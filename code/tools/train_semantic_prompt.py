#!/usr/bin/env python3
"""Train the CLIP semantic prompt through the frozen SR backbone.

    CLIP image embedding of the LR -> projector -> text-slot tokens

Only the projector (and, optionally, one learned negative token) is optimized.
The backbone, text_proj, the C/R adapter, the HBQ tokenizer and CLIP are frozen,
so the model without a semantic prompt stays the unchanged base model and can
serve as the negative branch of 'prompt' guidance at inference.

Each update runs ONE differentiable round of the real backbone (model.forward,
the same call the S5 trainer uses) on a bit state of a cached training pair:

  positive step   text slot = semantic tokens,       target = HR bits
  negative step   text slot = learned negative token, target = low-quality bits
                  (--negative-every N makes every N-th step a negative step)

Bit states are a mix of the teacher's cached rollout inputs (rounds 20/35/45/50)
and states constructed from the HR bits at the sampler's own retention schedule.

--probe N   No training. Perturb the empty-prompt token on N pairs and report
            how far the bit log-odds move. Run this first: if the frozen backbone
            no longer reads its text slot, a prompt cannot work without also
            adapting the backbone.

The last --holdout cache records are never trained on. On them the tool reports
bit cross-entropy with the empty prompt, with the image's own semantic tokens,
and with another image's tokens (a control for image-specific semantics). Nothing runs without
--execute. This is an experimental control with no quality claim: benefits must
be shown by complete guided inference on the benchmarks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / 'scripts', ROOT / 'code'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

SCHEMA = 's5_semantic_prompt_training_v1'
ROUNDS = 50
ROLLOUT_STEPS = (20, 35, 45, 50)
EVAL_ROUNDS = (0, 16, 32, 48)
PROBE_ROUNDS = (0, 20, 40)


def retention(round_index):
    """Scheduled input retention of the 50-round sampler at a zero-based round."""
    if type(round_index) is not int or not 0 <= round_index < ROUNDS:
        raise ValueError('round_index must be an integer in [0, 49]')
    return 0.95 * (1.0 - math.cos(math.pi / 2 * min(1.0, round_index / (ROUNDS - 1))))


def load_cache_manifest(path, ta):
    """Read a trajectory-adaptation teacher cache without hashing its feature files.

    Only state.pt and the HR/LR PNGs are used here; each is hashed when read.
    """
    path = ta.local_path(path)
    manifest = json.loads(path.read_text())
    if (manifest.get('schema') != ta.SCHEMA or manifest.get('status') != 'complete'
            or manifest.get('rollout_steps') != ROUNDS or manifest.get('state_steps') != list(ROLLOUT_STEPS)
            or not isinstance(manifest.get('records'), list) or not manifest['records']):
        raise ValueError('Expected a complete trajectory-adaptation teacher cache (cache.json)')
    return path, manifest


def read_image(path, size, device):
    """Saved RGB PNG -> [1,3,size,size] in [-1,1], the pipeline's LR/HR convention."""
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(path) as image:
        if image.mode != 'RGB' or image.size != (size, size):
            raise ValueError(f'Expected RGB {size}x{size}: {path}')
        pixels = np.array(image, copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0 * 2.0 - 1.0


def load_item(cache_dir, row, device, ta):
    """(validated state cache, HR [1,3,512,512], LR [1,3,128,128]) of one cached pair."""
    from grn.utils_t2iv.sr_state_training import load_pair_state
    state_path = ta.local_path(cache_dir / row['state_path'])
    if ta.sha256(state_path) != row['state_sha256']:
        raise ValueError('State cache bytes changed: ' + row['pair_id'])
    state = load_pair_state(state_path)
    if tuple(state['target_bits'].shape) != (1, 1024, 256):
        raise ValueError('Unexpected cached bit geometry: ' + row['pair_id'])
    images = []
    for name, size in (('hr', 512), ('lr', 128)):
        file = ta.local_path(row[name])
        if ta.sha256(file) != row[name + '_sha256']:
            raise ValueError(f'{name} image bytes changed: ' + row['pair_id'])
        images.append(read_image(file, size, device))
    return state, images[0], images[1]


def build_state(state, *, source, position, seed, device):
    """Input bits [1,1024,256] (uint8, on device) and their retention.

    rollout      the teacher's actual input at sampler step ``position`` (20/35/45/50)
    constructed  HR bits kept with the scheduled probability of zero-based round
                 ``position``, fresh random bits elsewhere
    """
    import torch
    target = state['target_bits']
    if source == 'rollout':
        cached = state['states'][position]
        return cached['input_bits'].to(device), float(cached['signal_fraction'])
    if source != 'constructed':
        raise ValueError('source must be rollout or constructed')
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    noise = torch.randint(0, 2, target.shape, generator=generator, dtype=torch.uint8)
    keep = torch.rand(target.shape, generator=generator) < retention(position)
    return torch.where(keep, target, noise).to(device), float(keep.float().mean())


def step_plan(step, order, *, state_mix, negative_every, seed):
    """Deterministic data/state choice for one update."""
    generator = random.Random(seed + step * 104729)
    rollout = generator.random() < state_mix
    return {'index': order[step % len(order)],
            'kind': 'negative' if negative_every and step % negative_every == negative_every - 1 else 'positive',
            'source': 'rollout' if rollout else 'constructed',
            'position': generator.choice(ROLLOUT_STEPS) if rollout else generator.randrange(ROUNDS),
            'state_seed': generator.randrange(2 ** 31)}


class FrozenBackbone:
    """One round of the frozen SR model on a given bit state, differentiable in the text slot."""

    def __init__(self, pipe, kappa):
        import numpy as np
        from grn.schedules.global_refine import get_visual_rope_embeds
        self.pipe, self.model = pipe, pipe.model
        if self.model.sr_condition_mode != 'prefix_text' or self.model.sr_latent_size != 32:
            raise ValueError('The semantic prompt requires the prefix_text 128->512 SR model')
        self.device = next(self.model.parameters()).device
        templates = self.model.h_div_w_templates
        template = templates[np.argmin(np.abs(templates - 1.0))]
        # Same visual RoPE the sampler and the trainer build for a (1,32,32) grid.
        self.rope = get_visual_rope_embeds(self.model.rope2d_freqs_grid, (1, 32, 32), self.device,
                                           template, t_offset=0)
        self.kappa = kappa if hasattr(self.model, 'sr_round_calibrator') else None

    def conditions(self, lr):
        """Frozen DINOv2 features of the LR if the base model fuses them, else None."""
        encoder = self.pipe.sr_semantic_encoder
        return None if encoder is None else encoder(lr)

    def __call__(self, *, text, bits, target, signal, lr, semantic):
        """Return (mean bit cross-entropy, logits [1,2,1024,256]).

        text    [L,4096] FP32 text-slot features (may require grad)
        bits    [1,1024,256] input bit state; target [1,1024,256] label bits
        signal  retention of ``bits`` in [0,1], the value of the signal token
        """
        import torch
        from grn.utils_t2iv.hbq_util_t2iv import multiclass_labels2onehot_input
        length = int(text.shape[0])
        label = (text, [length], torch.tensor([0, length], dtype=torch.int32, device=self.device), length, [1])
        labels = bits.transpose(1, 2).reshape(1, 256, 1, 32, 32)
        inputs = multiclass_labels2onehot_input(labels, 2).flatten(2).transpose(1, 2)
        info = [{'largest_scale': (1, 32, 32), 'wandb_plot_index': min(9, int(signal / 0.1)),
                 'cur_bits': 256, 'cur_lvl': 2, 'scale_token_id': float(signal),
                 'predict_tokens': 1024, 'all_tokens': -1, 'first_frame_condition': False}]
        kwargs = dict(lr_condition=lr, x_BLC_mask=None, gt_BL=[target.long()],
                      visual_rope_cache=[self.rope], sequece_packing_scales=[[(1, 32, 32)]],
                      super_scale_lengths=[2 * 1024 + length + 1], other_info_by_scale=info,
                      scale_or_time_ids=None, return_sr_aux=True)
        if semantic is not None:
            kwargs['sr_semantic_features'] = semantic
        if self.kappa is not None:
            kwargs['sr_calibrator_kappa'] = self.kappa
        with torch.autocast('cuda', enabled=self.device.type == 'cuda', dtype=torch.bfloat16):
            _, loss, _, _, aux = self.model(label, [inputs], **kwargs)
        return loss.float().mean(), aux['logits_by_scale'][0]


def run_probe(backbone, records, cache_dir, empty, ta, count):
    """How far do the bit log-odds move when only the text-slot token changes?"""
    import torch
    device = backbone.device
    rms = empty.square().mean().sqrt()
    generator = torch.Generator(device='cpu').manual_seed(0)
    rows = []
    with torch.no_grad():
        for row in records[:count]:
            state, _, lr = load_item(cache_dir, row, device, ta)
            semantic = backbone.conditions(lr)
            target = state['target_bits'].to(device)
            for position in PROBE_ROUNDS:
                bits, signal = build_state(state, source='constructed', position=position,
                                           seed=1000 + position, device=device)

                def log_odds(text):
                    _, logits = backbone(text=text, bits=bits, target=target, signal=signal, lr=lr, semantic=semantic)
                    return (logits[:, 1] - logits[:, 0]).float()

                reference = log_odds(empty)
                noise = torch.randn(empty.shape, generator=generator).to(device) * rms
                variants = {'zeros': torch.zeros_like(empty), 'double': 2.0 * empty, 'noise_1rms': empty + noise}
                for name, text in variants.items():
                    other = log_odds(text)
                    rows.append({'pair_id': row['pair_id'], 'round': position, 'variant': name,
                                 'mean_abs_shift': float((other - reference).abs().mean()),
                                 'flipped_bits': float(((other > 0) != (reference > 0)).float().mean()),
                                 'mean_abs_reference': float(reference.abs().mean())})
            print('PROBED', row['pair_id'], flush=True)
    summary = {}
    for name in ('zeros', 'double', 'noise_1rms'):
        chosen = [item for item in rows if item['variant'] == name]
        summary[name] = {key: sum(item[key] for item in chosen) / len(chosen)
                         for key in ('mean_abs_shift', 'flipped_bits', 'mean_abs_reference')}
    return {'summary': summary, 'rows': rows,
            'reading': 'mean_abs_shift is in log-odds units; the calibrator bound kappa is 2. Shifts far below '
                       '0.01 with almost no flipped bits mean the frozen backbone barely reads its text slot.'}


def evaluate(backbone, projector, clip_encoder, records, cache_dir, empty, ta, low_quality):
    """Held-out bit cross-entropy with and without the learned text-slot tokens.

    clean_ce_shuffled gives each image the semantic tokens of a different
    held-out image. If it equals clean_ce_semantic, the prompt acts as one
    constant token and carries no image-specific semantics.
    """
    import torch
    device = backbone.device
    totals = {}

    def add(key, value):
        totals.setdefault(key, []).append(float(value))

    with torch.no_grad():
        prompts = []
        for row in records:
            embedding = clip_encoder(load_item(cache_dir, row, device, ta)[2])
            prompts.append(projector.positive_text(embedding, empty))
            add('offset_rms', projector.offsets(embedding).square().mean().sqrt())
        for number, row in enumerate(records):
            state, _, lr = load_item(cache_dir, row, device, ta)
            semantic = backbone.conditions(lr)
            clean = state['target_bits'].to(device)
            positive = prompts[number]
            shuffled = prompts[(number + 1) % len(prompts)] if len(prompts) > 1 else None
            low = None if low_quality is None else low_quality(row)
            negative = None if low is None else projector.negative_text(empty)
            for position in EVAL_ROUNDS:
                bits, signal = build_state(state, source='constructed', position=position,
                                           seed=7919 * (number + 1) + position, device=device)
                common = dict(bits=bits, signal=signal, lr=lr, semantic=semantic)
                add('clean_ce_empty', backbone(text=empty, target=clean, **common)[0])
                add('clean_ce_semantic', backbone(text=positive, target=clean, **common)[0])
                if shuffled is not None:
                    add('clean_ce_shuffled', backbone(text=shuffled, target=clean, **common)[0])
                if low is not None:
                    add('low_ce_empty', backbone(text=empty, target=low, **common)[0])
                    add('low_ce_negative', backbone(text=negative, target=low, **common)[0])
    return {key: sum(values) / len(values) for key, values in totals.items()}


def train(args):
    import torch
    from _bundle import CONFIG, bundled
    from grn.models.sr_semantic_prompt import (FrozenCLIPImageEncoder, SemanticPromptProjector,
                                               save_semantic_prompt, tensor_sha256)
    from grn.utils_t2iv.sr_native import load_empty_text_cache
    from grn.utils_t2iv.sr_trajectory_adapter import load_trajectory_adapter
    from tools import trajectory_adaptation as ta
    from tools.train_negative_calibrator import encode_bits, low_quality_image

    scalars = (args.lr, args.negative_lr, args.state_mix, args.align_weight, args.kappa,
               args.blur_sigma, args.downscale)
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError('Training scalars must be finite')
    if (min(args.updates, args.accumulate, args.save_every, args.eval_every, args.tokens, args.hidden) < 1
            or min(args.lr, args.negative_lr) <= 0 or not 0 <= args.state_mix <= 1
            or min(args.align_weight, args.kappa, args.blur_sigma, args.negative_every, args.holdout, args.probe) < 0
            or args.downscale < 1):
        raise ValueError('Invalid training configuration')
    negative = args.negative_every > 0
    if negative and args.negative_target == 'blur' and args.blur_sigma == 0 and args.downscale == 1:
        raise ValueError('blur target needs --blur-sigma > 0 or --downscale > 1')
    cache_path, manifest = load_cache_manifest(args.cache, ta)
    cache_dir = cache_path.parent
    records = manifest['records']
    if args.holdout >= len(records):
        raise ValueError('--holdout leaves no training records')
    held_out = records[len(records) - args.holdout:] if args.holdout else []
    training = records[:len(records) - args.holdout]
    output = ta.fresh_output(args.output)
    status = {'schema': SCHEMA, 'status': 'running', 'updates_completed': 0}
    ta.write_json(output / 'status.json', status)
    try:
        base_sha = ta.sha256(bundled(CONFIG['checkpoint']))
        if base_sha != manifest['base_sha256']:
            raise ValueError('Cache belongs to a different base checkpoint')
        ta.require_vae_identity(manifest.get('vae_sha256'), ta.sha256(bundled(CONFIG['hbq'])),
                                context='Current tokenizer')
        pipe = ta.load_pipeline(args.device)
        adapter_sha = None
        if args.adapter:
            adapter_path = ta.local_path(args.adapter)
            adapter_sha = ta.sha256(adapter_path)
            load_trajectory_adapter(adapter_path, pipe.model, expected_base_sha256=base_sha, trainable=False)
        pipe.model.requires_grad_(False)
        pipe.model.eval()
        pipe.vae.requires_grad_(False)
        pipe.vae.eval()
        backbone = FrozenBackbone(pipe, args.kappa)
        device = backbone.device
        empty = load_empty_text_cache(str(bundled(CONFIG['empty_text'])), 4096).to(device)

        if args.probe:
            report = run_probe(backbone, records, cache_dir, empty, ta, min(args.probe, len(records)))
            ta.write_json(output / 'probe.json', report)
            print('PROBE', json.dumps(report['summary'], indent=2), flush=True)
            print(report['reading'], flush=True)
            status.update(status='complete', mode='probe')
            return output / 'probe.json'

        if not args.clip:
            raise ValueError('--clip is required for training')
        clip_encoder = FrozenCLIPImageEncoder(ta.local_path(args.clip)).to(device)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(args.seed)
            projector = SemanticPromptProjector(clip_dim=clip_encoder.embed_dim, hidden_dim=args.hidden,
                                                tokens=args.tokens, base_tokens=int(empty.shape[0]))
        projector.to(device)
        projector.train()
        main_parameters = [p for name, p in projector.named_parameters() if name != 'negative_delta']
        parameters = main_parameters + [projector.negative_delta]
        optimizer = torch.optim.AdamW([{'params': main_parameters, 'lr': args.lr},
                                       {'params': [projector.negative_delta], 'lr': args.negative_lr}],
                                      weight_decay=0.)

        def low_quality(row):
            return encode_bits(pipe.vae, low_quality_image(
                row, target=args.negative_target, sigma=args.blur_sigma, downscale=args.downscale, device=device))

        config = {'schema': SCHEMA, 'tokens': args.tokens, 'hidden': args.hidden, 'lr': args.lr,
                  'negative_lr': args.negative_lr, 'negative_every': args.negative_every,
                  'negative_target': args.negative_target if negative else None,
                  'blur_sigma': args.blur_sigma, 'downscale': args.downscale,
                  'align_weight': args.align_weight, 'state_mix': args.state_mix, 'kappa': backbone.kappa,
                  'accumulate': args.accumulate, 'updates_requested': args.updates, 'seed': args.seed,
                  'holdout': args.holdout, 'training_records': len(training),
                  'cache_sha256': ta.sha256(cache_path), 'pairs_sha256': manifest['pairs_sha256'],
                  'cache_teacher_adapter_sha256': manifest['teacher_adapter_sha256'],
                  'base_sha256': base_sha, 'adapter_sha256': adapter_sha, 'clip': clip_encoder.identity(),
                  'trainable_parameters': sum(p.numel() for p in parameters),
                  'scope': 'Text-slot prompt tuning through the frozen backbone; no quality claim.'}
        ta.write_json(output / 'config.json', config)
        history, evaluations = output / 'history.jsonl', output / 'eval.jsonl'
        empty_sha = tensor_sha256(empty)

        def checkpoint(name, step):
            return save_semantic_prompt(output / name, projector, base_sha256=base_sha, adapter_sha256=adapter_sha,
                clip_identity=clip_encoder.identity(), empty_text_sha256=empty_sha,
                negative_trained=negative, step=step, metadata={'training_config': config})

        def held_out_report(step):
            if not held_out:
                return None
            projector.eval()
            report = dict(step=step, **evaluate(backbone, projector, clip_encoder, held_out, cache_dir, empty,
                                                ta, low_quality if negative else None))
            projector.train()
            with evaluations.open('a') as stream:
                stream.write(json.dumps(report, allow_nan=False) + '\n')
            print('EVAL', json.dumps(report), flush=True)
            return report

        order = random.Random(args.seed).sample(range(len(training)), len(training))
        optimizer.zero_grad(set_to_none=True)
        total = args.updates * args.accumulate
        for step in range(total):
            choice = step_plan(step, order, state_mix=args.state_mix,
                               negative_every=args.negative_every, seed=args.seed)
            row = training[choice['index']]
            state, hr, lr = load_item(cache_dir, row, device, ta)
            semantic = backbone.conditions(lr)
            bits, signal = build_state(state, source=choice['source'], position=choice['position'],
                                       seed=choice['state_seed'], device=device)
            alignment = torch.zeros((), device=device)
            if choice['kind'] == 'positive':
                embedding = clip_encoder(lr)
                text = projector.positive_text(embedding, empty)
                target = state['target_bits'].to(device)
                if args.align_weight > 0:
                    alignment = projector.alignment_loss(embedding, clip_encoder(hr))
            else:
                text = projector.negative_text(empty)
                target = low_quality(row)
            bit_ce, _ = backbone(text=text, bits=bits, target=target, signal=signal, lr=lr, semantic=semantic)
            loss = bit_ce + args.align_weight * alignment
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite semantic prompt loss')
            (loss / args.accumulate).backward()
            log = dict(step=step + 1, pair_id=row['pair_id'], **choice, signal=signal,
                       bit_ce=float(bit_ce.detach()), alignment=float(alignment.detach()))
            if (step + 1) % args.baseline_every == 0:
                # Same state and target with the constant empty prompt: the reference this prompt must beat.
                with torch.no_grad():
                    log['bit_ce_empty'] = float(backbone(text=empty, bits=bits, target=target, signal=signal,
                                                         lr=lr, semantic=semantic)[0])
            if (step + 1) % args.accumulate == 0:
                if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in parameters):
                    raise RuntimeError('No nonzero projector gradient: the text slot may not reach the loss')
                log['grad_norm'] = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                status['updates_completed'] += 1
                update = status['updates_completed']
                if update % args.save_every == 0 and update != args.updates:
                    checkpoint(f'semantic-step-{update:08d}.pt', update)
                if update % args.eval_every == 0 and update != args.updates:
                    held_out_report(update)
            with history.open('a') as stream:
                stream.write(json.dumps(log, allow_nan=False) + '\n')
            if (step + 1) % 20 == 0 or step + 1 == total:
                ta.write_json(output / 'status.json', status)
                print('STEP', json.dumps(log), flush=True)
        final_report = held_out_report(args.updates)
        checkpoint('semantic.pt', args.updates)
        status.update(status='complete', semantic_sha256=ta.sha256(output / 'semantic.pt'),
                      held_out=final_report)
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        ta.write_json(output / 'status.json', status)
    return output / 'semantic.pt'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cache', required=True, type=Path, help='Teacher cache.json (pairs + cached bit states)')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--adapter', type=Path, help='C or C+R adapter the prompt will run beside (recommended)')
    parser.add_argument('--clip', type=Path, help='CLIP weights inside the bundle: OpenAI .pt file or transformers directory')
    parser.add_argument('--probe', type=int, default=0, help='Only measure text-slot sensitivity on this many pairs')
    parser.add_argument('--tokens', type=int, default=1, help='Text-slot tokens; 1 adds onto the empty-prompt token')
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--updates', type=int, default=2048, help='Optimizer updates (each of --accumulate samples)')
    parser.add_argument('--accumulate', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--negative-every', type=int, default=0,
                        help='Every N-th sample trains the learned negative token on a low-quality target; 0 = off')
    parser.add_argument('--negative-lr', type=float, default=1e-2)
    parser.add_argument('--negative-target', choices=('blur', 'lr_bicubic'), default='blur')
    parser.add_argument('--blur-sigma', type=float, default=2.0)
    parser.add_argument('--downscale', type=float, default=1.0)
    parser.add_argument('--align-weight', type=float, default=1.0,
                        help='Weight of the LR->HR CLIP embedding alignment loss; 0 = off')
    parser.add_argument('--state-mix', type=float, default=.5, help='Fraction of cached rollout states')
    parser.add_argument('--kappa', type=float, default=2.0)
    parser.add_argument('--holdout', type=int, default=32, help='Last N cache records kept for evaluation only')
    parser.add_argument('--eval-every', type=int, default=512)
    parser.add_argument('--save-every', type=int, default=512)
    parser.add_argument('--baseline-every', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.baseline_every < 1:
        parser.error('--baseline-every must be positive')
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2), flush=True)
    if not args.execute:
        return
    print('COMPLETE', train(args), flush=True)


if __name__ == '__main__':
    main()
