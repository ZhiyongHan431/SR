#!/usr/bin/env python3
"""Train the negative calibrator C- for quality-contrastive revision guidance.

C- has the architecture of the positive calibrator C and reads the same cached
frozen-backbone features (hidden states, head logits, LR latent, retention).
It is trained to predict the HBQ bits of a *low-quality* version of the HR
target. At inference the sampler extrapolates away from it:

    l_g = l_C + clamp(w * (l_C - l_C-), -bound, +bound) * plane_weight

No backbone forward runs here: only a 337K-parameter module is optimized on an
existing teacher cache, so this costs about as much as the C stage.

Low-quality targets (never the test set; same qualified pairs as the cache):
  blur        Gaussian-blurred HR (sigma in HR pixels), optional down/up resize
  lr_bicubic  the pair's LR bicubically upsampled to 512

Nothing runs without --execute. This is an experimental control with no
quality claim; benefits must be shown by complete guided inference.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / 'scripts', ROOT / 'code'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

TARGETS = ('blur', 'lr_bicubic')
SCHEMA = 's5_negative_calibrator_training_v1'


def low_quality_image(row, *, target, sigma, downscale, device):
    """Deterministic low-quality RGB[-1,1] tensor [1,3,512,512] for one pair."""
    import numpy as np
    import torch
    from PIL import Image
    from torch.nn import functional as F

    def read(path, size):
        with Image.open(path) as image:
            if image.mode != 'RGB' or image.size != (size, size):
                raise ValueError(f'Expected RGB {size}x{size}: {path}')
            pixels = np.array(image, copy=True)
        return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device).float() / 127.5 - 1.

    if target == 'lr_bicubic':
        image = F.interpolate(read(row['lr'], 128), size=(512, 512), mode='bicubic', align_corners=False)
        return image.clamp(-1, 1)
    image = read(row['hr'], 512)
    if sigma > 0:
        radius = max(1, int(math.ceil(3 * sigma)))
        offsets = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        kernel = torch.exp(-.5 * (offsets / sigma) ** 2)
        kernel = kernel / kernel.sum()
        padded = F.pad(image, (radius, radius, radius, radius), mode='reflect')
        padded = F.conv2d(padded, kernel.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
        image = F.conv2d(padded, kernel.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)
    if downscale > 1:
        small = max(1, round(512 / downscale))
        image = F.interpolate(image, size=(small, small), mode='bicubic', align_corners=False, antialias=True)
        image = F.interpolate(image, size=(512, 512), mode='bicubic', align_corners=False)
    return image.clamp(-1, 1)


def encode_bits(vae, image, *, rounds=4):
    """RGB[-1,1] [1,3,512,512] -> target bit labels [1,1024,256] (int64).

    Same encoder call and bit-plane order as tools.cache_sr_training_states.encode_target.
    """
    import torch
    from grn.utils_t2iv.hbq_util_t2iv import raw_feature2bit_label
    with torch.no_grad(), torch.autocast(image.device.type, enabled=False):
        raw, _, _ = vae.encode_for_raw_features(image.unsqueeze(2), scale_schedule=None, slice=True)
        labels = raw_feature2bit_label(raw[0], hbq_round=rounds)
    labels = labels.flatten(2).transpose(1, 2).contiguous().long()
    if labels.shape != (1, 1024, 256):
        raise ValueError(f'Unexpected negative target geometry: {tuple(labels.shape)}')
    return labels


def train(args):
    import torch
    from _bundle import CONFIG, bundled
    from grn.models.sr_quality_guidance import configure_negative_calibrator, save_negative_calibrator
    from grn.utils_t2iv.sr_trajectory_adapter import load_trajectory_adapter
    from tools import trajectory_adaptation as ta

    scalars = (args.lr, args.kappa_min, args.kappa_max, args.state_mix, args.blur_sigma, args.downscale)
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError('Training scalars must be finite')
    if (min(args.updates, args.save_every) < 1 or min(args.lr, args.kappa_min) <= 0
            or args.kappa_min > args.kappa_max or not 0 <= args.state_mix <= 1
            or args.blur_sigma < 0 or args.downscale < 1):
        raise ValueError('Invalid training configuration')
    if args.target == 'blur' and args.blur_sigma == 0 and args.downscale == 1:
        raise ValueError('blur target needs --blur-sigma > 0 or --downscale > 1')
    cache_path, manifest = ta.load_cache(args.cache)
    adapter_path = ta.local_path(args.adapter)
    output = ta.fresh_output(args.output)
    status = {'schema': SCHEMA, 'status': 'running', 'updates_completed': 0}
    ta.write_json(output / 'status.json', status)
    try:
        base_sha = ta.sha256(bundled(CONFIG['checkpoint']))
        if base_sha != manifest['base_sha256']:
            raise ValueError('Cache belongs to a different base checkpoint')
        ta.require_vae_identity(manifest.get('vae_sha256'), ta.sha256(bundled(CONFIG['hbq'])),
                                context='Current encoder')
        adapter_sha = ta.sha256(adapter_path)
        pipe = ta.load_pipeline(args.device)
        load_trajectory_adapter(adapter_path, pipe.model, expected_base_sha256=base_sha, trainable=False)
        pipe.model.requires_grad_(False)
        configure_negative_calibrator(pipe.model, trainable=True)
        negative = pipe.model.sr_negative_calibrator
        device = next(negative.parameters()).device
        parameters = list(negative.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.)
        config = {'schema': SCHEMA, 'target': args.target, 'blur_sigma': args.blur_sigma,
                  'downscale': args.downscale, 'lr': args.lr, 'state_mix': args.state_mix,
                  'kappa_min': args.kappa_min, 'kappa_max': args.kappa_max, 'seed': args.seed,
                  'updates_requested': args.updates, 'cache_sha256': ta.sha256(cache_path),
                  'cache_teacher_adapter_sha256': manifest['teacher_adapter_sha256'],
                  'pairs_sha256': manifest['pairs_sha256'], 'base_sha256': base_sha,
                  'positive_adapter_sha256': adapter_sha, 'initialization': 'copy_of_positive_calibrator',
                  'trainable_parameters': sum(p.numel() for p in parameters),
                  'cache_teacher_matches_positive_adapter': manifest['teacher_adapter_sha256'] == adapter_sha,
                  'scope': 'Negative calibrator for guidance; fixed-teacher cached features; no quality claim.'}
        ta.write_json(output / 'config.json', config)
        history = output / 'history.jsonl'
        records = manifest['records']
        for step in range(args.updates):
            choice = ta.schedule_step(step, len(records), args.state_mix, args.kappa_min, args.kappa_max, args.seed)
            row = records[choice['index']]
            features = torch.load(cache_path.parent / row['feature_path'], map_location='cpu', weights_only=True)
            state = {k: v.to(device) for k, v in features[choice['source']][choice['state_step']].items()}
            target = encode_bits(pipe.vae, low_quality_image(
                row, target=args.target, sigma=args.blur_sigma, downscale=args.downscale, device=device))
            clean = features['target_bits'].long().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = negative.calibrate_logits(state['head_logits'], state['hidden'], state['lr_features'],
                                               state['signal'], kappa=choice['kappa'])
            loss = ta.bit_cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite negative-calibrator loss')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            log = dict(step=step + 1, pair_id=row['pair_id'], **choice, bit_ce=float(loss.detach()),
                       target_differs_from_clean=float((target != clean).float().mean()),
                       grad_norm=float(grad_norm))
            with history.open('a') as stream:
                stream.write(json.dumps(log, allow_nan=False) + '\n')
            status['updates_completed'] += 1
            if (step + 1) % 50 == 0 or step + 1 == args.updates:
                ta.write_json(output / 'status.json', status)
                print('UPDATE', json.dumps(log), flush=True)
            if (step + 1) % args.save_every == 0 and step + 1 != args.updates:
                save_negative_calibrator(output / f'negative-step-{step + 1:08d}.pt', pipe.model,
                    base_sha256=base_sha, positive_adapter_sha256=adapter_sha, step=step + 1,
                    metadata={'training_config': config})
        save_negative_calibrator(output / 'negative.pt', pipe.model, base_sha256=base_sha,
            positive_adapter_sha256=adapter_sha, step=args.updates,
            metadata={'training_config': config, 'history_sha256': ta.sha256(history)})
        status.update(status='complete', negative_sha256=ta.sha256(output / 'negative.pt'))
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        ta.write_json(output / 'status.json', status)
    return output / 'negative.pt'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cache', required=True, type=Path, help='Teacher cache.json (ideally cached with --adapter as teacher)')
    parser.add_argument('--adapter', required=True, type=Path, help='Positive C or C+R adapter that guidance will run beside')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--target', choices=TARGETS, default='blur')
    parser.add_argument('--blur-sigma', type=float, default=2.0, help='HR-pixel Gaussian sigma for --target blur')
    parser.add_argument('--downscale', type=float, default=1.0, help='Extra bicubic down/up factor for --target blur')
    # 16,384 = 2 sources x 2,048 pairs x 4 cached rounds: fewer updates do not visit every round.
    parser.add_argument('--updates', type=int, default=16384)
    parser.add_argument('--save-every', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--state-mix', type=float, default=.5)
    parser.add_argument('--kappa-min', type=float, default=.25)
    parser.add_argument('--kappa-max', type=float, default=4.)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2), flush=True)
    if not args.execute:
        return
    print('COMPLETE', train(args), flush=True)


if __name__ == '__main__':
    main()
