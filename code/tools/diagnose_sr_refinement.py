#!/usr/bin/env python3
"""Read-only GRN SR refinement parity, trajectory, and condition diagnostics.

--pairs accepts schema_version=1 JSON with pairs:[{pair_id,source_id,hr,lr}],
where image paths are absolute or relative to JSON, RGB PNGs are512/128 pixels,
pair_id is one safe filename component, and source IDs are unique originals.
prepare_sr_diagnostic_pairs.py produces
this schema with content hashes and source reservation metadata.
Requires at least two pairs for cyclic LR derangement. --parity-trajectory-only
permits a single-pair smoke, omitting causal interventions. Teacher-corrupted
GT probes are diagnostics, not benchmark scores or root-cause proof.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from tools.sr_refinement_diagnostics import (
    bit_metrics, capture_refinement, compare_logits, disable_condition, file_hash,
    fresh_output, load_pairs, preserve_rng, replay_forward, require_identical_noise,
    summarize_run, teacher_snapshot, tensor_hash, tensor_tree, verify_final_rgb,
)


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument('--checkpoint', type=Path, required=True)
    result.add_argument('--vae', type=Path, required=True)
    result.add_argument('--pairs', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--native-empty-text-cache', '--sr_native_empty_text_path', type=Path)
    result.add_argument('--samples', type=int, default=0, help='0 means all; positive selects first N fixed pairs')
    result.add_argument('--seed', type=int, default=0)
    result.add_argument('--steps', type=int, default=50)
    result.add_argument('--parity-trajectory-only', action='store_true')
    return result


def effective_protocol(steps):
    if steps < 2:
        raise ValueError('steps must be >=2: real schedule divides by Tmax-1')
    return dict(guidance_scale=1., temperature=1., top_k=1, top_p=.75,
                complexity_aware_Tmin=steps, complexity_aware_Tmax=steps,
                complexity_aware_b=steps, complexity_aware_k=0, complexity_aware_wp=0,
                snr_shift=1., output_type='numpy')


def selected_steps(steps):
    return sorted({step for step in (1, 5, 10, 20, 35, 45, 50, steps) if step <= steps})


def validate_inference_args(args):
    # The real loop begins at repeat_idx=0 and leaks only if repeat_idx<gt_leak.
    if args.gt_leak > 0:
        raise ValueError('Diagnostic inference forbids GT leakage')
    if args.refine_mode != 'ar_discrete_GRN_bit' or args.add_scale_token != 1:
        raise ValueError('Diagnostic supports real binary refinement with one time token')


def score_metrics(metrics, prediction, target):
    """Score all saved uint8 images consistently, independent of caller autocast."""
    from tools.diagnose_grn_sr import score
    with torch.autocast(prediction.device.type, enabled=False):
        return score(metrics, prediction.float(), target.float())


def _rgb_tensor(rgb, device):
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(device).float() / 255.


def _bits_latent(bits, grid, rounds):
    from grn.utils_t2iv.hbq_util_t2iv import bit_label2raw_feature
    labels = bits.transpose(1, 2).reshape(bits.shape[0], bits.shape[2], *grid)
    return bit_label2raw_feature(labels, rounds)


def run(args):
    from tools.grn_pipeline import GRNPipeline
    from tools.diagnose_grn_sr import image_tensor, make_metrics
    from tools.sr_diagnostic_core import latent_error
    from grn.utils_t2iv.hbq_util_t2iv import raw_feature2bit_label, bit_label2raw_feature
    from grn.utils_t2iv.sr_native import load_empty_text_cache

    protocol = effective_protocol(args.steps)
    pairs = load_pairs(args.pairs, args.samples)
    modes = ['correct'] if args.parity_trajectory_only else ['correct', 'shuffled', 'disabled']
    if len(pairs) < 2 and not args.parity_trajectory_only:
        raise ValueError('Causal LR derangement requires at least two pairs; use --parity-trajectory-only for smoke')
    fresh_output(args.output)
    records, completed = [], []
    metadata = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        protocol=protocol, capture_steps=selected_steps(args.steps), teacher_signal_fractions=[0., .25, .5, .75, .9],
        metric_protocol='PyIQA PSNR-Y/LPIPS/DISTS; FP32 with autocast disabled; RGB PNG uint8; Y crop0; no color correction',
        parity_scope='REAL model.forward eval-mode BF16 autocast vs REAL autoregressive inference; not full FSDP/trainer equivalence',
        known_numeric_path_differences='Training forward disables BF16 autocast for word_embed; inference does not. Inference casts hidden states to FP32 before head; training forward does not.',
        trajectory_labels='Captured inputs are actual mixed rollout states. Argmax predictions equal sampled bit predictions under formal top_k=1; saved intermediate images are labelled argmax_prediction. Sampled final comes from the actual decode entrypoint.',
        teacher_warning='Teacher-corrupted GT probes are not benchmark scores; teacher versus rollout differences alone do not prove a root cause.',
        condition_disable='Native adapter residual bypass; prefix encoder output zeroed, preserving prefix length/positions layout and zero legacy cross inputs.',
        historical_isolation=json.loads(args.pairs.read_text()).get('metadata', {}).get('historical_isolation', 'Unknown for external pair manifest'),
        snapshots='Selected CPU snapshots retained in memory per image only; no logits/checkpoint dumps',
        modes=modes, status='running')

    def write_summary(status='running', error=None):
        summary = summarize_run(records, len(pairs), modes, status, completed, error,
                                expected_steps=selected_steps(args.steps),
                                teacher_signals=metadata['teacher_signal_fractions'])
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        return summary

    def append(row):
        with (args.output / 'records.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        records.append(row)
        write_summary()

    write_summary()
    try:
        metadata['identities'] = {name: dict(path=str(path.resolve()), sha256=file_hash(path))
                                  for name, path in [('checkpoint', args.checkpoint), ('vae', args.vae), ('pairs', args.pairs)]}
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        if not torch.cuda.is_available():
            raise RuntimeError('Real refinement diagnostics require CUDA')
        pipe = GRNPipeline.from_pretrained(model_path=str(args.checkpoint), vae_path=str(args.vae),
            task='SR', model='GRN2b', device='cuda',
            sr_native_empty_text_path=str(args.native_empty_text_cache) if args.native_empty_text_cache else '')
        model, vae = pipe.model, pipe.vae
        validate_inference_args(pipe.args)
        metadata.update(condition_mode=model.sr_condition_mode,
                        model_training=bool(model.training), parameter_dtype=str(next(model.parameters()).dtype),
                        forward_autocast='cuda BF16', actual_gt_leak=pipe.args.gt_leak)
        if model.training:
            raise RuntimeError('Pipeline must load the model in eval mode')
        features = None
        if model.sr_condition_mode == 'native_adapter':
            cache_path = Path(pipe.args.sr_native_empty_text_path)
            features = load_empty_text_cache(cache_path)
            metadata['identities']['native_empty_text_cache'] = dict(path=str(cache_path.resolve()), sha256=file_hash(cache_path))
        metrics = {key: value for key, value in make_metrics().items() if key in ('PSNR-Y', 'LPIPS', 'DISTS')}
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        with torch.no_grad(), preserve_rng(model):
            for index, pair in enumerate(pairs):
                target_rgb = image_tensor(pair['hr'], 'cuda')
                with torch.autocast('cuda', enabled=False):
                    raw, _, _ = vae.encode_for_raw_features((target_rgb * 2 - 1).unsqueeze(2), scale_schedule=None, slice=True)
                    labels = raw_feature2bit_label(raw[0], pipe.args.hbq_round)
                    target_latent = bit_label2raw_feature(labels, pipe.args.hbq_round)
                grid = tuple(labels.shape[-3:])
                target = labels.flatten(2).transpose(1, 2).contiguous().long()
                reference_noise, reference_rgb = None, None
                correct_condition = correct_text = correct_rope = correct_lengths = None
                for mode in modes:
                    source = pairs[(index + 1) % len(pairs)] if mode == 'shuffled' else pair
                    with capture_refinement(model, selected_steps(args.steps), target.shape[-1]) as capture:
                        with disable_condition(model, mode == 'disabled') as counts:
                            rgb = pipe(lr_image=source['lr'], seed=args.seed + index, **protocol)[0]
                    if (capture.steps_seen != args.steps or set(capture.snapshots) != set(selected_steps(args.steps))
                            or capture.final_latent is None):
                        raise RuntimeError('Missing expected refinement states or sampled final latent')
                    noise = dict(initial_hash=capture.initial_hash, rng_hashes=capture.rng_hashes)
                    if reference_noise is None:
                        reference_noise = noise
                    require_identical_noise(reference_noise, noise)
                    folder = args.output / mode / pair['pair_id']
                    folder.mkdir(parents=True)
                    Image.fromarray(rgb).save(folder / 'sampled_final.png')
                    prediction = _rgb_tensor(rgb, 'cuda')
                    if reference_rgb is None:
                        reference_rgb = prediction.clone()
                    sampled_latent = capture.final_latent.to('cuda')
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        decoded = model.summed_codes2images(vae, sampled_latent)
                    verify_final_rgb(decoded, rgb)
                    sampled_bits = raw_feature2bit_label(sampled_latent, pipe.args.hbq_round).flatten(2).transpose(1, 2)
                    base = dict(pair_id=pair['pair_id'], source_id=pair['source_id'], mode=mode,
                                lr_source_id=source['source_id'], seed=args.seed + index)
                    append(dict(base, kind='sampled_final', initial_hr_sha256=capture.initial_hash,
                        rng_hashes=capture.rng_hashes, disable_calls=dict(counts),
                        metrics=score_metrics(metrics, prediction, target_rgb), latent_metrics=latent_error(sampled_latent, target_latent),
                        sampled_bit_error=float((sampled_bits != target).float().mean()),
                        pipeline_uint8_max_difference=0,
                        differences=dict(pixel_mae_vs_correct=float((prediction - reference_rgb).abs().mean()),
                                         changed_pixel_fraction=float((prediction != reference_rgb).float().mean())),
                        image=str((folder / 'sampled_final.png').relative_to(args.output))))
                    # Reuse the REAL captured visual RoPE coordinates and actual sequence lengths.
                    lengths = capture.layout['sequence_lengths']
                    rope = capture.layout['packed_rope'][:, :, :target.shape[1]].permute(0, 1, 3, 2, 4).unsqueeze(1).to('cuda')
                    actual_inputs = tensor_tree(capture.condition_inputs, 'cuda')
                    if actual_inputs['scale_schedule'] != [grid]:
                        raise RuntimeError('Captured real inference grid differs from quantized HR')
                    condition = {'lr_condition': actual_inputs['lr_condition']}
                    text = None
                    if features is not None:
                        condition = {key: actual_inputs[key] for key in ('native_lr_input', 'native_lr_latent')}
                        text = (*actual_inputs['label_B_or_BLT'][0], [1])
                    if mode == 'correct':
                        correct_condition, correct_text, correct_rope, correct_lengths = condition, text, rope, lengths
                    with disable_condition(model, mode == 'disabled'), torch.autocast('cuda', dtype=torch.bfloat16):
                        for step, snapshot in capture.snapshots.items():
                            replay = replay_forward(model, snapshot, target, rope, text, condition, lengths, grid,
                                                    expected_packed_rope=capture.layout['packed_rope'])
                            actual = snapshot['logits'].to('cuda')
                            row = dict(base, kind='forward_parity', step=step,
                                actual_signal_fraction=snapshot['signal_fraction'], input_state_sha256=tensor_hash(snapshot['inputs']),
                                inference_logits_dtype=snapshot['dtype'], replay_logits_dtype=str(replay.dtype),
                                parity=compare_logits(actual, replay))
                            append(row)
                            if mode == 'correct':
                                latent = _bits_latent(actual.argmax(-1), grid, pipe.args.hbq_round)
                                image_bgr = model.summed_codes2images(vae, latent)
                                image_rgb = image_bgr[0, 0].flip(-1).cpu().numpy()
                                image_path = folder / f'argmax_step_{step:02d}.png'
                                Image.fromarray(image_rgb).save(image_path)
                                append(dict(base, kind='argmax_prediction', step=step,
                                    actual_signal_fraction=snapshot['signal_fraction'],
                                    prediction_equals_sampled_bits_under_top_k_1=True,
                                    bit_metrics=bit_metrics(actual, target), latent_metrics=latent_error(latent, target_latent),
                                    metrics=score_metrics(metrics, _rgb_tensor(image_rgb, 'cuda'), target_rgb),
                                    image=str(image_path.relative_to(args.output))))
                    del capture, sampled_latent, prediction, decoded
                for signal in metadata['teacher_signal_fractions']:
                    snapshot = teacher_snapshot(labels, signal, args.seed + index)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = replay_forward(model, snapshot, target, correct_rope, correct_text,
                                                correct_condition, correct_lengths, grid)
                    latent = _bits_latent(logits.argmax(-1), grid, pipe.args.hbq_round)
                    append(dict(pair_id=pair['pair_id'], source_id=pair['source_id'], mode='correct',
                        kind='teacher_probe', requested_signal_fraction=signal,
                        actual_signal_fraction=snapshot['signal_fraction'], seed=args.seed + index,
                        bit_metrics=bit_metrics(logits, target), latent_metrics=latent_error(latent, target_latent)))
                completed.append(pair['pair_id'])
                write_summary()
                print(f'{index + 1}/{len(pairs)} {pair["pair_id"]}: all diagnostic records saved', flush=True)
        metadata['status'] = 'complete'
        return write_summary('complete')
    except Exception as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_summary('failed', metadata['error'])
        raise
    finally:
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')


def main():
    args = parser().parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
