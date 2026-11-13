"""Single-thread/process-only reversible instrumentation of the real binary loop."""
from collections import defaultdict
from contextlib import ExitStack, contextmanager
import inspect
import threading
from pathlib import Path
from PIL import Image

import numpy as np
import torch

from tools.sr_refinement_diagnostics import capture_refinement, replace_method, tensor_hash


_capture_lock = threading.Lock()
_precision_lock = threading.Lock()


@contextmanager
def aligned_precision(model, enabled):
    """Match the eval training forward's embedding/time/head call boundaries.

    FsqHead owns its norm/projection autocast policy. Calling it with the original
    hidden dtype is intentional; casting its weights or the transformer is not.
    CPU autocast support exists for tiny-module tests of the same boundaries.
    """
    info = dict(enabled=bool(enabled), counts=dict(word_embed=0, pt_embedder=0, head=0),
        boundaries={'word_embed': 'autocast disabled; input.float()',
                    'pt_embedder': 'autocast disabled; input.float()',
                    'head': 'head(hidden_states, e), as get_loss_acc'},
        parity_scope='real model.forward eval only; not FSDP/training-stack equivalence')
    if not enabled:
        yield info
        return
    if not _precision_lock.acquire(blocking=False):
        raise RuntimeError('Precision context reentry/concurrent use is unsupported')
    try:
        with ExitStack() as stack:
            for name in ('word_embed', 'pt_embedder'):
                module = getattr(model, name)
                original = module.forward
                def boundary(value, *args, _name=name, _original=original, **kwargs):
                    info['counts'][_name] += 1
                    with torch.autocast(value.device.type, enabled=False):
                        return _original(value.float(), *args, **kwargs)
                stack.enter_context(replace_method(module, 'forward', boundary))
            def logits(hidden, e=None):
                info['counts']['head'] += 1
                return model.head(hidden, e)
            stack.enter_context(replace_method(model, 'get_logits_during_infer', logits))
            yield info
    finally:
        _precision_lock.release()


@contextmanager
def capture_sampling(model, selected_steps, bits, *, steps, stochastic, actual_seed):
    """Extend the existing real-loop capture with sampled labels and update masks.

    Only direct calls from this model's original autoregressive_infer code are
    intercepted. Calls from HBQ, decoder, other methods or threads pass through.
    No extra draws are made. The probability tensor is passed unchanged to the
    original multinomial, with only its generator replaced for stochastic arms.
    """
    if model.sr_condition_mode != 'prefix':
        raise ValueError('Sampling precision diagnostics support prefix only')
    if steps < 2 or bits < 1 or actual_seed < 0 or not set(selected_steps) <= set(range(1, steps + 1)):
        raise ValueError('Invalid capture steps, bit count or seed')
    original_where, original_multinomial = torch.where, torch.multinomial
    # The real entrypoint is decorated with torch.no_grad(). Match the body
    # issuing where/multinomial, not the decorator's context-manager frame.
    infer_code = inspect.unwrap(model.autoregressive_infer).__code__
    owner = threading.get_ident()
    rng = model.rng
    sampling_rng = torch.Generator(device=rng.device).manual_seed(1000000 + actual_seed)
    statistics = getattr(model, 'entrophy_statistics', None)
    statistics_length = len(statistics) if statistics is not None else None
    rope = getattr(model, 'rope2d_freqs_grid', None)
    rope_entries = dict(rope) if rope is not None else None
    current = dict(pending=False, shape=None, sampled=False)
    if not _capture_lock.acquire(blocking=False):
        raise RuntimeError('Sampling capture already active; reentry/concurrency unsupported')
    try:
        with capture_refinement(model, selected_steps, bits) as cap, ExitStack() as stack:
            cap.multinomial_calls = 0
            cap.sampling_seed = 1000000 + actual_seed
            cap.sampling_rng_hashes = [tensor_hash(sampling_rng.get_state())]
            cap.update_mask_hashes, cap.selected_bit_hashes, cap.update_counts = [], [], []
            cap.model_rng_at_updates = []
            original_codes = model.embeds_codes2input
            original_logits = model.get_logits_during_infer

            def codes(value):
                nonlocal rng, sampling_rng
                if current['pending']:
                    raise RuntimeError('Missing previous update mask capture')
                if value.ndim != 5 or value.shape[1] != bits * 2:
                    raise ValueError('Unexpected binary one-hot input shape')
                if cap.steps_seen == 0:
                    # Real infer may replace model.rng when its device changes.
                    rng = model.rng
                    sampling_rng = torch.Generator(device=rng.device).manual_seed(1000000 + actual_seed)
                    cap.sampling_rng_hashes = [tensor_hash(sampling_rng.get_state())]
                elif model.rng is not rng:
                    raise RuntimeError('Model RNG changed during refinement')
                current.update(shape=(value.shape[0], bits, *value.shape[2:]), sampled=False)
                return original_codes(value)

            def logits(hidden, *args, **kwargs):
                if current['shape'] is None or current['pending']:
                    raise RuntimeError('Invalid active-step logits context')
                result = original_logits(hidden, *args, **kwargs)
                current['pending'] = True
                return result

            def multinomial(probabilities, num_samples, replacement=False, *, generator=None, out=None):
                caller = inspect.currentframe().f_back
                relevant = (threading.get_ident() == owner and
                            caller.f_code is infer_code and caller.f_locals.get('self') is model)
                del caller
                if not relevant:
                    return original_multinomial(probabilities, num_samples, replacement, generator=generator, out=out)
                if (not stochastic or generator is not rng or model.rng is not rng or not current['pending']
                        or current['sampled'] or probabilities.shape != (int(np.prod(current['shape'])), 2)
                        or not probabilities.is_floating_point() or num_samples != 1 or not replacement or out is not None):
                    raise RuntimeError('Invalid real sampling generator/layout/active-step contract')
                result = original_multinomial(probabilities, num_samples, replacement, generator=sampling_rng)
                cap.multinomial_calls += 1
                current['sampled'] = True
                cap.sampling_rng_hashes.append(tensor_hash(sampling_rng.get_state()))
                return result

            def where(*args, **kwargs):
                caller = inspect.currentframe().f_back
                relevant = (threading.get_ident() == owner and
                            caller.f_code is infer_code and caller.f_locals.get('self') is model)
                del caller
                if not relevant:
                    return original_where(*args, **kwargs)
                if len(args) != 3 or kwargs or not current['pending']:
                    raise RuntimeError('Invalid real update mask active-step context')
                mask, selected, noise = args
                if (any(not torch.is_tensor(x) or tuple(x.shape) != current['shape'] for x in args)
                        or mask.dtype != torch.bool or selected.dtype != torch.int64
                        or noise.dtype not in (torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.int32)
                        or mask.device != selected.device or noise.device != selected.device):
                    raise ValueError('Invalid update mask/selected bits shape or dtype')
                if current['sampled'] != stochastic or not ((selected == 0) | (selected == 1)).all():
                    raise RuntimeError('Missing actual binary sampling or invalid labels')
                cap.update_mask_hashes.append(tensor_hash(mask))
                cap.selected_bit_hashes.append(tensor_hash(selected))
                cap.model_rng_at_updates.append(tensor_hash(rng.get_state()))
                cap.update_counts.append(dict(step=cap.steps_seen, n_total=mask.numel(), n_predict=int(mask.sum())))
                if cap.steps_seen in cap.snapshots:
                    cap.snapshots[cap.steps_seen].update(
                        selected_bits=selected.detach().flatten(2).transpose(1, 2).contiguous().cpu().clone(),
                        update_mask=mask.detach().flatten(2).transpose(1, 2).contiguous().cpu().clone())
                current['pending'] = False
                return original_where(*args, **kwargs)

            stack.enter_context(replace_method(model, 'embeds_codes2input', codes))
            stack.enter_context(replace_method(model, 'get_logits_during_infer', logits))
            torch.where, torch.multinomial = where, multinomial
            yield cap
            if (cap.steps_seen != steps or len(cap.update_mask_hashes) != steps
                    or cap.multinomial_calls != (steps if stochastic else 0)
                    or set(cap.snapshots) != set(selected_steps) or cap.final_latent is None
                    or any('selected_bits' not in s for s in cap.snapshots.values())):
                raise RuntimeError('Missing expected capture states or callback counts')
    finally:
        torch.where, torch.multinomial = original_where, original_multinomial
        if statistics is not None:
            del statistics[statistics_length:]
            model.entrophy_statistics = statistics
        if rope is not None:
            rope.clear()
            rope.update(rope_entries)
            model.rope2d_freqs_grid = rope
        _capture_lock.release()


def input_bits(onehot, bits):
    if onehot.ndim != 3 or onehot.shape[-1] != bits * 2:
        raise ValueError('Expected [B,L,bits*2] round-major one-hot input')
    value = onehot.reshape(*onehot.shape[:2], bits, 2)
    if not (((value == 0) | (value == 1)).all() and (value.sum(-1) == 1).all()):
        raise ValueError('Input is not binary one-hot')
    return value.argmax(-1)


def require_baseline_rgb(path, rgb):
    with Image.open(path) as image:
        reference = np.asarray(image)
        if image.mode != 'RGB' or reference.dtype != np.uint8 or not np.array_equal(reference, rgb):
            raise RuntimeError(f'Baseline uint8 RGB mismatch: {path}')
    return 0


def top1_selection_counts(logits, selected):
    """Audit the real probability argmax without changing dtype or drawing RNG.

    At the fixed guidance/temperature of one, conditional logits are the real
    sampler input. Softmax rounding can create ties absent in the raw logits.
    """
    if (logits.ndim != 4 or logits.shape[-1] != 2 or logits.shape[:-1] != selected.shape
            or not logits.is_floating_point() or logits.numel() == 0
            or logits.device != selected.device or not torch.isfinite(logits).all()
            or not ((selected == 0) | (selected == 1)).all()):
        raise ValueError('Expected finite [B,L,bits,2] logits and matching binary selections')
    raw_difference = int((logits.argmax(-1) != selected).sum())
    probability_difference = int((logits.softmax(-1).argmax(-1) != selected).sum())
    if probability_difference:
        raise ValueError(f'Actual top1 bits differ from real probability argmax at {probability_difference} positions')
    return dict(n_bits=selected.numel(), raw_logit_argmax_vs_actual_bit_count=raw_difference,
                probability_argmax_vs_actual_bit_count=probability_difference)


def legacy_baseline_evidence(reference_path, actual_rgb, logits, selected, decode_raw,
                             auxiliary_path, output_root):
    """Strictly reproduce old raw-logit images; report actual-image differences.

    The caller's actual PNG/metrics are never replaced. A raw-logit decode is
    needed only when its bits differ; that separate image is an auxiliary audit.
    """
    selection = top1_selection_counts(logits, selected)
    different_bits = selection['raw_logit_argmax_vs_actual_bit_count']
    raw_rgb = decode_raw(logits.argmax(-1)) if different_bits else actual_rgb
    exact = require_baseline_rgb(reference_path, raw_rgb)
    with Image.open(reference_path) as image:
        reference = np.asarray(image)
        if actual_rgb.shape != reference.shape or actual_rgb.dtype != np.uint8:
            raise ValueError('Actual baseline evidence requires matching uint8 RGB shape')
        delta = np.abs(actual_rgb.astype(np.int16) - reference.astype(np.int16))
    auxiliary_relative = None
    if different_bits:
        auxiliary_relative = str(Path(auxiliary_path).relative_to(output_root))
        Image.fromarray(raw_rgb).save(auxiliary_path)
    return dict(baseline_reference_kind='legacy_raw_logit_argmax',
        legacy_reconstruction_uint8_max_difference=exact,
        baseline_uint8_max_difference=int(delta.max()),
        baseline_changed_rgb_component_count=int(np.count_nonzero(delta)),
        baseline_changed_pixel_count=int(np.count_nonzero(np.any(delta != 0, axis=-1))),
        baseline_raw_logit_argmax_vs_actual_bit_count=different_bits,
        legacy_logit_argmax_image=auxiliary_relative)


def transition_metrics(inputs, selected, target, rounds):
    """Round-major [B,L,M*C] errors; rates use their at-risk populations."""
    if (inputs.shape != target.shape or selected.shape != target.shape or target.ndim != 3
            or rounds < 1 or target.shape[-1] % rounds or target.numel() == 0):
        raise ValueError('Expected matching nonempty round-major bit tensors')
    if any(not ((value == 0) | (value == 1)).all() for value in (inputs, selected, target)):
        raise ValueError('Expected binary labels')
    def count(x, y, gt):
        before, after = x != gt, y != gt
        total = gt.numel()
        wrong, output_wrong = int(before.sum()), int(after.sum())
        correction, damage = int((before & ~after).sum()), int((~before & after).sum())
        return dict(n_total=total, n_input_wrong=wrong, n_input_correct=total-wrong,
            n_output_wrong=output_wrong, n_wrong_to_correct=correction, n_correct_to_wrong=damage,
            wrong_correction_rate=correction/wrong if wrong else None,
            correct_damage_rate=damage/(total-wrong) if total-wrong else None,
            net_error_reduction=(correction-damage)/total, input_ber=wrong/total, output_ber=output_wrong/total)
    width = target.shape[-1] // rounds
    return dict(overall=count(inputs, selected, target), per_round=[
        dict(round_index=i, **count(inputs[..., i*width:(i+1)*width], selected[..., i*width:(i+1)*width],
                                   target[..., i*width:(i+1)*width])) for i in range(rounds)])


def validate_seed_bases(bases):
    if not bases or len(set(bases)) != len(bases) or any(type(v) is not int or v < 0 or v > 2**63-1000001 for v in bases):
        raise ValueError('Seed bases must be unique nonnegative bounded integers')


def validate_experiment(args, mode, seed_bases):
    from tools.diagnose_sr_refinement import validate_inference_args
    validate_seed_bases(seed_bases)
    validate_inference_args(args)
    if mode != 'prefix':
        raise ValueError('Only prefix conditioning is supported by this bounded experiment')


def summarize_sampling(rows, pairs, seed_bases, steps, status, completed, error=None):
    pair_indices = {p['pair_id']: i for i, p in enumerate(pairs)}
    if status == 'complete' and any(r.get('actual_seed') != r['seed_base'] + pair_indices.get(r['pair_id'], -1) for r in rows):
        raise ValueError('Invalid actual seed; expected seed_base + pair index')
    expected = set()
    for pair in pairs:
        for seed in seed_bases:
            expected.add((pair['pair_id'], seed, 'sampled_final', None))
            expected.update((pair['pair_id'], seed, kind, step)
                            for kind in ('sampled_prediction', 'forward_parity') for step in steps)
    actual = {(r['pair_id'], r['seed_base'], r['kind'], r.get('step')) for r in rows}
    wanted_completed = {(p['pair_id'], s) for p in pairs for s in seed_bases}
    finals = {r['image'] for r in rows if r['kind'] == 'sampled_final' and r.get('image')}
    trajectories = {r['image'] for r in rows if r['kind'] == 'sampled_prediction' and r.get('image')}
    count = len(wanted_completed)
    if status == 'complete' and (actual != expected or len(rows) != len(expected)
            or set(map(tuple, completed)) != wanted_completed or len(finals) != count
            or len(trajectories) != count * len(steps) or finals & trajectories):
        raise ValueError('Incomplete or duplicate diagnostic records/images cannot be complete')
    groups = defaultdict(list)
    for row in rows:
        groups[(row['seed_base'], row['kind'], row.get('step'))].append(row)
    summaries = []
    for (seed, kind, step), group in groups.items():
        entry = dict(seed_base=seed, kind=kind, step=step, original_images=len(group))
        for field in ('metrics', 'bit_metrics', 'latent_metrics', 'parity'):
            values = [r[field] for r in group if field in r]
            if values:
                entry[field] = {key: float(np.mean([v[key] for v in values if v.get(key) is not None]))
                               for key in values[0] if isinstance(values[0][key], (float, int))}
        summaries.append(entry)
    return dict(status=status, expected_sample_seeds=count, completed_sample_seeds=len(set(map(tuple, completed))),
        expected_original_images=len(pairs), expected_records=len(expected), records=len(rows),
        completed_unique_records=len(actual), expected_final_images=count, completed_final_images=len(finals),
        expected_trajectory_images=count*len(steps), completed_trajectory_images=len(trajectories),
        groups=summaries, error=error,
        aggregation='Per-seed means over originals. Average seeds per original before paired original-image bootstrap.')
