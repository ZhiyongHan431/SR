"""Bounded, reversible instrumentation for the real GRN SR refinement loop."""
from contextlib import contextmanager, ExitStack
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import random

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def tensor_hash(value):
    value = value.detach().contiguous().cpu()
    return hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode()
                          + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_tree(value, device='cpu'):
    if torch.is_tensor(value):
        return value.detach().to(device).clone()
    if isinstance(value, (list, tuple)):
        return type(value)(tensor_tree(item, device) for item in value)
    if isinstance(value, dict):
        return {key: tensor_tree(item, device) for key, item in value.items()}
    return value


@contextmanager
def preserve_rng(model=None):
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    generator = getattr(model, 'rng', None)
    state = generator.get_state() if generator is not None else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        if generator is not None:
            generator.set_state(state)
            model.rng = generator


@contextmanager
def replace_method(obj, name, replacement):
    """Restore descriptor lookup, not just an equivalent bound-method object."""
    existed, previous = name in obj.__dict__, obj.__dict__.get(name)
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        if existed:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


def select_target_logits(logits, target_tokens, bits, mode, sequence_lengths, target_offset=None):
    if (len(sequence_lengths) != 2 or sum(sequence_lengths) != logits.shape[1]
            or logits.ndim != 3 or logits.shape[-1] != bits * 2
            or any(n < target_tokens + 1 for n in sequence_lengths)):
        raise ValueError('Invalid packed conditional/unconditional logits layout')
    from grn.models.sr_condition_modes import uses_lr_prefix
    offset = target_tokens if uses_lr_prefix(mode) else 0
    # Preserve the legacy public API for callers with unequal LR/HR lengths.
    if mode == 'prefix' and target_offset is None:
        offset = sequence_lengths[0] - target_tokens - 1
    if target_offset is not None:
        offset = target_offset
    if offset < 0 or any(offset + target_tokens >= length for length in sequence_lengths):
        raise ValueError('HR target offset exceeds packed sequence bounds')
    return logits[:, offset:offset + target_tokens].reshape(logits.shape[0], target_tokens, bits, 2)


@contextmanager
def capture_refinement(model, steps, bits):
    """Snapshot HR codes before embedding, actual time, layout/RoPE, and target logits.

    Only selected states are retained on CPU. No RNG draws or extra model calls
    occur inside the rollout. LR's direct word_embed call is deliberately ignored.
    """
    capture = SimpleNamespace(snapshots={}, initial_hash=None, rng_hashes=[], final_latent=None,
                              steps_seen=0, layout=None, condition_inputs=None)
    selected = set(steps)
    originals = {name: getattr(model, name) for name in
                 ('embeds_codes2input', 'get_logits_during_infer', 'summed_codes2images')}
    training = [(module, module.training) for module in model.modules()]
    attention = [(block.attn, block.attn.caching, block.attn.cached_k, block.attn.cached_v,
                  block.attn.cached_split_cond_uncond)
                 for block in getattr(model, 'unregistered_blocks', [])]
    current = {}

    def codes(value):
        capture.steps_seen += 1
        step = capture.steps_seen
        if step == 1:
            capture.initial_hash = tensor_hash(value)
        capture.rng_hashes.append(tensor_hash(model.rng.get_state()))
        current.clear()
        current['tokens'] = value.flatten(2).shape[-1]
        if step in selected:
            current['inputs'] = value.detach().flatten(2).transpose(1, 2).contiguous().cpu().clone()
        return originals['embeds_codes2input'](value)

    def time_hook(module, inputs):
        current['signal_fraction'] = float(inputs[0].detach().flatten()[0])

    def layout_hook(module, inputs, kwargs):
        lengths = kwargs['cu_seqlens'].detach().cpu().diff().tolist()
        if capture.layout is None:
            capture.layout = {'sequence_lengths': lengths,
                              'packed_rope': kwargs['rope2d_freqs_grid'].detach().cpu().clone()}
        elif capture.layout['sequence_lengths'] != lengths:
            raise RuntimeError('Refinement packed sequence layout changed')

    def logits(hidden, *args, **kwargs):
        result = originals['get_logits_during_infer'](hidden, *args, **kwargs)
        if capture.steps_seen in selected:
            if 'signal_fraction' not in current or capture.layout is None:
                raise RuntimeError('Missing actual time or layout capture')
            # A has no text and retains legacy unequal-prefix capture support.
            # B's following text must never be mistaken for extra prefix tokens.
            offset = (None if model.sr_condition_mode == 'prefix' else
                      current['tokens'] if model.sr_condition_mode == 'prefix_text' else 0)
            target = select_target_logits(result, current['tokens'], bits,
                model.sr_condition_mode, capture.layout['sequence_lengths'],
                target_offset=offset)
            capture.snapshots[capture.steps_seen] = {
                'inputs': current['inputs'], 'signal_fraction': current['signal_fraction'],
                'logits': target.detach().cpu().clone(), 'dtype': str(target.dtype)}
        return result

    def decode(vae, value, *args, **kwargs):
        if capture.final_latent is not None:
            raise RuntimeError('Expected exactly one sampled final decode')
        capture.final_latent = value.detach().cpu().clone()
        capture.rng_hashes.append(tensor_hash(model.rng.get_state()))
        return originals['summed_codes2images'](vae, value, *args, **kwargs)

    original_infer = getattr(model, 'autoregressive_infer', None)
    def infer(*args, **kwargs):
        capture.condition_inputs = tensor_tree({key: kwargs.get(key) for key in
            ('lr_condition', 'native_lr_input', 'native_lr_latent', 'label_B_or_BLT', 'scale_schedule')})
        return original_infer(*args, **kwargs)

    with preserve_rng(model), ExitStack() as stack:
        if original_infer is not None:
            stack.enter_context(replace_method(model, 'autoregressive_infer', infer))
        for name, replacement in [('embeds_codes2input', codes), ('get_logits_during_infer', logits),
                                   ('summed_codes2images', decode)]:
            stack.enter_context(replace_method(model, name, replacement))
        handles = [model.pt_embedder.register_forward_pre_hook(time_hook),
                   model.block_chunks[0].register_forward_pre_hook(layout_hook, with_kwargs=True)]
        try:
            yield capture
        finally:
            for handle in handles:
                handle.remove()
            for module, state in training:
                module.training = state
            for attn, caching, cached_k, cached_v, cached_split in attention:
                attn.caching, attn.cached_k, attn.cached_v = caching, cached_k, cached_v
                attn.cached_split_cond_uncond = cached_split


@contextmanager
def disable_condition(model, enabled=True):
    """Native: bypass adapter residuals. Prefix: zero encoded prefix including positions.

    The zero prefix is also passed to legacy cross attention by the real model;
    prefix length, sequence ordering, and native empty text remain unchanged.
    """
    counts = {}
    with ExitStack() as stack:
        if enabled and model.sr_condition_mode in ('native_adapter', 'spatial_control'):
            controls = (model.sr_native_adapters if model.sr_condition_mode == 'native_adapter'
                        else model.sr_spatial_controls)
            for name, module in controls.items():
                counts[name] = 0
                def bypass(module, inputs, output, name=name):
                    counts[name] += 1
                    return inputs[0]
                handle = module.register_forward_hook(bypass)
                stack.callback(handle.remove)
        elif enabled and model.sr_condition_mode in ('prefix', 'prefix_text'):
            counts['prefix'] = 0
            original = model.encode_lr_condition
            def zero(*args, **kwargs):
                result = original(*args, **kwargs)
                counts['prefix'] += 1
                if isinstance(result, tuple):
                    return (torch.zeros_like(result[0]), *result[1:])
                return torch.zeros_like(result)
            stack.enter_context(replace_method(model, 'encode_lr_condition', zero))
        elif enabled:
            raise ValueError('Unsupported conditioning mode')
        yield counts
        if enabled and (not counts or not all(counts.values())):
            raise RuntimeError('Requested condition-disabling hooks were not all invoked')


def require_identical_noise(reference, observed):
    if reference['initial_hash'] is None or reference['initial_hash'] != observed['initial_hash']:
        raise RuntimeError('Initial HR noise mismatch invalidates causal comparison')
    if not reference['rng_hashes'] or reference['rng_hashes'] != observed['rng_hashes']:
        raise RuntimeError('Model RNG draws mismatch invalidates causal comparison')


def compare_logits(reference, replay):
    if reference.shape != replay.shape or not torch.isfinite(reference).all() or not torch.isfinite(replay).all():
        raise ValueError('Logits must have matching shapes and finite values')
    ref, got = reference.float(), replay.float()
    delta = (ref - got).abs()
    rms = float(ref.square().mean().sqrt())
    return dict(finite=True, max_absolute_error=float(delta.max()), mean_absolute_error=float(delta.mean()),
                rms_relative_error=float(delta.square().mean().sqrt()) / max(rms, 1e-12),
                bit_argmax_disagreement=float((ref.argmax(-1) != got.argmax(-1)).float().mean()),
                mean_probability_difference=float((ref.softmax(-1) - got.softmax(-1)).abs().mean()),
                max_probability_difference=float((ref.softmax(-1) - got.softmax(-1)).abs().max()))


def bit_metrics(logits, target):
    if logits.shape[:-1] != target.shape or not torch.isfinite(logits).all():
        raise ValueError('Expected finite [B,L,bits,2] logits and matching target')
    logits = logits.float()
    probs = logits.softmax(-1)
    return dict(ce=float(F.cross_entropy(logits.flatten(0, -2), target.long().flatten())),
                bit_error=float((logits.argmax(-1) != target).float().mean()),
                entropy_bits=float(-(probs * probs.clamp_min(1e-30).log2()).sum(-1).mean()))


def replay_forward(model, snapshot, target, visual_rope, text, condition, sequence_lengths, grid,
                   expected_packed_rope=None):
    """Real eval-mode model.forward parity; not FSDP/training-stack equivalence.

    GT is passed solely for the model's CE calculation after logits are computed.
    Its auxiliary logits are [B,classes,L,bits], not inference [B,L,bits,classes].
    """
    info = dict(cur_bits=target.shape[-1], cur_lvl=2, predict_tokens=target.shape[1],
                all_tokens=sequence_lengths[0], scale_token_id=snapshot['signal_fraction'])
    calls = []
    def verify_layout(module, inputs, kwargs):
        calls.append(True)
        if kwargs['cu_seqlens'].tolist() != [0, sequence_lengths[0]]:
            raise RuntimeError('Replay conditional sequence length differs from inference')
        if not torch.equal(kwargs['rope2d_freqs_grid'].detach().cpu(),
                           expected_packed_rope[:, :, :sequence_lengths[0]].cpu()):
            raise RuntimeError('Replay visual/text/time RoPE differs from inference')
    handle = (model.block_chunks[0].register_forward_pre_hook(verify_layout, with_kwargs=True)
              if expected_packed_rope is not None else None)
    try:
        outputs = model(text, [snapshot['inputs'].to(target.device)],
                        visual_rope_cache=[visual_rope], sequece_packing_scales=[[grid]],
                        super_scale_lengths=[sequence_lengths[0]], other_info_by_scale=[info],
                        gt_BL=[target], return_sr_aux=True, **condition)
        if handle is not None and not calls:
            raise RuntimeError('Replay layout verification hook was not invoked')
    finally:
        if handle is not None:
            handle.remove()
    logits = outputs[-1]['logits_by_scale'][0]
    if logits.shape != (target.shape[0], 2, target.shape[1], target.shape[2]):
        raise ValueError('Unexpected real training auxiliary logits layout')
    return logits.permute(0, 2, 3, 1).contiguous()


def teacher_snapshot(labels, signal, seed):
    from grn.utils_t2iv.hbq_util_t2iv import multiclass_labels2onehot_input
    if not 0 <= signal <= 1:
        raise ValueError('Teacher signal must be in [0,1]')
    generator = torch.Generator(device=labels.device).manual_seed(seed)
    noise = torch.randint(0, 2, labels.shape, device=labels.device, generator=generator)
    mask = torch.rand(labels.shape, device=labels.device, generator=generator) < signal
    mixed = torch.where(mask, labels, noise)
    return dict(inputs=multiclass_labels2onehot_input(mixed, 2).flatten(2).transpose(1, 2).contiguous(),
                signal_fraction=float(mask.float().mean()))


def verify_final_rgb(decoded_bgr, pipeline_rgb):
    decoded = decoded_bgr.detach().cpu()
    if decoded.ndim != 5 or decoded.shape[:2] != (1, 1) or decoded.shape[-1] != 3:
        raise RuntimeError('Unexpected final decode output shape')
    rgb = decoded[0, 0].flip(-1).numpy()
    if rgb.shape != pipeline_rgb.shape or not np.array_equal(rgb, pipeline_rgb):
        raise RuntimeError('Captured sampled latent decode differs from pipeline uint8 RGB')
    return 0


def fresh_output(path):
    Path(path).mkdir(parents=True, exist_ok=False)


def load_pairs(path, samples=0):
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get('schema_version') != 1 or not isinstance(payload.get('pairs'), list):
        raise ValueError('Pair JSON requires schema_version=1 and pairs list')
    rows = payload['pairs']
    if not rows or samples < 0 or samples > len(rows):
        raise ValueError('Invalid diagnostic pair count')
    seen_pairs, seen_sources = set(), set()
    for row in rows:
        if not all(isinstance(row.get(k), str) and row[k] for k in ('pair_id', 'source_id', 'hr', 'lr')):
            raise ValueError('Each pair requires nonempty pair_id, source_id, hr, lr')
        pair_id = row['pair_id']
        if (pair_id in ('.', '..') or Path(pair_id).name != pair_id
                or '\\' in pair_id or '\x00' in pair_id):
            raise ValueError('pair_id must be one safe filename component')
        if row['pair_id'] in seen_pairs or row['source_id'] in seen_sources:
            raise ValueError('Duplicate pair or source original ID')
        seen_pairs.add(row['pair_id'])
        seen_sources.add(row['source_id'])
        for kind, size in [('hr', 512), ('lr', 128)]:
            image_path = (path.parent / row[kind]).resolve()
            with Image.open(image_path) as image:
                if image.size != (size, size) or image.mode != 'RGB' or image.format != 'PNG':
                    raise ValueError(f'Invalid {kind} shape/mode/format: {image_path}')
                image.load()
            if row.get(kind + '_sha256') and file_hash(image_path) != row[kind + '_sha256']:
                raise ValueError(f'Pair {kind} hash mismatch')
            row[kind] = str(image_path)
    return rows[:samples] if samples else rows


def summarize_run(rows, expected, modes, status, completed_pairs, error=None,
                  expected_steps=None, teacher_signals=None):
    completed = set(completed_pairs)
    finals = {(row['pair_id'], row['mode']) for row in rows if row['kind'] == 'sampled_final'}
    if status == 'complete' and (len(completed) != expected or len(finals) != expected * len(modes)):
        raise ValueError('Incomplete diagnostic work cannot be marked complete')
    expected_records = None
    if expected_steps is not None and teacher_signals is not None:
        expected_records = expected * (len(modes) * (1 + len(expected_steps)) + len(expected_steps) + len(teacher_signals))
        if status == 'complete':
            wanted = set()
            for pair in completed:
                for mode in modes:
                    wanted.add((pair, 'sampled_final', mode, None, None))
                    wanted.update((pair, 'forward_parity', mode, step, None) for step in expected_steps)
                wanted.update((pair, 'argmax_prediction', 'correct', step, None) for step in expected_steps)
                wanted.update((pair, 'teacher_probe', 'correct', None, signal) for signal in teacher_signals)
            actual = {(r['pair_id'], r['kind'], r['mode'], r.get('step'), r.get('requested_signal_fraction')) for r in rows}
            if actual != wanted or len(rows) != expected_records:
                raise ValueError('Incomplete or duplicate parity/trajectory/teacher work')
    groups = defaultdict(list)
    for row in rows:
        groups[(row['kind'], row['mode'], row.get('step'), row.get('requested_signal_fraction'))].append(row)
    result = []
    for (kind, mode, step, signal), group in groups.items():
        entry = dict(kind=kind, mode=mode, step=step, requested_signal_fraction=signal, records=len(group))
        for field in ('metrics', 'bit_metrics', 'latent_metrics', 'parity', 'differences'):
            values = [r[field] for r in group if field in r]
            if values:
                entry[field] = {key: float(np.mean([value[key] for value in values]))
                                for key in values[0] if isinstance(values[0][key], (float, int))}
        for field in ('actual_signal_fraction', 'sampled_bit_error', 'pipeline_uint8_max_difference'):
            values = [r[field] for r in group if field in r]
            if values:
                entry[field + '_mean'] = float(np.mean(values))
        result.append(entry)
    return dict(status=status, expected_samples=expected, completed_samples=len(completed),
                expected_final_images=expected * len(modes), completed_final_images=len(finals),
                expected_records=expected_records, records=len(rows), groups=result, error=error)
