"""Opt-in fixed-pair U/T/R training contracts; no model or RNG interventions.

Cache tensors use BLD; ``grid`` declares the original THW layout. Provenance
mask hashes use the sampler's original BDTHW tensors, and rng_sha256 stores
capture_sampling.model_rng_at_updates (50 entries). Hashing/auditing transfers
actual bits to CPU and adds synchronous work in enabled runs.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch


STATE_STEPS = (20, 35, 45, 50)
CAPTURE_STEPS = (1, 19, 20, 34, 35, 44, 45, 49, 50)
IDENTITY_HASHES = ('source_sha256', 'hr_sha256', 'lr_sha256', 'checkpoint_sha256', 'vae_sha256')


def tensor_hash(value):
    """Exactly tools.sr_refinement_diagnostics.tensor_hash, including layout/dtype."""
    value = value.detach().contiguous().cpu()
    return hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode()
                          + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _verified_file(path, sha):
    path = Path(path).resolve(strict=True)
    if not _hash(sha):
        raise ValueError(f'Invalid SHA256 identity for {path}')
    with path.open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    if actual != sha:
        raise ValueError(f'SHA256 identity mismatch: {path}')
    return str(path)


def _bits(value, shape=None, *, mask=False):
    if (not isinstance(value, torch.Tensor) or value.ndim != 3
            or value.dtype != (torch.bool if mask else torch.uint8)
            or any(d <= 0 for d in value.shape)
            or (shape is not None and tuple(value.shape) != tuple(shape))
            or (not mask and not torch.all((value == 0) | (value == 1)).item())):
        raise ValueError('Cache requires binary uint8 BLD bits / bool mask with exact matching shape')


def _original_layout(bits, grid):
    return bits.transpose(1, 2).reshape(bits.shape[0], bits.shape[2], *grid)


def _validate_pair_state(cache):
    if not isinstance(cache, dict) or cache.get('schema_version') != 1:
        raise ValueError('Unsupported state cache schema')
    for key in ('pair_id', 'source_id'):
        if not isinstance(cache.get(key), str) or not cache[key]:
            raise ValueError(f'Missing cache {key}')
    target = cache.get('target_bits')
    _bits(target)
    grid = cache.get('grid')
    if (not isinstance(grid, (list, tuple)) or len(grid) != 3
            or any(type(d) is not int or d <= 0 for d in grid)
            or math.prod(grid) != target.shape[1] or target.shape[0] != 1):
        raise ValueError('Invalid original cache grid')
    _bits(cache.get('initial_noise'), target.shape)
    provenance = cache.get('provenance', {})
    if (not isinstance(provenance, dict) or provenance.get('source_id') != cache['source_id']
            or provenance.get('rollout_steps') != 50 or type(provenance.get('seed')) is not int
            or provenance['seed'] < 0 or any(not _hash(provenance.get(k)) for k in IDENTITY_HASHES)
            or provenance.get('capture_counts') != {j: 1 for j in CAPTURE_STEPS}):
        raise ValueError('Invalid state cache provenance')
    for key in ('rng_sha256', 'mask_sha256'):
        hashes = provenance.get(key)
        if not isinstance(hashes, list) or len(hashes) != 50 or not all(map(_hash, hashes)):
            raise ValueError(f'Invalid rollout provenance {key}')
    states = cache.get('states')
    if not isinstance(states, dict) or set(states) != set(STATE_STEPS):
        raise ValueError('Missing/invalid cached state steps')
    for step, state in states.items():
        if not isinstance(state, dict):
            raise ValueError('Invalid cached state')
        for key in ('input_bits', 'previous_prediction'):
            _bits(state.get(key), target.shape)
        mask = state.get('previous_mask')
        _bits(mask, target.shape, mask=True)
        if tensor_hash(_original_layout(mask, grid)) != provenance['mask_sha256'][step - 2]:
            raise ValueError('Previous mask does not match actual previous-step provenance')
        expected = torch.where(mask, state['previous_prediction'], cache['initial_noise'])
        signal = state.get('signal_fraction')
        if (type(signal) not in (int, float) or not math.isfinite(signal)
                or signal != mask.float().mean().item()
                or not torch.equal(expected, state['input_bits'])):
            raise ValueError('Cached actual input/time does not reconstruct from previous state')
    return cache


def load_pair_state(path):
    """Load only CPU tensors/basic containers and validate every actual state."""
    return _validate_pair_state(torch.load(path, map_location='cpu', weights_only=True))


def _check_identity(cache, meta):
    if cache['pair_id'] != meta['state_pair_id'] or cache['source_id'] != meta['source_id']:
        raise ValueError('State cache pair/source identity mismatch')
    if any(cache['provenance'][key] != meta[key] for key in IDENTITY_HASHES):
        raise ValueError('State cache image/model provenance identity mismatch')


def load_state_manifest(path):
    """Resolve and verify all files, geometry and provenance before data iteration."""
    path = Path(path).resolve(strict=True)
    with path.open(encoding='utf-8') as handle:
        manifest = json.load(handle)
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != 1
            or manifest.get('status') != 'complete' or manifest.get('bit_shape') != [1, 1024, 256]
            or manifest.get('hr_size') != 512 or manifest.get('lr_size') != 128
            or manifest.get('rollout_steps') != 50 or manifest.get('state_steps') != list(STATE_STEPS)):
        raise ValueError('State manifest must declare complete fixed 512/128, 50-step BLD geometry')
    model_hashes = {}
    for name in ('checkpoint', 'vae'):
        identity = manifest.get(name, {})
        if not isinstance(identity, dict) or not identity.get('path'):
            raise ValueError(f'Missing manifest {name} identity')
        identity['path'] = _verified_file(path.parent / identity['path'], identity.get('sha256'))
        model_hashes[name + '_sha256'] = identity['sha256']
    records = manifest.get('pairs')
    if not isinstance(records, list) or not records:
        raise ValueError('State manifest has no pairs')
    seen = {key: set() for key in ('pair_id', 'source_id', 'source_path')}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('Invalid manifest pair record')
        for name in ('source', 'hr', 'lr', 'state_cache'):
            key = name if name == 'state_cache' else name + '_path'
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f'Missing pair {key}')
            record[key] = _verified_file(path.parent / record[key], record.get(name + '_sha256'))
        for key, values in seen.items():
            if not isinstance(record.get(key), str) or not record[key] or record[key] in values:
                raise ValueError(f'Duplicate or missing source/pair {key}')
            values.add(record[key])
        for name, size in (('hr', 512), ('lr', 128)):
            with Image.open(record[name + '_path']) as image:
                if image.format != 'PNG' or image.mode != 'RGB' or image.size != (size, size):
                    raise ValueError(f'Fixed {name} must be exact RGB PNG {size}x{size}')
                image.load()
        record.update(model_hashes)
        cache = load_pair_state(record['state_cache'])
        if tuple(cache['target_bits'].shape) != (1, 1024, 256) or list(cache['grid']) != [1, 32, 32]:
            raise ValueError('Production cache geometry mismatch')
        _check_identity(cache, {**record, 'state_pair_id': record['pair_id']})
    return manifest


def build_state_occurrences(records, *, epoch, seed, total, rank, world_size):
    """Construct one global stream before rank slicing, without global RNG draws."""
    if (not records or any(type(x) is not int for x in (epoch, seed, total, rank, world_size))
            or epoch < 0 or seed < 0 or total <= 0 or world_size <= 0 or not 0 <= rank < world_size):
        raise ValueError('Invalid state occurrence stream arguments')
    order = np.random.default_rng(seed + epoch).permutation(len(records))
    result = []
    for occurrence in range(rank, total, world_size):
        record = records[int(order[occurrence % len(order)])]
        result.append({**record, 'image_path': record['hr_path'],
            'state_occurrence': occurrence, 'state_epoch': epoch,
            'state_pair_id': record['pair_id'], 'state_cache_path': record['state_cache'],
            'state_candidate': occurrence % 2 == 1, 'state_step': STATE_STEPS[(occurrence // 2) % 4]})
    return result


def select_training_state(labels, original_input, original_signal, meta, arm):
    """Override only selected inputs; even U verifies the encoded GT identity."""
    if not arm:
        return original_input, original_signal, {}
    if arm not in ('U', 'T', 'R'):
        raise ValueError('State training arm must be U, T or R')
    if (labels.ndim != 5 or original_input.shape != labels.shape
            or not torch.all((labels == 0) | (labels == 1)).item()
            or not torch.all((original_input == 0) | (original_input == 1)).item()
            or not math.isfinite(original_signal) or not 0 <= original_signal <= 1):
        raise ValueError('State training requires binary BDTHW inputs and finite signal')
    cache = load_pair_state(_verified_file(meta['state_cache_path'], meta['state_cache_sha256']))
    _check_identity(cache, meta)
    target = labels.permute(0, 2, 3, 4, 1).reshape(labels.shape[0], -1, labels.shape[1]).detach().cpu().byte()
    if list(labels.shape[2:]) != list(cache['grid']) or not torch.equal(target, cache['target_bits']):
        raise ValueError('Re-encoded target bits/layout do not match cached HR target')
    occurrence = meta['state_occurrence']
    if type(occurrence) is not int or occurrence < 0:
        raise ValueError('Invalid state occurrence')
    candidate = occurrence % 2 == 1
    step = STATE_STEPS[(occurrence // 2) % 4]
    actual, signal, kind = original_input, original_signal, 'original'
    if candidate and arm in ('T', 'R'):
        state = cache['states'][step]
        bits = (torch.where(state['previous_mask'], cache['target_bits'], cache['initial_noise'])
                if arm == 'T' else state['input_bits'])
        actual = _original_layout(bits, cache['grid']).to(device=labels.device, dtype=labels.dtype)
        signal, kind = state['signal_fraction'], 'teacher' if arm == 'T' else 'rollout'
    canonical = actual.permute(0, 2, 3, 4, 1).reshape(target.shape).byte()
    audit = dict(arm=arm, state_candidate=candidate, candidate_step=step,
        state_kind=kind, cache_step=step if kind != 'original' else None,
        actual_signal=float(signal), target_bits_sha256=tensor_hash(target),
        input_bits_sha256=tensor_hash(canonical))
    return actual, float(signal), audit


def append_state_audit(args, meta, audit):
    """Append actual consumption in process order; I/O errors abort the run."""
    if not getattr(args, 'sr_state_manifest', ''):
        return
    folder = Path(args.local_out_path)
    folder.mkdir(parents=True, exist_ok=True)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else int(getattr(args, 'rank', 0))
    keys = ('state_occurrence', 'state_epoch', 'state_pair_id', 'source_id', 'source_path',
            'hr_path', 'lr_path', 'state_cache_path', 'state_cache_sha256', *IDENTITY_HASHES)
    row = {key: meta[key] for key in keys if key in meta}
    with (folder / f'sr_state_audit_rank{rank}.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({**row, **audit}, allow_nan=False, sort_keys=True) + '\n')


def load_fixed_pair_images(meta):
    """Return exact saved RGB pixels in the existing [-1,1] tensor convention."""
    cache = load_pair_state(_verified_file(meta['state_cache_path'], meta['state_cache_sha256']))
    _check_identity(cache, meta)
    images = []
    for name, size in (('hr', 512), ('lr', 128)):
        path = _verified_file(meta[name + '_path'], meta[name + '_sha256'])
        with Image.open(path) as image:
            if image.format != 'PNG' or image.mode != 'RGB' or image.size != (size, size):
                raise ValueError(f'Fixed {name} must be exact RGB PNG {size}x{size}')
            pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div_(255)
            images.append(pixels.add(pixels).add_(-1))
    return images[0].unsqueeze(0), images[1]
