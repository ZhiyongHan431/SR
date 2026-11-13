"""Immutable materialized SR occurrences and independent original-U randomness."""
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Invalid SHA256 hash')


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid nonnegative integer')


def load_view_manifest(path):
    """Validate coverage once per process; verify image bytes when consumed."""
    path = Path(path).resolve()
    stat = path.stat()
    return _load_manifest(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _load_manifest(path, mtime_ns, size):
    path = Path(path)
    value = json.loads(path.read_text())
    if value.get('schema_version') != 1 or value.get('status') != 'complete' or value.get('corruption') != 'original_U':
        raise ValueError('Paired views require a complete original_U schema 1 manifest')
    for key in ('source_manifest_sha256', 'split_sha256', 'reference_code_sha256', 'degradation_config_sha256'):
        _digest(value.get(key))
    for key in ('train_unique', 'validation_unique', 'train_occurrences', 'positions_per_phase'):
        _integer(value.get(key), 1)
    if value['train_occurrences'] % value['positions_per_phase']:
        raise ValueError('Incomplete phase coverage')
    if json_hash(value.get('split')) != value['split_sha256']:
        raise ValueError('Split hash mismatch')
    owners, originals, views, partitions = {}, {}, set(), {}
    for partition, count in [('train', value['train_occurrences']), ('validation', value['validation_unique'])]:
        rows = value.get(partition)
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError('Incomplete occurrence coverage')
        source_ids = set()
        for index, row in enumerate(rows):
            if type(row.get('occurrence_id')) is not int or row['occurrence_id'] != index:
                raise ValueError('Occurrences must have contiguous ordered zero-based IDs')
            if not isinstance(row.get('view_id'), str) or not row['view_id'] or row['view_id'] in views:
                raise ValueError('Duplicate or invalid view ID')
            views.add(row['view_id'])
            for key in ('view_seed', 'corruption_seed'):
                _integer(row.get(key))
                if row[key] >= 2**63: raise ValueError('Seed exceeds signed 64-bit range')
            source = row.get('source_id')
            if not isinstance(source, str) or not source: raise ValueError('Missing source ID')
            for key in ('source_sha256', 'decoded_rgb_sha256', 'hr_sha256', 'lr_sha256'):
                _digest(row.get(key))
            for key in ('source_path', 'hr_path', 'lr_path'):
                if not isinstance(row.get(key), str) or not row[key]: raise ValueError('Missing image path')
                row[key] = str((path.parent / row[key]).resolve())
            identity = tuple(row[key] for key in ('source_path', 'source_sha256', 'decoded_rgb_sha256'))
            if source in originals and originals[source] != identity:
                raise ValueError('Source identity changed between views')
            originals[source] = identity
            for kind, ident in zip(('canonical', 'bytes', 'rgb'), identity):
                owner = owners.setdefault((kind, ident), source)
                if owner != source: raise ValueError('Distinct source IDs alias the same original')
            source_ids.add(source)
        if len(source_ids) != value[partition + '_unique']:
            raise ValueError('Unique original count mismatch')
        if set(value['split'].get(partition, [])) != source_ids or len(value['split'][partition]) != len(source_ids):
            raise ValueError('Split does not match occurrence sources')
        partitions[partition] = source_ids
    if partitions['train'] & partitions['validation']:
        raise ValueError('Validation/train original overlap')
    return value


def phase_rank_rows(manifest, phase, rank, world_size):
    for value, minimum in [(phase, 0), (rank, 0), (world_size, 1)]: _integer(value, minimum)
    count = manifest['positions_per_phase']
    if rank >= world_size or count % world_size or phase >= manifest['train_occurrences'] // count:
        raise ValueError('Invalid phase/rank or incomplete rank coverage')
    return manifest['train'][phase * count:(phase + 1) * count][rank::world_size]


def load_view_images(row):
    images = []
    for kind, size in [('hr', 512), ('lr', 128)]:
        data = Path(row[kind + '_path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row[kind + '_sha256']:
            raise ValueError(f'{kind} image hash mismatch')
        with Image.open(io.BytesIO(data)) as image:
            if image.mode != 'RGB' or image.size != (size, size):
                raise ValueError(f'Invalid saved {kind} RGB geometry')
            array = np.array(image, copy=True)
        images.append(torch.from_numpy(array).permute(2, 0, 1).float().div_(255).mul_(2).sub_(1))
    return images[0].unsqueeze(0), images[1]


def occurrence_generators(row, device):
    seed = row['corruption_seed']
    _integer(seed)
    if seed >= 2**63: raise ValueError('Seed exceeds signed 64-bit range')
    return np.random.default_rng(seed), torch.Generator(device=device).manual_seed(seed)


def validate_view_training(args, world_size):
    """Fail before model allocation when paired-view startup is inconsistent."""
    if not getattr(args, 'sr_view_manifest', ''): return None
    if getattr(args, 'sr_state_manifest', '') or getattr(args, 'sr_state_arm', ''):
        raise ValueError('sr_view_manifest is mutually exclusive with state training')
    if (not getattr(args, 'sr_condition', 0) or getattr(args, 'sr_hr_size', 512) != 512
            or getattr(args, 'sr_lr_size', 128) != 128 or getattr(args, 'pair_input', False)
            or getattr(args, 'loop_data_per_epoch', 1) != 1):
        raise ValueError('Paired views require fixed 512/128 SR, one image and one data pass')
    manifest = load_view_manifest(args.sr_view_manifest)
    accumulation = getattr(args, 'gradient_accumulation', 1)
    _integer(accumulation, 1)
    _integer(world_size, 1)
    if manifest['positions_per_phase'] % (world_size * accumulation):
        raise ValueError('Paired phases must be divisible by world size and accumulation')
    restricted = getattr(args, 'restrict_data_size', -1)
    if restricted > 0 and restricted != manifest['positions_per_phase']:
        raise ValueError('restrict_data_size must equal paired positions_per_phase')
    if getattr(args, 'ep', manifest['train_occurrences'] // manifest['positions_per_phase']) != manifest['train_occurrences'] // manifest['positions_per_phase']:
        raise ValueError('Training horizon must match paired-view phases')
    return manifest


def append_consumption_audit(args, meta, tensors, actual_signal):
    if not getattr(args, 'sr_view_manifest', ''): return
    from tools.sr_refinement_diagnostics import tensor_hash
    row = meta['sr_view']
    audit = {key: row[key] for key in ('occurrence_id', 'source_id', 'source_sha256',
        'decoded_rgb_sha256', 'view_id', 'view_seed', 'corruption_seed', 'hr_sha256', 'lr_sha256')}
    audit.update(phase=meta['sr_view_phase'], rank=int(getattr(args, 'rank', 0)),
                 scheduled_signal=float(meta['sr_view_scheduled_signal']), actual_signal=float(actual_signal))
    for key in ('target_labels', 'random_labels', 'mask', 'corrupted_input'):
        audit[key + '_sha256'] = tensor_hash(tensors[key])
    directory = Path(getattr(args, 'sr_view_audit_dir', '') or getattr(args, 'local_out_path', '.'))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f'sr_view_audit_rank{audit["rank"]}.jsonl').open('a') as handle:
        handle.write(json.dumps(audit, sort_keys=True) + '\n')
