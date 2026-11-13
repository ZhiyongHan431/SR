"""Qualified full-coverage S5 data primitives; not a training launch or GPU gate.

Requires the final pinned production qualification marker, not status.json.
Each image read hashes and decodes the same bytes, with no sample substitution.
Crop parameters are explicit research configuration, not selected defaults.
CPU RNG scope is intended for one sequential loader call per worker process.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image, ImageFile
import torch
from grn.utils_t2iv.s5_bundle_paths import resolve_path, mapped_path


SCHEMA = 'grn_sr_s5_qualified_originals_v1'
EXPECTED_CANDIDATES = 94991
EXPECTED_GROUPS = {'validation': 1024, 'old_holdout': 96, 'benchmark': 386}
IDENTITY_KEYS = ('source_id', 'source_path', 'source_sha256', 'decoded_rgb_sha256')
ARTIFACT_NAMES = {'inventory': 'inventory.jsonl', 'protected': 'protected_originals.json',
                  'train': 'qualified_train_originals.json', 'summary': 'qualification_summary.json'}
METADATA_LIMIT = 512 * 1024 * 1024


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, 'Invalid integer')
    return value


def int64(value, minimum=0):
    integer(value, minimum)
    require(value < 2**63, 'Integer exceeds signed 64-bit range')
    return value


def digest(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA256')
    return value


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def parse(raw):
    def invalid_constant(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=invalid_constant)


def checked_bytes(path, expected_sha, maximum, *, expected_path=None):
    digest(expected_sha)
    integer(maximum, 1)
    path = resolve_path(path)
    resolved = path.resolve(strict=True)
    if expected_path is not None:
        require(str(resolved) == str(mapped_path(expected_path)), 'Pinned canonical file path changed')
    with resolved.open('rb') as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, 'Input exceeds declared byte limit')
    require(sha(raw) == expected_sha, f'File content changed: {path}')
    require(path.resolve(strict=True) == resolved, 'File symlink changed while reading')
    return raw


def checked_binding(binding, *, base=None):
    require(isinstance(binding, dict), 'Missing file binding')
    path = Path(binding['path'])
    if base is not None:
        require(not path.is_absolute() and '..' not in path.parts and str(path) == binding['path'],
                'Artifact path must be normalized and relative')
        path = base / path
        require(path.resolve(strict=True).is_relative_to(base.resolve()), 'Artifact escapes output directory')
    else:
        require(path.is_absolute(), 'Provenance input must have an absolute path')
    raw = checked_bytes(path, binding['sha256'], METADATA_LIMIT)
    if 'bytes' in binding:
        require(integer(binding['bytes']) == len(raw), 'Artifact byte count differs')
    return raw


def validate_identity(row, protocol_root):
    require(isinstance(row, dict), 'Image identity must be an object')
    source = row['source_id']
    require(isinstance(source, str) and source and '\\' not in source, 'Invalid source ID')
    value = Path(source)
    require(not value.is_absolute() and '..' not in value.parts and str(value) == source,
            'Source ID must be normalized protocol-relative path')
    require(Path(row['source_path']).is_absolute(), 'Source identity must remain absolute')
    local_root = resolve_path(protocol_root, strict=True)
    expected = mapped_path(row['source_path'])
    require(expected.is_relative_to(local_root), 'Source path escapes protocol root')
    require((local_root / value).resolve(strict=True) == expected, 'Source canonical mapping changed')
    for key in ('source_sha256', 'decoded_rgb_sha256'):
        digest(row[key])


@dataclass(frozen=True)
class QualifiedSources:
    manifest_path: str
    manifest_sha256: str
    protocol_root: str
    rows: tuple
    max_source_bytes: int
    max_pixels: int

    def __post_init__(self):
        object.__setattr__(self, '_identity_by_ordinal', {
            row['ordinal']: tuple(row[key] for key in IDENTITY_KEYS) for row in self.rows})


def load_qualified_manifest(path, expected_sha):
    """Full artifact integrity/isolation audit once at startup, no image decoding."""
    path = resolve_path(path, strict=True)
    require(path.name == 'qualified_manifest.json', 'Final qualification marker required')
    manifest = parse(checked_bytes(path, expected_sha, 2 * 1024 * 1024))
    require(manifest.get('schema') == SCHEMA and manifest.get('status') == 'complete'
            and manifest.get('profile') == 'production_94991', 'Complete production qualification required')
    # Keep the archived root string as identity; relocate only filesystem access.
    root = Path(manifest['protocol_root'])
    require(root.is_absolute() and '..' not in root.parts and str(root) == manifest['protocol_root'],
            'Protocol root must be canonical')
    resolve_path(root, strict=True)
    inputs = {name: checked_binding(manifest['inputs'][name])
              for name in ('canonical', 'views', 'history', 'benchmarks', 'data_report')}
    # Scanner uses exact splitlines, not strip/comment filtering. Duplicates
    # stay in the inventory and are rejected there, preserving ordinal identity.
    canonical = inputs['canonical'].decode('utf-8').splitlines()
    require(len(canonical) == EXPECTED_CANDIDATES and all(canonical),
            'Canonical candidate coverage differs')
    checked_binding(manifest['scanner'])
    artifacts = manifest['artifacts']
    require(set(artifacts) == set(ARTIFACT_NAMES), 'Qualification artifact membership differs')
    raw = {}
    for key, filename in ARTIFACT_NAMES.items():
        require(artifacts[key]['path'] == filename, 'Unexpected qualification artifact filename')
        raw[key] = checked_binding(artifacts[key], base=path.parent)
    train, protected, summary = (parse(raw[k]) for k in ('train', 'protected', 'summary'))
    require(train['schema'] == protected['schema'] == SCHEMA and train['protocol_root'] == str(root),
            'Qualified data schema/root differs')
    require(manifest['summary'] == summary, 'Summary differs from commit marker')
    count = integer(train['count'], 1)
    require(count == len(train['rows']) == summary['qualified_count'], 'Actual qualified count differs')
    require(summary['candidate_count'] == EXPECTED_CANDIDATES
            and summary['protected_count'] == sum(EXPECTED_GROUPS.values())
            and summary['rejected_count'] == EXPECTED_CANDIDATES - count, 'Qualified coverage differs')
    groups = {name: 0 for name in EXPECTED_GROUPS}
    forbidden, protected_ids = set(), set()
    for row in protected['rows']:
        validate_identity(row, root)
        require(row['group'] in groups and row['protected_id'] not in protected_ids,
                'Invalid/duplicate protected identity')
        groups[row['group']] += 1
        protected_ids.add(row['protected_id'])
        forbidden.update((key, row[key]) for key in IDENTITY_KEYS)
        checked_bytes(root / row['source_id'], row['source_sha256'],
                      integer(manifest['execution']['max_source_bytes'], 1), expected_path=row['source_path'])
    require(groups == EXPECTED_GROUPS, 'Protected groups may not shrink')
    accepted, seen = [], set()
    for line_number, line in enumerate(raw['inventory'].splitlines()):
        row = parse(line)
        require(type(row['ordinal']) is int and row['ordinal'] == line_number, 'Inventory ordinals incomplete')
        require(line_number < len(canonical) and row['source_id'] == canonical[line_number],
                'Inventory source differs from pinned canonical ordinal')
        require(row['decision'] in ('accepted', 'rejected'), 'Unknown qualification decision')
        if row['decision'] == 'accepted':
            require(row.get('reasons') == [], 'Accepted identity has rejection reasons')
            accepted.append({key: row[key] for key in ('ordinal',) + IDENTITY_KEYS})
    require(len(raw['inventory'].splitlines()) == EXPECTED_CANDIDATES and accepted == train['rows'],
            'Qualified train list differs from complete inventory')
    for row in accepted:
        validate_identity(row, root)
        keys = {(key, row[key]) for key in IDENTITY_KEYS}
        require(keys.isdisjoint(forbidden), 'Protected identity leaked into training')
        require(keys.isdisjoint(seen), 'Exact training duplicate was accepted')
        seen.update(keys)
    contract = manifest['runtime_contract']
    for name in ('fail_fast_on_file_hash_or_decode_error', 'no_recursive_substitution',
                 'verify_manifest_and_all_artifact_hashes', 'protected_revalidated_at_start_and_end'):
        require(contract.get(name) is True, 'Required qualification runtime contract missing')
    require(contract['sr_expected_train_samples'] == count and contract['source_paths_relative_to'] == 'protocol_root'
            and contract['candidate_files_globally_atomic_snapshot'] is False, 'Runtime data contract differs')
    execution = manifest['execution']
    require(execution['original_files_written'] is False and execution['gpu_used'] is False
            and set(execution['image_formats']) == {'BMP', 'JPEG', 'PNG', 'WEBP'}, 'Unsupported qualification process')
    return QualifiedSources(str(path), expected_sha, str(root), tuple(accepted),
                            integer(execution['max_source_bytes'], 1), integer(execution['max_pixels'], 1))


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json_bytes(parts)).digest()[:8], 'little') & (2**63 - 1)


def epoch_occurrences(sources, *, epoch, seed, world_size, accumulation):
    """Global full-coverage order, padded to full optimizer windows; never drop originals."""
    for value, minimum in ((epoch, 0), (seed, 0), (world_size, 1), (accumulation, 1)):
        int64(value, minimum)
    rows = sources.rows
    require(rows, 'No qualified originals')
    order = list(range(len(rows)))
    random.Random(stable_seed(sources.manifest_sha256, seed, 'order', epoch)).shuffle(order)
    block = world_size * accumulation
    padded_count = ((len(rows) + block - 1) // block) * block
    int64(padded_count, 1)
    int64((epoch + 1) * padded_count - 1)
    result = []
    for position in range(padded_count):
        row = rows[order[position % len(order)]]
        key = (sources.manifest_sha256, seed, epoch, position, row['ordinal'])
        result.append(dict(row, epoch=epoch, global_position=position,
            epoch_seed=seed, world_size=world_size, accumulation=accumulation,
            padded_count=padded_count,
            occurrence_id=epoch * padded_count + position, rank=position % world_size,
            rank_position=position // world_size, padding=position >= len(rows),
            crop_seed=stable_seed(*key, 'crop'), degradation_seed=stable_seed(*key, 'degradation'),
            corruption_seed=stable_seed(*key, 'corruption')))
    return result


@contextmanager
def strict_image_decode(max_pixels):
    """Own decoder policy for this sequential worker call, including on failure.

    IQA/reference dataset imports can enable PIL's global truncated-image
    tolerance. Never inherit that policy for accepted training originals.
    """
    previous_limit = Image.MAX_IMAGE_PIXELS
    previous_truncated = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        Image.MAX_IMAGE_PIXELS = integer(max_pixels, 1)
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        yield
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_truncated


def read_qualified_rgb(sources, row):
    """Same-byte hash/decode, no resize/EXIF conversion and no recursive fallback."""
    root = Path(sources.protocol_root)
    validate_identity(row, root)
    require(sources._identity_by_ordinal.get(row.get('ordinal')) == tuple(row[key] for key in IDENTITY_KEYS),
            'Image is not an approved qualified original')
    raw = checked_bytes(root / row['source_id'], row['source_sha256'], sources.max_source_bytes,
                        expected_path=row['source_path'])
    with strict_image_decode(sources.max_pixels), Image.open(io.BytesIO(raw)) as opened:
        require(opened.format in ('BMP', 'JPEG', 'PNG', 'WEBP'), 'Unsupported source format')
        require(0 < opened.width * opened.height <= sources.max_pixels, 'Source exceeds pixel limit')
        image = opened.convert('RGB')
        actual_rgb = sha(f'RGB:{image.width}:{image.height}\0'.encode() + image.tobytes())
        require(actual_rgb == row['decoded_rgb_sha256'], 'Decoded original identity changed')
    return image


def dynamic_crop(image, *, seed, short_side_min, short_side_max, size=512):
    """Random short-side resize + aligned random crop; degrader owns horizontal flip."""
    for value in (seed, short_side_min, short_side_max, size):
        int64(value)
    require(size == 512 and size <= short_side_min <= short_side_max, 'Explicit valid 512 crop range required')
    require(isinstance(image, Image.Image) and image.mode == 'RGB', 'Expected RGB source')
    rng = random.Random(seed)
    side = rng.randint(short_side_min, short_side_max)
    scale = side / min(image.size)
    width, height = max(size, round(image.width * scale)), max(size, round(image.height * scale))
    left, top = rng.randrange(width - size + 1), rng.randrange(height - size + 1)
    # Resample only the selected source rectangle; avoid allocating an enormous
    # intermediate canvas for valid but very elongated original images.
    box = (left * image.width / width, top * image.height / height,
           (left + size) * image.width / width, (top + size) * image.height / height)
    crop = image.resize((size, size), Image.Resampling.LANCZOS, box=box)
    return crop, dict(algorithm='random_virtual_short_side_lanczos_crop_v1', crop_seed=seed,
                     short_side_range=[short_side_min, short_side_max], selected_short_side=side,
                     original_size=list(image.size), resized_size=[width, height],
                     source_rectangle=list(box),
                     crop_xywh=[left, top, size, size], crop_flip=False,
                     horizontal_flip_owner='reference_degrader')


@contextmanager
def cpu_random_scope(seed):
    """Own RNG and CPU execution width for one sequential degradation call.

    Fixed RNG alone is insufficient: reference floating-point reductions can
    yield different quantized LR pixels under DataLoader's one-thread worker
    versus a multi-thread main process. Match the worker's canonical width and
    restore caller settings, including on failure. This is not thread-safe for
    concurrent degradation calls in the same process (nor is the RNG scope).
    """
    int64(seed)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    previous_threads = torch.get_num_threads()
    try:
        if previous_threads != 1:
            torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.default_generator.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if previous_threads != 1:
            torch.set_num_threads(previous_threads)


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return sha(json_bytes([str(value.dtype), list(value.shape)]) + value.numpy().tobytes())


def validate_occurrence(sources, row):
    """Check routing and independent draw identities before opening source bytes."""
    for key in ('epoch', 'epoch_seed', 'global_position', 'occurrence_id', 'rank',
                'rank_position', 'crop_seed', 'degradation_seed', 'corruption_seed'):
        int64(row[key])
    for key in ('world_size', 'accumulation', 'padded_count'):
        int64(row[key], 1)
    require(type(row['padding']) is bool, 'Padding must be explicit boolean')
    count, world, acc = len(sources.rows), row['world_size'], row['accumulation']
    expected_count = ((count + world * acc - 1) // (world * acc)) * world * acc
    position = row['global_position']
    require(row['padded_count'] == expected_count and position < expected_count
            and row['rank'] == position % world and row['rank_position'] == position // world
            and row['occurrence_id'] == row['epoch'] * expected_count + position
            and row['padding'] == (position >= count), 'Occurrence routing/padding differs')
    integer(row['ordinal'])
    require(sources._identity_by_ordinal.get(row['ordinal']) == tuple(row[key] for key in IDENTITY_KEYS),
            'Occurrence is not an approved original')
    key = (sources.manifest_sha256, row['epoch_seed'], row['epoch'], position, row['ordinal'])
    for tag in ('crop', 'degradation', 'corruption'):
        require(row[tag + '_seed'] == stable_seed(*key, tag), 'Occurrence random seed differs')


def materialize_occurrence(sources, row, degrader, *, short_side_min, short_side_max):
    """Return production-shaped CPU HR[1,3,512,512]/LR[3,128,128] plus consumption identity."""
    validate_occurrence(sources, row)
    image = read_qualified_rgb(sources, row)
    crop, crop_record = dynamic_crop(image, seed=row['crop_seed'], short_side_min=short_side_min,
                                     short_side_max=short_side_max)
    array = np.asarray(crop, dtype=np.float32) / 255.
    with cpu_random_scope(row['degradation_seed']):
        hr, lr = degrader.degrade_process(array, resize_bak=False)
    for value, shape in ((hr, (1, 3, 512, 512)), (lr, (1, 3, 128, 128))):
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == torch.float32
                and tuple(value.shape) == shape and not value.requires_grad and torch.isfinite(value).all()
                and value.min() >= 0 and value.max() <= 1, 'Invalid reference degradation result')
    hr, lr = hr * 2. - 1., lr.squeeze(0) * 2. - 1.
    record = {key: row[key] for key in ('ordinal',) + IDENTITY_KEYS +
              ('epoch', 'global_position', 'occurrence_id', 'rank', 'rank_position', 'padding',
               'epoch_seed', 'world_size', 'accumulation', 'padded_count',
               'crop_seed', 'degradation_seed', 'corruption_seed')}
    record.update(qualification_manifest_sha256=sources.manifest_sha256, crop=crop_record,
                  hr_tensor_sha256=tensor_sha(hr), lr_tensor_sha256=tensor_sha(lr))
    return hr, lr, record
