#!/usr/bin/env python3
"""Qualified fixed-pair, fixed-teacher trajectory adaptation.

prepare -> cache -> train. Nothing runs without --execute. The cache is a
frozen teacher's actual 50-step trajectory, NOT the changing student's on-policy
trajectory. Cached frozen-backbone features make bounded C/R adaptation cheap;
all claimed quality improvements still require complete student inference.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import shutil
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / 'scripts', ROOT / 'code'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from _bundle import CONFIG, bundled, setup
setup()

SCHEMA = 's5_trajectory_adaptation_v1'
STEPS = (20, 35, 45, 50)
CAPTURE_STEPS = (1, 19, 20, 34, 35, 44, 45, 49, 50)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def execution_identity():
    import torch
    names = ['code/tools/trajectory_adaptation.py', 'code/grn/models/grn.py',
        'code/grn/models/sr_v2_calibrator.py', 'code/grn/models/sr_latent_residual.py',
        'code/grn/utils_t2iv/sr_trajectory_adapter.py', 'code/tools/grn_pipeline.py',
        'code/grn/utils_t2iv/sr_v2_image_loss.py']
    return {'source_sha256': {name:sha256(ROOT/name) for name in names if (ROOT/name).is_file()},
        'numeric_policy': {'torch':str(torch.__version__),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cudnn_enabled':torch.backends.cudnn.enabled,
            'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def local_path(path):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('All adaptation assets must stay inside the S5 bundle')
    return path


def fresh_output(path):
    path = local_path(path)
    if not any(path.is_relative_to(ROOT / name) for name in ('outputs', 'verification')):
        raise ValueError('Outputs must be under outputs/ or verification/')
    path.mkdir(parents=True, exist_ok=False)
    return path


def split_qualified_rows(rows, train_count, validation_count, seed):
    """Stable source-level split, with exact-content isolation."""
    if min(train_count, validation_count) < 1 or train_count + validation_count > len(rows):
        raise ValueError('Invalid train/validation counts')
    ordered = sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{r['source_id']}".encode()).digest())
    validation = ordered[:validation_count]
    train = ordered[validation_count:validation_count + train_count]
    for key in ('source_id', 'source_sha256', 'decoded_rgb_sha256'):
        if len({r[key] for r in train + validation}) != len(train) + len(validation):
            raise ValueError('Source or content duplicate across adapter splits')
    return train, validation


def prepare(args):
    import numpy as np
    from PIL import Image
    from grn.utils_t2iv import sr_v2_qualified_data as q
    from grn.utils_t2iv.s5_bundle_paths import resolve_path
    from tools.prepare_sr_diagnostic_pairs import reference_engine
    output = fresh_output(args.output)
    status = {'schema': SCHEMA, 'status': 'running', 'stage': 'prepare'}
    write_json(output / 'status.json', status)
    try:
        manifest = ROOT / 'manifests/qualification_v1/qualified_manifest.json'
        original = json.loads((ROOT / 'manifests/s5_training_args.json').read_text())
        sources = q.load_qualified_manifest(manifest, original['original_qualified_manifest_sha256'])
        train, validation = split_qualified_rows(list(sources.rows), args.train_count, args.validation_count, args.seed)
        protocol = resolve_path(sources.protocol_root, strict=True)
        recipe = protocol / 'dataloader/params_realesrgan.yml'
        engine = reference_engine(protocol, recipe)
        payload = {'schema': SCHEMA, 'status': 'complete', 'qualification_manifest_sha256': sources.manifest_sha256,
                   'degradation': 'packaged_reference_RealESRGAN', 'degradation_config_sha256': sha256(recipe),
                   'split_seed': args.seed, 'hr_size': 512, 'lr_size': 128, 'execution_identity': execution_identity(),
                   'scope': 'Fixed qualified-source crops and degradations for adapter experiments; not a full dynamic-data epoch.',
                   'validation_scope': 'Source-disjoint from this adapter training subset; the S5 base was already trained on qualified sources.',
                   'splits': {'train': [], 'validation': []}}
        for role, rows in (('train', train), ('validation', validation)):
            (output / role / 'hr').mkdir(parents=True)
            (output / role / 'lr').mkdir()
            for index, row in enumerate(rows):
                crop_seed = q.stable_seed(args.seed, row['source_id'], 'crop')
                degradation_seed = q.stable_seed(args.seed, row['source_id'], 'degradation')
                image = q.read_qualified_rgb(sources, row)
                crop, crop_meta = q.dynamic_crop(image, seed=crop_seed, short_side_min=args.crop_min, short_side_max=args.crop_max)
                with q.cpu_random_scope(degradation_seed):
                    hr, lr = engine.degrade_process(np.asarray(crop, dtype=np.float32) / 255., resize_bak=False)
                paths = {}
                for name, tensor, size in (('hr', hr, 512), ('lr', lr, 128)):
                    if tuple(tensor.shape) != (1, 3, size, size) or not bool(tensor.isfinite().all()):
                        raise ValueError('Reference degradation returned invalid image')
                    path = output / role / name / f'pair_{index:05d}.png'
                    pixels = tensor[0].permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().numpy()
                    Image.fromarray(pixels).save(path)
                    paths[name] = str(path.relative_to(output))
                    paths[name + '_sha256'] = sha256(path)
                record = {key: row[key] for key in ('ordinal', 'source_id', 'source_sha256', 'decoded_rgb_sha256')}
                record.update(paths, pair_id=f'{role}_{index:05d}', source_path=str(resolve_path(row['source_path'], strict=True)),
                              split=role, crop=crop_meta, degradation_seed=degradation_seed, seed=args.seed + index)
                payload['splits'][role].append(record)
                print('PREPARED', role, index, row['source_id'], flush=True)
        write_json(output / 'pairs.json', payload)
        status.update(status='complete', pairs_sha256=sha256(output / 'pairs.json'))
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'status.json', status)
    return output / 'pairs.json'


def qualified_index():
    """Revalidate the committed allowed-source list for every consumer startup."""
    original = json.loads((ROOT/'manifests/s5_training_args.json').read_text())
    marker = ROOT/'manifests/qualification_v1/qualified_manifest.json'
    if sha256(marker) != original['original_qualified_manifest_sha256']:
        raise ValueError('Qualification marker changed')
    manifest = json.loads(marker.read_text())
    artifact = marker.parent/manifest['artifacts']['train']['path']
    if sha256(artifact) != manifest['artifacts']['train']['sha256']:
        raise ValueError('Qualified source list changed')
    accepted = json.loads(artifact.read_text())['rows']
    return original['original_qualified_manifest_sha256'], {r['source_id']:r for r in accepted}


def load_pairs(path, split='train'):
    path = local_path(path)
    value = json.loads(path.read_text())
    if value.get('schema') != SCHEMA or value.get('status') != 'complete' or split not in ('train', 'validation'):
        raise ValueError('Expected complete adaptation pair manifest')
    qualification_sha, allowed = qualified_index()
    if value.get('qualification_manifest_sha256') != qualification_sha:
        raise ValueError('Pair manifest qualification mismatch')
    from grn.utils_t2iv.s5_bundle_paths import resolve_path
    all_rows = value['splits']['train'] + value['splits']['validation']
    for row in all_rows:
        approved = allowed.get(row['source_id'])
        if approved is None or any(row[k] != approved[k] for k in ('source_sha256','decoded_rgb_sha256')):
            raise ValueError('Unqualified or test-set source in adaptation pairs')
        if local_path(row['source_path']) != resolve_path(approved['source_path'], strict=True):
            raise ValueError('Qualified source canonical path mismatch')
    for key in ('source_id', 'source_sha256', 'decoded_rgb_sha256'):
        if len({r[key] for r in all_rows}) != len(all_rows):
            raise ValueError('Pair manifest contains source/content overlap')
    rows = []
    for original in value['splits'][split]:
        row = dict(original)
        if row['split'] != split:
            raise ValueError('Split label mismatch')
        for key in ('hr', 'lr', 'source_path'):
            p = local_path(path.parent / row[key])
            expected = row['source_sha256'] if key == 'source_path' else row[key + '_sha256']
            if sha256(p) != expected:
                raise ValueError('Pair/source hash mismatch: ' + str(p))
            row[key] = str(p)
        rows.append(row)
    if not rows:
        raise ValueError('Empty requested split')
    return value, rows


def load_pipeline(device):
    import torch
    from tools.grn_pipeline import GRNPipeline
    pipe = GRNPipeline.from_pretrained(model_path=str(bundled(CONFIG['checkpoint'])),
        vae_path=str(bundled(CONFIG['hbq'])), sr_native_empty_text_path=str(bundled(CONFIG['empty_text'])),
        task='SR', model='GRN2b', device=torch.device(device), torch_dtype=torch.bfloat16)
    pipe.model.eval().requires_grad_(False)
    pipe.vae.eval().requires_grad_(False)
    return pipe


def protocol():
    return dict(guidance_scale=1., temperature=1., complexity_aware_Tmin=50,
        complexity_aware_Tmax=50, complexity_aware_k=0, complexity_aware_b=50,
        complexity_aware_wp=0, snr_shift=1., top_k=1, top_p=.75, output_type='numpy')


@contextmanager
def capture_teacher(pipe, *, constructed=None):
    """Capture the calibrated production sampler without altering random draws.

    Optional constructed inputs are an explicitly separate control run. They
    replace only selected backbone inputs, with original teacher masks/signals.
    """
    import torch
    from grn.utils_t2iv.hbq_util_t2iv import multiclass_labels2onehot_input
    from grn.utils_t2iv.sr_state_training import tensor_hash
    from tools.sr_refinement_diagnostics import replace_method, preserve_rng
    model = pipe.model
    cap = SimpleNamespace(snapshots={}, features={}, final_latent=None, steps_seen=0,
        multinomial_calls=0, update_mask_hashes=[], model_rng_at_updates=[], selected_bit_hashes=[], update_counts=[])
    codes = model.embeds_codes2input
    calibrate = model._calibrate_sr_hr_logits
    decode = model.summed_codes2images
    original_where = torch.where
    owner = threading.get_ident()
    infer_code = inspect.unwrap(model.autoregressive_infer).__code__
    current = {}
    def capture_codes(value):
        cap.steps_seen += 1
        if constructed is not None and cap.steps_seen in constructed:
            bits = constructed[cap.steps_seen].to(value.device)
            value = multiclass_labels2onehot_input(bits, 2)
        if cap.steps_seen in CAPTURE_STEPS:
            current['inputs'] = value.detach().flatten(2).transpose(1, 2).contiguous().cpu().clone()
        return codes(value)
    def capture_calibrate(logits, hidden, lr_latent, retention, *args, **kwargs):
        result = calibrate(logits, hidden, lr_latent, retention, *args, **kwargs)
        step = cap.steps_seen
        if step in CAPTURE_STEPS:
            cap.snapshots[step] = {'inputs': current['inputs'], 'signal_fraction': float(retention.item()),
                'logits': result.detach().permute(0, 2, 3, 1).contiguous().cpu()}
        if step in STEPS:
            cap.features[step] = {'hidden': hidden.detach().cpu().clone(), 'head_logits': logits.detach().cpu().clone(),
                'lr_features': lr_latent.detach().cpu().clone(), 'signal': retention.detach().float().cpu().clone()}
        return result
    def capture_where(*args, **kwargs):
        caller = inspect.currentframe().f_back
        relevant = threading.get_ident() == owner and caller.f_code is infer_code and caller.f_locals.get('self') is model
        del caller
        if relevant:
            if len(args) != 3 or kwargs:
                raise ValueError('Unexpected sampler where invocation')
            mask, selected, noise = args
            if mask.dtype != torch.bool or mask.shape != selected.shape or selected.shape != noise.shape:
                raise ValueError('Invalid sampling mask geometry')
            cap.update_mask_hashes.append(tensor_hash(mask))
            cap.model_rng_at_updates.append(tensor_hash(model.rng.get_state()))
            cap.selected_bit_hashes.append(tensor_hash(selected))
            cap.update_counts.append({'step': cap.steps_seen, 'n_total': mask.numel(), 'n_predict': int(mask.sum())})
            if cap.steps_seen in cap.snapshots:
                cap.snapshots[cap.steps_seen].update(selected_bits=selected.detach().flatten(2).transpose(1, 2).contiguous().cpu().byte(),
                    update_mask=mask.detach().flatten(2).transpose(1, 2).contiguous().cpu().clone())
        return original_where(*args, **kwargs)
    def capture_decode(vae, latent, *args, **kwargs):
        if cap.final_latent is not None:
            raise ValueError('Expected one final decoder call')
        cap.final_latent = latent.detach().cpu().clone()
        return decode(vae, latent, *args, **kwargs)
    with preserve_rng(model), replace_method(model, 'embeds_codes2input', capture_codes), \
         replace_method(model, '_calibrate_sr_hr_logits', capture_calibrate), \
         replace_method(model, 'summed_codes2images', capture_decode):
        torch.where = capture_where
        try:
            yield cap
            if cap.steps_seen != 50 or len(cap.update_counts) != 50 or set(cap.features) != set(STEPS) or cap.final_latent is None:
                raise ValueError('Incomplete 50-step teacher capture')
        finally:
            torch.where = original_where


def constructed_inputs(cache, seed):
    import torch
    result = {}
    for step in STEPS:
        generator = torch.Generator(device='cpu').manual_seed(seed + step)
        fresh_noise = torch.randint(0, 2, cache['target_bits'].shape, generator=generator, dtype=torch.uint8)
        bits = torch.where(cache['states'][step]['previous_mask'], cache['target_bits'], fresh_noise)
        result[step] = bits.transpose(1, 2).reshape(1, 256, 1, 32, 32)
    return result


def cache_storage_estimate(count):
    if type(count) is not int or count < 1:
        raise ValueError('Cache requires a positive image count')
    return {'image_count':count, 'estimated_bytes_per_image':64*1024**2,
            'estimated_payload_bytes':count*64*1024**2, 'reserve_bytes':1024**3,
            'scope':'Uncompressed fixed-teacher features plus bit states; estimate, not a byte-exact upper bound.'}


def cache(args):
    import numpy as np
    import torch
    from PIL import Image
    from tools.cache_sr_training_states import encode_target, cache_from_capture
    from grn.utils_t2iv.sr_state_training import _validate_pair_state, tensor_hash
    from grn.utils_t2iv.sr_trajectory_adapter import load_trajectory_adapter
    preparation, rows = load_pairs(args.pairs, 'train')
    output = fresh_output(args.output)
    storage = cache_storage_estimate(len(rows))
    storage['available_bytes'] = shutil.disk_usage(output).free
    status = {'schema': SCHEMA, 'status': 'running', 'stage': 'cache', 'completed': [], 'storage_estimate':storage}
    write_json(output / 'status.json', status)
    try:
        if storage['available_bytes'] < storage['estimated_payload_bytes'] + storage['reserve_bytes']:
            raise RuntimeError('Insufficient free disk for requested fixed-teacher feature cache plus 1 GiB reserve')
        identities = {'checkpoint_sha256': sha256(bundled(CONFIG['checkpoint'])), 'vae_sha256': sha256(bundled(CONFIG['hbq']))}
        teacher_adapter_sha = sha256(local_path(args.adapter)) if args.adapter else None
        pipe = load_pipeline(args.device)
        if args.adapter:
            load_trajectory_adapter(local_path(args.adapter), pipe.model, expected_base_sha256=identities['checkpoint_sha256'], trainable=False)
            if hasattr(pipe.model, 'sr_latent_residual'):
                raise ValueError('Cache teacher must be base/C-only: an R changes the decoder latent away from sampled hard bits')
        manifest = {'schema': SCHEMA, 'status': 'complete', 'pairs_sha256': sha256(local_path(args.pairs)),
            'pairs_path': str(local_path(args.pairs)), 'base_sha256': identities['checkpoint_sha256'],
            'vae_sha256': identities['vae_sha256'], 'teacher_adapter_sha256': teacher_adapter_sha,
            'teacher_adapter': str(local_path(args.adapter)) if args.adapter else None,
            'state_steps': list(STEPS), 'rollout_steps': 50, 'state_policy': 'fixed_teacher_features_not_student_on_policy',
            'degradation': preparation['degradation'], 'execution_identity': execution_identity(), 'storage_estimate':storage, 'records': []}
        device = next(pipe.model.parameters()).device
        for index, row in enumerate(rows):
            pair = dict(row, **identities)
            with torch.no_grad():
                target, grid = encode_target(pipe.vae, row['hr'], rounds=4, device=device)
                with capture_teacher(pipe) as captured:
                    rgb = pipe(lr_image=row['lr'], seed=row['seed'], **protocol())[0]
                state = cache_from_capture(captured, target, grid=grid, pair=pair, seed=row['seed'], validation_device=device)
                _validate_pair_state(state)
                from grn.utils_t2iv.hbq_util_t2iv import bit_label2raw_feature
                final_bits = captured.snapshots[50]['selected_bits'].transpose(1, 2).reshape(1, 256, 1, 32, 32)
                if not torch.equal(bit_label2raw_feature(final_bits, 4), captured.final_latent):
                    raise ValueError('Captured final latent differs from final hard bits')
                controls = constructed_inputs(state, row['seed'] + 1000000)
                with capture_teacher(pipe, constructed=controls) as constructed_cap:
                    pipe(lr_image=row['lr'], seed=row['seed'], **protocol())
                if captured.model_rng_at_updates != constructed_cap.model_rng_at_updates:
                    raise ValueError('Constructed-state control changed sampler RNG consumption')
                if index == 0 and args.verify_plain:
                    plain = pipe(lr_image=row['lr'], seed=row['seed'], **protocol())[0]
                    if not np.array_equal(rgb, plain):
                        raise ValueError('Capture instrumentation changed sampled pixels')
            folder = output / row['pair_id']; folder.mkdir()
            state_path, feature_path = folder / 'state.pt', folder / 'features.pt'
            torch.save(state, state_path)
            features = {'schema': SCHEMA, 'pair_id': row['pair_id'], 'base_sha256': identities['checkpoint_sha256'],
                'teacher_adapter_sha256': teacher_adapter_sha, 'target_bits': target,
                'final_latent': captured.final_latent, 'rollout': captured.features, 'constructed': constructed_cap.features}
            torch.save(features, feature_path)
            Image.fromarray(rgb).save(folder / 'teacher.png')
            record = dict(row, state_path=str(state_path.relative_to(output)), state_sha256=sha256(state_path),
                feature_path=str(feature_path.relative_to(output)), feature_sha256=sha256(feature_path),
                final_latent_sha256=tensor_hash(captured.final_latent), teacher_png_sha256=sha256(folder / 'teacher.png'))
            manifest['records'].append(record)
            status['completed'].append(row['pair_id'])
            write_json(output / 'status.json', status)
            print('CACHED', row['pair_id'], flush=True)
        write_json(output / 'cache.json', manifest)
        status.update(status='complete', cache_sha256=sha256(output / 'cache.json'))
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'status.json', status)
    return output / 'cache.json'


def require_vae_identity(expected, actual, *, context):
    def valid(value):
        return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
    if not valid(expected) or not valid(actual) or expected != actual:
        raise ValueError(f'{context}: HBQ/VAE checkpoint identity mismatch')


def training_source_provenance(current_rows, ancestor_reports):
    """Union all adapter training exposure, including teacher-feature lineage.

    The frozen original S5 base is outside this adapter-only isolation claim.
    An ancestor without complete source records permanently marks the union
    incomplete; known source records are still retained for overlap detection.
    """
    keys = ('source_id', 'source_sha256', 'decoded_rgb_sha256')
    combined = {}
    def add(rows):
        if not isinstance(rows, list):
            raise ValueError('Training source provenance must be a list')
        for row in rows:
            if not isinstance(row, dict) or any(key not in row for key in keys):
                raise ValueError('Training source provenance lacks an identity field')
            identity = {key: row[key] for key in keys}
            source = identity['source_id']
            if not isinstance(source, str) or not source or Path(source).is_absolute() or '..' in Path(source).parts:
                raise ValueError('Training source ID must be a normalized relative identity')
            for key in keys[1:]:
                value = identity[key]
                if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError('Training source digest is invalid')
            previous = combined.get(source)
            if previous is not None and previous != identity:
                raise ValueError('Training source lineage contains conflicting identities')
            combined[source] = identity
    add(current_rows)
    complete = True
    for report in ancestor_reports:
        metadata = report.get('metadata') if isinstance(report, dict) else None
        config = metadata.get('training_config') if isinstance(metadata, dict) else None
        if not isinstance(config, dict):
            complete = False
            continue
        rows = config.get('training_sources')
        if rows is None:
            complete = False
            continue
        add(rows)
        if config.get('training_source_provenance_complete') is not True or not rows:
            complete = False
    return {'training_sources': [combined[key] for key in sorted(combined)],
            'training_source_provenance_complete': complete}


def load_cache(path):
    import torch
    from grn.utils_t2iv.sr_state_training import load_pair_state
    path = local_path(path)
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != SCHEMA or manifest.get('status') != 'complete' or manifest.get('rollout_steps') != 50:
        raise ValueError('Expected complete teacher cache')
    if manifest.get('state_policy') != 'fixed_teacher_features_not_student_on_policy' or manifest.get('state_steps') != list(STEPS):
        raise ValueError('Unexpected teacher cache policy')
    if sha256(local_path(manifest['pairs_path'])) != manifest['pairs_sha256']:
        raise ValueError('Pair manifest changed after caching')
    _, original_rows = load_pairs(manifest['pairs_path'], 'train')
    by_id = {r['pair_id']: r for r in original_rows}
    if set(by_id) != {r['pair_id'] for r in manifest['records']}:
        raise ValueError('Cached image set differs from training pairs')
    for row in manifest['records']:
        original = by_id[row['pair_id']]
        for key in ('source_id', 'source_sha256', 'hr_sha256', 'lr_sha256', 'split'):
            if row[key] != original[key]:
                raise ValueError('Cache source identity mismatch')
        for name in ('state', 'feature'):
            file = local_path(path.parent / row[name + '_path'])
            if sha256(file) != row[name + '_sha256']:
                raise ValueError('Cache tensor bytes changed')
        state = load_pair_state(path.parent / row['state_path'])
        require_vae_identity(manifest.get('vae_sha256'), state['provenance'].get('vae_sha256'), context='State cache')
        feature = torch.load(path.parent / row['feature_path'], map_location='cpu', weights_only=True)
        if (feature['schema'] != SCHEMA or feature['teacher_adapter_sha256'] != manifest['teacher_adapter_sha256']
                or feature['base_sha256'] != manifest['base_sha256'] or feature['pair_id'] != row['pair_id']
                or state['provenance']['checkpoint_sha256'] != manifest['base_sha256']
                or any(state['provenance'][key] != row[key] for key in ('source_sha256','hr_sha256','lr_sha256'))):
            raise ValueError('Feature checkpoint/pair mismatch')
        if not torch.equal(state['target_bits'], feature['target_bits']):
            raise ValueError('Feature supervision target differs from validated cache')
    return path, manifest


def image_tensor(path, device):
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(path) as image:
        pixels = np.array(image.convert('RGB'), copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device).float() / 127.5 - 1.


def schedule_step(step, count, state_mix, kappa_min, kappa_max, seed):
    """A local per-step RNG makes data/κ choices exactly resumable."""
    generator = random.Random(seed + step * 104729)
    # Exact 50/50 alternation is the default, not approximate Bernoulli mixing.
    source = ('constructed' if step % 2 == 0 else 'rollout') if state_mix == .5 else (
        'rollout' if generator.random() < state_mix else 'constructed')
    return {'index': (step // 2) % count, 'source': source, 'state_step': STEPS[(step // (2 * count)) % len(STEPS)],
            'kappa': math.exp(generator.uniform(math.log(kappa_min), math.log(kappa_max)))}


def bit_cross_entropy(logits, target):
    """Canonical bit CE with a deterministic, separate mean reduction.

    MACA's spatial NLL internal mean uses a nondeterministic CUDA kernel.
    The existing GRN training path also requests per-bit CE first; reducing
    those losses with torch.mean preserves the objective and deterministic mode.
    """
    from torch.nn import functional as F
    return F.cross_entropy(logits, target, reduction='none').mean()


def train(args):
    import torch
    from torch.nn import functional as F
    from grn.utils_t2iv.sr_trajectory_adapter import (configure_trajectory_adapter, save_trajectory_adapter,
        load_trajectory_adapter, compute_trajectory_image_loss)
    from grn.utils_t2iv.sr_v2_image_loss import _logits_to_latent, build_offline_lpips
    cache_path, manifest = load_cache(args.cache)
    if not all(math.isfinite(value) for value in (args.lr,args.kappa_min,args.kappa_max,args.state_mix,args.image_weight,args.lpips_weight,args.gram_weight)):
        raise ValueError('Training scalars must be finite')
    if args.initial_adapter and args.resume:
        raise ValueError('Use initial-adapter or resume, not both')
    if args.save_every < 1:
        raise ValueError('Checkpoint interval must be positive')
    if min(args.updates, args.lr, args.kappa_min, args.kappa_max) <= 0 or args.kappa_min > args.kappa_max:
        raise ValueError('Invalid training configuration')
    if not 0 <= args.state_mix <= 1 or min(args.image_weight, args.lpips_weight, args.gram_weight) < 0:
        raise ValueError('Invalid mixture/loss weight')
    output = fresh_output(args.output)
    status = {'schema': SCHEMA, 'status': 'running', 'stage': 'train', 'updates_completed': 0}
    write_json(output / 'status.json', status)
    try:
        base_sha = sha256(bundled(CONFIG['checkpoint']))
        if base_sha != manifest['base_sha256']:
            raise ValueError('Cache belongs to a different base checkpoint')
        require_vae_identity(manifest.get('vae_sha256'), sha256(bundled(CONFIG['hbq'])), context='Current decoder')
        initial_sha = sha256(local_path(args.initial_adapter)) if args.initial_adapter else None
        if args.arm == 'R' and initial_sha is not None and initial_sha != manifest['teacher_adapter_sha256']:
            raise ValueError('R initial adapter differs from the cached teacher; regenerate cache with that C adapter')
        pipe = load_pipeline(args.device)
        ancestor_reports = []
        if manifest['teacher_adapter']:
            if sha256(local_path(manifest['teacher_adapter'])) != manifest['teacher_adapter_sha256']:
                raise ValueError('Teacher adapter changed')
            ancestor_reports.append(load_trajectory_adapter(local_path(manifest['teacher_adapter']), pipe.model, expected_base_sha256=base_sha, trainable=False))
        if args.initial_adapter:
            ancestor_reports.append(load_trajectory_adapter(local_path(args.initial_adapter), pipe.model, expected_base_sha256=base_sha, trainable=False))
        configure_trajectory_adapter(pipe.model, enable_lrc=args.arm in ('R', 'CR'), blocks=args.blocks, seed=args.seed, trainable=True)
        pipe.model.sr_round_calibrator.requires_grad_(args.arm in ('C', 'CR'))
        if hasattr(pipe.model, 'sr_latent_residual'):
            pipe.model.sr_latent_residual.requires_grad_(args.arm in ('R', 'CR'))
        parameters = [p for p in pipe.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.)
        trainable_names = [n for n,p in pipe.model.named_parameters() if p.requires_grad]
        if not parameters or any(not n.startswith(('sr_round_calibrator.', 'sr_latent_residual.')) for n in trainable_names):
            raise ValueError('Unexpected trainable parameter set')
        config = {'arm': args.arm, 'lr': args.lr, 'state_mix': args.state_mix, 'kappa_min': args.kappa_min,
            'kappa_max': args.kappa_max, 'image_weight': args.image_weight, 'lpips_weight': args.lpips_weight,
            'gram_weight': args.gram_weight, 'blocks': args.blocks, 'seed': args.seed,
            'cache_sha256': sha256(cache_path), 'state_policy': manifest['state_policy'], 'degradation': manifest['degradation'],
            'initial_adapter_sha256':initial_sha, 'teacher_adapter_sha256':manifest['teacher_adapter_sha256'],
            'r_input_policy':'final_frozen_teacher_latent', 'execution_identity':execution_identity(),
            'pairs_sha256': manifest['pairs_sha256'],
            **training_source_provenance(manifest['records'], ancestor_reports)}
        start_step = 0
        if args.resume:
            restored = load_trajectory_adapter(local_path(args.resume), pipe.model, expected_base_sha256=base_sha,
                                               optimizer=optimizer, trainable=True)
            saved_config = restored.get('metadata', {}).get('training_config')
            if not isinstance(saved_config, dict):
                raise ValueError('Resume checkpoint lacks training configuration')
            # Resume restores the trained state, not the original initializer.
            # Carry its immutable lineage forward without requiring --initial-adapter.
            inherited_initial = saved_config.get('initial_adapter_sha256')
            if inherited_initial is not None and (not isinstance(inherited_initial,str)
                    or len(inherited_initial)!=64 or any(c not in '0123456789abcdef' for c in inherited_initial)):
                raise ValueError('Invalid resumed initial adapter identity')
            config['initial_adapter_sha256'] = inherited_initial
            config.update(training_source_provenance(manifest['records'], ancestor_reports + [restored]))
            if args.arm == 'R' and inherited_initial is not None and inherited_initial != manifest['teacher_adapter_sha256']:
                raise ValueError('Resumed R initializer differs from cached teacher')
            if saved_config != config:
                raise ValueError('Resume training configuration/cache differs')
            start_step = int(restored['step'])
            # Loader restores trainability generically; keep this arm's exact optimizer contract.
            pipe.model.sr_round_calibrator.requires_grad_(args.arm in ('C', 'CR'))
            if hasattr(pipe.model, 'sr_latent_residual'):
                pipe.model.sr_latent_residual.requires_grad_(args.arm in ('R', 'CR'))
        device = next(pipe.model.sr_round_calibrator.parameters()).device
        perceptual = None
        if args.lpips_weight > 0 and (args.image_weight > 0 or args.arm in ('R','CR')):
            perceptual = build_offline_lpips(str(ROOT/'protocol/offline_cache/pyiqa/LPIPS_v0.1_alex-df73285e.pth'),
                str(ROOT/'protocol/offline_cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'), device=device)
        semantic = pipe.sr_semantic_encoder if args.gram_weight > 0 else None
        trainable_before = {n:p.detach().cpu().clone() for n,p in pipe.model.named_parameters() if p.requires_grad}
        write_json(output/'config.json', dict(config, updates_requested=args.updates, start_step=start_step,
            trainable_parameters=sum(p.numel() for p in parameters), trainable_names=trainable_names,
            scope='Bounded fixed-teacher-state adapter training; no claim of benchmark improvement.'))
        history = output/'history.jsonl'
        for step in range(start_step, start_step + args.updates):
            choice = schedule_step(step, len(manifest['records']), args.state_mix, args.kappa_min, args.kappa_max, args.seed)
            if args.arm == 'R':
                choice.update(index=step % len(manifest['records']), source='rollout', state_step=50)
            row = manifest['records'][choice['index']]
            features = torch.load(cache_path.parent/row['feature_path'], map_location='cpu', weights_only=True)
            state = {k:v.to(device) for k,v in features[choice['source']][choice['state_step']].items()}
            target = features['target_bits'].long().to(device)
            hr = image_tensor(row['hr'], device)
            optimizer.zero_grad(set_to_none=True)
            ce = torch.zeros((),device=device)
            image_loss = torch.zeros((),device=device)
            r_loss = torch.zeros((),device=device)
            if args.arm in ('C', 'CR'):
                logits = pipe.model.sr_round_calibrator.calibrate_logits(state['head_logits'], state['hidden'],
                    state['lr_features'], state['signal'], kappa=choice['kappa'])
                ce = bit_cross_entropy(logits, target)
                if args.image_weight > 0:
                    latent = _logits_to_latent(logits, 32, 32, 4, 64, 'hard_st')
                    image_loss = compute_trajectory_image_loss(latent,hr,pipe.vae,perceptual_model=perceptual,
                        semantic_encoder=semantic,lpips_weight=args.lpips_weight,gram_weight=args.gram_weight)['total']
                loss = ce + args.image_weight * image_loss
                # Keep C and final-latent R objectives separate; no 50-step graph is retained.
                loss.backward()
            if args.arm in ('R', 'CR'):
                final = features['final_latent'].to(device)
                lr_features = features['rollout'][50]['lr_features'].to(device)
                corrected = pipe.model.sr_latent_residual(final, lr_features, choice['kappa'])
                r_loss = compute_trajectory_image_loss(corrected,hr,pipe.vae,perceptual_model=perceptual,
                    semantic_encoder=semantic,lpips_weight=args.lpips_weight,gram_weight=args.gram_weight)['total']
                r_loss.backward()
            total = ce.detach()+args.image_weight*image_loss.detach()+r_loss.detach()
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite training loss')
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            if not any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in parameters):
                raise RuntimeError('No nonzero adapter gradient')
            if any(p.grad is not None for p in pipe.model.parameters() if not p.requires_grad):
                raise RuntimeError('Frozen base accumulated parameter gradients')
            optimizer.step()
            log = dict(step=step+1, pair_id=row['pair_id'], source_id=row['source_id'], **choice,
                bit_ce=float(ce.detach()), c_image=float(image_loss.detach()), r_image=float(r_loss.detach()),
                total=float(total), grad_norm=float(grad_norm), peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0)
            with history.open('a') as stream:stream.write(json.dumps(log,allow_nan=False)+'\n')
            status['updates_completed'] += 1
            write_json(output/'status.json',status)
            print('UPDATE',json.dumps(log),flush=True)
            if (step+1) % args.save_every == 0:
                save_trajectory_adapter(output/f'adapter-step-{step+1:08d}.pt',pipe.model,base_sha256=base_sha,optimizer=optimizer,
                    step=step+1,metadata={'training_config':config,'history_sha256':sha256(history)})
        changed = [n for n,p in pipe.model.named_parameters() if n in trainable_before and not torch.equal(p.detach().cpu(),trainable_before[n])]
        if not changed:
            raise RuntimeError('Optimizer did not update adapter parameters')
        save_trajectory_adapter(output/'adapter.pt',pipe.model,base_sha256=base_sha,optimizer=optimizer,
            step=start_step+args.updates,metadata={'training_config':config,'history_sha256':sha256(history)})
        status.update(status='complete', adapter_sha256=sha256(output/'adapter.pt'), changed_parameters=changed,
            frozen_base_parameter_gradients_absent=True)
    except BaseException as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output/'status.json',status)
    return output/'adapter.pt'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--train-count',type=int,default=2048);p.add_argument('--validation-count',type=int,default=64)
    p.add_argument('--crop-min',type=int,default=640);p.add_argument('--crop-max',type=int,default=896)
    c=sub.add_parser('cache');c.add_argument('--pairs',required=True,type=Path);c.add_argument('--adapter',type=Path)
    c.add_argument('--verify-plain',action='store_true');c.add_argument('--device',default='cuda:0')
    t=sub.add_parser('train');t.add_argument('--cache',required=True,type=Path);t.add_argument('--arm',choices=['C','R','CR'],default='C')
    t.add_argument('--updates',type=int,default=8);t.add_argument('--save-every',type=int,default=1000);t.add_argument('--lr',type=float,default=1e-5)
    t.add_argument('--state-mix',type=float,default=.5);t.add_argument('--kappa-min',type=float,default=.25);t.add_argument('--kappa-max',type=float,default=4.)
    t.add_argument('--image-weight',type=float,default=.1);t.add_argument('--lpips-weight',type=float,default=2.);t.add_argument('--gram-weight',type=float,default=0.)
    t.add_argument('--blocks',type=int,default=6);t.add_argument('--initial-adapter',type=Path);t.add_argument('--resume',type=Path);t.add_argument('--device',default='cuda:0')
    for child in (p,c,t):
        child.add_argument('--output',required=True,type=Path);child.add_argument('--seed',type=int,default=20261108);child.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    plan = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    if args.command == 'cache' and args.pairs.is_file():
        candidate = json.loads(args.pairs.read_text())
        count = len(candidate.get('splits',{}).get('train',[]))
        if count:
            plan['storage_estimate'] = cache_storage_estimate(count)
    print(json.dumps(plan,indent=2),flush=True)
    if not args.execute:return
    path={'prepare':prepare,'cache':cache,'train':train}[args.command](args)
    print('COMPLETE',path,flush=True)


if __name__=='__main__':
    main()
