"""Explicit qualified-original data mode, independent of immutable pilot views.

This implements data admission/consumption, not original-GRN initialization or
an S5 training approval. Full-model and distributed gates remain necessary.
"""
from pathlib import Path
from grn.utils_t2iv.s5_bundle_paths import resolve_path
import platform

import numpy as np
import PIL
import torch

from grn.utils_t2iv import sr_v2_qualified_data as data


def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    paths = ['utils_t2iv/sr_v2_qualified_data.py', 'utils_t2iv/sr_v2_qualified_training.py',
             'utils_t2iv/sr_v2_rollout.py', 'dataset/dataset_joint_vi.py', 'schedules/global_refine.py']
    hashes = {path: data.sha((root / path).read_bytes()) for path in paths}
    return data.sha(data.json_bytes(hashes))


def enabled(args):
    return bool(getattr(args, 'sr_qualified_manifest', ''))


def reference_recipe(args):
    root = resolve_path(args.sr_protocol_root, strict=True)
    config = Path(args.sr_degradation_config)
    config = (root / config).resolve(strict=True)
    data.require(config.is_relative_to(root), 'Degradation config escapes protocol bundle')
    source = root / 'dataloader/realesrgan.py'
    config_raw = data.checked_bytes(config, args.sr_qualified_degradation_config_sha256, 2**20)
    data.checked_bytes(source, args.sr_qualified_reference_code_sha256, 2**20)
    # Bind transitive local implementation, not only the entry-point file.
    dependencies = sorted(set((root / 'basicsr').rglob('*.py')) |
                          set((root / 'dataloader').rglob('*.py')))
    data.require(dependencies and (root / 'basicsr/__init__.py') in dependencies,
                 'Reference BasicSR source tree missing')
    inventory = {}
    for path in dependencies:
        data.require(path.resolve(strict=True).is_relative_to(root), 'Reference dependency escapes bundle')
        inventory[str(path.relative_to(root))] = data.sha(path.read_bytes())
    import yaml
    options = yaml.safe_load(config_raw)
    data.require(isinstance(options, dict) and options.get('scale') == 4
                 and options.get('gray_prob', 0) == 0 and options.get('color_jitter_prob', 0) == 0,
                 'Qualified reference recipe requires RGB x4 with no output color conversion')
    return dict(config_path=str(config), config_sha256=data.sha(config_raw),
                reference_path=str(source), reference_sha256=args.sr_qualified_reference_code_sha256,
                local_python_inventory_sha256=data.sha(data.json_bytes(inventory)),
                local_python_files=len(inventory))


def validate_qualified_training(args, world_size):
    """Fail before model allocation; cache the expensive original-isolation audit."""
    if not enabled(args):
        return None
    data.require(not any(getattr(args, key, '') for key in
        ('sr_view_manifest', 'sr_state_manifest', 'sr_state_arm', 'sr_train_manifest')),
        'Qualified originals are exclusive with static views, cached states and legacy manifests')
    data.require(getattr(args, 'sr_condition', 0) == 1
        and getattr(args, 'sr_condition_mode', '') == 'prefix_text'
        and (getattr(args, 'sr_hr_size', 0), getattr(args, 'sr_lr_size', 0)) == (512, 128)
        and getattr(args, 'sr_degradation', '') == 'realesrgan'
        and not getattr(args, 'pair_input', False)
        and getattr(args, 'loop_data_per_epoch', 1) == 1,
        'Qualified training requires single-image prefix_text RGB 512/128 SR')
    data.require(getattr(args, 'restrict_data_size', -1) == -1,
                 'Qualified training must not truncate full original coverage')
    data.require(getattr(args, 'video_scale_probs', [1.0]) == [1.0],
                 'Qualified training requires preserving the sole HR scale')
    data.require(not getattr(args, 'use_vae_token_cache', 0)
                 and not getattr(args, 'save_vae_token_cache', 0)
                 and getattr(args, 'allow_online_vae_feature_extraction', 1),
                 'Dynamic HR views cannot use a fixed VAE cache')
    data.require(getattr(args, 'checkpoint_type', 'torch') == 'torch',
                 'Qualified data resume currently supports torch checkpoints only')
    if getattr(args, 'auto_resume', False):
        data.require(getattr(args, 'sr_v2_resume_audit', 0) == 1,
                     'Qualified auto-resume requires pinned exact audited resume, never fallback')
    world = data.int64(world_size, 1)
    accumulation = data.int64(getattr(args, 'gradient_accumulation', 1), 1)
    seed = data.int64(args.sr_qualified_seed)
    low, high = (data.int64(args.sr_qualified_crop_min, 512), data.int64(args.sr_qualified_crop_max, 512))
    data.require(low <= high, 'Qualified crop range reversed')
    data.require(Path(args.sr_qualified_audit_dir).is_absolute(), 'Explicit absolute data audit directory required')
    data.require(all(getattr(args, key, '') and Path(getattr(args, key)).is_absolute()
        for key in ('sr_v2_audit_dir', 'sr_v2_runtime_audit_dir')),
        'Qualified training requires explicit actual-input and update audit directories')
    path = resolve_path(args.sr_qualified_manifest, strict=True)
    digest = data.digest(args.sr_qualified_manifest_sha256)
    # Every startup/epoch still rechecks the small final commit marker. Source
    # originals are hashed+decoded when consumed; we never trust cached pixels.
    data.checked_bytes(path, digest, 2**21)
    cached = getattr(args, '_sr_qualified_sources', None)
    if cached is None:
        cached = data.load_qualified_manifest(path, digest)
        args._sr_qualified_sources = cached
    data.require(cached.manifest_path == str(path) and cached.manifest_sha256 == digest,
                 'Qualified source changed after admission')
    data.require(resolve_path(cached.protocol_root, strict=True) == resolve_path(args.sr_protocol_root, strict=True),
                 'Qualified and degradation protocol roots differ')
    data.require(type(args.sr_expected_train_samples) is int and args.sr_expected_train_samples == len(cached.rows),
                 'Expected train samples must equal actual qualified originals, not raw candidates')
    recipe = reference_recipe(args)
    padded = ((len(cached.rows) + world * accumulation - 1) // (world * accumulation)) * world * accumulation
    data.int64(padded, 1)
    import cv2
    import scipy
    import torchvision
    contract = dict(schema='grn_sr_qualified_data_training_v1', manifest_path=str(path),
        manifest_sha256=digest, originals=len(cached.rows), world_size=world, accumulation=accumulation,
        epoch_seed=seed, short_side_range=[low, high], padded_occurrences_per_epoch=padded,
        optimizer_updates_per_epoch=padded // (world * accumulation),
        crop_algorithm='random_virtual_short_side_lanczos_crop_v1',
        degradation_cpu_threads=1,
        ordering='manifest_seed_epoch_global_shuffle_pad_full_optimizer_windows_v1',
        normalized_tensor_format='HR_1x3x512x512_LR_3x128x128_float32_minus1_plus1',
        reference=recipe, implementation_sha256=implementation_identity(),
        versions=dict(python=platform.python_version(), pillow=PIL.__version__,
                      numpy=np.__version__, torch=str(torch.__version__),
                      torchvision=str(torchvision.__version__), scipy=scipy.__version__, opencv=cv2.__version__))
    encoded = data.json_bytes(contract).decode()
    previous = getattr(args, 'sr_qualified_data_contract', '')
    data.require(not previous or previous == encoded, 'Qualified runtime data contract changed')
    args.sr_qualified_data_contract = encoded
    return cached


def validate_qualified_resume(args, saved_args, *, start_ep, start_it):
    current, old = enabled(args), bool(saved_args.get('sr_qualified_manifest', ''))
    if not current and not old:
        return
    data.require(current and old, 'Resume cannot switch qualified/legacy data modes')
    data.require(bool(getattr(args, 'sr_qualified_data_contract', ''))
                 and saved_args.get('sr_qualified_data_contract') == args.sr_qualified_data_contract,
                 'Resume qualified data contract differs or is missing')
    contract = data.parse(args.sr_qualified_data_contract)
    data.int64(start_ep)
    data.int64(start_it)
    per_rank = contract['padded_occurrences_per_epoch'] // contract['world_size']
    data.require(start_it < per_rank and start_it % contract['accumulation'] == 0,
                 'Qualified resume must start at an optimizer boundary within the epoch')


def loader_resume_view(args, dataset, start_it):
    """Skip already-consumed dynamic originals by index, without re-degrading them.

    Legacy datasets retain their historical iterator-skip path. The training
    loop keeps the full epoch length and absolute local microstep numbering.
    """
    if not enabled(args):
        return dataset, start_it
    data.int64(start_it)
    data.require(start_it < len(dataset) and start_it % args.gradient_accumulation == 0,
                 'Invalid qualified loader resume boundary')
    if start_it == 0:
        return dataset, 0
    return torch.utils.data.Subset(dataset, range(start_it, len(dataset))), 0


def rank_records(args, sources, *, epoch, rank, world_size):
    data.int64(rank)
    data.require(rank < world_size, 'Invalid qualified data rank')
    occurrences = data.epoch_occurrences(sources, epoch=epoch, seed=args.sr_qualified_seed,
        world_size=world_size, accumulation=args.gradient_accumulation)
    return [dict(sr_qualified=row, image_path=row['source_path']) for row in occurrences[rank::world_size]]


def check_materialized_meta(args, meta):
    data.require(enabled(args) and 'sr_view' not in meta and 'sr_qualified' in meta
                 and 'sr_qualified_materialized' in meta, 'Qualified materialized metadata required')
    row, record = meta['sr_qualified'], meta['sr_qualified_materialized']
    data.validate_occurrence(args._sr_qualified_sources, row)
    data.require(all(record.get(key) == value for key, value in row.items()),
                 'Materialized identity differs from requested occurrence')
    data.require(record['qualification_manifest_sha256'] == args.sr_qualified_manifest_sha256,
                 'Materialized source qualification differs')
    data.require(row['rank'] == getattr(args, 'rank', 0), 'Qualified sample reached a different rank')
    return row, record


def append_consumption_audit(args, meta, tensors, *, scheduled_signal, actual_signal):
    """Main-process consumption, not worker-prefetch: binds original-U draws."""
    if not enabled(args):
        return
    _, record = check_materialized_meta(args, meta)
    from tools.sr_refinement_diagnostics import tensor_hash
    value = dict(schema='grn_sr_qualified_consumption_v1', **record,
        audit_stage='gt_corruption_before_optional_rollout',
        data_contract_sha256=data.sha(args.sr_qualified_data_contract.encode()),
        scheduled_signal=float(scheduled_signal), actual_signal=float(actual_signal))
    for key in ('target_labels', 'random_labels', 'mask', 'corrupted_input'):
        value[key + '_sha256'] = tensor_hash(tensors[key])
    directory = Path(args.sr_qualified_audit_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f'sr_qualified_consumption_rank{value["rank"]}.jsonl').open('a') as handle:
        handle.write(data.json_bytes(value).decode() + '\n')


def bind_actual_input(args, meta, lr_condition):
    """Actual GRN-input audit supplement; never labels dynamic pixels as PNGs."""
    _, record = check_materialized_meta(args, meta)
    if lr_condition is not None:
        data.require(data.tensor_sha(lr_condition) == record['lr_tensor_sha256'],
                     'Actual model LR tensor differs from dynamically materialized condition')
    return dict(data_mode='qualified_dynamic_originals',
                data_contract_sha256=data.sha(args.sr_qualified_data_contract.encode()),
                qualified_materialization=record)
