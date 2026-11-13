"""Opt-in finite execution prefix without changing the full data/LR horizon.

This is a runtime stop boundary, not a subsampled dataset or a completed epoch.
It uses the normal Trainer and CKPTSaver; CPU contracts are not GPU evidence.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def validate(args):
    stop = getattr(args, 'sr_run_stop_microstep', 0)
    require(type(stop) is int and stop >= 0, 'Run stop must be a nonnegative integer')
    if not stop:
        return 0
    bundle_smoke = getattr(args, 's5_bundle_smoke_updates', 0)
    if bundle_smoke:
        from grn.utils_t2iv.s5_bundle_paths import bundle_root, resolve_path
        require(type(bundle_smoke) is int and bundle_smoke > 0,
                'S5 smoke updates must be a positive integer')
        require(stop == bundle_smoke * 16 and args.gradient_accumulation == 16 and args.ep == 2,
                'S5 smoke must preserve the original accumulation and two-epoch horizon')
        require(args.sr_qualified_manifest_sha256 ==
                '43c0d463cd1f976ce1b7b4946061908034b9ca310f223e20b62375d5a5dc4e60',
                'S5 smoke requires the unchanged original qualified manifest')
        require(resolve_path(args.rush_resume, strict=True) ==
                bundle_root() / 'weights/s5_initialization_step_16000.pth',
                'S5 smoke requires the packaged original S5 initialization')
        require(Path(args.local_out_path).resolve().is_relative_to(bundle_root() / 'verification'),
                'S5 smoke outputs must stay within bundle verification')
    require((getattr(args, 'sr_initialization_kind', '') == 'original_grn' or bundle_smoke)
            and bool(getattr(args, 'sr_qualified_manifest', '')),
            'Bounded run requires explicit original GRN or packaged S5 smoke with qualified full data')
    accumulation = getattr(args, 'gradient_accumulation', 0)
    require(type(accumulation) is int and accumulation > 0 and stop % accumulation == 0,
            'Run stop must be an optimizer accumulation boundary')
    require(getattr(args, 'checkpoint_type', '') == 'torch'
            and getattr(args, 'restrict_data_size', None) == -1
            and not getattr(args, 'sr_checkpoint_microsteps', ''),
            'Run stop requires normal torch saves and untruncated full data, not phase saves')
    frequency = getattr(args, 'save_model_iters_freq', 0)
    require(type(frequency) is int and frequency >= 0 and frequency % accumulation == 0,
            'Periodic saves must align with full optimizer windows')
    for key in ('local_out_path', 'bed', 'sr_v2_runtime_audit_dir'):
        require(bool(getattr(args, key, '')) and Path(getattr(args, key)).is_absolute(),
                'Explicit bounded-run output/audit directories required')
    require(Path(args.local_out_path).resolve() == Path(args.bed).resolve(),
            'Bounded run requires one local checkpoint destination')
    return stop


def validate_resume(args, saved_args):
    if not (getattr(args, 'sr_run_stop_microstep', 0) or saved_args.get('sr_run_stop_microstep', 0)):
        return
    # A different execution boundary is allowed; a different LR/data horizon is
    # a new training experiment, not exact continuation of this bounded run.
    for key in ('ep', 'gradient_accumulation', 'sche', 'tlr', 'twd', 'wp', 'wp0', 'wpe',
                'adam_beta2', 'fp16', 'tclip', 'sr_backbone_lr_scale',
                'sr_head_lr_scale', 'sr_new_lr_scale', 'sr_qualified_data_contract'):
        require(key in saved_args and getattr(args, key, None) == saved_args[key],
                'Bounded resume changes the full training recipe: ' + key)


def planned_stop(args, *, epoch, full_epoch_microsteps, start_it, start_global_it):
    stop = validate(args)
    if not stop:
        return 0
    contract = json.loads(args.sr_qualified_data_contract)
    require(all(type(value) is int for value in (epoch, full_epoch_microsteps, start_it, start_global_it))
            and 0 <= epoch < args.ep and full_epoch_microsteps > 0
            and 0 <= start_it < full_epoch_microsteps
            and start_it % args.gradient_accumulation == 0
            and start_global_it == epoch * full_epoch_microsteps + start_it,
            'Run start does not match the actual full-epoch offset')
    require(contract['padded_occurrences_per_epoch'] == full_epoch_microsteps * contract['world_size']
            and contract['accumulation'] == args.gradient_accumulation
            and full_epoch_microsteps % args.gradient_accumulation == 0,
            'Run budget cannot shorten the qualified epoch or alter accumulation')
    require(start_global_it < stop < args.ep * full_epoch_microsteps,
            'Bounded prefix must perform new work and end before the full training horizon')
    return stop


def record_stop(args, trainer, *, epoch, next_it, global_it, full_epoch_microsteps, rank):
    """Called after the real saver returns on every rank, never from a plan."""
    require(global_it == validate(args) and 0 < next_it <= full_epoch_microsteps
            and global_it == epoch * full_epoch_microsteps + next_it,
            'Actual run stop position differs from the requested boundary')
    state = trainer.sr_v2_state
    require(state['microsteps_completed'] == global_it
            and state['optimizer_updates_completed'] == global_it // args.gradient_accumulation,
            'Trainer counters differ from the completed bounded run')
    checkpoint = Path(args.local_out_path) / f'ar-ckpt-global_step_{global_it}.pth'
    require(checkpoint.is_file() and not checkpoint.is_symlink() and checkpoint.stat().st_size > 0,
            'Actual bounded checkpoint was not saved')
    next_epoch = epoch + int(next_it == full_epoch_microsteps)
    value = dict(schema_version=1, kind=('sr_v2_bounded_s5_bundle_smoke'
        if getattr(args, 's5_bundle_smoke_updates', 0) else 'sr_v2_bounded_original_training_run'),
        status='completed_bounded_prefix', pid=os.getpid(), rank=rank,
        finished_at=datetime.now(timezone.utc).isoformat(),
        original_phase=(None if getattr(args, 's5_bundle_smoke_updates', 0) else args.sr_original_phase),
        s5_bundle_smoke_updates=getattr(args, 's5_bundle_smoke_updates', 0), global_microstep=global_it,
        next_epoch=next_epoch, next_it=0 if next_epoch != epoch else next_it,
        full_epoch_microsteps=full_epoch_microsteps,
        full_training_horizon_microsteps=args.ep * full_epoch_microsteps,
        full_training_horizon_completed=False,
        dataset_truncated=False, learning_rate_horizon_shortened=False,
        actual_counters=dict(state), checkpoint=str(checkpoint),
        checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_restoration_verified=False, quality_evaluated=False)
    directory = Path(args.sr_v2_runtime_audit_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f'bounded_run_rank{rank}.json').open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    args._sr_run_stopped_at = global_it
    return value
