"""Opt-in S1 configuration and actual warm-start evidence contracts."""
from __future__ import annotations
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import torch

def restored_v2_evidence(trainer, saved_trainer_state):
    """Require actual restored V2 counters; leave legacy report schemas untouched."""
    if 'sr_v2' not in saved_trainer_state:
        return {}
    expected = saved_trainer_state['sr_v2']
    actual = getattr(trainer, 'sr_v2_state', None)
    if (not getattr(trainer, 'sr_v2_enabled', False)
            or not isinstance(expected, dict) or not isinstance(actual, dict)
            or actual != expected):
        raise RuntimeError('Actual restored V2 trainer counters differ from checkpoint')
    return dict(restored_v2_state=dict(actual), v2_state_exact=True)




IMAGE_WEIGHTS = ('sr_v2_rgb_weight', 'sr_v2_lowfreq_weight',
                 'sr_v2_edge_weight', 'sr_v2_perceptual_weight')


def validate_v2_config(args):
    weights = [getattr(args, key, 0.) for key in IMAGE_WEIGHTS]
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or v < 0 for v in weights):
        raise ValueError('V2 image weights must be finite nonnegative numbers')
    every = getattr(args, 'sr_v2_rollout_every', 0)
    if type(every) is not int or every < 0 or every == 1:
        raise ValueError('V2 rollout_every must be 0 (disabled) or >= 2')
    enabled = bool(every or any(weights) or getattr(args, 'sr_v2_audit_dir', '')
                   or getattr(args, 'sr_v2_runtime_audit_dir', ''))
    if not enabled:
        return False
    if not getattr(args, 'sr_condition', 0) or getattr(args, 'sr_condition_mode', '') != 'prefix_text':
        raise ValueError('V2 S1 requires the audited prefix_text SR architecture')
    if (getattr(args, 'sr_lr_size', 128), getattr(args, 'sr_hr_size', 512),
            getattr(args, 'hbq_round', 4), getattr(args, 'detail_scale_dim', 64)) != (128, 512, 4, 64):
        raise ValueError('V2 S1 requires 128/512 images and 64-channel M4 HBQ')
    if getattr(args, 'sr_state_manifest', '') or getattr(args, 'sr_state_arm', ''):
        raise ValueError('V2 S1 cannot combine legacy cached-state training')
    drop_rate = float(getattr(args, 'sr_condition_drop_rate', 0.))
    pilot = getattr(args, 'sr_v2_condition_dropout_pilot', 0)
    if type(pilot) is not int or pilot not in (0, 1):
        raise ValueError('condition-dropout pilot flag must be integer 0/1')
    if drop_rate != 0. and not pilot:
        raise ValueError('V2 S1 requires condition dropout=0 unless the explicit pilot flag is set')
    if pilot and not 0. < drop_rate < 1.:
        raise ValueError('condition-dropout pilot requires a rate strictly between 0 and 1')
    # Args defaults to zero; arg_util only initializes sequence parallelism
    # for values greater than one. Both zero and one are disabled sentinels.
    sp_size = getattr(args, 'sp_size', 0)
    if type(sp_size) is not int or sp_size not in (0, 1):
        raise ValueError('V2 S1 sequence parallelism has not been validated')
    if (int(getattr(args, 'use_ada_layer_norm', 0)) != 0
            or int(getattr(args, 'add_scale_token', 1)) != 1
            or bool(getattr(args, 'use_slow_attn', False))):
        raise ValueError('V2 S1 requires native time tokens, no AdaLN, and the fast attention path')
    warmup = getattr(args, 'sr_v2_image_warmup_updates', 100)
    if type(warmup) is not int or warmup < 0:
        raise ValueError('V2 image warmup must be nonnegative optimizer updates')
    if getattr(args, 'sr_v2_image_mode', 'hard_st') not in ('hard_st', 'soft'):
        raise ValueError('V2 image mode must be hard_st or soft')
    if every:
        steps = getattr(args, 'sr_v2_rollout_steps', 20)
        if type(steps) is not int or not 2 <= steps <= 50:
            raise ValueError('V2 rollout steps must be in [2, 50]')
        raw = getattr(args, 'sr_v2_rollout_input_steps', '[5, 10, 15, 20]')
        try:
            selected = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError('Invalid V2 rollout input steps') from error
        if (not isinstance(selected, (list, tuple)) or not selected
                or any(type(v) is not int or not 2 <= v <= steps for v in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError('V2 rollout input steps must be unique integers in [2, steps]')
        seed = getattr(args, 'sr_v2_rollout_seed', 20260916)
        if type(seed) is not int or seed < 0:
            raise ValueError('V2 rollout seed must be a nonnegative integer')
    return True


def verify_warm_start(model, source_state, source_path, *, rank):
    """Compare loaded tensors before FSDP, one tensor at a time, against mmap source."""
    actual = model.state_dict()
    if set(actual) != set(source_state):
        raise RuntimeError('V2 warm-start state keys differ from the trained B checkpoint')
    for name, value in actual.items():
        reference = source_state[name]
        if (not torch.is_tensor(reference) or value.shape != reference.shape
                or not torch.equal(value.detach().cpu(), reference.detach().cpu().to(value.dtype))):
            raise RuntimeError(f'V2 actual warm-start tensor mismatch: {name}')
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            raise RuntimeError(f'V2 nonfinite warm-start tensor: {name}')
    return dict(rank=int(rank), source_path=str(Path(source_path).resolve(strict=True)),
                source_tensors_exact=True, source_tensor_count=len(actual),
                backend_flags=dict(
                    torch_version=str(torch.__version__),
                    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    matmul_allow_bf16_reduced_precision_reduction=
                        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                    cudnn_benchmark=torch.backends.cudnn.benchmark,
                    cudnn_deterministic=torch.backends.cudnn.deterministic,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
                pid=os.getpid(), timestamp_utc=datetime.now(timezone.utc).isoformat())


def write_initialization_audit(directory, record, optimizer):
    """Require a fresh optimizer and refuse overwriting previous run evidence."""
    if not record or not record.get('source_tensors_exact') or not record.get('source_tensor_count'):
        raise RuntimeError('V2 initialization audit requires actual source comparison')
    if len(optimizer.state):
        raise RuntimeError('V2 warm-start optimizer unexpectedly contains inherited states')
    value = {**record, 'optimizer_states_at_start': 0, 'optimizer_updates_at_start': 0}
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"init_rank{record['rank']}.json").open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    return value
