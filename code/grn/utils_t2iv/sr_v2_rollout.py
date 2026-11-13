"""Opt-in, FSDP-forward-safe on-policy SR input generation.

This implements inputs to the existing fixed cosine / shift=1 / retention=.95,
temperature=1, guidance=1, argmax sampler.  It is not confidence masking and
does not substitute generated targets.  Only the final caller's forward has
gradients.  Legacy paired-view audits describe the *pre-replacement* input;
``append_actual_input_audit`` explicitly records the consumed input separately.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.distributed as distributed

from grn.utils_t2iv.hbq_util_t2iv import multiclass_labels2onehot_input


def tensor_sha256(value):
    value = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rollout_plan(args, microstep):
    """One deterministic decision shared by ranks; no RNG is consumed."""
    every = int(getattr(args, 'sr_v2_rollout_every', 0))
    if every == 0:
        return None
    if every < 1 or microstep < 0:
        raise ValueError('V2 rollout interval must be positive and microstep nonnegative')
    steps = int(getattr(args, 'sr_v2_rollout_steps', 20))
    positions = getattr(args, 'sr_v2_rollout_input_steps', '[5, 10, 15, 20]')
    positions = json.loads(positions) if isinstance(positions, str) else positions
    if steps < 2 or not isinstance(positions, (list, tuple)) or not positions:
        raise ValueError('V2 rollout requires at least two steps and nonempty input positions')
    if any(type(p) is not int or not 1 <= p <= steps for p in positions):
        raise ValueError('V2 input positions are one-based integers within rollout_steps')
    active = microstep % every == every - 1
    position = positions[(microstep // every) % len(positions)] if active else 0
    return {'active': active, 'input_step': position, 'total_steps': steps}


def assert_shared_plan(plan, microstep, device):
    """Fail collectively before FSDP forwards if ranks disagree on loop count."""
    if not distributed.is_available() or not distributed.is_initialized():
        return
    signature = torch.tensor([microstep, int(plan['active']), plan['input_step'],
                              plan['total_steps']], device=device, dtype=torch.int64)
    signatures = [torch.empty_like(signature) for _ in range(distributed.get_world_size())]
    distributed.all_gather(signatures, signature)
    if any(not torch.equal(signature, peer) for peer in signatures):
        raise RuntimeError('V2 rollout plans differ between distributed ranks')


def rollout_seed(base_seed, microstep, rank, occurrence_id):
    payload = f'grn-sr-v2:{base_seed}:{microstep}:{rank}:{occurrence_id}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2 ** 63 - 1)


@contextmanager
def temporary_eval(model):
    """Restore every module flag, including intentionally frozen/eval children."""
    flags = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in flags:
            module.training = training


def labels_to_input(labels):
    return multiclass_labels2onehot_input(labels, 2).flatten(2).transpose(1, 2)


def prepare_rollout_input(model, text_cond_tuple, x_BLC, forward_kwargs, *,
                          args, microstep, rank, meta_list):
    """Return inputs, scale metadata, and provenance (identity when disabled).

    The current pilot supports one packed image per rank, continuous B prefix,
    and no cached state replacement.  This narrow contract fails closed rather
    than silently changing sampling semantics for other training recipes.
    """
    info = forward_kwargs['other_info_by_scale']
    plan = rollout_plan(args, microstep)
    provenance = {'source': 'gt_corruption', 'seed': None, 'input_step': None,
                  'rollout_total_steps': None, 'no_grad_forwards': 0}
    if plan is None:
        return x_BLC, info, provenance
    assert_shared_plan(plan, microstep, x_BLC[0].device)
    if not plan['active']:
        return x_BLC, info, provenance
    if not getattr(args, 'sr_condition', 0) or getattr(args, 'sr_condition_mode', '') != 'prefix_text':
        raise ValueError('V2 online rollout currently requires B / prefix_text SR')
    if getattr(args, 'refine_mode', '') != 'ar_discrete_GRN_bit' or getattr(args, 'sr_state_manifest', ''):
        raise ValueError('V2 rollout requires bit GRN without cached-state replacement')
    scales = forward_kwargs['sequece_packing_scales']
    if len(x_BLC) != 1 or len(scales) != 1 or len(scales[0]) != 1 or len(info) != 1:
        raise ValueError('V2 rollout requires one image and one scale per rank')
    pt, ph, pw = scales[0][0]
    if pt != 1 or any(m.get('first_frame_condition', False) for m in meta_list):
        raise ValueError('V2 rollout supports natural images, not first-frame conditioning')
    gt = forward_kwargs['gt_BL'][0]
    if gt.shape != (1, ph * pw, args.hbq_round * args.detail_scale_dim):
        raise ValueError('V2 rollout target bit dimensions are inconsistent')
    occurrence = meta_list[0].get('sr_view', {}).get('occurrence_id', '')
    if getattr(args, 'sr_qualified_manifest', ''):
        from grn.utils_t2iv.sr_v2_qualified_training import check_materialized_meta
        qualified_row, _ = check_materialized_meta(args, meta_list[0])
        occurrence = f"qualified:{args.sr_qualified_manifest_sha256}:{qualified_row['occurrence_id']}"
    seed = rollout_seed(getattr(args, 'sr_v2_rollout_seed', 20260916), microstep, rank, occurrence)
    generator = torch.Generator(device=x_BLC[0].device).manual_seed(seed)
    sampling_dtype = torch.get_autocast_dtype(gt.device.type) if torch.is_autocast_enabled(gt.device.type) else torch.float32
    # Integer 0/1 values in floating point match autoregressive_infer's label
    # initialization.  Layout is [B, bit_channel, T, H, W], including mask draws.
    pure_noise = torch.randint(0, 2, (1, gt.shape[-1], pt, ph, pw),
                               device=gt.device, dtype=sampling_dtype, generator=generator)
    mixed = pure_noise
    actual_signal = 0.0
    current_info = [dict(info[0])]
    model_kwargs = dict(forward_kwargs, return_sr_aux=True, other_info_by_scale=current_info,
                        sr_v2_inference_precision=True)
    with temporary_eval(model), torch.no_grad():
        for step_index in range(plan['input_step'] - 1):
            current_info[0]['scale_token_id'] = actual_signal
            output = model(text_cond_tuple, [labels_to_input(mixed)], **model_kwargs)
            logits = output[-1]['logits_by_scale'][0]
            if logits.shape != (1, 2, ph * pw, gt.shape[-1]):
                raise ValueError('V2 rollout expects [1, 2, spatial_tokens, bit_channels] logits')
            require_finite_on_all_ranks(logits, 'V2 rollout logits')
            # Do not replace softmax->argmax with logits.argmax: bf16 rounding
            # can introduce ties after the actual inference softmax.
            predicted = logits.softmax(dim=1).argmax(dim=1)
            predicted = predicted.transpose(1, 2).reshape_as(pure_noise)
            scheduled = (1.0 - math.cos(math.pi / 2 * min(1., (step_index + 1) / (plan['total_steps'] - 1)))) * .95
            mask = torch.rand(predicted.shape, device=gt.device, generator=generator) < scheduled
            mixed = torch.where(mask, predicted, pure_noise)
            actual_signal = mask.float().mean().item()
    current_info[0]['scale_token_id'] = actual_signal
    current_info[0]['wandb_plot_index'] = min(9, int(actual_signal / .1))
    provenance.update(source='online_current_model', seed=seed,
                      input_step=plan['input_step'], rollout_total_steps=plan['total_steps'],
                      no_grad_forwards=plan['input_step'] - 1,
                      sampler='fixed_cosine_shift1_retention0.95_argmax_tau1_guidance1',
                      initial_noise_sha256=tensor_sha256(pure_noise),
                      sampler_precision=str(sampling_dtype))
    return [labels_to_input(mixed)], current_info, provenance


def append_actual_input_audit(args, *, x_BLC, gt_BLC, other_info_by_scale,
                             provenance, meta_list, microstep, rank, counters, lr_condition=None):
    """Audit the tensors immediately before the single gradient forward."""
    directory = getattr(args, 'sr_v2_audit_dir', '')
    if not directory:
        return
    rows = []
    for i, (inputs, gt, info) in enumerate(zip(x_BLC, gt_BLC, other_info_by_scale)):
        row = dict(schema_version=1, audit_stage='actual_gradient_forward_input',
                   rank=rank, microstep=microstep, sample_index=i,
                   occurrence_id=meta_list[i].get('sr_view', {}).get('occurrence_id'),
                   actual_signal=float(info['scale_token_id']),
                   input_sha256=tensor_sha256(inputs), gt_sha256=tensor_sha256(gt),
                   target_source='ground_truth_hr', **provenance, **counters)
        if lr_condition is not None:
            row['lr_tensor_sha256'] = tensor_sha256(lr_condition[i:i + 1])
        view = meta_list[i].get('sr_view', {})
        for key in ('source_sha256', 'hr_sha256', 'lr_sha256', 'view_id', 'corruption_seed'):
            if key in view:
                row['bound_' + key] = view[key]
        if getattr(args, 'sr_qualified_manifest', ''):
            from grn.utils_t2iv.sr_v2_qualified_training import bind_actual_input
            row['occurrence_id'] = meta_list[i]['sr_qualified']['occurrence_id']
            row.update(bind_actual_input(args, meta_list[i],
                None if lr_condition is None else lr_condition[i]))
        rows.append(json.dumps(row, sort_keys=True) + '\n')
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path / f'sr_v2_actual_input_rank{rank}.jsonl').open('a') as handle:
        handle.writelines(rows)


def require_finite_on_all_ranks(value, description):
    finite = torch.isfinite(value.detach()).all().to(dtype=torch.int32)
    if distributed.is_available() and distributed.is_initialized():
        distributed.all_reduce(finite, op=distributed.ReduceOp.MIN)
    if not finite.item():
        raise FloatingPointError(f'{description} is nonfinite on at least one rank')


def append_optimizer_audit(args, *, microstep, rank, counters, total_loss, grad_norm,
                           bit_ce, latent_losses, image_losses, model=None, interval_counts=None):
    directory = getattr(args, 'sr_v2_audit_dir', '')
    runtime_directory = getattr(args, 'sr_v2_runtime_audit_dir', '')
    if not directory and not runtime_directory:
        return
    losses = {'bit_ce': float(bit_ce.detach()), **{key: float(value.detach())
              for key, value in latent_losses.items()}}
    if image_losses is not None:
        losses.update({'image_' + key: float(value.detach()) if torch.is_tensor(value) else float(value)
                       for key, value in image_losses.items()})
    row = dict(schema_version=1, audit_stage='optimizer_step_completed', rank=rank,
               microstep=microstep, finite_loss=bool(torch.isfinite(total_loss.detach()).all()),
               finite_grad_norm=bool(torch.isfinite(grad_norm.detach()).all()),
               total_loss=float(total_loss.detach()), grad_norm=float(grad_norm.detach()),
               losses=losses, **counters)
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        with (path / f'sr_v2_optimizer_rank{rank}.jsonl').open('a') as handle:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    if runtime_directory:
        if model is None or interval_counts is None:
            raise ValueError('Runtime update audit requires model gradients and accumulation counts')
        count = sum(p.grad is not None and p.grad.numel() > 0 for p in model.parameters())
        if count < 1:
            raise RuntimeError('V2 optimizer update has no observed parameter gradients')
        # The pre-step clip norm covers every model gradient (and all FSDP
        # shards).  A finite norm certifies those participating gradients.
        runtime_row = dict(rank=rank, optimizer_update=counters['optimizer_updates_completed'],
            loss=row['total_loss'], grad_norm=row['grad_norm'],
            all_gradients_finite=row['finite_grad_norm'], gradient_tensors=count,
            microstep=microstep, **interval_counts)
        path = Path(runtime_directory)
        path.mkdir(parents=True, exist_ok=True)
        with (path / f'updates_rank{rank}.jsonl').open('a') as handle:
            handle.write(json.dumps(runtime_row, sort_keys=True) + '\n')
