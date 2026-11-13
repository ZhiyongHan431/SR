import random
import time
import gc
from functools import partial
from pprint import pformat
from typing import List, Optional, Tuple, Union
import os
import os.path as osp
import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import torch.distributed as tdist
from torch.amp import autocast
import cv2

import grn.utils_t2iv.dist as dist
from grn.models.ema import update_ema
from grn.utils_t2iv import arg_util, misc
from grn.utils import wandb_utils
from grn.schedules import get_encode_decode_func
from grn.schedules.dynamic_resolution import get_dynamic_resolution_meta
from grn.utils.compress_tokens import save_packed_tensor
from grn.models.grn import GRN

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
fulloptstate_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)


def _charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + eps ** 2).mean()


def _latent_highpass(latent: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Return the latent detail residual after removing a local low-pass mean."""
    if kernel_size < 3 or kernel_size % 2 != 1:
        raise ValueError(
            f"sr_latent_highpass_kernel must be an odd integer >= 3, got {kernel_size}"
        )
    if kernel_size > min(latent.shape[-2:]):
        raise ValueError(
            f"High-pass kernel {kernel_size} exceeds latent grid {tuple(latent.shape[-2:])}"
        )
    radius = kernel_size // 2
    lowpass = F.avg_pool2d(
        F.pad(latent, (radius, radius, radius, radius), mode='reflect'),
        kernel_size=kernel_size,
        stride=1,
    )
    return latent - lowpass


def _hbq_bits_to_latent(
    bit_values: torch.Tensor,
    hbq_round: int,
    latent_dim: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Differentiably convert hard/soft HBQ bits to the decoder latent grid."""
    if bit_values.ndim != 3:
        raise ValueError(f"HBQ bits must be [B, N, R*C], got {tuple(bit_values.shape)}")
    batch_size, token_count, bit_channels = bit_values.shape
    if token_count != height * width or bit_channels != hbq_round * latent_dim:
        raise ValueError(
            f"HBQ bit shape {tuple(bit_values.shape)} is incompatible with "
            f"rounds={hbq_round}, channels={latent_dim}, grid={height}x{width}"
        )
    bits = bit_values.reshape(batch_size, token_count, hbq_round, latent_dim)
    intervals = torch.pow(
        torch.tensor(0.5, device=bits.device, dtype=bits.dtype),
        torch.arange(1, hbq_round + 1, device=bits.device, dtype=bits.dtype),
    )
    latent = ((bits * 2.0 - 1.0) * intervals[None, None, :, None]).sum(dim=2)
    return latent.transpose(1, 2).reshape(batch_size, latent_dim, height, width)


def _compute_sr_latent_losses(
    sr_aux: dict,
    gt_bits_by_scale: List[torch.Tensor],
    sequence_packing_scales: List[List[Tuple[int, int, int]]],
    args: arg_util.Args,
) -> dict:
    """Compute SR supervision entirely in the quantized HBQ latent domain."""
    native = getattr(args, 'sr_condition_mode', 'prefix') == 'native_adapter'
    if native and float(getattr(args, 'sr_condition_align_weight', 0)) != 0:
        raise ValueError('native SR requires sr_condition_align_weight=0')
    logits_by_scale = sr_aux['logits_by_scale']
    if len(logits_by_scale) != len(gt_bits_by_scale):
        raise ValueError(
            f"Predicted/target scale count mismatch: {len(logits_by_scale)} vs "
            f"{len(gt_bits_by_scale)}"
        )

    pred_latents, target_latents = [], []
    scale_index = 0
    for sample_scales in sequence_packing_scales:
        if len(sample_scales) != 1:
            raise ValueError(
                "The fixed 128-to-512 SR objective expects exactly one 32x32 scale per sample"
            )
        pt, ph, pw = sample_scales[0]
        if pt != 1:
            raise ValueError(f"Natural-image SR expects pt=1, got {pt}")
        logits = logits_by_scale[scale_index]
        if logits.shape[1] != 2:
            raise ValueError(f"Bit-level SR needs two logits per bit, got {logits.shape[1]}")
        soft_bits = logits.float().softmax(dim=1)[:, 1]
        target_bits = gt_bits_by_scale[scale_index].float()
        pred_latents.append(
            _hbq_bits_to_latent(
                soft_bits, args.hbq_round, args.detail_scale_dim, ph, pw
            )
        )
        target_latents.append(
            _hbq_bits_to_latent(
                target_bits, args.hbq_round, args.detail_scale_dim, ph, pw
            )
        )
        scale_index += 1

    pred_latent = torch.cat(pred_latents, dim=0)
    target_latent = torch.cat(target_latents, dim=0).detach()
    condition_latent = sr_aux['condition_latent'].float()
    if condition_latent.shape != target_latent.shape:
        raise ValueError(
            f"Condition/HR latent shape mismatch: {tuple(condition_latent.shape)} vs "
            f"{tuple(target_latent.shape)}"
        )

    latent_recon = _charbonnier_loss(pred_latent, target_latent)
    structure = sum(
        weight * _charbonnier_loss(
            F.avg_pool2d(pred_latent, kernel_size=pool),
            F.avg_pool2d(target_latent, kernel_size=pool),
        )
        for pool, weight in ((2, 0.4), (4, 0.6))
    )
    edge = 0.5 * (
        _charbonnier_loss(
            pred_latent[:, :, :, 1:] - pred_latent[:, :, :, :-1],
            target_latent[:, :, :, 1:] - target_latent[:, :, :, :-1],
        )
        + _charbonnier_loss(
            pred_latent[:, :, 1:, :] - pred_latent[:, :, :-1, :],
            target_latent[:, :, 1:, :] - target_latent[:, :, :-1, :],
        )
    )

    alignment_terms = []
    for pool in (() if native else (2, 4)):
        condition_low = F.avg_pool2d(condition_latent, kernel_size=pool)
        target_low = F.avg_pool2d(target_latent, kernel_size=pool)
        cosine = 1.0 - F.cosine_similarity(
            condition_low, target_low, dim=1, eps=1e-6
        ).mean()
        alignment_terms.append(
            _charbonnier_loss(condition_low, target_low) + 0.1 * cosine
        )
    condition_alignment = torch.stack(alignment_terms).mean() if alignment_terms else pred_latent.new_zeros(())
    highpass_weight = float(getattr(args, 'sr_latent_highpass_weight', 0.0))
    if highpass_weight < 0:
        raise ValueError(
            f"sr_latent_highpass_weight must be non-negative, got {highpass_weight}"
        )
    latent_highpass = pred_latent.new_zeros(())
    if highpass_weight > 0.0:
        highpass_kernel = int(getattr(args, 'sr_latent_highpass_kernel', 3))
        latent_highpass = _charbonnier_loss(
            _latent_highpass(pred_latent, highpass_kernel),
            _latent_highpass(target_latent, highpass_kernel),
        )
    return {
        'latent_recon': latent_recon,
        'latent_structure': structure,
        'latent_edge': edge,
        'condition_alignment': condition_alignment,
        'latent_highpass': latent_highpass,
    }

def _collect_sr_cross_attn_gate_metrics(
    model: nn.Module,
    layers: List[int],
) -> dict:
    """Collect sharding-safe scalar gate values and gradients from an FSDP model."""
    device = next(model.parameters()).device
    local_stats = {
        str(layer): torch.zeros(4, device=device, dtype=torch.float32)
        for layer in layers
    }
    for name, parameter in model.named_parameters():
        parts = name.split('.')
        if 'sr_cross_attn' not in parts or not name.endswith('.gate'):
            continue
        marker = parts.index('sr_cross_attn')
        if marker + 1 >= len(parts) or parts[marker + 1] not in local_stats:
            continue
        if parameter.numel() == 0:
            continue
        stats = local_stats[parts[marker + 1]]
        stats[0] += parameter.detach().float().sum()
        stats[1] += parameter.numel()
        if parameter.grad is not None and parameter.grad.numel() > 0:
            stats[2] += parameter.grad.detach().float().abs().sum()
            stats[3] += parameter.grad.numel()

    metrics = {}
    for layer, stats in local_stats.items():
        if tdist.is_initialized():
            tdist.all_reduce(stats, op=tdist.ReduceOp.SUM)
        if stats[1].item() > 0:
            metrics[f'SR/cross_attn_gate/layer_{layer}'] = (stats[0] / stats[1]).item()
        if stats[3].item() > 0:
            metrics[f'SR/cross_attn_gate_grad_abs/layer_{layer}'] = (
                stats[2] / stats[3]
            ).item()
    return metrics

def sum_dict(acc_pt2scale_acc):
    for full_pt in acc_pt2scale_acc:
        for si in range(len(acc_pt2scale_acc[full_pt])):
            acc_pt2scale_acc[full_pt][si] = torch.tensor(acc_pt2scale_acc[full_pt][si]).sum()
    return acc_pt2scale_acc

def dict2list(acc_pt2scale_acc):
    flatten_acc_pt2scale_acc = []
    for key, val in acc_pt2scale_acc.items():
        flatten_acc_pt2scale_acc.extend(val)
    return flatten_acc_pt2scale_acc

def list2dict(acc_pt2scale_acc, flatten_acc_pt2scale_acc):
    ptr = 0
    for key in acc_pt2scale_acc:
        for ind in range(len(acc_pt2scale_acc[key])):
            acc_pt2scale_acc[key][ind] = flatten_acc_pt2scale_acc[ptr]
            ptr += 1
    return acc_pt2scale_acc

import queue
import threading

def save_token():
    while True:
        try:
            raw_features, feature_cache_files4images = save_token_queue.get()
            for i in range(len(feature_cache_files4images)):
                if not osp.exists(feature_cache_files4images[i]):
                    os.makedirs(osp.dirname(feature_cache_files4images[i]), exist_ok=True)
                    # torch.save(raw_features[i], feature_cache_files4images[i])
                    save_packed_tensor(feature_cache_files4images[i], raw_features[i])
                    print(f'Save to {feature_cache_files4images[i]}')
                else:
                    print(f'{feature_cache_files4images[i]} exists, skip')
        except Exception as e:
            print(f"Error saving token: {e}")
        finally:
            save_token_queue.task_done()

save_token_queue = queue.Queue()
saver = threading.Thread(target=save_token, daemon=True)
saver.start()

class Trainer(object):
    def __init__(
        self, is_visualizer: bool, device, 
        vae_local, gpt_wo_ddp: GRN, gpt: DDP,  gpt_opt: torch.optim.Optimizer, 
        dbg_unused=False,zero=0, vae_latent_dim=True, reweight_loss_by_scale=0,
        gpt_wo_ddp_ema=None, gpt_ema=None, use_fsdp_model_ema=False, other_args=None,
    ):
        super(Trainer, self).__init__()
        self.zero = zero
        self.device = device
        self.vae_latent_dim = vae_latent_dim
        self.gpt: Union[DDP, FSDP, nn.Module]
        self.gpt, self.vae_local = gpt, vae_local
        from grn.utils_t2iv.sr_v2_original_runtime import trainer_origin
        self.sr_original_init = trainer_origin(other_args)
        from grn.utils_t2iv.sr_v2_original_joint import trainer_lineage
        self.sr_original_joint = trainer_lineage(other_args)
        self.dynamic_scale_schedule = other_args.dynamic_scale_schedule
        self.dynamic_resolution_h_w, self.h_div_w_templates = get_dynamic_resolution_meta(other_args.dynamic_scale_schedule, other_args.train_h_div_w_list, other_args.video_frames)
        self.gpt_opt = gpt_opt
        self.gpt_wo_ddp = gpt_wo_ddp
        self.gpt_wo_ddp_ema = gpt_wo_ddp_ema
        self.gpt_ema = gpt_ema
        self.use_fsdp_model_ema = use_fsdp_model_ema
        self.batch_size, self.seq_len = 0, 0
        self.reweight_loss_by_scale = reweight_loss_by_scale
        video_encode, _, _, _ = get_encode_decode_func(other_args.dynamic_scale_schedule)
        self.video_encode = video_encode
        self.is_visualizer = is_visualizer
        self.set_seed(other_args.seed)
        from grn.utils_t2iv.sr_v2_condition_runtime import condition_config, build_semantic_encoder
        self.sr_v2_condition_config = condition_config(other_args)
        from grn.utils_t2iv.sr_v2_calibrator_runtime import calibrator_config
        s3_config = calibrator_config(other_args)
        # Full-schema checks belong before FSDP; rank-local shards are not full tensors.
        self.sr_v2_calibrator_config = s3_config
        self.sr_v2_semantic_encoder = build_semantic_encoder(other_args, device)
        if self.sr_v2_condition_config.enabled:
            self.sr_v2_condition_state = dict(microsteps_completed=0, semantic_batches=0, coarse_batches=0)
        self.sr_v2_enabled = bool(
            getattr(other_args, 'sr_v2_audit_dir', '')
            or getattr(other_args, 'sr_v2_runtime_audit_dir', '')
            or getattr(other_args, 'sr_v2_rollout_every', 0)
            or any(float(getattr(other_args, 'sr_v2_' + name + '_weight', 0))
                   for name in ('rgb', 'lowfreq', 'edge', 'perceptual'))
            or self.sr_v2_condition_config.enabled or s3_config.enabled
        )
        if self.sr_v2_enabled:
            self.sr_v2_state = dict(optimizer_updates_completed=0, microsteps_completed=0,
                                   rollout_microsteps_completed=0, image_microsteps_completed=0,
                                   image_gradient_chain_microsteps=0)
            self.sr_v2_perceptual_model = None
            self.sr_v2_accum_rollout = 0
            self.sr_v2_accum_image = 0
        self.sr_v2_condition_observer = None
        if getattr(other_args, 'sr_v2_condition_audit', 0):
            from grn.utils_t2iv.sr_v2_condition_audit import ConditionRuntimeAudit
            self.sr_v2_condition_observer = ConditionRuntimeAudit(other_args,
                self.sr_v2_condition_config, gpt_wo_ddp, self.sr_v2_semantic_encoder, rank=tdist.get_rank())

    def set_seed(self, seed: int) -> None:
        rank_seed = int(seed) + tdist.get_rank()
        numpy_generator = np.random.default_rng(rank_seed)
        torch_cuda_generator = torch.Generator(device=self.device)
        torch_cuda_generator.manual_seed(rank_seed)
        self.rank_vary_generator = {
            'numpy_generator': numpy_generator,
            'torch_cuda_generator': torch_cuda_generator,
        }
        gpt_uncompiled = (
            self.gpt_wo_ddp._orig_mod
            if hasattr(self.gpt_wo_ddp, '_orig_mod')
            else self.gpt_wo_ddp
        )
        gpt_uncompiled.rng = torch.Generator(device=self.device)
        gpt_uncompiled.rng.manual_seed(rank_seed)

    def train_step(
        self, ep: int, it: int, g_it: int, stepping: bool, clip_decay_ratio: float, metric_lg: misc.MetricLogger, logging_params: bool,
        raw_features_bcthw: FTen, feature_cache_files4images: list, media: str, meta_list: list,
        inp_B3HW: FTen, lr_conditions: Optional[List[FTen]],
        text_cond_tuple: Union[ITen, FTen], args: arg_util.Args,
        accumulation_divisor: int = 1,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        device = args.device
        B = len(inp_B3HW) + len(raw_features_bcthw)
        lr_condition = None
        native_lr_input, native_lr_latent = None, None
        if getattr(args, 'sr_condition', 0):
            if len(raw_features_bcthw):
                raise ValueError(
                    "SR training needs HR pixels and paired LR conditions; disable VAE token cache"
                )
            if lr_conditions is None or len(lr_conditions) != len(inp_B3HW):
                raise ValueError(
                    f"SR expects one LR condition per HR image, got "
                    f"{0 if lr_conditions is None else len(lr_conditions)} and {len(inp_B3HW)}"
                )
            expected_hr = int(getattr(args, 'sr_hr_size', 512))
            expected_lr = int(getattr(args, 'sr_lr_size', 128))
            for hr, lr in zip(inp_B3HW, lr_conditions):
                if hr.ndim != 4 or hr.shape[:2] != (3, 1):
                    raise ValueError(
                        f"SR expects HR tensors shaped [3, 1, H, W], got {tuple(hr.shape)}"
                    )
                if tuple(hr.shape[-2:]) != (expected_hr, expected_hr):
                    raise ValueError(
                        f"SR expects {expected_hr} x {expected_hr} HR crops, got {tuple(hr.shape[-2:])}"
                    )
                if lr.shape != (3, expected_lr, expected_lr):
                    raise ValueError(
                        f"SR expects LR tensors shaped [3, {expected_lr}, {expected_lr}], "
                        f"got {tuple(lr.shape)}"
                    )
            lr_condition = torch.stack(lr_conditions, dim=0)

        if media == 'images':
            is_image_batch = 1
        else:
            is_image_batch = 0
        # [forward]
        with torch.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=args.zero == 0):
            with torch.amp.autocast('cuda', dtype=torch.float32):
                if lr_condition is not None and getattr(args, 'sr_condition_mode', 'prefix') == 'native_adapter':
                    from grn.utils_t2iv.sr_native import encode_native_lr
                    native_lr_input, native_lr_latent = encode_native_lr(
                        self.vae_local, lr_condition, args.sr_hr_size, args.hbq_round,
                    )
                raw_features_list = []
                if len(inp_B3HW):
                    with torch.no_grad():
                        for inp_ind, inp in enumerate(inp_B3HW):
                            raw_features_, _, _ = self.vae_local.encode_for_raw_features(inp.unsqueeze(0), scale_schedule=None, slice=args.use_slice)
                            raw_features = raw_features_[0]
                            save_tokens = args.use_vae_token_cache and args.save_vae_token_cache and (not osp.exists(feature_cache_files4images[inp_ind]))
                            if save_tokens:
                                from grn.utils_t2iv.hbq_util_t2iv import raw_feature2bit_label
                                saved_visual_features = raw_feature2bit_label(raw_features, args.hbq_round).to(torch.bool)
                                saved_visual_features = saved_visual_features.cpu().data
                                save_token_queue.put((saved_visual_features, [feature_cache_files4images[inp_ind]]))
                            raw_features_list.append([raw_features])
                                
                if len(raw_features_bcthw):
                    raw_features_bcthw = [[item.unsqueeze(0)] for item in raw_features_bcthw]
                    raw_features_list = raw_features_list + raw_features_bcthw

            assert isinstance(raw_features_list[0], list)
            full_pts_this_batch = [item[0].shape[-3] for item in raw_features_list]
            kv_compact, lens, cu_seqlens_k, max_seqlen_k, caption_nums = text_cond_tuple
            
            with torch.no_grad():
                x_BLC, x_BLC_mask, scale_or_time_ids, gt_BLC, _, visual_rope_cache, sequece_packing_scales, super_scale_lengths, other_info_by_scale = self.video_encode(
                    vae=self.vae_local,
                    inp_B3HW=None,
                    vae_features=raw_features_list,
                    args=args,
                    device=device,
                    rope2d_freqs_grid=self.gpt.rope2d_freqs_grid,
                    dynamic_resolution_h_w=self.dynamic_resolution_h_w,
                    text_lens=lens,
                    caption_nums=caption_nums,
                    tokens_remain=args.train_max_token_len,
                    rank_vary_generator=self.rank_vary_generator,
                    vis_verbose=False, # g_it%20==0,
                    meta_list=meta_list,
                )
            
            # x_BLC_wo_prefix: torch.Size([bs, 2*2+3*3+...+64*64, d or 4d])

            # import pdb; pdb.set_trace()
            # from torchinfo import summary
            # res = summary(self.gpt, input_data=(text_cond_tuple, x_BLC_wo_prefix, scale_schedule))

            semantic_features = None
            observer = getattr(self, 'sr_v2_condition_observer', None)
            if observer is not None:
                observer.begin(lr_condition, g_it - 1)
            if getattr(args, 'sr_v2_semantic', 0):
                encoder = getattr(self, 'sr_v2_semantic_encoder', None)
                if encoder is None or lr_condition is None:
                    raise RuntimeError('S2 semantic training requires the frozen encoder and LR pixels')
                semantic_features = encoder(lr_condition)
            forward_kwargs = dict(
                lr_condition=lr_condition,
                native_lr_input=native_lr_input,
                native_lr_latent=native_lr_latent,
                x_BLC_mask=x_BLC_mask,
                gt_BL=gt_BLC,
                is_image_batch=is_image_batch,
                visual_rope_cache=visual_rope_cache,
                sequece_packing_scales=sequece_packing_scales,
                super_scale_lengths=super_scale_lengths,
                other_info_by_scale=other_info_by_scale,
                scale_or_time_ids=scale_or_time_ids,
                return_sr_aux=bool(getattr(args, 'sr_condition', 0)),
            )
            if semantic_features is not None:
                forward_kwargs['sr_semantic_features'] = semantic_features
            v2_provenance = None
            if getattr(self, 'sr_v2_enabled', False):
                from grn.utils_t2iv.sr_v2_rollout import prepare_rollout_input, append_actual_input_audit
                # t2iv_train increments g_it before calling train_step.  V2
                # schedules and occurrence audits use zero-based microsteps.
                v2_microstep = g_it - 1
                if v2_microstep < 0:
                    raise ValueError('Trainer g_it is one-based; V2 microstep must be nonnegative')
                x_BLC, other_info_by_scale, v2_provenance = prepare_rollout_input(
                    self.gpt, text_cond_tuple, x_BLC, forward_kwargs,
                    args=args, microstep=v2_microstep, rank=dist.get_rank(), meta_list=meta_list,
                )
                forward_kwargs['other_info_by_scale'] = other_info_by_scale
                append_actual_input_audit(
                    args, x_BLC=x_BLC, gt_BLC=gt_BLC, other_info_by_scale=other_info_by_scale,
                    provenance=v2_provenance, meta_list=meta_list, microstep=v2_microstep,
                    rank=dist.get_rank(), counters=self.sr_v2_state, lr_condition=lr_condition,
                )
            model_outputs = self.gpt(text_cond_tuple, x_BLC, **forward_kwargs)
            if getattr(args, 'sr_condition', 0):
                logits_norm, loss, acc_bit, valid_sequence_ratio, sr_aux = model_outputs
            else:
                logits_norm, loss, acc_bit, valid_sequence_ratio = model_outputs
                sr_aux = None

            # [loss reweight]
            # import pdb; pdb.set_trace()
            example_global_scales = 10
            acc_pt2scale_acc = {}
            acc_pt2scale_acc_counter = {}
            for full_pt in self.dynamic_resolution_h_w[self.h_div_w_templates[0]]['0.06M']['pt2scale_schedule']:
                full_pt = int(np.round((full_pt-1) / 4)) * 4 + 1
                if full_pt not in acc_pt2scale_acc:
                    acc_pt2scale_acc[full_pt] = [[] for _ in range(example_global_scales)]
                    acc_pt2scale_acc_counter[full_pt] = [0 for _ in range(example_global_scales)]
            
            flatten_L_list, flatten_acc_bit_list, flatten_weight_list = [], [], []
            ptr = 0
            global_scale_ind = 0
            for sample_ind, item in enumerate(sequece_packing_scales):
                full_pt = full_pts_this_batch[sample_ind]
                full_pt = int(np.round((full_pt-1) / 4)) * 4 + 1
                for si, (pt, ph, pw) in enumerate(item):
                    mul_pt_ph_pw = pt * ph * pw
                    start, end = ptr, ptr+mul_pt_ph_pw
                    ptr = end
                    loss_this_scale = loss[start:end].mean()
                    acc_this_scale = acc_bit[start:end].mean()
                    wandb_plot_index = other_info_by_scale[global_scale_ind]['wandb_plot_index']
                    acc_pt2scale_acc[full_pt][wandb_plot_index].append(acc_this_scale)
                    acc_pt2scale_acc_counter[full_pt][wandb_plot_index] += 1
                    flatten_weight_list.append(mul_pt_ph_pw ** args.reweight_loss_by_scale)
                    flatten_L_list.append(loss_this_scale)
                    flatten_acc_bit_list.append(acc_this_scale)
                    global_scale_ind += 1
            flatten_weight_list = torch.tensor(flatten_weight_list, dtype=loss.dtype, device=loss.device)
            flatten_weight_list = flatten_weight_list / flatten_weight_list.sum()
            bit_ce_loss = (torch.stack(flatten_L_list) * flatten_weight_list).sum()
            final_acc_bit = (torch.stack(flatten_acc_bit_list) * flatten_weight_list).sum()

            zero_loss = bit_ce_loss.new_zeros(())
            sr_losses = {
                'latent_recon': zero_loss,
                'latent_structure': zero_loss,
                'latent_edge': zero_loss,
                'condition_alignment': zero_loss,
                'latent_highpass': zero_loss,
            }
            if sr_aux is not None:
                sr_losses = _compute_sr_latent_losses(
                    sr_aux, gt_BLC, sequece_packing_scales, args
                )
            final_loss = (
                bit_ce_loss
                + float(getattr(args, 'sr_latent_recon_weight', 0.0)) * sr_losses['latent_recon']
                + float(getattr(args, 'sr_latent_structure_weight', 0.0)) * sr_losses['latent_structure']
                + float(getattr(args, 'sr_latent_edge_weight', 0.0)) * sr_losses['latent_edge']
                + float(getattr(args, 'sr_condition_align_weight', 0.0)) * sr_losses['condition_alignment']
                + float(getattr(args, 'sr_latent_highpass_weight', 0.0)) * sr_losses['latent_highpass']
            )
            coarse_weight = float(getattr(args, 'sr_v2_condition_coarse_weight', 0))
            if coarse_weight > 0:
                from grn.utils_t2iv.sr_v2_condition_runtime import condition_coarse_loss
                if sr_aux is None or 'condition_coarse_rgb' not in sr_aux:
                    raise RuntimeError('S2 coarse objective requires the actual conditional prediction')
                coarse_loss = condition_coarse_loss(sr_aux['condition_coarse_rgb'], inp_B3HW)
                if not coarse_loss.requires_grad or coarse_loss.grad_fn is None:
                    raise RuntimeError('S2 coarse objective lost its condition-encoder gradient chain')
                sr_losses['condition_coarse_rgb'] = coarse_loss
                final_loss = final_loss + coarse_weight * coarse_loss
            image_losses = None
            if getattr(self, 'sr_v2_enabled', False):
                image_weights = {name: float(getattr(args, 'sr_v2_' + name + '_weight', 0))
                                 for name in ('rgb', 'lowfreq', 'edge', 'perceptual')}
                if any(weight != 0 for weight in image_weights.values()):
                    if sr_aux is None:
                        raise ValueError('V2 image supervision requires SR auxiliary logits')
                    from grn.utils_t2iv.sr_v2_image_loss import compute_sr_v2_image_losses, build_offline_lpips
                    warmup_updates = int(getattr(args, 'sr_v2_image_warmup_updates', 100))
                    if warmup_updates < 0:
                        raise ValueError('V2 image warmup updates must be nonnegative')
                    warmup_factor = (min(1., (self.sr_v2_state['optimizer_updates_completed'] + 1) / warmup_updates)
                                     if warmup_updates else 1.)
                    if image_weights['perceptual'] > 0 and self.sr_v2_perceptual_model is None:
                        self.sr_v2_perceptual_model = build_offline_lpips(
                            getattr(args, 'sr_v2_perceptual_path', ''),
                            getattr(args, 'sr_v2_perceptual_backbone_path', ''), device=device,
                        )
                    image_losses = compute_sr_v2_image_losses(
                        sr_aux, sequece_packing_scales, inp_B3HW, self.vae_local,
                        hbq_round=args.hbq_round, latent_dim=args.detail_scale_dim,
                        weights=image_weights, mode=getattr(args, 'sr_v2_image_mode', 'hard_st'),
                        warmup_factor=warmup_factor, perceptual_model=self.sr_v2_perceptual_model,
                    )
                    if not image_losses['total'].requires_grad or image_losses['total'].grad_fn is None:
                        raise RuntimeError('V2 image loss must retain its gradient chain to GRN logits')
                    final_loss = final_loss + image_losses['total']
        
        # [backward]
        if accumulation_divisor < 1:
            raise ValueError(f'accumulation_divisor must be positive, got {accumulation_divisor}')
        if getattr(self, 'sr_v2_enabled', False):
            from grn.utils_t2iv.sr_v2_rollout import require_finite_on_all_ranks
            require_finite_on_all_ranks(final_loss, 'V2 total loss')
        (final_loss / accumulation_divisor).backward(
            retain_graph=False, create_graph=False
        )
        if observer is not None:
            observer.finish()
        if hasattr(self, 'sr_v2_condition_state'):
            self.sr_v2_condition_state['microsteps_completed'] += 1
            self.sr_v2_condition_state['semantic_batches'] += int(semantic_features is not None)
            self.sr_v2_condition_state['coarse_batches'] += int(coarse_weight > 0)
        if getattr(self, 'sr_v2_enabled', False):
            self.sr_v2_state['microsteps_completed'] += 1
            self.sr_v2_state['rollout_microsteps_completed'] += int(v2_provenance['source'] == 'online_current_model')
            self.sr_v2_state['image_microsteps_completed'] += int(image_losses is not None)
            self.sr_v2_state['image_gradient_chain_microsteps'] += int(image_losses is not None)
            self.sr_v2_accum_rollout += int(v2_provenance['source'] == 'online_current_model')
            self.sr_v2_accum_image += int(image_losses is not None)
        should_log = metric_lg.log_every_iter or it == 0 or it in metric_lg.log_iters
        gate_metrics = {}
        if (
            should_log
            and bool(getattr(args, 'sr_log_cross_attn_gates', 0))
            and bool(getattr(args, 'sr_cross_attn', 0))
        ):
            cross_layers = getattr(args, 'sr_cross_attn_layers', '[4, 12, 20, 28]')
            if isinstance(cross_layers, str):
                cross_layers = json.loads(cross_layers)
            gate_metrics = _collect_sr_cross_attn_gate_metrics(
                self.gpt, [int(layer) for layer in cross_layers]
            )
        grad_norm_t = torch.zeros((), device=final_loss.device)
        if stepping:
            if self.zero:
                grad_norm_t = self.gpt.clip_grad_norm_(args.tclip)
            else:
                grad_norm_t = torch.nn.utils.clip_grad_norm_(
                    self.gpt.parameters(), args.tclip
                )
            if getattr(self, 'sr_v2_enabled', False):
                require_finite_on_all_ranks(grad_norm_t, 'V2 gradient norm')
            self.gpt_opt.step()
            if getattr(self, 'sr_v2_enabled', False):
                self.sr_v2_state['optimizer_updates_completed'] += 1
                from grn.utils_t2iv.sr_v2_rollout import append_optimizer_audit
                append_optimizer_audit(
                    args, microstep=v2_microstep, rank=dist.get_rank(), counters=self.sr_v2_state,
                    total_loss=final_loss, grad_norm=grad_norm_t, bit_ce=bit_ce_loss,
                    latent_losses=sr_losses, image_losses=image_losses,
                    model=self.gpt, interval_counts=dict(rollout_microsteps=self.sr_v2_accum_rollout,
                        image_microsteps=self.sr_v2_accum_image),
                )
                self.sr_v2_accum_rollout = 0
                self.sr_v2_accum_image = 0

            if args.use_fsdp_model_ema and (args.model_ema_decay < 1):
                update_ema(self.gpt_ema, self.gpt, args.model_ema_decay)

            self.gpt_opt.zero_grad(set_to_none=True)
        
        # [metric logging]
        if should_log:
            acc_pt2scale_acc = sum_dict(acc_pt2scale_acc)
            flatten_acc_pt2scale_acc = dict2list(acc_pt2scale_acc)
            flatten_acc_pt2scale_acc_counter = dict2list(acc_pt2scale_acc_counter)

            train_loss = final_loss.item()
            train_acc = final_acc_bit.item()
            component_values = [
                bit_ce_loss.item(),
                sr_losses['latent_recon'].item(),
                sr_losses['latent_structure'].item(),
                sr_losses['latent_edge'].item(),
                sr_losses['condition_alignment'].item(),
                sr_losses['latent_highpass'].item(),
            ]
            metrics = torch.tensor(flatten_acc_pt2scale_acc + flatten_acc_pt2scale_acc_counter + [grad_norm_t.item(), train_loss, train_acc, is_image_batch, valid_sequence_ratio] + component_values, device=loss.device)
            tdist.all_reduce(metrics, op=tdist.ReduceOp.SUM)
            flatten_acc_pt2scale_acc, flatten_acc_pt2scale_acc_counter = metrics[:len(flatten_acc_pt2scale_acc)], metrics[len(flatten_acc_pt2scale_acc):2*len(flatten_acc_pt2scale_acc)]
            flatten_acc_pt2scale_acc = flatten_acc_pt2scale_acc / (flatten_acc_pt2scale_acc_counter + 1e-16)
            acc_pt2scale_acc = list2dict(acc_pt2scale_acc, flatten_acc_pt2scale_acc)
            acc_pt2scale_acc_counter = list2dict(acc_pt2scale_acc_counter, flatten_acc_pt2scale_acc_counter)
            metric_tail = metrics[2*len(flatten_acc_pt2scale_acc):] / (dist.get_world_size() + 1e-16)
            (
                grad_norm_t, train_loss, train_acc, is_image_batch,
                valid_sequence_ratio, train_bit_ce, train_latent_recon,
                train_latent_structure, train_latent_edge, train_condition_alignment,
                train_latent_highpass,
            ) = metric_tail
            if args.num_of_label_value == 1:
                key, base = 'Loss', 1
            else:
                key, base = 'Acc', 100
            metric_lg.update(L=train_loss, Acc=train_acc*base, L_i=0., Acc_i=0., L_v=0., Acc_v=0., tnm=grad_norm_t, seq_usage=valid_sequence_ratio*100.)    # todo: Accm, Acct
            wandb_log_dict = {
                'Overall/train_loss': train_loss,
                'Overall/bit_ce_loss': train_bit_ce,
                'SR/latent_recon_loss': train_latent_recon,
                'SR/latent_structure_loss': train_latent_structure,
                'SR/latent_edge_loss': train_latent_edge,
                'SR/condition_alignment_loss': train_condition_alignment,
                'Overall/train_acc': train_acc*base,
                'Overall/grad_norm_t': grad_norm_t,
                'Overall/logits_abs_mean': logits_norm.item(),
                'Overall/video_batch_ratio': (1-is_image_batch)*100., 
                'Overall/valid_sequence_ratio': valid_sequence_ratio*100.,
            }
            if float(getattr(args, 'sr_latent_highpass_weight', 0.0)) > 0:
                wandb_log_dict['SR/latent_highpass_loss'] = train_latent_highpass
            if bool(getattr(args, 'sr_log_loss_contributions', 0)):
                wandb_log_dict.update({
                    'SR/weighted_latent_recon_loss': train_latent_recon * float(getattr(args, 'sr_latent_recon_weight', 0.0)),
                    'SR/weighted_latent_structure_loss': train_latent_structure * float(getattr(args, 'sr_latent_structure_weight', 0.0)),
                    'SR/weighted_latent_edge_loss': train_latent_edge * float(getattr(args, 'sr_latent_edge_weight', 0.0)),
                    'SR/weighted_condition_alignment_loss': train_condition_alignment * float(getattr(args, 'sr_condition_align_weight', 0.0)),
                    'SR/weighted_latent_highpass_loss': train_latent_highpass * float(getattr(args, 'sr_latent_highpass_weight', 0.0)),
                })

            wandb_log_dict.update(gate_metrics)
            if coarse_weight > 0:
                coarse_metric = sr_losses['condition_coarse_rgb'].detach().clone()
                tdist.all_reduce(coarse_metric, op=tdist.ReduceOp.SUM)
                coarse_metric /= dist.get_world_size()
                wandb_log_dict['SR_V2/condition_coarse_rgb'] = coarse_metric
                wandb_log_dict['SR_V2/weighted_condition_coarse_rgb'] = coarse_metric * coarse_weight
            if hasattr(self, 'sr_v2_condition_state'):
                wandb_log_dict.update({f'SR_V2/condition_{key}': value for key, value in self.sr_v2_condition_state.items()})
            if getattr(self, 'sr_v2_enabled', False):
                v2_values = torch.tensor([
                    float(v2_provenance['source'] == 'online_current_model'),
                    float(other_info_by_scale[0]['scale_token_id']),
                    *[float(image_losses[name].detach()) if image_losses is not None else 0.
                      for name in ('rgb', 'lowfreq', 'edge', 'perceptual', 'total')],
                ], device=loss.device)
                tdist.all_reduce(v2_values, op=tdist.ReduceOp.SUM)
                v2_values /= dist.get_world_size()
                wandb_log_dict.update(zip(
                    ('SR_V2/online_rollout', 'SR_V2/input_signal', 'SR_V2/rgb_loss',
                     'SR_V2/lowfreq_loss', 'SR_V2/edge_loss', 'SR_V2/perceptual_loss',
                     'SR_V2/weighted_image_total'), v2_values.unbind(),
                ))
                wandb_log_dict.update({f'SR_V2/{key}': value for key, value in self.sr_v2_state.items()})
            for full_pt in acc_pt2scale_acc:
                for si in range(len(acc_pt2scale_acc[full_pt])):
                    if acc_pt2scale_acc_counter[full_pt][si] > 0:
                        duration = (full_pt-1) / 4
                        prefix = f't{duration:04.1f}s/signal_{si*0.1:.01f}_{(si+1)*0.1:.01f}'
                        wandb_log_dict[f'Details/{key}/{prefix}'] = acc_pt2scale_acc[full_pt][si].item() * base
                        wandb_log_dict[f'Details/Num/{prefix}'] = acc_pt2scale_acc_counter[full_pt][si]
            wandb_utils.log(wandb_log_dict, step=g_it)
        
    def __repr__(self):
        return (
            f'\n'
            f'[VGPTTr.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[VGPTTr.structure]: {super(Trainer, self).__repr__().replace(Trainer.__name__, "")}'
        )
      
    def get_config(self):
        return {}
    
    def state_dict(self):
        m = self.vae_local
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        state = {'config': self.get_config()}
        s3_config = getattr(self, 'sr_v2_calibrator_config', None)
        if s3_config is not None and s3_config.enabled:
            state['sr_v2_calibrator'] = s3_config.metadata()
        if getattr(self, 'sr_original_init', None) is not None:
            from copy import deepcopy
            state['sr_original_init'] = deepcopy(self.sr_original_init)
        if getattr(self, 'sr_original_joint', None) is not None:
            from copy import deepcopy
            state['sr_original_joint'] = deepcopy(self.sr_original_joint)
        observer = getattr(self, 'sr_v2_condition_observer', None)
        if observer is not None:
            # The checkpoint path can call state_dict() twice at an epoch
            # boundary (the periodic save and the epoch-complete save share
            # the same global step).  Keep the audit file exclusive, but make
            # the trainer-side call idempotent for that same boundary.
            boundary = self.sr_v2_state['microsteps_completed']
            last_boundary = getattr(self, '_sr_v2_condition_snapshot_boundary', None)
            if last_boundary != boundary:
                observer.snapshot(boundary)
                self._sr_v2_condition_snapshot_boundary = boundary
        if getattr(self, 'sr_v2_enabled', False):
            state['sr_v2'] = dict(self.sr_v2_state)
        if hasattr(self, 'sr_v2_condition_state'):
            state['sr_v2_condition'] = dict(self.sr_v2_condition_state)
            encoder = getattr(self, 'sr_v2_semantic_encoder', None)
            state['sr_v2_semantic_provenance'] = dict(encoder.provenance) if encoder is not None else None
        
        if self.zero:
            state['gpt_fsdp'] = None
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                state['gpt_fsdp'] = self.gpt.state_dict()
                if self.use_fsdp_model_ema:
                    state['gpt_ema_fsdp'] = self.gpt_ema.state_dict()
                state['gpt_fsdp_opt'] = FSDP.optim_state_dict(
                    self.gpt, self.gpt_opt
                )
        else:
            if self.using_ema:
                self.ema_load()
                state['gpt_ema_for_vis'] = {k: v.cpu() for k, v in self.gpt_wo_ddp.state_dict().items()}
                self.ema_recover()
            
            for k in ('gpt_wo_ddp', 'gpt_opt'):
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        from grn.utils_t2iv.sr_v2_calibrator_runtime import calibrator_config, check_trainer_calibrator
        s3_config = getattr(self, 'sr_v2_calibrator_config', None)
        check_trainer_calibrator(s3_config if s3_config is not None else calibrator_config({}), state)
        from grn.utils_t2iv.sr_v2_original_runtime import check_trainer_origin
        check_trainer_origin(getattr(self, 'sr_original_init', None), state)
        from grn.utils_t2iv.sr_v2_original_joint import check_trainer
        check_trainer(getattr(self, 'sr_original_joint', None), state)
        if getattr(self, 'sr_v2_enabled', False):
            if 'sr_v2' not in state:
                raise ValueError('V2 resume requires saved sr_v2 counters; use weights-only warm-start for B')
            saved_v2 = state['sr_v2']
            if set(saved_v2) != set(self.sr_v2_state) or any(type(v) is not int or v < 0 for v in saved_v2.values()):
                raise ValueError('Invalid V2 training counters in checkpoint')
        if hasattr(self, 'sr_v2_condition_state'):
            from grn.utils_t2iv.sr_v2_condition_runtime import validate_condition_counters
            validated = validate_condition_counters(state.get('sr_v2_condition'),
                self.sr_v2_condition_config, saved_v2['microsteps_completed'])
            encoder = getattr(self, 'sr_v2_semantic_encoder', None)
            provenance = dict(encoder.provenance) if encoder is not None else None
            if 'sr_v2_semantic_provenance' not in state or state['sr_v2_semantic_provenance'] != provenance:
                raise ValueError('S2 resume semantic asset provenance changed or is missing')
            self.sr_v2_condition_state.update(validated)
        if getattr(self, 'sr_v2_enabled', False):
            self.sr_v2_state.update(saved_v2)
        if self.zero:
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                self.gpt.load_state_dict(state['gpt_fsdp'])
                if self.use_fsdp_model_ema:
                    self.gpt_ema.load_state_dict(state['gpt_ema_fsdp'])
                one_group_opt_state = state.get('gpt_fsdp_opt')
                if one_group_opt_state:
                    optim_state_dict = FSDP.optim_state_dict_to_load(
                        model=self.gpt,
                        optim=self.gpt_opt,
                        optim_state_dict=one_group_opt_state,
                    )
                    self.gpt_opt.load_state_dict(optim_state_dict)
        else:
            raise NotImplementedError
