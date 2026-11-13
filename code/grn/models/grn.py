import json
import math
import time
from copy import deepcopy
from contextlib import nullcontext
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import tqdm
from timm.models import register_model

import grn.utils_t2iv.dist as dist
from grn.models.basic import FastRMSNorm, SelfAttnBlock
from grn.models.sr_attention_config import resolve_cross_attention_windows
from grn.models.sr_native_adapter import NativeLRSpatialAdapter
from grn.models.sr_condition_modes import condition_layout, uses_continuous_lr, uses_lr_prefix, uses_empty_text
from grn.models.sr_spatial_control import HRQuerySpatialControl, apply_packed_spatial_control
from grn.models.sr_v2_calibrator import resolve_calibrator_kappa
from grn.utils_t2iv.sr_native import validate_native_config
from grn.models.hbq_tokenizer import CausalConv3d, Resample, patchify
from grn.models.rope import precompute_rope3d_freqs_grid
from grn.schedules.dynamic_resolution import get_dynamic_resolution_meta
from grn.utils_t2iv.dist import for_visualize
from grn.utils_t2iv.hbq_util_t2iv import multiclass_labels2onehot_input
from grn.utils_t2iv.sequence_parallel import SequenceParallelManager as sp_manager
from grn.utils_t2iv.sequence_parallel import sp_gather_sequence_by_dim, sp_split_sequence_by_dim


class MultipleLayers(nn.Module):
    """A sequential container for a chunk of multiple transformer blocks."""

    def __init__(self, layers: List[nn.Module], num_blocks: int, start_index: int):
        super().__init__()
        self.module = nn.ModuleList([
            layers[i] for i in range(start_index, start_index + num_blocks)
        ])

    def forward(
        self, x, cu_seqlens, max_seqlen, e0: Optional[torch.Tensor], 
        attn_bias_or_two_vector: Optional[Any], attn_fn: Optional[Any] = None, 
        checkpointing_full_block: bool = False, rope2d_freqs_grid: Optional[torch.Tensor] = None, 
        scale_ind: Optional[Any] = None, context_info: Optional[Any] = None, 
        last_diffusion_step: bool = True, ref_text_scale_inds: Optional[List[Any]] = None, 
        use_cfg: bool = False, split_cond_uncond: Optional[List[Any]] = None
    ) -> torch.Tensor:
        h = x
        for m in self.module:
            if checkpointing_full_block:
                h = torch.utils.checkpoint.checkpoint(
                    m, h, cu_seqlens, max_seqlen, e0, attn_bias_or_two_vector, attn_fn, 
                    rope2d_freqs_grid, scale_ind, context_info, 
                    last_diffusion_step, ref_text_scale_inds, 
                    use_cfg, split_cond_uncond, use_reentrant=False
                )
            else:
                h = m(
                    h, cu_seqlens, max_seqlen, e0, attn_bias_or_two_vector, attn_fn, 
                    rope2d_freqs_grid, scale_ind, context_info, 
                    last_diffusion_step, ref_text_scale_inds, 
                    use_cfg, split_cond_uncond
                )
        return h


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """
    Generate 1D sinusoidal embeddings.
    
    Args:
        dim (int): Embedding dimension (must be even).
        position (torch.Tensor): Position tensor of shape [B, L].
        
    Returns:
        torch.Tensor: Embeddings of shape [B, L, dim].
    """
    if dim % 2 != 0:
        raise ValueError(f"Embedding dimension must be even, got {dim}")
        
    half = dim // 2
    b, l = position.shape
    position = position.reshape(-1).type(torch.float64)

    sinusoid = torch.outer(
        position, 
        torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.reshape(b, l, dim)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class HBQStrideOneShortcut(nn.Module):
    """Trainable channel projection replacing HBQ's pixel-unshuffle shortcut.

    The original shortcut doubles channels while halving H/W. Once the spatial
    stride is removed, a 1x1x1 projection is required for the same channel
    transition. Channel-repeat initialization is the exact analogue of the
    original average shortcut for a spatially constant feature map.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels % in_channels:
            raise ValueError(
                f"Stride-one HBQ shortcut requires an integer channel ratio, "
                f"got {in_channels} -> {out_channels}"
            )
        self.proj = CausalConv3d(in_channels, out_channels, kernel_size=1)
        self.reset_as_channel_repeat()

    @torch.no_grad()
    def reset_as_channel_repeat(self) -> None:
        ratio = self.proj.out_channels // self.proj.in_channels
        self.proj.weight.zero_()
        output_channels = torch.arange(
            self.proj.out_channels, device=self.proj.weight.device
        )
        input_channels = torch.div(output_channels, ratio, rounding_mode='floor')
        self.proj.weight[output_channels, input_channels, 0, 0, 0] = 1
        if self.proj.bias is not None:
            self.proj.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class HBQConditionEncoder(nn.Module):
    """Trainable stride-4 HBQ encoder copy for native 128x128 LR conditions.

    HBQ first performs parameter-free 2x patchification, followed by three
    spatial downsampling stages. Keeping patchification and stage 0, while
    converting stages 1 and 2 to stride one, gives an exact total stride of 4:
    128 -> 64 -> 32 -> 32 -> 32. All compatible HBQ weights are inherited.
    """

    STRIDE_ONE_STAGES = (1, 2)

    def __init__(self, hbq: Any, embedding_channels: int) -> None:
        super().__init__()
        self.codebook_dim = int(hbq.codebook_dim)
        self.encoder = deepcopy(hbq.encoder)
        self.pre_quant = deepcopy(hbq.conv1)
        self._convert_encoder_to_stride4(self.encoder)
        self.feature_proj = nn.Conv2d(
            self.codebook_dim, embedding_channels, kernel_size=1
        )
        nn.init.trunc_normal_(self.feature_proj.weight, std=0.02)
        if self.feature_proj.bias is not None:
            nn.init.zeros_(self.feature_proj.bias)
        self.requires_grad_(True)

    @classmethod
    def _convert_encoder_to_stride4(cls, encoder: nn.Module) -> None:
        for stage_index in cls.STRIDE_ONE_STAGES:
            stage = encoder.downsamples[stage_index]
            if not isinstance(stage.downsamples[-1], Resample):
                raise TypeError(
                    f"HBQ encoder stage {stage_index} does not end in Resample"
                )
            resample = stage.downsamples[-1]
            if not resample.mode.startswith('downsample'):
                raise ValueError(
                    f"HBQ encoder stage {stage_index} has unexpected mode {resample.mode}"
                )
            if not (
                isinstance(resample.resample, nn.Sequential)
                and len(resample.resample) == 2
                and isinstance(resample.resample[1], nn.Conv2d)
            ):
                raise TypeError(
                    f"HBQ encoder stage {stage_index} has an unsupported resampler"
                )

            # Reuse the pretrained 3x3 kernel at full spatial resolution.
            spatial_conv = resample.resample[1]
            spatial_conv.stride = (1, 1)
            spatial_conv.padding = (1, 1)
            resample.resample[0] = nn.Identity()
            resample.mode = 'downsample2d'
            if hasattr(resample, 'time_conv'):
                delattr(resample, 'time_conv')

            old_shortcut = stage.avg_shortcut
            shortcut = HBQStrideOneShortcut(
                old_shortcut.in_channels, old_shortcut.out_channels
            )
            reference = next(stage.parameters())
            stage.avg_shortcut = shortcut.to(
                device=reference.device, dtype=reference.dtype
            )

    @torch.no_grad()
    def initialize_from_hbq(self, hbq: Any) -> Dict[str, int]:
        """Restore the condition backbone after GRN's global initialization."""
        if int(hbq.codebook_dim) != self.codebook_dim:
            raise ValueError(
                f"HBQ latent channels changed: {hbq.codebook_dim} != {self.codebook_dim}"
            )
        source_encoder = deepcopy(hbq.encoder)
        self._convert_encoder_to_stride4(source_encoder)
        self.encoder.load_state_dict(source_encoder.state_dict(), strict=True)
        self.pre_quant.load_state_dict(hbq.conv1.state_dict(), strict=True)
        self.requires_grad_(True)
        return {
            'encoder_parameter_tensors': len(source_encoder.state_dict()),
            'pre_quant_parameter_tensors': len(hbq.conv1.state_dict()),
            'stride_one_stages': len(self.STRIDE_ONE_STAGES),
        }

    def forward(
        self, condition: torch.Tensor, return_latent: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if condition.ndim != 4 or condition.shape[1] != 3:
            raise ValueError(
                f"HBQ condition input must be [B, 3, H, W], got {tuple(condition.shape)}"
            )
        feature = patchify(condition.unsqueeze(2), patch_size=2)
        feature = self.encoder(feature)
        mu, _ = self.pre_quant(feature).chunk(2, dim=1)
        feature = torch.tanh(mu)
        if feature.shape[2] != 1:
            raise ValueError(
                f"Natural-image SR expects one temporal feature, got {feature.shape[2]}"
            )
        latent = feature[:, :, 0]
        projected = self.feature_proj(latent)
        if return_latent:
            return projected, latent
        return projected


class GatedLRCrossAttention(nn.Module):
    """Inject an LR feature grid into HR tokens through a gated residual.

    The first injection stages use a square local window; the last stage uses
    global cross-attention. A small non-zero gate keeps a warm-started GRN close
    to its original behavior while still allowing gradients to reach Q/K/V on
    the first optimization step.
    """

    def __init__(
        self,
        embed_dim: int,
        inner_dim: int,
        num_heads: int,
        spatial_size: int,
        window_size: int = 7,
        gate_init: float = 1e-3,
    ) -> None:
        super().__init__()
        if inner_dim % num_heads:
            raise ValueError(
                f"Cross-attention inner_dim must be divisible by num_heads, "
                f"got {inner_dim} and {num_heads}"
            )
        if window_size and window_size % 2 != 1:
            raise ValueError(f"Local window must be odd, got {window_size}")
        self.inner_dim = int(inner_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.inner_dim // self.num_heads
        self.spatial_size = int(spatial_size)
        self.window_size = int(window_size)
        self.hr_norm = FastRMSNorm(embed_dim)
        self.lr_norm = FastRMSNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, embed_dim, bias=False)
        self.gate = nn.Parameter(torch.tensor([float(gate_init)]))

        if self.window_size:
            radius = self.window_size // 2
            axis = torch.arange(self.spatial_size)
            yy, xx = torch.meshgrid(axis, axis, indexing='ij')
            coords = torch.stack((yy.flatten(), xx.flatten()), dim=-1)
            delta = (coords[:, None] - coords[None, :]).abs()
            local_mask = (delta[..., 0] <= radius) & (delta[..., 1] <= radius)
            local_mask = local_mask[None, None]
        else:
            local_mask = torch.empty(0, dtype=torch.bool)
        self.register_buffer('local_mask', local_mask, persistent=False)

    def forward(self, hr_tokens: torch.Tensor, lr_tokens: torch.Tensor) -> torch.Tensor:
        if hr_tokens.ndim != 3 or lr_tokens.ndim != 3:
            raise ValueError("Cross-attention expects [B, N, C] token tensors")
        expected_tokens = self.spatial_size ** 2
        if hr_tokens.shape[1] != expected_tokens or lr_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Cross-attention expects {expected_tokens} aligned spatial tokens, "
                f"got HR={hr_tokens.shape[1]} and LR={lr_tokens.shape[1]}"
            )

        batch_size, token_count = hr_tokens.shape[:2]
        q = self.q_proj(self.hr_norm(hr_tokens))
        k = self.k_proj(self.lr_norm(lr_tokens))
        v = self.v_proj(self.lr_norm(lr_tokens))
        q = q.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        attn_mask = self.local_mask if self.window_size else None
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        attended = attended.transpose(1, 2).reshape(batch_size, token_count, self.inner_dim)
        update = self.out_proj(attended)
        return hr_tokens + self.gate.to(dtype=update.dtype) * update


def bld_to_bthwd(item: torch.Tensor, patch_time: int, patch_height: int, patch_width: int, apply_spatial_patchify: bool = False) -> torch.Tensor:
    """Reshape a sequence tensor to a spatial tensor."""
    batch_size = item.shape[0]
    return item.reshape(batch_size, patch_time, patch_height, patch_width, -1)


def build_attn_mask(seqlens, device):
    attn_mask = torch.zeros((1, 1, sum(seqlens), sum(seqlens)), dtype=torch.bool, device=device)
    q_start = 0
    for i in range(len(seqlens)):
        q_len = seqlens[i]
        q_end = q_start + q_len
        attn_mask[:, :, q_start:q_end, q_start:q_end] = True
        q_start = q_end
    return attn_mask


class FsqHead(nn.Module):
    """Classification head for Finite Scalar Quantization (FSQ)."""

    def __init__(self, hidden_dim: int, fsq_dim: int, fsq_lvl: int, use_ada_layer_norm: bool, eps: float = 1e-6):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, fsq_dim * fsq_lvl)
        self.norm = FastRMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor, e: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.amp.autocast('cuda', dtype=torch.float32):
            return self.proj(self.norm(x))


class GRN(nn.Module):
    def __init__(
        self,
        vae_local: Any,
        arch: str = 'var',
        qwen_qkvo_bias: bool = False,
        text_channels: int = 0,
        text_maxlen: int = 0,
        embed_dim: int = 1024,
        depth: int = 16,
        num_key_value_heads: int = -1,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_eps: float = 1e-6,
        block_chunks: int = 1,
        checkpointing: Optional[str] = None,
        use_flex_attn: bool = False,
        num_of_label_value: int = 2,
        rope2d_normalized_by_hw: int = 0,
        pn: Optional[str] = None,
        video_frames: int = 1,
        always_training_scales: int = 20,
        apply_spatial_patchify: int = 0,
        inference_mode: bool = False,
        other_args: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__()
        # 1. Model Configuration
        self.embed_dim = embed_dim
        self.C = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.arch = arch
        self.mlp_ratio = mlp_ratio
        self.norm_eps = norm_eps
        self.drop_path_rate = drop_path_rate
        self.use_flex_attn = use_flex_attn
        self.checkpointing = checkpointing
        self.inference_mode = inference_mode
        self.other_args = other_args
        from grn.models.sr_deterministic_attention import validate_policy
        self.sr_deterministic_attention = validate_policy(other_args)

        # 2. Embedding & Scale Configuration
        self.vae_embed_dim = vae_local.codebook_dim
        self.apply_spatial_patchify = apply_spatial_patchify
        self.text_channels = text_channels
        self.text_maxlen = text_maxlen
        self.is_text_to_image = text_channels != 0
        self.sr_condition = bool(getattr(other_args, 'sr_condition', 0))
        self.sr_condition_mode = getattr(other_args, 'sr_condition_mode', 'prefix')
        validate_native_config(other_args)
        self.sr_lr_size = int(getattr(other_args, 'sr_lr_size', 128))
        self.sr_hr_size = int(getattr(other_args, 'sr_hr_size', 512))
        self.sr_latent_size = self.sr_hr_size // 16
        self.sr_condition_drop_rate = float(getattr(other_args, 'sr_condition_drop_rate', 0.1))
        self.use_sr_cross_attn = bool(getattr(other_args, 'sr_cross_attn', 0))
        cross_layers = getattr(other_args, 'sr_cross_attn_layers', '[4, 12, 20, 28]')
        if isinstance(cross_layers, str):
            cross_layers = json.loads(cross_layers)
        self.sr_cross_attn_layers = tuple(int(layer) for layer in cross_layers)
        self.sr_cross_attn_dim = int(getattr(other_args, 'sr_cross_attn_dim', 512))
        self.sr_cross_attn_heads = int(getattr(other_args, 'sr_cross_attn_heads', 8))
        self.sr_cross_attn_window = int(getattr(other_args, 'sr_cross_attn_window', 7))
        self.sr_cross_attn_gate_init = float(
            getattr(other_args, 'sr_cross_attn_gate_init', 1e-3)
        )
        if self.sr_condition:
            if self.sr_hr_size != self.sr_lr_size * 4:
                raise ValueError(
                    f"SR mode requires 4x scaling, got {self.sr_lr_size} -> {self.sr_hr_size}"
                )
            if self.sr_hr_size % 16:
                raise ValueError(f"sr_hr_size must be divisible by 16, got {self.sr_hr_size}")

        classifier_head_dim = other_args.detail_scale_dim
        classifier_head_lvl = other_args.detail_num_lvl
        hbq_round = other_args.hbq_round

        if other_args.refine_mode in ['ar_discrete_GRN_ind']:
            self.visual_embedding_in_dim = vae_local.codebook_dim * (2**hbq_round)
            classifier_head_dim = vae_local.codebook_dim
        elif other_args.refine_mode in ['ar_discrete_GRN_bit']:
            self.visual_embedding_in_dim = hbq_round * vae_local.codebook_dim * 2
            classifier_head_dim = hbq_round * vae_local.codebook_dim
        else:
            self.visual_embedding_in_dim = vae_local.codebook_dim

        if self.apply_spatial_patchify:
            self.visual_embedding_in_dim *= 4

        # 3. Dynamic Resolution & Video Specifics
        self.video_frames = video_frames
        self.always_training_scales = always_training_scales
        self.num_of_label_value = num_of_label_value
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw

        self.dynamic_resolution_h_w, self.h_div_w_templates = get_dynamic_resolution_meta(
            other_args.dynamic_scale_schedule, other_args.train_h_div_w_list, other_args.video_frames
        )
        self.train_h_div_w_list = self.h_div_w_templates
        print(f"train_h_div_w_list: {self.train_h_div_w_list}")

        # 4. Utilities
        self.entrophy_statistics = []
        self.top_p, self.top_k = 1.0, 100
        self.rng = torch.Generator(device=dist.get_device())
        self.maybe_record_function = nullcontext
        self.infer_ts = None

        # 5. Model Components (Projections, Embeddings)
        self.norm0_cond = nn.Identity()
        self.text_proj = nn.Linear(self.text_channels, self.embed_dim)

        if self.other_args.use_ada_layer_norm:
            self.scale_or_time_dim = 256
            self.scale_or_time_embedding = nn.Sequential(
                nn.Linear(self.scale_or_time_dim, self.embed_dim), nn.SiLU(), nn.Linear(self.embed_dim, self.embed_dim),
            )
            self.scale_or_time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.embed_dim, self.embed_dim * 6))

        # RoPE grid initialization
        with torch.amp.autocast('cuda', dtype=torch.float32):
            self.rope2d_freqs_grid = precompute_rope3d_freqs_grid(
                dim=self.embed_dim // self.num_heads,
                rope2d_normalized_by_hw=self.rope2d_normalized_by_hw,
                activated_h_div_w_templates=self.train_h_div_w_list,
                max_scales=1010, # never used
                max_frames=int(self.video_frames / other_args.temporal_compress_rate + 1),
                max_height=1800 // 8, 
                max_width=1800 // 8,
                text_maxlen=self.text_maxlen,
                args=other_args,
            )

        self.word_embed = nn.Linear(self.visual_embedding_in_dim, self.embed_dim)
        if self.sr_condition and uses_continuous_lr(self.sr_condition_mode):
            self.sr_condition_encoder = HBQConditionEncoder(vae_local, self.embed_dim)
            self.sr_prefix_pos = nn.Parameter(
                torch.empty(1, self.sr_latent_size ** 2, self.embed_dim)
            )
            nn.init.trunc_normal_(self.sr_prefix_pos, std=0.02)
            self.sr_cross_attn = nn.ModuleDict()
            if self.use_sr_cross_attn:
                if not self.sr_cross_attn_layers:
                    raise ValueError("sr_cross_attn_layers cannot be empty when enabled")
                invalid_layers = [
                    layer for layer in self.sr_cross_attn_layers
                    if layer < 1 or layer > depth
                ]
                if invalid_layers:
                    raise ValueError(
                        f"Cross-attention layers outside depth {depth}: {invalid_layers}"
                    )
        self.head = FsqHead(
            hidden_dim=self.embed_dim, 
            fsq_dim=classifier_head_dim, 
            fsq_lvl=classifier_head_lvl, 
            use_ada_layer_norm=other_args.use_ada_layer_norm,
        )

        if other_args.add_scale_token > 0:
            self.pt_embedder = TimestepEmbedder(self.embed_dim)

        # 6. Transformer Blocks
        self.attn_fn_compile_dict = {}
        self.unregistered_blocks = []
        for block_idx in range(depth):
            block = SelfAttnBlock(
                embed_dim=self.embed_dim, 
                num_heads=num_heads, 
                num_key_value_heads=num_key_value_heads, 
                mlp_ratio=mlp_ratio,
                use_flex_attn=use_flex_attn,
                sr_deterministic_attention=self.sr_deterministic_attention,
                qwen_qkvo_bias=qwen_qkvo_bias,
                use_ada_layer_norm=other_args.use_ada_layer_norm,
            )
            self.unregistered_blocks.append(block)

        self.num_block_chunks = block_chunks or 1
        self.num_blocks_in_a_chunk = depth // self.num_block_chunks
        assert self.num_blocks_in_a_chunk * self.num_block_chunks == depth, "Depth must be divisible by block_chunks"
        
        self.block_chunks = nn.ModuleList([
            MultipleLayers(self.unregistered_blocks, self.num_blocks_in_a_chunk, i * self.num_blocks_in_a_chunk)
            for i in range(self.num_block_chunks)
        ])

        if self.sr_condition and self.sr_condition_mode == 'native_adapter':
            layers = getattr(other_args, 'sr_native_adapter_layers', '[4, 12, 20, 28]')
            if isinstance(layers, str):
                layers = json.loads(layers)
            chunk_ends = {(i + 1) * self.num_blocks_in_a_chunk for i in range(self.num_block_chunks)}
            if (not isinstance(layers, (list, tuple)) or not layers
                or any(type(layer) is not int for layer in layers)
                or len(set(layers)) != len(layers) or not set(layers).issubset(chunk_ends)):
                raise ValueError(f'sr_native_adapter_layers must be distinct chunk ends in {sorted(chunk_ends)}')
            self.sr_native_adapters = nn.ModuleDict({
                str(layer): NativeLRSpatialAdapter(self.embed_dim, getattr(other_args, 'sr_native_adapter_dim', 256))
                for layer in layers
            })

        if self.sr_condition and self.sr_condition_mode == 'spatial_control':
            layers = getattr(other_args, 'sr_control_layers', '[4, 12, 20, 28]')
            if isinstance(layers, str):
                layers = json.loads(layers)
            chunk_ends = {(i + 1) * self.num_blocks_in_a_chunk for i in range(self.num_block_chunks)}
            if (not isinstance(layers, (list, tuple)) or not layers
                or any(type(layer) is not int for layer in layers)
                or len(set(layers)) != len(layers) or not set(layers).issubset(chunk_ends)):
                raise ValueError(f'sr_control_layers must be distinct chunk ends in {sorted(chunk_ends)}')
            self.sr_spatial_controls = nn.ModuleDict({
                str(layer): HRQuerySpatialControl(
                    self.embed_dim,
                    inner_dim=getattr(other_args, 'sr_control_dim', 512),
                    num_heads=getattr(other_args, 'sr_control_heads', 8),
                    window_size=getattr(other_args, 'sr_control_window', 7),
                )
                for layer in layers
            })

        if self.sr_condition and self.use_sr_cross_attn:
            chunk_ends = {
                (index + 1) * self.num_blocks_in_a_chunk
                for index in range(self.num_block_chunks)
            }
            unsupported = set(self.sr_cross_attn_layers) - chunk_ends
            if unsupported:
                raise ValueError(
                    "Cross-attention is inserted after transformer chunks; "
                    f"layers {sorted(unsupported)} are not chunk ends {sorted(chunk_ends)}"
                )
            windows = resolve_cross_attention_windows(
                self.sr_cross_attn_layers, self.sr_cross_attn_window,
                getattr(other_args, 'sr_cross_attn_global_layers', ''),
            )
            for layer in self.sr_cross_attn_layers:
                window_size = windows[layer]
                self.sr_cross_attn[str(layer)] = GatedLRCrossAttention(
                    embed_dim=self.embed_dim,
                    inner_dim=self.sr_cross_attn_dim,
                    num_heads=self.sr_cross_attn_heads,
                    spatial_size=self.sr_latent_size,
                    window_size=window_size,
                    gate_init=self.sr_cross_attn_gate_init,
                )
            print(
                f"    [SR CrossAttn] layers={self.sr_cross_attn_layers}, "
                f"windows={windows}, "
                f"inner_dim={self.sr_cross_attn_dim}, heads={self.sr_cross_attn_heads}"
            )

        print(f"    [Model Config] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, "
              f"mlp_ratio={mlp_ratio}, num_blocks_in_a_chunk={self.num_blocks_in_a_chunk}")
        print(f"    drop_path_rate={drop_path_rate:g}", end='\n\n', flush=True)
        
    def get_loss_acc(
        self,
        hidden_states: torch.Tensor,
        hidden_states_mask: Optional[torch.Tensor],
        e: Optional[torch.Tensor],
        sequence_packing_scales: List[List[Tuple[int, int, int]]],
        gt: List[torch.Tensor],
        other_info_by_scale: List[Dict[str, Any]],
        return_last_hidden_states: bool,
        sr_v2_inference_precision: bool = False,
        sr_calibrator_latent: Optional[torch.Tensor] = None,
        sr_calibrator_kappa: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Calculate loss and accuracy for the predicted logits.
        
        Args:
            hidden_states: shaped (B, L, C)
            hidden_states_mask: Optional mask for hidden states
            e: scale or time embeddings
            sequence_packing_scales: List of scales for sequence packing
            gt: Ground truth labels
            other_info_by_scale: Meta information for each scale
            return_last_hidden_states: Whether to return the last hidden states
            
        Returns:
            Tuple of (logits_norm, loss_list, acc_list)
        """
        calibration_enabled = hasattr(self, 'sr_round_calibrator')
        calibration_kwargs = {}
        if sr_calibrator_kappa is not None:
            if not calibration_enabled:
                raise ValueError('Runtime kappa requires an enabled SR calibrator')
            calibration_kwargs['kappa'] = resolve_calibrator_kappa(sr_calibrator_kappa)
        if calibration_enabled:
            if (sr_calibrator_latent is None or sr_calibrator_latent.ndim != 4
                    or sr_calibrator_latent.shape[0] != len(sequence_packing_scales)
                    or any(len(scales) != 1 or scales[0][0] != 1
                           for scales in sequence_packing_scales)):
                raise ValueError('S3 calibration requires one aligned LR latent per packed image')
        logits_norm = []
        logits_full = self.head(hidden_states.float() if sr_v2_inference_precision else hidden_states, e)
        global_token_ptr, global_scale_ptr = 0, 0
        loss_list, acc_list, predicted_logits = [], [], []

        for pack_scales in sequence_packing_scales:
            for pt, ph, pw in pack_scales:
                mul_pt_ph_pw = pt * ph * pw
                cur_bits = other_info_by_scale[global_scale_ptr]['cur_bits']
                cur_lvl = other_info_by_scale[global_scale_ptr]['cur_lvl']
                predict_tokens = other_info_by_scale[global_scale_ptr]['predict_tokens']
                all_tokens = other_info_by_scale[global_scale_ptr]['all_tokens']
                target_token_offset = other_info_by_scale[global_scale_ptr].get('target_token_offset', 0)
                target_start = global_token_ptr + target_token_offset
                logits = logits_full[:, target_start:target_start + predict_tokens]
                logits = logits.reshape(hidden_states.shape[0], mul_pt_ph_pw, cur_bits, cur_lvl)
                logits = logits.permute(0, 3, 1, 2) # [1, num_of_label_value, mul_pt_ph_pw, d]
                if calibration_enabled:
                    if (cur_bits, cur_lvl, predict_tokens) != (256, 2, mul_pt_ph_pw):
                        raise ValueError('S3 calibration requires absolute M4 x 64 binary HR logits')
                    logits = self._calibrate_sr_hr_logits(
                        logits,
                        hidden_states[:, target_start:target_start + predict_tokens],
                        sr_calibrator_latent[global_scale_ptr:global_scale_ptr + 1],
                        hidden_states.new_tensor(
                            [other_info_by_scale[global_scale_ptr]['scale_token_id']],
                            dtype=torch.float32),
                        **calibration_kwargs,
                    )
                predicted_logits.append(logits)
                
                logits_norm.append(logits.abs().mean())
                
                # gt[global_scale_ptr]: [1, mul_pt_ph_pw, d]
                loss_this_scale = F.cross_entropy(logits, gt[global_scale_ptr], reduction='none')[0] # [mul_pt_ph_pw, d]
                acc_this_scale = (logits.argmax(1) == gt[global_scale_ptr]).float()[0] # [mul_pt_ph_pw, d]
                
                loss_list.append(loss_this_scale.mean(-1))
                acc_list.append(acc_this_scale.mean(-1))
                
                global_scale_ptr += 1
                global_token_ptr += all_tokens
                
        loss_tensor = torch.cat(loss_list) if loss_list else torch.tensor([], device=hidden_states.device)
        acc_tensor = torch.cat(acc_list) if acc_list else torch.tensor([], device=hidden_states.device)
        logits_norm_tensor = torch.stack(logits_norm).mean() if logits_norm else torch.tensor(0.0, device=hidden_states.device)
        
        return logits_norm_tensor, loss_tensor, acc_tensor, predicted_logits
    
    def get_logits_during_infer(self, hidden_states: torch.Tensor, e: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get logits during inference."""
        return self.head(hidden_states.float(), e)

    def configure_sr_v2_calibrator(
        self, *, mode: str = 'off', width: int = 128,
        max_log_odds: float = 2., seed: int = 20260917,
    ) -> None:
        """Candidate-only materialization after common init and before FSDP.

        This method does not migrate checkpoints or authorize a training run.
        Default off adds no state and consumes no RNG. Old state loading remains
        strict until a separately audited S3 promotion path is implemented.
        """
        if mode not in ('off', 'plain', 'shared_round', 'per_round'):
            raise ValueError('Unknown S3 calibration mode')
        if hasattr(self, 'sr_round_calibrator'):
            raise RuntimeError('S3 calibrator already configured; refusing overwrite')
        if mode == 'off':
            return
        if (not self.sr_condition or self.sr_condition_mode != 'prefix_text'
                or self.sr_condition_encoder is None
                or (self.sr_condition_drop_rate != 0
                    and getattr(self.other_args, 'sr_v2_condition_dropout_pilot', 0) != 1)
                or (self.other_args.hbq_round, self.other_args.detail_scale_dim,
                    self.other_args.detail_num_lvl) != (4, 64, 2)):
            raise ValueError('S3 requires prefix_text, M4 x 64 binary HBQ and no condition dropout')
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('S3 seed must be an integer in [0, 2**63)')
        reference = next(self.sr_condition_encoder.parameters())
        if reference.dtype != torch.float32:
            raise ValueError('S3 candidate materialization requires FP32 parameters')
        from grn.models.sr_v2_calibrator import BoundedRoundCalibrator
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            module = BoundedRoundCalibrator(hr_dim=self.C, lr_dim=64, width=width,
                mode=mode, max_log_odds=max_log_odds)
        module.to(device=reference.device)
        module.train(self.training)
        self.add_module('sr_round_calibrator', module)

    def _calibrate_sr_hr_logits(self, logits, hidden, lr_latent, retention, *, kappa=None):
        # Do not silently replace a reduced-precision baseline with FP32.
        # Actual GPU inference/training dtype and zero-delta identity remain gates.
        if logits.dtype != torch.float32:
            raise ValueError('S3 integration currently requires native FP32 head logits')
        control = {} if kappa is None else {'kappa': kappa}
        return self.sr_round_calibrator.calibrate_logits(logits, hidden, lr_latent, retention, **control)

    def initialize_sr_condition_from_hbq(self, vae_local: Any) -> Dict[str, int]:
        """Initialize the trainable LR branch from the frozen target HBQ."""
        if not self.sr_condition:
            return {}
        return self.sr_condition_encoder.initialize_from_hbq(vae_local)

    def configure_sr_v2_condition_modules(
        self, *, semantic: bool = False, semantic_hidden: int = 128,
        coarse: bool = False, seed: int = 20260916,
    ) -> None:
        """Create optional S2 modules after common initialization, before FSDP.

        Off is a true no-op. Each enabled branch has an isolated CPU RNG stream;
        changing the coarse switch does not change semantic initialization.
        """
        if type(semantic) is not bool or type(coarse) is not bool:
            raise ValueError('semantic and coarse switches must be bool')
        if any(hasattr(self, name) for name in ('sr_semantic_fusion', 'sr_coarse_head')):
            raise RuntimeError('S2 condition modules are already configured; refusing parameter overwrite')
        if not semantic and not coarse:
            return
        if (not self.sr_condition or self.sr_condition_mode not in ('prefix', 'prefix_text')
                or self.sr_condition_encoder is None):
            raise ValueError('S2 condition modules require continuous LR prefix or prefix_text mode')
        if type(semantic_hidden) is not int or semantic_hidden < 1:
            raise ValueError('semantic_hidden must be a positive integer')
        if type(seed) is not int or not 0 <= seed < 2 ** 63:
            raise ValueError('S2 initialization seed must be an integer in [0, 2**63)')
        from grn.models.sr_v2_condition import SRSemanticFusion, SRCoarseHead
        reference = next(self.sr_condition_encoder.parameters())
        modules = {}
        with torch.random.fork_rng(devices=[]):
            if semantic:
                torch.random.default_generator.manual_seed(seed)
                modules['sr_semantic_fusion'] = SRSemanticFusion(self.C, semantic_hidden)
            if coarse:
                torch.random.default_generator.manual_seed(seed + 1)
                modules['sr_coarse_head'] = SRCoarseHead()
        for name, module in modules.items():
            module.to(device=reference.device, dtype=reference.dtype)
            module.train(self.training)
            self.add_module(name, module)

    def _validate_sr_v2_semantic_condition(self, lr_condition, sr_semantic_features):
        enabled = hasattr(self, 'sr_semantic_fusion')
        if enabled and (lr_condition is None or sr_semantic_features is None):
            raise ValueError('Enabled semantic fusion requires LR and sr_semantic_features')
        if not enabled and sr_semantic_features is not None:
            raise ValueError('Received sr_semantic_features but semantic fusion is disabled')

    def encode_lr_condition(
        self,
        lr_condition: torch.Tensor,
        target_hw: Optional[Tuple[int, int]] = None,
        drop_condition: bool = False,
        return_latent: bool = False,
        sr_semantic_features: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Turn native LR images into scale-aligned HBQ prefix tokens."""
        self._validate_sr_v2_semantic_condition(lr_condition, sr_semantic_features)
        if not self.sr_condition:
            raise RuntimeError("LR conditioning is disabled; set --sr_condition=1")
        if lr_condition.ndim != 4 or lr_condition.shape[1] != 3:
            raise ValueError(
                f"lr_condition must have shape [B, 3, H, W], got {tuple(lr_condition.shape)}"
            )
        input_hw = tuple(lr_condition.shape[-2:])
        native_hw = (self.sr_lr_size, self.sr_lr_size)
        if input_hw != native_hw:
            raise ValueError(
                f"Direct SR condition encoder requires native {native_hw}, got {input_hw}"
            )

        condition_features, condition_latent = self.sr_condition_encoder(
            lr_condition.float(), return_latent=True
        )
        condition_hw = tuple(condition_features.shape[-2:])
        expected_hw = target_hw or (self.sr_latent_size, self.sr_latent_size)
        if condition_hw != expected_hw:
            raise ValueError(
                f"LR prefix grid {condition_hw} does not match HR latent grid {expected_hw}"
            )
        if hasattr(self, 'sr_semantic_fusion'):
            condition_features = self.sr_semantic_fusion(condition_features, sr_semantic_features)
        prefix = condition_features.flatten(2).transpose(1, 2)
        if prefix.shape[1] != self.sr_prefix_pos.shape[1]:
            raise ValueError(
                f"Only fixed {self.sr_lr_size} -> {self.sr_hr_size} SR is supported; "
                f"got {prefix.shape[1]} prefix tokens"
            )
        prefix = prefix + self.sr_prefix_pos.to(dtype=prefix.dtype)
        if drop_condition:
            prefix = torch.zeros_like(prefix)
        elif self.training and self.sr_condition_drop_rate > 0:
            drop_mask = torch.rand(prefix.shape[0], 1, 1, device=prefix.device)
            prefix = torch.where(
                drop_mask < self.sr_condition_drop_rate, torch.zeros_like(prefix), prefix
            )
        if return_latent:
            return prefix, condition_latent
        return prefix

    def _apply_sr_cross_attention(
        self,
        hidden_states: torch.Tensor,
        lr_prefix: torch.Tensor,
        contexts: List[Tuple[int, int, int, int]],
        layer_number: int,
        checkpointing: bool = False,
    ) -> torch.Tensor:
        """Update only the HR portion of each packed SR sequence."""
        layer_key = str(layer_number)
        if layer_key not in self.sr_cross_attn:
            return hidden_states
        cross_attn = self.sr_cross_attn[layer_key]
        pieces = []
        cursor = 0
        for sequence_start, prefix_len, target_len, condition_index in contexts:
            target_start = sequence_start + prefix_len
            target_end = target_start + target_len
            pieces.append(hidden_states[:, cursor:target_start])
            target = hidden_states[:, target_start:target_end]
            condition = lr_prefix[condition_index:condition_index + 1]
            if checkpointing:
                target = torch.utils.checkpoint.checkpoint(
                    cross_attn, target, condition, use_reentrant=False
                )
            else:
                target = cross_attn(target, condition)
            pieces.append(target)
            cursor = target_end
        pieces.append(hidden_states[:, cursor:])
        return torch.cat(pieces, dim=1)
    
    def _apply_sr_native_adapters(self, hidden_states, source, contexts, layer_number, checkpointing=False):
        """Inject source residuals over HR spans in the original packed sequence."""
        adapter = self.sr_native_adapters[str(layer_number)]
        pieces, cursor = [], 0
        for sequence_start, hr_len, source_index, grid_hw in contexts:
            end = sequence_start + hr_len
            pieces.append(hidden_states[:, cursor:sequence_start])
            target = hidden_states[:, sequence_start:end]
            condition = source[source_index:source_index + 1]
            if checkpointing:
                target = torch.utils.checkpoint.checkpoint(adapter, target, condition, grid_hw, use_reentrant=False)
            else:
                target = adapter(target, condition, grid_hw)
            pieces.append(target)
            cursor = end
        pieces.append(hidden_states[:, cursor:])
        return torch.cat(pieces, dim=1)

    def forward(
        self,
        label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]],
        x_BLC: torch.Tensor,
        visual_rope_cache: Optional[List[torch.Tensor]] = None,
        sequece_packing_scales: Optional[List[List[Tuple[int, int, int]]]] = None,
        super_scale_lengths: Optional[List[int]] = None,
        other_info_by_scale: Optional[List[Dict[str, Any]]] = None,
        gt_BL: Optional[List[torch.Tensor]] = None,
        x_BLC_mask: Optional[torch.Tensor] = None,
        scale_or_time_ids: Optional[torch.Tensor] = None,
        return_last_hidden_states: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Forward pass for the GRN model.
        
        Args:
            label_B_or_BLT: Text conditions or labels
            x_BLC: Input sequence hidden states
            visual_rope_cache: Cache for visual RoPE embeddings
            sequece_packing_scales: Scales for sequence packing
            super_scale_lengths: Lengths of super scales
            other_info_by_scale: Meta info for scales
            gt_BL: Ground truth
            x_BLC_mask: Mask for input sequence
            scale_or_time_ids: IDs for scale or time embeddings
            return_last_hidden_states: Whether to return last hidden states
            
        Returns:
            Tuple of (logits_norm, loss_list, acc_list, valid_sequence_ratio)
        """
        device = x_BLC[0].device
        lr_condition = kwargs.pop('lr_condition', None)
        sr_semantic_features = kwargs.pop('sr_semantic_features', None)
        self._validate_sr_v2_semantic_condition(lr_condition, sr_semantic_features)
        native_lr_input = kwargs.pop('native_lr_input', None)
        native_lr_latent = kwargs.pop('native_lr_latent', None)
        return_sr_aux = bool(kwargs.pop('return_sr_aux', False))
        sr_v2_inference_precision = bool(kwargs.pop('sr_v2_inference_precision', False))
        sr_calibrator_kappa = kwargs.pop('sr_calibrator_kappa', None)
        calibration_kwargs = {}
        if sr_calibrator_kappa is not None:
            if not hasattr(self, 'sr_round_calibrator'):
                raise ValueError('Runtime kappa requires an enabled SR calibrator')
            calibration_kwargs['sr_calibrator_kappa'] = resolve_calibrator_kappa(sr_calibrator_kappa)
        if sr_v2_inference_precision and (self.training or torch.is_grad_enabled()):
            raise ValueError('V2 inference precision is only valid for eval/no-grad rollout')
        if sr_v2_inference_precision and (len(x_BLC) != 1 or self.sr_condition_mode != 'prefix_text'):
            raise ValueError('V2 inference precision currently requires one B / prefix_text sample')
        if hasattr(self, 'sr_round_calibrator') and lr_condition is None:
            raise ValueError('Enabled S3 calibration requires LR conditioning')
        native_mode = self.sr_condition_mode == 'native_adapter'
        use_native_condition = native_lr_input is not None
        use_sr_condition = lr_condition is not None and uses_continuous_lr(self.sr_condition_mode)
        use_sr_prefix = use_sr_condition and uses_lr_prefix(self.sr_condition_mode)
        use_text = not use_sr_condition or uses_empty_text(self.sr_condition_mode)
        use_spatial_control = use_sr_condition and self.sr_condition_mode == 'spatial_control'
        if (lr_condition is not None or use_native_condition) and not self.sr_condition:
            raise RuntimeError("Received lr_condition but the model was not built in SR mode")
        if use_native_condition and not native_mode:
            raise ValueError('native_lr_input requires native SR mode')
        if native_mode and lr_condition is not None and not use_native_condition:
            raise ValueError('Encode native LR outside GRN and pass native_lr_input explicitly')

        # [1. get input sequence x_BLC]
        # word embedding
        sub_L_list = [item.shape[1] for item in x_BLC]
        # Production infer embeds the noncontiguous one-hot view directly.
        # Even cat([single_tensor]) changes strides and BF16 GEMM rounding.
        cat_x_BLC = x_BLC[0] if sr_v2_inference_precision else torch.cat(x_BLC, dim=1)
        with (nullcontext() if sr_v2_inference_precision else torch.amp.autocast('cuda', dtype=torch.float32)):
            cat_x_BLC = self.word_embed(cat_x_BLC.float())
        x_BLC = list(torch.split(cat_x_BLC, sub_L_list, dim=1))

        if use_native_condition:
            if native_lr_input.ndim != 3 or native_lr_input.shape[0] != len(x_BLC):
                raise ValueError('native source batch must match the packed HR samples')
            with torch.amp.autocast('cuda', dtype=torch.float32):
                native_source = self.word_embed(native_lr_input.float())
            sr_condition_latent = native_lr_latent
            if return_sr_aux and sr_condition_latent is None:
                raise ValueError('native SR auxiliary output requires native_lr_latent')

        if use_sr_condition:
            if len(x_BLC) != lr_condition.shape[0]:
                raise ValueError(
                    f"Got {len(x_BLC)} packed HR samples but {lr_condition.shape[0]} LR images"
                )
            if any(len(scales) != 1 or scales[0][0] != 1 for scales in sequece_packing_scales):
                raise ValueError('continuous SR requires a single image scale per sample')
            target_hws = [
                (int(scale[0][1]), int(scale[0][2])) for scale in sequece_packing_scales
            ]
            if any(hw != target_hws[0] for hw in target_hws):
                raise ValueError("An SR batch must use one shared target latent resolution")
            lr_prefix, sr_condition_latent = self.encode_lr_condition(
                lr_condition, target_hw=target_hws[0], return_latent=True,
                sr_semantic_features=sr_semantic_features,
            )
            lr_prefix_splits = list(lr_prefix.split(1, dim=0))
        if use_text:
            # text tokens embedding
            kv_compact, lens, cu_seqlens_k, max_seqlen_k, _ = label_B_or_BLT
            with (nullcontext() if sr_v2_inference_precision else torch.amp.autocast('cuda', dtype=torch.float32)):
                if sr_v2_inference_precision:
                    # Guidance=1 infer still projects cond+uncond empty text
                    # together; preserve that GEMM shape before taking cond.
                    count = kv_compact.shape[0]
                    kv_compact = self.text_proj(torch.cat((kv_compact, kv_compact), dim=0))[:count].contiguous()
                else:
                    kv_compact = self.text_proj(kv_compact).contiguous() # [sum(lens), C]
            kv_compact_splits = torch.split(kv_compact, lens, dim=0)

        # scale tokens embedding
        scale_token_ids = torch.tensor([info["scale_token_id"] for info in other_info_by_scale], device=device)
        with (nullcontext() if sr_v2_inference_precision else torch.amp.autocast("cuda", dtype=torch.float32)):
            pt_tokens = self.pt_embedder((scale_token_ids)) # [num_scales, C]
        
        # Pack the explicit SR layout; other tasks retain [visual, text, scale].
        x_BLC_lists = []
        actual_scale_lengths = []
        sr_cross_contexts = []
        sr_native_contexts = []
        sr_spatial_contexts = []
        sequence_cursor = 0
        for i in range(len(x_BLC)):
            if use_sr_condition:
                layout = condition_layout(self.sr_condition_mode, x_BLC[i].shape[1], lens[i] if use_text else 0)
                other_info_by_scale[i]['target_token_offset'] = layout.target_offset
                other_info_by_scale[i]['all_tokens'] = layout.total_tokens
                current_parts = ([lr_prefix_splits[i]] if layout.prefix_tokens else []) + [x_BLC[i]]
                if layout.text_tokens:
                    current_parts.append(kv_compact_splits[i].unsqueeze(0))
                current_parts.append(pt_tokens[i][None, None])
                sr_cross_contexts.append(
                    (sequence_cursor, layout.target_offset, x_BLC[i].shape[1], i)
                )
                if use_spatial_control:
                    sr_spatial_contexts.append((sequence_cursor + layout.target_offset,
                        x_BLC[i].shape[1], i, target_hws[i]))
            else:
                current_parts = [
                    x_BLC[i], kv_compact_splits[i].unsqueeze(0), pt_tokens[i][None, None]
                ]
                if use_native_condition:
                    scales = sequece_packing_scales[i]
                    if len(scales) != 1 or scales[0][0] != 1:
                        raise ValueError('native SR requires a single image scale per sample')
                    grid_hw = (int(scales[0][1]), int(scales[0][2]))
                    sr_native_contexts.append((sequence_cursor, x_BLC[i].shape[1], i, grid_hw))
            x_BLC_lists.extend(current_parts)
            actual_scale_lengths.append(sum(item.shape[1] for item in current_parts))
            sequence_cursor += actual_scale_lengths[-1]
        x_BLC = torch.cat(x_BLC_lists, dim=1)
        if use_spatial_control:
            # The encoder may run under AMP while the residual stream is FP32.
            lr_prefix = lr_prefix.to(dtype=x_BLC.dtype)

        token_budget = max(int(getattr(self.other_args, 'train_max_token_len', x_BLC.shape[1])), 1)
        valid_sequence_ratio = x_BLC.shape[1] / token_budget
        attn_fn, attn_bias_or_two_vector = None, None

        # Calculate the final RoPE cache.
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(x_BLC.device)
        rope_cache_list = []
        for i in range(len(visual_rope_cache)):
            if use_sr_prefix:
                rope_cache_list.append(visual_rope_cache[i])
            rope_cache_list.append(visual_rope_cache[i])
            if use_text:
                rope_cache_list.append(self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:lens[i]])
            rope_cache_list.append(self.rope2d_freqs_grid['freqs_text'][:,:,:,:,512:512+self.other_args.add_scale_token])
        rope_cache = torch.cat(rope_cache_list, dim=4) # (2, 1, 1, 1, seq_len, head_dim / 2)
        assert rope_cache.shape[4] == x_BLC.shape[1], f'{rope_cache.shape[4]} != {x_BLC.shape[1]}'
        rope_cache = rope_cache[:,0].permute(0, 1, 3, 2, 4) # (2, 1, 1, 1, seq_len, head_dim / 2) -> (2, 1, 1, seq_len, head_dim / 2) -> (2, 1, seq_len, 1, head_dim / 2)

        # calculate time or scale embeddings
        if self.other_args.use_ada_layer_norm:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                e = self.scale_or_time_embedding(sinusoidal_embedding_1d(self.scale_or_time_dim, scale_or_time_ids).float()) # [1, visual_seq_len,] -> [1, visual_seq_len, 256] -> [1, visual_seq_len, C]
                if e.shape[1] < x_BLC.shape[1]:
                    e = F.pad(e, (0,0,0,x_BLC.shape[1]-e.shape[1]), 'constant', 0.) # [1, visual_seq_len, C] -> [1, L, C]
                e0 = self.scale_or_time_projection(e).unflatten(2, (6, self.C)) # [1, L, C] -> [1, L, 6C] -> [1, L, 6, C]
                assert e.dtype == torch.float32 and e0.dtype == torch.float32
        else:
            e, e0 = None, None
        
        # [2. block loop]
        checkpointing_full_block = self.checkpointing == 'full-block' and self.training

        if sp_manager.sp_on():
            # [B, raw_L, C] --> [B, raw_L/sp_size, C]
            x_BLC = sp_split_sequence_by_dim(x_BLC, 1)

        if use_sr_condition or use_native_condition:
            super_scale_lengths = actual_scale_lengths
        cu_seqlens = torch.tensor([0]+super_scale_lengths, device=device).cumsum(-1).to(torch.int32)
        max_seqlen = max(super_scale_lengths)
        for i, chunk in enumerate(self.block_chunks): # this path
            x_BLC = chunk(x=x_BLC, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, e0=e0, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, checkpointing_full_block=checkpointing_full_block, rope2d_freqs_grid=rope_cache)
            layer_number = (i + 1) * self.num_blocks_in_a_chunk
            if use_sr_condition and str(layer_number) in self.sr_cross_attn:
                if sp_manager.sp_on():
                    x_BLC = sp_gather_sequence_by_dim(x_BLC, 1)
                x_BLC = self._apply_sr_cross_attention(
                    x_BLC,
                    lr_prefix,
                    sr_cross_contexts,
                    layer_number,
                    checkpointing=checkpointing_full_block,
                )
                if sp_manager.sp_on():
                    x_BLC = sp_split_sequence_by_dim(x_BLC, 1)
            if use_spatial_control and str(layer_number) in self.sr_spatial_controls:
                if sp_manager.sp_on():
                    x_BLC = sp_gather_sequence_by_dim(x_BLC, 1)
                x_BLC = apply_packed_spatial_control(
                    x_BLC, lr_prefix, sr_spatial_contexts,
                    self.sr_spatial_controls[str(layer_number)],
                    checkpointing=checkpointing_full_block,
                )
                if sp_manager.sp_on():
                    x_BLC = sp_split_sequence_by_dim(x_BLC, 1)
            if use_native_condition and str(layer_number) in self.sr_native_adapters:
                if sp_manager.sp_on():
                    x_BLC = sp_gather_sequence_by_dim(x_BLC, 1)
                x_BLC = self._apply_sr_native_adapters(
                    x_BLC, native_source, sr_native_contexts, layer_number,
                    checkpointing=checkpointing_full_block,
                )
                if sp_manager.sp_on():
                    x_BLC = sp_split_sequence_by_dim(x_BLC, 1)

        if sp_manager.sp_on():
            # [B, raw_L/sp_size, C] --> [B, raw_L, C]
            x_BLC = sp_gather_sequence_by_dim(x_BLC, 1)

        # [3. unpad the seqlen dim, and then get logits]
        logits_norm, loss_list, acc_list, predicted_logits = self.get_loss_acc(
            x_BLC, x_BLC_mask, e, sequece_packing_scales, gt_BL, other_info_by_scale,
            return_last_hidden_states, sr_v2_inference_precision=sr_v2_inference_precision,
            sr_calibrator_latent=(sr_condition_latent
                if hasattr(self, 'sr_round_calibrator') else None),
            **calibration_kwargs,
        )
        outputs = (logits_norm, loss_list, acc_list, valid_sequence_ratio)
        if return_sr_aux:
            if not (use_sr_condition or use_native_condition):
                raise ValueError("return_sr_aux is only valid for SR training")
            sr_aux = {
                'logits_by_scale': predicted_logits,
                'condition_latent': sr_condition_latent,
            }
            if torch.is_grad_enabled() and hasattr(self, 'sr_coarse_head'):
                sr_aux['condition_coarse_rgb'] = self.sr_coarse_head(sr_condition_latent)
            return outputs + (sr_aux,)
        return outputs

    def prepare_text_conditions(
        self,
        label_B_or_BLT: Tuple[torch.Tensor, ...],
        negative_label_B_or_BLT: Optional[Tuple[torch.Tensor, ...]],
        use_cfg: bool = False,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Prepare text conditions for inference."""
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        if use_cfg:
            kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
            kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
            cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:] + cu_seqlens_k[-1]), dim=0)
            max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
            lens = lens + lens_un
        kv_compact = self.text_proj(kv_compact).contiguous()
        return kv_compact, lens
    
    def embeds_codes2input(self, last_stage: torch.Tensor) -> torch.Tensor:
        """Embed discrete codes into continuous input representations."""
        last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, t*h*w] or [B, 4d, t*h*w]
        last_stage = torch.permute(last_stage, [0, 2, 1]) # [B, t*h*w, d] or [B, t*h*w, 4d]
        last_stage = self.word_embed(last_stage) # norm0_ve is Identity
        return last_stage
    
    def encode_first_frame(self, vae, image_path, tgt_h, tgt_w):
        from grn.utils_t2iv.infer import transform
        from grn.utils_t2iv.hbq_util_t2iv import raw_feature2bit_label
        raw_video = [cv2.imread(image_path)[:,:,::-1]] # bgr
        img_T3HW = [transform(Image.fromarray(frame).convert("RGB"), tgt_h, tgt_w) for frame in raw_video] # bgr to rgb
        img_T3HW = torch.stack(img_T3HW, 0) # [t,3,h,w]
        img_bcthw = img_T3HW.permute(1,0,2,3).unsqueeze(0) # [c,t,h,w] -> [b,c,t,h,w]
        raw_features, _, _ = vae.encode_for_raw_features(img_bcthw.to('cuda'), scale_schedule=None, slice=True)
        labels = raw_feature2bit_label(raw_features[0], hbq_round=self.other_args.hbq_round) # [B, hbq_round * d, t, h, w]
        return labels

    @torch.no_grad()
    def autoregressive_infer(
        self,
        vae: Optional[Any] = None,
        scale_schedule: Optional[List[Tuple[int, int, int]]] = None,
        label_B_or_BLT: Optional[List[Tuple[torch.Tensor, ...]]] = None,
        negative_label_B_or_BLT: Optional[List[Tuple[torch.Tensor, ...]]] = None,
        g_seed: Optional[int] = None,
        cfg_list: Optional[List[float]] = None,
        tau_list: Optional[List[float]] = None,
        gt_leak: int = 0,
        args: Optional[Any] = None,
        get_visual_rope_embeds: Optional[Any] = None,
        noise_list: Optional[List[torch.Tensor]] = None,
        first_frame_condition: bool = False,
        first_frame_path: Optional[str] = None,
        lr_condition: Optional[torch.Tensor] = None,
        sr_semantic_features: Optional[torch.Tensor] = None,
        sr_calibrator_kappa: Optional[float] = None,
        sr_quality_guidance: Optional[Any] = None,
        sr_negative_lr_condition: Optional[torch.Tensor] = None,
        sr_negative_semantic_features: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        """Autoregressive inference loop for the GRN model."""
        self._validate_sr_v2_semantic_condition(lr_condition, sr_semantic_features)
        if cfg_list is None: cfg_list = []
        if tau_list is None: tau_list = []
        calibration_enabled = hasattr(self, 'sr_round_calibrator')
        calibration_kwargs = {}
        if sr_calibrator_kappa is not None:
            if not calibration_enabled:
                raise ValueError('Runtime kappa requires an enabled SR calibrator')
            calibration_kwargs['kappa'] = resolve_calibrator_kappa(sr_calibrator_kappa)
        residual_enabled = hasattr(self, 'sr_latent_residual')
        if residual_enabled and (not calibration_enabled or lr_condition is None
                or self.sr_condition_mode != 'prefix_text' or first_frame_condition
                or getattr(args, 'refine_mode', None) != 'ar_discrete_GRN_bit'):
            raise ValueError('Latent residual requires calibrated prefix_text image SR with binary refinement')
        if calibration_enabled and (lr_condition is None
                or not cfg_list or any(float(cfg) != 1.0 for cfg in cfg_list)):
            raise ValueError('S3 inference requires LR and guidance=1; CFG is not validated')
        # Opt-in quality-contrastive guidance. It is separate from cfg_list, which
        # stays at 1: the zero-prefix unconditional branch was never trained.
        quality_guidance = sr_quality_guidance
        if quality_guidance is None:
            if sr_negative_lr_condition is not None or sr_negative_semantic_features is not None:
                raise ValueError('Negative LR inputs require sr_quality_guidance')
        else:
            from grn.models.sr_quality_guidance import (
                SRQualityGuidance, guided_logits, sample_bits_by_plane)
            if not isinstance(quality_guidance, SRQualityGuidance):
                raise ValueError('sr_quality_guidance must be an SRQualityGuidance')
            if (lr_condition is None or self.sr_condition_mode != 'prefix_text' or first_frame_condition
                    or getattr(args, 'refine_mode', None) != 'ar_discrete_GRN_bit'
                    or any(float(cfg) != 1.0 for cfg in cfg_list)):
                raise ValueError('Quality guidance requires prefix_text image SR, binary refinement and guidance=1')
            if quality_guidance.needs_negative_lr:
                if (sr_negative_lr_condition is None
                        or sr_negative_lr_condition.shape != lr_condition.shape):
                    raise ValueError('degraded_lr guidance requires a negative LR with the LR shape')
                self._validate_sr_v2_semantic_condition(
                    sr_negative_lr_condition, sr_negative_semantic_features)
            elif sr_negative_lr_condition is not None or sr_negative_semantic_features is not None:
                raise ValueError('This guidance setting does not use a negative LR')
            if quality_guidance.needs_negative_calibrator and (
                    not calibration_enabled or not hasattr(self, 'sr_negative_calibrator')):
                raise ValueError('calibrator guidance requires loaded positive and negative calibrators')
        native_lr_input = kwargs.pop('native_lr_input', None)
        use_native_condition = native_lr_input is not None
        if use_native_condition:
            if not self.sr_condition or self.sr_condition_mode != 'native_adapter':
                raise ValueError('native source requires a native SR model')
            if any(float(cfg) != 1.0 for cfg in cfg_list):
                raise ValueError('native SR Stage1 requires guidance=1')
        if self.sr_condition_mode == 'native_adapter' and lr_condition is not None and not use_native_condition:
            raise ValueError('Encode native LR once before refinement and pass native_lr_input')
        
        from grn.schedules.global_refine import shift_pt
        
        assert len(cfg_list) >= len(scale_schedule), "Not enough CFG values for scales"
        assert len(tau_list) >= len(scale_schedule), "Not enough tau values for scales"

        ret, idx_Bl_list = [], []  # current length, list of reconstructed images
        for b in self.unregistered_blocks: b.attn.kv_caching(True)
        total_steps = args.max_infer_steps
        pbar = tqdm.tqdm(total=total_steps)
        block_chunks = self.block_chunks
        use_cfg = True
        cfg_interval = float(args.cfg_type.split('_')[-1])
        model_device = self.word_embed.weight.device
        full_pt, ph, pw = scale_schedule[0]
        if first_frame_condition:
            pt = full_pt - 1
            visual_rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, (pt, ph, pw), model_device, args.mapped_h_div_w_template, t_offset=1)
        else:
            pt = full_pt
            visual_rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, (pt, ph, pw), model_device, args.mapped_h_div_w_template, t_offset=0)

        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(model_device)
        use_sr_condition = lr_condition is not None and uses_continuous_lr(self.sr_condition_mode)
        use_text = not use_sr_condition or uses_empty_text(self.sr_condition_mode)
        use_spatial_control = use_sr_condition and self.sr_condition_mode == 'spatial_control'
        if use_sr_condition or use_native_condition:
            if first_frame_condition:
                raise ValueError("SR and first-frame conditioning cannot be enabled together")
            if (pt, ph, pw) != (1, self.sr_latent_size, self.sr_latent_size):
                raise ValueError(
                    f"SR inference requires latent scale "
                    f"{(1, self.sr_latent_size, self.sr_latent_size)}, got {(pt, ph, pw)}"
                )
        if use_native_condition:
            if native_lr_input.shape[:2] != (1, ph * pw):
                raise ValueError('native SR inference requires one source with the target spatial grid')
            native_source = self.word_embed(native_lr_input.float()).repeat(2, 1, 1)
        if use_sr_condition:
            if lr_condition.shape[0] != 1:
                raise ValueError('SR inference requires one LR image')
            if calibration_enabled:
                sr_prefix_cond, sr_calibrator_latent = self.encode_lr_condition(
                    lr_condition, target_hw=(ph, pw), return_latent=True,
                    sr_semantic_features=sr_semantic_features)
            else:
                sr_prefix_cond = self.encode_lr_condition(
                    lr_condition, target_hw=(ph, pw), sr_semantic_features=sr_semantic_features)
            sr_prefix_uncond = torch.zeros_like(sr_prefix_cond)
            sr_negative_latent = None
            if quality_guidance is not None and quality_guidance.needs_negative_lr:
                # The second packed branch is computed at guidance=1 anyway and
                # then discarded. Give it a further-degraded LR instead of zeros.
                sr_prefix_uncond, sr_negative_latent = self.encode_lr_condition(
                    sr_negative_lr_condition, target_hw=(ph, pw), return_latent=True,
                    sr_semantic_features=sr_negative_semantic_features)
                sr_prefix_uncond = sr_prefix_uncond.to(dtype=sr_prefix_cond.dtype)
            lr_memory = torch.cat((sr_prefix_cond, sr_prefix_uncond), dim=0)
            lens = [0, 0]
            device = sr_prefix_cond.device
            infer_device, infer_dtype = device, sr_prefix_cond.dtype
        if use_text:
            prefix_tokens, lens = self.prepare_text_conditions(
                label_B_or_BLT[0], negative_label_B_or_BLT, use_cfg
            )
            device = prefix_tokens.device
            infer_device, infer_dtype = prefix_tokens.device, prefix_tokens.dtype
            prefix_tokens = torch.split(prefix_tokens, lens, dim=0)
            rope_cache_text_cond = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:lens[0]]
            rope_cache_text_uncond = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,:lens[1]]

        infer_device = torch.device(infer_device)
        if self.rng.device != infer_device:
            self.rng = torch.Generator(device=infer_device)
        rng = self.rng
        if g_seed is not None:
            rng.manual_seed(int(g_seed))

        if args.refine_mode in ['ar_discrete_GRN_bit']:
            classes = 2
            labels_shape = (1,args.detail_scale_dim*args.hbq_round,pt,ph,pw)
        elif args.refine_mode in ['ar_discrete_GRN_index']:
            classes = 2**args.hbq_round
            labels_shape = (1,args.detail_scale_dim,pt,ph,pw)

        mul_pt_ph_pw = pt * ph * pw
        repeat_idx = -1
        scale_token_rope_cache = self.rope2d_freqs_grid['freqs_text'][:,:,:,:,512:512+args.add_scale_token]
        if noise_list is not None:
            absolute_gt_labels = noise_list[0].to(infer_device).permute(0,2,3,4,1) # [B,d,t,h,w] -> [B,t,h,w,d]
        assert len(scale_schedule) == 1
        if use_sr_condition:
            layouts = [condition_layout(self.sr_condition_mode, mul_pt_ph_pw, n) for n in lens]
            target_token_offset = layouts[0].target_offset
            tmp_seqlens = [layout.total_tokens for layout in layouts]
            sr_cross_prefix = lr_memory
            sr_cross_contexts = [
                (0, layouts[0].target_offset, mul_pt_ph_pw, 0),
                (tmp_seqlens[0], layouts[1].target_offset, mul_pt_ph_pw, 1),
            ]
            sr_spatial_contexts = [
                (layouts[0].target_offset, mul_pt_ph_pw, 0, (ph, pw)),
                (tmp_seqlens[0] + layouts[1].target_offset, mul_pt_ph_pw, 1, (ph, pw)),
            ]
        elif first_frame_condition:
            target_token_offset = 0
            first_frame_labels = self.encode_first_frame(vae, first_frame_path, ph*16, pw*16)[:,:,:1] # [B,d,1,h,w]
            first_frame_tokens_cond = self.embeds_codes2input(multiclass_labels2onehot_input(first_frame_labels, classes))
            fist_frame_rope_cache = get_visual_rope_embeds(self.rope2d_freqs_grid, (1, ph, pw), device, args.mapped_h_div_w_template, t_offset=0)
            visual_rope_cache = torch.cat((visual_rope_cache, fist_frame_rope_cache), dim=4)
            tmp_seqlens = [mul_pt_ph_pw + ph * pw + lens[0] + args.add_scale_token, mul_pt_ph_pw + lens[1] + ph * pw + args.add_scale_token]
        else:
            target_token_offset = 0
            tmp_seqlens = [mul_pt_ph_pw+lens[0]+args.add_scale_token, mul_pt_ph_pw+lens[1]+args.add_scale_token]
        if use_native_condition:
            sr_native_contexts = [(0, mul_pt_ph_pw, 0, (ph, pw)),
                                  (tmp_seqlens[0], mul_pt_ph_pw, 1, (ph, pw))]

        if use_sr_condition:
            rope_parts = []
            for layout in layouts:
                if layout.prefix_tokens:
                    rope_parts.append(visual_rope_cache)
                rope_parts.append(visual_rope_cache)
                if layout.text_tokens:
                    rope_parts.append(self.rope2d_freqs_grid['freqs_text'][..., :layout.text_tokens, :])
                rope_parts.append(scale_token_rope_cache)
            rope_cache = torch.cat(rope_parts, dim=4)
        else:
            rope_cache = torch.cat([visual_rope_cache, rope_cache_text_cond, scale_token_rope_cache, visual_rope_cache, rope_cache_text_uncond, scale_token_rope_cache], dim=4) # (2, 1, 1, 1, seq_len, dim / 2)
        rope_cache = rope_cache[:,0].permute(0, 1, 3, 2, 4) # (2, 1, 1, 1, seq_len, dim / 2) -> (2, 1, 1, seq_len, dim / 2) -> (2, 1, seq_len, 1, dim / 2)

        cu_seqlens = torch.tensor([0]+tmp_seqlens, device=device).cumsum(-1).to(torch.int32)
        max_seqlen = max(tmp_seqlens)

        pure_rand_labels = torch.randint(
            low=0,
            high=classes,
            size=labels_shape,
            device=infer_device,
            dtype=infer_dtype,
            generator=rng,
        )
        mixed_xt = pure_rand_labels
        next_pt = 0.
        attn_mask = build_attn_mask(tmp_seqlens, device) if args.use_slow_attn else None
        for cur_inner_round_si in range(args.max_infer_steps):
            cur_pt = next_pt
            is_last_step = np.abs(cur_pt - 1) < 0.02
            if cur_inner_round_si == 0:
                self.entrophy_statistics.append([])
            repeat_idx += 1 # index scale tokens, very important
            cfg = cfg_list[0] if cur_pt >= cfg_interval else 1.0
            last_stage = self.embeds_codes2input(multiclass_labels2onehot_input(mixed_xt, classes))
            pt_tokens = self.pt_embedder(torch.tensor([cur_pt], device=device)).unsqueeze(0)
            if use_sr_condition:
                sequences = []
                for index, layout in enumerate(layouts):
                    parts = ([lr_memory[index:index+1]] if layout.prefix_tokens else []) + [last_stage]
                    if layout.text_tokens:
                        parts.append(prefix_tokens[index].unsqueeze(0))
                    parts.append(pt_tokens)
                    sequences.append(torch.cat(parts, dim=1))
                last_stage_cond, last_stage_uncond = sequences
            elif first_frame_condition:
                last_stage_cond = torch.cat((last_stage, first_frame_tokens_cond, prefix_tokens[0].unsqueeze(0), pt_tokens), dim=1)
                last_stage_uncond = torch.cat((last_stage, first_frame_tokens_cond, prefix_tokens[1].unsqueeze(0), pt_tokens), dim=1)
            else:
                last_stage_cond = torch.cat((last_stage, prefix_tokens[0].unsqueeze(0), pt_tokens), dim=1)
                last_stage_uncond = torch.cat((last_stage, prefix_tokens[1].unsqueeze(0), pt_tokens), dim=1)
            last_stage = torch.cat([last_stage_cond, last_stage_uncond], dim=1)
            if use_spatial_control:
                lr_memory = lr_memory.to(dtype=last_stage.dtype)

            e, e0 = None, None
            last_diffusion_step = False
            for block_idx, b in enumerate(block_chunks):
                last_stage = b(x=last_stage, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, e0=e0, attn_bias_or_two_vector=attn_mask, attn_fn=None, rope2d_freqs_grid=rope_cache, last_diffusion_step=last_diffusion_step)
                layer_number = (block_idx + 1) * self.num_blocks_in_a_chunk
                if use_sr_condition and str(layer_number) in self.sr_cross_attn:
                    last_stage = self._apply_sr_cross_attention(
                        last_stage,
                        sr_cross_prefix,
                        sr_cross_contexts,
                        layer_number,
                        checkpointing=False,
                    )
                if use_spatial_control and str(layer_number) in self.sr_spatial_controls:
                    last_stage = apply_packed_spatial_control(
                        last_stage, lr_memory, sr_spatial_contexts,
                        self.sr_spatial_controls[str(layer_number)],
                    )
                if use_native_condition and str(layer_number) in self.sr_native_adapters:
                    last_stage = self._apply_sr_native_adapters(
                        last_stage, native_source, sr_native_contexts, layer_number,
                    )
            logits = self.get_logits_during_infer(last_stage, e=e)
            tmp_bs, tmp_seq_len = logits.shape[:2]
            logits = logits.reshape(tmp_bs, tmp_seq_len, -1, args.detail_num_lvl) # [B,thw+...,d,2]
            pred_cond_logits = logits[
                :, target_token_offset:target_token_offset + mul_pt_ph_pw
            ]
            uncond_target_start = tmp_seqlens[0] + target_token_offset
            pred_uncond_logits = logits[
                :, uncond_target_start:uncond_target_start + mul_pt_ph_pw
            ]
            guide_this_round = quality_guidance is not None and quality_guidance.active_at(cur_pt)
            head_cond_logits = pred_cond_logits
            if calibration_enabled:
                pred_cond_logits = self._calibrate_sr_hr_logits(
                    pred_cond_logits.permute(0, 3, 1, 2),
                    last_stage[:, target_token_offset:target_token_offset + mul_pt_ph_pw],
                    sr_calibrator_latent,
                    last_stage.new_tensor([cur_pt], dtype=torch.float32),
                    **calibration_kwargs,
                ).permute(0, 2, 3, 1)
            if guide_this_round:
                retention = last_stage.new_tensor([cur_pt], dtype=torch.float32)
                if quality_guidance.needs_negative_lr:
                    negative_logits = pred_uncond_logits
                    if calibration_enabled:
                        # Same calibrator and kappa on both branches, so their
                        # difference isolates the effect of the LR degradation.
                        negative_logits = self._calibrate_sr_hr_logits(
                            negative_logits.permute(0, 3, 1, 2),
                            last_stage[:, uncond_target_start:uncond_target_start + mul_pt_ph_pw],
                            sr_negative_latent, retention, **calibration_kwargs,
                        ).permute(0, 2, 3, 1)
                else:
                    # C- reads the same hidden states and head logits as C.
                    negative_logits = self.sr_negative_calibrator.calibrate_logits(
                        head_cond_logits.permute(0, 3, 1, 2),
                        last_stage[:, target_token_offset:target_token_offset + mul_pt_ph_pw],
                        sr_calibrator_latent, retention, **calibration_kwargs,
                    ).permute(0, 2, 3, 1)
                pred_cond_logits = guided_logits(pred_cond_logits, negative_logits, quality_guidance)
            pred_cond_probs = pred_cond_logits.softmax(-1) # [B,thw,d,2]
            categories = pred_cond_logits.shape[-1]
            entrophy = (-pred_cond_probs * torch.log2(pred_cond_probs)).sum(-1).mean().item() / np.log2(categories)

            pt_unshift = (cur_inner_round_si + 1) / (args.complexity_aware_Tmax - 1)
            pt_shift = shift_pt(min(1., pt_unshift), args.snr_shift)
            next_pt = 1 - np.cos(np.pi/2*pt_shift)
            next_pt = next_pt * 0.95

            pred_cond_labels = torch.argmax(pred_cond_probs, dim=-1) # [B,thw,d]
            pred_cond_labels = bld_to_bthwd(pred_cond_labels, pt, ph, pw)
            if cfg != 1:
                pred_cfg_logits = pred_uncond_logits + cfg * (pred_cond_logits - pred_uncond_logits)
            else:
                pred_cfg_logits = pred_cond_logits
            pred_cfg_logits = pred_cfg_logits.mul(1/tau_list[0]) # [B,thw,d,2]
            pred_cfg_probs = pred_cfg_logits.softmax(dim=-1) # [B,thw,d,2]
            pred_cfg_labels = torch.argmax(pred_cfg_probs, dim=-1) # [B,thw,d]
            pred_cfg_labels = bld_to_bthwd(pred_cfg_labels, pt, ph, pw) # [B,t,h,w,d]
            plane_sample_probs = None
            if quality_guidance is not None and quality_guidance.plane_temperatures is not None:
                # Per-plane greedy/stochastic decisions replace the global sampler.
                pred_sample_labels, plane_sample_probs = sample_bits_by_plane(
                    pred_cfg_logits, quality_guidance.plane_temperatures, rng)
            elif int(getattr(args, 'sr_top_k', 0)) == 1:
                pred_sample_labels = pred_cfg_probs.argmax(dim=-1)
            else:
                pred_sample_labels = torch.multinomial(
                    pred_cfg_probs.view(-1, args.detail_num_lvl),
                    num_samples=1,
                    replacement=True,
                    generator=rng,
                ).view(tmp_bs, mul_pt_ph_pw, -1)
            pred_sample_probs = torch.gather(pred_cfg_probs, dim=3, index=pred_sample_labels.unsqueeze(-1)).squeeze(-1) # [B,thw,d]
            if plane_sample_probs is not None:
                pred_sample_probs = plane_sample_probs.to(dtype=pred_sample_probs.dtype)
            pred_sample_probs = bld_to_bthwd(pred_sample_probs, pt, ph, pw) # [B,t,h,w,d]
            pred_sample_labels = bld_to_bthwd(pred_sample_labels, pt, ph, pw) # [B,t,h,w,d]

            assume_flip_ratio = (1 - cur_pt) / args.detail_num_lvl * 100. # different ratio between prediciton and input
            pred_zero_ratio = (pred_cond_labels == 0).sum() / pred_cond_labels.numel() * 100.
            pred_one_ratio = (pred_cond_labels == 1).sum() / pred_cond_labels.numel() * 100.
            mixed_xt_Bthwd_01 = mixed_xt.clone().permute(0,2,3,4,1)
            mixed_xt_Bthwd_01[mixed_xt_Bthwd_01<0] = 0
            pred_cond_flip_ratio = (pred_cond_labels != mixed_xt_Bthwd_01).sum() / pred_cond_labels.numel() * 100.
            pred_cfg_flip_ratio = (pred_cfg_labels != mixed_xt_Bthwd_01).sum() / pred_cfg_labels.numel() * 100.
            pred_sample_flip_ratio = (pred_sample_labels != mixed_xt_Bthwd_01).sum() / pred_sample_labels.numel() * 100.
            self.entrophy_statistics[-1].append({
                'cur_inner_round_si': cur_inner_round_si,
                'cur_pt': cur_pt,
                # 'cur_tau': cur_tau,
                # 'cur_cfg': cur_cfg,
                'entrophy': entrophy,
                'assume_flip_ratio': assume_flip_ratio,
                'pred_cond_flip_ratio': pred_cond_flip_ratio.item(),
                'pred_cfg_flip_ratio': pred_cfg_flip_ratio.item(),
                'pred_sample_flip_ratio': pred_sample_flip_ratio.item(),
                'pred_zero_ratio': pred_zero_ratio.item(),
                'pred_one_ratio': pred_one_ratio.item(),
                'meta': args.meta,
            })
            print(f'{repeat_idx=} {cur_inner_round_si=} {cur_pt=:.3f} {pred_sample_labels.shape=}')
            print(f'{assume_flip_ratio=:.2f}% {pred_cond_flip_ratio=:.2f}% {pred_cfg_flip_ratio=:.2f}% {pred_sample_flip_ratio=:.2f}%')
            if repeat_idx < gt_leak:
                gt_labels = absolute_gt_labels
                gt_flip_ratio = (gt_labels != mixed_xt_Bthwd_01).sum() / gt_labels.numel() * 100.
                gt_flip_ratio = gt_flip_ratio.item()
                pred_cond_acc = (gt_labels==pred_cond_labels).to(float).mean().item()
                pred_cfg_acc = (gt_labels==pred_cfg_labels).to(float).mean().item()
                pred_sample_acc = (gt_labels==pred_sample_labels).to(float).mean().item()
                print(f'{repeat_idx=} {entrophy=:.4f} {pred_cond_acc=:.4f} {pred_cfg_acc=:.4f} {pred_sample_acc=:.4f}')
                self.entrophy_statistics[-1][-1].update({
                    'gt_flip_ratio': gt_flip_ratio,
                    'pred_cond_acc': pred_cond_acc,
                    'pred_cfg_acc': pred_cfg_acc,
                    'pred_sample_acc': pred_sample_acc,
                })
                pred_sample_labels = gt_labels
            
            pred_sample_labels = pred_sample_labels.permute(0,4,1,2,3) # [B,t,h,w,d] -> [B,d,t,h,w]
            pred_sample_probs = pred_sample_probs.permute(0,4,1,2,3) # [B,t,h,w,d] -> [B,d,t,h,w]
            use_predict_mask = torch.rand(
                pred_sample_labels.shape, device=device, generator=rng
            ) < next_pt
            mixed_xt = torch.where(use_predict_mask, pred_sample_labels, pure_rand_labels)
            next_pt = use_predict_mask.float().mean().item()
            pbar.update(1)
            if is_last_step: break

        if first_frame_condition:
            pred_sample_labels = torch.cat((first_frame_labels, pred_sample_labels), dim=2)
         
        if args.refine_mode == 'ar_discrete_GRN_ind':
            from grn.utils_t2iv.hbq_util_t2iv import index_label2quant_features
            approx_signal = index_label2quant_features(pred_sample_labels, hbq_round=args.hbq_round)
        elif args.refine_mode == 'ar_discrete_GRN_bit':
            from grn.utils_t2iv.hbq_util_t2iv import bit_label2raw_feature
            approx_signal = bit_label2raw_feature(pred_sample_labels, hbq_round=args.hbq_round) # [B, hbq_round_mul_d, t, h, w] -> [B,d,t,h,w]
        for b in self.unregistered_blocks: b.attn.kv_caching(False)
        if residual_enabled:
            # Apply only once, after all bit refinement. The sampling trajectory
            # and random-mask draws above do not depend on this residual head.
            residual_kappa = resolve_calibrator_kappa(
                sr_calibrator_kappa, self.sr_round_calibrator.max_log_odds)
            approx_signal = self.sr_latent_residual(
                approx_signal, sr_calibrator_latent, residual_kappa)
        img = self.summed_codes2images(vae, approx_signal)
        return ret, idx_Bl_list, img

    def summed_codes2images(self, vae: Any, summed_codes: torch.Tensor) -> torch.Tensor:
        """Decode summed codes into images using the VAE."""
        t1 = time.time()
        img = vae.decode(summed_codes, slice=True)
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)
        img = img.permute(0, 2, 3, 4, 1) # [bs, 3, t, h, w] -> [bs, t, h, w, 3]
        img = img.mul_(255).to(torch.uint8).flip(dims=(4,))
        print(f"Decode takes {time.time() - t1:.1f}s")
        return img # bgr order

    @for_visualize
    def vis_key_params(self, ep: int) -> None:
        return
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = False, assign: bool = False) -> Any:       
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
    
    def special_init(self, **kwargs: Any) -> None:
        """Apply special initialization to specific layers."""
        std = math.sqrt(1 / self.C / 3)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
        residual_scale = 1 / math.sqrt(2 * self.depth)
        for block in self.unregistered_blocks:
            block.attn.o_proj.weight.data.mul_(residual_scale)
            block.mlp.down_proj.weight.data.mul_(residual_scale)
        for adapter in getattr(self, 'sr_native_adapters', {}).values():
            adapter.reset_output()
        for control in getattr(self, 'sr_spatial_controls', {}).values():
            control.reset_output()
        if hasattr(self, 'sr_semantic_fusion'):
            self.sr_semantic_fusion.reset_output()
        if hasattr(self, 'sr_round_calibrator'):
            self.sr_round_calibrator.reset_output()
    
    def extra_repr(self) -> str:
        return f'drop_path_rate={self.drop_path_rate}'
    
    def get_layer_id_and_scale_exp(self, para_name: str) -> Any:
        raise NotImplementedError

TIMM_KEYS = {'img_size', 'pretrained', 'pretrained_cfg', 'pretrained_cfg_overlay', 'global_pool'}

@register_model
def GRN0b(depth: int = 4, block_chunks: int = 2, embed_dim: int = 512, num_heads: int = 4, num_key_value_heads: int = 4, drop_path_rate: float = 0.0, **kwargs: Any) -> GRN: 
    return GRN(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.55,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def GRN2b(depth: int = 28, block_chunks: int = 7, embed_dim: int = 2304, num_heads: int = 18, num_key_value_heads: int = 18, drop_path_rate: float = 0.0, **kwargs: Any) -> GRN: 
    return GRN(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=3.55,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def GRN8b(depth: int = 36, block_chunks: int = 6, embed_dim: int = 4096, num_heads: int = 32, num_key_value_heads: int = 8, mlp_ratio: float = 4, drop_path_rate: float = 0.0, **kwargs: Any) -> GRN: 
    return GRN(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )

@register_model
def GRN8b(depth: int = 36, block_chunks: int = 6, embed_dim: int = 4096, num_heads: int = 32, num_key_value_heads: int = 32, mlp_ratio: float = 3, drop_path_rate: float = 0.0, **kwargs: Any) -> GRN: 
    return GRN(
        arch='qwen',
        qwen_qkvo_bias=False,
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )
