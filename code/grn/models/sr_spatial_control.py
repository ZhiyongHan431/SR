"""Local HR-query spatial control for continuous LR feature memory."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from grn.models.basic import FastRMSNorm


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_grid(grid_hw: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(grid_hw, tuple)
        or len(grid_hw) != 2
        or any(not _positive_integer(size) for size in grid_hw)
    ):
        raise ValueError('grid_hw must be a tuple of two positive integers')
    return grid_hw


class HRQuerySpatialControl(nn.Module):
    """Add local cross-attention with HR queries and aligned LR keys/values."""

    def __init__(
        self,
        embed_dim: int,
        inner_dim: int = 512,
        num_heads: int = 8,
        window_size: int = 7,
    ) -> None:
        super().__init__()
        if not all(_positive_integer(value) for value in (embed_dim, inner_dim, num_heads)):
            raise ValueError('embed_dim, inner_dim, and num_heads must be positive integers')
        if not _positive_integer(window_size) or window_size % 2 == 0:
            raise ValueError('window_size must be a positive odd integer')
        if inner_dim % num_heads:
            raise ValueError(
                f'inner_dim must be divisible by num_heads, got {inner_dim} and {num_heads}'
            )

        self.embed_dim = embed_dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        self.window_size = window_size
        self.hr_norm = FastRMSNorm(embed_dim)
        self.lr_norm = FastRMSNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, embed_dim, bias=False)

        self._reset_attention_path()
        self.reset_output()

    def _reset_attention_path(self) -> None:
        """Initialize inputs explicitly when framework reset hooks are disabled."""
        nn.init.ones_(self.hr_norm.weight)
        nn.init.ones_(self.lr_norm.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)

    def reset_output(self) -> None:
        """Restore the exact-identity output without resetting the attention path."""
        nn.init.zeros_(self.out_proj.weight)

    def forward(
        self,
        hr_tokens: Tensor,
        lr_tokens: Tensor,
        grid_hw: tuple[int, int],
    ) -> Tensor:
        if hr_tokens.ndim != 3 or lr_tokens.ndim != 3:
            raise ValueError('hr_tokens and lr_tokens must be rank-3 tensors')
        if hr_tokens.shape[0] != lr_tokens.shape[0]:
            raise ValueError('hr_tokens and lr_tokens must have the same batch size')
        if hr_tokens.shape[2] != self.embed_dim or lr_tokens.shape[2] != self.embed_dim:
            raise ValueError(
                f'token channel dimensions must equal embed_dim={self.embed_dim}, got '
                f'HR={hr_tokens.shape[2]} and LR={lr_tokens.shape[2]}'
            )
        if hr_tokens.device != lr_tokens.device:
            raise ValueError('hr_tokens and lr_tokens must use the same device')
        if hr_tokens.dtype != lr_tokens.dtype:
            raise ValueError('hr_tokens and lr_tokens must use the same dtype')

        height, width = _validate_grid(grid_hw)
        count = height * width
        if hr_tokens.shape[1] != count or lr_tokens.shape[1] != count:
            raise ValueError(
                f'grid H*W must equal both token counts, got {height}*{width}, '
                f'HR={hr_tokens.shape[1]}, and LR={lr_tokens.shape[1]}'
            )

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(height, device=hr_tokens.device),
                torch.arange(width, device=hr_tokens.device),
                indexing='ij',
            ),
            dim=-1,
        ).flatten(0, 1)
        allowed = (
            (coords[:, None] - coords[None, :]).abs().amax(-1)
            <= self.window_size // 2
        )
        batch = hr_tokens.shape[0]
        q = self.q_proj(self.hr_norm(hr_tokens))
        k = self.k_proj(self.lr_norm(lr_tokens))
        v = self.v_proj(self.lr_norm(lr_tokens))
        q = q.view(batch, count, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, count, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, count, self.num_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed[None, None],
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, count, self.inner_dim)
        delta = self.out_proj(attended)
        return (hr_tokens.float() + delta.float()).to(hr_tokens.dtype)


def apply_packed_spatial_control(
    hidden_states: Tensor,
    lr_memory: Tensor,
    contexts: list[tuple[int, int, int, tuple[int, int]]],
    module: HRQuerySpatialControl,
    checkpointing: bool = False,
) -> Tensor:
    """Apply independent LR control to absolute HR spans in a packed sequence."""
    if hidden_states.ndim != 3 or lr_memory.ndim != 3:
        raise ValueError('hidden_states and lr_memory must be rank-3 tensors')
    if hidden_states.shape[0] != 1:
        raise ValueError('packed hidden_states must have batch size 1')
    if hidden_states.shape[2] != lr_memory.shape[2]:
        raise ValueError('hidden_states and lr_memory must have the same channel count')
    if hidden_states.device != lr_memory.device:
        raise ValueError('hidden_states and lr_memory must use the same device')
    if hidden_states.dtype != lr_memory.dtype:
        raise ValueError('hidden_states and lr_memory must use the same dtype')
    if not isinstance(contexts, list):
        raise ValueError('contexts must be a list')

    pieces = []
    cursor = 0
    total_tokens = hidden_states.shape[1]
    sample_count = lr_memory.shape[0]
    for context in contexts:
        if not isinstance(context, tuple) or len(context) != 4:
            raise ValueError('each context must be a four-item tuple')
        start, length, sample_index, grid_hw = context
        if not _positive_integer(length):
            raise ValueError('context HR length must be a positive integer')
        if not isinstance(start, int) or isinstance(start, bool):
            raise ValueError('context HR start must be a nonnegative integer')
        if start < 0 or start + length > total_tokens:
            raise ValueError('context HR span is out of range')
        if start < cursor:
            raise ValueError('context HR spans must be sorted and must not overlap')
        if (
            not isinstance(sample_index, int)
            or isinstance(sample_index, bool)
            or not 0 <= sample_index < sample_count
        ):
            raise ValueError('context LR sample index is out of range')

        end = start + length
        pieces.append(hidden_states[:, cursor:start])
        target = hidden_states[:, start:end]
        memory = lr_memory[sample_index:sample_index + 1]
        if checkpointing:
            target = checkpoint(
                module,
                target,
                memory,
                grid_hw,
                use_reentrant=False,
            )
        else:
            target = module(target, memory, grid_hw)
        pieces.append(target)
        cursor = end

    pieces.append(hidden_states[:, cursor:])
    return torch.cat(pieces, dim=1)
