"""Native low-resolution spatial residual adapter."""

import math

from torch import Tensor, nn
from torch.nn import functional as F


class NativeLRSpatialAdapter(nn.Module):
    """Add a learned spatial residual derived from low-resolution tokens."""

    def __init__(self, embed_dim: int, bottleneck_dim: int = 256):
        super().__init__()
        if not isinstance(embed_dim, int) or not isinstance(bottleneck_dim, int):
            raise ValueError("embed_dim and bottleneck_dim must be positive integers")
        if embed_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("embed_dim and bottleneck_dim must be positive integers")

        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim)
        self.in_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.spatial_conv = nn.Conv2d(
            bottleneck_dim,
            bottleneck_dim,
            kernel_size=3,
            padding=1,
            groups=bottleneck_dim,
        )
        self.out_proj = nn.Linear(bottleneck_dim, embed_dim)

        self._reset_input_path()
        self.reset_output()

    def _reset_input_path(self) -> None:
        """Initialize layers whose framework resets may be globally disabled."""
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        nn.init.kaiming_uniform_(self.in_proj.weight, a=math.sqrt(5))
        if self.in_proj.bias is not None:
            bound = 1 / math.sqrt(self.in_proj.in_features)
            nn.init.uniform_(self.in_proj.bias, -bound, bound)

    def reset_output(self) -> None:
        """Restore exact identity behavior without changing the feature path."""
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        hr_tokens: Tensor,
        lr_tokens: Tensor,
        grid_hw: tuple[int, int],
    ) -> Tensor:
        if hr_tokens.ndim != 3 or lr_tokens.ndim != 3:
            raise ValueError("hr_tokens and lr_tokens must be rank-3 tensors")
        if hr_tokens.shape != lr_tokens.shape:
            raise ValueError("hr_tokens and lr_tokens must have the same shape")
        batch, token_count, channels = hr_tokens.shape
        if channels != self.embed_dim:
            raise ValueError(
                f"token channel dimension must equal embed_dim={self.embed_dim}, got {channels}"
            )
        if (
            not isinstance(grid_hw, tuple)
            or len(grid_hw) != 2
            or any(not isinstance(size, int) or size <= 0 for size in grid_hw)
        ):
            raise ValueError("grid_hw must be a tuple of two positive integers")
        height, width = grid_hw
        if height * width != token_count:
            raise ValueError(
                f"grid H*W must equal token count N={token_count}, got {height}*{width}"
            )

        features = F.silu(self.in_proj(self.norm(lr_tokens)))
        features = features.reshape(batch, height, width, -1).permute(0, 3, 1, 2)
        features = self.spatial_conv(features)
        features = features.permute(0, 2, 3, 1).reshape(batch, token_count, -1)
        spatial_features = F.silu(features)
        delta = self.out_proj(spatial_features)
        return (hr_tokens.float() + delta.float()).to(hr_tokens.dtype)
