"""Opt-in final-latent correction; the original GRN sampling loop is unchanged."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def _kappa_tensor(kappa, reference):
    if isinstance(kappa, bool):
        raise ValueError('kappa must be a finite nonnegative scalar')
    if torch.is_tensor(kappa):
        if kappa.ndim != 0 or kappa.dtype == torch.bool or kappa.is_complex():
            raise ValueError('kappa must be a finite nonnegative scalar')
        value = kappa.to(device=reference.device, dtype=torch.float32)
    else:
        if not isinstance(kappa, (int, float)):
            raise ValueError('kappa must be a finite nonnegative scalar')
        value = reference.new_tensor(kappa, dtype=torch.float32)
    if not bool(torch.isfinite(value)) or bool(value < 0):
        raise ValueError('kappa must be finite, nonnegative and representable in FP32')
    return value


class _ResidualBlock(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.kappa_proj = nn.Linear(channels, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, feature, embedding):
        value = self.conv1(F.silu(self.norm1(feature)))
        value = value + self.kappa_proj(embedding)[None, :, None, None]
        return feature + self.conv2(F.silu(self.norm2(value)))


class LatentResidualCorrector(nn.Module):
    """Correct the last HBQ latent once using LR structure and a scalar kappa.

    Inputs are z[B,64,H,W] or z[B,64,1,H,W], LR[B,64,H,W]. Computation uses
    FP32 parameters outside autocast; the result keeps z's shape and dtype.
    The zero output projection starts as identity but retains its gradient.
    """
    def __init__(self, *, blocks=6, channels=64, groups=8, kappa_dim=64):
        super().__init__()
        if type(blocks) is not int or not 1 <= blocks <= 32:
            raise ValueError('blocks must be an integer in [1,32]')
        if (channels, groups, kappa_dim) != (64, 8, 64) or any(
                type(value) is not int for value in (channels, groups, kappa_dim)):
            raise ValueError('LRC requires 64 channels, 8 norm groups and 64 kappa dimensions')
        self.blocks_count, self.channels = blocks, channels
        self.groups, self.kappa_dim = groups, kappa_dim
        self.input_proj = nn.Conv2d(2 * channels, channels, 3, padding=1)
        self.kappa_proj = nn.Sequential(nn.Linear(kappa_dim, channels), nn.SiLU(),
                                        nn.Linear(channels, channels))
        self.blocks = nn.ModuleList(_ResidualBlock(channels, groups) for _ in range(blocks))
        self.output_proj = nn.Conv2d(channels, channels, 3, padding=1)
        self.register_buffer('kappa_frequencies', torch.exp(
            -math.log(10000.) * torch.arange(kappa_dim // 2, dtype=torch.float32) / (kappa_dim // 2)))
        # GRN factories disable standard reset_parameters globally. Initialize
        # every newly created parameter explicitly, including normalization.
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    nn.init.uniform_(module.bias, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def architecture(self):
        return dict(blocks=self.blocks_count, channels=self.channels,
                    groups=self.groups, kappa_dim=self.kappa_dim)

    def residual(self, z, lr_features, kappa=2.0):
        for name, value in (('latent', z), ('LR features', lr_features)):
            if not torch.is_tensor(value) or not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f'{name} must be a finite floating tensor')
        temporal = z.ndim == 5
        if temporal and z.shape[2] != 1:
            raise ValueError('LRC only supports a single image temporal dimension')
        value = z.squeeze(2) if temporal else z
        if (value.ndim != 4 or value.shape[0] < 1 or value.shape[1] != self.channels
                or min(value.shape[-2:]) < 1 or lr_features.shape != value.shape):
            raise ValueError('Expected z[B,64,(1),H,W] and aligned LR[B,64,H,W]')
        if z.device != lr_features.device or z.device != self.input_proj.weight.device:
            raise ValueError('LRC inputs and parameters must share a device')
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError('LRC parameters must remain FP32')
        scalar = _kappa_tensor(kappa, value)
        with torch.autocast(device_type=value.device.type, enabled=False):
            phase = scalar * self.kappa_frequencies
            embedding = self.kappa_proj(torch.cat((phase.sin(), phase.cos())))
            feature = self.input_proj(torch.cat((value.float(), lr_features.float()), dim=1))
            for block in self.blocks:
                feature = block(feature, embedding)
            delta = self.output_proj(feature)
            if not bool(torch.isfinite(delta).all()):
                raise FloatingPointError('LRC produced a nonfinite residual')
            delta = delta.to(dtype=z.dtype)
            if not bool(torch.isfinite(delta).all()):
                raise FloatingPointError('LRC residual cannot be represented in latent dtype')
        return delta.unsqueeze(2) if temporal else delta

    def forward(self, z, lr_features, kappa=2.0):
        result = z + self.residual(z, lr_features, kappa)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError('LRC corrected latent is nonfinite')
        return result
