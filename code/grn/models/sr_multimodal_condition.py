"""A shared-capacity residual conditioner for cached, LR-only visual tokens.

The feature extractor owns source-specific fixed projections and normalization.
Both Qwen and DINO must supply the same [B, N, input_dim] contract. This module
never receives HR pixels, captions derived from HR, or features from a decoder.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def condition_alpha(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('alpha must be an explicit finite nonnegative scalar')
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('alpha must be finite and nonnegative')
    return value


class MultimodalConditionAdapter(nn.Module):
    """Read external tokens with LR-grid queries and add a gated residual.

    The zero output projection preserves the initial conditioning and receives
    gradients on the first update. alpha=0 returns the original tensor object
    before reading external features. All new trainable computations use FP32;
    the returned grid preserves the original shape, device, and dtype.
    """

    def __init__(self, condition_dim=2304, input_dim=512, hidden_dim=256,
                 heads=8, num_tokens=65):
        super().__init__()
        self.condition_dim = _positive_int(condition_dim, 'condition_dim')
        self.input_dim = _positive_int(input_dim, 'input_dim')
        self.hidden_dim = _positive_int(hidden_dim, 'hidden_dim')
        self.heads = _positive_int(heads, 'heads')
        self.num_tokens = _positive_int(num_tokens, 'num_tokens')
        if hidden_dim % heads:
            raise ValueError('hidden_dim must be divisible by heads')
        self.query_norm = nn.LayerNorm(condition_dim, elementwise_affine=False)
        self.query_proj = nn.Linear(condition_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output_proj = nn.Linear(hidden_dim, condition_dim, bias=False)
        self.gate_proj = nn.Linear(hidden_dim, 1)
        # GRN may globally disable module reset_parameters: initialize explicitly.
        for projection in (self.query_proj, self.key_proj, self.value_proj):
            nn.init.xavier_uniform_(projection.weight)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def architecture(self):
        return {name: getattr(self, name) for name in
                ('condition_dim', 'input_dim', 'hidden_dim', 'heads', 'num_tokens')}

    def forward(self, condition_grid, tokens=None, mask=None, alpha=1.0):
        alpha = condition_alpha(alpha)
        if alpha == 0:
            return condition_grid
        if (not torch.is_tensor(condition_grid) or condition_grid.ndim != 4
                or not condition_grid.is_floating_point()
                or condition_grid.shape[0] < 1
                or condition_grid.shape[1] != self.condition_dim
                or min(condition_grid.shape[-2:]) < 1):
            raise ValueError('Expected floating condition grid [B, condition_dim, H, W]')
        batch, _, height, width = condition_grid.shape
        if (not torch.is_tensor(tokens) or not tokens.is_floating_point()
                or tuple(tokens.shape) != (batch, self.num_tokens, self.input_dim)
                or tokens.requires_grad):
            raise ValueError('Expected frozen feature tokens [B, num_tokens, input_dim]')
        if (tokens.device != condition_grid.device
                or condition_grid.device != self.query_proj.weight.device):
            raise ValueError('Condition grid, feature tokens, and adapter must share a device')
        if any(parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise ValueError('Multimodal adapter parameters must remain FP32')
        if not bool(torch.isfinite(condition_grid).all()) or not bool(torch.isfinite(tokens).all()):
            raise ValueError('Condition grid and feature tokens must be finite')
        if mask is None:
            mask = torch.ones((batch, self.num_tokens), dtype=torch.bool, device=tokens.device)
        if (not torch.is_tensor(mask) or mask.dtype != torch.bool
                or tuple(mask.shape) != (batch, self.num_tokens)
                or mask.device != tokens.device or not bool(mask.any(dim=1).all())):
            raise ValueError('Feature mask must be boolean [B, num_tokens] with a valid token per image')
        with torch.autocast(device_type=condition_grid.device.type, enabled=False):
            queries = condition_grid.float().flatten(2).transpose(1, 2)
            queries = self.query_proj(self.query_norm(queries))
            keys = self.key_proj(tokens.float())
            values = self.value_proj(tokens.float())
            head_dim = self.hidden_dim // self.heads
            split = lambda value: value.reshape(batch, -1, self.heads, head_dim).transpose(1, 2)
            scores = torch.matmul(split(queries), split(keys).transpose(-2, -1)) / math.sqrt(head_dim)
            scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))
            weights = F.softmax(scores, dim=-1)
            readout = torch.matmul(weights, split(values)).transpose(1, 2).reshape(batch, height * width, self.hidden_dim)
            delta = self.output_proj(readout) * torch.sigmoid(self.gate_proj(queries))
            delta = (alpha * delta).transpose(1, 2).reshape_as(condition_grid).to(condition_grid.dtype)
            if not bool(torch.isfinite(delta).all()):
                raise FloatingPointError('Multimodal residual is not finite in the conditioning dtype')
            result = condition_grid + delta
            if not bool(torch.isfinite(result).all()):
                raise FloatingPointError('Multimodal conditioning output is not finite')
        return result
