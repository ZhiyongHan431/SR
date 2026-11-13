"""Isolated S3 integration candidate: bounded spatial/time/round log-odds correction.

This candidate is not promoted to the active training tree. It has no GT argument, no
token locking, no recurrent state, and no quality claim. LR features must already
correspond to the HR token grid; this module does not infer their alignment.
"""
from __future__ import annotations

import math
from numbers import Real
import torch
from torch import nn
from torch.nn import functional as F


MODES = ('plain', 'shared_round', 'per_round')


def resolve_calibrator_kappa(kappa, default=2.0):
    """Validate a runtime control without changing checkpoint architecture metadata.

    A scalar tensor is a control value, not a trainable parameter. Converting it
    to a Python scalar preserves the existing FP32 scalar multiplication path.
    """
    value = default if kappa is None else kappa
    if torch.is_tensor(value):
        if value.ndim != 0 or value.dtype == torch.bool or value.is_complex():
            raise ValueError('kappa must be a nonnegative finite real scalar')
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('kappa must be a nonnegative finite real scalar')
    value = float(value)
    if not math.isfinite(value) or value < 0 or value > torch.finfo(torch.float32).max:
        raise ValueError('kappa must be nonnegative, finite, and representable in FP32')
    return value


def _finite_float(name, value):
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise ValueError(f'{name} must be a floating point tensor')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} contains nonfinite values')


def _finite_float32(name, value):
    _finite_float(name, value)
    converted = value.float()
    if not bool(torch.isfinite(converted).all()):
        raise ValueError(f'{name} cannot be represented as finite FP32 values')
    return converted


def _safe_normalize(value):
    # Scale first: ordinary LayerNorm can overflow its variance on finite 1e30
    # inputs. Scaling by a positive channelwise maximum preserves direction.
    value = _finite_float32('normalization input', value)
    value = value / value.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    return F.layer_norm(value, (value.shape[-1],), eps=1e-5)


def apply_log_odds_residual(logits, delta):
    """Return FP32 logits after symmetric +/- delta/2 accumulation.

    logits: [B,2,N,256], delta: [B,N,4,64]. Inputs remain unchanged.
    The log-odds identity is exact in real arithmetic, not for arbitrary finite
    FP32 magnitudes. A zero delta equals logits.float() numerically; promotion
    alone can change a downstream BF16 softmax tie, so this is NOT a claim of
    original GRN bit/sampling identity.
    """
    _finite_float('logits', logits)
    _finite_float('delta', delta)
    if (delta.ndim != 4 or delta.shape[-2:] != (4, 64)
            or logits.shape != (delta.shape[0], 2, delta.shape[1], 256)):
        raise ValueError('Expected logits [B,2,N,256] and delta [B,N,4,64]')
    if logits.device != delta.device:
        raise ValueError('logits and delta must be on the same device')
    with torch.autocast(device_type=logits.device.type, enabled=False):
        base = _finite_float32('logits', logits)
        correction = _finite_float32('delta', delta).flatten(2) * .5
        # Preserve the native head's class-last storage order. A class-first
        # stack changes CUDA reduction order even for a numerically zero delta.
        # Slice writes retain autograd into both the base and learned correction;
        # no zero-delta branch bypasses the calibrator or changes the input tensor.
        corrected = torch.empty_like(base, memory_format=torch.preserve_format)
        corrected[:, 0] = base[:, 0] - correction
        corrected[:, 1] = base[:, 1] + correction
        if not bool(torch.isfinite(corrected).all()):
            raise FloatingPointError('FP32 logit accumulation overflowed')
    return corrected


class BoundedRoundCalibrator(nn.Module):
    """A shared input bottleneck and one of three explicitly comparable heads.

    plain:        one width->256 projection, no round embedding.
    shared_round: a shared width->64 head evaluated at four round embeddings.
    per_round:    four width->64 heads evaluated at four round embeddings.

    All modes use the same HR/LR/time bottleneck and learned sigmoid gate. The
    shared head has fewer parameters; compare both parameter count and measured
    compute, rather than describing these variants as exactly budget-matched.
    """
    def __init__(self, hr_dim=2304, lr_dim=64, width=128, *, mode='per_round',
                 rounds=4, bits_per_round=64, max_log_odds=2., initial_gate=.5):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        if any(type(d) is not int or d < 1 for d in (hr_dim, lr_dim, width)):
            raise ValueError('feature and bottleneck dimensions must be positive integers')
        if type(rounds) is not int or type(bits_per_round) is not int or (rounds, bits_per_round) != (4, 64):
            raise ValueError('This prototype requires M4 x 64-bit-channel HBQ')
        if (isinstance(max_log_odds, bool) or not math.isfinite(max_log_odds) or max_log_odds <= 0
                or max_log_odds > torch.finfo(torch.float32).max
                or isinstance(initial_gate, bool) or not math.isfinite(initial_gate) or not 0 < initial_gate < 1):
            raise ValueError('max_log_odds must be positive finite FP32; initial_gate must lie inside (0,1)')
        self.hr_dim, self.lr_dim, self.width = hr_dim, lr_dim, width
        self.mode, self.rounds, self.bits_per_round = mode, rounds, bits_per_round
        self.max_log_odds = float(max_log_odds)
        self.initial_gate = float(initial_gate)
        self.hr_proj = nn.Linear(hr_dim, width)
        self.lr_proj = nn.Linear(lr_dim, width)
        self.time_proj = nn.Linear(2, width)
        self.round_embedding = None if mode == 'plain' else nn.Embedding(rounds, width)
        if self.round_embedding is not None:
            nn.init.normal_(self.round_embedding.weight, std=.02)
        if mode == 'plain':
            self.output_heads = nn.ModuleList([nn.Linear(width, rounds * bits_per_round)])
            self.gate_heads = nn.ModuleList([nn.Linear(width, rounds)])
        elif mode == 'shared_round':
            self.output_heads = nn.ModuleList([nn.Linear(width, bits_per_round)])
            self.gate_heads = nn.ModuleList([nn.Linear(width, 1)])
        else:
            self.output_heads = nn.ModuleList([nn.Linear(width, bits_per_round) for _ in range(rounds)])
            self.gate_heads = nn.ModuleList([nn.Linear(width, 1) for _ in range(rounds)])
        # The production factory disables nn.Linear.reset_parameters.
        # New modules created after global init must initialize every tensor.
        for projection in (self.hr_proj, self.lr_proj, self.time_proj):
            nn.init.kaiming_uniform_(projection.weight, a=math.sqrt(5))
            bound = 1 / math.sqrt(projection.in_features)
            nn.init.uniform_(projection.bias, -bound, bound)
        self.reset_output()

    def reset_output(self):
        """Reapply zero residual and nonzero gate after global initialization."""
        for head in self.output_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        # Zero *gate logits* mean a nonzero gate of .5, not a dead zero gate.
        gate_logit = math.log(self.initial_gate / (1. - self.initial_gate))
        for head in self.gate_heads:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, gate_logit)

    def _validate(self, hidden, lr_features, time, round_ids):
        for name, value in (('hidden', hidden), ('lr_features', lr_features), ('time', time)):
            _finite_float(name, value)
        if hidden.ndim != 3 or hidden.shape[-1] != self.hr_dim or hidden.shape[0] < 1 or hidden.shape[1] < 1:
            raise ValueError('hidden must be nonempty [B,N,hr_dim]')
        if (lr_features.ndim != 4 or lr_features.shape[:2] != (hidden.shape[0], self.lr_dim)
                or min(lr_features.shape[-2:]) < 1 or lr_features.shape[-2] * lr_features.shape[-1] != hidden.shape[1]):
            raise ValueError('LR must be [B,lr_dim,H,W] with H*W matching HR tokens')
        if time.shape != (hidden.shape[0],) or bool(((time < 0) | (time > 1)).any()):
            raise ValueError('time must be [B] actual retention fractions in [0,1]')
        if lr_features.device != hidden.device or time.device != hidden.device or self.hr_proj.weight.device != hidden.device:
            raise ValueError('features, time, and module parameters must share a device')
        if self.hr_proj.weight.dtype != torch.float32:
            raise ValueError('S3 calibration requires FP32 module parameters')
        if round_ids is None:
            round_ids = torch.arange(self.rounds, device=hidden.device)
        if (not torch.is_tensor(round_ids) or round_ids.dtype != torch.long or round_ids.ndim != 1
                or round_ids.numel() < 1 or round_ids.device != hidden.device
                or bool(((round_ids < 0) | (round_ids >= self.rounds)).any())
                or round_ids.unique().numel() != round_ids.numel()):
            raise ValueError('round_ids must be a nonempty unique int64 vector inside [0,3]')
        return round_ids

    def forward(self, hidden, lr_features, time, round_ids=None, *, return_gate=False, kappa=None):
        cap = self.max_log_odds if kappa is None else resolve_calibrator_kappa(kappa)
        ids = self._validate(hidden, lr_features, time, round_ids)
        # No caller AMP may narrow the learned residual, gate, or tanh. This is
        # an explicit prototype precision policy, not yet a GRN sampler policy.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            lr_tokens = lr_features.flatten(2).transpose(1, 2)
            time32 = _finite_float32('time', time)
            time_features = torch.stack((time32, time32.square()), dim=-1)
            base = (self.hr_proj(_safe_normalize(hidden)) + self.lr_proj(_safe_normalize(lr_tokens))
                    + self.time_proj(time_features)[:, None])
            _finite_float('calibrator preactivation', base)
            if self.mode == 'plain':
                features = F.silu(base)
                raw = self.output_heads[0](features).reshape(*base.shape[:2], 4, 64).index_select(2, ids)
                gate_logits = self.gate_heads[0](features).index_select(2, ids).unsqueeze(-1)
            else:
                features = F.silu(base[:, :, None, :] + self.round_embedding(ids)[None, None])
                if self.mode == 'shared_round':
                    raw = self.output_heads[0](features)
                    gate_logits = self.gate_heads[0](features)
                else:
                    raw = torch.stack([self.output_heads[r](features[:, :, index])
                                       for index, r in enumerate(ids.tolist())], dim=2)
                    gate_logits = torch.stack([self.gate_heads[r](features[:, :, index])
                                               for index, r in enumerate(ids.tolist())], dim=2)
            _finite_float('raw delta', raw)
            _finite_float('gate logits', gate_logits)
            gates = gate_logits.sigmoid()
            delta = cap * gates * raw.tanh()
            if not bool(torch.isfinite(delta).all()):
                raise FloatingPointError('Nonfinite calibrator output; inspect learned parameters')
        return (delta, gates) if return_gate else delta

    def calibrate_logits(self, logits, hidden, lr_features, time, *, kappa=None):
        control = {} if kappa is None else {'kappa': kappa}
        return apply_log_odds_residual(logits, self(hidden, lr_features, time, **control))

    def parameter_counts(self):
        return {**{name: sum(p.numel() for p in module.parameters()) for name, module in self.named_children()},
                'total': sum(p.numel() for p in self.parameters())}
