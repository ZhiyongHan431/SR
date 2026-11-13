"""Opt-in quality-contrastive revision guidance for the 50-round SR sampler.

Nothing here runs unless a caller passes ``sr_quality_guidance`` to
``GRN.autoregressive_infer``; with it left at ``None`` the sampler executes the
same statements as before this module existed.

Every round already predicts the log-odds of all 262,144 bits. Guidance moves
that full prediction away from a *lower-quality* prediction of the same bits:

    l_g = l_pos + clamp(w * (l_pos - l_neg), -bound, +bound) * plane_weight

The negative prediction comes from one of these sources:

``degraded_lr``  the same network conditioned on a further-degraded LR image.
                 It occupies the second packed branch that guidance=1 inference
                 already computes and discards, so it adds no backbone tokens.
``calibrator``   a second bounded calibrator (C-) trained to predict the bits of
                 a low-quality target from the same hidden states. It adds no
                 backbone forward at all.
``prompt``       the second branch keeps the same LR anchor and differs only in
                 its text slot (empty prompt, a learned negative token, or a
                 cached negative prompt), while the first branch carries the
                 semantic prompt. See ``sr_semantic_prompt``.
``degraded_lr_prompt``  both at once: degraded LR and the negative text slot.

The same config can also set per-bit-plane sampling temperatures. Plane 0 is
the most significant HBQ plane (interval 1/2), plane 3 the least significant.

These are experimental controls. No quality claim is attached to any default.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re

import torch
from torch.nn import functional as F

from grn.models.sr_v2_calibrator import BoundedRoundCalibrator

NEGATIVE_SOURCES = ('degraded_lr', 'calibrator', 'prompt', 'degraded_lr_prompt')
NEGATIVE_SCHEMA = 'grn_s5_negative_calibrator_v1'
PLANES, CHANNELS = 4, 64


def _finite_scalar(name, value, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite real scalar')
    value = float(value)
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f'{name} is outside [{minimum}, {maximum}]')
    return value


def _plane_vector(name, values, *, minimum, maximum):
    if values is None:
        return None
    if isinstance(values, str):
        values = [item for item in values.split(',') if item.strip()]
    values = tuple(_finite_scalar(f'{name}[{index}]', float(value), minimum=minimum, maximum=maximum)
                   for index, value in enumerate(values))
    if len(values) != PLANES:
        raise ValueError(f'{name} needs exactly {PLANES} values, most significant plane first')
    return values


class SRQualityGuidance:
    """Validated, immutable description of one guidance/sampling setting.

    scale               w >= 0. Zero disables guidance but keeps plane sampling.
    bound               largest absolute log-odds shift per bit, like kappa.
    start               guidance acts on rounds whose input retention >= start.
    plane_weights       per-plane multiplier in [0,1]; default acts on the two
                        least significant planes only, which carry texture.
    negative            one of NEGATIVE_SOURCES (see the module docstring).
    lr_blur_sigma       Gaussian sigma in LR pixels for the degraded-LR branch.
    lr_downscale        optional extra bicubic down/up factor (>= 1) for it.
    plane_temperatures  None keeps the caller's sampler. Otherwise four values:
                        0 means greedy for that plane, t > 0 samples that plane
                        from the guided log-odds divided by t.
    """
    __slots__ = ('scale', 'bound', 'start', 'plane_weights', 'negative',
                 'lr_blur_sigma', 'lr_downscale', 'plane_temperatures')

    def __init__(self, *, scale=0., bound=2., start=0.3, plane_weights=(0., 0., 1., 1.),
                 negative='degraded_lr', lr_blur_sigma=1.2, lr_downscale=1.,
                 plane_temperatures=None):
        if negative not in NEGATIVE_SOURCES:
            raise ValueError(f'negative must be one of {NEGATIVE_SOURCES}')
        object.__setattr__(self, 'scale', _finite_scalar('scale', scale, minimum=0., maximum=64.))
        object.__setattr__(self, 'bound', _finite_scalar('bound', bound, minimum=0., maximum=64.))
        object.__setattr__(self, 'start', _finite_scalar('start', start, minimum=0., maximum=1.))
        object.__setattr__(self, 'plane_weights',
                           _plane_vector('plane_weights', plane_weights, minimum=0., maximum=1.))
        object.__setattr__(self, 'negative', negative)
        object.__setattr__(self, 'lr_blur_sigma',
                           _finite_scalar('lr_blur_sigma', lr_blur_sigma, minimum=0., maximum=16.))
        object.__setattr__(self, 'lr_downscale',
                           _finite_scalar('lr_downscale', lr_downscale, minimum=1., maximum=8.))
        object.__setattr__(self, 'plane_temperatures',
                           _plane_vector('plane_temperatures', plane_temperatures, minimum=0., maximum=8.))
        if self.guides and self.degrades_lr and self.lr_blur_sigma == 0 and self.lr_downscale == 1:
            raise ValueError('degraded-LR guidance needs lr_blur_sigma > 0 or lr_downscale > 1')

    def __setattr__(self, name, value):
        raise AttributeError('SRQualityGuidance is immutable')

    @property
    def guides(self):
        return self.scale > 0 and self.bound > 0 and any(self.plane_weights)

    @property
    def degrades_lr(self):
        return self.negative in ('degraded_lr', 'degraded_lr_prompt')

    @property
    def uses_negative_text(self):
        return self.negative in ('prompt', 'degraded_lr_prompt')

    @property
    def needs_negative_lr(self):
        # True whenever the second packed branch supplies the negative
        # prediction. It then needs its own LR input: a degraded one, or for
        # 'prompt' the unchanged LR.
        return self.guides and self.negative != 'calibrator'

    @property
    def needs_negative_calibrator(self):
        return self.guides and self.negative == 'calibrator'

    def active_at(self, retention):
        return self.guides and float(retention) >= self.start

    def metadata(self):
        return {name: (list(value) if isinstance(value, tuple) else value)
                for name, value in ((slot, getattr(self, slot)) for slot in self.__slots__)}


def _class_last_logits(name, logits):
    if (not torch.is_tensor(logits) or not logits.is_floating_point() or logits.ndim != 4
            or logits.shape[-2:] != (PLANES * CHANNELS, 2)):
        raise ValueError(f'{name} must be floating [B,N,{PLANES * CHANNELS},2] logits')
    if not bool(torch.isfinite(logits).all()):
        raise ValueError(f'{name} contains nonfinite values')
    return logits.float()


def _per_bit(values, reference):
    """[4] plane values -> [256] in the plane-major order used by the bit head."""
    return reference.new_tensor(values, dtype=torch.float32).repeat_interleave(CHANNELS)


def guided_logits(positive, negative, guidance):
    """Bounded log-odds extrapolation away from the negative prediction.

    positive, negative: [B,N,256,2] class-last logits of the same bits.
    Returns FP32 logits of the same shape. The shift is split symmetrically
    over the two classes, exactly like the calibrator's log-odds residual.
    """
    positive = _class_last_logits('positive logits', positive)
    negative = _class_last_logits('negative logits', negative)
    if positive.shape != negative.shape or positive.device != negative.device:
        raise ValueError('positive and negative logits must share shape and device')
    if not isinstance(guidance, SRQualityGuidance):
        raise ValueError('guidance must be an SRQualityGuidance')
    with torch.autocast(device_type=positive.device.type, enabled=False):
        difference = (positive[..., 1] - positive[..., 0]) - (negative[..., 1] - negative[..., 0])
        shift = (guidance.scale * difference).clamp(-guidance.bound, guidance.bound)
        shift = shift * _per_bit(guidance.plane_weights, positive)
        result = torch.stack((positive[..., 0] - .5 * shift, positive[..., 1] + .5 * shift), dim=-1)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError('Guided logits are nonfinite')
    return result


def sample_bits_by_plane(logits, temperatures, generator):
    """Per-plane greedy or stochastic bit decisions.

    logits: [B,N,256,2] class-last. temperatures: four values, 0 = greedy.
    Returns (labels [B,N,256] int64, probability of the chosen label [B,N,256]).
    One uniform draw per bit is consumed from ``generator`` for every call, also
    for greedy planes, so the number of draws does not depend on the setting.
    """
    logits = _class_last_logits('logits', logits)
    temperatures = _plane_vector('temperatures', temperatures, minimum=0., maximum=8.)
    with torch.autocast(device_type=logits.device.type, enabled=False):
        log_odds = logits[..., 1] - logits[..., 0]
        greedy = log_odds > 0
        temperature = _per_bit(temperatures, logits)
        stochastic = temperature > 0
        probability_one = torch.sigmoid(log_odds / temperature.clamp_min(1e-6))
        uniform = torch.rand(log_odds.shape, device=log_odds.device, generator=generator)
        labels = torch.where(stochastic, uniform < probability_one, greedy)
        plain = torch.sigmoid(log_odds)
        chosen = torch.where(labels, plain, 1. - plain)
    return labels.long(), chosen


def degrade_lr_condition(lr_condition, guidance):
    """Deterministic extra degradation of a normalized LR batch [B,3,H,W] in [-1,1].

    Gaussian blur with reflect padding, then an optional antialiased bicubic
    down/up resize. Blur and resize are both operations the Real-ESRGAN training
    degradations already contain, so the result stays a plausible LR input.
    """
    if (not torch.is_tensor(lr_condition) or lr_condition.ndim != 4 or lr_condition.shape[1] != 3
            or not lr_condition.is_floating_point() or not bool(torch.isfinite(lr_condition).all())):
        raise ValueError('lr_condition must be a finite floating [B,3,H,W] tensor')
    if not isinstance(guidance, SRQualityGuidance):
        raise ValueError('guidance must be an SRQualityGuidance')
    image = lr_condition.float()
    sigma = guidance.lr_blur_sigma
    if sigma > 0:
        radius = max(1, int(math.ceil(3 * sigma)))
        radius = min(radius, min(image.shape[-2:]) - 1)
        offsets = torch.arange(-radius, radius + 1, device=image.device, dtype=torch.float32)
        kernel = torch.exp(-.5 * (offsets / sigma) ** 2)
        kernel = kernel / kernel.sum()
        padded = F.pad(image, (radius, radius, radius, radius), mode='reflect')
        channels = image.shape[1]
        padded = F.conv2d(padded, kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels)
        image = F.conv2d(padded, kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels)
    if guidance.lr_downscale > 1:
        height, width = image.shape[-2:]
        small = (max(1, round(height / guidance.lr_downscale)), max(1, round(width / guidance.lr_downscale)))
        image = F.interpolate(image, size=small, mode='bicubic', align_corners=False, antialias=True)
        image = F.interpolate(image, size=(height, width), mode='bicubic', align_corners=False)
    return image.clamp(-1, 1).to(dtype=lr_condition.dtype)


# --------------------------------------------------------------------------
# Negative calibrator (C-) lifecycle. It is stored in its own file and never
# enters the C/R adapter checkpoint, whose loader requires exact membership.
# --------------------------------------------------------------------------

def _sha256_text(name, value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError(f'{name} must be a lowercase SHA256 digest')
    return value


def _calibrator_architecture(module):
    return {key: getattr(module, key) for key in (
        'mode', 'hr_dim', 'lr_dim', 'width', 'rounds', 'bits_per_round',
        'max_log_odds', 'initial_gate')}


def configure_negative_calibrator(model, *, trainable=False):
    """Attach C- as a copy of the loaded positive calibrator.

    Starting from the positive weights makes the initial guidance difference
    exactly zero; training then only has to learn the quality direction.
    """
    positive = getattr(model, 'sr_round_calibrator', None)
    if not isinstance(positive, BoundedRoundCalibrator):
        raise ValueError('A negative calibrator requires the loaded positive calibrator')
    if hasattr(model, 'sr_negative_calibrator'):
        raise RuntimeError('Negative calibrator already configured; refusing overwrite')
    module = deepcopy(positive)
    module.requires_grad_(bool(trainable))
    module.train(bool(trainable))
    model.sr_negative_calibrator = module
    return _calibrator_architecture(module)


def save_negative_calibrator(path, model, *, base_sha256, positive_adapter_sha256, step, metadata=None):
    module = getattr(model, 'sr_negative_calibrator', None)
    if not isinstance(module, BoundedRoundCalibrator):
        raise ValueError('No negative calibrator to save')
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Refusing to overwrite existing checkpoint: {path}')
    if type(step) is not int or step < 0:
        raise ValueError('step must be a nonnegative integer')
    payload = {'schema': NEGATIVE_SCHEMA, 'base_sha256': _sha256_text('base SHA256', base_sha256),
               'positive_adapter_sha256': (None if positive_adapter_sha256 is None
                                           else _sha256_text('positive adapter SHA256', positive_adapter_sha256)),
               'architecture': _calibrator_architecture(module),
               'state_dict': {key: value.detach().cpu().clone() for key, value in module.state_dict().items()},
               'step': step, 'metadata': {} if metadata is None else metadata}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    return {'path': str(path), 'step': step}


def load_negative_calibrator(path, model, *, expected_base_sha256, expected_positive_adapter_sha256=None,
                             trainable=False):
    """Validate binding and tensors, then attach C- to the model."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    keys = {'schema', 'base_sha256', 'positive_adapter_sha256', 'architecture', 'state_dict', 'step', 'metadata'}
    if type(payload) is not dict or set(payload) != keys or payload['schema'] != NEGATIVE_SCHEMA:
        raise ValueError('Unsupported negative calibrator checkpoint')
    if payload['base_sha256'] != _sha256_text('expected base SHA256', expected_base_sha256):
        raise ValueError('Negative calibrator belongs to a different base checkpoint')
    if (expected_positive_adapter_sha256 is not None
            and payload['positive_adapter_sha256'] != expected_positive_adapter_sha256):
        raise ValueError('Negative calibrator was trained beside a different positive adapter')
    architecture = configure_negative_calibrator(model, trainable=trainable)
    try:
        if payload['architecture'] != architecture:
            raise ValueError('Negative calibrator architecture differs from the positive calibrator')
        reference = model.sr_negative_calibrator.state_dict()
        state = payload['state_dict']
        if type(state) is not dict or set(state) != set(reference):
            raise ValueError('Negative calibrator state membership differs')
        for key, value in state.items():
            if (not torch.is_tensor(value) or value.shape != reference[key].shape
                    or value.dtype != reference[key].dtype or not bool(torch.isfinite(value).all())):
                raise ValueError(f'Invalid negative calibrator tensor: {key}')
        model.sr_negative_calibrator.load_state_dict(state, strict=True)
    except BaseException:
        del model.sr_negative_calibrator
        raise
    return {'step': payload['step'], 'metadata': payload['metadata'], 'architecture': architecture,
            'positive_adapter_sha256': payload['positive_adapter_sha256']}
