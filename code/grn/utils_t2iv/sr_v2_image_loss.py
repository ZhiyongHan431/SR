"""Opt-in differentiable image objectives for the fixed-grid GRN SR recipe.

The production HBQ ``decode`` returns unclamped BCHTW RGB in [-1, 1] nominally;
unlike the inference image writer it neither disables gradients nor quantizes to
uint8. We freeze its parameters, *not* the input graph. Hard-ST bits and the
straight-through output clamp intentionally use biased gradient estimators.
Nothing in this module downloads weights or constructs an evaluation-only IQA
wrapper (which would silently disable the input gradient).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


IMAGE_LOSS_KEYS = ('rgb', 'lowfreq', 'edge', 'perceptual')


def _freeze_for_input_gradients(module: nn.Module) -> None:
    module.eval()
    module.requires_grad_(False)


def _charbonnier(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((prediction - target).square() + 1e-6).sqrt().mean()


def _image_batch(hr_images: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    if not torch.is_tensor(hr_images):
        if not hr_images:
            raise ValueError('Image supervision requires nonempty HR images')
        hr_images = torch.stack([
            image.squeeze(1) if image.ndim == 4 and image.shape[1] == 1 else image
            for image in hr_images
        ])
    if hr_images.ndim == 5 and hr_images.shape[2] == 1:
        hr_images = hr_images.squeeze(2)
    if hr_images.ndim != 4 or hr_images.shape[1] != 3:
        raise ValueError('HR images must be [B,3,H,W], [B,3,1,H,W], or a list of [3,1,H,W]')
    return hr_images.detach().float()


def _logits_to_latent(logits: torch.Tensor, height: int, width: int,
                      hbq_round: int, latent_dim: int, mode: str) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[1:] != (2, height * width, hbq_round * latent_dim):
        raise ValueError(
            f'Expected bit logits [B,2,{height * width},{hbq_round * latent_dim}], '
            f'got {tuple(logits.shape)}'
        )
    soft_bits = logits.float().softmax(dim=1)[:, 1]
    if mode == 'hard_st':
        # Inference takes argmax after softmax in the logits' original dtype.
        # Close FP32/BF16 logits can round to tied probabilities, so using
        # argmax(logits) would not reproduce the actual inference forward.
        hard_bits = logits.softmax(dim=1).argmax(dim=1).to(soft_bits.dtype)
        bits = soft_bits + (hard_bits - soft_bits).detach()
    else:
        bits = soft_bits
    bits = bits.reshape(logits.shape[0], height * width, hbq_round, latent_dim)
    intervals = bits.new_tensor([0.5 ** (index + 1) for index in range(hbq_round)])
    latent = ((bits * 2 - 1) * intervals[None, None, :, None]).sum(dim=2)
    return latent.transpose(1, 2).reshape(logits.shape[0], latent_dim, 1, height, width)


def compute_sr_v2_image_losses(
    sr_aux: dict,
    sequence_packing_scales: Sequence[Sequence[tuple[int, int, int]]],
    hr_images: torch.Tensor | Sequence[torch.Tensor],
    vae: nn.Module,
    *,
    hbq_round: int = 4,
    latent_dim: int = 64,
    weights: Mapping[str, float] | None = None,
    mode: str = 'hard_st',
    warmup_factor: float = 1.0,
    perceptual_model: nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    """Return unweighted RGB/lowfreq/edge/perceptual terms and weighted ``total``.

    ``sr_aux['logits_by_scale']`` contains [B,2,H*W,R*C] logits, with one
    ``(1,H,W)`` grid per packed sample. HR pixels are RGB floats in [-1,1].
    Weights default to zero; zero weights/factor perform no decoder/perceptual
    call, no parameter or mode mutation, and no RNG consumption. Warmup is an
    externally computed optimizer-update factor in [0,1], applied exactly once
    to ``total``. The returned raw terms retain gradients for trainer logging or
    diagnostics; callers should detach logs themselves.

    Positive perceptual weight requires a supplied frozen differentiable model
    accepting [0,1] RGB (``build_offline_lpips`` provides one). Decoder and
    perceptual inference run outside AMP, preserving float32 objective math.
    """
    weights = {} if weights is None else dict(weights)
    unknown = set(weights) - set(IMAGE_LOSS_KEYS)
    if unknown:
        raise ValueError(f'Unknown image loss weights: {sorted(unknown)}')
    active_weights = {key: float(weights.get(key, 0.0)) for key in IMAGE_LOSS_KEYS}
    if any(not math.isfinite(value) or value < 0 for value in active_weights.values()):
        raise ValueError('Image loss weights must be finite and non-negative')
    factor = float(warmup_factor)
    if not math.isfinite(factor) or not 0 <= factor <= 1:
        raise ValueError('Image warmup_factor must be finite and in [0,1]')
    logits_by_scale = sr_aux.get('logits_by_scale', [])
    zero = logits_by_scale[0].new_zeros((), dtype=torch.float32) if logits_by_scale else torch.zeros(())
    result = {key: zero for key in IMAGE_LOSS_KEYS}
    result.update(total=zero, warmup_factor=zero.new_tensor(factor))
    if not any(active_weights.values()) or factor == 0:
        return result
    if mode not in ('soft', 'hard_st'):
        raise ValueError(f'Image latent mode must be soft or hard_st, got {mode!r}')
    if hbq_round < 1 or latent_dim < 1:
        raise ValueError('HBQ rounds and latent dimension must be positive')
    if not logits_by_scale or len(logits_by_scale) != len(sequence_packing_scales):
        raise ValueError('Image logits and packed scale counts must match and be nonempty')
    if active_weights['perceptual']:
        if perceptual_model is None:
            raise ValueError('Positive perceptual weight requires an offline perceptual_model')
        if getattr(perceptual_model, 'as_loss', True) is not True:
            raise ValueError('Perceptual model must run as_loss=True to retain input gradients')

    latents = []
    for logits, scales in zip(logits_by_scale, sequence_packing_scales):
        if len(scales) != 1 or len(scales[0]) != 3 or scales[0][0] != 1:
            raise ValueError('Image supervision requires exactly one (1,H,W) scale per sample')
        _, height, width = scales[0]
        if height < 1 or width < 1:
            raise ValueError('Latent grid dimensions must be positive')
        latents.append(_logits_to_latent(logits, height, width, hbq_round, latent_dim, mode))
    if any(latent.shape[1:] != latents[0].shape[1:] for latent in latents):
        raise ValueError('Image supervision requires a shared latent grid for the batch')
    latent = torch.cat(latents)
    target = _image_batch(hr_images).to(device=latent.device)
    if target.shape[0] != latent.shape[0]:
        raise ValueError('Predicted latent and HR image batch counts differ')
    if min(target.shape[-2:]) < 4:
        raise ValueError('Image objectives require HR spatial dimensions >= 4')

    _freeze_for_input_gradients(vae)
    # Production VAE is float32. Accommodate explicitly low-precision toy/user
    # decoders without mutating decoder dtypes or registering it in the GRN.
    decoder_parameter = next((p for p in vae.parameters() if p.is_floating_point()), None)
    decoder_dtype = torch.float32 if decoder_parameter is None else decoder_parameter.dtype
    with torch.autocast(device_type=latent.device.type, enabled=False):
        decoded = vae.decode(latent.to(dtype=decoder_dtype), slice=True)
        if decoded.ndim != 5 or decoded.shape[2] != 1:
            raise ValueError('HBQ decoder must return [B,3,1,H,W] RGB')
        prediction = (decoded.squeeze(2).float() + 1) / 2
        if prediction.shape != target.shape:
            raise ValueError(f'Decoded/HR image shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}')
        # Match saved-image forward values but do not kill correction gradients
        # when the raw HBQ decoder overshoots its nominal [-1,1] output range.
        prediction = prediction + (prediction.clamp(0, 1) - prediction).detach()
        target = (target + 1) / 2
        if active_weights['rgb']:
            result['rgb'] = _charbonnier(prediction, target)
        if active_weights['lowfreq']:
            result['lowfreq'] = sum(
                weight * _charbonnier(F.avg_pool2d(prediction, pool), F.avg_pool2d(target, pool))
                for pool, weight in ((2, 0.4), (4, 0.6))
            )
        if active_weights['edge']:
            result['edge'] = 0.5 * (
                _charbonnier(prediction[..., :, 1:] - prediction[..., :, :-1],
                             target[..., :, 1:] - target[..., :, :-1])
                + _charbonnier(prediction[..., 1:, :] - prediction[..., :-1, :],
                               target[..., 1:, :] - target[..., :-1, :])
            )
        if active_weights['perceptual']:
            _freeze_for_input_gradients(perceptual_model)
            result['perceptual'] = perceptual_model(prediction, target).float().mean()
        result['total'] = factor * sum(active_weights[key] * result[key] for key in IMAGE_LOSS_KEYS)
    return result


class OfflineAlexLPIPS(nn.Module):
    """LPIPS-v0.1 AlexNet loss with exclusively explicit local weight loading.

    Matches PyIQA's LPIPS eval-mode calculation, including its 1e-10 feature
    normalization epsilon and channel scaling. The installed PyIQA AlexNet
    ignores ``pretrained=False`` and requests torchvision weights, so relying
    on ``pnet_rand=True`` is insufficient to guarantee offline construction.
    """

    as_loss = True

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride=4, padding=2), nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2), nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2), nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2),
        )
        self.calibration = nn.ModuleList(nn.Conv2d(channels, 1, 1, bias=False)
                                         for channels in (64, 192, 384, 256, 256))
        self.register_buffer('shift', torch.tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
            raise ValueError('LPIPS requires matching BCHW RGB tensors')
        if min(prediction.shape[-2:]) < 32:
            raise ValueError('AlexNet LPIPS requires image dimensions >= 32')
        prediction = (2 * prediction - 1 - self.shift) / self.scale
        target = (2 * target - 1 - self.shift) / self.scale
        per_sample = prediction.new_zeros(prediction.shape[0])
        calibration_index = 0
        for layer_index, layer in enumerate(self.features):
            if layer_index == 12:  # PyIQA features stop at relu5, before pool5.
                break
            prediction, target = layer(prediction), layer(target)
            if layer_index in (1, 4, 7, 9, 11):
                pred_norm = prediction / (prediction.square().sum(1, keepdim=True).sqrt() + 1e-10)
                target_norm = target / (target.square().sum(1, keepdim=True).sqrt() + 1e-10)
                per_sample = per_sample + self.calibration[calibration_index](
                    (pred_norm - target_norm).square()).mean((1, 2, 3))
                calibration_index += 1
        return per_sample.mean()


def build_offline_lpips(calibration_path: str | Path, backbone_path: str | Path,
                        device: str | torch.device = 'cpu') -> OfflineAlexLPIPS:
    """Load frozen, differentiable AlexNet LPIPS from two *existing local* files.

    No PyIQA/torchvision constructor, URL, implicit cache lookup, or network API
    is called. All expected feature/calibration keys are strict; classifier
    tensors in the official AlexNet checkpoint are intentionally not used.
    CPU initialization RNG is restored so lazy construction cannot alter the
    training RNG stream. Missing paths fail before any model allocation.
    """
    calibration_path, backbone_path = Path(calibration_path), Path(backbone_path)
    for path in (calibration_path, backbone_path):
        if not path.is_file():
            raise FileNotFoundError(f'Required offline LPIPS weight file does not exist: {path}')
    with torch.random.fork_rng(devices=[]):
        model = OfflineAlexLPIPS()
    backbone = torch.load(backbone_path, map_location='cpu', weights_only=True)
    feature_weights = {key.removeprefix('features.'): value for key, value in backbone.items()
                       if key.startswith('features.')}
    model.features.load_state_dict(feature_weights, strict=True)
    calibration = torch.load(calibration_path, map_location='cpu', weights_only=True)
    expected = {f'lin{index}.model.1.weight' for index in range(5)}
    if set(calibration) != expected:
        raise ValueError(f'Expected local AlexNet LPIPS v0.1 calibration keys {sorted(expected)}')
    model.calibration.load_state_dict(
        {f'{index}.weight': calibration[f'lin{index}.model.1.weight'] for index in range(5)}, strict=True)
    _freeze_for_input_gradients(model)
    return model.to(device=device, dtype=torch.float32)
