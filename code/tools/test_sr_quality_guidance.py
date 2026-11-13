#!/usr/bin/env python3
"""CPU checks for quality guidance math. Run: python code/tools/test_sr_quality_guidance.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from grn.models.sr_quality_guidance import (
    SRQualityGuidance, degrade_lr_condition, guided_logits, sample_bits_by_plane)


def main():
    generator = torch.Generator().manual_seed(0)
    positive = torch.randn(1, 6, 256, 2, generator=generator)
    negative = torch.randn(1, 6, 256, 2, generator=generator)
    odds = lambda x: x[..., 1] - x[..., 0]

    # Equal branches: guidance is an exact no-op on the log-odds.
    config = SRQualityGuidance(scale=3., bound=2., plane_weights=(1, 1, 1, 1))
    assert torch.allclose(odds(guided_logits(positive, positive, config)), odds(positive), atol=1e-6)

    # Shift is bounded and confined to the selected planes.
    config = SRQualityGuidance(scale=50., bound=1.5, plane_weights=(0, 0, 1, 1))
    shift = odds(guided_logits(positive, negative, config)) - odds(positive)
    assert float(shift.abs().max()) <= 1.5 + 1e-5
    assert float(shift[..., :128].abs().max()) < 1e-6 and float(shift[..., 128:].abs().max()) > 0
    expected = (50. * (odds(positive) - odds(negative))).clamp(-1.5, 1.5)[..., 128:]
    assert torch.allclose(shift[..., 128:], expected, atol=1e-5)

    # Zero temperatures reproduce the greedy sampler.
    labels, chosen = sample_bits_by_plane(positive, (0, 0, 0, 0), torch.Generator().manual_seed(1))
    assert torch.equal(labels, positive.argmax(-1))
    assert torch.allclose(chosen, positive.softmax(-1).max(-1).values, atol=1e-6)

    # Stochastic planes change only their own bits and follow the Bernoulli rate.
    big = torch.zeros(1, 4096, 256, 2)
    big[..., 1] = 1.0
    labels, _ = sample_bits_by_plane(big, (0, 0, 1, 1), torch.Generator().manual_seed(2))
    assert bool((labels[..., :128] == 1).all())
    rate = labels[..., 128:].float().mean().item()
    assert abs(rate - torch.sigmoid(torch.tensor(1.0)).item()) < 0.01, rate

    # Degraded LR keeps shape/range and removes high-frequency energy.
    lr = torch.rand(1, 3, 128, 128, generator=generator) * 2 - 1
    degraded = degrade_lr_condition(lr, SRQualityGuidance(scale=1., lr_blur_sigma=1.2, lr_downscale=1.5))
    assert degraded.shape == lr.shape and float(degraded.abs().max()) <= 1
    assert float(degraded.std()) < float(lr.std())
    flat = torch.full((1, 3, 128, 128), 0.25)
    assert torch.allclose(degrade_lr_condition(flat, SRQualityGuidance(scale=1., lr_blur_sigma=2.)), flat, atol=1e-5)

    # Invalid settings are rejected.
    for bad in (dict(scale=-1.), dict(plane_weights=(1, 1, 1)), dict(negative='zeros'),
                dict(scale=1., lr_blur_sigma=0., lr_downscale=1.)):
        try:
            SRQualityGuidance(**bad)
        except ValueError:
            continue
        raise AssertionError(f'accepted invalid config {bad}')
    print('sr_quality_guidance checks passed')


if __name__ == '__main__':
    main()
