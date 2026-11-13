#!/usr/bin/env python3
"""CPU checks for the semantic prompt math. Run: python code/tools/test_sr_semantic_prompt.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from grn.models.sr_quality_guidance import SRQualityGuidance
from grn.models.sr_semantic_prompt import (
    SemanticPrompt, SemanticPromptProjector, checked_text_features,
    load_semantic_prompt, save_semantic_prompt, tensor_sha256)


class StubEncoder:
    """Stands in for FrozenCLIPImageEncoder: no weights are needed for the math."""
    embed_dim = 16

    def identity(self):
        return {'backend': 'stub', 'weight_sha256': 'a' * 64, 'embed_dim': 16, 'resolution': 224}

    def __call__(self, image):
        return torch.ones(image.shape[0], 16)

    def to(self, device=None):
        return self


def rejects(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return True
    return False


def main():
    torch.manual_seed(0)
    sizes = dict(clip_dim=16, text_dim=32, hidden_dim=8)
    base, embedding = torch.randn(1, 32), torch.randn(1, 16)

    # A fresh projector reproduces the empty prompt exactly, for one or several tokens.
    projector = SemanticPromptProjector(tokens=1, base_tokens=1, **sizes)
    assert torch.equal(projector.positive_text(embedding, base), base)
    assert torch.equal(projector.negative_text(base), base)
    wide = SemanticPromptProjector(tokens=3, base_tokens=1, **sizes)
    assert torch.equal(wide.positive_text(embedding, base), base.expand(3, 32))

    # Offsets scale with `scale` and the base RMS, and ignore the CLIP embedding norm.
    with torch.no_grad():
        projector.token_out.weight.normal_(std=0.1)
    one = projector.positive_text(embedding, base) - base
    assert float(one.abs().max()) > 0
    assert torch.allclose(projector.positive_text(embedding, base, scale=0.5) - base, 0.5 * one, atol=1e-6)
    assert torch.equal(projector.positive_text(embedding, base, scale=0.), base)
    assert torch.allclose(projector.positive_text(7.0 * embedding, base), base + one, atol=1e-5)
    assert torch.allclose(projector.positive_text(embedding, 3.0 * base) - 3.0 * base, 3.0 * one, atol=1e-5)

    # The tokens carry gradient back into the projector.
    projector.zero_grad()
    projector.positive_text(embedding, base).sum().backward()
    assert float(projector.token_out.weight.grad.abs().sum()) > 0

    # Alignment loss is 1 - cosine: 0 for identical embeddings, 2 for opposite ones.
    fresh = SemanticPromptProjector(tokens=1, base_tokens=1, **sizes)
    assert float(fresh.alignment_loss(embedding, embedding)) < 1e-6
    assert abs(float(fresh.alignment_loss(embedding, -embedding)) - 2.0) < 1e-5

    # The learned negative token is the empty prompt plus an offset in RMS units.
    with torch.no_grad():
        projector.negative_delta.fill_(0.5)
    assert torch.allclose(projector.negative_text(base), base + 0.5 * base.square().mean().sqrt(), atol=1e-6)

    # Save/load round trip and identity bindings.
    encoder = StubEncoder()
    bindings = dict(expected_base_sha256='b' * 64, expected_adapter_sha256=None,
                    clip_identity=encoder.identity(), empty_features=base)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'semantic.pt'
        save_semantic_prompt(path, projector, base_sha256='b' * 64, adapter_sha256=None,
                             clip_identity=encoder.identity(), empty_text_sha256=tensor_sha256(base),
                             negative_trained=True, step=3, metadata={'note': 'test'})
        loaded, report = load_semantic_prompt(path, **bindings)
        assert report['negative_trained'] is True and report['step'] == 3
        assert torch.equal(loaded.positive_text(embedding, base), projector.positive_text(embedding, base))
        for change in (dict(expected_base_sha256='c' * 64), dict(expected_adapter_sha256='d' * 64),
                       dict(empty_features=base + 1.0)):
            assert rejects(load_semantic_prompt, path, **{**bindings, **change}), change
        try:
            save_semantic_prompt(path, projector, base_sha256='b' * 64, adapter_sha256=None,
                                 clip_identity=encoder.identity(), empty_text_sha256=tensor_sha256(base),
                                 negative_trained=True, step=3)
        except FileExistsError:
            pass
        else:
            raise AssertionError('overwrote an existing checkpoint')
    runtime = SemanticPrompt(encoder, loaded, base, negative_trained=True)
    positive, negative = runtime.text_features(torch.zeros(1, 3, 8, 8))
    assert positive.shape == (1, 32) and negative.shape == (1, 32) and not positive.requires_grad
    assert SemanticPrompt(encoder, loaded, base, negative_trained=False).text_features(torch.zeros(1, 3, 8, 8))[1] is None

    # Text-slot feature validation used by the sampler.
    checked_text_features(torch.zeros(2, 4096))
    for bad in (torch.zeros(4096), torch.zeros(0, 4096), torch.zeros(2, 8), torch.full((1, 4096), float('nan'))):
        assert rejects(checked_text_features, bad)

    # Guidance negative sources.
    prompt = SRQualityGuidance(scale=1., negative='prompt', lr_blur_sigma=0.)
    assert prompt.needs_negative_lr and not prompt.degrades_lr and prompt.uses_negative_text
    degraded = SRQualityGuidance(scale=1., negative='degraded_lr')
    assert degraded.needs_negative_lr and degraded.degrades_lr and not degraded.uses_negative_text
    both = SRQualityGuidance(scale=1., negative='degraded_lr_prompt')
    assert both.needs_negative_lr and both.degrades_lr and both.uses_negative_text
    calibrator = SRQualityGuidance(scale=1., negative='calibrator')
    assert not calibrator.needs_negative_lr and calibrator.needs_negative_calibrator
    assert rejects(SRQualityGuidance, scale=1., negative='degraded_lr_prompt', lr_blur_sigma=0., lr_downscale=1.)
    print('sr_semantic_prompt checks passed')


if __name__ == '__main__':
    main()
