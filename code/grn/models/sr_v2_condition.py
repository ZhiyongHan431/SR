"""Small, optional S2 condition modules; no backbone or image preprocessing.

The frozen semantic encoder is owned by the caller. Only its detached patch
features are reused across rollout forwards; these trainable modules are not
cached. GRN creates them after its common initialization, before FSDP wrapping.
"""
import torch
from torch import nn
from torch.nn import functional as F


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _finite_image_features(value, name, channels):
    if (not torch.is_tensor(value) or value.ndim != 4
            or value.shape[0] < 1 or value.shape[1] != channels
            or min(value.shape[-2:]) < 1 or not value.is_floating_point()):
        raise ValueError(f'{name} must be floating [B,{channels},H,W], B,H,W > 0')
    if not torch.isfinite(value).all().item():
        raise ValueError(f'{name} must be finite')


class SRSemanticFusion(nn.Module):
    """Zero-initialized semantic residual, preserving the existing LR grid."""

    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        _positive_int(embed_dim, 'embed_dim')
        _positive_int(hidden_dim, 'hidden_dim')
        self.embed_dim = embed_dim
        self.input_proj = nn.Conv2d(384, hidden_dim, 1, device='cpu', dtype=torch.float32)
        self.output_proj = nn.Conv2d(hidden_dim, embed_dim, 1, device='cpu', dtype=torch.float32)
        # Explicit initialization also works if callers disable builtin resets.
        nn.init.trunc_normal_(self.input_proj.weight, std=.02)
        nn.init.zeros_(self.input_proj.bias)
        self.reset_output()

    @torch.no_grad()
    def reset_output(self):
        self.output_proj.weight.zero_()
        self.output_proj.bias.zero_()

    def forward(self, structure_features, semantic_features):
        _finite_image_features(structure_features, 'structure_features', self.embed_dim)
        _finite_image_features(semantic_features, 'sr_semantic_features', 384)
        if semantic_features.shape[-2:] != (16, 16):
            raise ValueError('sr_semantic_features must have a 16x16 patch grid')
        if semantic_features.shape[0] != structure_features.shape[0]:
            raise ValueError('semantic and structure batch sizes must match')
        if (semantic_features.device != structure_features.device
                or structure_features.device != self.input_proj.weight.device):
            raise ValueError('semantic, structure and fusion module devices must match')
        semantic = semantic_features.to(dtype=self.input_proj.weight.dtype)
        semantic = F.interpolate(semantic, size=structure_features.shape[-2:],
                                 mode='bilinear', align_corners=False)
        residual = self.output_proj(F.silu(self.input_proj(semantic)))
        # Do not promote a BF16 prefix merely because the module uses FP32 weights.
        residual = residual.to(dtype=structure_features.dtype)
        fused = structure_features + residual
        if not torch.isfinite(fused).all().item():
            raise ValueError('fused semantic condition must be finite')
        return fused


class SRCoarseHead(nn.Module):
    """Auxiliary RGB prediction from the unmodified 64-channel LR latent."""

    def __init__(self):
        super().__init__()
        self.input_proj = nn.Conv2d(64, 64, 3, padding=1, device='cpu', dtype=torch.float32)
        self.output_proj = nn.Conv2d(64, 3, 1, device='cpu', dtype=torch.float32)
        for module in (self.input_proj, self.output_proj):
            nn.init.trunc_normal_(module.weight, std=.02)
            nn.init.zeros_(module.bias)

    def forward(self, condition_latent):
        _finite_image_features(condition_latent, 'condition_latent', 64)
        if condition_latent.device != self.input_proj.weight.device:
            raise ValueError('condition latent and coarse head devices must match')
        latent = condition_latent.to(dtype=self.input_proj.weight.dtype)
        return self.output_proj(F.silu(self.input_proj(latent)))
