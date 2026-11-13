"""Offline frozen DINOv2-S/14 semantic features, external to the GRN backbone.

Only the explicitly audited no-register official S/14 checkpoint is accepted.
Construction is CPU/FP32 and preserves CPU RNG. Callers own device placement:
``encoder.to(device=device)``; converting parameters to BF16/FP16 is rejected.
Forward accepts finite floating RGB [B,3,H,W] in [-1,1] and returns ordinary
detached FP32 [B,384,16,16] tensors, even inside caller inference_mode/autocast.
"Normalized" patches use the pretrained final LayerNorm, not extra L2 norm.

Image preprocessing is an SR project choice: whole-image bicubic resize224,
antialias=True, clamp [0,1], ImageNet normalize; no classification center crop.
Position interpolation separately matches Meta's no-register algorithm with
offset=0.1, antialias=False. Timm's default position resampler is NOT used.

Position algorithm attribution: Copyright (c) Meta Platforms, Inc. and
affiliates, Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
Official revision 7764ea0f912e53c92e82eb78a2a1631e92725fc8:
https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/models/vision_transformer.py
The official LICENSE is retained beside the audited asset manifest. Strict
weight loading and position-algorithm parity do not prove full-network output
equivalence between the local timm implementation and Meta's implementation.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
from pathlib import Path
from grn.utils_t2iv.s5_bundle_paths import resolve_path, bundle_root
import re

import torch
from torch import nn
from torch.nn import functional as F


OFFICIAL_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
OFFICIAL_WEIGHT_NAME = "dinov2_vits14_pretrain.pth"
OFFICIAL_WEIGHT_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
OFFICIAL_WEIGHT_BYTES = 88283115
TIMM_MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
AUDITED_TIMM_VERSION = "1.0.27"
SEMANTIC_CHANNELS = 384
SEMANTIC_GRID_SIZE = 16


def _local_file(path: str | Path) -> Path:
    value = str(path)
    if not value or "://" in value:
        raise ValueError("Expected an explicit local file path, not a URL")
    result = resolve_path(value)
    if not result.is_file():
        raise FileNotFoundError(f"Required offline file does not exist: {result}")
    return result


def _verified_bytes(path: str | Path, expected_sha256: str, *, expected_bytes: int | None = None) -> bytes:
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("Expected SHA256 must be an explicit lowercase 64-hex digest")
    resolved = _local_file(path)
    if expected_bytes is not None and resolved.stat().st_size != expected_bytes:
        raise ValueError(f"Offline file size mismatch: {resolved}")
    # Validate and deserialize the very same bytes; do not reopen a checked path.
    data = resolved.read_bytes()
    if expected_bytes is not None and len(data) != expected_bytes:
        raise ValueError(f"Offline file changed size during read: {resolved}")
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"Offline file SHA256 mismatch: {resolved}")
    return data


def _read_manifest(manifest_path: str | Path, expected_sha256: str) -> tuple[Path, dict]:
    manifest_file = _local_file(manifest_path)
    manifest = json.loads(_verified_bytes(manifest_file, expected_sha256))
    if not isinstance(manifest, dict) or manifest.get("status") != "passed":
        raise ValueError("Asset manifest must describe passed validation")
    if manifest.get("official_revision") != OFFICIAL_REVISION:
        raise ValueError("Unsupported official DINOv2 revision in asset manifest")
    try:
        entry = manifest["files"][OFFICIAL_WEIGHT_NAME]
    except (KeyError, TypeError) as exc:
        raise ValueError("Manifest is missing the approved DINOv2-S/14 weight entry") from exc
    if not isinstance(entry, dict):
        raise ValueError("Manifest weight entry must be an object")
    if entry.get("sha256") != OFFICIAL_WEIGHT_SHA256 or entry.get("bytes") != OFFICIAL_WEIGHT_BYTES:
        raise ValueError("Manifest weight metadata does not match the fixed official S/14 asset")
    return manifest_file, manifest


def _verified_timm_factory(manifest: dict):
    import timm

    if timm.__version__ != AUDITED_TIMM_VERSION:
        raise RuntimeError(f"Expected audited timm {AUDITED_TIMM_VERSION}, got {timm.__version__}")
    try:
        source_hashes = manifest["cpu_validation"]["timm_source_files"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Manifest is missing audited local timm source hashes") from exc
    if not isinstance(source_hashes, dict) or len(source_hashes) != 2:
        raise ValueError("Expected exactly two audited timm source entries")
    pinned = {str(_local_file(path)): digest for path, digest in source_hashes.items()}
    for name in ("timm.models.vision_transformer", "timm.layers.pos_embed"):
        module = importlib.import_module(name)
        actual_path = Path(module.__file__).resolve(strict=True)
        if not actual_path.is_relative_to(bundle_root()):
            raise RuntimeError(f"Imported timm is outside the S5 bundle: {actual_path}")
        expected = pinned.get(str(actual_path))
        if expected is None:
            raise RuntimeError(f"Actually imported timm source was not audited: {actual_path}")
        _verified_bytes(actual_path, expected)
    return timm.create_model


def _load_audited_backbone(manifest_path: str | Path, expected_sha256: str) -> tuple[nn.Module, dict]:
    manifest_file, manifest = _read_manifest(manifest_path, expected_sha256)
    weight_path = _local_file(manifest["files"][OFFICIAL_WEIGHT_NAME]["path"])
    weight_bytes = _verified_bytes(weight_path, OFFICIAL_WEIGHT_SHA256, expected_bytes=OFFICIAL_WEIGHT_BYTES)
    create_model = _verified_timm_factory(manifest)
    with torch.inference_mode(False), torch.no_grad(), torch.device("cpu"), torch.random.fork_rng(devices=[]):
        backbone = create_model(TIMM_MODEL_NAME, pretrained=False, device="cpu", dtype=torch.float32)
        state = torch.load(io.BytesIO(weight_bytes), map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or len(state) != 175:
            raise ValueError("Unexpected official state dictionary")
        if any(not isinstance(t, torch.Tensor) or t.dtype != torch.float32 or not torch.isfinite(t).all()
               for t in state.values()):
            raise ValueError("Official weights must contain only finite FP32 tensors")
        if tuple(state.get("mask_token", torch.empty(0)).shape) != (1, SEMANTIC_CHANNELS):
            raise ValueError("Unexpected official mask_token shape")
        state.pop("mask_token")  # Unused without masked input; no other conversion.
        backbone.load_state_dict(state, strict=True)
    provenance = {
        "manifest": str(manifest_file), "manifest_sha256": expected_sha256,
        "weights": str(weight_path), "weight_sha256": OFFICIAL_WEIGHT_SHA256,
        "official_revision": OFFICIAL_REVISION, "timm_version": AUDITED_TIMM_VERSION,
        "torch_version": str(torch.__version__), "full_network_official_parity_tested": False,
    }
    return backbone, provenance


def official_no_reg_position(pos_embed: torch.Tensor, height: int, width: int,
                             token_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Meta's offset-0.1 bicubic position interpolation, with no antialias.

    Spatial axes use row-major height,width order. The unchanged square-grid
    branch intentionally returns the original dtype, just like Meta's method.
    """
    if height <= 0 or width <= 0 or height % 14 or width % 14:
        raise ValueError("Position target must have positive multiples-of-14 sides")
    if pos_embed.ndim != 3 or pos_embed.shape[0] != 1 or pos_embed.shape[1] <= 1:
        raise ValueError("Position embedding must be [1,1+N,D]")
    count = pos_embed.shape[1] - 1
    source_side = math.isqrt(count)
    if source_side * source_side != count:
        raise ValueError("Source patch-position grid must be square")
    rows, cols = height // 14, width // 14
    if rows * cols == count and height == width:
        return pos_embed
    grid = pos_embed[:, 1:].float().reshape(1, source_side, source_side, -1).permute(0, 3, 1, 2)
    resized = F.interpolate(grid, scale_factor=((rows + 0.1) / source_side, (cols + 0.1) / source_side),
                            mode="bicubic", align_corners=False, antialias=False)
    if tuple(resized.shape[-2:]) != (rows, cols):
        raise RuntimeError("Official position interpolation produced an unexpected grid")
    patches = resized.permute(0, 2, 3, 1).reshape(1, rows * cols, -1)
    return torch.cat((pos_embed[:, :1].float(), patches), dim=1).to(token_dtype)


class FrozenDinoV2SemanticEncoder(nn.Module):
    """Audited local encoder, permanently frozen/eval and external to FSDP GRN.

    Device placement is explicit. The caller should move with ``.to(device=...)``
    only and should not include this encoder in a trainable optimizer or wrap it
    with mixed-precision FSDP. Input/device/dtype mismatches fail closed.
    """

    def __init__(self, manifest_path: str | Path, expected_manifest_sha256: str) -> None:
        super().__init__()
        with torch.inference_mode(False), torch.device("cpu"), torch.random.fork_rng(devices=[]):
            self.backbone, self.provenance = _load_audited_backbone(manifest_path, expected_manifest_sha256)
        model = self.backbone
        if not (model.embed_dim == SEMANTIC_CHANNELS and len(model.blocks) == 12
                and tuple(model.patch_embed.patch_size) == (14, 14)
                and tuple(model.pos_embed.shape) == (1, 1370, SEMANTIC_CHANNELS)
                and model.num_prefix_tokens == 1 and model.reg_token is None
                and not model.no_embed_class and isinstance(model.patch_embed.norm, nn.Identity)
                and isinstance(model.norm_pre, nn.Identity)):
            raise RuntimeError("Model differs from audited no-register DINOv2-S/14 configuration")
        with torch.inference_mode(False):
            self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32,
                                                         device="cpu").reshape(1, 3, 1, 1))
            self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32,
                                                        device="cpu").reshape(1, 3, 1, 1))
        self.train(False)
        self._check_runtime(torch.device("cpu"))

    def train(self, mode: bool = True) -> "FrozenDinoV2SemanticEncoder":
        super().train(False)
        if hasattr(self, "backbone"):
            self.backbone.requires_grad_(False)
        return self

    def _check_runtime(self, device: torch.device) -> None:
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Semantic encoder supports only CPU or CUDA tensors")
        for name, tensor in list(self.named_parameters()) + list(self.named_buffers()):
            if tensor.device != device:
                raise ValueError(f"Semantic encoder/input device mismatch for {name}: {tensor.device} vs {device}")
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                raise ValueError(f"Semantic encoder requires FP32 parameters/buffers: {name} has {tensor.dtype}")
            if tensor.is_inference():
                raise ValueError(f"Semantic encoder parameter/buffer must be an ordinary tensor: {name}")

    def _preprocess(self, lr: torch.Tensor) -> torch.Tensor:
        if lr.ndim != 4 or lr.shape[0] < 1 or lr.shape[1] != 3 or min(lr.shape[-2:]) < 1:
            raise ValueError("Expected nonempty RGB [B,3,H,W]")
        if not lr.is_floating_point():
            raise ValueError("LR input must be floating RGB in [-1,1]")
        if not torch.isfinite(lr).all() or lr.min() < -1 or lr.max() > 1:
            raise ValueError("LR input must be finite and within [-1,1]")
        rgb = (lr.float() + 1.0) * 0.5
        rgb = F.interpolate(rgb, size=(224, 224), mode="bicubic", align_corners=False,
                            antialias=True).clamp(0.0, 1.0)
        return (rgb - self.rgb_mean) / self.rgb_std

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        if not isinstance(lr, torch.Tensor):
            raise TypeError("Semantic encoder input must be a Tensor")
        self.train(False)
        self._check_runtime(lr.device)
        # Override external inference_mode as well as AMP. Plain no_grad outputs
        # are required because the downstream semantic adapter is trainable.
        with torch.inference_mode(False), torch.no_grad(), torch.autocast(device_type=lr.device.type, enabled=False):
            rgb = self._preprocess(lr)
            model = self.backbone
            grid = model.patch_embed.proj(rgb)
            tokens = model.patch_embed.norm(grid.flatten(2).transpose(1, 2))
            tokens = torch.cat((model.cls_token.expand(lr.shape[0], -1, -1), tokens), dim=1)
            tokens = model.pos_drop(tokens + official_no_reg_position(model.pos_embed, 224, 224, tokens.dtype))
            tokens = model.norm_pre(model.patch_drop(tokens))
            tokens = model.norm(model.blocks(tokens))
            output = tokens[:, 1:].transpose(1, 2).reshape(lr.shape[0], SEMANTIC_CHANNELS,
                                                         SEMANTIC_GRID_SIZE, SEMANTIC_GRID_SIZE).contiguous()
            if not torch.isfinite(output).all():
                raise RuntimeError("Semantic encoder produced nonfinite patch features")
            return output
