"""Data-only adapters for the official VARSR model.

Two mutually exclusive training modes are exposed:

* no_cfg: LSDIR + FFHQ only, every sample has label_B=0.
* synthetic_cfg: the same positive corpus plus an on-the-fly low-quality
  target subset with label_B=1.

The model, tokenizer, losses, and autoregressive implementation are untouched.
"""

from __future__ import annotations

import io
import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFile, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .realesrgan import RealESRGAN_degradation


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and ".ipynb_checkpoints" not in path.parts
    )


def read_manifest(manifest_path: str | Path) -> list[Path]:
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = PROJECT_ROOT / manifest
    if not manifest.is_file():
        raise FileNotFoundError(f"Training manifest does not exist: {manifest}")

    paths: list[Path] = []
    for raw_line in manifest.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        paths.append(path)

    if not paths:
        raise RuntimeError(f"Training manifest is empty: {manifest}")
    for path in paths[:100]:
        if not path.is_file():
            raise FileNotFoundError(f"Manifest path is unavailable: {path}")
    return paths


def official_hr_preprocess(image: Image.Image, image_size: int) -> Image.Image:
    """VARSR paper preprocessing: resize to 1.25x, center crop, random flip."""
    transform = transforms.Compose(
        [
            transforms.Resize(
                round(1.25 * image_size),
                interpolation=InterpolationMode.LANCZOS,
            ),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )
    return transform(image)


def _jpeg(image: Image.Image) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=random.randint(15, 60), subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def _noise(image: Image.Image) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    sigma = random.uniform(8.0, 30.0)
    if random.random() < 0.25:
        noise = np.random.normal(0.0, sigma, (*array.shape[:2], 1))
    else:
        noise = np.random.normal(0.0, sigma, array.shape)
    return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="RGB")


def _ringing(image: Image.Image) -> Image.Image:
    width, height = image.size
    scale = random.uniform(0.35, 0.70)
    small = image.resize(
        (max(16, round(width * scale)), max(16, round(height * scale))),
        Image.Resampling.BICUBIC,
    )
    restored = small.resize((width, height), Image.Resampling.LANCZOS)
    return restored.filter(
        ImageFilter.UnsharpMask(
            radius=random.uniform(1.5, 3.0),
            percent=random.randint(180, 380),
            threshold=0,
        )
    )


def _color_exposure(image: Image.Image) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.55, 1.45))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.55, 1.55))
    image = ImageEnhance.Color(image).enhance(random.uniform(0.35, 1.65))
    if random.random() < 0.5:
        gamma = random.uniform(0.55, 1.75)
        lut = [round(255 * ((value / 255) ** gamma)) for value in range(256)]
        image = image.point(lut * 3)
    return image


def synthesize_low_quality_target(image: Image.Image) -> Image.Image:
    """Create a visibly low-quality target for the image-based CFG branch.

    This is intentionally an ablation, not a claim to reproduce VARSR's private
    50K IQA/IAA negative subset.
    """
    operations = [
        lambda x: x.filter(ImageFilter.GaussianBlur(random.uniform(0.8, 3.2))),
        _jpeg,
        _noise,
        lambda x: x.filter(
            ImageFilter.UnsharpMask(
                radius=random.uniform(1.0, 3.0),
                percent=random.randint(220, 500),
                threshold=0,
            )
        ),
        _ringing,
        _color_exposure,
    ]
    for operation in random.sample(operations, k=random.randint(2, 4)):
        image = operation(image)
    return image.convert("RGB")


class LSDIRFFHQDataset(Dataset):
    """LSDIR + FFHQ-10K with official VARSR preprocessing and Real-ESRGAN."""

    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 512,
        degradation_config: str | Path = "dataloader/params_realesrgan.yml",
        cfg_mode: str = "no_cfg",
        synthetic_negative_fraction: float = 0.02,
        synthetic_negative_seed: int = 2025,
    ) -> None:
        if cfg_mode not in {"no_cfg", "synthetic_cfg"}:
            raise ValueError(f"Unknown cfg_mode: {cfg_mode}")
        if not 0 <= synthetic_negative_fraction < 0.5:
            raise ValueError("synthetic_negative_fraction must be in [0, 0.5)")

        self.positive_paths = read_manifest(manifest_path)
        self.image_size = image_size
        self.cfg_mode = cfg_mode
        config_path = Path(degradation_config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        self.degradation = RealESRGAN_degradation(str(config_path), device="cpu")

        self.negative_paths: list[Path] = []
        if cfg_mode == "synthetic_cfg" and synthetic_negative_fraction > 0:
            positive_count = len(self.positive_paths)
            negative_count = math.ceil(
                positive_count
                * synthetic_negative_fraction
                / (1.0 - synthetic_negative_fraction)
            )
            selector = random.Random(synthetic_negative_seed)
            indices = list(range(positive_count))
            selector.shuffle(indices)
            self.negative_paths = [
                self.positive_paths[indices[index % positive_count]]
                for index in range(negative_count)
            ]

        total = len(self)
        actual_fraction = len(self.negative_paths) / total if total else 0.0
        print(
            "[LSDIRFFHQDataset] "
            f"mode={cfg_mode}, positives={len(self.positive_paths)}, "
            f"synthetic_negatives={len(self.negative_paths)}, "
            f"negative_fraction={actual_fraction:.4f}, HR={image_size}"
        )

    def __len__(self) -> int:
        return len(self.positive_paths) + len(self.negative_paths)

    def __getitem__(self, index: int) -> dict:
        is_negative = index >= len(self.positive_paths)
        path = (
            self.negative_paths[index - len(self.positive_paths)]
            if is_negative
            else self.positive_paths[index]
        )
        with Image.open(path) as handle:
            target = official_hr_preprocess(handle.convert("RGB"), self.image_size)
        if is_negative:
            target = synthesize_low_quality_target(target)

        target_tensor, condition_tensor = self.degradation.degrade_process(
            np.asarray(target, dtype=np.float32) / 255.0,
            resize_bak=True,
        )
        return {
            "pixel_values": target_tensor.squeeze(0).mul(2).sub(1),
            "conditioning_pixel_values": condition_tensor.squeeze(0).mul(2).sub(1),
            "label_B": int(is_negative),
            "path": str(path),
            "cfg_mode": self.cfg_mode,
        }


def benchmark_dirs(root: Path) -> tuple[Path, Path]:
    candidates: Sequence[tuple[str, str]] = (("gt", "lq"), ("test_HR", "test_LR"))
    for hr_name, lr_name in candidates:
        hr_dir, lr_dir = root / hr_name, root / lr_name
        if hr_dir.is_dir() and lr_dir.is_dir():
            return hr_dir, lr_dir
    raise FileNotFoundError(
        f"Expected gt/lq or test_HR/test_LR under benchmark root: {root}"
    )


class AlignVARPairDataset(Dataset):
    """Fixed AlignVAR 512x512 HR / 128x128 LR benchmark pairs."""

    def __init__(self, root: str | Path, image_size: int = 512) -> None:
        self.root = Path(root)
        if not self.root.is_absolute():
            self.root = PROJECT_ROOT / self.root
        self.hr_dir, self.lr_dir = benchmark_dirs(self.root)
        hr_paths = image_files(self.hr_dir)
        lr_by_name = {path.name: path for path in image_files(self.lr_dir)}
        self.pairs = [(hr, lr_by_name[hr.name]) for hr in hr_paths if hr.name in lr_by_name]
        if len(self.pairs) != len(hr_paths):
            missing = [path.name for path in hr_paths if path.name not in lr_by_name]
            raise RuntimeError(
                f"{len(missing)} HR images have no same-name LR pair; first={missing[0]}"
            )
        if not self.pairs:
            raise RuntimeError(f"No benchmark pairs found under {self.root}")
        self.image_size = image_size
        print(f"[AlignVARPairDataset] root={self.root}, pairs={len(self.pairs)}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        hr_path, lr_path = self.pairs[index]
        with Image.open(hr_path) as handle:
            target = TF.to_tensor(handle.convert("RGB"))
        with Image.open(lr_path) as handle:
            native_lr = TF.to_tensor(handle.convert("RGB"))

        expected_hr = (self.image_size, self.image_size)
        expected_lr = (self.image_size // 4, self.image_size // 4)
        if tuple(target.shape[-2:]) != expected_hr:
            raise ValueError(f"{hr_path} has shape {tuple(target.shape[-2:])}, expected {expected_hr}")
        if tuple(native_lr.shape[-2:]) != expected_lr:
            raise ValueError(f"{lr_path} has shape {tuple(native_lr.shape[-2:])}, expected {expected_lr}")

        condition = F.interpolate(
            native_lr.unsqueeze(0),
            size=expected_hr,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0).clamp_(0, 1)
        return {
            "pixel_values": target.mul(2).sub(1),
            "conditioning_pixel_values": condition.mul(2).sub(1),
            "native_lr_pixel_values": native_lr,
            "label_B": 0,
            "path": str(hr_path),
            "lr_path": str(lr_path),
        }

