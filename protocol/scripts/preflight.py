#!/usr/bin/env python3
"""Fail-fast checks for the offline four-GPU VARSR reproduction bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
EXPECTED_WEIGHT_HASHES = {
    "checkpoints/VARSR_C2I.pth": "e6fc619171c810be79b5c3cae671c883b2ded21b912a093419a7ef3f79140984",
    "checkpoints/VQVAE.pth": "b4b60aaa47e7cff7fdebb479521138efbbd968faa893fe0d4f8c703fd49ddf45",
}
IMPORTS = [
    "torch",
    "torchvision",
    "cv2",
    "numpy",
    "PIL",
    "yaml",
    "tap",
    "wandb",
    "transformers",
    "timm",
    "fairscale",
    "pyiqa",
    "skimage",
    "scipy",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--required_gpus", type=int, default=4)
    parser.add_argument("--allow_fewer_gpus", action="store_true")
    parser.add_argument("--skip_weight_hash", action="store_true")
    parser.add_argument("--skip_data_sample", action="store_true")
    parser.add_argument("--skip_metric_init", action="store_true")
    args = parser.parse_args()

    print("[1/6] Python dependencies")
    for name in IMPORTS:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "installed")
        print(f"  {name}: {version}")

    print("[2/6] CUDA devices")
    device_count = torch.cuda.device_count()
    print(f"  torch={torch.__version__}, cuda_runtime={torch.version.cuda}")
    print(f"  visible CUDA devices={device_count}")
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        print(
            f"  cuda:{index}: {properties.name}, "
            f"{properties.total_memory / 1024**3:.1f} GiB"
        )
    if device_count < args.required_gpus and not args.allow_fewer_gpus:
        raise RuntimeError(
            f"Expected at least {args.required_gpus} visible GPUs, found {device_count}. "
            "Use --allow_fewer_gpus only for inspection, not the four-GPU run."
        )

    print("[3/6] Official pretrained weights")
    for relative, expected_hash in EXPECTED_WEIGHT_HASHES.items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"  {relative}: {path.stat().st_size} bytes")
        if not args.skip_weight_hash:
            actual_hash = sha256(path)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"{relative}: SHA-256 mismatch; "
                    f"expected={expected_hash}, actual={actual_hash}"
                )
            print(f"    SHA-256 OK: {actual_hash}")
    critical_manifest = PROJECT_ROOT / "config/official_critical_files.sha256"
    for line in critical_manifest.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        expected_hash, relative = line.split(maxsplit=1)
        path = PROJECT_ROOT / relative
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Official critical source changed: {relative}; "
                f"expected={expected_hash}, actual={actual_hash}"
            )
    print("  official model/loss/trainer/degradation source hashes: OK")

    print("[4/6] Training and benchmark manifests")
    manifest = PROJECT_ROOT / "data/manifests/lsdir_ffhq10k.txt"
    lines = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
    if len(lines) != 94991:
        raise RuntimeError(f"Expected 94991 manifest entries, found {len(lines)}")
    missing = [line for line in lines if not (PROJECT_ROOT / line).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing manifest files, first={missing[0]}")
    report = json.loads(
        (PROJECT_ROOT / "data/manifests/data_report.json").read_text()
    )
    expected_pairs = {"DIV2K-Val": 3000, "RealSR": 100, "DRealSR": 93}
    found_pairs = {
        name: info["pairs"] for name, info in report["benchmarks"].items()
    }
    if found_pairs != expected_pairs:
        raise RuntimeError(
            f"Benchmark report mismatch: expected={expected_pairs}, found={found_pairs}"
        )
    print(f"  training entries={len(lines)}, benchmarks={found_pairs}")

    print("[5/6] Bundled metric cache")
    cache_root = PROJECT_ROOT / "offline_cache"
    cache_manifest = json.loads((cache_root / "manifest.json").read_text())
    for relative in cache_manifest["files"]:
        path = cache_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"  {relative}: {path.stat().st_size} bytes")
    if not args.skip_metric_init:
        import gc
        import pyiqa

        metric_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        metric_specs = [
            ("psnry", {"crop_border": 0}),
            ("ssim", {"crop_border": 0}),
            ("lpips", {}),
            ("dists", {}),
            ("maniqa", {}),
            ("musiq", {}),
            ("clipiqa", {}),
            ("fid", {}),
        ]
        for metric_name, kwargs in metric_specs:
            metric = pyiqa.create_metric(metric_name, device=metric_device, **kwargs)
            del metric
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  initialized offline: {metric_name}")

    print("[6/6] Data-mode smoke test")
    if args.skip_data_sample:
        print("  skipped by request")
    else:
        from dataloader.alignvar_repro import LSDIRFFHQDataset

        common = {
            "manifest_path": "data/manifests/lsdir_ffhq10k.txt",
            "image_size": 512,
            "degradation_config": "dataloader/params_realesrgan.yml",
        }
        no_cfg = LSDIRFFHQDataset(cfg_mode="no_cfg", **common)
        sample = no_cfg[0]
        assert sample["label_B"] == 0
        assert tuple(sample["pixel_values"].shape) == (3, 512, 512)
        assert tuple(sample["conditioning_pixel_values"].shape) == (3, 512, 512)
        assert len(no_cfg) == 94991
        print("  no_cfg: label=0 and HR/condition shapes OK")

        synthetic = LSDIRFFHQDataset(
            cfg_mode="synthetic_cfg",
            synthetic_negative_fraction=0.02,
            synthetic_negative_seed=2025,
            **common,
        )
        negative = synthetic[len(synthetic.positive_paths)]
        assert negative["label_B"] == 1
        assert tuple(negative["pixel_values"].shape) == (3, 512, 512)
        assert tuple(negative["conditioning_pixel_values"].shape) == (3, 512, 512)
        actual_fraction = len(synthetic.negative_paths) / len(synthetic)
        if abs(actual_fraction - 0.02) > 1e-4:
            raise RuntimeError(f"Synthetic negative fraction mismatch: {actual_fraction}")
        print(
            f"  synthetic_cfg: label=1 and shapes OK, "
            f"fraction={actual_fraction:.6f}"
        )

    print("OK: offline VARSR reproduction bundle passed preflight")


if __name__ == "__main__":
    main()
