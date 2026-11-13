#!/usr/bin/env python3
"""Read-only dependency audit for the offline VARSR reproduction bundle."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


CHECKS = [
    ("torch", "torch", True),
    ("torchvision", "torchvision", True),
    ("opencv-python-headless", "cv2", True),
    ("numpy", "numpy", True),
    ("Pillow", "PIL", True),
    ("PyYAML", "yaml", True),
    ("typed-argument-parser", "tap", True),
    ("wandb", "wandb", True),
    ("transformers", "transformers", True),
    ("huggingface-hub", "huggingface_hub", True),
    ("timm", "timm", True),
    ("fairscale", "fairscale", True),
    ("pyiqa", "pyiqa", True),
    ("scikit-image", "skimage", True),
    ("scipy", "scipy", True),
    ("pytz", "pytz", True),
    ("einops", "einops", True),
    ("imageio", "imageio", True),
    ("lmdb", "lmdb", True),
    ("pandas", "pandas", True),
    ("requests", "requests", True),
    ("safetensors", "safetensors", True),
    ("tqdm", "tqdm", True),
    # trainer.py imports test_varsr.py, which imports accelerate at module load.
    ("accelerate", "accelerate", True),
    ("diffusers", "diffusers", False),
]


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def compact_error(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "unknown import failure"


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    print("=" * 78)
    print("VARSR offline environment audit (read-only)")
    print("=" * 78)
    print(f"project       : {project_root}")
    print(f"python        : {sys.executable}")
    print(f"python version: {platform.python_version()}")
    print(f"platform      : {platform.platform()}")
    print(f"architecture  : {platform.machine()}")
    print(f"PATH torchrun : {shutil.which('torchrun') or 'NOT FOUND'}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")

    missing: list[str] = []
    broken: list[tuple[str, str]] = []
    optional_missing: list[str] = []

    print("\n[1/4] Python imports")
    for distribution, module, required in CHECKS:
        installed_version = distribution_version(distribution)
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    f"import {module} as m; "
                    "print(getattr(m, '__version__', 'imported'))"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        version_text = installed_version or "metadata unavailable"
        label = f"{distribution} (import {module})"
        if probe.returncode == 0:
            imported_version = probe.stdout.strip().splitlines()[-1]
            print(f"  OK      {label:<54} dist={version_text}; import={imported_version}")
        elif installed_version is None:
            status = "MISSING" if required else "OPTIONAL"
            print(f"  {status:<7} {label:<54} not installed")
            (missing if required else optional_missing).append(distribution)
        else:
            error = compact_error(probe.stderr or probe.stdout)
            print(f"  BROKEN  {label:<54} dist={version_text}; {error}")
            broken.append((distribution, error))

    # cv2 can be supplied by mutually overlapping distributions. Report both.
    print("\n[2/4] OpenCV package ownership")
    for name in ("opencv-python", "opencv-python-headless"):
        print(f"  {name:<26}: {distribution_version(name) or 'not installed'}")
    if distribution_version("opencv-python") and distribution_version("opencv-python-headless"):
        print("  WARNING: both OpenCV variants are installed; prefer only headless on this server.")

    print("\n[3/4] CUDA and distributed launcher")
    cuda_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, torch; "
                "print(json.dumps({'torch': torch.__version__, "
                "'cuda_runtime': torch.version.cuda, "
                "'cuda_available': torch.cuda.is_available(), "
                "'device_count': torch.cuda.device_count(), "
                "'devices': [torch.cuda.get_device_name(i) "
                "for i in range(torch.cuda.device_count())]}))"
            ),
        ],
        capture_output=True,
        text=True,
    )
    if cuda_probe.returncode == 0:
        info = json.loads(cuda_probe.stdout.strip().splitlines()[-1])
        print(json.dumps(info, ensure_ascii=False, indent=2))
        if info["device_count"] < 4:
            broken.append(("CUDA", f"only {info['device_count']} visible GPUs; need 4"))
    else:
        error = compact_error(cuda_probe.stderr or cuda_probe.stdout)
        print(f"  CUDA PROBE FAILED: {error}")
        broken.append(("CUDA", error))

    distributed_probe = subprocess.run(
        [sys.executable, "-c", "import torch.distributed.run; print('OK')"],
        capture_output=True,
        text=True,
    )
    if shutil.which("torchrun") and distributed_probe.returncode == 0:
        print("  torchrun and torch.distributed.run: OK")
    else:
        error = compact_error(distributed_probe.stderr or distributed_probe.stdout)
        print(f"  torchrun check failed: {error}")
        broken.append(("torchrun", error))

    print("\n[4/4] pip dependency consistency")
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
    )
    pip_text = (pip_check.stdout + pip_check.stderr).strip()
    print(pip_text or "No output from pip check")
    pip_has_warnings = pip_check.returncode != 0

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if missing:
        print("Required distributions not installed:")
        for name in missing:
            print(f"  - {name}")
    if broken:
        print("Installed/runtime components that are broken or incompatible:")
        for name, error in broken:
            print(f"  - {name}: {error}")
    if optional_missing:
        print("Optional for this fuse=0 workflow (absence does not block training):")
        for name in optional_missing:
            print(f"  - {name}")
    if pip_has_warnings:
        print("NOTE: pip check conflicts are warnings unless a required import above is BROKEN.")
    if not missing and not broken:
        print("PASS: all required imports, CUDA visibility, torchrun and pip check passed.")
        return 0
    print("FAIL: fix the required MISSING/BROKEN entries, then rerun this script.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
