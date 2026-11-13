#!/usr/bin/env python3
"""Build and validate the portable LSDIR/FFHQ and AlignVAR benchmark manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
EXPECTED = {
    "LSDIR": 84991,
    "FFHQ": 10000,
    "DIV2K-Val": 3000,
    "RealSR": 100,
    "DRealSR": 93,
}
BENCHMARKS = {
    "DIV2K-Val": ("data/benchmarks/StableSR_testsets/DIV2K_V2_val", "gt", "lq"),
    "RealSR": ("data/benchmarks/StableSR_testsets/RealSRVal_crop128", "test_HR", "test_LR"),
    "DRealSR": ("data/benchmarks/StableSR_testsets/DrealSRVal_crop128", "test_HR", "test_LR"),
}


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and ".ipynb_checkpoints" not in path.parts
    )


def relative_posix(path: Path, project_root: Path) -> str:
    return path.relative_to(project_root).as_posix()


def validate_benchmark(
    project_root: Path,
    name: str,
    relative_root: str,
    hr_name: str,
    lr_name: str,
    check_shapes: bool,
) -> dict:
    root = project_root / relative_root
    hr_files = image_files(root / hr_name)
    lr_files = image_files(root / lr_name)
    hr_by_name = {path.name: path for path in hr_files}
    lr_by_name = {path.name: path for path in lr_files}
    if set(hr_by_name) != set(lr_by_name):
        only_hr = sorted(set(hr_by_name) - set(lr_by_name))
        only_lr = sorted(set(lr_by_name) - set(hr_by_name))
        raise RuntimeError(
            f"{name}: HR/LR names differ; only_hr={only_hr[:3]}, only_lr={only_lr[:3]}"
        )
    if len(hr_files) != EXPECTED[name]:
        raise RuntimeError(
            f"{name}: expected {EXPECTED[name]} pairs, found {len(hr_files)}"
        )

    shape_errors: list[str] = []
    if check_shapes:
        filenames = sorted(hr_by_name)
        for index, filename in enumerate(filenames, start=1):
            if index == 1 or index % 500 == 0 or index == len(filenames):
                print(
                    f"[data] {name}: shape check {index}/{len(filenames)}",
                    flush=True,
                )
            with Image.open(hr_by_name[filename]) as hr:
                hr_size = hr.size
            with Image.open(lr_by_name[filename]) as lr:
                lr_size = lr.size
            if hr_size != (512, 512) or lr_size != (128, 128):
                shape_errors.append(
                    f"{filename}: HR={hr_size}, LR={lr_size}, expected (512,512)/(128,128)"
                )
                if len(shape_errors) >= 10:
                    break
    if shape_errors:
        raise RuntimeError(f"{name}: invalid dimensions: {shape_errors}")

    return {
        "root": relative_root,
        "hr_dir": hr_name,
        "lr_dir": lr_name,
        "pairs": len(hr_files),
        "expected_hr_size": [512, 512],
        "expected_lr_size": [128, 128],
        "shape_check": check_shapes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--skip_shape_check",
        action="store_true",
        help="Only check counts and names; full PIL header validation is the default.",
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()

    print("[data] scanning LSDIR ...", flush=True)
    lsdir = image_files(project_root / "data/raw/LSDIR")
    print(f"[data] LSDIR: found {len(lsdir)} images", flush=True)
    print("[data] scanning FFHQ_Part1 ...", flush=True)
    ffhq_all = image_files(project_root / "data/raw/FFHQ_Part1")
    print(f"[data] FFHQ_Part1: found {len(ffhq_all)} images", flush=True)
    if len(lsdir) != EXPECTED["LSDIR"]:
        raise RuntimeError(
            f"LSDIR: expected {EXPECTED['LSDIR']} images, found {len(lsdir)}"
        )
    if len(ffhq_all) < EXPECTED["FFHQ"]:
        raise RuntimeError(
            f"FFHQ: expected at least {EXPECTED['FFHQ']} images, found {len(ffhq_all)}"
        )
    ffhq = ffhq_all[: EXPECTED["FFHQ"]]

    manifest_dir = project_root / "data/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "lsdir_ffhq10k.txt"
    training_paths = lsdir + ffhq
    manifest_path.write_text(
        "\n".join(relative_posix(path, project_root) for path in training_paths) + "\n"
    )
    print(
        f"[data] wrote training manifest with {len(training_paths)} entries",
        flush=True,
    )

    benchmark_report = {}
    for index, (name, (root, hr_dir, lr_dir)) in enumerate(
        BENCHMARKS.items(), start=1
    ):
        print(f"[data] benchmark {index}/{len(BENCHMARKS)}: {name}", flush=True)
        benchmark_report[name] = validate_benchmark(
            project_root,
            name,
            root,
            hr_dir,
            lr_dir,
            check_shapes=not args.skip_shape_check,
        )

    report = {
        "training": {
            "LSDIR": len(lsdir),
            "FFHQ_first_sorted": len(ffhq),
            "total_positive": len(training_paths),
            "manifest": relative_posix(manifest_path, project_root),
            "manifest_paths_are_portable": True,
        },
        "benchmarks": benchmark_report,
    }
    report_path = manifest_dir / "data_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"OK: wrote {manifest_path}")
    print(f"OK: wrote {report_path}")


if __name__ == "__main__":
    main()
