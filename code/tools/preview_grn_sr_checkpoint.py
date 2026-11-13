#!/usr/bin/env python3
"""Create a small deterministic visual preview for a GRN-SR checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from tools.evaluate_grn_sr_alignvar import BENCHMARKS, benchmark_pairs
from tools.grn_pipeline import GRNPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--protocol_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--model", default="GRN2b")
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    parser.add_argument("--samples_per_benchmark", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.samples_per_benchmark < 1:
        parser.error("--samples_per_benchmark must be positive")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    unknown = sorted(set(args.benchmarks) - set(BENCHMARKS))
    if unknown:
        parser.error(f"unknown benchmarks: {unknown}")
    return args


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    count = min(length, count)
    return [min(length - 1, int((index + 0.5) * length / count)) for index in range(count)]


def rgb_to_y(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)
    return (
        16.0
        + (65.481 * image[..., 0] + 128.553 * image[..., 1] + 24.966 * image[..., 2])
        / 255.0
    )


def psnr_y(prediction: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((rgb_to_y(prediction) - rgb_to_y(target)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((255.0**2) / mse)


def resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image)
    return image.resize(size, resampling.BICUBIC)


def labeled_panel(image: Image.Image, label: str, size: int = 256) -> Image.Image:
    panel = Image.new("RGB", (size, size + 28), "white")
    panel.paste(resize(image.convert("RGB"), (size, size)), (0, 28))
    ImageDraw.Draw(panel).text((6, 7), label, fill="black")
    return panel


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("GRN-SR preview requires CUDA")

    device = torch.device("cuda:0")
    pipeline = GRNPipeline.from_pretrained(
        model_path=str(args.checkpoint.resolve()),
        vae_path=str(args.vae.resolve()),
        task="SR",
        model=args.model,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    pipeline.model.eval()
    pipeline.vae.eval()

    records = []
    sheet_rows = []
    started = time.time()
    for benchmark in args.benchmarks:
        pairs = benchmark_pairs(args.protocol_root.resolve(), BENCHMARKS[benchmark])
        indices = evenly_spaced_indices(len(pairs), args.samples_per_benchmark)
        benchmark_root = args.output_root / benchmark
        benchmark_root.mkdir(parents=True, exist_ok=True)
        for index in indices:
            hr_path, lr_path = pairs[index]
            with Image.open(lr_path) as handle:
                lr = handle.convert("RGB")
            with Image.open(hr_path) as handle:
                gt = handle.convert("RGB")
            if lr.size != (128, 128) or gt.size != (512, 512):
                raise ValueError(
                    f"Unexpected pair sizes for {lr_path.name}: LR={lr.size}, GT={gt.size}"
                )

            sample_started = time.time()
            prediction = pipeline(
                lr_image=lr,
                guidance_scale=1.0,
                temperature=1.0,
                complexity_aware_Tmin=args.steps,
                complexity_aware_Tmax=args.steps,
                complexity_aware_k=0,
                complexity_aware_b=args.steps,
                complexity_aware_wp=0,
                snr_shift=1.0,
                seed=args.seed + index,
                output_type="numpy",
                top_k=1,
                top_p=0.75,
            )[0]
            sr = Image.fromarray(prediction, mode="RGB")
            bicubic = resize(lr, (512, 512))

            stem = hr_path.stem
            sr.save(benchmark_root / f"{stem}_sr.png")
            bicubic.save(benchmark_root / f"{stem}_bicubic.png")
            gt.save(benchmark_root / f"{stem}_gt.png")
            lr.save(benchmark_root / f"{stem}_lr.png")

            sr_psnr = psnr_y(np.asarray(sr), np.asarray(gt))
            bicubic_psnr = psnr_y(np.asarray(bicubic), np.asarray(gt))
            record = {
                "benchmark": benchmark,
                "name": hr_path.name,
                "index": index,
                "seed": args.seed + index,
                "steps": args.steps,
                "psnr_y_sr": sr_psnr,
                "psnr_y_bicubic": bicubic_psnr,
                "seconds": time.time() - sample_started,
            }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)

            row = Image.new("RGB", (256 * 3, 256 + 52), "white")
            title = f"{benchmark} / {hr_path.name}"
            ImageDraw.Draw(row).text((6, 4), title, fill="black")
            panels = (
                labeled_panel(bicubic, f"LR bicubic  PSNR-Y {bicubic_psnr:.2f}"),
                labeled_panel(sr, f"GRN-SR {args.steps} steps  PSNR-Y {sr_psnr:.2f}"),
                labeled_panel(gt, "Ground truth"),
            )
            for panel_index, panel in enumerate(panels):
                row.paste(panel, (panel_index * 256, 24))
            sheet_rows.append(row)

    sheet = Image.new("RGB", (256 * 3, sum(row.height for row in sheet_rows)), "white")
    y = 0
    for row in sheet_rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(args.output_root / "comparison_sheet.png")

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "steps": args.steps,
        "samples": records,
        "mean_psnr_y_sr": float(np.mean([item["psnr_y_sr"] for item in records])),
        "mean_psnr_y_bicubic": float(
            np.mean([item["psnr_y_bicubic"] for item in records])
        ),
        "total_seconds": time.time() - started,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
