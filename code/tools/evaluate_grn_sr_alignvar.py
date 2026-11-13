#!/usr/bin/env python3
"""Generate and score GRN-SR with the fixed VARSR/AlignVAR offline protocol."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.distributed as torch_dist
from PIL import Image

from tools.grn_pipeline import GRNPipeline
from tools.sr_eval_protocol import (
    BENCHMARKS,
    EXPECTED_COUNTS,
    build_generation_signature,
    resolve_benchmarks,
)


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
PAPER_METRICS = (
    "PSNR-Y",
    "SSIM-Y",
    "LPIPS",
    "DISTS",
    "FID",
    "MANIQA",
    "CLIPIQA",
    "MUSIQ",
)
PAPER_TABLE1_VALUES = {
    "VARSR": {
        "DIV2K-Val": (24.41, 0.6189, 0.2985, 0.2167, 28.64, 0.4137, 0.6312, 66.88),
        "RealSR": (26.08, 0.7381, 0.2777, 0.2111, 118.84, 0.4316, 0.5953, 66.65),
        "DRealSR": (28.84, 0.7735, 0.3102, 0.2311, 141.99, 0.4225, 0.6395, 62.66),
    },
    "AlignVAR": {
        "DIV2K-Val": (24.35, 0.6021, 0.2955, 0.2162, 25.71, 0.4665, 0.6754, 67.32),
        "RealSR": (26.11, 0.7125, 0.2871, 0.2123, 138.81, 0.4553, 0.6784, 68.53),
        "DRealSR": (28.54, 0.7682, 0.3184, 0.2340, 145.44, 0.4685, 0.6534, 63.83),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--protocol_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--model", default="GRN2b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--metric_workers", type=int, default=8)
    parser.add_argument("--metric_batch_size", type=int, default=32)
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument("--metrics_only", action="store_true")
    args = parser.parse_args()
    if args.generate_only and args.metrics_only:
        parser.error("--generate_only and --metrics_only are mutually exclusive")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    try:
        resolve_benchmarks(args.benchmarks)
    except ValueError as error:
        parser.error(str(error))
    return args


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def benchmark_pairs(protocol_root: Path, relative_root: str) -> list[tuple[Path, Path]]:
    root = protocol_root / relative_root
    for hr_name, lr_name in (("gt", "lq"), ("test_HR", "test_LR")):
        hr_dir, lr_dir = root / hr_name, root / lr_name
        if hr_dir.is_dir() and lr_dir.is_dir():
            break
    else:
        raise FileNotFoundError(f"Missing HR/LR benchmark directories under {root}")
    lr_by_name = {path.name: path for path in image_files(lr_dir)}
    hr_paths = image_files(hr_dir)
    pairs = [(hr, lr_by_name[hr.name]) for hr in hr_paths if hr.name in lr_by_name]
    if len(pairs) != len(hr_paths):
        raise RuntimeError(f"{root}: not every HR image has a same-name LR image")
    return pairs


def valid_prediction(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            return image.size == (512, 512)
    except OSError:
        return False


def distributed_context() -> tuple[int, int, torch.device]:
    if "RANK" not in os.environ:
        torch.cuda.set_device(0)
        return 0, 1, torch.device("cuda:0")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch_dist.init_process_group("nccl")
    return torch_dist.get_rank(), torch_dist.get_world_size(), torch.device(f"cuda:{local_rank}")


def generate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GRN-SR generation requires CUDA")
    rank, world_size, device = distributed_context()
    mode_root = args.output_root / "no_cfg"
    metadata_path = mode_root / "generation.json"
    active_benchmarks = resolve_benchmarks(args.benchmarks)
    signature = build_generation_signature(args, list(active_benchmarks))

    if rank == 0:
        mode_root.mkdir(parents=True, exist_ok=True)
        if metadata_path.is_file():
            existing = json.loads(metadata_path.read_text())
            if existing.get("generation_signature") != signature:
                raise RuntimeError(
                    f"Output directory has a different generation signature: {metadata_path}"
                )
        metadata_path.write_text(
            json.dumps(
                {
                    "status": "generating",
                    "experiment": "GRN2B-SR 128-to-512 without CFG",
                    "generation_signature": signature,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
    if torch_dist.is_initialized():
        torch_dist.barrier()

    pipeline = GRNPipeline.from_pretrained(
        model_path=str(args.checkpoint),
        vae_path=str(args.vae),
        task="SR",
        model=args.model,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    pipeline.model.eval()
    pipeline.vae.eval()

    counts = {}
    for benchmark_name, relative_root in active_benchmarks.items():
        pairs = benchmark_pairs(args.protocol_root, relative_root)
        if len(pairs) != EXPECTED_COUNTS[benchmark_name]:
            raise RuntimeError(
                f"{benchmark_name}: expected {EXPECTED_COUNTS[benchmark_name]}, found {len(pairs)}"
            )
        destination = mode_root / benchmark_name
        destination.mkdir(parents=True, exist_ok=True)
        local_indices = list(range(rank, len(pairs), world_size))
        for local_number, index in enumerate(local_indices, start=1):
            hr_path, lr_path = pairs[index]
            output_path = destination / hr_path.name
            if valid_prediction(output_path):
                continue
            with Image.open(lr_path) as handle:
                lr_image = handle.convert("RGB")
                if lr_image.size != (128, 128):
                    raise ValueError(f"{lr_path}: expected 128x128, got {lr_image.size}")
                prediction = pipeline(
                    lr_image=lr_image,
                    guidance_scale=1.0,
                    temperature=args.temperature,
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
                )
            if prediction.shape != (1, 512, 512, 3):
                raise RuntimeError(f"Unexpected GRN output shape: {prediction.shape}")
            Image.fromarray(prediction[0], mode="RGB").save(output_path, format="PNG")
            if local_number % 25 == 0:
                print(
                    f"[rank {rank}] {benchmark_name}: "
                    f"{local_number}/{len(local_indices)} local samples",
                    flush=True,
                )
        counts[benchmark_name] = len(pairs)

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    if torch_dist.is_initialized():
        torch_dist.barrier()
    if rank == 0:
        metadata_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "experiment": "GRN2B-SR 128-to-512 without CFG",
                    "generation_signature": signature,
                    "counts": counts,
                    "sampling": {
                        "temperature": args.temperature,
                        "steps": args.steps,
                        "seed_rule": "seed + global benchmark index",
                        "top_k": 1,
                        "top_p": 0.75,
                        "top_p_effect": "top_k=1 selects the maximum-probability HBQ bit",
                    },
                    "protocol": {
                        "native_lr": [128, 128],
                        "output": [512, 512],
                        "postprocess": "none",
                        "color_correction": "none",
                    },
                    "generated_at_unix": time.time(),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
    if torch_dist.is_initialized():
        torch_dist.destroy_process_group()


def evaluate_exact_reference_metrics(args: argparse.Namespace) -> None:
    active_benchmarks = resolve_benchmarks(args.benchmarks)
    script = args.protocol_root / "scripts" / "evaluate_alignvar.py"
    spec = importlib.util.spec_from_file_location("alignvar_reference_evaluator", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPORT_NAMES["no_cfg"] = "GRN2B-SR 128-to-512 without CFG"
    module.BENCHMARKS = active_benchmarks
    module.EXPECTED_COUNTS = {
        name: EXPECTED_COUNTS[name] for name in active_benchmarks
    }
    report = module.evaluate(
        mode="no_cfg",
        cfg=1.0,
        checkpoint=args.checkpoint,
        output_root=args.output_root,
        crop_border=0,
        metric_workers=args.metric_workers,
        metric_batch_size=args.metric_batch_size,
    )
    report["protocol"]["guidance"] = (
        "disabled; GRN uses internal scale=1 for the conditional prediction"
    )
    report["protocol"]["native_lr"] = [128, 128]
    report["protocol"]["output"] = [512, 512]
    report["protocol"]["reference_evaluator"] = str(script)
    paper_results = {
        model: {
            benchmark: dict(zip(PAPER_METRICS, values))
            for benchmark, values in rows.items()
            if benchmark in active_benchmarks
        }
        for model, rows in PAPER_TABLE1_VALUES.items()
    }
    report["paper_table1_baselines"] = {
        "source": str(
            args.protocol_root / "reports" / "VARSR_CFG0_CFG6_vs_AlignVAR_Table1_ZH.md"
        ),
        "results": paper_results,
    }
    report["deltas_grn_minus_paper"] = {
        model: {
            benchmark: {
                metric: report["results"][benchmark][metric] - value
                for metric, value in metrics.items()
            }
            for benchmark, metrics in rows.items()
        }
        for model, rows in paper_results.items()
    }
    report_path = args.output_root / "no_cfg" / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.vae = args.vae.resolve()
    args.protocol_root = args.protocol_root.resolve()
    args.output_root = args.output_root.resolve()
    for path in (args.checkpoint, args.vae):
        if not path.is_file():
            raise FileNotFoundError(path)

    if not args.metrics_only:
        generate(args)
        if args.generate_only:
            return

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        evaluate_exact_reference_metrics(args)


if __name__ == "__main__":
    main()
