#!/usr/bin/env python3
"""Generate VARSR results and evaluate with the public AlignVAR benchmark protocol."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

import dist
from dataloader.alignvar_repro import benchmark_dirs, image_files
from models import VQVAE, build_var


PATCH_NUMS = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
BENCHMARKS = {
    "DIV2K-Val": "data/benchmarks/StableSR_testsets/DIV2K_V2_val",
    "RealSR": "data/benchmarks/StableSR_testsets/RealSRVal_crop128",
    "DRealSR": "data/benchmarks/StableSR_testsets/DrealSRVal_crop128",
}
EXPECTED_COUNTS = {"DIV2K-Val": 3000, "RealSR": 100, "DRealSR": 93}
REPORT_NAMES = {
    "no_cfg": "VARSR without Image-CFG",
    "synthetic_cfg": "VARSR synthetic negative CFG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["auto", "no_cfg", "synthetic_cfg"],
        default="auto",
    )
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--vae", type=Path, default=Path("checkpoints/VQVAE.pth"))
    parser.add_argument("--output_root", type=Path, default=Path("outputs/alignvar_eval"))
    parser.add_argument("--crop_border", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metric_workers", type=int, default=8)
    parser.add_argument("--metric_batch_size", type=int, default=32)
    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument("--metrics_only", action="store_true")
    args = parser.parse_args()
    if args.generate_only and args.metrics_only:
        parser.error("--generate_only and --metrics_only are mutually exclusive")
    if args.crop_border < 0:
        parser.error("--crop_border must be non-negative")
    return args


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path
def checkpoint_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }




def checkpoint_mode(state: dict, requested: str) -> str:
    saved_args = state.get("args", {})
    if hasattr(saved_args, "get"):
        saved_mode = saved_args.get("cfg_mode")
    else:
        saved_mode = getattr(saved_args, "cfg_mode", None)
    if requested == "auto":
        if saved_mode not in REPORT_NAMES:
            raise RuntimeError(
                "Checkpoint has no cfg_mode metadata. Pass --mode explicitly only "
                "for a checkpoint trained by this bundle; do not evaluate C2I directly."
            )
        return saved_mode
    if saved_mode in REPORT_NAMES and requested != saved_mode:
        raise RuntimeError(
            f"Requested mode={requested}, but checkpoint records cfg_mode={saved_mode}"
        )
    return requested


def resolve_cfg(mode: str, requested_cfg: float | None) -> float:
    default = 0.0 if mode == "no_cfg" else 6.0
    cfg = default if requested_cfg is None else requested_cfg
    if mode == "no_cfg" and cfg != 0:
        raise RuntimeError(
            "no_cfg checkpoints must use cfg=0: label=1 was never trained and "
            "there is no meaningful negative branch."
        )
    return float(cfg)


def paired_paths(relative_root: str) -> tuple[Path, list[tuple[Path, Path]]]:
    root = absolute(Path(relative_root))
    hr_dir, lr_dir = benchmark_dirs(root)
    hr_files = image_files(hr_dir)
    lr_by_name = {path.name: path for path in image_files(lr_dir)}
    pairs = [(hr, lr_by_name[hr.name]) for hr in hr_files if hr.name in lr_by_name]
    return hr_dir, pairs


def load_model(checkpoint_state: dict, vae_path: Path, device: torch.device):
    vae, var = build_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        controlnet_depth=24,
        device=device,
        patch_nums=PATCH_NUMS,
        control_patch_nums=PATCH_NUMS,
        num_classes=2,
        depth=24,
        shared_aln=False,
        attn_l2_norm=True,
        flash_if_available=False,
        fused_if_available=False,
        init_adaln=0.5,
        init_adaln_gamma=1e-4,
        init_head=0.02,
        init_std=-1,
    )
    vae_state = torch.load(vae_path, map_location="cpu")["trainer"]["vae_local"]
    vae.load_state_dict(vae_state, strict=True)
    var.load_state_dict(
        checkpoint_state["trainer"]["var_wo_ddp"],
        strict=True,
    )
    vae.eval()
    var.eval()
    del vae_state
    return vae, var


def load_condition(lr_path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(lr_path) as handle:
        lr = TF.to_tensor(handle.convert("RGB"))
    if tuple(lr.shape[-2:]) != (128, 128):
        raise ValueError(f"{lr_path}: expected LR 128x128, got {tuple(lr.shape[-2:])}")
    condition = F.interpolate(
        lr.unsqueeze(0),
        size=(512, 512),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp_(0, 1)
    return condition.mul(2).sub(1).to(device, non_blocking=True)


def save_tensor(image: torch.Tensor, path: Path) -> None:
    array = (
        image.detach()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(array).save(path)


def valid_existing_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as handle:
            return handle.size == (512, 512)
    except OSError:
        return False


def generate(
    var,
    mode: str,
    cfg: float,
    output_root: Path,
    device: torch.device,
    rank: int,
    world_size: int,
    top_k: int,
    top_p: float,
    seed: int,
) -> dict:
    counts = {}
    for benchmark_name, relative_root in BENCHMARKS.items():
        _, pairs = paired_paths(relative_root)
        if len(pairs) != EXPECTED_COUNTS[benchmark_name]:
            raise RuntimeError(
                f"{benchmark_name}: expected {EXPECTED_COUNTS[benchmark_name]}, "
                f"found {len(pairs)}"
            )
        destination = output_root / mode / benchmark_name
        destination.mkdir(parents=True, exist_ok=True)
        local_indices = list(range(rank, len(pairs), world_size))
        for local_number, index in enumerate(local_indices, start=1):
            _, lr_path = pairs[index]
            output_path = destination / pairs[index][0].name
            if valid_existing_image(output_path):
                continue
            condition = load_condition(lr_path, device)
            label = torch.zeros(1, dtype=torch.long, device=device)
            with torch.inference_mode(), torch.autocast(
                "cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
                cache_enabled=True,
            ):
                prediction = var.autoregressive_infer_cfg(
                    B=1,
                    cfg=cfg,
                    top_k=top_k,
                    top_p=top_p,
                    text_hidden=None,
                    lr_inp=condition,
                    negative_text=None,
                    label_B=label,
                    g_seed=seed + index,
                    lr_inp_scale=None,
                    more_smooth=False,
                )
            save_tensor(prediction[0], output_path)
            if local_number % 25 == 0:
                print(
                    f"[rank {rank}] {benchmark_name}: "
                    f"{local_number}/{len(local_indices)} local samples"
                )
        counts[benchmark_name] = len(pairs)
    return counts


def ensure_clean_gt(
    hr_dir: Path,
    pairs: list[tuple[Path, Path]],
    stage_dir: Path,
) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    for hr_path, _ in pairs:
        destination = stage_dir / hr_path.name
        if destination.exists():
            continue
        try:
            os.link(hr_path, destination)
        except OSError:
            try:
                destination.symlink_to(hr_path)
            except OSError:
                shutil.copy2(hr_path, destination)
    staged = image_files(stage_dir)
    if len(staged) != len(pairs):
        raise RuntimeError(
            f"Clean GT staging mismatch: expected {len(pairs)}, found {len(staged)}"
        )
    return stage_dir


def mean_metric(metric, pairs, prediction_dir: Path, reference: bool) -> float:
    scores = []
    with torch.inference_mode():
        for index, (hr_path, _) in enumerate(pairs, start=1):
            prediction = prediction_dir / hr_path.name
            score = metric(str(prediction), str(hr_path)) if reference else metric(str(prediction))
            scores.append(float(torch.as_tensor(score).mean().item()))
            if index % 250 == 0:
                print(f"  metric progress: {index}/{len(pairs)}")
    return float(np.mean(scores))


def verify_predictions(pairs, prediction_dir: Path) -> None:
    expected = {hr_path.name for hr_path, _ in pairs}
    actual = {path.name for path in image_files(prediction_dir)}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"{prediction_dir}: prediction set mismatch; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )


def evaluate(
    mode: str,
    cfg: float,
    checkpoint: Path,
    output_root: Path,
    crop_border: int,
    metric_workers: int,
    metric_batch_size: int,
) -> dict:
    import pyiqa

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metric_specs = [
        ("PSNR-Y", "psnry", True, {"crop_border": crop_border}),
        ("SSIM-Y", "ssim", True, {"crop_border": crop_border}),
        ("LPIPS", "lpips", True, {}),
        ("DISTS", "dists", True, {}),
        ("MANIQA", "maniqa", False, {}),
        ("MUSIQ", "musiq", False, {}),
        ("CLIPIQA", "clipiqa", False, {}),
    ]
    all_results = {}
    clean_gt_root = output_root / "_paired_gt_without_notebook_copies"

    for benchmark_name, relative_root in BENCHMARKS.items():
        hr_dir, pairs = paired_paths(relative_root)
        prediction_dir = output_root / mode / benchmark_name
        verify_predictions(pairs, prediction_dir)
        benchmark_result = {}
        print(f"[metrics] {benchmark_name}: {len(pairs)} pairs")

        for display_name, metric_name, reference, kwargs in metric_specs:
            print(f"[metrics] {benchmark_name}: {display_name}")
            metric = pyiqa.create_metric(metric_name, device=device, **kwargs)
            benchmark_result[display_name] = mean_metric(
                metric,
                pairs,
                prediction_dir,
                reference,
            )
            del metric
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        clean_gt = ensure_clean_gt(
            hr_dir,
            pairs,
            clean_gt_root / benchmark_name,
        )
        print(f"[metrics] {benchmark_name}: FID (clean)")
        fid = pyiqa.create_metric("fid", device=device)
        fid_value = fid(
            str(clean_gt),
            str(prediction_dir),
            mode="clean",
            num_workers=metric_workers,
            batch_size=metric_batch_size,
            verbose=True,
        )
        benchmark_result["FID"] = float(fid_value)
        del fid
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        all_results[benchmark_name] = benchmark_result

    report = {
        "experiment": REPORT_NAMES[mode],
        "mode": mode,
        "cfg": cfg,
        "checkpoint": str(checkpoint),
        "protocol": {
            "benchmarks": EXPECTED_COUNTS,
            "crop_border": crop_border,
            "PSNR_SSIM": "Y channel",
            "FID": "clean",
            "color_correction": "none",
            "adain": False,
            "wavelet_color_fix": False,
        },
        "results": all_results,
    }
    report_path = output_root / mode / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"OK: metrics written to {report_path}")
    return report


def main() -> None:
    args = parse_args()
    checkpoint = absolute(args.checkpoint)
    vae_path = absolute(args.vae)
    output_root = absolute(args.output_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not vae_path.is_file():
        raise FileNotFoundError(vae_path)

    state = torch.load(checkpoint, map_location="cpu", mmap=True)
    mode = checkpoint_mode(state, args.mode)
    cfg = resolve_cfg(mode, args.cfg)
    print(f"experiment={REPORT_NAMES[mode]}, mode={mode}, cfg={cfg}")
    signature = checkpoint_signature(checkpoint)
    generation_signature = {
        "checkpoint": signature,
        "mode": mode,
        "cfg": cfg,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    metadata_path = output_root / mode / "generation.json"

    distributed_generation = not args.metrics_only
    if distributed_generation:
        if not torch.cuda.is_available():
            raise RuntimeError("VARSR generation requires CUDA")
        dist.initialize(gpu_id_if_not_distibuted=0)
        rank, world_size = dist.get_rank(), dist.get_world_size()
        device = dist.get_device()
        if rank == 0:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            if metadata_path.is_file():
                existing = json.loads(metadata_path.read_text())
                if existing.get("generation_signature") != generation_signature:
                    raise RuntimeError(
                        "The prediction directory belongs to a different run "
                        f"version: {metadata_path}. Use a new output_root."
                    )
            elif image_files(metadata_path.parent):
                raise RuntimeError(
                    "Predictions exist without checkpoint metadata. Use a new "
                    f"output_root instead of mixing results: {metadata_path.parent}"
                )
            metadata_path.write_text(
                json.dumps(
                    {
                        "status": "generating",
                        "experiment": REPORT_NAMES[mode],
                        "mode": mode,
                        "cfg": cfg,
                        "generation_signature": generation_signature,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            )
        dist.barrier()
        vae, var = load_model(state, vae_path, device)
        del state
        counts = generate(
            var,
            mode,
            cfg,
            output_root,
            device,
            rank,
            world_size,
            args.top_k,
            args.top_p,
            args.seed,
        )
        del var, vae
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        if rank == 0:
            metadata = {
                "status": "complete",
                "experiment": REPORT_NAMES[mode],
                "mode": mode,
                "cfg": cfg,
                "checkpoint": str(checkpoint),
                "checkpoint_signature": signature,
                "generation_signature": generation_signature,
                "counts": counts,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "seed_rule": "seed + global benchmark index",
                "color_correction": "none",
                "generated_at_unix": time.time(),
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
            )
        if rank != 0 or args.generate_only:
            return
    else:
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            return
        if not metadata_path.is_file():
            raise RuntimeError(
                f"Cannot run metrics-only without completed generation metadata: {metadata_path}"
            )
        existing = json.loads(metadata_path.read_text())
        if (
            existing.get("status") != "complete"
            or existing.get("generation_signature") != generation_signature
        ):
            raise RuntimeError(
                "metrics-only checkpoint/sampling arguments do not match the "
                f"completed predictions: {metadata_path}"
            )
        del state

    evaluate(
        mode,
        cfg,
        checkpoint,
        output_root,
        args.crop_border,
        args.metric_workers,
        args.metric_batch_size,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        dist.finalize()
