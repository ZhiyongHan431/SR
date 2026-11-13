"""Shared, dependency-light protocol helpers for GRN-SR evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


BENCHMARKS = {
    "DIV2K-Val": "data/benchmarks/StableSR_testsets/DIV2K_V2_val",
    "RealSR": "data/benchmarks/StableSR_testsets/RealSRVal_crop128",
    "DRealSR": "data/benchmarks/StableSR_testsets/DrealSRVal_crop128",
}
EXPECTED_COUNTS = {"DIV2K-Val": 3000, "RealSR": 100, "DRealSR": 93}


def resolve_benchmarks(names: Sequence[str]) -> dict[str, str]:
    """Validate benchmark names and return their protocol paths in order."""
    selected = list(names)
    if not selected:
        raise ValueError("At least one benchmark is required")
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate benchmark names are not allowed")
    unknown = [name for name in selected if name not in BENCHMARKS]
    if unknown:
        raise ValueError(f"Unknown benchmark: {unknown[0]}")
    return {name: BENCHMARKS[name] for name in selected}


def checkpoint_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_generation_signature(args: Any, benchmark_names: Sequence[str]) -> dict[str, Any]:
    """Build the resume-safety signature for one prediction directory."""
    selected = list(resolve_benchmarks(benchmark_names))
    return {
        "checkpoint": checkpoint_signature(args.checkpoint),
        "model": args.model,
        "seed": args.seed,
        "temperature": args.temperature,
        "steps": args.steps,
        "benchmarks": selected,
        "guidance": "disabled (GRN internal scale=1, conditional prediction)",
        "color_correction": "none",
    }
