"""Expose only the exact benchmark PNG set to recursively scanning FID code."""
from __future__ import annotations

import os
from pathlib import Path


def stage_predictions(source: Path, stage: Path, expected: set[str], root: Path) -> dict:
    root = root.resolve()
    source = source.resolve()
    if not source.is_relative_to(root):
        raise ValueError(f"Prediction source is outside this bundle: {source}")
    if stage.is_symlink():
        raise ValueError(f"Prediction staging must be a real directory, not a directory symlink: {stage}")
    stage.mkdir(parents=True, exist_ok=True)
    if not stage.resolve().is_relative_to(root / "outputs"):
        raise ValueError(f"Prediction staging is outside bundle outputs: {stage}")
    # Reject stale directories/files, including notebook backups. FID recursively
    # scans directories and does not share the benchmark helper's exclusion rule.
    unexpected = {path.name for path in stage.iterdir()} - expected
    if unexpected:
        raise ValueError(f"Unexpected entries in prediction staging: {sorted(unexpected)[:3]}")
    for name in sorted(expected):
        if Path(name).name != name or Path(name).suffix.lower() != ".png":
            raise ValueError(f"Invalid benchmark PNG name: {name}")
        target = source / name
        if not target.is_file() or not target.resolve().is_relative_to(root):
            raise ValueError(f"Prediction is missing or leaves this bundle: {target}")
        destination = stage / name
        if destination.exists() or destination.is_symlink():
            if not destination.is_symlink() or destination.resolve() != target.resolve():
                raise ValueError(f"Prediction staging belongs to another input: {destination}")
        else:
            destination.symlink_to(os.path.relpath(target, stage))
    visible = list(stage.rglob("*"))
    if len(visible) != len(expected) or {path.name for path in visible} != expected:
        raise ValueError("Recursive prediction staging count/identity mismatch")
    return {"source": str(source.relative_to(root)), "stage": str(stage.relative_to(root)),
            "expected_count": len(expected), "recursive_file_count": len(visible),
            "first": min(expected), "last": max(expected),
            "policy": "Individual package-relative symlinks for exactly the paired benchmark PNGs."}
