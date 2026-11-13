"""Resolve archived S5 paths without changing provenance bytes or RNG identity.

All original manifests and checkpoints remain byte-for-byte copies. Only filesystem
access is relocated. An unmapped external path fails rather than reading the source
project. The map is explicit, longest-prefix-first, and cannot escape this bundle.
"""
from functools import lru_cache
import json
import os
from pathlib import Path


def bundle_root():
    discovered = Path(__file__).resolve().parents[3]
    configured = os.environ.get("GRN_S5_ROOT")
    if configured and Path(configured).resolve() != discovered:
        raise ValueError("GRN_S5_ROOT differs from the executing bundle")
    return discovered


@lru_cache(maxsize=1)
def _mappings():
    root = bundle_root()
    value = json.loads((root / "manifests/path_map.json").read_text())
    if value.get("schema") != "grn_s5_path_map_v1":
        raise ValueError("Unsupported S5 bundle path-map schema")
    result = []
    seen = set()
    for row in value["mappings"]:
        source = Path(row["source_prefix"])
        relative = Path(row["destination_prefix"])
        if not source.is_absolute() or ".." in source.parts:
            raise ValueError("Source mapping must be a normalized absolute path")
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Destination mapping must remain within the bundle")
        if source in seen:
            raise ValueError("Duplicate source path mapping")
        seen.add(source)
        result.append((source, root / relative))
    return tuple(sorted(result, key=lambda item: len(item[0].parts), reverse=True))


def mapped_path(value):
    """Relocate lexically, preserving canonical-path checks before symlink resolution."""
    root = bundle_root()
    requested = Path(value).expanduser()
    if ".." in requested.parts:
        raise ValueError(f"Parent traversal is not allowed in bundle paths: {value}")
    if not requested.is_absolute():
        candidate = root / requested
    elif requested.is_relative_to(root):
        candidate = requested
    else:
        for source, destination in _mappings():
            if requested.is_relative_to(source):
                candidate = destination / requested.relative_to(source)
                break
        else:
            raise ValueError(f"External path has no S5 bundle mapping: {value}")
    if not candidate.resolve().is_relative_to(root):
        raise ValueError(f"Bundle path escapes through a symlink: {candidate}")
    return candidate


def resolve_path(value, *, strict=False):
    """Return the local I/O path; never alter the caller's identity string."""
    resolved = mapped_path(value).resolve(strict=strict)
    if not resolved.is_relative_to(bundle_root()):
        raise ValueError(f"Bundle path escapes through a symlink: {value}")
    return resolved
