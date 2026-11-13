#!/usr/bin/env python3
"""Install bundled metric weights into the current user's standard cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify_existing",
        action="store_true",
        help="Hash existing destination files; default trusts matching file sizes.",
    )
    args = parser.parse_args()

    source_root = PROJECT_ROOT / "offline_cache"
    manifest = json.loads((source_root / "manifest.json").read_text())
    cache_root = Path.home() / ".cache"
    installed = 0
    reused = 0

    for relative, expected_hash in manifest["files"].items():
        source = source_root / relative
        destination = cache_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"Bundled cache file is missing: {source}")

        source_size = source.stat().st_size
        destination_ok = False
        if destination.is_file() and destination.stat().st_size == source_size:
            destination_ok = not args.verify_existing or sha256(destination) == expected_hash

        if destination_ok:
            print(f"reuse: {destination}")
            reused += 1
            continue

        actual_hash = sha256(source)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Bundled cache checksum mismatch: {source}; "
                f"expected={expected_hash}, actual={actual_hash}"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            backup = destination.with_name(
                destination.name + f".corrupt.{int(time.time())}"
            )
            destination.replace(backup)
            print(f"preserved unexpected existing file: {backup}")
        shutil.copy2(source, destination)
        if sha256(destination) != expected_hash:
            raise RuntimeError(f"Copied cache verification failed: {destination}")
        print(f"installed: {destination}")
        installed += 1

    print(f"OK: metric cache ready; installed={installed}, reused={reused}")


if __name__ == "__main__":
    main()
