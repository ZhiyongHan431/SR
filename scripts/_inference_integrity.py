"""Content identities for immutable S5 assets and resumable generated images."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_index(root: Path) -> dict:
    """Read completed copy records; a missing or partial record is never trusted."""
    index = {}
    for log in sorted((root / "verification/copy").glob("*.jsonl")):
        with log.open() as handle:
            for line in handle:
                row = json.loads(line)
                key = row["destination"]
                if key in index and (index[key]["sha256"], index[key]["bytes"]) != (row["sha256"], row["bytes"]):
                    raise ValueError(f"Conflicting verified copy records for {key}")
                index[key] = row
    if not index:
        raise FileNotFoundError("No verified package copy records")
    return index


def verified_file(path: Path, expected: dict) -> dict:
    """Hash bytes at every invocation; equal file size alone is insufficient."""
    if expected.get("destination_sha256_verified") is not True:
        raise ValueError(f"No completed destination hash verification: {path}")
    before = path.stat()
    if before.st_size != expected["bytes"]:
        raise ValueError(f"Asset size differs from copied source: {path}")
    actual = digest_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f"Asset changed while being hashed: {path}")
    if actual != expected["sha256"]:
        raise ValueError(f"Asset SHA256 differs from verified copied source: {path}")
    return {"bytes": before.st_size, "sha256": actual}


def source_identity(root: Path, index: dict) -> dict:
    """Reject an incomplete copied source tree and bind the actual portable code."""
    prefix = "reference/source_code/"
    expected = [key[len(prefix):] for key in index if key.startswith(prefix) and key.endswith(".py")]
    if not expected:
        raise ValueError("Missing original source inventory")
    for relative in expected:
        if not (root / "code" / relative).is_file():
            raise FileNotFoundError(f"Incomplete executable source copy: code/{relative}")
    paths = list((root / "code").rglob("*.py"))
    paths.extend(root / "scripts" / name for name in ("infer.py", "_bundle.py", "_inference_integrity.py"))
    paths.extend((root / "manifests/bundle.json", root / "manifests/path_map.json"))
    identities = {}
    for path in sorted(paths):
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Missing or external inference source: {path}")
        identities[str(path.relative_to(root))] = digest_file(path)
    return {"sha256": digest_json(identities), "files": identities}


def prediction_provenance(output: Path, record: dict, signature_sha256: str) -> dict:
    return {"schema": "grn_s5_prediction_v1", "generation_signature_sha256": signature_sha256,
            **{key: record[key] for key in ("benchmark", "global_index", "seed", "filename", "lr_sha256")},
            "prediction_png_sha256": digest_file(output)}


def verify_existing_prediction(output: Path, record: dict, signature_sha256: str, previous_record: dict | None = None) -> bool:
    """Skip only images bound to the same run/input whose exact PNG hash matches."""
    if not output.exists():
        return False
    receipt = output.with_suffix(output.suffix + ".provenance.json")
    if receipt.is_file():
        provenance = json.loads(receipt.read_text())
        if provenance.get("schema") != "grn_s5_prediction_v1" or provenance.get("generation_signature_sha256") != signature_sha256:
            raise ValueError(f"Prediction receipt belongs to another generation: {output}")
    elif previous_record and previous_record.get("prediction_png_sha256"):
        provenance = previous_record
    else:
        raise ValueError(f"Prediction has no verified checksum receipt; use a new output directory: {output}")
    for key in ("benchmark", "global_index", "seed", "filename", "lr_sha256"):
        if provenance.get(key) != record[key]:
            raise ValueError(f"Prediction input identity changed ({key}): {output}")
    if digest_file(output) != provenance.get("prediction_png_sha256"):
        raise ValueError(f"Existing PNG checksum mismatch: {output}")
    from PIL import Image
    with Image.open(output) as image:
        if image.size != (512, 512) or image.mode != "RGB":
            raise ValueError(f"Invalid existing prediction: {output}")
        image.load()
    return True
