#!/usr/bin/env python3
"""Compare package-generated predictions to copied historical S5 RGB pixels."""
from __future__ import annotations
import argparse
import hashlib
import json
import numpy as np
from PIL import Image
from _bundle import CONFIG, ROOT, bundled, output_path, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/smoke")
    parser.add_argument("--report", help="Report path under outputs; default INPUT/reproduction.json")
    args = parser.parse_args()
    root = output_path(args.input)
    generation = json.loads((root / "no_cfg/generation.json").read_text())
    if generation.get("status") != "complete":
        raise ValueError("Only completed inference manifests can be verified")
    records = []
    for item in generation["records"]:
        original = bundled(CONFIG["benchmarks"][item["benchmark"]]["historical_predictions"]) / item["filename"]
        current = root / "no_cfg" / item["benchmark"] / item["filename"]
        with Image.open(original) as handle:
            expected = np.array(handle.convert("RGB"))
        with Image.open(current) as handle:
            actual = np.array(handle.convert("RGB"))
        if actual.shape != expected.shape:
            raise ValueError(f"Shape mismatch: {current}")
        difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
        records.append({"benchmark": item["benchmark"], "global_index": item["global_index"], "seed": item["seed"],
                        "filename": item["filename"], "rgb_byte_exact": bool(np.array_equal(actual, expected)),
                        "different_pixels": int(np.count_nonzero(np.any(difference, axis=-1))),
                        "different_channel_values": int(np.count_nonzero(difference)),
                        "max_abs_uint8": int(difference.max()), "mean_abs_uint8": float(difference.mean()),
                        "historical_png_sha256": sha256(original), "generated_png_sha256": sha256(current),
                        "historical_rgb_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                        "generated_rgb_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                        "historical_path": str(original.relative_to(ROOT)), "generated_path": str(current.relative_to(ROOT))})
    passed = bool(records) and all(item["rgb_byte_exact"] for item in records)
    report = {"schema": "grn_s5_rgb_reproduction_v1", "status": "passed" if passed else "failed",
              "comparison": "decoded RGB bytes; PNG compression identity is reported separately",
              "image_count": len(records), "results": records}
    path = output_path(args.report) if args.report else root / "reproduction.json"
    write_json(path, report)
    print(json.dumps({"status": report["status"], "images": len(records), "report": str(path)}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
