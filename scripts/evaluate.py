#!/usr/bin/env python3
"""Recompute all eight historical S5 metrics with the copied original evaluator."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from _bundle import CONFIG, ROOT, bundled, metric_asset_paths, offline_metric_assets, output_path, pairs, setup, write_json
from _evaluation_staging import stage_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=list(CONFIG["benchmarks"]), default=list(CONFIG["benchmarks"]))
    parser.add_argument("--input", help="Package inference root containing no_cfg/DATASET; omit with --historical")
    parser.add_argument("--historical", action="store_true", help="Re-score copied historical PNGs without re-generating")
    parser.add_argument("--output", default="outputs/metrics")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8, help="FID feature batch size; seven image metrics remain per image")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if a recomputed historical mean differs beyond tolerance")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--fid-tolerance", type=float, default=1e-4)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.historical == bool(args.input):
        parser.error("Select exactly one of --historical and --input")
    if len(set(args.benchmarks)) != len(args.benchmarks):
        parser.error("Duplicate benchmarks")
    setup()
    destination = output_path(args.output)
    input_root = output_path(args.input) if args.input else None
    inputs = {}
    for benchmark in args.benchmarks:
        expected = {hr.name for hr, _ in pairs(benchmark)}
        source = (bundled(CONFIG["benchmarks"][benchmark]["historical_predictions"]) if args.historical
                  else input_root / "no_cfg" / benchmark)
        actual = {path.name for path in source.iterdir() if path.is_file() and path.suffix.lower() == ".png"}
        if actual != expected:
            raise ValueError(f"Full-set metrics require exact {benchmark} image identities/count; found {len(actual)}, expected {len(expected)}")
        inputs[benchmark] = source
    if args.preflight:
        assets = metric_asset_paths()
        print(json.dumps({"status": "preflight_passed", "inputs": {k: str(v) for k, v in inputs.items()},
                          "counts": {k: CONFIG['benchmarks'][k]['count'] for k in inputs},
                          "metric_assets": {k: str(v) for k, v in assets.items()}}, indent=2))
        return
    staging = {benchmark: stage_predictions(source, destination / "no_cfg" / benchmark,
                                            {hr.name for hr, _ in pairs(benchmark)}, ROOT)
               for benchmark, source in inputs.items()}
    write_json(destination / "prediction_staging.json", staging)
    print(json.dumps({"prediction_staging": staging}), flush=True)
    assets = offline_metric_assets()
    script = bundled(CONFIG["protocol_root"]) / "scripts/evaluate_alignvar.py"
    if not script.is_file() or not script.resolve().is_relative_to(ROOT):
        raise ValueError("The original reference evaluator must be a package-local file")
    spec = importlib.util.spec_from_file_location("s5_original_reference_evaluator", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REPORT_NAMES["no_cfg"] = "GRN2B-SR 128-to-512 without CFG"
    module.BENCHMARKS = {name: CONFIG["benchmarks"][name]["relative_root"] for name in args.benchmarks}
    module.EXPECTED_COUNTS = {name: CONFIG["benchmarks"][name]["count"] for name in args.benchmarks}
    import torch
    numeric_policy = {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                      "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                      "cudnn_benchmark": torch.backends.cudnn.benchmark,
                      "cudnn_deterministic": torch.backends.cudnn.deterministic,
                      "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                      "policy": "Preserve original evaluation-path defaults; no precision flags changed by this wrapper."}
    write_json(destination / "numeric_policy.json", numeric_policy)
    print(json.dumps({"numeric_policy": numeric_policy}), flush=True)
    # Preserve each original metric call while recording its elapsed time and mean.
    # This also makes a long, interrupted run's completed metrics inspectable.
    progress_path = destination / "metric_progress.jsonl"
    def progress(event):
        event["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with progress_path.open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        print("[timing] " + json.dumps(event), flush=True)
    original_mean = module.mean_metric
    def timed_mean(metric, paired, prediction_dir, reference):
        metric_name = getattr(metric, "metric_name", type(metric).__name__)
        identity = {"benchmark": prediction_dir.name, "metric": metric_name}
        progress({**identity, "event": "start", "count": len(paired)})
        started = time.monotonic()
        result = original_mean(metric, paired, prediction_dir, reference)
        progress({**identity, "event": "complete", "elapsed_seconds": time.monotonic() - started, "value": result})
        return result
    module.mean_metric = timed_mean
    import pyiqa
    original_factory = pyiqa.create_metric
    def timed_factory(name, *positional, **kwargs):
        metric = original_factory(name, *positional, **kwargs)
        if name == "fid":
            state = {}
            def before(_module, positional):
                state["benchmark"] = Path(positional[1]).name
                state["started"] = time.monotonic()
                progress({"benchmark": state["benchmark"], "metric": "fid", "event": "start"})
            def after(_module, _positional, result):
                progress({"benchmark": state["benchmark"], "metric": "fid", "event": "complete",
                          "elapsed_seconds": time.monotonic() - state["started"], "value": float(result)})
            metric.register_forward_pre_hook(before)
            metric.register_forward_hook(after)
        return metric
    pyiqa.create_metric = timed_factory
    report = module.evaluate(mode="no_cfg", cfg=1.0, checkpoint=bundled(CONFIG["checkpoint"]),
                             output_root=destination, crop_border=0, metric_workers=args.workers,
                             metric_batch_size=args.batch_size)
    comparison = {}
    for benchmark, values in report["results"].items():
        historical = json.loads(bundled(CONFIG["benchmarks"][benchmark]["historical_metrics"]).read_text())["results"][benchmark]
        comparison[benchmark] = {name: {"actual": value, "historical": historical[name],
                                        "difference": value - historical[name],
                                        "tolerance": args.fid_tolerance if name == "FID" else args.tolerance,
                                        "within_tolerance": abs(value - historical[name]) <= (args.fid_tolerance if name == "FID" else args.tolerance)}
                                 for name, value in values.items()}
    passed = all(item["within_tolerance"] for values in comparison.values() for item in values.values())
    report["portable_reproduction"] = {"status": "passed" if passed else "differences_detected", "assets": assets,
                                        "numeric_policy": numeric_policy,
                                        "prediction_staging": staging,
                                        "evaluator": str(script.relative_to(ROOT)), "python": sys.executable,
                                        "inputs": {key: str(value.relative_to(ROOT)) for key, value in inputs.items()},
                                        "historical_comparison": comparison,
                                        "note": "Original seven per-image metrics and clean FID; no RNG reset or scoring-option change was added."}
    write_json(destination / "no_cfg/metrics.json", report)
    write_json(destination / "metrics_comparison.json", {"status": report["portable_reproduction"]["status"], "comparisons": comparison})
    if args.strict and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
