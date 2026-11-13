#!/usr/bin/env python3
"""S5 native inference with the historical 50-step protocol and stable indices."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from _bundle import CONFIG, ROOT, bundled, output_path, pairs, setup, sha256, write_json
from _inference_integrity import (copy_index, digest_json, prediction_provenance,
                                 source_identity, verified_file, verify_existing_prediction)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=list(CONFIG["benchmarks"]), default=list(CONFIG["benchmarks"]))
    parser.add_argument("--output", default="outputs/inference")
    parser.add_argument("--smoke", action="store_true", help="Generate only original global indices 0 and 1 in each selected benchmark")
    parser.add_argument("--indices", help="Explicit comma-separated ORIGINAL benchmark indices; never reindexed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kappa", type=float, default=None, help="Nonnegative runtime calibrator strength; omit to preserve checkpoint cap")
    parser.add_argument("--adapter", help="Project-local C/R adaptation checkpoint bound to the packaged S5 base")
    parser.add_argument("--resume", action="store_true")
    # Opt-in quality-contrastive guidance and per-plane sampling. With none of
    # these flags the generation signature and sampler are unchanged.
    parser.add_argument("--qg-scale", type=float, default=0.0, help="Guidance scale w; 0 disables guidance")
    parser.add_argument("--qg-bound", type=float, default=2.0, help="Largest absolute log-odds shift per bit")
    parser.add_argument("--qg-start", type=float, default=0.3, help="Guide rounds whose input retention is at least this")
    parser.add_argument("--qg-planes", default="0,0,1,1", help="Per-plane guidance weights, most significant plane first")
    parser.add_argument("--qg-negative", choices=["degraded_lr", "calibrator", "prompt", "degraded_lr_prompt"],
                        default="degraded_lr")
    parser.add_argument("--qg-lr-sigma", type=float, default=1.2, help="Gaussian sigma (LR pixels) of the degraded-LR branch")
    parser.add_argument("--qg-lr-downscale", type=float, default=1.0, help="Extra bicubic down/up factor of the degraded-LR branch")
    parser.add_argument("--qg-negative-adapter", help="Project-local negative calibrator (C-) for --qg-negative calibrator")
    parser.add_argument("--plane-temperatures", help="Four per-plane sampling temperatures, 0 = greedy, e.g. 0,0,0.7,1")
    # Opt-in text-slot conditioning: a trained CLIP semantic prompt or cached umT5
    # prompts. Without these flags both branches keep the cached empty prompt.
    parser.add_argument("--semantic-prompt", help="Project-local semantic prompt sidecar from scripts/train_semantic.py")
    parser.add_argument("--clip", help="Project-local CLIP weights: an OpenAI .pt file or a transformers directory")
    parser.add_argument("--semantic-scale", type=float, default=1.0, help="Semantic prompt strength; 0 disables it")
    parser.add_argument("--text-prompt", help="Project-local cached umT5 prompt for the positive text slot")
    parser.add_argument("--negative-text-prompt", help="Project-local cached umT5 prompt for the negative text slot")
    parser.add_argument("--preflight", action="store_true", help="Validate assets, runtime policy and pair counts without loading a model")
    args = parser.parse_args()
    if args.smoke and args.indices:
        parser.error("--smoke and --indices are mutually exclusive")
    if len(set(args.benchmarks)) != len(args.benchmarks):
        parser.error("Duplicate benchmarks")
    if args.kappa is not None and (not math.isfinite(args.kappa) or args.kappa < 0 or args.kappa > 3.4028234663852886e38):
        parser.error("--kappa must be finite and nonnegative")
    setup()
    adapter_path = None
    if args.adapter:
        adapter_path = Path(args.adapter).expanduser()
        adapter_path = (ROOT / adapter_path).resolve() if not adapter_path.is_absolute() else adapter_path.resolve()
        if not adapter_path.is_relative_to(ROOT) or not adapter_path.is_file():
            raise ValueError("Adapter must be an existing package-local file")
    negative_adapter_path = None
    if args.qg_negative_adapter:
        negative_adapter_path = Path(args.qg_negative_adapter).expanduser()
        negative_adapter_path = ((ROOT / negative_adapter_path).resolve() if not negative_adapter_path.is_absolute()
                                 else negative_adapter_path.resolve())
        if not negative_adapter_path.is_relative_to(ROOT) or not negative_adapter_path.is_file():
            raise ValueError("Negative calibrator must be an existing package-local file")

    def project_path(value, description, *, directory=False):
        if not value:
            return None
        path = Path(value).expanduser()
        path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(ROOT) or not (path.is_file() or (directory and path.is_dir())):
            raise ValueError(f"{description} must be an existing package-local path")
        return path

    semantic_path = project_path(args.semantic_prompt, "Semantic prompt")
    clip_path = project_path(args.clip, "CLIP weights", directory=True)
    text_prompt_path = project_path(args.text_prompt, "Text prompt cache")
    negative_text_prompt_path = project_path(args.negative_text_prompt, "Negative text prompt cache")
    if (semantic_path is None) != (clip_path is None):
        parser.error("--semantic-prompt and --clip must be given together")
    if semantic_path is not None and text_prompt_path is not None:
        parser.error("Use either --semantic-prompt or --text-prompt for the positive text slot")
    if not math.isfinite(args.semantic_scale) or args.semantic_scale < 0:
        parser.error("--semantic-scale must be finite and nonnegative")
    clip_weight_path = clip_path
    if clip_path is not None and clip_path.is_dir():
        clip_weight_path = next((clip_path / name for name in ("model.safetensors", "pytorch_model.bin")
                                 if (clip_path / name).is_file()), None)
        if clip_weight_path is None:
            parser.error("--clip directory has no model.safetensors or pytorch_model.bin")
    text_slot_used = any(path is not None for path in (semantic_path, text_prompt_path, negative_text_prompt_path))
    import torch
    quality_guidance = None
    if args.qg_scale > 0 or args.plane_temperatures:
        from grn.models.sr_quality_guidance import SRQualityGuidance
        quality_guidance = SRQualityGuidance(scale=args.qg_scale, bound=args.qg_bound, start=args.qg_start,
            plane_weights=args.qg_planes, negative=args.qg_negative, lr_blur_sigma=args.qg_lr_sigma,
            lr_downscale=args.qg_lr_downscale, plane_temperatures=args.plane_temperatures)
        if quality_guidance.needs_negative_calibrator and (negative_adapter_path is None or adapter_path is None):
            parser.error("--qg-negative calibrator requires --adapter and --qg-negative-adapter")
    if negative_adapter_path is not None and (quality_guidance is None or not quality_guidance.needs_negative_calibrator):
        parser.error("--qg-negative-adapter requires --qg-scale > 0 with --qg-negative calibrator")
    if (quality_guidance is not None and quality_guidance.needs_negative_lr
            and quality_guidance.uses_negative_text and not text_slot_used):
        parser.error("--qg-negative prompt needs --semantic-prompt, --text-prompt or --negative-text-prompt")
    numeric_policy = {
        "backend": os.environ.get("GRN_S5_BACKEND", "unknown"), "torch": torch.__version__,
        "cuda": torch.version.cuda, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_enabled": torch.backends.cudnn.enabled, "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    destination = output_path(args.output)
    assets = {key: bundled(CONFIG[key]) for key in ("checkpoint", "hbq", "empty_text")}
    for path in assets.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    selected = {}
    records = []
    for benchmark in args.benchmarks:
        benchmark_pairs = pairs(benchmark)
        indices = list(range(len(benchmark_pairs)))
        if args.smoke:
            indices = [0, 1]
        elif args.indices:
            indices = [int(value) for value in args.indices.split(",")]
        if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(benchmark_pairs) for i in indices):
            raise ValueError(f"Invalid original indices for {benchmark}: {indices}")
        selected[benchmark] = indices
        for index in indices:
            hr, lr = benchmark_pairs[index]
            records.append({"benchmark": benchmark, "global_index": index, "seed": CONFIG["sampling"]["seed"] + index,
                            "filename": hr.name, "hr": str(hr.relative_to(ROOT)), "lr": str(lr.relative_to(ROOT))})
    rank = int(os.environ.get("RANK", "0"))
    signature = None
    validation_error = None
    # Rank zero validates once; its result is broadcast before any model loads.
    if rank == 0 or args.preflight:
        try:
            inventory = copy_index(ROOT)
            asset_identities = {}
            for key, path in assets.items():
                relative = str(path.relative_to(ROOT))
                if relative not in inventory:
                    raise ValueError(f"Asset has no verified copy record: {relative}")
                asset_identities[key] = {"path": relative, **verified_file(path, inventory[relative])}
            source = source_identity(ROOT, inventory)
            for record in records:
                lr_path = bundled(record["lr"])
                record["lr_sha256"] = sha256(lr_path)
                copied_lr = inventory.get(record["lr"])
                if copied_lr is None or copied_lr.get("destination_sha256_verified") is not True or record["lr_sha256"] != copied_lr["sha256"]:
                    raise ValueError(f"LR input is not the verified copied benchmark image: {record['lr']}")
            signature = {"assets": asset_identities, "source_sha256": source["sha256"],
                         "sampling": CONFIG["sampling"], "selected_indices": selected,
                         "input_records_sha256": digest_json(records),
                         "numeric_policy": numeric_policy,
                         "adaptation": {"kappa": args.kappa, "adapter": None if adapter_path is None else {
                             "path": str(adapter_path.relative_to(ROOT)), "bytes": adapter_path.stat().st_size,
                             "sha256": sha256(adapter_path)}}}
            if quality_guidance is not None:
                # Added only when requested: default runs keep their old signature.
                signature["adaptation"]["quality_guidance"] = {
                    **quality_guidance.metadata(),
                    "negative_adapter": None if negative_adapter_path is None else {
                        "path": str(negative_adapter_path.relative_to(ROOT)),
                        "bytes": negative_adapter_path.stat().st_size,
                        "sha256": sha256(negative_adapter_path)}}
            if text_slot_used:
                # Added only when requested: default runs keep their old signature.
                def file_identity(path):
                    return None if path is None else {"path": str(path.relative_to(ROOT)),
                                                      "bytes": path.stat().st_size, "sha256": sha256(path)}
                signature["adaptation"]["text_slot"] = {
                    "semantic_prompt": file_identity(semantic_path), "clip": file_identity(clip_weight_path),
                    "semantic_scale": args.semantic_scale, "text_prompt": file_identity(text_prompt_path),
                    "negative_text_prompt": file_identity(negative_text_prompt_path)}
        except Exception as error:
            validation_error = f"{type(error).__name__}: {error}"
    if args.preflight:
        if validation_error:
            raise RuntimeError(validation_error)
        print(json.dumps({"status": "preflight_passed", "signature": signature, "images": len(records)}, indent=2))
        return
    import numpy as np
    import torch
    import torch.distributed as distributed
    from PIL import Image
    from tools.grn_pipeline import GRNPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("Native S5 inference requires an available CUDA-compatible GPU")
    world_size = 1
    device = args.device
    if "RANK" in os.environ:
        rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        torch.cuda.set_device(device)
        distributed.init_process_group("nccl")
    else:
        torch.cuda.set_device(device)
    validation = [{"signature": signature, "records": records, "error": validation_error}]
    if world_size > 1:
        distributed.broadcast_object_list(validation, src=0)
    if validation[0]["error"]:
        raise RuntimeError(validation[0]["error"])
    signature, records = validation[0]["signature"], validation[0]["records"]
    signature_sha256 = digest_json(signature)
    metadata_path = destination / "no_cfg/generation.json"
    resume_state = [{"previous_records": {}, "error": None}]
    if rank == 0:
        try:
            previous_records = {}
            if metadata_path.exists():
                previous = json.loads(metadata_path.read_text())
                if not args.resume or previous.get("generation_signature") != signature:
                    raise RuntimeError("Existing generation requires --resume and an identical content signature; choose a new output directory")
                previous_records = {f"{record['benchmark']}/{record['filename']}": record for record in previous["records"]}
            elif (destination / "no_cfg").exists() and any((destination / "no_cfg").iterdir()):
                raise RuntimeError("Refusing outputs without a matching generation manifest")
            # Validate every image before loading the model or mutating the manifest.
            for record in records:
                output = destination / "no_cfg" / record["benchmark"] / record["filename"]
                if not output.resolve().is_relative_to(destination):
                    raise ValueError(f"Prediction path escapes output directory: {output}")
                if args.resume:
                    verify_existing_prediction(output, record, signature_sha256,
                                               previous_records.get(f"{record['benchmark']}/{record['filename']}"))
            resume_state[0]["previous_records"] = previous_records
            write_json(metadata_path, {"schema": "grn_s5_portable_generation_v2", "status": "generating",
                                      "generation_signature": signature, "records": records,
                                      "source_files": source["files"]})
        except Exception as error:
            resume_state[0]["error"] = f"{type(error).__name__}: {error}"
    if world_size > 1:
        distributed.broadcast_object_list(resume_state, src=0)
    if resume_state[0]["error"]:
        raise RuntimeError(resume_state[0]["error"])
    started = time.time()
    pipeline = GRNPipeline.from_pretrained(model_path=str(assets["checkpoint"]), vae_path=str(assets["hbq"]),
                                         sr_native_empty_text_path=str(assets["empty_text"]), task="SR", model="GRN2b",
                                         device=torch.device(device), torch_dtype=torch.bfloat16)
    adapter_report = None
    if adapter_path is not None:
        if sha256(adapter_path) != signature["adaptation"]["adapter"]["sha256"]:
            raise ValueError("Adapter changed after preflight")
        from grn.utils_t2iv.sr_trajectory_adapter import load_trajectory_adapter
        adapter_report = load_trajectory_adapter(adapter_path, pipeline.model,
            expected_base_sha256=signature["assets"]["checkpoint"]["sha256"], trainable=False)
    negative_report = None
    if negative_adapter_path is not None:
        recorded = signature["adaptation"]["quality_guidance"]["negative_adapter"]
        if sha256(negative_adapter_path) != recorded["sha256"]:
            raise ValueError("Negative calibrator changed after preflight")
        from grn.models.sr_quality_guidance import load_negative_calibrator
        negative_report = load_negative_calibrator(negative_adapter_path, pipeline.model,
            expected_base_sha256=signature["assets"]["checkpoint"]["sha256"],
            expected_positive_adapter_sha256=signature["adaptation"]["adapter"]["sha256"])
        calibrator_device = next(pipeline.model.sr_round_calibrator.parameters()).device
        pipeline.model.sr_negative_calibrator.to(device=calibrator_device, dtype=torch.float32)
    pipeline.model.eval()
    pipeline.vae.eval()
    adaptation_kwargs = {} if args.kappa is None else {"sr_calibrator_kappa": args.kappa}
    if quality_guidance is not None:
        adaptation_kwargs["sr_quality_guidance"] = quality_guidance
    semantic_report = None
    if text_slot_used:
        from grn.models.sr_semantic_prompt import (FrozenCLIPImageEncoder, SemanticPrompt,
                                                   load_semantic_prompt, load_text_prompt_cache)
        recorded = signature["adaptation"]["text_slot"]
        for path, key in ((semantic_path, "semantic_prompt"), (text_prompt_path, "text_prompt"),
                          (negative_text_prompt_path, "negative_text_prompt")):
            if path is not None and sha256(path) != recorded[key]["sha256"]:
                raise ValueError(f"{key} changed after preflight")
        if semantic_path is not None:
            from grn.utils_t2iv.sr_native import load_empty_text_cache
            clip_encoder = FrozenCLIPImageEncoder(clip_path)
            if clip_encoder.weight_sha256 != recorded["clip"]["sha256"]:
                raise ValueError("CLIP weights changed after preflight")
            empty_features = load_empty_text_cache(str(assets["empty_text"]), 4096)
            projector, semantic_report = load_semantic_prompt(
                semantic_path, expected_base_sha256=signature["assets"]["checkpoint"]["sha256"],
                expected_adapter_sha256=(None if adapter_path is None else signature["adaptation"]["adapter"]["sha256"]),
                clip_identity=clip_encoder.identity(), empty_features=empty_features)
            pipeline.sr_semantic_prompt = SemanticPrompt(
                clip_encoder, projector, empty_features,
                negative_trained=semantic_report["negative_trained"]).to(torch.device(device))
            adaptation_kwargs["sr_semantic_scale"] = args.semantic_scale
        if text_prompt_path is not None:
            adaptation_kwargs["sr_text_features"] = load_text_prompt_cache(text_prompt_path)[0]
        if negative_text_prompt_path is not None:
            adaptation_kwargs["sr_negative_text_features"] = load_text_prompt_cache(negative_text_prompt_path)[0]
    sampling = CONFIG["sampling"]
    for number, record in enumerate(records):
        if number % world_size != rank:
            continue
        output = destination / "no_cfg" / record["benchmark"] / record["filename"]
        if not output.resolve().is_relative_to(destination):
            raise ValueError(f"Prediction path escapes output directory: {output}")
        if args.resume and verify_existing_prediction(output, record, signature_sha256,
                resume_state[0]["previous_records"].get(f"{record['benchmark']}/{record['filename']}")):
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(bundled(record["lr"])) as handle:
            image = handle.convert("RGB")
            if image.size != (128, 128):
                raise ValueError(f"Unexpected LR image size: {record['lr']}")
            prediction = pipeline(lr_image=image, guidance_scale=1.0, temperature=1.0,
                                  complexity_aware_Tmin=sampling["steps"], complexity_aware_Tmax=sampling["steps"],
                                  complexity_aware_k=0, complexity_aware_b=sampling["steps"], complexity_aware_wp=0,
                                  snr_shift=1.0, seed=record["seed"], output_type="numpy", top_k=1, top_p=0.75, **adaptation_kwargs)
        if prediction.shape != (1, 512, 512, 3) or prediction.dtype != np.uint8:
            raise ValueError(f"Unexpected native output: {prediction.shape}, {prediction.dtype}")
        temporary = output.with_suffix(output.suffix + ".tmp")
        Image.fromarray(prediction[0], mode="RGB").save(temporary, format="PNG")
        temporary.replace(output)
        write_json(output.with_suffix(output.suffix + ".provenance.json"),
                   prediction_provenance(output, record, signature_sha256))
        print(json.dumps({"rank": rank, "benchmark": record["benchmark"], "global_index": record["global_index"], "output": str(output)}, ensure_ascii=False), flush=True)
    if world_size > 1:
        distributed.barrier()
    if rank == 0:
        for record in records:
            prediction = destination / "no_cfg" / record["benchmark"] / record["filename"]
            record["prediction_png_sha256"] = sha256(prediction)
            record["lr_sha256"] = sha256(bundled(record["lr"]))
        write_json(metadata_path, {"schema": "grn_s5_portable_generation_v2", "status": "complete",
                                  "generation_signature": signature, "records": records,
                                  "source_files": source["files"],
                                  "counts": {name: len(indices) for name, indices in selected.items()},
                                  "python": sys.executable, "torch": torch.__version__, "cuda": torch.version.cuda,
                                  "pipeline_module": str(Path(sys.modules[GRNPipeline.__module__].__file__).resolve()),
                                  "elapsed_seconds_including_model_load": time.time() - started,
                                  "adapter_load": adapter_report,
                                  "negative_adapter_load": negative_report,
                                  "semantic_prompt_load": semantic_report,
                                  "note": "Original benchmark indices are preserved; no RGB postprocessing or color correction."})
    if world_size > 1:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
