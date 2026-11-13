#!/usr/bin/env python3
"""HBQ reconstruction and read-only GRN loss/gradient diagnostics."""
from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tools.evaluate_grn_sr_alignvar import BENCHMARKS, benchmark_pairs
from tools.grn_pipeline import GRNPipeline
from tools.preview_grn_sr_checkpoint import evenly_spaced_indices
from tools.sr_diagnostic_core import gradient_comparison, residual_comparison, latent_error, fp32_residual_pre_hook
from grn.trainer.sft_trainer import _compute_sr_latent_losses, _hbq_bits_to_latent
from grn.utils_t2iv.hbq_util_t2iv import (
    raw_feature2bit_label, bit_label2raw_feature, multiclass_labels2onehot_input,
)
from grn.utils_t2iv.load import load_visual_tokenizer


def image_tensor(path, device):
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"), copy=True)
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255


def decoded_rgb(vae, latent):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        image = vae.decode(latent.float(), slice=True)
    if image.ndim == 5:
        image = image[:, :, 0]
    return ((image + 1) / 2).clamp(0, 1)


def png_tensor(image):
    return image.mul(255).clamp(0, 255).to(torch.uint8).float() / 255


def save_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = image[0].mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(path)


def make_metrics():
    import pyiqa
    return {
        "PSNR-Y": pyiqa.create_metric("psnry", device="cuda", crop_border=0),
        "SSIM-Y": pyiqa.create_metric("ssim", device="cuda", crop_border=0),
        "LPIPS": pyiqa.create_metric("lpips", device="cuda"),
        "DISTS": pyiqa.create_metric("dists", device="cuda"),
    }


def score(metrics, prediction, target):
    with torch.no_grad():
        return {name: float(metric(prediction, target).mean()) for name, metric in metrics.items()}


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        key = (row["benchmark"], row["mode"], row.get("signal_fraction"))
        groups[key].append(row)
    result = []
    for (benchmark, mode, signal), rows in groups.items():
        entry = {"benchmark": benchmark, "mode": mode, "signal_fraction": signal, "samples": len(rows)}
        for key in ("metrics", "latent_metrics", "feature_gradient", "pre_quant_parameter_gradient"):
            values = [r[key] for r in rows if key in r]
            if values:
                entry[key] = {
                    name: float(np.mean([v[name] for v in values if v[name] is not None]))
                    if any(v[name] is not None for v in values) else None
                    for name in values[0]
                }
                if key.endswith("gradient"):
                    cosines = [v["cosine"] for v in values if v["cosine"] is not None]
                    entry[key]["negative_cosine_fraction"] = (
                        sum(v < 0 for v in cosines) / len(cosines) if cosines else None
                    )
        result.append(entry)
    return result


def append_record(output, records, row):
    records.append(row)
    with (output / "records.jsonl").open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


def select_pairs(args, benchmark):
    pairs = benchmark_pairs(args.protocol_root, BENCHMARKS[benchmark])
    indices = range(len(pairs)) if args.samples == 0 else evenly_spaced_indices(len(pairs), args.samples)
    return [(index, *pairs[index]) for index in indices]


def hbq_probe(args, metrics, records):
    vae_args = GRNPipeline._get_default_args()
    vae_args.detail_scale_dim = 64
    vae_args.vae_path = str(args.vae)
    vae_args.video_frames = 1
    vae = load_visual_tokenizer(vae_args, device="cuda")
    for benchmark in args.benchmarks:
        for count, (index, hr_path, lr_path) in enumerate(select_pairs(args, benchmark), 1):
            gt, lr = image_tensor(hr_path, "cuda"), image_tensor(lr_path, "cuda")
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                raw_list, _, _ = vae.encode_for_raw_features((gt * 2 - 1).unsqueeze(2), scale_schedule=None, slice=True)
                raw = raw_list[0]
                hard = bit_label2raw_feature(raw_feature2bit_label(raw, 4), 4)
                images = {
                    "hbq_continuous": png_tensor(decoded_rgb(vae, raw)),
                    "hbq_quantized": png_tensor(decoded_rgb(vae, hard)),
                }
                with Image.open(lr_path) as handle:
                    bicubic = np.array(handle.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC), copy=True)
                images["bicubic"] = torch.from_numpy(bicubic).permute(2, 0, 1)[None].to("cuda").float() / 255
            for mode, prediction in images.items():
                row = {"benchmark": benchmark, "name": hr_path.name, "index": index, "mode": mode,
                       "metrics": score(metrics, prediction, gt)}
                append_record(args.output, records, row)
                save_image(prediction, args.output / mode / benchmark / hr_path.name)
            if count % 10 == 0 or count == 1:
                print(f"HBQ {benchmark}: {count} pairs completed", flush=True)
    del vae


def model_probe(args, metrics, records):
    pipe = GRNPipeline.from_pretrained(model_path=str(args.checkpoint), vae_path=str(args.vae),
                                      task="SR", model="GRN2b", device="cuda")
    model, vae = pipe.model, pipe.vae
    gate_records, hooks = [], []
    if args.fp32_residual:
        hooks.append(model.block_chunks[0].register_forward_pre_hook(
            fp32_residual_pre_hook, with_kwargs=True))
    phase = {"value": ""}
    raw_rms = {}

    def raw_hook(layer):
        def collect(module, inputs, output):
            raw_rms[layer] = float(output.detach().float().square().mean().sqrt())
        return collect

    def gate_hook(layer):
        def collect(module, inputs, output):
            incoming = float(inputs[0].detach().float().square().mean().sqrt())
            gated = abs(float(module.gate.detach())) * raw_rms[layer]
            gate_records.append({"phase": phase["value"], "layer": int(layer),
                                 "gate": float(module.gate.detach()), "window": module.window_size,
                                 "condition_is_zero": float(inputs[1].detach().abs().max()) == 0.0,
                                 "update_to_hr_rms": gated / max(incoming, 1e-12),
                                 **residual_comparison(inputs[0], output)})
        return collect

    for layer, module in model.sr_cross_attn.items():
        hooks += [module.out_proj.register_forward_hook(raw_hook(layer)),
                  module.register_forward_hook(gate_hook(layer))]
    # Restrict parameter-gradient measurement to the shared LR pre-quant projection.
    parameters = tuple(model.sr_condition_encoder.pre_quant.parameters())
    model.eval()
    for benchmark in args.benchmarks:
        for index, hr_path, lr_path in select_pairs(args, benchmark):
            gt, lr = image_tensor(hr_path, "cuda"), image_tensor(lr_path, "cuda")
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                raw_list, _, _ = vae.encode_for_raw_features((gt * 2 - 1).unsqueeze(2), scale_schedule=None, slice=True)
                labels = raw_feature2bit_label(raw_list[0], 4)
            target = labels.permute(0, 2, 3, 4, 1).reshape(1, 1024, 256).long()
            target_latent = _hbq_bits_to_latent(target.float(), 4, 64, 32, 32).unsqueeze(2)
            rope = pipe.get_visual_rope_embeds(model.rope2d_freqs_grid, (1, 32, 32),
                                             "cuda", 1.0, t_offset=0)
            for p in parameters:
                p.requires_grad_(True)
            for signal in args.signal_fractions:
                phase["value"] = f"teacher_signal_{signal}"
                rng = torch.Generator(device="cuda").manual_seed(args.seed + index)
                noise = torch.randint(0, 2, labels.shape, device="cuda", generator=rng)
                mask = torch.rand(labels.shape, device="cuda", generator=rng) < signal
                mixed = torch.where(mask, labels, noise)
                inputs = multiclass_labels2onehot_input(mixed, 2).reshape(1, 512, -1).transpose(1, 2)
                info = [{"cur_bits": 256, "cur_lvl": 2, "predict_tokens": 1024,
                         "all_tokens": 2049, "scale_token_id": float(mask.float().mean())}]
                with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(None, [inputs], lr_condition=lr * 2 - 1,
                                    visual_rope_cache=[rope], sequece_packing_scales=[[(1, 32, 32)]],
                                    super_scale_lengths=[2049], other_info_by_scale=info,
                                    gt_BL=[target], return_sr_aux=True)
                    aux = outputs[-1]
                    losses = _compute_sr_latent_losses(aux, [target], [[(1, 32, 32)]], pipe.args)
                    main = outputs[1].mean() + .25 * losses["latent_recon"] + .10 * losses["latent_structure"] + .05 * losses["latent_edge"]
                    align = .05 * losses["condition_alignment"]
                    variables = (aux["condition_latent"],) + parameters
                    main_grads = torch.autograd.grad(main, variables, retain_graph=True)
                    align_grads = torch.autograd.grad(align, variables)
                    feature_comparison = gradient_comparison(main_grads[0], align_grads[0])
                    parameter_comparison = gradient_comparison(
                        torch.cat([g.flatten() for g in main_grads[1:]]),
                        torch.cat([g.flatten() for g in align_grads[1:]]))
                    logits = aux["logits_by_scale"][0].detach().float()
                    probabilities = logits.softmax(1)[:, 1]
                    hard_bits = logits.argmax(1).float()
                base = {"benchmark": benchmark, "name": hr_path.name, "index": index,
                        "signal_fraction": signal, "feature_gradient": feature_comparison,
                        "pre_quant_parameter_gradient": parameter_comparison,
                        "main_loss": float(main.detach()), "weighted_alignment_loss": float(align.detach())}
                del main_grads, align_grads, outputs, aux, losses, main, align, variables
                for mode, bits in (("teacher_soft", probabilities), ("teacher_hard", hard_bits)):
                    latent = _hbq_bits_to_latent(bits, 4, 64, 32, 32).unsqueeze(2)
                    prediction = png_tensor(decoded_rgb(vae, latent))
                    append_record(args.output, records, dict(base, mode=mode, metrics=score(metrics, prediction, gt),
                                                             latent_metrics=latent_error(latent, target_latent)))
                    if index == select_pairs(args, benchmark)[0][0]:
                        save_image(prediction, args.output / f"{mode}_{signal}" / benchmark / hr_path.name)
                del logits, probabilities, hard_bits, inputs, prediction
            for p in parameters:
                p.requires_grad_(False)
            if args.inference_steps:
                phase["value"] = "actual_inference"
                captured = {}
                def capture_head(module, inputs, output):
                    captured["logits"] = output[:, 1024:2048].detach()
                handle = model.head.register_forward_hook(capture_head)
                with Image.open(lr_path) as image:
                    result = pipe(lr_image=image.convert("RGB"), guidance_scale=1., temperature=1.,
                                  complexity_aware_Tmin=args.inference_steps, complexity_aware_Tmax=args.inference_steps,
                                  complexity_aware_k=0, complexity_aware_b=args.inference_steps, complexity_aware_wp=0,
                                  snr_shift=1., seed=args.seed + index, output_type="numpy", top_k=1, top_p=.75)
                handle.remove()
                logits = captured["logits"].reshape(1, 1024, 256, 2).float()
                soft_bits, hard_bits = logits.softmax(-1)[..., 1], logits.argmax(-1).float()
                hard_reference = torch.from_numpy(result).permute(0, 3, 1, 2).to("cuda").float() / 255
                for mode, bits in (("inference_soft", soft_bits), ("inference_hard", hard_bits)):
                    latent = _hbq_bits_to_latent(bits, 4, 64, 32, 32).unsqueeze(2)
                    prediction = png_tensor(decoded_rgb(vae, latent))
                    row = {"benchmark": benchmark, "name": hr_path.name, "index": index, "mode": mode,
                           "metrics": score(metrics, prediction, gt),
                           "latent_metrics": latent_error(latent, target_latent)}
                    if mode == "inference_hard":
                        row["pipeline_pixel_max_difference"] = float((prediction - hard_reference).abs().max())
                        if row["pipeline_pixel_max_difference"] > 1.01 / 255:
                            raise RuntimeError(f"Diagnostic hard decoding differs from pipeline: {row}")
                    append_record(args.output, records, row)
                    save_image(prediction, args.output / mode / benchmark / hr_path.name)
                del captured, logits, soft_bits, hard_bits
            print(f"MODEL {benchmark}/{hr_path.name} completed", flush=True)
            gc.collect()
    for hook in hooks:
        hook.remove()
    (args.output / "gate_contributions.json").write_text(json.dumps(gate_records, indent=2) + "\n")
    del pipe, model, vae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("hbq", "model"), required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--protocol_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmarks", nargs="+", choices=("RealSR", "DRealSR"), default=["RealSR", "DRealSR"])
    parser.add_argument("--samples", type=int, default=6, help="per benchmark; 0 means all")
    parser.add_argument("--signal_fractions", nargs="+", type=float, default=[0., .5, .9])
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp32_residual", action="store_true", help="diagnostic-only: FP32 residual stream, BF16 operators")
    args = parser.parse_args()
    if args.mode == "model" and args.checkpoint is None:
        parser.error("--checkpoint is required for model probes")
    if args.samples < 0 or any(not 0 <= p <= 1 for p in args.signal_fractions):
        parser.error("Invalid sample count or signal fraction")
    if args.inference_steps not in (0,) and args.inference_steps < 2:
        parser.error("inference_steps must be zero or >=2")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "records.jsonl").exists():
        raise RuntimeError("Output already contains diagnostic records; choose a new output")
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata["gradient_scope"] = "shared condition latent and LR pre_quant parameters; not the full encoder"
    metadata["teacher_probe_warning"] = "teacher-corrupted GT inputs are diagnostics, not SR benchmark scores"
    metadata["metric_protocol"] = "PyIQA 0.1.12; PNG uint8; Y PSNR/SSIM; crop 0; no color correction"
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    torch.cuda.set_device(0)
    records = []
    metrics = make_metrics()
    (hbq_probe if args.mode == "hbq" else model_probe)(args, metrics, records)
    summary = {"status": "complete", "records": len(records), "groups": summarize(records)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

