"""Portable paths and offline assets; no model or metric math is changed."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "manifests/bundle.json").read_text())
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def bundled(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path leaves the S5 bundle: {relative}")
    return path


def setup() -> None:
    os.environ["GRN_S5_ROOT"] = str(ROOT)
    os.environ["TORCH_HOME"] = str(bundled(CONFIG["offline_cache"]) / "torch")
    os.environ["HF_HOME"] = str(bundled(CONFIG["offline_cache"]) / "huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "offline"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    sys.path.insert(0, str(bundled(CONFIG["code"])))


def output_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT / "outputs"):
        raise ValueError("Write outputs under GRN_S5/outputs; reference assets stay read-only")
    return path


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pairs(name: str):
    spec = CONFIG["benchmarks"][name]
    directory = bundled(CONFIG["protocol_root"]) / spec["relative_root"]
    for hr_name, lr_name in (("gt", "lq"), ("test_HR", "test_LR")):
        hr, lr = directory / hr_name, directory / lr_name
        if hr.is_dir() and lr.is_dir():
            break
    else:
        raise FileNotFoundError(f"Missing paired benchmark directories: {directory}")
    hr_files = sorted(p for p in hr.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    lr_files = {p.name: p for p in lr.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    if len(hr_files) != spec["count"] or {p.name for p in hr_files} != set(lr_files):
        raise ValueError(f"Unexpected pair identities/count in {name}")
    return [(p, lr_files[p.name]) for p in hr_files]


def metric_asset_paths() -> dict[str, Path]:
    cache = bundled(CONFIG["offline_cache"])
    names = (
        "pyiqa/LPIPS_v0.1_alex-df73285e.pth", "pyiqa/DISTS_weights-f5e65c96.pth",
        "pyiqa/ckpt_koniq10k.pt", "pyiqa/musiq_koniq_ckpt-e95806b9.pth",
        "pyiqa/pt_inception-2015-12-05-6726825d.pth", "clip/RN50.pt",
        "torch/hub/checkpoints/alexnet-owt-7be5be79.pth", "torch/hub/checkpoints/vgg16-397923af.pth",
        "huggingface/hub/models--timm--vit_base_patch8_224.augreg2_in21k_ft_in1k/refs/main",
    )
    result = {}
    for name in names:
        path = (cache / name).resolve()
        if not path.is_relative_to(cache) or not path.is_file():
            raise FileNotFoundError(f"Missing or external offline metric asset: {cache / name}")
        result[name] = path
    revision_file = result[names[-1]]
    revision = revision_file.read_text().strip()
    if not revision or "/" in revision or ".." in revision:
        raise ValueError("Invalid packaged MANIQA backbone cache revision")
    model = revision_file.parent.parent / "snapshots" / revision / "model.safetensors"
    model = model.resolve()
    if not model.is_relative_to(cache) or not model.is_file():
        raise FileNotFoundError(f"Missing or external MANIQA backbone: {model}")
    result["maniqa_backbone"] = model
    return result


def offline_metric_assets() -> dict:
    """Redirect existing PyIQA loaders to identical package-local weight bytes.

    PyIQA 0.1.12 hardcodes ~/.cache in two helper functions. Redirect these
    lookups without changing HOME, architectures, preprocessing, or scores.
    No missing asset is downloaded.
    """
    local_assets = metric_asset_paths()
    import pyiqa
    from pyiqa.utils import download_util
    from pyiqa.archs import clip_model
    from urllib.parse import urlparse

    if not Path(pyiqa.__file__).resolve().is_relative_to(ROOT / "environment"):
        raise RuntimeError("Use GRN_S5/environment/python: importing PyIQA from outside the bundle is not allowed")
    cache = bundled(CONFIG["offline_cache"])
    original = download_util.load_file_from_url

    def local_file(url, model_dir=None, progress=True, file_name=None):
        name = file_name or Path(urlparse(url).path).name
        path = (cache / "pyiqa" / name).resolve()
        if not path.is_relative_to(cache) or not path.is_file():
            raise FileNotFoundError(f"Missing offline PyIQA asset: {path}")
        return str(path)

    redirects = []
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("pyiqa") and module is not None:
            for name, value in list(vars(module).items()):
                if value is original:
                    setattr(module, name, local_file)
                    redirects.append(f"{module_name}.{name}")

    def local_clip(url, root):
        path = (cache / "clip" / Path(urlparse(url).path).name).resolve()
        if not path.is_relative_to(cache) or not path.is_file():
            raise FileNotFoundError(f"Missing offline CLIP asset: {path}")
        expected = url.rstrip("/").split("/")[-2]
        if len(expected) == 64 and sha256(path) != expected:
            raise ValueError(f"CLIP weight SHA256 mismatch: {path}")
        return str(path)

    clip_model._download = local_clip
    return {"pyiqa_version": pyiqa.__version__, "cache": str(cache), "loader_redirects": redirects,
            "verified_local_asset_paths": {name: str(path.relative_to(ROOT)) for name, path in local_assets.items()},
            "note": "Only local weight lookup paths are redirected; metric options and computation are unchanged."}
