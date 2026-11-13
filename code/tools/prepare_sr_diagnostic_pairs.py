#!/usr/bin/env python3
"""Prepare fixed, source-reserved CPU Real-ESRGAN diagnostic pairs.

Pair JSON schema: {"schema_version": 1, "metadata": {...}, "pairs": [
{"pair_id": "pair_0000", "source_id": "manifest/original.png", "source_path":
"/absolute/original.png", "seed": 20260909, "hr": "hr/pair_0000.png",
"lr": "lr/pair_0000.png", "source_sha256": "...", "input_sha256": "...",
"hr_sha256": "...", "lr_sha256": "..."}]}.
HR/LR paths are relative to the JSON; absolute paths are also accepted by the
diagnostic consumer. PNGs must be uint8 RGB, exactly 512/128 pixels respectively.
Known pilot exclusion reconstructs numpy.default_rng(3407+epoch).shuffle over
the full ordered manifest, then first16000 originals for epochs0,1. This matches
two-rank slicing with8000/rank. Earlier warmstart exposure remains unknown.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch

from tools.sr_refinement_diagnostics import file_hash, fresh_output, preserve_rng


def reference_engine(root, config):
    module_path = root / 'dataloader/realesrgan.py'
    spec = importlib.util.spec_from_file_location('grn_reference_realesrgan_diagnostic', module_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(root))
    basicsr = sys.modules.get('basicsr')
    if basicsr is None or not Path(basicsr.__file__).resolve().is_relative_to(root):
        raise RuntimeError('Reference degradation imported a non-reference BasicSR package')
    return module.RealESRGAN_degradation(str(config), device='cpu')


def prepare_pairs(protocol_root, manifest, config, output, samples=32, seed=20260909,
                  pilot_seed=3407, pilot_epochs=(0, 1), pilot_count=16000, engine=None):
    from torchvision.transforms import Compose, Resize, CenterCrop, RandomHorizontalFlip, InterpolationMode
    root, output = Path(protocol_root).resolve(), Path(output)
    manifest, config = root / manifest, root / config
    sources = [line.strip() for line in manifest.read_text().splitlines()
               if line.strip() and not line.startswith('#')]
    if len(set(sources)) != len(sources):
        raise ValueError('Duplicate source IDs in reference manifest')
    resolved = [(root / source).resolve() for source in sources]
    if len(set(resolved)) != len(resolved):
        raise ValueError('Manifest aliases duplicate source originals')
    if not 0 <= pilot_count <= len(sources) or samples < 1:
        raise ValueError('Invalid pilot or diagnostic sample count')
    excluded, epoch_ids = set(), {}
    for epoch in pilot_epochs:
        shuffled = list(sources)
        np.random.default_rng(pilot_seed + epoch).shuffle(shuffled)
        selected = shuffled[:pilot_count]
        excluded.update(selected)
        epoch_ids[str(epoch)] = hashlib.sha256('\n'.join(selected).encode()).hexdigest()
    available = [source for source in sources if source not in excluded]
    if samples > len(available):
        raise ValueError('Insufficient original sources after known pilot exclusion')
    np.random.default_rng(seed).shuffle(available)
    chosen = available[:samples]
    fresh_output(output)
    metadata = dict(protocol_root=str(root), manifest=str(manifest), manifest_sha256=file_hash(manifest),
        reference_config=str(config), reference_config_sha256=file_hash(config), seed=seed,
        expected_samples=samples, source_manifest_count=len(sources),
        known_pilot_excluded_count=len(excluded), known_pilot_seed=pilot_seed,
        known_pilot_epochs=list(pilot_epochs), known_pilot_originals_per_epoch=pilot_count,
        known_pilot_epoch_ids_sha256=epoch_ids,
        exclusion_sha256=hashlib.sha256('\n'.join(sorted(excluded)).encode()).hexdigest(),
        historical_isolation='Known pilot selected originals excluded; earlier warmstart exposure unknown.',
        transform='torchvision Resize(640,LANCZOS), CenterCrop(512), RandomHorizontalFlip(.5); reference degradation may flip HR again',
        degradation='reference dataloader/realesrgan.py, CPU, resize_bak=False; no fallback',
        seed_rule='Python, NumPy legacy, torch.manual_seed(seed + pair index) before HR transform and degradation',
        png_convention='clip [0,1], multiply255, round, uint8 RGB')
    source_code = root / 'dataloader/realesrgan.py'
    if source_code.is_file():
        metadata['reference_engine_sha256'] = file_hash(source_code)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    rows, summary = [], dict(status='running', expected_samples=samples, completed_samples=0)
    try:
        with preserve_rng():
            engine = engine if engine is not None else reference_engine(root, config)
            transform = Compose([Resize(640, interpolation=InterpolationMode.LANCZOS), CenterCrop(512),
                                 RandomHorizontalFlip(.5)])
            for index, source in enumerate(chosen):
                pair_seed = seed + index
                random.seed(pair_seed)
                np.random.seed(pair_seed % (2 ** 32))
                torch.manual_seed(pair_seed)
                source_path = (root / source).resolve()
                source_digest = file_hash(source_path)
                with Image.open(source_path) as image:
                    pixels = np.asarray(transform(image.convert('RGB')), dtype=np.float32) / 255.
                hr, lr = engine.degrade_process(pixels, resize_bak=False)
                row = dict(pair_id=f'pair_{index:04d}', source_id=source, source_path=str(source_path),
                           source_sha256=source_digest, seed=pair_seed,
                           input_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
                for kind, value, size in [('hr', hr, 512), ('lr', lr, 128)]:
                    if value.shape != (1, 3, size, size) or not torch.isfinite(value).all():
                        raise ValueError(f'Reference engine returned invalid {kind} shape/values')
                    relative = f'{kind}/{row["pair_id"]}.png'
                    path = output / relative
                    path.parent.mkdir(exist_ok=True)
                    array = value[0].detach().cpu().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
                    Image.fromarray(array).save(path)
                    row[kind], row[kind + '_sha256'] = relative, file_hash(path)
                rows.append(row)
                with (output / 'records.jsonl').open('a') as handle:
                    handle.write(json.dumps(row) + '\n')
                summary['completed_samples'] = len(rows)
                (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        (output / 'future_training_exclusions.txt').write_text('\n'.join(chosen) + '\n')
        (output / 'pairs.json').write_text(json.dumps(dict(schema_version=1, metadata=metadata, pairs=rows), indent=2) + '\n')
        summary['status'] = 'complete'
    except Exception as error:
        summary.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return output / 'pairs.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--protocol-root', '--protocol_root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('data/manifests/lsdir_ffhq10k.txt'))
    parser.add_argument('--config', type=Path, default=Path('dataloader/params_realesrgan.yml'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20260909)
    parser.add_argument('--pilot-seed', type=int, default=3407)
    parser.add_argument('--pilot-epochs', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--pilot-count', type=int, default=16000)
    args = parser.parse_args()
    print(prepare_pairs(**vars(args)))


if __name__ == '__main__':
    main()
