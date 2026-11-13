#!/usr/bin/env python3
"""Cache the original, byte-verified UMT5 encoder's actual empty prompt."""
import argparse
import json
from pathlib import Path

import torch

from tools.sr_diagnostic_core import GRN_UMT5_MANIFEST, GRN_UMT5_REVISION, verify_file_manifest


def cache_payload(features, dependency_hashes):
    if (not isinstance(features, torch.Tensor) or features.ndim != 2
            or features.shape[0] < 1 or features.shape[1] != 4096
            or not features.is_floating_point() or not torch.isfinite(features).all()):
        raise ValueError('Original empty encoder output must be finite floating [L,4096], L>0')
    return dict(schema_version=1, prompt='', features=features.detach().float().cpu().contiguous(),
                umt5_revision=GRN_UMT5_REVISION, dependency_hashes=dependency_hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text-encoder', type=Path, default=Path('/tmp/grn_pretrain_http_dependencies/umt5-xxl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'Refusing to overwrite cache: {args.output}')
    hashes = verify_file_manifest(args.text_encoder, GRN_UMT5_MANIFEST)
    print('Original UMT5 dependency bytes verified', flush=True)
    from grn.utils_t2iv.infer import load_tokenizer
    _, encoder = load_tokenizer(t5_path=str(args.text_encoder), device=args.device)
    encoder.model.eval().requires_grad_(False)
    with torch.no_grad():
        features = encoder([''], torch.device(args.device))
    if len(features) != 1:
        raise ValueError('Expected exactly one empty prompt result')
    payload = cache_payload(features[0], hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        torch.save(payload, handle)
    metadata = {key: value for key, value in payload.items() if key != 'features'}
    metadata.update(shape=list(payload['features'].shape), dtype=str(payload['features'].dtype),
                    text_encoder=str(args.text_encoder.resolve()), cache=str(args.output.resolve()))
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
