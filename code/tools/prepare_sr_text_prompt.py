#!/usr/bin/env python3
"""Cache the umT5 features of one real prompt for the SR text slot.

The SR model keeps GRN's text-to-image prompt pathway but always receives the
cached empty prompt. This tool encodes a real prompt with the same byte-verified
umT5 encoder, so it can be placed in the positive or negative text slot:

    scripts/infer.py --text-prompt P.pt --negative-text-prompt N.pt --qg-negative prompt --qg-scale 2

Typical use is a quality prompt against a low-quality prompt. Whether the SR
fine-tuned backbone still responds to real prompts must be measured; nothing is
claimed here.
"""
import argparse
import json
from pathlib import Path

import torch

from tools.sr_diagnostic_core import GRN_UMT5_MANIFEST, GRN_UMT5_REVISION, verify_file_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--text-encoder', type=Path, required=True,
                        help='Directory holding models_t5_umt5-xxl-enc-bf16.pth and umt5-xxl/')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if not args.prompt.strip():
        parser.error('--prompt must not be empty; the empty prompt is weights/native_empty_text.pt')
    if args.output.exists():
        parser.error(f'Refusing to overwrite cache: {args.output}')
    hashes = verify_file_manifest(args.text_encoder, GRN_UMT5_MANIFEST)
    print('Original UMT5 dependency bytes verified', flush=True)
    from grn.models.sr_semantic_prompt import text_prompt_payload
    from grn.utils_t2iv.infer import load_tokenizer
    _, encoder = load_tokenizer(t5_path=str(args.text_encoder), device=args.device)
    encoder.model.eval().requires_grad_(False)
    with torch.no_grad():
        features = encoder([args.prompt], torch.device(args.device))
    if len(features) != 1:
        raise ValueError('Expected exactly one prompt result')
    payload = text_prompt_payload(args.prompt, features[0].detach().float().cpu(), GRN_UMT5_REVISION, hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as handle:
        torch.save(payload, handle)
    metadata = {key: value for key, value in payload.items() if key != 'features'}
    metadata.update(shape=list(payload['features'].shape), cache=str(args.output.resolve()))
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
