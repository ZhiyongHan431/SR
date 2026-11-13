#!/usr/bin/env python3
"""CPU-only verification of relocated S5 data identity and historical random seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from grn.utils_t2iv.s5_bundle_paths import resolve_path
from grn.utils_t2iv import sr_v2_qualified_data as data

SHA = '43c0d463cd1f976ce1b7b4946061908034b9ca310f223e20b62375d5a5dc4e60'


def verify(full_manifest=False, limit=0):
    path = ROOT / 'manifests/qualification_v1/qualified_manifest.json'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA:
        raise ValueError('Original qualification manifest bytes changed')
    manifest = json.loads(raw)
    train_raw = data.checked_binding(manifest['artifacts']['train'], base=path.parent)
    train = json.loads(train_raw)
    sources = data.QualifiedSources(str(path), SHA, manifest['protocol_root'], tuple(train['rows']),
        manifest['execution']['max_source_bytes'], manifest['execution']['max_pixels'])
    if len(sources.rows) != 93854:
        raise ValueError('Qualified source count changed')
    if full_manifest:
        audited = data.load_qualified_manifest(path, SHA)
        if audited.rows != sources.rows:
            raise ValueError('Full manifest audit differs from the sampled identity')
    keys = ('source_id', 'source_path', 'source_sha256', 'decoded_rgb_sha256', 'ordinal',
            'epoch', 'global_position', 'epoch_seed', 'world_size', 'accumulation',
            'padded_count', 'occurrence_id', 'rank', 'rank_position', 'padding',
            'crop_seed', 'degradation_seed', 'corruption_seed')
    epochs, counts = {}, {}
    for rank in (0, 1):
        count = 0
        history = ROOT / f'reference/s5_run/qualified_consumption/sr_qualified_consumption_rank{rank}.jsonl'
        with history.open() as stream:
            for line in stream:
                previous = json.loads(line)
                epoch = previous['epoch']
                if epoch not in epochs:
                    epochs[epoch] = data.epoch_occurrences(sources, epoch=epoch, seed=3407,
                                                          world_size=2, accumulation=16)
                current = epochs[epoch][previous['global_position']]
                if previous['qualification_manifest_sha256'] != SHA:
                    raise ValueError('Historical qualification identity differs')
                for key in keys:
                    if current[key] != previous[key]:
                        raise ValueError(f'Seed/order mismatch rank={rank}, row={count}, field={key}')
                if count < 4:
                    local = resolve_path(current['source_path'], strict=True)
                    expected = ROOT / 'protocol' / current['source_id']
                    if local != expected.resolve(strict=True):
                        raise ValueError('Image path failed independent relocation')
                count += 1
                if limit and count >= limit:
                    break
        counts[str(rank)] = count
        if not limit and count != 93856:
            raise ValueError('Historical full-run record count differs')
    return {'status': 'passed', 'gpu_used': False, 'model_loaded': False,
            'original_qualified_manifest_sha256': SHA, 'manifest_bytes_unchanged': True,
            'qualified_samples': len(sources.rows), 'historical_records_checked': counts,
            'checked_fields': list(keys), 'full_manifest_integrity_audited': full_manifest,
            'crop_degradation_corruption_seeds_exact': True,
            'image_pixels_recomputed': False, 'training_executed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-manifest', action='store_true',
                        help='Also run the original complete artifact/protected-file integrity audit')
    parser.add_argument('--records-per-rank', type=int, default=0, help='0 checks all 93856 records per rank')
    args = parser.parse_args()
    if args.records_per_rank < 0:
        parser.error('--records-per-rank must be nonnegative')
    result = verify(args.full_manifest, args.records_per_rank)
    output = ROOT / 'verification/qualified_seed_check.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
