#!/usr/bin/env python3
"""Reserve original sources and prepare immutable U/T/R HR512/LR128 PNG pairs."""
import argparse
import hashlib
import json
from pathlib import Path


SOURCE_MANIFEST_SHA256 = 'af81ca691e23a820ec0b6dd82080a3bcf5deff184392767680cd421e6ad29bb4'
SOURCE_MANIFEST_COUNT = 94991
PREVIOUS_EXCLUSIONS_SHA256 = '9919a2580311eaa8ed00093b89244bc9e9d88ba487928f8a5696dd2eec35d751'
PREVIOUS_EXCLUSIONS_COUNT = 32


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
    temporary.replace(path)


def read_sources(path):
    return [line.strip() for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')]


def official_sources(manifest, previous_exclusions):
    """Production identity lock; exact file bytes also bind original order."""
    sources, old = read_sources(manifest), read_sources(previous_exclusions)
    if (len(sources) != SOURCE_MANIFEST_COUNT or file_hash(manifest) != SOURCE_MANIFEST_SHA256
            or len(old) != PREVIOUS_EXCLUSIONS_COUNT or file_hash(previous_exclusions) != PREVIOUS_EXCLUSIONS_SHA256):
        raise ValueError('Production requires the pinned full original manifest and exact old32 exclusion file')
    return sources


def split_source_hashes(root, train, validation, old):
    """Hash selected/old originals only; reject content overlap despite distinct paths."""
    hashes = {name: {source: file_hash(Path(root) / source) for source in sources}
              for name, sources in [('train', train), ('validation', validation), ('old', old)]}
    sets = {name: set(values.values()) for name, values in hashes.items()}
    if (any(len(sets[name]) != len(hashes[name]) for name in ('train', 'validation'))
            or sets['train'] & sets['validation'] or sets['train'] & sets['old']
            or sets['validation'] & sets['old']):
        raise ValueError('Selected original contents duplicate or overlap train/validation/old exclusions')
    return hashes


def select_originals(sources, *, protocol_root, previous_exclusions, seed=20260910,
                     train_count=2048, validation_count=64, pilot_count=16000):
    """Reconstruct both pilot epochs from the full ordered pool before filtering."""
    import numpy as np
    root = Path(protocol_root).resolve(strict=True)
    sources = list(sources)
    if (any(type(v) is not int or v < 0 for v in (seed, train_count, validation_count, pilot_count))
            or not train_count or not validation_count or pilot_count > len(sources)):
        raise ValueError('Invalid source selection counts/seed')
    resolved = [(root / source).resolve(strict=True) for source in sources]
    if (len(set(sources)) != len(sources) or len(set(resolved)) != len(resolved)
            or any(not path.is_file() for path in resolved)):
        raise ValueError('Duplicate/aliased or invalid source originals')
    old_paths = [(root / source).resolve(strict=True) for source in read_sources(previous_exclusions)]
    if not old_paths or len(set(old_paths)) != len(old_paths) or not set(old_paths) <= set(resolved):
        raise ValueError('Previous exclusions must uniquely identify originals from full manifest')
    source_by_path = dict(zip(resolved, sources))
    old = {source_by_path[path] for path in old_paths}
    pilot, epoch_hashes = set(), {}
    for epoch in (0, 1):
        order = sources.copy()
        np.random.default_rng(3407 + epoch).shuffle(order)
        chosen = order[:pilot_count]
        pilot.update(chosen)
        epoch_hashes[str(epoch)] = hashlib.sha256('\n'.join(chosen).encode()).hexdigest()
    validation_excluded = pilot | old
    pool = [source for source in sources if source not in validation_excluded]
    np.random.default_rng(seed).shuffle(pool)
    if len(pool) < validation_count:
        raise ValueError('Insufficient validation originals after pilot/old exclusions')
    validation = pool[:validation_count]
    training_excluded = old | set(validation)
    pool = [source for source in sources if source not in training_excluded]
    np.random.default_rng(seed).shuffle(pool)
    if len(pool) < train_count:
        raise ValueError('Insufficient training originals after validation exclusions')
    train = pool[:train_count]
    metadata = dict(seed=seed, source_manifest_count=len(sources),
        known_pilot_seed=3407, known_pilot_epochs=[0, 1], known_pilot_originals_per_epoch=pilot_count,
        known_pilot_originals=sorted(pilot), known_pilot_epoch_ids_sha256=epoch_hashes,
        previous_exclusions=sorted(old), future_training_exclusions=sorted(old | set(validation)),
        train_originals=train, validation_originals=validation,
        historical_isolation='New validation excludes known matched pilot epochs; earlier warmstart exposure unknown.',
        selection_rule='Validate full ordered manifest; new validation excludes old and pilot; train excludes old and new validation; each pool shuffled with local default_rng(seed).')
    return train, validation, metadata


def prepare_training_pairs(*, protocol_root, previous_exclusions, output,
                           manifest=Path('data/manifests/lsdir_ffhq10k.txt'),
                           config=Path('dataloader/params_realesrgan.yml'), seed=20260910,
                           train_count=2048, validation_count=64):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', completed_splits=[])
    atomic_json(output / 'status.json', status)
    try:
        root = Path(protocol_root).resolve(strict=True)
        manifest, config = root / manifest, root / config
        if (train_count, validation_count) not in ((2048, 64), (2, 2)):
            raise ValueError('Production pair counts must be full 2048/64 or explicit smoke 2/2')
        sources = official_sources(manifest, previous_exclusions)
        train, validation, metadata = select_originals(sources, protocol_root=root,
            previous_exclusions=previous_exclusions, seed=seed, train_count=train_count,
            validation_count=validation_count)
        metadata.update(protocol_root=str(root), source_manifest=str(manifest.resolve()),
            source_manifest_sha256=file_hash(manifest), previous_exclusions_path=str(Path(previous_exclusions).resolve()),
            previous_exclusions_sha256=file_hash(previous_exclusions), reference_config=str(config.resolve()),
            reference_config_sha256=file_hash(config),
            pair_preparation='Existing prepare_pairs on preselected manifests with pilot_epochs=(), pilot_count=0. Its seed shuffle determines emitted pair order; split lists record selection order.',
            training_pool_scope='Repeated fixed degradation mechanism pilot; not 16000 unique originals per epoch.')
        metadata['selected_source_sha256'] = split_source_hashes(root, train, validation, metadata['previous_exclusions'])
        atomic_json(output / 'source_split.json', metadata)
        with (output / 'future_training_exclusions.txt').open('x') as handle:
            handle.write('\n'.join(metadata['future_training_exclusions']) + '\n')
        # Heavy torchvision/reference-engine imports happen inside the live failure guard.
        from tools.prepare_sr_diagnostic_pairs import prepare_pairs, reference_engine
        engine = reference_engine(root, config)
        for name, sources in [('train', train), ('validation', validation)]:
            selected = output / (name + '_sources.txt')
            with selected.open('x') as handle:
                handle.write('\n'.join(sources) + '\n')
            prepare_pairs(root, selected.resolve(), config, output / name, samples=len(sources),
                          seed=seed, pilot_epochs=(), pilot_count=0, engine=engine)
            pair_path = output / name / 'pairs.json'
            payload = json.loads(pair_path.read_text())
            expected_hashes = metadata['selected_source_sha256'][name]
            if ({row['source_id'] for row in payload['pairs']} != set(sources)
                    or any(row['source_sha256'] != expected_hashes[row['source_id']] for row in payload['pairs'])):
                raise ValueError('Prepared source identities differ from reserved split')
            payload['metadata'].update(split_role=name, source_split=str((output / 'source_split.json').resolve()),
                source_split_sha256=file_hash(output / 'source_split.json'),
                historical_isolation=(metadata['historical_isolation'] if name == 'validation' else
                    'Training excludes old/new validation originals; known matched pilot originals may be reused.'))
            atomic_json(output / name / 'metadata.json', payload['metadata'])
            atomic_json(pair_path, payload)
            status['completed_splits'].append(name)
            atomic_json(output / 'status.json', status)
        status['status'] = 'complete'
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        atomic_json(output / 'status.json', status)
    return output / 'source_split.json'


def load_training_pairs(path):
    """Validate the completed, locked CPU preparation before cache use/publication."""
    from tools.sr_refinement_diagnostics import load_pairs
    path = Path(path).resolve(strict=True)
    payload = json.loads(path.read_text())
    metadata = payload.get('metadata', {})
    if (not isinstance(metadata, dict) or metadata.get('split_role') != 'train'
            or path.name != 'pairs.json' or path.parent.name != 'train'):
        raise ValueError('Cache requires the CPU producer training split, not validation/diagnostic pairs')
    preparation = path.parent.parent
    split_path = preparation / 'source_split.json'
    split_sha = file_hash(split_path)

    def check_split_metadata(meta, role, pair_path):
        link = meta.get('source_split') if isinstance(meta, dict) else None
        if (not isinstance(link, str) or not link or meta.get('split_role') != role
                or (pair_path.parent / link).resolve(strict=True) != split_path.resolve(strict=True)
                or meta.get('source_split_sha256') != split_sha):
            raise ValueError('Missing, stale or noncanonical CPU source_split provenance')

    check_split_metadata(metadata, 'train', path)
    status = json.loads((preparation / 'status.json').read_text())
    if status.get('status') != 'complete' or status.get('completed_splits') != ['train', 'validation']:
        raise ValueError('CPU pair preparation is not complete')
    split = json.loads(split_path.read_text())
    required = ('protocol_root', 'source_manifest', 'previous_exclusions_path', 'train_originals',
                'validation_originals', 'selected_source_sha256', 'seed')
    if not isinstance(split, dict) or any(key not in split for key in required):
        raise ValueError('Incomplete CPU source_split provenance')
    if (split.get('source_manifest_sha256') != SOURCE_MANIFEST_SHA256
            or split.get('previous_exclusions_sha256') != PREVIOUS_EXCLUSIONS_SHA256):
        raise ValueError('CPU source_split does not bind the pinned official originals/exclusions')
    sources = official_sources(split['source_manifest'], split['previous_exclusions_path'])
    train, validation = split['train_originals'], split['validation_originals']
    if (not isinstance(train, list) or not isinstance(validation, list)
            or (len(train), len(validation)) not in ((2048, 64), (2, 2))):
        raise ValueError('Cached preparation must reserve full 2048/64 or smoke 2/2 originals')
    root = Path(split['protocol_root']).resolve(strict=True)
    expected_train, expected_validation, expected = select_originals(sources, protocol_root=root,
        previous_exclusions=split['previous_exclusions_path'], seed=split['seed'],
        train_count=len(train), validation_count=len(validation))
    for key in ('source_manifest_count', 'known_pilot_seed', 'known_pilot_epochs',
                'known_pilot_originals_per_epoch', 'known_pilot_originals', 'known_pilot_epoch_ids_sha256',
                'previous_exclusions', 'future_training_exclusions', 'train_originals', 'validation_originals'):
        if split.get(key) != expected[key]:
            raise ValueError(f'CPU reserved original selection mismatch: {key}')
    hashes = split_source_hashes(root, expected_train, expected_validation, expected['previous_exclusions'])
    if split['selected_source_sha256'] != hashes:
        raise ValueError('CPU reserved source SHA evidence differs from actual disjoint originals')
    training_rows = None
    for role, originals in [('train', expected_train), ('validation', expected_validation)]:
        pair_path = preparation / role / 'pairs.json'
        split_payload = json.loads(pair_path.read_text())
        check_split_metadata(split_payload.get('metadata', {}), role, pair_path)
        summary = json.loads((pair_path.parent / 'summary.json').read_text())
        if (summary.get('status') != 'complete' or summary.get('expected_samples') != len(originals)
                or summary.get('completed_samples') != len(originals)):
            raise ValueError('Incomplete fixed PNG split preparation evidence')
        rows = load_pairs(pair_path)
        if len(rows) != len(originals) or {row['source_id'] for row in rows} != set(originals):
            raise ValueError('Fixed pair membership differs from reserved originals')
        for row in rows:
            if ((pair_path.parent / row['source_path']).resolve(strict=True) != (root / row['source_id']).resolve(strict=True)
                    or row.get('source_sha256') != hashes[role][row['source_id']]):
                raise ValueError('Fixed pair source path/SHA differs from reserved original')
            for kind in ('hr', 'lr'):
                if (Path(row[kind]) != (pair_path.parent / kind / (row['pair_id'] + '.png')).resolve(strict=True)
                        or row.get(kind + '_sha256') != file_hash(row[kind])):
                    raise ValueError('Fixed saved PNG path/SHA differs from prepared split')
        if role == 'train':
            training_rows = rows
    return training_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('protocol-root', 'previous-exclusions', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('data/manifests/lsdir_ffhq10k.txt'))
    parser.add_argument('--config', type=Path, default=Path('dataloader/params_realesrgan.yml'))
    parser.add_argument('--seed', type=int, default=20260910)
    parser.add_argument('--train-count', type=int, default=2048)
    parser.add_argument('--validation-count', type=int, default=64)
    print(prepare_training_pairs(**vars(parser.parse_args())))


if __name__ == '__main__':
    main()
