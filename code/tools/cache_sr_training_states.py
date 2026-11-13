#!/usr/bin/env python3
"""Cache real original-precision 50-step rollouts; assemble only verified shards.

Each --output/shard_NNNN must be fresh. --assemble reserves output/assembly;
failed jobs cannot resume or overwrite. Only complete assembly publishes the
consumer's training_manifest.json. Pair seeds use the global input pair index.
"""
import argparse
import json
import os
from pathlib import Path

from tools.prepare_sr_state_training import atomic_json, file_hash, load_training_pairs


def shard_indices(count, shard_index, shards):
    if (any(type(v) is not int for v in (count, shard_index, shards))
            or not 0 < shards <= count or not 0 <= shard_index < shards):
        raise ValueError('Invalid shard count/index; empty shards are forbidden')
    return list(range(shard_index, count, shards))


def encode_target(vae, hr_path, *, rounds, device):
    """Re-read saved PNG using the exact fixed training loader arithmetic."""
    import numpy as np
    from PIL import Image
    import torch
    from grn.utils_t2iv.hbq_util_t2iv import raw_feature2bit_label
    with Image.open(hr_path) as image:
        if image.format != 'PNG' or image.mode != 'RGB' or image.size != (512, 512):
            raise ValueError('Target must be saved RGB HR512 PNG')
        pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div_(255)
    pixels = pixels.add(pixels).add_(-1).unsqueeze(0).unsqueeze(2).to(device)
    with torch.no_grad(), torch.autocast(torch.device(device).type, enabled=False):
        raw, _, _ = vae.encode_for_raw_features(pixels, scale_schedule=None, slice=True)
        labels = raw_feature2bit_label(raw[0], hbq_round=rounds)
    return labels.flatten(2).transpose(1, 2).contiguous().cpu().byte(), list(labels.shape[2:])


def cache_from_capture(cap, target, *, grid, pair, seed, validation_device):
    import torch
    from grn.utils_t2iv.sr_state_training import CAPTURE_STEPS, STATE_STEPS
    from tools.sr_sampling_precision import input_bits, top1_selection_counts
    if (cap.steps_seen != 50 or cap.multinomial_calls != 0 or set(cap.snapshots) != set(CAPTURE_STEPS)
            or any(len(getattr(cap, key)) != 50 for key in
                   ('update_mask_hashes', 'model_rng_at_updates', 'selected_bit_hashes', 'update_counts'))
            or [row['step'] for row in cap.update_counts] != list(range(1, 51))):
        raise ValueError('Incomplete real 50-step capture or unexpected multinomial calls')
    for snapshot in cap.snapshots.values():
        # Softmax rounding depends on device; audit where the actual sampler ran.
        top1_selection_counts(snapshot['logits'].to(device=validation_device),
                              snapshot['selected_bits'].to(device=validation_device))
    noise = input_bits(cap.snapshots[1]['inputs'], target.shape[-1]).byte()
    states = {}
    for step in STATE_STEPS:
        previous, actual = cap.snapshots[step - 1], cap.snapshots[step]
        state = dict(input_bits=input_bits(actual['inputs'], target.shape[-1]).byte(),
            previous_prediction=previous['selected_bits'].byte(), previous_mask=previous['update_mask'].clone(),
            signal_fraction=actual['signal_fraction'])
        if (not torch.equal(state['input_bits'], torch.where(state['previous_mask'], state['previous_prediction'], noise))
                or state['signal_fraction'] != state['previous_mask'].float().mean().item()):
            raise ValueError('Actual input/time does not match previous update')
        states[step] = state
    return dict(schema_version=1, pair_id=pair['pair_id'], source_id=pair['source_id'], grid=list(grid),
        target_bits=target.clone(), initial_noise=noise, states=states,
        provenance=dict(source_id=pair['source_id'], seed=seed, rollout_steps=50,
            **{key: pair[key] for key in ('source_sha256', 'hr_sha256', 'lr_sha256', 'checkpoint_sha256', 'vae_sha256')},
            rng_sha256=list(cap.model_rng_at_updates), mask_sha256=list(cap.update_mask_hashes),
            capture_counts={step: 1 for step in CAPTURE_STEPS}))


def capture_pair(pipe, pair, *, seed, verify_plain=False):
    import numpy as np
    import torch
    from grn.utils_t2iv.sr_state_training import CAPTURE_STEPS
    from tools.diagnose_sr_refinement import effective_protocol, _bits_latent
    from tools.sr_refinement_diagnostics import preserve_rng, verify_final_rgb
    from tools.sr_sampling_precision import capture_sampling, validate_experiment
    model = pipe.model
    device = next(model.parameters()).device
    validate_experiment(pipe.args, model.sr_condition_mode, [seed])
    if model.training or pipe.vae.training:
        raise ValueError('Frozen source model and VAE must be eval')
    with torch.no_grad(), preserve_rng(model):
        target, grid = encode_target(pipe.vae, pair['hr'], rounds=pipe.args.hbq_round, device=device)
        if target.shape != (1, 1024, 256) or grid != [1, 32, 32]:
            raise ValueError('Production HBQ target geometry mismatch')
        plain = None
        if verify_plain:
            with preserve_rng(model):
                plain = pipe(lr_image=pair['lr'], seed=seed, **effective_protocol(50))[0]
        with capture_sampling(model, CAPTURE_STEPS, 256, steps=50, stochastic=False, actual_seed=seed) as cap:
            rgb = pipe(lr_image=pair['lr'], seed=seed, **effective_protocol(50))[0]
        if plain is not None and not np.array_equal(plain, rgb):
            raise ValueError('Plain and captured actual PNG pixels differ')
        with torch.autocast(device.type, dtype=torch.bfloat16):
            latent = _bits_latent(cap.snapshots[50]['selected_bits'].to(device), grid, pipe.args.hbq_round)
            if not torch.equal(latent, cap.final_latent.to(device)):
                raise ValueError('Final selected bits differ from actual decoder latent')
            verify_final_rgb(model.summed_codes2images(pipe.vae, latent), rgb)
        cache = cache_from_capture(cap, target, grid=grid, pair=pair, seed=seed, validation_device=device)
        evidence = dict(multinomial_calls=cap.multinomial_calls, selected_final_latent_exact=True,
            pipeline_uint8_max_difference=0, plain_capture_uint8_max_difference=0 if verify_plain else None,
            selected_bit_hashes=cap.selected_bit_hashes, update_counts=cap.update_counts)
    return cache, rgb, evidence


def parameter_hashes(pipe):
    from grn.utils_t2iv.sr_state_training import tensor_hash
    return {f'{owner}.{name}': tensor_hash(value) for owner, module in [('model', pipe.model), ('vae', pipe.vae)]
            for name, value in module.named_parameters()}


def run_shard(*, checkpoint, vae, pairs, output, shard_index=0, shards=1, verify_plain_first=False):
    output = Path(output)
    if type(shard_index) is not int or type(shards) is not int or not 0 <= shard_index < shards:
        raise ValueError('Invalid shard index/count')
    folder = output / f'shard_{shard_index:04d}'
    folder.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', shard_index=shard_index, shards=shards, completed_indices=[])
    atomic_json(folder / 'status.json', status)
    try:
        identities = {name: dict(path=str(Path(path).resolve(strict=True)), sha256=file_hash(path))
                      for name, path in [('checkpoint', checkpoint), ('vae', vae), ('pairs', pairs)]}
        from PIL import Image
        import torch
        from tools.sr_refinement_diagnostics import preserve_rng
        from grn.utils_t2iv.sr_state_training import load_pair_state
        rows = load_training_pairs(pairs)
        indices = shard_indices(len(rows), shard_index, shards)
        seen_paths = set()
        for pair in rows:
            source = (Path(pairs).resolve().parent / pair['source_path']).resolve(strict=True)
            if source in seen_paths or file_hash(source) != pair['source_sha256']:
                raise ValueError('Duplicate source original or source hash mismatch')
            seen_paths.add(source)
            pair['source_path'] = str(source)
            for name in ('hr', 'lr'):
                if file_hash(pair[name]) != pair[name + '_sha256']:
                    raise ValueError('Missing or inconsistent fixed PNG hash')
            pair.update(checkpoint_sha256=identities['checkpoint']['sha256'], vae_sha256=identities['vae']['sha256'])
        status.update(identities=identities, expected_indices=indices, pair_count=len(rows), seed_base=20260910)
        atomic_json(folder / 'status.json', status)
        if not torch.cuda.is_available():
            raise RuntimeError('Real state production requires CUDA')
        from tools.grn_pipeline import GRNPipeline
        with preserve_rng():
            pipe = GRNPipeline.from_pretrained(model_path=str(checkpoint), vae_path=str(vae),
                                              task='SR', model='GRN2b', device='cuda')
            pipe.model.requires_grad_(False)
            pipe.vae.requires_grad_(False)
            before = parameter_hashes(pipe)
            try:
                with (folder / 'records.jsonl').open('x') as records:
                    for index in indices:
                        pair = rows[index]
                        cache, rgb, evidence = capture_pair(pipe, pair, seed=20260910 + index,
                            verify_plain=verify_plain_first and index == indices[0])
                        pair_dir = folder / pair['pair_id']
                        pair_dir.mkdir(exist_ok=False)
                        state_path, final_path = pair_dir / 'state.pt', pair_dir / 'sampled_final.png'
                        with state_path.open('xb') as handle:
                            torch.save(cache, handle)
                        load_pair_state(state_path)
                        Image.fromarray(rgb).save(final_path)
                        record = {key: pair[key] for key in ('pair_id', 'source_id', 'source_path', 'source_sha256', 'hr_sha256', 'lr_sha256')}
                        record.update(global_pair_index=index, seed=20260910 + index, hr_path=pair['hr'], lr_path=pair['lr'],
                            state_cache=str(state_path.resolve()), state_cache_sha256=file_hash(state_path),
                            sampled_final=str(final_path.resolve()), sampled_final_sha256=file_hash(final_path), evidence=evidence)
                        records.write(json.dumps(record, allow_nan=False) + '\n')
                        records.flush()
                        status['completed_indices'].append(index)
                        atomic_json(folder / 'status.json', status)
            finally:
                if parameter_hashes(pipe) != before:
                    raise RuntimeError('Frozen model/VAE parameters changed during shard')
        status.update(status='complete', parameters_unchanged=True, records_sha256=file_hash(folder / 'records.jsonl'))
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        atomic_json(folder / 'status.json', status)
    return folder


def assemble(*, output, shards=1):
    output = Path(output).resolve()
    if (output / 'training_manifest.json').exists():
        raise FileExistsError(output / 'training_manifest.json')
    folder = output / 'assembly'
    folder.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', shards=shards)
    atomic_json(folder / 'status.json', status)
    try:
        if type(shards) is not int or shards <= 0:
            raise ValueError('Invalid shard count')
        from grn.utils_t2iv.sr_state_training import load_state_manifest, load_pair_state
        records, common = [], None
        for index in range(shards):
            shard_dir = output / f'shard_{index:04d}'
            shard = json.loads((shard_dir / 'status.json').read_text())
            if (shard['status'] != 'complete' or shard['shards'] != shards or shard['shard_index'] != index
                    or not shard.get('parameters_unchanged') or shard.get('seed_base') != 20260910):
                raise ValueError('Incomplete or inconsistent shard status')
            if common is None:
                common = shard
            if shard['identities'] != common['identities'] or shard['pair_count'] != common['pair_count']:
                raise ValueError('Shard identities/counts differ')
            expected = shard_indices(shard['pair_count'], index, shards)
            if shard['expected_indices'] != expected or shard['completed_indices'] != expected:
                raise ValueError('Shard coverage mismatch')
            record_path = shard_dir / 'records.jsonl'
            if file_hash(record_path) != shard['records_sha256']:
                raise ValueError('Shard record hash mismatch')
            actual = [json.loads(line) for line in record_path.read_text().splitlines()]
            if [row['global_pair_index'] for row in actual] != expected:
                raise ValueError('Duplicate/missing global pair coverage')
            records.extend(actual)
        identities = common['identities']
        for identity in identities.values():
            if file_hash(identity['path']) != identity['sha256']:
                raise ValueError('Assembly source identity mismatch')
        source_pairs = load_training_pairs(identities['pairs']['path'])
        records.sort(key=lambda row: row['global_pair_index'])
        if len(source_pairs) != len(records) or len(records) != common['pair_count']:
            raise ValueError('Incomplete global coverage')
        for index, (row, pair) in enumerate(zip(records, source_pairs)):
            source_path = (Path(identities['pairs']['path']).parent / pair['source_path']).resolve(strict=True)
            if (row['global_pair_index'] != index or row['seed'] != 20260910 + index
                    or any(row[key] != pair[key] for key in ('pair_id', 'source_id', 'source_sha256', 'hr_sha256', 'lr_sha256'))
                    or Path(row['source_path']).resolve(strict=True) != source_path
                    or any(row[k + '_path'] != pair[k] for k in ('hr', 'lr'))):
                raise ValueError('Global pair identity/seed mismatch')
            evidence = row.get('evidence', {})
            if (evidence.get('multinomial_calls') != 0 or evidence.get('selected_final_latent_exact') is not True
                    or evidence.get('pipeline_uint8_max_difference') != 0):
                raise ValueError('Missing original top1/final decode verification evidence')
            if file_hash(row['sampled_final']) != row['sampled_final_sha256']:
                raise ValueError('Final PNG hash mismatch')
            if load_pair_state(row['state_cache'])['provenance']['seed'] != row['seed']:
                raise ValueError('Cached seed mismatch')
        manifest = dict(schema_version=1, status='complete', bit_shape=[1, 1024, 256], hr_size=512,
            lr_size=128, rollout_steps=50, state_steps=[20, 35, 45, 50],
            checkpoint=identities['checkpoint'], vae=identities['vae'], pairs=records)
        candidate = folder / 'validated_manifest.json'
        atomic_json(candidate, manifest)
        load_state_manifest(candidate)
        # All consumer identities and native previous-mask hashes verified before publication.
        os.link(candidate, output / 'training_manifest.json')
        status.update(status='complete', pair_count=len(records), manifest_sha256=file_hash(output / 'training_manifest.json'))
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        atomic_json(folder / 'status.json', status)
    return output / 'training_manifest.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assemble', action='store_true')
    for name in ('checkpoint', 'vae', 'pairs'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--verify-plain-first', action='store_true')
    args = parser.parse_args()
    if args.assemble:
        print(assemble(output=args.output, shards=args.shards))
    else:
        if any(getattr(args, key) is None for key in ('checkpoint', 'vae', 'pairs')):
            parser.error('--checkpoint, --vae and --pairs are required for cache production')
        kwargs = vars(args)
        kwargs.pop('assemble')
        print(run_shard(**kwargs))


if __name__ == '__main__':
    main()
