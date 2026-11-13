#!/usr/bin/env python3
"""Durable expanded C/R experiment: caches, training, full benchmarks, report.

The coordinator uses only the standard library. Model math stays in the tested
public entry points. A completed task is skipped on coordinator restart; failed
tasks are retained for diagnosis rather than silently overwriting their data.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / 'environment/python')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def absolute_pairs(manifest, parent):
    result = copy.deepcopy(manifest)
    for rows in result['splits'].values():
        for row in rows:
            for key in ('hr', 'lr', 'source_path'):
                row[key] = str((parent / row[key]).resolve())
    return result


def merge_caches(pair_path, shard_paths, destination):
    """Preserve tensor files/identities; merge only the small cache manifests."""
    pairs = read(pair_path)
    caches = [read(path) for path in shard_paths]
    keys = ('schema', 'status', 'base_sha256', 'vae_sha256',
            'teacher_adapter_sha256', 'teacher_adapter', 'state_steps',
            'rollout_steps', 'state_policy', 'degradation', 'execution_identity')
    for cache in caches:
        if cache['status'] != 'complete' or any(cache[k] != caches[0][k] for k in keys):
            raise ValueError('Shard cache policies/teachers differ')
    merged = copy.deepcopy(caches[0])
    merged.update(pairs_path=str(pair_path.resolve()), pairs_sha256=digest(pair_path), records=[])
    records = {}
    for path, cache in zip(shard_paths, caches):
        if digest(Path(cache['pairs_path'])) != cache['pairs_sha256']:
            raise ValueError('Shard pair manifest changed')
        for item in cache['records']:
            row = copy.deepcopy(item)
            if row['pair_id'] in records:
                raise ValueError('Duplicate cached pair')
            for key in ('state_path', 'feature_path'):
                source = (path.parent / row[key]).resolve()
                if not source.is_relative_to(ROOT) or not source.is_file():
                    raise ValueError('Missing/external tensor cache')
                row[key] = os.path.relpath(source, destination.parent)
            records[row['pair_id']] = row
    expected = pairs['splits']['train']
    if set(records) != {r['pair_id'] for r in expected}:
        raise ValueError('Merged cache does not cover training pairs exactly')
    for original in expected:
        row = records[original['pair_id']]
        for key in ('source_id', 'source_sha256', 'hr_sha256', 'lr_sha256'):
            if row[key] != original[key]:
                raise ValueError('Merged pair identity mismatch')
        merged['records'].append(row)
    merged['storage_estimate'] = {'image_count': len(expected), 'scope': 'Merged sharded cache'}
    merged['shards'] = [{'path': str(p), 'sha256': digest(p)} for p in shard_paths]
    write(destination, merged)


class Experiment:
    def __init__(self, output):
        self.output = output
        self.state_path = output / 'experiment_status.json'
        self.state = read(self.state_path) if self.state_path.exists() else {
            'status': 'running', 'tasks': {}, 'started_utc': datetime.now(timezone.utc).isoformat()}
        paths = list((ROOT / 'code').rglob('*.py'))
        paths += list((ROOT / 'scripts').glob('*.py'))
        paths += [ROOT / 'environment/python-maca', ROOT / 'environment/bootstrap/maca_numeric_policy.py']
        sources = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
        previous = self.state.setdefault('source_sha256', sources)
        if previous != sources:
            raise ValueError('Experiment source files changed since coordinator launch')

    def update(self, **values):
        self.state.update(values, updated_utc=datetime.now(timezone.utc).isoformat())
        write(self.state_path, self.state)

    def task(self, name, args, gpu=0):
        return {'name': name, 'argv': [PYTHON, *map(str, args)], 'gpu': gpu}

    def run(self, stage, tasks):
        if any(digest(ROOT / name) != sha for name, sha in self.state['source_sha256'].items()):
            raise ValueError('Experiment source files changed during the run')
        self.update(status='running', stage=stage)
        pending = []
        for task in tasks:
            prior = self.state['tasks'].get(task['name'])
            identity = {'argv': task['argv'], 'gpu': task['gpu']}
            if prior and any(prior[k] != v for k, v in identity.items()):
                raise ValueError('Task command changed: ' + task['name'])
            if prior and prior['status'] == 'complete':
                continue
            if prior:
                raise RuntimeError('Incomplete task requires diagnosis before restart: ' + task['name'])
            pending.append(task)
        active = {}
        try:
            while pending or active:
                occupied = {task['gpu'] for task, proc, handle in active.values()}
                for task in pending[:]:
                    if task['gpu'] in occupied:
                        continue
                    log = self.output / 'logs' / (task['name'] + '.log')
                    log.parent.mkdir(parents=True, exist_ok=True)
                    handle = log.open('x')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(task['gpu']),
                               S5_BUNDLE_GUARD='1', PYTHONUNBUFFERED='1')
                    proc = subprocess.Popen(task['argv'], cwd=ROOT, env=env, stdout=handle,
                                            stderr=subprocess.STDOUT)
                    self.state['tasks'][task['name']] = dict(task, pid=proc.pid,
                        status='running', log=str(log), started_utc=datetime.now(timezone.utc).isoformat())
                    active[task['name']] = (task, proc, handle)
                    occupied.add(task['gpu']); pending.remove(task)
                    print('START', task['name'], 'GPU', task['gpu'], flush=True)
                for name, (task, proc, handle) in list(active.items()):
                    code = proc.poll()
                    if code is None:
                        continue
                    handle.close(); del active[name]
                    self.state['tasks'][name].update(status='complete' if code == 0 else 'failed',
                        exit_code=code, ended_utc=datetime.now(timezone.utc).isoformat())
                    self.update()
                    print('END', name, code, flush=True)
                    if code:
                        raise RuntimeError('Task failed: ' + name)
                self.update()
                if active:
                    time.sleep(10)
        finally:
            for name, (task, proc, handle) in active.items():
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
                handle.close()
                self.state['tasks'][name].update(status='interrupted', exit_code=proc.returncode)
            self.update()

    def cache(self, name, pairs, adapter=None):
        manifest = absolute_pairs(read(pairs), pairs.parent)
        tasks, shards = [], []
        for gpu in range(8):
            subset = copy.deepcopy(manifest)
            subset['splits']['train'] = subset['splits']['train'][gpu::8]
            pair_shard = self.output / 'splits' / f'train_{gpu}.json'
            write(pair_shard, subset)
            target = self.output / name / f'shard_{gpu}'
            args = ['scripts/adapt.py', 'cache', '--pairs', pair_shard, '--output', target,
                    '--verify-plain', '--execute']
            if adapter:
                args += ['--adapter', adapter]
            tasks.append(self.task(f'{name}_{gpu}', args, gpu))
            shards.append(target / 'cache.json')
        self.run(name, tasks)
        merged = self.output / name / 'cache.json'
        merge_caches(pairs, shards, merged)
        return merged

    def inference(self, name, adapter=None):
        config = read(ROOT / 'manifests/bundle.json')
        tasks, shards = [], []
        for benchmark, spec in config['benchmarks'].items():
            for gpu in range(8):
                target = self.output / 'predictions' / name / 'shards' / benchmark / str(gpu)
                args = ['scripts/infer.py', '--benchmarks', benchmark, '--indices',
                    ','.join(map(str, range(gpu, spec['count'], 8))), '--kappa', '2', '--output', target]
                if adapter:
                    args += ['--adapter', adapter]
                tasks.append(self.task(f'infer_{name}_{benchmark}_{gpu}', args, gpu))
                shards.append((benchmark, target))
        self.run('inference_' + name, tasks)
        combined = self.output / 'predictions' / name / 'full'
        records, child_manifests, signature = [], [], None
        for benchmark, target in shards:
            receipt = target / 'no_cfg/generation.json'
            child = read(receipt)
            if child['status'] != 'complete':
                raise ValueError('Incomplete inference shard')
            identity = {key: child['generation_signature'][key] for key in
                        ('assets', 'source_sha256', 'sampling', 'numeric_policy', 'adaptation')}
            if signature is not None and signature != identity:
                raise ValueError('Inference shard code/model/numeric policies differ')
            signature = identity
            child_manifests.append({'path': str(receipt), 'sha256': digest(receipt)})
            for row in child['records']:
                source = target / 'no_cfg' / benchmark / row['filename']
                if digest(source) != row['prediction_png_sha256']:
                    raise ValueError('Prediction changed')
                dest = combined / 'no_cfg' / benchmark / row['filename']
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.is_symlink():
                    if dest.resolve() != source.resolve():
                        raise ValueError('Unexpected prediction link')
                elif dest.exists():
                    raise ValueError('Refusing prediction overwrite')
                else:
                    dest.symlink_to(os.path.relpath(source, dest.parent))
                records.append(row)
        for benchmark, spec in config['benchmarks'].items():
            selected = [r for r in records if r['benchmark'] == benchmark]
            if len(selected) != spec['count'] or {r['global_index'] for r in selected} != set(range(spec['count'])):
                raise ValueError('Benchmark coverage is not exact')
        write(combined / 'no_cfg/sharded_generation.json', {
            'status': 'complete', 'adapter': str(adapter) if adapter else None,
            'adapter_sha256': digest(adapter) if adapter else None, 'kappa': 2.,
            'shards': child_manifests, 'records': records})
        return combined

    def execute(self):
        out = self.output
        pair_path = out / 'pairs/pairs.json'
        pair_manifest = read(pair_path)
        if pair_manifest['status'] != 'complete' or len(pair_manifest['splits']['train']) != 2048:
            raise ValueError('Expected complete 2048-image training preparation')
        base_cache = self.cache('cache_base', pair_path)
        c_dir, r_dir = out / 'train_c', out / 'train_cr'
        self.run('train_C', [self.task('train_C', ['scripts/adapt.py', 'train', '--cache', base_cache,
            '--arm', 'C', '--updates', '16384', '--save-every', '256', '--lr', '5e-5',
            '--seed', '20261109', '--output', c_dir, '--execute'])])
        c_adapter = c_dir / 'adapter.pt'
        c_cache = self.cache('cache_c', pair_path, c_adapter)
        self.run('train_R', [self.task('train_R', ['scripts/adapt.py', 'train', '--cache', c_cache,
            '--arm', 'R', '--updates', '4096', '--save-every', '256', '--lr', '5e-5',
            '--kappa-min', '2', '--kappa-max', '2', '--seed', '20261109',
            '--output', r_dir, '--execute'])])
        r_adapter = r_dir / 'adapter.pt'
        # All final checkpoints are evaluated, regardless of observed score direction.
        predictions = {name: self.inference(name, adapter) for name, adapter in
                       [('S5', None), ('C', c_adapter), ('CR', r_adapter)]}
        self.run('full_metrics', [self.task('metrics_' + name, ['scripts/evaluate.py', '--input', path,
            '--output', out / 'metrics' / name, '--batch-size', '8'], gpu)
            for gpu, (name, path) in enumerate(predictions.items())])
        absolute = absolute_pairs(pair_manifest, pair_path.parent)
        tasks = []
        for gpu in range(8):
            subset = copy.deepcopy(absolute)
            subset['splits']['validation'] = subset['splits']['validation'][gpu::8]
            path = out / 'splits' / f'validation_{gpu}.json'; write(path, subset)
            tasks.append(self.task(f'internal_validation_{gpu}', ['scripts/evaluate_adapter.py',
                '--pairs', path, '--adapter', f'C={c_adapter}', '--adapter', f'CR={r_adapter}',
                '--kappas', '2', '--output', out / 'internal_validation' / str(gpu), '--execute'], gpu))
        self.run('internal_validation', tasks)
        self.report(c_adapter, r_adapter)

    def report(self, c_adapter, r_adapter):
        out = self.output
        results = {name: read(out / 'metrics' / name / 'no_cfg/metrics.json')['results']
                   for name in ('S5', 'C', 'CR')}
        config = read(ROOT / 'manifests/bundle.json')
        differences = {name: {bench: {key: value - results['S5'][bench][key]
            for key, value in metrics.items()} for bench, metrics in results[name].items()}
            for name in ('C', 'CR')}
        internal = [read(out / 'internal_validation' / str(gpu) / 'validation.json') for gpu in range(8)]
        if any(p['status'] != 'complete' for p in internal):
            raise ValueError('Internal validation incomplete')
        records = [r for p in internal for r in p['records']]
        methods = sorted({r['method'] for r in records})
        if any(sum(r['method'] == method for r in records) != 128 for method in methods):
            raise ValueError('Internal validation does not cover 128 pairs')
        means = {method: {metric: sum(r['metrics'][metric] for r in records if r['method'] == method)/128
            for metric in ('psnry', 'ssim', 'lpips', 'dists')} for method in methods}
        result = {'status': 'complete', 'completed_utc': datetime.now(timezone.utc).isoformat(),
            'training': {'sources': 2048, 'C_updates': 16384, 'R_updates': 4096,
                         'C_kappa': [.25,4.], 'R_kappa': 2., 'gram_weight': 0.},
            'checkpoints': {name: {'path': str(path), 'sha256': digest(path)}
                            for name, path in [('C', c_adapter), ('CR', r_adapter)]},
            'counts': {k:v['count'] for k,v in config['benchmarks'].items()},
            'results': results, 'difference_from_same_machine_S5': differences,
            'internal_128_means': means,
            'scope': 'Frozen S5, fixed teacher caches refreshed after C; final checkpoints at kappa=2. '
                     'Full benchmark sets never used for training/checkpoint selection. '
                     'Only internal validation is source-disjoint from this adapter; base S5 saw qualified sources.'}
        write(out / 'report.json', result)
        lines = ['扩大训练与全验证集评测结果', '', '2048图；C 16384更新；R 4096更新；κ=2。',
                 '所有模型均在本机重新推理，使用相同的50轮协议与完整验证集。', '']
        for bench in config['benchmarks']:
            lines += [f"{bench}（{config['benchmarks'][bench]['count']}图）"]
            for name in ('S5', 'C', 'CR'):
                lines += [name + ': ' + ', '.join(f'{k}={v:.6f}' for k,v in results[name][bench].items())]
            lines += ['C+R减原S5: ' + ', '.join(f'{k}={v:+.6f}' for k,v in differences['CR'][bench].items()), '']
        lines += ['PSNR/SSIM/MANIQA/CLIPIQA/MUSIQ 越高越好；LPIPS/DISTS/FID 越低越好。',
                  '请逐项核对，不以单项指标改善代替整体提升。详细来源、检查点哈希及内部128图结果见 report.json。']
        (out / 'report.txt').write_text('\n'.join(lines) + '\n')
        self.update(status='complete', stage='report_complete', report=str(out / 'report.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/trajectory_expanded_20261108')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--prepare-pid', type=int, help='Wait for the already started preparation process')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Output must be inside project outputs')
    plan = {'train_sources': 2048, 'internal_validation': 128, 'C_updates':16384,
            'R_updates':4096, 'gpus':list(range(8)), 'kappa_evaluation':2.,
            'benchmarks':{'RealSR':100,'DRealSR':93,'DIV2K-Val':3000},
            'methods':['S5','C','CR'], 'execute':args.execute}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(output / 'expanded_plan.json', plan)
    experiment = Experiment(output)
    try:
        while not (output / 'pairs/pairs.json').is_file():
            experiment.update(status='running', stage='preparing_data', preparation_pid=args.prepare_pid)
            status = output / 'pairs/status.json'
            if status.exists() and read(status).get('status') == 'failed':
                raise RuntimeError('Data preparation failed; inspect logs/prepare.log')
            if args.prepare_pid is None:
                raise RuntimeError('Prepare pairs first or supply --prepare-pid')
            try:
                os.kill(args.prepare_pid, 0)
            except ProcessLookupError:
                raise RuntimeError('Preparation exited without a complete pair manifest')
            time.sleep(15)
        experiment.execute()
    except BaseException as error:
        experiment.update(status='failed', error=f'{type(error).__name__}: {error}')
        write(output / 'failure_report.json', experiment.state)
        raise


if __name__ == '__main__':
    main()
