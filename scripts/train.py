#!/usr/bin/env python3
"""Print the original S5 training plan; launch only when --execute is explicit."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from grn.utils_t2iv.s5_bundle_paths import resolve_path


def plan(args):
    record = json.loads((ROOT / 'manifests/s5_training_args.json').read_text())
    options = dict(item[2:].split('=', 1) for item in record['original_argv'])
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name):
        raise ValueError('Run name must contain only letters, digits, underscore, dot or hyphen')
    if args.smoke_updates < 0:
        raise ValueError('--smoke-updates must be nonnegative')
    relative = ('verification/train_smoke' if args.run_name == 's5_retrain' else
                'verification/' + args.run_name) if args.smoke_updates else 'runs/' + args.run_name
    output = (ROOT / relative).resolve()
    if not output.is_relative_to(ROOT / ('verification' if args.smoke_updates else 'runs')):
        raise ValueError('Training output must remain inside the bundle')
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing training output: ' + str(output))
    for key, value in list(options.items()):
        if value.startswith('/') and key != 'token_cache_dir':
            options[key] = str(resolve_path(value))
    options.update(local_out_path=str(output), bed=str(output),
        exp_name=args.run_name, token_cache_dir=str(output / 'token_cache'),
        sr_view_audit_dir=str(output / 'consumption'),
        sr_qualified_audit_dir=str(output / 'qualified_consumption'),
        sr_v2_audit_dir=str(output / 'actual_input_audit'),
        sr_v2_runtime_audit_dir=str(output / 'runtime'),
        rush_resume=str(ROOT / 'weights/s5_initialization_step_16000.pth'),
        vae_path=str(ROOT / 'weights/HBQ_image_video_tokenizer_64dim_M4_20260626.ckpt'),
        sr_native_empty_text_path=str(ROOT / 'weights/native_empty_text.pt'),
        sr_protocol_root=str(ROOT / 'protocol'),
        sr_qualified_manifest=str(ROOT / 'manifests/qualification_v1/qualified_manifest.json'),
        sr_v2_semantic_manifest=str(ROOT / 'weights/dinov2_vits14/ASSET_MANIFEST.json'),
        auto_resume='False')
    if args.smoke_updates:
        options.update(sr_run_stop_microstep=str(args.smoke_updates * 16),
                       s5_bundle_smoke_updates=str(args.smoke_updates))
    fixed = {'ep': '2', 'gradient_accumulation': '16', 'restrict_data_size': '-1',
             'sr_expected_train_samples': '93854', 'sr_latent_edge_weight': '0.0',
             'sr_qualified_manifest_sha256': record['original_qualified_manifest_sha256']}
    for key, value in fixed.items():
        if options.get(key) != value:
            raise ValueError('Archived S5 recipe drift: ' + key)
    argv = ['--' + key + '=' + value for key, value in options.items()]
    command = [str(ROOT / 'environment/python'), '-m', 'torch.distributed.run',
               '--standalone', '--master_port=' + str(args.master_port), '--nproc_per_node=2',
               str(ROOT / 'code/t2iv_train.py'), *argv]
    return {'schema': 'grn_s5_training_plan_v1', 'execution_requested': args.execute,
            'mode': 'bounded_smoke' if args.smoke_updates else 'full_two_epochs',
            'world_size': 2, 'gradient_accumulation': 16, 'epochs': 2,
            'optimizer_updates_requested': args.smoke_updates or 5866,
            'qualified_samples': 93854, 'original_identity_sha256': fixed['sr_qualified_manifest_sha256'],
            'output': str(output), 'working_directory': str(ROOT / 'code'), 'command': command}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Actually start the printed training command')
    parser.add_argument('--run-name', default='s5_retrain')
    parser.add_argument('--master-port', type=int, default=29610)
    parser.add_argument('--smoke-updates', type=int, default=0,
                        help='Stop after this many real optimizer updates, preserving the full data/LR horizon')
    parser.add_argument('--json', action='store_true', help='Print the complete plan as JSON')
    args = parser.parse_args()
    if not 1024 <= args.master_port <= 65535:
        parser.error('--master-port must be between 1024 and 65535')
    value = plan(args)
    if args.json:
        print(json.dumps(value, indent=2))
    else:
        print('S5 training plan (no training starts without --execute):')
        print('Output:', value['output'])
        print(shlex.join(value['command']))
    if not args.execute:
        return
    output = Path(value['output'])
    required = [ROOT / 'environment/python', ROOT / 'weights/s5_initialization_step_16000.pth',
        ROOT / 'weights/HBQ_image_video_tokenizer_64dim_M4_20260626.ckpt',
        ROOT / 'weights/native_empty_text.pt', ROOT / 'weights/dinov2_vits14/ASSET_MANIFEST.json',
        ROOT / 'manifests/qualification_v1/qualified_manifest.json']
    for path in required:
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise FileNotFoundError('Missing or external packaged dependency: ' + str(path))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'launch_plan.json').write_text(json.dumps(value, indent=2) + '\n')
    env = os.environ.copy()
    env.update(GRN_S5_ROOT=str(ROOT), PYTHONPATH=str(ROOT / 'code'),
               S5_BUNDLE_GUARD='1',
               S5_GUARD_REPORT_DIR=env.get('S5_GUARD_REPORT_DIR', str(output / 'isolation')),
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', WANDB_MODE='offline',
               WANDB_DIR=str(output),
               HF_HOME=str(output / 'cache/huggingface'), TORCH_HOME=str(ROOT / 'protocol/offline_cache/torch'),
               XDG_CACHE_HOME=str(output / 'cache'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    env.pop('AUTO_RESUME', None)
    # AF_UNIX socket names have a short OS limit. This transient directory holds
    # multiprocessing sockets, not model/code/data dependencies, and is removed
    # when training exits; placing it below a long bundle path breaks DataLoader.
    with tempfile.TemporaryDirectory(prefix='s5_', dir='/tmp') as transient_tmp:
        env['TMPDIR'] = transient_tmp
        (output / 'transient_runtime.json').write_text(json.dumps({
            'tmpdir': transient_tmp, 'purpose': 'multiprocessing sockets and transient files',
            'contains_external_dependency': False}, indent=2) + '\n')
        with (output / 'stdout.log').open('x') as stream:
            result = subprocess.run(value['command'], cwd=ROOT / 'code', env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
    (output / 'launcher_result.json').write_text(json.dumps({'exit_code': result.returncode,
        'training_completed': result.returncode == 0 and not args.smoke_updates,
        'smoke_run': bool(args.smoke_updates), 'stdout': str(output / 'stdout.log'),
        'transient_tmp_removed': not Path(transient_tmp).exists()}, indent=2) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
