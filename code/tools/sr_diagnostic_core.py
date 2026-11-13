"""Small numerical helpers for read-only SR model diagnostics."""
import torch


def gradient_comparison(main, alignment):
    main, alignment = main.detach().float().flatten(), alignment.detach().float().flatten()
    if main.shape != alignment.shape:
        raise ValueError('Gradient shapes must match')
    dot = torch.dot(main, alignment).item()
    main_norm, alignment_norm = main.norm().item(), alignment.norm().item()
    return {
        'dot': dot, 'main_norm': main_norm, 'alignment_norm': alignment_norm,
        'cosine': dot / (main_norm * alignment_norm) if main_norm and alignment_norm else None,
        'alignment_to_main_norm': alignment_norm / main_norm if main_norm else None,
    }


def residual_comparison(incoming, outgoing):
    """Measure the realized residual, including rounding during the actual addition."""
    if incoming.shape != outgoing.shape:
        raise ValueError('Residual shapes must match')
    before, after = incoming.detach().float(), outgoing.detach().float()
    incoming_rms = before.square().mean().sqrt().item()
    actual_rms = (after - before).square().mean().sqrt().item()
    return {
        'input_dtype': str(incoming.dtype), 'output_dtype': str(outgoing.dtype),
        'actual_update_to_hr_rms': actual_rms / incoming_rms if incoming_rms else None,
        'changed_element_fraction': (after != before).float().mean().item(),
    }


def latent_error(prediction, target):
    """Error against the quantized HR latent used as the model target."""
    if prediction.shape != target.shape:
        raise ValueError('Latent shapes must match')
    delta = prediction.detach().float() - target.detach().float()
    return {'MAE': delta.abs().mean().item(), 'MSE': delta.square().mean().item()}


def fp32_residual_pre_hook(module, args, kwargs):
    """Diagnostic-only intervention: keep BF16 operators but promote the residual stream."""
    return args, dict(kwargs, x=kwargs['x'].float())



def native_checkpoint_state(reference, checkpoint):
    """Extract native tensors; never silently tolerate a partial backbone."""
    source = checkpoint['trainer']['gpt_fsdp'] if 'trainer' in checkpoint else checkpoint
    missing = set(reference) - set(source)
    unexpected = {key for key in source if key not in reference
                  and key != 'sr_prefix_pos'
                  and not key.startswith(('sr_condition_encoder.', 'sr_cross_attn.', 'sr_native_adapters.'))}
    shapes = {key for key in reference if key in source
              and reference[key].shape != source[key].shape}
    if missing or unexpected or shapes:
        raise ValueError(f'native checkpoint mismatch: missing={sorted(missing)}, '
                         f'unexpected={sorted(unexpected)}, shape={sorted(shapes)}')
    return {key: source[key] for key in reference}


from contextlib import contextmanager


@contextmanager
def without_sr_cross_attention(model, enabled=True):
    """Bypass added residuals, preserving the LR prefix and all weights."""
    handles, counts = [], {}

    def bypass(layer):
        def hook(module, inputs, output):
            counts[layer] = counts.get(layer, 0) + 1
            return inputs[0]
        return hook

    try:
        if enabled:
            for layer, module in model.sr_cross_attn.items():
                handles.append(module.register_forward_hook(bypass(layer)))
        yield counts
    finally:
        for handle in handles:
            handle.remove()



def native_weight_audit(reference, checkpoint):
    """Compare every native tensor value, not a sampled weight-drift estimate."""
    state = native_checkpoint_state(reference, checkpoint)
    changed = sorted(key for key in reference if not torch.equal(reference[key], state[key]))
    return {'native_tensors': len(reference), 'changed_native_tensors': len(changed),
            'changed_names': changed}


def pretrain_control_audit(checkpoint, arm, microsteps):
    """Check the resolved source paths, stored settings, and actual update count."""
    import json
    from pathlib import Path
    args = checkpoint['args']
    project = Path(__file__).resolve().parents[1]
    paths = {'rush_resume': project / 'weights/GRN_T2I_2B_FSA_251600.pth',
             'vae_path': project / 'weights/HBQ_image_video_tokenizer_64dim_M4_20260626.ckpt',
             'sr_protocol_root': project.parent / 'VARSR-AlignVAR-Offline4GPU'}
    for key, expected_path in paths.items():
        if Path(args.get(key, '')).resolve() != expected_path.resolve():
            raise ValueError(f'Control original source/protocol path mismatch: {key}')
    if arm not in ('full', 'frozen') or bool(args['sr_freeze_backbone']) != (arm == 'frozen'):
        raise ValueError('Checkpoint freeze scope does not match the arm')
    expected = {'sr_fixed_seed': 3407, 'restrict_data_size': 16000,
                'gradient_accumulation': 16, 'ep': 2,
                'sr_latent_recon_weight': .25, 'sr_latent_structure_weight': .1,
                'sr_latent_edge_weight': .05, 'sr_condition_align_weight': .05,
                'sr_latent_highpass_weight': 0., 'sr_backbone_lr_scale': .2,
                'sr_head_lr_scale': .4, 'sr_new_lr_scale': 1.,
                'sr_condition': 1, 'sr_cross_attn': 1, 'tlr': 5e-5, 'sche': 'lin0',
                'wp': .1, 'wp0': .005, 'wpe': .01, 'twd': .05, 'adam_beta2': .95,
                'tclip': 2., 'fp16': 2, 'sr_cross_attn_dim': 512, 'sr_cross_attn_heads': 8,
                'sr_cross_attn_window': 7, 'sr_cross_attn_gate_init': .001,
                'sr_degradation': 'realesrgan', 'sr_train_manifest': 'data/manifests/lsdir_ffhq10k.txt',
                'sr_degradation_config': 'dataloader/params_realesrgan.yml',
                'sr_lr_size': 128, 'sr_hr_size': 512, 'sr_condition_drop_rate': 0., 'hbq_round': 4}
    for key, value in expected.items():
        if args.get(key) != value:
            raise ValueError(f'Control configuration mismatch: {key}={args.get(key)}')
    if (json.loads(args['sr_cross_attn_layers']) != [4, 12, 20, 28]
            or json.loads(args['sr_cross_attn_global_layers']) != [28]):
        raise ValueError('Control cross-attention configuration mismatch')
    states = checkpoint['trainer']['gpt_fsdp_opt']['state']
    steps = {int(state['step']) for state in states.values() if 'step' in state}
    if microsteps not in (8000, 16000) or steps != {microsteps // 16} or checkpoint['g_it'] != microsteps:
        raise ValueError(f'Unexpected optimizer updates: {steps}; microsteps={microsteps}')
    return {'optimizer_updates': microsteps // 16, 'freeze_backbone': arm == 'frozen',
            'warm_start': args['rush_resume']}



GRN_UMT5_REVISION = '3d4699d4d31fe5e0bf7cc8f25c4d98b315a87a18'
# Official repository object identities at GRN_UMT5_REVISION. LFS uses SHA256;
# the two small tokenizer files use Git's blob SHA1 (including the blob header).
GRN_UMT5_MANIFEST = {
    'models_t5_umt5-xxl-enc-bf16.pth': ('sha256', '7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d'),
    'umt5-xxl/tokenizer.json': ('sha256', '6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b'),
    'umt5-xxl/special_tokens_map.json': ('git_blob_sha1', '14855e7052ffbb595057dfd791d293c1c940db2c'),
    'umt5-xxl/tokenizer_config.json': ('git_blob_sha1', '4e1cc1cd85599ce0b47fd0a746af188fe4043ff2'),
}


def verify_file_manifest(root, manifest):
    """Hash actual dependency bytes; completion marker contents are not trusted."""
    import hashlib
    from pathlib import Path
    observed = {}
    for filename, (algorithm, expected) in manifest.items():
        path = Path(root) / filename
        if not path.is_file():
            raise ValueError(f'Missing dependency: {filename}')
        if algorithm == 'sha256':
            with path.open('rb') as handle:
                actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        elif algorithm == 'git_blob_sha1':
            payload = path.read_bytes()
            actual = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
        else:
            raise ValueError(f'Unsupported dependency hash: {algorithm}')
        if actual != expected:
            raise ValueError(f'Dependency content mismatch: {filename}')
        observed[filename] = actual
    return observed


def require_complete_controls(result, root):
    """Fail closed unless all four evaluations and 36 native images exist."""
    from pathlib import Path
    expected = {(arm, step) for arm in ('full', 'frozen') for step in (500, 1000)}
    evaluations = result['evaluations']
    if len(evaluations) != 4 or {(r['arm'], r['optimizer_updates']) for r in evaluations} != expected:
        raise ValueError('Incomplete or duplicate control evaluations')
    native = result.get('native_regression')
    models = {'pristine', 'baseline', 'enhanced'}
    if not native or set(native['models']) != models or native['images'] != 36:
        raise ValueError('Incomplete native regression image manifest')
    for model in models:
        for prompt in range(6):
            for seed in (42, 3407):
                path = Path(root) / 'native_regression' / model / f'prompt_{prompt:02d}_seed_{seed}.png'
                if not path.is_file() or not path.stat().st_size:
                    raise ValueError(f'Missing native regression image: {path}')
