"""Frozen native SR inputs and architecture/checkpoint contracts."""
import torch
from torch.nn import functional as F

from grn.utils_t2iv.hbq_util_t2iv import raw_feature2bit_label, multiclass_labels2onehot_input
from grn.models.sr_condition_modes import uses_continuous_lr


def validate_native_config(args):
    mode = getattr(args, 'sr_condition_mode', 'prefix')
    uses_continuous_lr(mode)
    # ``sr_cross_attn`` is also the module-topology switch for the new
    # S3->S3+Cross-Attention promotion.  Only reject it as the removed legacy
    # path when no explicit promotion marker is present.
    legacy_cross_attn = (
        getattr(args, 'sr_cross_attn', 0)
        and not getattr(args, 'sr_v2_cross_attn_promote', 0)
    )
    if mode in ('prefix_text', 'spatial_control') and getattr(args, 'sr_condition', 0):
        if legacy_cross_attn:
            raise ValueError(f'{mode} does not support legacy sr_cross_attn')
    if mode == 'native_adapter' and getattr(args, 'sr_condition', 0):
        if legacy_cross_attn:
            raise ValueError('native SR does not support legacy sr_cross_attn')
        if float(getattr(args, 'sr_condition_align_weight', 0)) != 0:
            raise ValueError('native SR requires sr_condition_align_weight=0')
        if getattr(args, 'refine_mode', 'ar_discrete_GRN_bit') != 'ar_discrete_GRN_bit':
            raise ValueError('native SR requires ar_discrete_GRN_bit')
        if getattr(args, 'apply_spatial_patchify', 0):
            raise ValueError('native SR does not support spatial patchify')
    return mode


@torch.no_grad()
def encode_native_lr(vae, lr_condition, hr_size, hbq_round):
    """Use the frozen tokenizer's image slicing and native bit/one-hot layout."""
    if lr_condition.ndim != 4 or lr_condition.shape[1] != 3 or lr_condition.shape[0] < 1:
        raise ValueError('native LR must be a nonempty [B,3,H,W] tensor')
    if hr_size <= 0 or hr_size % 16 or hbq_round < 1:
        raise ValueError('native SR requires positive HBQ rounds and HR size divisible by 16')
    pixels = F.interpolate(lr_condition.float(), size=(hr_size, hr_size), mode='bicubic',
                           align_corners=False, antialias=True).clamp(-1, 1)
    latents = []
    for item in pixels:
        features, _, _ = vae.encode_for_raw_features(item[None, :, None], scale_schedule=None, slice=True)
        latent = features[0]
        if latent.ndim != 5 or latent.shape[2] != 1:
            raise ValueError('native image HBQ must return [1,C,1,H,W] features')
        latents.append(latent)
    latent = torch.cat(latents, dim=0)
    bits = raw_feature2bit_label(latent, hbq_round)
    onehot = multiclass_labels2onehot_input(bits, 2).flatten(2).transpose(1, 2).contiguous()
    return onehot, latent.squeeze(2)


def load_empty_text_cache(path, text_channels=4096):
    if not path:
        raise ValueError('native SR requires sr_native_empty_text_path with real cached empty UMT5 features')
    from grn.utils_t2iv.s5_bundle_paths import resolve_path
    payload = torch.load(resolve_path(path, strict=True), map_location='cpu', weights_only=True)
    required = {'schema_version', 'prompt', 'features', 'umt5_revision', 'dependency_hashes'}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError('empty text cache is missing schema/provenance metadata')
    if type(payload['schema_version']) is not int or payload['schema_version'] != 1 or payload['prompt'] != '':
        raise ValueError('empty text cache requires schema_version=1 and prompt=""')
    revision, hashes = payload['umt5_revision'], payload['dependency_hashes']
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError('empty text cache requires a nonempty umt5_revision')
    if not isinstance(hashes, dict) or not hashes or any(
        not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
        for k, v in hashes.items()
    ):
        raise ValueError('empty text cache requires nonempty dependency_hashes')
    features = payload['features']
    if (not isinstance(features, torch.Tensor) or features.ndim != 2
        or features.shape[0] == 0 or features.shape[1] != text_channels
        or not features.is_floating_point() or not torch.isfinite(features).all()):
        raise ValueError(f'empty text features must be finite floating [L,{text_channels}], L>0')
    return features.detach().float().contiguous()


def pack_empty_text(features, batch_size, device):
    if features.ndim != 2 or features.shape[0] == 0 or batch_size < 1:
        raise ValueError('empty text packing requires [L,C], L>0 and a positive batch size')
    length = features.shape[0]
    compact = features.detach().to(device=device, dtype=torch.float32).repeat(batch_size, 1)
    return (compact, [length] * batch_size,
            torch.arange(batch_size + 1, device=device, dtype=torch.int32) * length,
            length, [1] * batch_size)


def restore_sr_architecture(args, checkpoint_args):
    """Old checkpoints are prefix; a caller's relocated cache path wins."""
    cache_override = getattr(args, 'sr_native_empty_text_path', '')
    from grn.models.sr_deterministic_attention import attention_flag
    args.sr_deterministic_attention = attention_flag(checkpoint_args)
    args.sr_condition_mode = checkpoint_args.get('sr_condition_mode', 'prefix')
    for field in ('sr_condition', 'sr_lr_size', 'sr_hr_size', 'sr_cross_attn',
                  'sr_cross_attn_layers', 'sr_cross_attn_dim', 'sr_cross_attn_heads',
                  'sr_cross_attn_window', 'sr_cross_attn_global_layers', 'sr_cross_attn_gate_init',
                  'sr_native_adapter_layers', 'sr_native_adapter_dim', 'sr_native_empty_text_path',
                  'sr_control_layers', 'sr_control_dim', 'sr_control_heads', 'sr_control_window',
                  'sr_condition_init_path', 'sr_v2_cross_attn_promote',
                  'sr_v2_cross_attn_init_seed'):
        if field in checkpoint_args:
            setattr(args, field, checkpoint_args[field])
    if cache_override:
        args.sr_native_empty_text_path = cache_override
    validate_native_config(args)
    from grn.utils_t2iv.sr_v2_condition_runtime import restore_condition_architecture
    restore_condition_architecture(args, checkpoint_args)
    from grn.utils_t2iv.sr_v2_calibrator_runtime import restore_calibrator_architecture
    restore_calibrator_architecture(args, checkpoint_args)


def load_native_state(model, state, warm_start=False):
    """A T2I warm start may miss only all new adapters; trained native loads are exact."""
    expected = model.state_dict()
    missing, unexpected = set(expected) - set(state), set(state) - set(expected)
    new_keys = {key for key in expected if key.startswith('sr_native_adapters.')}
    has_native = any(key.startswith('sr_native_adapters.') for key in state)
    permitted = new_keys if warm_start and not has_native else set()
    if missing != permitted or unexpected:
        raise RuntimeError(f'native SR checkpoint mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}')
    return model.load_state_dict(state, strict=not permitted)
