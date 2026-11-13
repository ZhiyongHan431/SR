"""Explicit SR attention layouts with legacy-checkpoint compatibility."""
import json
from grn.models.sr_condition_modes import uses_empty_text


def resolve_cross_attention_windows(layers, local_window, global_layers=None):
    layers = tuple(int(layer) for layer in layers)
    if len(set(layers)) != len(layers) or any(layer < 1 for layer in layers):
        raise ValueError('Cross-attention layers must be distinct positive integers')
    if local_window < 0 or (local_window and local_window % 2 != 1):
        raise ValueError('Local window must be zero or a positive odd number')
    if global_layers is None or global_layers == '':
        global_layers = [max(layers)] if layers else []
    elif isinstance(global_layers, str):
        global_layers = json.loads(global_layers)
    if not isinstance(global_layers, (list, tuple)):
        raise ValueError('Global cross-attention layers must be a JSON list')
    global_layers = set(int(layer) for layer in global_layers)
    return {layer: 0 if layer in global_layers else local_window for layer in layers}

def configure_sr_trainability(model, freeze_backbone=False):
    """Select the SR adaptation scope before FSDP and optimizer construction."""
    if not model.sr_condition:
        raise ValueError('SR trainability policy requires an SR model')
    report = {'trainable_numel': 0, 'frozen_numel': 0, 'frozen_names': []}
    for name, parameter in model.named_parameters():
        condition_parameter = name == 'sr_prefix_pos' or name.startswith(
            ('sr_condition_encoder.', 'sr_cross_attn.', 'sr_native_adapters.', 'sr_spatial_controls.',
             'sr_semantic_fusion.', 'sr_coarse_head.', 'sr_round_calibrator.'))
        has_text = uses_empty_text(getattr(model, 'sr_condition_mode', 'prefix'))
        trainable = condition_parameter if freeze_backbone else (has_text or not name.startswith('text_proj.'))
        parameter.requires_grad_(trainable)
        report['trainable_numel' if trainable else 'frozen_numel'] += parameter.numel()
        if not trainable:
            report['frozen_names'].append(name)
    if not report['trainable_numel']:
        raise ValueError('SR policy selected no trainable condition parameters')
    return report


def sr_parameter_lr_scales(args):
    """Explicit SR parameter groups shared by startup and policy audits."""
    return (
        ('sr_condition_encoder.', args.sr_new_lr_scale),
        ('sr_cross_attn.', args.sr_new_lr_scale),
        ('sr_native_adapters.', args.sr_new_lr_scale),
        ('sr_spatial_controls.', args.sr_new_lr_scale),
        ('sr_semantic_fusion.', args.sr_new_lr_scale),
        ('sr_coarse_head.', args.sr_new_lr_scale),
        ('sr_round_calibrator.', args.sr_new_lr_scale),
        ('sr_prefix_pos', args.sr_new_lr_scale),
        ('head.', args.sr_head_lr_scale),
    )
