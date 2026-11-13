"""Opt-in C/R lifecycle and image losses after an independently verified S5 load.

Adapters bind to the caller's verified base checkpoint SHA256. These helpers do
not load the base, change its manifest, or replace its existing source audit.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import tempfile

import torch
from torch import nn
from torch.nn import functional as F

from grn.models.sr_latent_residual import LatentResidualCorrector
from grn.models.sr_v2_calibrator import BoundedRoundCalibrator

SCHEMA = 'grn_s5_trajectory_adapter_v1'
PREFIXES = ('sr_round_calibrator.', 'sr_latent_residual.')


def _base_hash(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('base SHA256 must be a lowercase 64-character digest from the verified base')
    return value


def _calibrator(model):
    module = getattr(model, 'sr_round_calibrator', None)
    if not isinstance(module, BoundedRoundCalibrator):
        raise ValueError('Trajectory adaptation requires the already-loaded S5 calibrator')
    return module


def _calibrator_architecture(module):
    return {key: getattr(module, key) for key in (
        'mode', 'hr_dim', 'lr_dim', 'width', 'rounds', 'bits_per_round',
        'max_log_odds', 'initial_gate')}


def _architecture(model):
    module = getattr(model, 'sr_latent_residual', None)
    if module is not None and not isinstance(module, LatentResidualCorrector):
        raise ValueError('Unexpected existing latent residual module')
    return {'calibrator': _calibrator_architecture(_calibrator(model)),
            'lrc': None if module is None else module.architecture()}


def freeze_trajectory_parameters(model, *, trainable=True):
    if type(trainable) is not bool:
        raise ValueError('trainable must be an explicit bool')
    calibrator = _calibrator(model)
    model.requires_grad_(False)
    model.eval()
    calibrator.requires_grad_(trainable)
    calibrator.train(trainable)
    lrc = getattr(model, 'sr_latent_residual', None)
    if lrc is not None:
        if not isinstance(lrc, LatentResidualCorrector):
            raise ValueError('Unexpected latent residual module')
        lrc.requires_grad_(trainable)
        lrc.train(trainable)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def configure_trajectory_adapter(model, *, enable_lrc=True, blocks=6,
                                 seed=20261108, trainable=True):
    if type(enable_lrc) is not bool or type(trainable) is not bool:
        raise ValueError('enable_lrc and trainable must be explicit bools')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Adapter initialization seed must be a nonnegative int64')
    if type(blocks) is not int or not 1 <= blocks <= 32:
        raise ValueError('blocks must be an integer in [1,32]')
    calibrator = _calibrator(model)
    reference = next(calibrator.parameters())
    if reference.dtype != torch.float32:
        raise ValueError('Existing S5 calibrator must use FP32 parameters')
    existing = getattr(model, 'sr_latent_residual', None)
    if existing is not None:
        if (not enable_lrc or not isinstance(existing, LatentResidualCorrector)
                or existing.blocks_count != blocks):
            raise ValueError('Cannot silently replace an existing LRC architecture')
    elif enable_lrc:
        # Seed only the CPU generator. torch.manual_seed would also reseed CUDA.
        with torch.device('cpu'), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            module = LatentResidualCorrector(blocks=blocks)
        model.sr_latent_residual = module.to(device=reference.device, dtype=torch.float32)
    freeze_trajectory_parameters(model, trainable=trainable)
    return _architecture(model)


def _adapter_state(model):
    result = {'sr_round_calibrator.' + k: v for k, v in _calibrator(model).state_dict().items()}
    lrc = getattr(model, 'sr_latent_residual', None)
    if lrc is not None:
        result.update({'sr_latent_residual.' + k: v for k, v in lrc.state_dict().items()})
    return result


def _plain_metadata(value, *, depth=0):
    if depth > 32:
        raise ValueError('Adapter metadata nesting is too deep')
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) in (list, tuple):
        for item in value:
            _plain_metadata(item, depth=depth + 1)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _plain_metadata(item, depth=depth + 1)
        return
    raise ValueError('Adapter metadata must contain finite JSON-compatible values')


def _finite_tree(value):
    if torch.is_tensor(value):
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all()):
                raise ValueError('Checkpoint contains nonfinite tensor values')
    elif isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_tree(item)
    elif type(value) is float and not math.isfinite(value):
        raise ValueError('Checkpoint contains nonfinite scalar values')


def _optimizer_names(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    groups, seen = [], set()
    for group in optimizer.param_groups:
        group_names = []
        for parameter in group['params']:
            name = names.get(id(parameter))
            if name is None or not name.startswith(PREFIXES) or name in seen:
                raise ValueError('Adapter optimizer may contain only distinct C/R parameters')
            group_names.append(name)
            seen.add(name)
        groups.append(group_names)
    if not seen:
        raise ValueError('Adapter optimizer is empty')
    return groups


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def save_trajectory_adapter(path, model, *, base_sha256, optimizer=None,
                            step=0, metadata=None):
    """Save only C/R and optional optimizer state; never overwrite any file."""
    base_sha256 = _base_hash(base_sha256)
    if type(step) is not int or step < 0:
        raise ValueError('Adapter step must be a nonnegative integer')
    metadata = {} if metadata is None else metadata
    if type(metadata) is not dict:
        raise ValueError('Adapter metadata must be a dictionary')
    _plain_metadata(metadata)
    payload = {'schema': SCHEMA, 'base_sha256': base_sha256,
               'architecture': _architecture(model), 'state_dict': _cpu_tree(_adapter_state(model)),
               'step': step, 'metadata': metadata,
               'optimizer': None if optimizer is None else _cpu_tree(optimizer.state_dict()),
               'optimizer_parameter_names': None if optimizer is None else _optimizer_names(model, optimizer)}
    _finite_tree(payload)
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Refusing to overwrite existing checkpoint: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='.' + path.name + '.', suffix='.tmp',
                                     dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        # Atomic create-if-absent; unlike replace(), this cannot overwrite a base.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {'path': str(path), 'step': step, 'base_sha256': base_sha256,
            'architecture': payload['architecture']}


def load_trajectory_adapter(path, model, *, expected_base_sha256,
                            optimizer=None, trainable=False):
    """Validate schema, base binding and every C/R tensor before copying weights."""
    expected_base_sha256 = _base_hash(expected_base_sha256)
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    keys = {'schema', 'base_sha256', 'architecture', 'state_dict', 'step', 'metadata',
            'optimizer', 'optimizer_parameter_names'}
    if type(payload) is not dict or set(payload) != keys or payload['schema'] != SCHEMA:
        raise ValueError('Unsupported or incomplete trajectory adapter schema')
    if _base_hash(payload['base_sha256']) != expected_base_sha256:
        raise ValueError('Adapter source hash does not match the verified base SHA256')
    if type(payload['step']) is not int or payload['step'] < 0:
        raise ValueError('Invalid adapter optimizer step')
    if type(payload['metadata']) is not dict:
        raise ValueError('Invalid adapter metadata')
    _plain_metadata(payload['metadata'])
    _finite_tree(payload)
    arch = payload['architecture']
    if type(arch) is not dict or set(arch) != {'calibrator', 'lrc'}:
        raise ValueError('Invalid adapter architecture metadata')
    expected_calibrator = _calibrator_architecture(_calibrator(model))
    saved_calibrator = arch['calibrator']
    if (type(saved_calibrator) is not dict or set(saved_calibrator) != set(expected_calibrator)
            or any(type(saved_calibrator[key]) is not type(value) or saved_calibrator[key] != value
                   for key, value in expected_calibrator.items())):
        raise ValueError('Adapter calibrator architecture differs from the loaded base')
    expected = {'sr_round_calibrator.' + k: v for k, v in _calibrator(model).state_dict().items()}
    prototype = None
    if arch['lrc'] is not None:
        spec = arch['lrc']
        if type(spec) is not dict or set(spec) != {'blocks', 'channels', 'groups', 'kappa_dim'}:
            raise ValueError('Invalid LRC architecture metadata')
        with torch.device('cpu'), torch.random.fork_rng(devices=[]):
            prototype = LatentResidualCorrector(**spec)
        expected.update({'sr_latent_residual.' + k: v for k, v in prototype.state_dict().items()})
    existing = getattr(model, 'sr_latent_residual', None)
    if existing is not None and (prototype is None or not isinstance(existing, LatentResidualCorrector)
                                 or existing.architecture() != arch['lrc']):
        raise ValueError('Existing LRC differs from adapter architecture')
    state = payload['state_dict']
    if type(state) is not dict or set(state) != set(expected):
        raise ValueError('Adapter must contain the exact C/R state membership')
    for key, value in state.items():
        reference = expected[key]
        if (not torch.is_tensor(value) or value.layout != torch.strided
                or value.shape != reference.shape or value.dtype != reference.dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f'Invalid adapter tensor shape/dtype/finite values: {key}')
    if (payload['optimizer'] is None) != (payload['optimizer_parameter_names'] is None):
        raise ValueError('Optimizer state and parameter names must appear together')
    if payload['optimizer'] is not None:
        saved_optimizer, saved_names = payload['optimizer'], payload['optimizer_parameter_names']
        if (type(saved_optimizer) is not dict or set(saved_optimizer) != {'state', 'param_groups'}
                or type(saved_optimizer['state']) is not dict
                or type(saved_optimizer['param_groups']) is not list or type(saved_names) is not list
                or len(saved_optimizer['param_groups']) != len(saved_names)):
            raise ValueError('Invalid adapter optimizer state')
        parameter_names = {PREFIXES[0] + name for name, _ in _calibrator(model).named_parameters()}
        if prototype is not None:
            parameter_names.update(PREFIXES[1] + name for name, _ in prototype.named_parameters())
        id_to_name, seen_names = {}, set()
        for group, group_names in zip(saved_optimizer['param_groups'], saved_names):
            if (type(group) is not dict or type(group.get('params')) is not list
                    or type(group_names) is not list or len(group['params']) != len(group_names)):
                raise ValueError('Invalid optimizer parameter metadata')
            for identifier, name in zip(group['params'], group_names):
                if (type(identifier) is not int or identifier < 0 or identifier in id_to_name
                        or type(name) is not str or name not in parameter_names or name in seen_names):
                    raise ValueError('Invalid optimizer parameter membership')
                id_to_name[identifier] = name
                seen_names.add(name)
        if not id_to_name or not set(saved_optimizer['state']).issubset(id_to_name):
            raise ValueError('Optimizer state contains parameters outside C/R')
        for identifier, parameter_state in saved_optimizer['state'].items():
            if type(parameter_state) is not dict:
                raise ValueError('Invalid optimizer per-parameter state')
            shape = expected[id_to_name[identifier]].shape
            for value in parameter_state.values():
                if torch.is_tensor(value) and value.ndim != 0 and value.shape != shape:
                    raise ValueError('Optimizer tensor shape differs from C/R parameter')
    if optimizer is not None:
        if payload['optimizer'] is None:
            raise ValueError('Adapter has no optimizer state to resume')
        if payload['optimizer_parameter_names'] != _optimizer_names(model, optimizer):
            raise ValueError('Adapter optimizer parameter order differs')
        saved_optimizer = payload['optimizer']
        if type(saved_optimizer) is not dict or set(saved_optimizer) != {'state', 'param_groups'}:
            raise ValueError('Invalid adapter optimizer state')
        if len(saved_optimizer['param_groups']) != len(optimizer.param_groups):
            raise ValueError('Adapter optimizer group count differs')
        for saved_group, group in zip(saved_optimizer['param_groups'], optimizer.param_groups):
            if len(saved_group['params']) != len(group['params']):
                raise ValueError('Adapter optimizer group size differs')
    configure_trajectory_adapter(model, enable_lrc=prototype is not None,
                                 blocks=6 if prototype is None else prototype.blocks_count,
                                 trainable=trainable)
    calibrator_state = {key.removeprefix(PREFIXES[0]): value for key, value in state.items()
                        if key.startswith(PREFIXES[0])}
    _calibrator(model).load_state_dict(calibrator_state, strict=True)
    if prototype is not None:
        model.sr_latent_residual.load_state_dict({key.removeprefix(PREFIXES[1]): value
            for key, value in state.items() if key.startswith(PREFIXES[1])}, strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload['optimizer'])
    return {'step': payload['step'], 'metadata': payload['metadata'],
            'base_sha256': payload['base_sha256'], 'architecture': arch,
            'optimizer_loaded': optimizer is not None}


class _DeterministicBicubicAA(torch.autograd.Function):
    """GPU-identical antialiased forward, deterministic CPU linear backward.

    MACA/PyTorch rejects the CUDA AA bicubic backward in deterministic mode.
    Resize is a fixed linear operator: recomputing its CPU derivative needs no
    image values. Host transfers occur only on this optional Gram loss branch.
    """
    @staticmethod
    def forward(ctx, image):
        ctx.input_shape, ctx.input_device, ctx.input_dtype = image.shape, image.device, image.dtype
        return F.interpolate(image, size=(224, 224), mode='bicubic',
                             align_corners=False, antialias=True)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        with torch.enable_grad():
            source = torch.zeros(ctx.input_shape, device='cpu', dtype=ctx.input_dtype, requires_grad=True)
            resized = F.interpolate(source, size=(224, 224), mode='bicubic',
                                    align_corners=False, antialias=True)
            result, = torch.autograd.grad(resized, source, gradient.detach().to('cpu'))
        return result.to(device=ctx.input_device)


def semantic_features_with_input_gradients(encoder, image01):
    """Reuse the audited DINO math without its condition-only no_grad wrapper."""
    encoder.eval()
    encoder.requires_grad_(False)
    from grn.utils_t2iv.sr_v2_semantic import FrozenDinoV2SemanticEncoder, official_no_reg_position
    if isinstance(encoder, FrozenDinoV2SemanticEncoder):
        encoder._check_runtime(image01.device)
        if image01.device.type == 'cuda' and image01.requires_grad:
            # Keep exactly the audited input normalization/resize forward. Only
            # the unsupported GPU resize derivative is evaluated on the CPU.
            input11 = image01 * 2 - 1
            rgb = _DeterministicBicubicAA.apply((input11.float() + 1) * .5).clamp(0, 1)
            rgb = (rgb - encoder.rgb_mean) / encoder.rgb_std
        else:
            rgb = encoder._preprocess(image01 * 2 - 1)
        model = encoder.backbone
        grid = model.patch_embed.proj(rgb)
        tokens = model.patch_embed.norm(grid.flatten(2).transpose(1, 2))
        tokens = torch.cat((model.cls_token.expand(rgb.shape[0], -1, -1), tokens), dim=1)
        tokens = model.pos_drop(tokens + official_no_reg_position(model.pos_embed, 224, 224, tokens.dtype))
        tokens = model.norm_pre(model.patch_drop(tokens))
        tokens = model.norm(model.blocks(tokens))
        features = tokens[:, 1:].transpose(1, 2).reshape(rgb.shape[0], 384, 16, 16)
    else:
        # Allows small gradient-preserving feature modules in CPU tests and
        # explicitly supplied alternate encoders with the same [-1,1] contract.
        features = encoder(image01 * 2 - 1)
    if not torch.is_tensor(features) or features.ndim != 4 or not bool(torch.isfinite(features).all()):
        raise ValueError('Semantic loss encoder must return finite [B,C,H,W] features')
    if image01.requires_grad and not features.requires_grad:
        raise ValueError('Semantic loss encoder cut the SR input gradient')
    return features


def compute_trajectory_image_loss(latent, hr_images, vae, *, perceptual_model=None,
                                  semantic_encoder=None, pixel_weight=1.,
                                  lpips_weight=2., gram_weight=0.):
    """Decode a differentiable latent and compare with detached RGB HR[-1,1].

    LPIPS receives RGB[0,1]. Gram compares channel-normalized patch relations
    F F^T on a 32x32 feature grid; its mean supplies the 1/N^2 factor.
    Only active objective networks are called. Frozen parameters keep no grads,
    while decoder/perceptual/DINO operations preserve the SR input gradient.
    """
    weights = (pixel_weight, lpips_weight, gram_weight)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in weights):
        raise ValueError('Image loss weights must be finite nonnegative scalars')
    if not torch.is_tensor(latent) or not latent.is_floating_point() or not bool(torch.isfinite(latent).all()):
        raise ValueError('Latent must be a finite floating tensor')
    if latent.ndim == 4:
        latent = latent.unsqueeze(2)
    if latent.ndim != 5 or latent.shape[2] != 1:
        raise ValueError('Expected latent [B,C,1,H,W] or [B,C,H,W]')
    from grn.utils_t2iv.sr_v2_image_loss import _charbonnier, _image_batch, _freeze_for_input_gradients
    target = _image_batch(hr_images).to(device=latent.device)
    if not bool(torch.isfinite(target).all()) or bool((target < -1).any()) or bool((target > 1).any()):
        raise ValueError('HR images must be finite RGB in [-1,1]')
    zero = latent.new_zeros((), dtype=torch.float32)
    result = dict(charbonnier=zero, lpips=zero, gram=zero, total=zero)
    if not any(weights):
        return result
    if lpips_weight and (perceptual_model is None or getattr(perceptual_model, 'as_loss', True) is not True):
        raise ValueError('LPIPS loss requires a differentiable perceptual model')
    if gram_weight and semantic_encoder is None:
        raise ValueError('Gram loss requires a frozen semantic encoder')
    _freeze_for_input_gradients(vae)
    parameter = next((p for p in vae.parameters() if p.is_floating_point()), None)
    dtype = torch.float32 if parameter is None else parameter.dtype
    with torch.autocast(device_type=latent.device.type, enabled=False):
        decoded = vae.decode(latent.to(dtype=dtype), slice=True)
        if decoded.ndim != 5 or decoded.shape[1:3] != (3, 1):
            raise ValueError('Decoder must return RGB [B,3,1,H,W]')
        prediction = (decoded.squeeze(2).float() + 1) * .5
        if prediction.shape != target.shape or not bool(torch.isfinite(prediction).all()):
            raise ValueError('Decoded prediction must be finite and match HR shape')
        prediction = prediction + (prediction.clamp(0, 1) - prediction).detach()
        target01 = (target + 1) * .5
        if pixel_weight:
            result['charbonnier'] = _charbonnier(prediction, target01)
        if lpips_weight:
            _freeze_for_input_gradients(perceptual_model)
            result['lpips'] = perceptual_model(prediction, target01).float().mean()
            if prediction.requires_grad and not result['lpips'].requires_grad:
                raise ValueError('Perceptual model cut the SR input gradient')
        if gram_weight:
            predicted_features = semantic_features_with_input_gradients(semantic_encoder, prediction)
            with torch.no_grad():
                target_features = semantic_features_with_input_gradients(semantic_encoder, target01)
            def relations(features):
                features = F.interpolate(features.float(), size=(32, 32), mode='bilinear', align_corners=False)
                tokens = F.normalize(features.flatten(2).transpose(1, 2), dim=-1, eps=1e-6)
                return tokens @ tokens.transpose(1, 2)
            result['gram'] = (relations(predicted_features) - relations(target_features)).square().mean()
        result['total'] = pixel_weight * result['charbonnier'] + lpips_weight * result['lpips'] + gram_weight * result['gram']
        if any(not bool(torch.isfinite(value)) for value in result.values()):
            raise FloatingPointError('Trajectory image loss is nonfinite')
    return result
