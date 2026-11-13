"""Isolated S2 configuration, exact B promotion, and coarse supervision contracts.

These helpers do not relax the frozen S1 loader. Architecture changes require an
explicit promotion, whereas S2 resume/inference always require an exact schema.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re

import torch
from torch.nn import functional as F


NEW_PREFIXES = ('sr_semantic_fusion.', 'sr_coarse_head.')
FIELD_DEFAULTS = {
    'sr_v2_semantic': 0,
    'sr_v2_semantic_hidden': 128,
    'sr_v2_semantic_manifest': '',
    'sr_v2_semantic_manifest_sha256': '',
    'sr_v2_condition_coarse_weight': 0.0,
    'sr_v2_condition_seed': 20260916,
}


@dataclass(frozen=True)
class ConditionConfig:
    semantic: bool
    semantic_hidden: int
    manifest: str
    manifest_sha256: str
    coarse_weight: float
    seed: int

    @property
    def enabled(self):
        return self.semantic or self.coarse_weight > 0


def condition_config(args):
    values = {name: getattr(args, name, default) for name, default in FIELD_DEFAULTS.items()}
    semantic = values['sr_v2_semantic']
    hidden = values['sr_v2_semantic_hidden']
    weight = values['sr_v2_condition_coarse_weight']
    seed = values['sr_v2_condition_seed']
    manifest = values['sr_v2_semantic_manifest']
    digest = values['sr_v2_semantic_manifest_sha256']
    if type(semantic) is not int or semantic not in (0, 1):
        raise ValueError('sr_v2_semantic must be integer 0/1')
    if type(hidden) is not int or hidden < 1:
        raise ValueError('S2 semantic hidden width must be a positive integer')
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError('S2 coarse weight must be finite and nonnegative')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('S2 initialization seed must be a nonnegative int64')
    if not isinstance(manifest, str) or not isinstance(digest, str):
        raise ValueError('S2 asset path and hash must be strings')
    if semantic:
        if not manifest or not Path(manifest).is_absolute() or re.fullmatch('[0-9a-f]{64}', digest) is None:
            raise ValueError('S2 semantic encoder requires an absolute manifest path and SHA256')
    elif manifest or digest:
        raise ValueError('Semantic-off must not specify a semantic asset')
    result = ConditionConfig(bool(semantic), hidden, manifest, digest, float(weight), seed)
    if result.enabled:
        if not getattr(args, 'sr_condition', 0) or getattr(args, 'sr_condition_mode', '') != 'prefix_text':
            raise ValueError('S2 condition modules require prefix_text SR')
        if (getattr(args, 'sr_lr_size', 128), getattr(args, 'sr_hr_size', 512),
                getattr(args, 'hbq_round', 4), getattr(args, 'detail_scale_dim', 64)) != (128, 512, 4, 64):
            raise ValueError('S2 supports only 128/512, 64-channel M4 HBQ')
        if (getattr(args, 'sr_cross_attn', 0)
                and not getattr(args, 'sr_v2_cross_attn_promote', 0)):
            raise ValueError('S2 initial comparison retains B without Cross-Attention unless explicit promotion is enabled')
    return result


def materialize_condition_modules(model, args):
    config = condition_config(args)
    if config.enabled:
        model.configure_sr_v2_condition_modules(semantic=config.semantic,
            semantic_hidden=config.semantic_hidden, coarse=config.coarse_weight > 0,
            seed=config.seed)
    return config


def restore_condition_architecture(args, saved_args):
    """Restore full architecture metadata; old/bare checkpoints explicitly stay off."""
    if not isinstance(saved_args, dict):
        raise ValueError('S2 checkpoint args must be a dictionary')
    present = set(FIELD_DEFAULTS) & set(saved_args)
    if present and present != set(FIELD_DEFAULTS):
        raise ValueError('S2 checkpoint requires complete architecture metadata')
    for name, default in FIELD_DEFAULTS.items():
        setattr(args, name, saved_args.get(name, default))
    return condition_config(args)


def build_semantic_encoder(args, device):
    config = condition_config(args)
    if not config.semantic:
        return None
    from grn.utils_t2iv.sr_v2_semantic import FrozenDinoV2SemanticEncoder
    return FrozenDinoV2SemanticEncoder(config.manifest, config.manifest_sha256).to(device=device)


def load_condition_state(model, state, *, promote_b=False):
    """Only an explicit complete B→S2 promotion may omit the entire new branch.

Validate all membership/shapes/dtypes before any copy. This deliberately does
not implement original-T2I→S2 initialization or partial S2→S2 architecture edits.
"""
    if type(promote_b) is not bool or not isinstance(state, dict):
        raise ValueError('Expected explicit boolean promotion and a tensor state dict')
    expected = model.state_dict()
    new_keys = {name for name in expected if name.startswith(NEW_PREFIXES)}
    missing, unexpected = set(expected) - set(state), set(state) - set(expected)
    if promote_b:
        if (not new_keys or any(name.startswith(NEW_PREFIXES) for name in state)
                or 'sr_prefix_pos' not in state
                or not any(name.startswith('sr_condition_encoder.') for name in state)):
            raise RuntimeError('S2 promotion requires a complete trained B source without S2 tensors')
        permitted = new_keys
    else:
        permitted = set()
    mismatched = [name for name in set(expected) & set(state)
        if (not torch.is_tensor(state[name]) or state[name].shape != expected[name].shape
            or state[name].dtype != expected[name].dtype)]
    if missing != permitted or unexpected or mismatched:
        raise RuntimeError(f'S2 checkpoint mismatch: missing={sorted(missing)}, '
                           f'unexpected={sorted(unexpected)}, shape_or_dtype={sorted(mismatched)}')
    for name, value in state.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite S2 checkpoint tensor: {name}')
    if promote_b:
        for name in new_keys:
            value = expected[name]
            if not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite newly initialized S2 tensor: {name}')
            if name.startswith('sr_semantic_fusion.output_proj.') and torch.count_nonzero(value):
                raise RuntimeError('S2 semantic output must remain zero before promotion')
    return model.load_state_dict(state, strict=not promote_b)


def verify_condition_warm_start(model, source_state, source_path, *, rank, promote_b):
    """Evidence after loading: source tensors exact, only declared new tensors added."""
    actual = model.state_dict()
    new_keys = {name for name in actual if name.startswith(NEW_PREFIXES)} if promote_b else set()
    if set(actual) != set(source_state) | new_keys or (set(source_state) & new_keys):
        raise RuntimeError('S2 warm-start schema drift')
    for name, value in actual.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f'Nonfinite loaded S2 tensor: {name}')
        if name not in new_keys:
            source = source_state[name]
            if (value.dtype != source.dtype or value.shape != source.shape or not torch.equal(
                    value.detach().cpu().contiguous().view(torch.uint8), source.detach().cpu().contiguous().view(torch.uint8))):
                raise RuntimeError(f'S2 source tensor changed: {name}')
        if name in new_keys and name.startswith('sr_semantic_fusion.output_proj.') and torch.count_nonzero(value):
            raise RuntimeError('S2 semantic output is not zero after B promotion')
    from grn.utils_t2iv.sr_condition_init import tensor_state_hash
    from grn.utils_t2iv.sr_v2_condition_audit import tensor_state_digest
    new_state = {name: actual[name] for name in new_keys}
    initialization_digest = tensor_state_hash({name: actual[name] for name in new_keys}) if new_keys else None
    return dict(rank=int(rank), source_path=str(Path(source_path).resolve(strict=True)),
                source_tensors_exact=True, source_tensor_count=len(source_state),
                condition_promotion=bool(promote_b), new_condition_keys=sorted(new_keys),
                new_condition_initialization_sha256=initialization_digest,
                new_initialization_sha256=tensor_state_digest(new_state) if new_state else None,
                branch_initialization_sha256={branch: tensor_state_digest(
                    {name: value for name, value in new_state.items() if name.startswith(prefix)})
                    for branch, prefix in (('semantic', 'sr_semantic_fusion.'), ('coarse', 'sr_coarse_head.'))
                    if any(name.startswith(prefix) for name in new_state)},
                source_file_binding='requires_external_dependency_manifest',
                pid=os.getpid(), timestamp_utc=datetime.now(timezone.utc).isoformat())


def condition_coarse_loss(prediction, hr_images):
    """Charbonnier on coarse RGB only; no HR information enters conditioning."""
    if prediction.ndim != 4 or tuple(prediction.shape[1:]) != (3, 32, 32) or prediction.shape[0] < 1:
        raise ValueError('Coarse prediction must be nonempty [B,3,32,32]')
    if not prediction.is_floating_point() or not torch.isfinite(prediction).all():
        raise ValueError('Coarse prediction must be finite floating RGB')
    if len(hr_images) != prediction.shape[0]:
        raise ValueError('Coarse prediction/HR batch mismatch')
    targets = []
    for hr in hr_images:
        if tuple(hr.shape) != (3, 1, 512, 512) or not hr.is_floating_point() or not torch.isfinite(hr).all():
            raise ValueError('Coarse supervision expects finite HR [3,1,512,512]')
        if hr.min() < -1 or hr.max() > 1:
            raise ValueError('Coarse HR pixels must be within [-1,1]')
        targets.append(hr.detach()[:, 0].to(device=prediction.device, dtype=torch.float32))
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        target = F.adaptive_avg_pool2d(torch.stack(targets), (32, 32))
        error = prediction.float() - target
        loss = (torch.sqrt(error.square() + 1e-6) - 1e-3).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite coarse RGB loss')
    return loss


def check_condition_source_config(args, saved_args, *, promote_b):
    """Forbid silent changes of branch topology or frozen semantic asset identity."""
    from types import SimpleNamespace
    requested = condition_config(args)
    saved = SimpleNamespace(**saved_args)
    source = restore_condition_architecture(saved, saved_args)
    if promote_b:
        if (source.enabled or not requested.enabled or not saved_args.get('sr_condition', 0)
                or saved_args.get('sr_condition_mode') != 'prefix_text'):
            raise ValueError('S2 promotion metadata must describe B to new S2 modules')
        return
    old_identity = (source.semantic, source.semantic_hidden if source.semantic else None,
                    source.manifest_sha256 if source.semantic else None, source.coarse_weight > 0)
    new_identity = (requested.semantic, requested.semantic_hidden if requested.semantic else None,
                    requested.manifest_sha256 if requested.semantic else None, requested.coarse_weight > 0)
    if old_identity != new_identity:
        raise ValueError('S2 source architecture/semantic asset identity mismatch')


def validate_condition_counters(saved, config, microsteps):
    required = {'microsteps_completed', 'semantic_batches', 'coarse_batches'}
    if (not isinstance(saved, dict) or set(saved) != required
            or any(type(v) is not int or v < 0 for v in saved.values())):
        raise ValueError('Invalid S2 condition counters')
    expected = dict(microsteps_completed=microsteps,
        semantic_batches=microsteps if config.semantic else 0,
        coarse_batches=microsteps if config.coarse_weight > 0 else 0)
    if saved != expected:
        raise ValueError('S2 condition counters disagree with saved microsteps/branches')
    return dict(saved)
