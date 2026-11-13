"""Canonical common-condition bundles and exact continuous-SR checkpoint loading."""
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re

import torch
from torch import Tensor, nn


def condition_state(model: nn.Module) -> dict[str, Tensor]:
    """Snapshot only the shared encoder/projection and positional tensors."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if name == 'sr_prefix_pos' or name.startswith('sr_condition_encoder.')}


def tensor_state_hash(state: Mapping[str, Tensor]) -> str:
    """Hash sorted names, dtypes, shapes, and raw bytes with length framing."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        fields = (name.encode(), str(value.dtype).encode(),
                  json.dumps(list(value.shape), separators=(',', ':')).encode(),
                  value.reshape(-1).view(torch.uint8).numpy().tobytes())
        for field in fields:
            digest.update(len(field).to_bytes(8, 'big'))
            digest.update(field)
    return digest.hexdigest()


def load_shared_condition(model: nn.Module, path: str | Path) -> dict[str, str]:
    """Validate the full bundle before copying; return hashes of actual loaded state."""
    payload = torch.load(path, map_location='cpu', weights_only=True)
    required = {'schema_version', 'source_sha256', 'vae_sha256', 'state_sha256', 'state'}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError('Shared condition bundle requires complete schema/provenance')
    if type(payload['schema_version']) is not int or payload['schema_version'] != 1:
        raise ValueError('Shared condition bundle requires schema_version=1')
    for key in ('source_sha256', 'vae_sha256', 'state_sha256'):
        if not isinstance(payload[key], str) or re.fullmatch('[0-9a-f]{64}', payload[key]) is None:
            raise ValueError(f'Invalid {key} in shared condition bundle')
    expected = condition_state(model)
    state = payload['state']
    if not expected or not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('Shared condition keys do not match the complete model condition state')
    for name, value in state.items():
        if (not isinstance(value, Tensor) or value.shape != expected[name].shape
                or value.dtype != expected[name].dtype or not torch.isfinite(value).all()):
            raise ValueError(f'Invalid shared condition tensor: {name}')
    if tensor_state_hash(state) != payload['state_sha256']:
        raise ValueError('Shared condition state_sha256 does not match actual tensors')
    with torch.no_grad():
        target = model.state_dict()
        for name, value in state.items():
            target[name].copy_(value)
    actual_hash = tensor_state_hash(condition_state(model))
    if actual_hash != payload['state_sha256']:
        raise RuntimeError('Loaded common condition tensor hash changed')
    return dict(state_sha256=actual_hash, source_sha256=payload['source_sha256'],
                vae_sha256=payload['vae_sha256'])


def load_abc_state(model: nn.Module, state: Mapping[str, Tensor], warm_start: bool = False) -> object:
    """Only a complete original-T2I state can omit all new SR tensors.

    A and B have identical tensor schemas; checkpoint args, restored by the caller,
    carry their architecture identity. No state-dict metadata is invented here.
    """
    expected = model.state_dict()
    new_keys = {name for name in expected if name == 'sr_prefix_pos' or name.startswith(
        ('sr_condition_encoder.', 'sr_cross_attn.', 'sr_spatial_controls.'))}
    missing, unexpected = set(expected) - set(state), set(state) - set(expected)
    permitted = new_keys if warm_start and not (new_keys & set(state)) else set()
    mismatched = [name for name in set(expected) & set(state)
                  if not isinstance(state[name], Tensor) or state[name].shape != expected[name].shape]
    if missing != permitted or unexpected or mismatched:
        raise RuntimeError(f'ABC checkpoint mismatch: missing={sorted(missing)}, '
                           f'unexpected={sorted(unexpected)}, shapes={sorted(mismatched)}')
    return model.load_state_dict(state, strict=not permitted)
