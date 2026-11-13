"""Opt-in LR-only multimodal conditioning without altering the GRN source path.

Load and verify the fixed base and C+R sidecar before calling configure(). The
caller retains responsibility for enforcing kappa=2 and verifying LR-feature
cache provenance. No context means no adapter computation. A feature context
must enclose forward AND backward when activation checkpointing is used.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import weakref

import torch
from torch import nn

from grn.models.sr_multimodal_condition import MultimodalConditionAdapter, condition_alpha

SCHEMA = 'grn_s5_multimodal_condition_v1'
ADAPTER_NAME = 'sr_multimodal_condition'
_CONTEXT = threading.local()
_ARCH_KEYS = {'condition_dim', 'input_dim', 'hidden_dim', 'heads', 'num_tokens'}


def _plain(value, depth=0):
    if depth > 32:
        raise ValueError('Metadata nesting is too deep')
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) in (list, tuple):
        for item in value:
            _plain(item, depth + 1)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _plain(item, depth + 1)
        return
    raise ValueError('Metadata must contain only finite JSON-compatible values')


def _identity(identity):
    required = {'base_sha256', 'trajectory_adapter_sha256', 'feature_manifest_sha256', 'feature_source'}
    if type(identity) is not dict or not required.issubset(identity):
        raise ValueError('Identity must bind base, C+R, feature cache, and feature source')
    _plain(identity)
    for name in required - {'feature_source'}:
        if type(identity[name]) is not str or not re.fullmatch('[0-9a-f]{64}', identity[name]):
            raise ValueError(f'Invalid identity SHA256: {name}')
    if identity['feature_source'] not in ('qwen', 'dino'):
        raise ValueError('feature_source must be qwen or dino')
    return copy.deepcopy(identity)


def _adapter(model):
    value = getattr(model, ADAPTER_NAME, None)
    seen = set()
    while isinstance(value, nn.Module) and not isinstance(value, MultimodalConditionAdapter):
        if id(value) in seen:
            break
        seen.add(id(value))
        value = getattr(value, 'module', None)
    if not isinstance(value, MultimodalConditionAdapter):
        raise ValueError('A configured MultimodalConditionAdapter is required')
    return value


def _binding(model):
    stack = getattr(_CONTEXT, 'bindings', {}).get(id(model), ())
    return stack[-1] if stack else None


def freeze_multimodal_parameters(model, *, trainable=True):
    """Keep every old weight frozen/eval; only the new adapter is trainable."""
    if type(trainable) is not bool:
        raise ValueError('trainable must be an explicit bool')
    adapter = _adapter(model)
    model.requires_grad_(False)
    model.eval()
    for parameter in model.parameters():
        parameter.grad = None
    adapter.requires_grad_(trainable)
    adapter.train(trainable)
    selected = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if any(not name.startswith(ADAPTER_NAME + '.') for name in selected):
        raise RuntimeError('An old parameter was unexpectedly selected for training')
    return selected


def configure_multimodal_condition(model, input_dim=512, hidden_dim=256, heads=8,
                                  num_tokens=65, seed=20261110, trainable=True):
    """Attach one zero-output residual hook to the existing fused condition grid.

    The hook modifies the shared memory used by Prefix and all Cross-Attention
    layers. It leaves the original LR latent used by C/R intact. Existing model
    weights are never copied into or modified by this function.
    """
    fusion = getattr(model, 'sr_semantic_fusion', None)
    if (not isinstance(fusion, nn.Module) or not hasattr(model, 'sr_round_calibrator')
            or not hasattr(model, 'sr_latent_residual')):
        raise ValueError('Load the verified S5 semantic model and fixed C+R sidecar first')
    condition_dim = getattr(fusion, 'embed_dim', None)
    if type(condition_dim) is not int or condition_dim < 1:
        raise ValueError('The semantic fusion must expose its condition feature dimension')
    if type(seed) is not int or not 0 <= seed < 2**63 or type(trainable) is not bool:
        raise ValueError('Expected a nonnegative CPU seed and explicit trainable bool')
    architecture = dict(condition_dim=condition_dim, input_dim=input_dim, hidden_dim=hidden_dim,
                        heads=heads, num_tokens=num_tokens)
    existing = getattr(model, ADAPTER_NAME, None)
    if existing is None:
        with torch.device('cpu'), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            adapter = MultimodalConditionAdapter(**architecture)
        reference = next(fusion.parameters(), None)
        if reference is None:
            raise ValueError('The existing semantic fusion has no device reference')
        setattr(model, ADAPTER_NAME, adapter.to(device=reference.device, dtype=torch.float32))
    else:
        adapter = _adapter(model)
        if adapter.architecture() != architecture:
            raise ValueError('Refusing to replace an existing multimodal adapter architecture')
    handle = getattr(model, '_sr_multimodal_hook_handle', None)
    if handle is not None and handle.id not in fusion._forward_hooks:
        raise ValueError('Multimodal hook ownership changed; detach before reconfiguring')
    if handle is None:
        owner = weakref.ref(model)
        def inject(_module, _inputs, output):
            current = owner()
            binding = _binding(current) if current is not None else None
            if binding is None or not binding['enabled'] or binding['alpha'] == 0:
                return None  # Preserve the original output tensor exactly.
            return getattr(current, ADAPTER_NAME)(output, binding['tokens'],
                                                   mask=binding['mask'], alpha=binding['alpha'])
        model._sr_multimodal_hook_handle = fusion.register_forward_hook(inject)
    freeze_multimodal_parameters(model, trainable=trainable)
    return _adapter(model)


def detach_multimodal_hook(model):
    """Remove the opt-in hook, retaining sidecar parameters for later reuse."""
    if _binding(model) is not None:
        raise RuntimeError('Cannot detach a hook inside a live feature context')
    handle = getattr(model, '_sr_multimodal_hook_handle', None)
    if handle is not None:
        handle.remove()
        delattr(model, '_sr_multimodal_hook_handle')


@contextmanager
def multimodal_condition(model, tokens=None, mask=None, alpha=1.0, enabled=True):
    """Bind one LR feature batch until context exit, including backward replay.

    Contexts are thread-local and may nest. Exceptions restore the previous
    context. Feature masks use True for valid tokens. alpha=0 or enabled=False
    needs no tokens and executes the original condition path unchanged.
    """
    _adapter(model)
    handle = getattr(model, '_sr_multimodal_hook_handle', None)
    fusion = getattr(model, 'sr_semantic_fusion', None)
    if handle is None or handle.id not in fusion._forward_hooks:
        raise ValueError('Install the multimodal hook before binding features')
    if type(enabled) is not bool:
        raise ValueError('enabled must be an explicit bool')
    alpha = condition_alpha(alpha)
    if enabled and alpha != 0 and not torch.is_tensor(tokens):
        raise ValueError('An active multimodal context requires cached LR feature tokens')
    bindings = getattr(_CONTEXT, 'bindings', None)
    if bindings is None:
        bindings = _CONTEXT.bindings = {}
    stack = bindings.setdefault(id(model), [])
    binding = dict(tokens=tokens, mask=mask, alpha=alpha, enabled=enabled)
    stack.append(binding)
    try:
        yield _adapter(model)
    finally:
        if not stack or stack[-1] is not binding:
            raise RuntimeError('Multimodal feature contexts exited out of order')
        stack.pop()
        if not stack:
            del bindings[id(model)]


def save_multimodal_condition(path, model, *, identity, step=0, metadata=None):
    """Save only the new adapter, atomically and without overwriting any file."""
    adapter = _adapter(model)
    identity = _identity(identity)
    if type(step) is not int or step < 0:
        raise ValueError('step must be a nonnegative integer')
    metadata = {} if metadata is None else metadata
    if type(metadata) is not dict:
        raise ValueError('metadata must be a dict')
    _plain(metadata)
    state = {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError('Cannot save nonfinite adapter parameters')
    payload = dict(schema=SCHEMA, identity=identity, architecture=adapter.architecture(),
                   state_dict=state, step=step, metadata=copy.deepcopy(metadata))
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Refusing to overwrite an existing file: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='.' + path.name + '.', suffix='.tmp',
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(path=str(path), architecture=payload['architecture'], identity=identity, step=step)


def load_multimodal_condition(path, model, *, expected_identity, trainable=False):
    """Validate identities and exact tensor membership before any weight copy."""
    expected_identity = _identity(expected_identity)
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    expected_keys = {'schema', 'identity', 'architecture', 'state_dict', 'step', 'metadata'}
    if type(payload) is not dict or set(payload) != expected_keys or payload['schema'] != SCHEMA:
        raise ValueError('Unsupported or incomplete multimodal sidecar schema')
    if _identity(payload['identity']) != expected_identity:
        raise ValueError('Base/C+R/feature identity does not match the requested sidecar')
    architecture = payload['architecture']
    if type(architecture) is not dict or set(architecture) != _ARCH_KEYS:
        raise ValueError('Invalid multimodal adapter architecture')
    if type(payload['step']) is not int or payload['step'] < 0 or type(payload['metadata']) is not dict:
        raise ValueError('Invalid step or metadata')
    _plain(payload['metadata'])
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        prototype = MultimodalConditionAdapter(**architecture)
    state, expected = payload['state_dict'], prototype.state_dict()
    if type(state) is not dict or set(state) != set(expected):
        raise ValueError('Sidecar must contain exactly the new adapter tensors')
    for key, value in state.items():
        reference = expected[key]
        if (not torch.is_tensor(value) or value.layout != torch.strided
                or value.shape != reference.shape or value.dtype != reference.dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f'Invalid sidecar tensor: {key}')
    if architecture['condition_dim'] != getattr(getattr(model, 'sr_semantic_fusion', None), 'embed_dim', None):
        raise ValueError('Sidecar condition dimension does not match the loaded base')
    spec = {key: value for key, value in architecture.items() if key != 'condition_dim'}
    adapter = configure_multimodal_condition(model, **spec, trainable=trainable)
    adapter.load_state_dict(state, strict=True)
    return dict(path=str(path), identity=expected_identity, architecture=architecture,
                step=payload['step'], metadata=payload['metadata'])
