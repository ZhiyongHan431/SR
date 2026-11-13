"""Isolated S5 original-GRN initialization prototype; never a B/S2 loader.

The public entry accepts only the actually audited original and HBQ file hashes.
It owns common-init -> original -> HBQ -> fresh LR projection -> S2 ordering.
No optimizer is accepted, loaded, created, or described as restored by this API.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import torch

ORIGINAL_SHA = '8647af94c6499ba1b34f759ca60732a88fc35c59cd131587812cfbc31a42dd9c'
HBQ_SHA = '288019ad1908b72f9b4e451fe95503632066d7a3ff12c85bbd6fbd37396511c0'
METADATA_SHA = '791e3d72fcdd77e7007bec6b3cf624d917e812bff27fc4136a5597f9363a51e1'
DEFAULT_METADATA = Path('/inspire/store/project/fdu-nicu/public/hanzhiyong/GRN/outputs/sr_v2_20260916.QCinZF/original_grn_cpu_audit.1bYR33/metadata.json')
BRANCHES = ('sr_semantic_fusion.', 'sr_coarse_head.')


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple
    dtype: str


@dataclass(frozen=True)
class OriginalRequest:
    # Explicit different lifecycle; not a renamed condition_promote flag.
    initialization_kind: str
    semantic: bool = False
    coarse: bool = False
    semantic_hidden: int = 128
    legacy_seed: int = 3407
    condition_seed: int = 20260916
    auto_resume: bool = False
    promote_b: bool = False
    resume: bool = False
    common_condition_path: str = ''
    calibrator: object = None  # explicit CalibratorConfig; absent retains original S2 lifecycle

    def validate(self):
        if self.calibrator is not None:
            from grn.utils_t2iv.sr_v2_calibrator_runtime import CalibratorConfig, calibrator_config, config_namespace
            if not isinstance(self.calibrator, CalibratorConfig):
                raise ValueError('Original S3 request requires typed complete configuration')
            calibrator_config(config_namespace(self.calibrator))
        if self.initialization_kind != 'original_grn':
            raise ValueError('Explicit original_grn initialization kind required')
        for name in ('semantic', 'coarse', 'auto_resume', 'promote_b', 'resume'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be a strict boolean')
        if self.auto_resume or self.promote_b or self.resume:
            raise ValueError('Original initialization is mutually exclusive with resume/B promotion')
        if self.common_condition_path != '':
            raise ValueError('Prototype does not accept external/shared trained condition bundles')
        if type(self.semantic_hidden) is not int or self.semantic_hidden != 128:
            raise ValueError('Only the audited semantic hidden width 128 is supported')
        for name in ('legacy_seed', 'condition_seed'):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError(f'{name} must be an integer in [0,2**63)')


@dataclass(frozen=True)
class Contract:
    original: dict
    legacy: dict
    hbq: dict
    hbq_copies: dict
    shortcuts: tuple
    hbq_removed: tuple
    fresh_lr: tuple
    semantic: dict
    coarse: dict


class OriginalInitializationFailure(RuntimeError):
    def __init__(self, stage, evidence, cause):
        super().__init__(f'Original initialization failed at {stage}: {cause}')
        self.stage, self.evidence = stage, evidence


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while data := source.read(8 * 1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def pin_file(path, expected):
    if not isinstance(expected, str) or re.fullmatch('[0-9a-f]{64}', expected) is None:
        raise ValueError('Explicit lowercase SHA256 required')
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file() or file_sha(resolved) != expected:
        raise ValueError('Pinned source file SHA mismatch')
    info = resolved.stat()
    return dict(path=str(resolved), sha256=expected, bytes=info.st_size,
        inode=info.st_ino, device=info.st_dev, mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)


def recheck_file(binding):
    if pin_file(binding['path'], binding['sha256']) != binding:
        raise ValueError('Source identity changed during initialization')


def specs(state):
    return {name: TensorSpec(tuple(value.shape), str(value.dtype)) for name, value in state.items()}


def load_audited_contract(path=DEFAULT_METADATA):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != METADATA_SHA:
        raise ValueError('Only the pinned real original/B/HBQ schema audit is accepted')
    audit = json.loads(data)
    def convert(items):
        return {name: TensorSpec(tuple(value['shape']), value['dtype']) for name, value in items.items()}
    original = convert(audit['checkpoints']['original']['schema'])
    base = convert(audit['checkpoints']['B']['model_schema'])
    legacy = {name: base[name] for name in audit['original_to_B']['B_only_keys']}
    hbq = convert(audit['checkpoints']['HBQ']['selected_schema'])
    copies = audit['HBQ_to_legacy_condition_schema']['direct_mapping']
    shortcuts = tuple(sorted(name for name in legacy if '.avg_shortcut.proj.' in name))
    removed = tuple(sorted(set(name for name in hbq if name.startswith(('encoder.', 'conv1.'))) - set(copies.values())))
    fresh = ('sr_condition_encoder.feature_proj.weight', 'sr_condition_encoder.feature_proj.bias', 'sr_prefix_pos')
    semantic = convert(audit['S2_new_module_schema']['semantic'])
    coarse = convert(audit['S2_new_module_schema']['coarse'])
    if (audit['status'] != 'passed' or len(original) != 319 or len(legacy) != 89
            or len(copies) != 82 or len(shortcuts) != 4 or len(removed) != 4
            or len(semantic) != 4 or len(coarse) != 4 or len(hbq) != 196
            or set(original) & set(legacy) or set(original) | set(legacy) != set(base)
            or set(copies) | set(shortcuts) | set(fresh) != set(legacy)):
        raise ValueError('Audited exact original/condition schema contract is inconsistent')
    return Contract(original, legacy, hbq, dict(copies), shortcuts, removed, fresh, semantic, coarse)


def validate_state(state, expected, label, *, finite=True):
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError(f'{label} must contain the exact complete declared tensor keys')
    for name, spec in expected.items():
        value = state[name]
        if (not torch.is_tensor(value) or tuple(value.shape) != spec.shape
                or str(value.dtype) != spec.dtype or value.layout != torch.strided
                or value.device.type == 'meta'):
            raise ValueError(f'{label} tensor shape/dtype/layout mismatch: {name}')
        if finite and (not value.is_floating_point() or not torch.isfinite(value).all().item()):
            raise ValueError(f'{label} tensor must be finite floating: {name}')


def tensor_bytes(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def exact(actual, expected, label):
    if (actual.shape != expected.shape or actual.dtype != expected.dtype
            or not torch.equal(tensor_bytes(actual), tensor_bytes(expected))):
        raise ValueError(f'{label} bytes changed')


def state_digest(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        for part in (name.encode(), str(value.dtype).encode(), repr(tuple(value.shape)).encode(),
                     tensor_bytes(value).numpy().tobytes()):
            result.update(len(part).to_bytes(8, 'big'))
            result.update(part)
    return result.hexdigest()


def shortcut_expected(value):
    if value.ndim == 1:
        return torch.zeros_like(value, device='cpu')
    if value.ndim != 5 or tuple(value.shape[2:]) != (1, 1, 1) or value.shape[0] % value.shape[1]:
        raise ValueError('Stride-one shortcut requires declared integer channel-repeat layout')
    result = torch.zeros_like(value, device='cpu')
    rows = torch.arange(value.shape[0])
    result[rows, rows // (value.shape[0] // value.shape[1]), 0, 0, 0] = 1
    return result


@torch.no_grad()
def _initialize_checked(model, hbq, original, hbq_state, contract, request, common_initialize):
    """Internal engine; tiny contracts are for CPU tests, not the public loader."""
    request.validate()
    if not callable(common_initialize):
        raise ValueError('Caller must supply the actual common/global+special initialization callback')
    if getattr(model, '_original_init_lifecycle', None) is not None:
        raise ValueError('Repeated or failed original initialization cannot be reused')
    if getattr(model, 'sr_condition', None) is not True or getattr(model, 'sr_condition_mode', None) != 'prefix_text':
        raise ValueError('Original S5 prototype requires prefix_text SR')
    if any(hasattr(model, branch.removesuffix('.')) for branch in (*BRANCHES, 'sr_round_calibrator.')):
        raise ValueError('S2 branches must be absent until after original and HBQ restoration')
    base = dict(contract.original, **contract.legacy)
    validate_state(original, contract.original, 'Original bare checkpoint')
    validate_state(model.state_dict(), base, 'Target base topology', finite=False)
    validate_state(hbq_state, contract.hbq, 'HBQ source')
    validate_state(hbq.state_dict(), contract.hbq, 'Actual HBQ')
    if hbq.training or any(p.requires_grad or p.grad is not None for p in hbq.parameters()):
        raise ValueError('Actual source HBQ must be frozen/eval without gradients')
    for name, value in hbq.state_dict().items():
        exact(value, hbq_state[name], f'Actual HBQ source {name}')
    if (set(contract.hbq_copies) | set(contract.shortcuts) | set(contract.fresh_lr) != set(contract.legacy)
            or set(contract.hbq_copies) & set(contract.shortcuts)
            or any(name not in contract.hbq for name in contract.hbq_copies.values())
            or any(name not in contract.hbq for name in contract.hbq_removed)):
        raise ValueError('Condition mapping must partition the exact legacy keys')
    for target, source in contract.hbq_copies.items():
        if contract.legacy[target] != contract.hbq[source]:
            raise ValueError('HBQ-copy schema mismatch before mutation')
    evidence = dict(kind='original_grn_initialization', status='running', events=[],
        optimizer_loaded=False, resume_performed=False, trained_B_state_used=False,
        original_tensor_count=len(original), hbq_copy_count=len(contract.hbq_copies),
        hbq_removed_keys=list(contract.hbq_removed), shortcut_new_keys=list(contract.shortcuts),
        source_shared_tensor_digest=state_digest(original), hbq_tensor_digest=state_digest(hbq_state),
        legacy_seed=request.legacy_seed, condition_seed=request.condition_seed)
    stage = 'common_initialization'
    model._original_init_lifecycle = 'running'
    try:
        common_initialize(model)
        evidence['events'].append(stage)
        validate_state(model.state_dict(), base, 'Post-common target', finite=False)
        stage = 'original_shared_load'
        result = model.load_state_dict(original, strict=False)
        if set(result.missing_keys) != set(contract.legacy) or result.unexpected_keys:
            raise ValueError('Actual shared load missing/unexpected keys differ from exact allowlist')
        for name, value in original.items():
            exact(model.state_dict()[name], value, f'Original immediately loaded {name}')
        evidence['events'].append(stage)
        stage = 'hbq_condition_restore'
        model.initialize_sr_condition_from_hbq(hbq)
        current = model.state_dict()
        validate_state(current, base, 'Post-HBQ target', finite=False)
        for target, source in contract.hbq_copies.items():
            exact(current[target], hbq_state[source], f'HBQ copy {target}')
        for name in contract.shortcuts:
            exact(current[name], shortcut_expected(current[name]), f'Channel-repeat shortcut {name}')
        evidence['events'].append(stage)
        stage = 'fresh_lr_projection_and_position'
        generator = torch.Generator(device='cpu').manual_seed(request.legacy_seed)
        for name in contract.fresh_lr:
            target = current[name]
            fresh = torch.empty_like(target, device='cpu')
            if name.endswith('.bias'):
                fresh.zero_()
            else:
                torch.nn.init.trunc_normal_(fresh, std=.02, generator=generator)
            target.copy_(fresh)
        legacy_digest = state_digest({name: current[name] for name in contract.legacy})
        evidence['events'].append(stage)
        stage = 's2_branch_materialization'
        model.configure_sr_v2_condition_modules(semantic=request.semantic, coarse=request.coarse,
            semantic_hidden=request.semantic_hidden, seed=request.condition_seed)
        evidence['events'].append(stage)
        expected = dict(base)
        if request.semantic:
            expected.update(contract.semantic)
        if request.coarse:
            expected.update(contract.coarse)
        if request.calibrator is not None and request.calibrator.enabled:
            stage = 's3_branch_materialization'
            from grn.utils_t2iv.sr_v2_calibrator_runtime import materialize_calibrator, config_namespace, original_specs
            materialize_calibrator(model, config_namespace(request.calibrator))
            expected.update(original_specs(request.calibrator, contract))
            evidence['s3_calibrator'] = request.calibrator.metadata()
            evidence['events'].append(stage)
        stage = 'final_byte_and_schema_audit'
        current = model.state_dict()
        validate_state(current, expected, 'Final initialized model')
        for name, value in original.items():
            exact(current[name], value, f'Final original shared {name}')
        if state_digest({name: current[name] for name in contract.legacy}) != legacy_digest:
            raise ValueError('S2 creation changed previously established legacy condition bytes')
        for name, value in hbq.state_dict().items():
            exact(value, hbq_state[name], f'Frozen HBQ after initialization {name}')
        if hbq.training or any(p.requires_grad or p.grad is not None for p in hbq.parameters()):
            raise ValueError('Source HBQ freeze/eval/gradient state changed')
        for name in contract.semantic if request.semantic else ():
            if name.startswith('sr_semantic_fusion.output_proj.') and torch.count_nonzero(current[name]).item():
                raise ValueError('S2 semantic output must remain exactly zero')
        evidence['events'].append(stage)
        new_names = sorted(set(expected) - set(contract.original))
        evidence.update(status='passed', source_shared_tensors_byte_exact=True,
            hbq_copied_tensors_byte_exact=True, shortcut_channel_repeat_exact=True,
            permitted_original_missing_keys=new_names, final_tensor_count=len(expected),
            legacy_condition_digest=legacy_digest,
            new_branches_digest=state_digest({name: current[name] for name in set(expected) - set(base)}),
            branch_keys=sorted(set(expected) - set(base)))
        model._original_init_lifecycle = 'complete'
        return evidence
    except BaseException as error:
        model._original_init_lifecycle = 'failed_discard_model'
        evidence.update(status='failed', failure_stage=stage, error=f'{type(error).__name__}: {error}')
        raise OriginalInitializationFailure(stage, evidence, error) from error


def initialize_original(model, hbq, request, *, source, source_sha256,
                        hbq_checkpoint, hbq_sha256, common_initialize,
                        audit_metadata=DEFAULT_METADATA):
    """Public, version-pinned original-only boundary; no source/container fallback.

    Validation failures before common_initialize do not mutate the model. A later
    failure poisons the model: discard it, never continue or construct optimizer.
    The two complete source files are rehashed after initialization.
    """
    request.validate()
    if source_sha256 != ORIGINAL_SHA or hbq_sha256 != HBQ_SHA:
        raise ValueError('Only the explicitly audited original GRN and HBQ identities are supported')
    contract = load_audited_contract(audit_metadata)
    source_binding = pin_file(source, source_sha256)
    hbq_binding = pin_file(hbq_checkpoint, hbq_sha256)
    original = torch.load(source_binding['path'], map_location='cpu', mmap=True, weights_only=True)
    hbq_payload = torch.load(hbq_binding['path'], map_location='cpu', mmap=True, weights_only=True)
    if not isinstance(hbq_payload, dict) or set(hbq_payload) != {'ema'}:
        raise ValueError('Only the exact audited HBQ EMA container is supported')
    result = _initialize_checked(model, hbq, original, hbq_payload['ema'], contract,
        request, common_initialize)
    try:
        recheck_file(source_binding)
        recheck_file(hbq_binding)
        load_audited_contract(audit_metadata)
    except BaseException as error:
        model._original_init_lifecycle = 'failed_discard_model'
        result.update(status='failed', failure_stage='final_source_identity', error=str(error))
        raise OriginalInitializationFailure('final_source_identity', result, error) from error
    result.update(source=source_binding, hbq_source=hbq_binding,
        audit_metadata_sha256=METADATA_SHA, source_files_rehashed_after_initialization=True)
    return result
