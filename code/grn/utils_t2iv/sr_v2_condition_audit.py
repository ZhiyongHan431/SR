"""Opt-in observations of real S2 calls and exact distributed resume state.

These are evidence producers, not model logic. No audit claims a tensor was
restored merely because its checkpoint contains the expected metadata.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct

import torch


def tensor_bytes(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def tensor_state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'big'))
        digest.update(header)
        digest.update(tensor_bytes(value).numpy().tobytes())
    return digest.hexdigest()


def compare_state_tree(actual, expected, path='state'):
    """Type/shape/dtype/bytes exact, including signed zero and optimizer trees."""
    if torch.is_tensor(expected):
        if (not torch.is_tensor(actual) or actual.shape != expected.shape or actual.dtype != expected.dtype
                or not torch.isfinite(actual).all() or not torch.isfinite(expected).all()
                or not torch.equal(tensor_bytes(actual), tensor_bytes(expected))):
            raise ValueError(f'Actual restored tensor differs: {path}')
    elif isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f'Actual restored mapping differs: {path}')
        for name in expected:
            compare_state_tree(actual[name], expected[name], f'{path}/{name}')
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError(f'Actual restored sequence differs: {path}')
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_state_tree(left, right, f'{path}/{index}')
    else:
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f'Actual restored scalar differs: {path}')
        if isinstance(expected, float) and (not math.isfinite(actual)
                or struct.pack('!d', actual) != struct.pack('!d', expected)):
            raise ValueError(f'Actual restored float bytes differ: {path}')


def write_fresh_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def validate_audit_flags(args):
    for name in ('sr_v2_condition_audit', 'sr_v2_resume_audit'):
        value = getattr(args, name, 0)
        if type(value) is not int or value not in (0, 1):
            raise ValueError(f'{name} must be integer 0/1')
    if getattr(args, 'sr_v2_condition_audit', 0) or getattr(args, 'sr_v2_resume_audit', 0):
        if not getattr(args, 'sr_v2_runtime_audit_dir', ''):
            raise ValueError('S2 observations require an explicit runtime audit directory')
        if not getattr(args, 'sr_condition', 0) or getattr(args, 'sr_condition_mode', '') != 'prefix_text':
            raise ValueError('S2 observations require continuous prefix_text SR')
    source = getattr(args, 'sr_v2_resume_source', '')
    digest = getattr(args, 'sr_v2_resume_source_sha256', '')
    if not getattr(args, 'sr_v2_resume_audit', 0):
        if source or digest:
            raise ValueError('Explicit resume source requires the strict resume audit')
        return
    import re
    if (not isinstance(source, str) or not Path(source).is_absolute()
            or not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None
            or getattr(args, 'auto_resume', False) is not True
            or getattr(args, 'sr_v2_condition_promote', 0) != 0):
        raise ValueError('Audited resume requires explicit source+SHA, auto_resume=True and promote=0')


def prepare_exact_resume(args, pattern):
    """Resolve only the explicitly pinned checkpoint, before the legacy loader.

    Hashes the actual file on each rank; startup cost is recorded, not hidden.
    No fallback to a different checkpoint or to a fresh optimizer is allowed.
    """
    validate_audit_flags(args)
    if not getattr(args, 'sr_v2_resume_audit', 0):
        return None
    source = Path(args.sr_v2_resume_source).resolve(strict=True)
    folder = Path(args.local_out_path).resolve(strict=True)
    if folder != Path(args.bed).resolve(strict=True):
        raise ValueError('Audited continuation requires one isolated checkpoint directory')
    candidates = list(folder.glob(pattern))
    if len(candidates) != 1 or candidates[0].resolve(strict=True) != source or not source.is_file():
        raise ValueError('Audited continuation must expose exactly its one pinned source checkpoint')
    before = source.stat()
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    after = source.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if digest.hexdigest() != args.sr_v2_resume_source_sha256 or identity(before) != identity(after):
        raise ValueError('Audited resume source hash/read stability mismatch')
    args.sr_v2_resume_binding = dict(path=str(source), sha256=digest.hexdigest())
    args.sr_v2_resume_file_identity = identity(after)
    return str(source)  # Never reopen a mutable directory alias/symlink.


def recheck_exact_resume(args, loaded_path):
    source = Path(loaded_path)
    binding = getattr(args, 'sr_v2_resume_binding', None)
    if not binding or str(source) != binding['path'] or source.resolve(strict=True) != source:
        raise ValueError('Actual resume load path differs from its pinned canonical source')
    stat = source.stat()
    actual = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if actual != args.sr_v2_resume_file_identity:
        raise ValueError('Pinned resume source file identity changed during load')
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    after = source.stat()
    if (digest.hexdigest() != binding['sha256']
            or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != actual):
        raise ValueError('Pinned resume source bytes changed during load/restore')


def validate_resume_architecture_request(args, saved_args):
    from grn.models.sr_deterministic_attention import assert_same_policy
    assert_same_policy(args, saved_args, 'S2/S3 resume')
    from grn.utils_t2iv.sr_v2_calibrator_runtime import assert_same_calibrator_config
    assert_same_calibrator_config(args, saved_args)
    from grn.utils_t2iv.sr_v2_condition_runtime import condition_config, check_condition_source_config
    if (condition_config(args).enabled or getattr(args, 'sr_v2_resume_audit', 0)
            or getattr(args, 'sr_v2_condition_audit', 0)):
        check_condition_source_config(args, saved_args, promote_b=False)


class ConditionRuntimeAudit:
    """Observe actual module calls rather than inferred consumption counters."""
    def __init__(self, args, config, model, encoder, *, rank):
        validate_audit_flags(args)
        if not getattr(args, 'sr_v2_condition_audit', 0):
            raise ValueError('Construct the observer only when explicitly requested')
        self.folder = Path(args.sr_v2_runtime_audit_dir)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.rank, self.config, self.encoder = int(rank), config, encoder
        self.active = None
        self.last_microstep = None
        self.counts = dict(microsteps=0, semantic_encoder_calls=0, coarse_forward_calls=0)
        self.handles = []
        coarse = getattr(model, 'sr_coarse_head', None)
        if (encoder is not None) != config.semantic or (coarse is not None) != (config.coarse_weight > 0):
            raise ValueError('Observed module topology differs from S2 configuration')
        self.frozen_hash = tensor_state_digest(encoder.state_dict()) if encoder is not None else None
        self.provenance = dict(encoder.provenance) if encoder is not None else None
        self.records = self.folder / f'condition_calls_rank{rank}.jsonl'
        # Exclusive creation even if no microstep follows; never append a new
        # process to an old rank's call history.
        with self.records.open('x'):
            pass
        if encoder is not None:
            self.handles.append(encoder.register_forward_pre_hook(self._semantic_pre))
            self.handles.append(encoder.register_forward_hook(self._semantic_post))
        if coarse is not None:
            self.handles.append(coarse.register_forward_hook(self._coarse_post))

    def begin(self, lr, microstep):
        if not torch.is_tensor(lr) or lr.ndim < 2 or lr.shape[0] != 1:
            raise ValueError('S2 runtime evidence requires the declared one-image microbatch')
        if self.active is not None or type(microstep) is not int or microstep < 0:
            raise ValueError('Invalid or nested S2 observed microstep')
        if self.last_microstep is not None and microstep != self.last_microstep + 1:
            raise ValueError('Observed S2 microsteps are not consecutive')
        self.active = dict(lr=lr, microstep=microstep, semantic_encoder_calls=0, coarse_forward_calls=0)

    def _semantic_pre(self, module, inputs):
        if self.active is None or len(inputs) != 1 or inputs[0] is not self.active['lr']:
            raise ValueError('Semantic encoder did not consume this microstep exact LR object')
        if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise ValueError('Observed semantic encoder is not frozen/eval/gradient-free')

    def _semantic_post(self, module, inputs, output):
        if output.requires_grad or output.is_inference() or not torch.isfinite(output).all():
            raise ValueError('Semantic features must be ordinary detached finite tensors')
        self.active['semantic_encoder_calls'] += 1

    def _coarse_post(self, module, inputs, output):
        if self.active is None or not torch.is_grad_enabled() or not output.requires_grad:
            raise ValueError('Coarse head executed outside the final gradient-enabled forward')
        self.active['coarse_forward_calls'] += 1

    def finish(self):
        if self.active is None:
            raise ValueError('No active S2 observed microstep')
        current = self.active
        if (current['semantic_encoder_calls'] != int(self.config.semantic)
                or current['coarse_forward_calls'] != int(self.config.coarse_weight > 0)):
            raise ValueError('Missing or repeated actual S2 condition module calls')
        row = {key: value for key, value in current.items() if key != 'lr'}
        from grn.utils_t2iv.sr_v2_rollout import tensor_sha256
        observed_lr = current['lr'].detach().cpu()
        row.update(rank=self.rank, lr_tensor_sha256=tensor_state_digest({'lr': observed_lr}),
                   lr_actual_input_sha256=tensor_sha256(observed_lr))
        with self.records.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        self.last_microstep = current['microstep']
        self.counts['microsteps'] += 1
        for key in ('semantic_encoder_calls', 'coarse_forward_calls'):
            self.counts[key] += current[key]
        self.active = None

    def snapshot(self, completed_microsteps):
        if self.active is not None or self.last_microstep != completed_microsteps - 1:
            raise ValueError('S2 observed call snapshot is not at the actual checkpoint boundary')
        if self.encoder is not None:
            if (tensor_state_digest(self.encoder.state_dict()) != self.frozen_hash
                    or dict(self.encoder.provenance) != self.provenance
                    or self.encoder.training
                    or any(p.requires_grad or p.grad is not None for p in self.encoder.parameters())):
                raise ValueError('External semantic encoder changed during real training')
        report = dict(rank=self.rank, completed_microsteps=completed_microsteps,
            process_observed_counts=dict(self.counts), last_microstep=self.last_microstep,
            frozen_semantic_state_sha256=self.frozen_hash, semantic_provenance=self.provenance,
            semantic_frozen_unchanged=True if self.encoder is not None else None)
        write_fresh_json(self.folder / f'condition_snapshot_{completed_microsteps}_rank{self.rank}.json', report)
        return report


def audit_distributed_resume(trainer, saved, args):
    """After real Trainer restore, export actual FSDP model and optimizer state."""
    from torch import distributed as dist
    from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, StateDictType,
        FullStateDictConfig, FullOptimStateDictConfig)
    validate_audit_flags(args)
    if (not getattr(args, 'sr_v2_resume_audit', 0) or not dist.is_initialized()
            or dist.get_world_size() != 2 or not isinstance(trainer.gpt, FSDP)
            or not torch.cuda.is_available()):
        raise ValueError('Resume evidence requires the actual two-rank CUDA FSDP restore')
    binding = getattr(args, 'sr_v2_resume_binding', None)
    if not binding:
        raise ValueError('Missing actual pinned resume-file loading evidence')
    with FSDP.state_dict_type(trainer.gpt, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)):
        restored_model = trainer.gpt.state_dict()
        restored_optimizer = FSDP.optim_state_dict(trainer.gpt, trainer.gpt_opt)
    compare_state_tree(restored_model, saved['gpt_fsdp'], 'model')
    compare_state_tree(restored_optimizer, saved['gpt_fsdp_opt'], 'optimizer')
    compare_state_tree(trainer.sr_v2_state, saved['sr_v2'], 'sr_v2')
    condition = getattr(trainer, 'sr_v2_condition_state', None)
    compare_state_tree(condition, saved.get('sr_v2_condition'), 'sr_v2_condition')
    encoder = getattr(trainer, 'sr_v2_semantic_encoder', None)
    provenance = dict(encoder.provenance) if encoder is not None else None
    compare_state_tree(provenance, saved.get('sr_v2_semantic_provenance'), 'semantic_provenance')
    update = trainer.sr_v2_state['optimizer_updates_completed']
    recheck_exact_resume(args, binding['path'])
    if not restored_optimizer['state'] or any(float(value['step']) != update
                                              for value in restored_optimizer['state'].values()):
        raise ValueError('Restored Adam optimizer steps differ from real trainer counters')
    record = dict(status='passed', rank=dist.get_rank(), world_size=dist.get_world_size(),
        pid=os.getpid(), timestamp_utc=datetime.now(timezone.utc).isoformat(), source=binding,
        actual_distributed_optimizer_restore=True, model_bytes_exact=True, optimizer_bytes_exact=True,
        model_tensors=len(restored_model), optimizer_states=len(restored_optimizer['state']),
        restored_optimizer_updates=update, additional_updates=0,
        before=dict(sr_v2=dict(trainer.sr_v2_state), sr_v2_condition=condition),
        semantic_provenance_exact=True)
    from grn.utils_t2iv.sr_v2_original_runtime import original_resume_evidence
    record.update(original_resume_evidence(trainer, saved, args))
    write_fresh_json(Path(args.sr_v2_runtime_audit_dir) / f'resume_initial_rank{dist.get_rank()}.json', record)
    return record
