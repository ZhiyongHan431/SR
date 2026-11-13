"""Explicit original-condition adaptation -> low-LR joint weights-only startup.

This never resumes an adaptation optimizer or relabels B as original adaptation.
The CPU-loaded source must prove unchanged original shared bytes, changed condition
weights and positive actual training counters. GPU/stage-selection gates remain
the responsibility of the producer/runner, not inferred from these tensor checks.
"""
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import re

import torch

from grn.utils_t2iv import sr_v2_original_init as original
from grn.utils_t2iv import sr_v2_original_runtime as runtime
from grn.utils_t2iv.sr_v2_condition_runtime import (
    FIELD_DEFAULTS, condition_config, load_condition_state)
from grn.utils_t2iv.sr_v2_qualified_data import parse

FIELDS = ('sr_original_joint_source', 'sr_original_joint_source_sha256', 'sr_original_joint_provenance')
IDENTITY = ('sr_initialization_kind', 'sr_original_path', 'sr_original_sha256',
    'sr_original_hbq_sha256', 'sr_original_legacy_seed', 'vae_path', 'model', 'Ct5',
    'sr_lr_size', 'sr_hr_size', 'hbq_round', 'detail_scale_dim', 'semantic_scale_dim',
    'num_of_label_value', 'simple_text_proj', 'use_fsq_cls_head', 'use_ada_layer_norm',
    'add_scale_token', 'add_class_token', 'apply_spatial_patchify', 'video_frames',
    'refine_mode', 'sr_cross_attn', 'sr_condition', 'sr_condition_mode')


def learning_policy(args):
    values = {name: getattr(args, name, None) for name in
        ('tlr', 'sr_backbone_lr_scale', 'sr_head_lr_scale', 'sr_new_lr_scale')}
    runtime.require(all(type(value) in (int, float) and math.isfinite(value) and value > 0
        for value in values.values()), 'Joint learning rates/scales must be finite and positive')
    runtime.require(0 < values['sr_backbone_lr_scale'] <= values['sr_head_lr_scale']
        < values['sr_new_lr_scale'] and values['sr_backbone_lr_scale'] <= .2
        and values['tlr'] * values['sr_backbone_lr_scale'] <= 1e-5,
        'Joint phase requires explicitly bounded low backbone LR and stronger condition LR')
    return values


def validate_flags(args):
    if runtime.phase(args) != 'joint':
        runtime.require(not any(getattr(args, key, '') for key in FIELDS),
            'Joint-specific fields require explicit joint phase')
        return
    source, digest = (getattr(args, key, '') for key in FIELDS[:2])
    runtime.require(isinstance(source, str) and Path(source).is_absolute()
        and isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest),
        'Joint phase requires an absolute adaptation source and explicit SHA256')
    learning_policy(args)
    if not args.auto_resume:
        runtime.require(not getattr(args, FIELDS[2], ''), 'Fresh joint phase cannot inherit a transition report')


def validate_lineage(value, args):
    runtime.require(isinstance(value, dict) and value.get('kind') == 'original_grn_joint_weights_only'
        and value.get('status') == 'passed', 'Completed original joint transition provenance required')
    for name in ('source_model_bytes_exact', 'original_shared_unchanged_at_transition',
                 'condition_weights_changed_during_adaptation', 'source_rehashed_after_load'):
        runtime.require(value.get(name) is True, f'Missing actual joint transition check: {name}')
    runtime.require(value.get('optimizer_loaded') is False and value.get('resume_performed') is False
        and type(value.get('source_optimizer_updates')) is int and value['source_optimizer_updates'] > 0,
        'Joint phase is a fresh optimizer, not an adaptation resume')
    binding = value.get('source', {})
    runtime.require(binding.get('path') == str(Path(args.sr_original_joint_source).resolve())
        and binding.get('sha256') == args.sr_original_joint_source_sha256,
        'Joint adaptation source identity differs')
    origin = runtime.trainer_origin(args)
    runtime.require(value.get('original_initialization_sha256') == hashlib.sha256(runtime.encode(origin).encode()).hexdigest(),
        'Joint source original initialization lineage differs')
    runtime.require(value.get('learning_policy') == learning_policy(args), 'Joint learning policy differs')
    for name in ('source_model_digest', 'source_condition_digest'):
        runtime.require(isinstance(value.get(name), str) and re.fullmatch('[0-9a-f]{64}', value[name]),
            'Joint source tensor digest missing')
    return value


def prepare(args, resume_state):
    """Read-only before model allocation. Returns source tensors, never optimizer."""
    if not runtime.enabled(args) or runtime.phase(args) != 'joint' or resume_state is not None:
        return None
    validate_flags(args)
    runtime.require(not args.auto_resume, 'Joint resume must have an admitted checkpoint')
    binding = original.pin_file(args.sr_original_joint_source, args.sr_original_joint_source_sha256)
    payload = torch.load(binding['path'], map_location='cpu', mmap=True, weights_only=True)
    runtime.require(isinstance(payload, dict) and isinstance(payload.get('args'), dict)
        and isinstance(payload.get('trainer'), dict), 'Joint source must be a complete trained checkpoint')
    saved, state = payload['args'], payload['trainer']
    runtime.require(saved.get('sr_initialization_kind') == 'original_grn'
        and saved.get('sr_original_phase', 'adaptation') == 'adaptation'
        and saved.get('sr_freeze_backbone') == 1 and not state.get('sr_original_joint')
        and not saved.get('sr_original_joint_provenance'), 'Joint source must be original frozen condition adaptation')
    from grn.models.sr_deterministic_attention import assert_same_policy
    assert_same_policy(args, saved, 'Joint source')
    from grn.utils_t2iv.sr_v2_calibrator_runtime import assert_same_calibrator_config, original_specs, check_trainer_calibrator
    s3 = assert_same_calibrator_config(args, saved)
    check_trainer_calibrator(s3, state)
    for key in IDENTITY:
        runtime.require(getattr(args, key, None) == saved.get(key), f'Joint source identity differs: {key}')
    for key, default in FIELD_DEFAULTS.items():
        runtime.require(getattr(args, key, default) == saved.get(key, default), f'Joint source condition configuration differs: {key}')
    origin = runtime.validate_provenance(parse(saved.get('sr_original_provenance', 'null')), args)
    runtime.check_trainer_origin(origin, state)
    contract = original.load_audited_contract(getattr(args, 'sr_original_metadata', '') or original.DEFAULT_METADATA)
    config = condition_config(args)
    schema = dict(contract.original, **contract.legacy)
    if config.semantic: schema.update(contract.semantic)
    if config.coarse_weight > 0: schema.update(contract.coarse)
    schema.update(original_specs(s3, contract))
    model = state.get('gpt_fsdp')
    original.validate_state(model, schema, 'Trained original adaptation model')
    shared_digest = original.state_digest({name: model[name] for name in contract.original})
    runtime.require(shared_digest == origin['source_shared_tensor_digest'],
        'Adaptation changed original shared GRN weights; cannot claim frozen adaptation')
    legacy_digest = original.state_digest({name: model[name] for name in contract.legacy})
    branches = {name: model[name] for name in set(schema) - set(contract.original) - set(contract.legacy)}
    runtime.require(legacy_digest != origin['legacy_condition_digest']
        or original.state_digest(branches) != origin['new_branches_digest'],
        'Adaptation checkpoint has no changed condition weights')
    counters = state.get('sr_v2', {})
    update, microstep, accumulation = (counters.get('optimizer_updates_completed'),
        counters.get('microsteps_completed'), saved.get('gradient_accumulation'))
    runtime.require(type(update) is int and update > 0 and type(microstep) is int
        and type(accumulation) is int and accumulation > 0 and microstep == update * accumulation,
        'Adaptation must have positive complete optimizer-window counters')
    opt = state.get('gpt_fsdp_opt', {})
    runtime.require(isinstance(opt, dict) and isinstance(opt.get('state'), dict) and opt['state']
        and isinstance(opt.get('param_groups'), list), 'Actual adaptation optimizer evidence missing')
    names = [name for group in opt['param_groups'] for name in group['params']]
    condition_names = set(schema) - set(contract.original)
    runtime.require(len(names) == len(set(names)) and set(names) == condition_names
        and set(opt['state']) <= condition_names, 'Adaptation optimizer must cover only declared condition parameters')
    for name, entry in opt['state'].items():
        step = entry.get('step')
        runtime.require(torch.is_tensor(step) and step.numel() == 1 and torch.isfinite(step).all()
            and float(step) == update, f'Adaptation optimizer step differs: {name}')
    original.recheck_file(binding)
    hbq_binding = original.pin_file(args.vae_path, args.sr_original_hbq_sha256)
    args.sr_original_provenance = runtime.encode(origin)
    return dict(model=model, origin=origin, contract=contract, source=binding, saved_args=saved,
        hbq_source=hbq_binding, source_optimizer_updates=update,
        source_model_digest=original.state_digest(model),
        source_condition_digest=original.state_digest({name: model[name] for name in condition_names}))


@torch.no_grad()
def initialize(model, hbq, args, start):
    runtime.require(getattr(model, '_original_joint_lifecycle', None) is None, 'Repeated joint initialization forbidden')
    model._original_joint_lifecycle = 'running'
    try:
        original.validate_state(hbq.state_dict(), start['contract'].hbq, 'Actual joint HBQ')
        runtime.require(not hbq.training and all(not p.requires_grad and p.grad is None for p in hbq.parameters())
            and all(v.device.type == 'cpu' for v in hbq.state_dict().values())
            and original.state_digest(hbq.state_dict()) == start['origin']['hbq_tensor_digest'],
            'Joint HBQ must be the unchanged frozen original tokenizer on CPU')
        from grn.utils_t2iv.sr_v2_calibrator_runtime import calibrator_config, load_calibrator_state
        if calibrator_config(args).enabled:
            load_calibrator_state(model, start['model'], args, start['saved_args'])
        else:
            load_condition_state(model, start['model'])
        for name, value in model.state_dict().items(): original.exact(value, start['model'][name], name)
        original.recheck_file(start['source'])
        original.recheck_file(start['hbq_source'])
        evidence = dict(kind='original_grn_joint_weights_only', status='passed', source=start['source'],
            source_optimizer_updates=start['source_optimizer_updates'],
            source_model_digest=start['source_model_digest'], source_condition_digest=start['source_condition_digest'],
            original_initialization_sha256=hashlib.sha256(runtime.encode(start['origin']).encode()).hexdigest(),
            learning_policy=learning_policy(args), source_model_bytes_exact=True,
            original_shared_unchanged_at_transition=True, condition_weights_changed_during_adaptation=True,
            source_rehashed_after_load=True, optimizer_loaded=False, resume_performed=False)
        validate_lineage(evidence, args)
        args.sr_original_joint_provenance = runtime.encode(evidence)
        model._original_joint_lifecycle = 'complete'
        return evidence
    except BaseException:
        model._original_joint_lifecycle = 'failed_discard_model'
        raise


def trainer_lineage(args):
    if not runtime.enabled(args) or runtime.phase(args) != 'joint': return None
    return validate_lineage(parse(getattr(args, 'sr_original_joint_provenance', 'null')), args)


def admit_resume(args, saved_args, state):
    runtime.require(runtime.phase(args) == saved_args.get('sr_original_phase', 'adaptation'),
        'Exact resume cannot change adaptation/joint phase')
    if runtime.phase(args) != 'joint':
        runtime.require(not state.get('sr_original_joint') and not saved_args.get('sr_original_joint_provenance'),
            'Adaptation checkpoint carries unexpected joint lineage')
        return
    for key in FIELDS[:2]:
        runtime.require(getattr(args, key, '') == saved_args.get(key), 'Joint resume changes adaptation source')
    value = validate_lineage(parse(saved_args.get('sr_original_joint_provenance', 'null')), args)
    check_trainer(value, state)
    requested = getattr(args, 'sr_original_joint_provenance', '')
    runtime.require(not requested or requested == runtime.encode(value), 'Requested joint lineage differs')
    args.sr_original_joint_provenance = runtime.encode(value)


def check_trainer(actual, state):
    saved = state.get('sr_original_joint')
    if actual is None and saved is None: return
    runtime.require(actual is not None and saved is not None and runtime.encode(actual) == runtime.encode(saved),
        'Actual trainer joint lineage differs from checkpoint')


def write_audit(directory, evidence, optimizer, *, args, model, rank, policy):
    validate_lineage(evidence, args)
    runtime.require(not optimizer.state and args.sr_freeze_backbone == 0 and not policy['frozen_names'],
        'Joint phase must open GRN with a new empty optimizer')
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    runtime.require(expected and set(actual) == expected and len(actual) == len(set(actual)),
        'Joint optimizer coverage differs')
    record = dict(kind='sr_v2_original_joint_initialization', status='passed', rank=int(rank),
        pid=os.getpid(), timestamp_utc=datetime.now(timezone.utc).isoformat(), transition=evidence,
        optimizer_states_at_start=0, optimizer_updates_at_start=0, previous_optimizer_not_loaded=True,
        trainability_policy=policy, optimizer_parameter_coverage_exact=True)
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path / f'original_joint_init_rank{rank}.json').open('x') as handle:
        handle.write(runtime.encode(record) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    return record
