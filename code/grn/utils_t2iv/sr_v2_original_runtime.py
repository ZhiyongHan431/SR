"""Explicit original-GRN condition-adaptation startup and exact resume provenance.

Legacy B/S2 initialization is unchanged. This stage freezes original parameters;
opening the backbone is a separate weights-only continuation, not an exact resume.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

import torch

from grn.utils_t2iv import sr_v2_original_init as original
from grn.utils_t2iv.sr_v2_condition_runtime import condition_config
from grn.utils_t2iv.sr_v2_qualified_data import parse as strict_json


RAW_FIELDS = ('sr_original_path', 'sr_original_sha256', 'sr_original_hbq_sha256',
              'sr_original_metadata', 'sr_original_provenance',
              'sr_original_joint_source', 'sr_original_joint_source_sha256',
              'sr_original_joint_provenance')
EVENTS = ['common_initialization', 'original_shared_load', 'hbq_condition_restore',
          'fresh_lr_projection_and_position', 's2_branch_materialization', 'final_byte_and_schema_audit']


def require(value, message):
    if not value:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def enabled(args):
    kind = getattr(args, 'sr_initialization_kind', 'legacy')
    require(kind in ('legacy', 'original_grn'), 'Unknown SR initialization kind')
    return kind == 'original_grn'


def phase(args):
    value = getattr(args, 'sr_original_phase', 'adaptation')
    require(value in ('adaptation', 'joint'), 'Unknown original training phase')
    require(enabled(args) or value == 'adaptation', 'Joint phase requires original_grn initialization')
    return value


def validate_original_config(args):
    """Typed startup validation; original source bytes are pinned before allocation."""
    from grn.models.sr_deterministic_attention import validate_policy
    validate_policy(args)
    phase_name = phase(args)
    if not enabled(args):
        require(not any(getattr(args, name, '') for name in RAW_FIELDS),
                'Original-specific fields require explicit original_grn mode')
        return None
    required = dict(sr_condition=1, sr_condition_mode='prefix_text', model='GRN2b',
        Ct5=4096, sr_lr_size=128, sr_hr_size=512, hbq_round=4, detail_scale_dim=64,
        semantic_scale_dim=64, num_of_label_value=2, simple_text_proj=1, use_fsq_cls_head=1,
        use_ada_layer_norm=0, add_scale_token=1, add_class_token=0, apply_spatial_patchify=0,
        video_frames=1, refine_mode='ar_discrete_GRN_bit', sr_cross_attn=0,
        sr_freeze_backbone=1 if phase_name == 'adaptation' else 0)
    for key, value in required.items():
        actual = getattr(args, key, None)
        require(type(actual) is type(value) and actual == value,
                f'Original condition-adaptation configuration differs: {key}')
    require(not getattr(args, 'use_slow_attn', False) and not getattr(args, 'use_fsdp_model_ema', False),
            'Original adaptation currently supports fast attention without EMA')
    require(str(getattr(args, 'model_init_device', '')) == 'cpu', 'Original initialization must first occur on CPU')
    require(not getattr(args, 'rush_resume', '') and not getattr(args, 'sr_condition_init_path', '')
            and not getattr(args, 'sr_v2_condition_promote', 0),
            'Original initialization cannot mix trained warm-start, shared-condition bundle or B promotion')
    require(getattr(args, 'checkpoint_type', '') == 'torch', 'Original resume requires torch checkpoints')
    require(type(getattr(args, 'auto_resume', None)) is bool, 'Original mode requires explicit boolean auto_resume')
    for key in ('sr_v2_runtime_audit_dir', 'sr_v2_audit_dir'):
        require(getattr(args, key, '') and Path(getattr(args, key)).is_absolute(), 'Explicit original audit directories required')
    require(getattr(args, 'sr_original_sha256', '') == original.ORIGINAL_SHA
            and getattr(args, 'sr_original_hbq_sha256', '') == original.HBQ_SHA,
            'Only the audited original GRN/HBQ assets are accepted')
    require(Path(getattr(args, 'sr_original_path', '')).is_absolute()
            and Path(getattr(args, 'vae_path', '')).is_absolute(), 'Explicit absolute original/HBQ paths required')
    metadata = getattr(args, 'sr_original_metadata', '') or str(original.DEFAULT_METADATA)
    require(Path(metadata).is_absolute(), 'Original metadata path must be absolute')
    config = condition_config(args)
    from grn.utils_t2iv.sr_v2_calibrator_runtime import calibrator_config
    s3 = calibrator_config(args)
    require(not getattr(args, 'sr_v2_calibrator_promote', 0), 'Original initialization cannot promote a trained S3 branch')
    request = original.OriginalRequest('original_grn', semantic=config.semantic,
        coarse=config.coarse_weight > 0, semantic_hidden=config.semantic_hidden,
        legacy_seed=getattr(args, 'sr_original_legacy_seed', 3407), condition_seed=config.seed,
        calibrator=s3 if s3.enabled else None)
    request.validate()
    if args.auto_resume:
        from grn.utils_t2iv.sr_v2_condition_audit import validate_audit_flags
        validate_audit_flags(args)
        require(getattr(args, 'sr_v2_resume_audit', 0) == 1, 'Original resume must use an exact pinned audited checkpoint')
    else:
        require(not getattr(args, 'sr_v2_resume_audit', 0) and not getattr(args, 'sr_original_provenance', ''),
                'Fresh original initialization cannot inherit resume or origin evidence')
    from grn.utils_t2iv.sr_v2_original_joint import validate_flags
    validate_flags(args)
    return request


def validate_provenance(value, args):
    """Validate the explicit producer schema; not a substitute for checkpoint bytes."""
    require(isinstance(value, dict) and value.get('kind') == 'original_grn_initialization'
            and value.get('status') == 'passed', 'Completed original initialization provenance required')
    for key in ('source_shared_tensors_byte_exact', 'hbq_copied_tensors_byte_exact',
                'shortcut_channel_repeat_exact', 'source_files_rehashed_after_initialization'):
        require(value.get(key) is True, f'Original provenance missing actual check: {key}')
    for key in ('optimizer_loaded', 'resume_performed', 'trained_B_state_used'):
        require(value.get(key) is False, 'Original origin cannot claim B/resume/optimizer inheritance')
    from grn.utils_t2iv.sr_v2_calibrator_runtime import calibrator_config, original_specs
    s3 = calibrator_config(args)
    events = EVENTS[:-1] + ['s3_branch_materialization', EVENTS[-1]] if s3.enabled else EVENTS
    require(value.get('s3_calibrator') == (s3.metadata() if s3.enabled else None),
            'Original S3 metadata provenance differs')
    require(value.get('events') == events and value.get('audit_metadata_sha256') == original.METADATA_SHA,
            'Original initialization sequence/metadata differs')
    for key, sha in (('source', original.ORIGINAL_SHA), ('hbq_source', original.HBQ_SHA)):
        binding = value.get(key)
        require(isinstance(binding, dict) and binding.get('sha256') == sha
                and isinstance(binding.get('path'), str) and Path(binding['path']).is_absolute()
                and type(binding.get('bytes')) is int and binding['bytes'] > 0,
                'Invalid original asset provenance')
    require(value['source']['path'] == str(Path(args.sr_original_path).resolve())
            and value['hbq_source']['path'] == str(Path(args.vae_path).resolve()), 'Original source path provenance differs')
    config = condition_config(args)
    contract = original.load_audited_contract(getattr(args, 'sr_original_metadata', '') or original.DEFAULT_METADATA)
    branches = dict(contract.semantic if config.semantic else {})
    if config.coarse_weight > 0: branches.update(contract.coarse)
    branches.update(original_specs(s3, contract))
    new_keys = set(contract.legacy) | set(branches)
    require(value.get('original_tensor_count') == len(contract.original)
            and value.get('hbq_copy_count') == len(contract.hbq_copies)
            and value.get('final_tensor_count') == len(contract.original) + len(new_keys)
            and value.get('permitted_original_missing_keys') == sorted(new_keys)
            and value.get('branch_keys') == sorted(branches)
            and value.get('hbq_removed_keys') == list(contract.hbq_removed)
            and value.get('shortcut_new_keys') == list(contract.shortcuts),
            'Original provenance schema/topology differs')
    require(value.get('legacy_seed') == args.sr_original_legacy_seed and value.get('condition_seed') == config.seed,
            'Original initialization seed provenance differs')
    for key in ('source_shared_tensor_digest', 'hbq_tensor_digest', 'legacy_condition_digest', 'new_branches_digest'):
        require(isinstance(value.get(key), str) and re.fullmatch('[0-9a-f]{64}', value[key]),
                'Invalid original tensor digest')
    return value


def admit_original_resume(args, saved_args, trainer_state):
    """Called only after the exact checkpoint load, before model construction."""
    require(isinstance(saved_args, dict) and isinstance(trainer_state, dict), 'Checkpoint header must be dictionaries')
    from grn.models.sr_deterministic_attention import assert_same_policy
    assert_same_policy(args, saved_args, 'Original resume')
    from grn.utils_t2iv.sr_v2_calibrator_runtime import assert_same_calibrator_config, check_trainer_calibrator
    s3 = assert_same_calibrator_config(args, saved_args)
    check_trainer_calibrator(s3, trainer_state)
    saved_kind = saved_args.get('sr_initialization_kind', 'legacy')
    require(saved_kind in ('legacy', 'original_grn'), 'Unknown checkpoint SR initialization kind')
    current, saved = enabled(args), saved_kind == 'original_grn'
    if not current and not saved:
        require('sr_original_init' not in trainer_state and 'sr_original_joint' not in trainer_state,
                'Legacy checkpoint unexpectedly carries original provenance')
        return
    require(current and saved, 'Resume cannot switch original/legacy initialization lifecycles')
    require(args.auto_resume and getattr(args, 'sr_v2_resume_audit', 0) == 1,
            'Original resume must be explicit and audited')
    for key in ('sr_original_path', 'sr_original_sha256', 'sr_original_hbq_sha256',
                'sr_original_legacy_seed', 'sr_freeze_backbone', 'vae_path',
                'model', 'Ct5', 'sr_lr_size', 'sr_hr_size', 'hbq_round', 'detail_scale_dim',
                'semantic_scale_dim', 'num_of_label_value', 'simple_text_proj', 'use_fsq_cls_head',
                'use_ada_layer_norm', 'add_scale_token', 'add_class_token', 'apply_spatial_patchify',
                'video_frames', 'refine_mode', 'sr_cross_attn', 'sr_condition', 'sr_condition_mode'):
        require(getattr(args, key, None) == saved_args.get(key), f'Original resume changes {key}')
    encoded = saved_args.get('sr_original_provenance', '')
    require(isinstance(encoded, str) and encoded, 'Saved original provenance missing')
    origin = validate_provenance(strict_json(encoded), args)
    require(encode(trainer_state.get('sr_original_init')) == encode(origin),
            'Checkpoint trainer/Args original provenance differs')
    requested = getattr(args, 'sr_original_provenance', '')
    require(not requested or requested == encode(origin), 'Requested original provenance differs')
    args.sr_original_provenance = encode(origin)
    from grn.utils_t2iv.sr_v2_original_joint import admit_resume
    admit_resume(args, saved_args, trainer_state)
    args._sr_original_resume_admitted = True


def prepare_fresh_request(args, resume_state):
    request = validate_original_config(args)
    if request is None:
        return None
    if resume_state is not None:
        require(args.auto_resume and getattr(args, '_sr_original_resume_admitted', False),
                'Original resume must admit actual checkpoint header first')
        require(encode(resume_state.get('sr_original_init')) == args.sr_original_provenance,
                'Admitted original resume state changed')
        # HBQ is reconstructed separately, not stored in the trained GRN state.
        args._sr_original_resume_hbq_pin = original.pin_file(args.vae_path, args.sr_original_hbq_sha256)
        return None  # exact trained GRN restoration; never raw-original/condition recopy
    require(not args.auto_resume, 'Requested original resume did not load a checkpoint')
    if phase(args) == 'joint':
        return None  # separate trained-weight source, no raw original initialization
    metadata = getattr(args, 'sr_original_metadata', '') or original.DEFAULT_METADATA
    original.load_audited_contract(metadata)
    args._sr_original_preallocation_pins = dict(
        source=original.pin_file(args.sr_original_path, args.sr_original_sha256),
        hbq=original.pin_file(args.vae_path, args.sr_original_hbq_sha256))
    return request


def initialize_fresh(model, hbq, args, request, common_initialize):
    pins = args._sr_original_preallocation_pins
    original.recheck_file(pins['source'])
    original.recheck_file(pins['hbq'])
    evidence = original.initialize_original(model, hbq, request,
        source=pins['source']['path'], source_sha256=args.sr_original_sha256,
        hbq_checkpoint=pins['hbq']['path'], hbq_sha256=args.sr_original_hbq_sha256,
        common_initialize=common_initialize,
        audit_metadata=getattr(args, 'sr_original_metadata', '') or original.DEFAULT_METADATA)
    validate_provenance(evidence, args)
    args.sr_original_provenance = encode(evidence)
    return evidence


def trainer_origin(args):
    if not enabled(args):
        return None
    raw = getattr(args, 'sr_original_provenance', '')
    require(isinstance(raw, str) and raw, 'Actual original initialization/resume provenance absent')
    return validate_provenance(strict_json(raw), args)


def verify_resume_hbq(hbq, args):
    """Check the actual separately loaded CPU tokenizer before any device move."""
    require(enabled(args) and getattr(args, '_sr_original_resume_admitted', False),
            'Original HBQ resume check needs admitted checkpoint provenance')
    binding = getattr(args, '_sr_original_resume_hbq_pin', None)
    require(binding is not None, 'HBQ source was not pinned before resume allocation')
    origin = trainer_origin(args)
    contract = original.load_audited_contract(getattr(args, 'sr_original_metadata', '') or original.DEFAULT_METADATA)
    original.validate_state(hbq.state_dict(), contract.hbq, 'Resumed actual HBQ')
    require(all(value.device.type == 'cpu' for value in hbq.state_dict().values()),
            'Resumed HBQ must be audited before CPU-to-device transfer')
    require(not hbq.training and all(not p.requires_grad and p.grad is None for p in hbq.parameters()),
            'Resumed HBQ must remain frozen/eval/gradient-free')
    actual = original.state_digest(hbq.state_dict())
    require(actual == origin['hbq_tensor_digest'], 'Resumed actual HBQ bytes differ from original provenance')
    original.recheck_file(binding)
    args._sr_original_resume_hbq_evidence = dict(source=binding,
        actual_hbq_tensor_digest=actual, actual_hbq_matches_original=True,
        actual_hbq_frozen_eval=True, source_rehashed_after_load=True)
    return args._sr_original_resume_hbq_evidence


def original_resume_evidence(trainer, state, args):
    if not enabled(args):
        check_trainer_origin(getattr(trainer, 'sr_original_init', None), state)
        return {}
    origin = trainer_origin(args)
    check_trainer_origin(origin, state)
    require(encode(getattr(trainer, 'sr_original_init', None)) == encode(origin),
            'Restored Trainer lost original initialization lineage')
    hbq = getattr(args, '_sr_original_resume_hbq_evidence', None)
    require(isinstance(hbq, dict) and hbq.get('actual_hbq_matches_original') is True,
            'Actual original HBQ resume evidence missing')
    require(not trainer.vae_local.training
            and all(not p.requires_grad and p.grad is None for p in trainer.vae_local.parameters())
            and original.state_digest(trainer.vae_local.state_dict()) == origin['hbq_tensor_digest'],
            'Actual tokenizer changed after resume device transfer')
    from grn.utils_t2iv.sr_v2_original_joint import trainer_lineage, check_trainer
    joint = trainer_lineage(args)
    check_trainer(getattr(trainer, 'sr_original_joint', None), state)
    require(encode(getattr(trainer, 'sr_original_joint', None)) == encode(joint),
            'Restored Trainer joint lineage differs from actual args')
    extra = {} if joint is None else dict(original_joint_provenance_exact=True,
        original_joint_provenance_sha256=hashlib.sha256(encode(joint).encode()).hexdigest())
    return dict(original_initialization_provenance_exact=True,
        original_initialization_sha256=hashlib.sha256(encode(origin).encode()).hexdigest(),
        original_hbq_restore=hbq, **extra)


def check_trainer_origin(actual, state):
    saved = state.get('sr_original_init')
    if actual is None and saved is None:
        return
    require(actual is not None and saved is not None and encode(actual) == encode(saved),
            'Actual trainer original origin differs from saved checkpoint')


def write_original_initialization_audit(directory, evidence, optimizer, *, args, model, rank, policy):
    validate_provenance(evidence, args)
    require(not optimizer.state, 'Original adaptation must start with an empty optimizer')
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    actual = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
    require(expected and len(actual) == len(set(actual)) and set(actual) == expected,
            'Original optimizer parameter coverage differs')
    require(args.sr_freeze_backbone == 1 and policy['frozen_names'] and policy['trainable_numel'] > 0,
            'Original adaptation must freeze shared GRN parameters')
    record = dict(kind='sr_v2_original_condition_adaptation_initialization', rank=int(rank),
        status='passed', pid=os.getpid(), timestamp_utc=datetime.now(timezone.utc).isoformat(),
        origin=evidence, origin_sha256=hashlib.sha256(encode(evidence).encode()).hexdigest(),
        optimizer_states_at_start=0, optimizer_updates_at_start=0,
        optimizer_parameter_coverage_exact=True, trainability_policy=policy,
        attention_execution=getattr(args, '_sr_attention_execution', None))
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f'original_init_rank{rank}.json').open('x') as handle:
        json.dump(record, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    return record
