"""S3 configuration, declared topology, lifecycle admission and exact state loading.

These helpers do not select a model or certify quality, FSDP or a GPU run.
Source/data identity and original-pretraining lineage have separate checks.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import SimpleNamespace

import torch


PREFIX = 'sr_round_calibrator.'
CROSS_PREFIX = 'sr_cross_attn.'
CROSS_FIELDS = ('sr_cross_attn', 'sr_cross_attn_layers', 'sr_cross_attn_dim',
                'sr_cross_attn_heads', 'sr_cross_attn_window',
                'sr_cross_attn_global_layers', 'sr_cross_attn_gate_init')
FIELD_DEFAULTS = {
    'sr_v2_calibrator_mode': 'off',
    'sr_v2_calibrator_width': 128,
    'sr_v2_calibrator_cap': 2.,
    'sr_v2_calibrator_seed': 20260917,
    'sr_v2_calibrator_precision': 'native_fp32',
}


@dataclass(frozen=True)
class CalibratorConfig:
    mode: str
    width: int
    cap: float
    seed: int
    precision: str

    @property
    def enabled(self):
        return self.mode != 'off'

    def metadata(self):
        return dict(zip(FIELD_DEFAULTS, (self.mode, self.width, self.cap, self.seed, self.precision)))


def _value(args, name, default):
    return args.get(name, default) if isinstance(args, Mapping) else getattr(args, name, default)


def calibrator_config(args):
    mode, width, cap, seed, precision = (
        _value(args, name, default) for name, default in FIELD_DEFAULTS.items())
    if type(mode) is not str or mode not in ('off', 'plain', 'shared_round', 'per_round'):
        raise ValueError('Invalid S3 calibration mode')
    if type(width) is not int or width < 1:
        raise ValueError('S3 width must be a positive integer')
    if (isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap)
            or cap <= 0 or cap > torch.finfo(torch.float32).max
            or float(torch.tensor(cap, dtype=torch.float32)) <= 0):
        raise ValueError('S3 cap must be positive and representable in FP32')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('S3 seed must be an integer in [0, 2**63)')
    if type(precision) is not str or precision != 'native_fp32':
        raise ValueError('Only explicit native_fp32 S3 precision is supported')
    config = CalibratorConfig(mode, width, float(cap), seed, precision)
    if config.enabled:
        if (_value(args, 'sr_condition', 0) != 1
                or _value(args, 'sr_condition_mode', '') != 'prefix_text'
                or tuple(_value(args, key, default) for key, default in
                    [('sr_lr_size', 128), ('sr_hr_size', 512), ('hbq_round', 4),
                     ('detail_scale_dim', 64), ('detail_num_lvl', 2)]) != (128, 512, 4, 64, 2)
                or (_value(args, 'sr_condition_drop_rate', 0.) != 0.
                    and _value(args, 'sr_v2_condition_dropout_pilot', 0) != 1)):
            raise ValueError('S3 requires fixed prefix_text RGB128/512, M4 x 64 binary HBQ; nonzero dropout requires explicit pilot authorization')
    return config


def saved_calibrator_config(saved_args):
    if not isinstance(saved_args, dict):
        raise ValueError('S3 saved args must be a dictionary')
    present = set(FIELD_DEFAULTS) & set(saved_args)
    if present and present != set(FIELD_DEFAULTS):
        raise ValueError('S3 checkpoint requires complete architecture metadata')
    return calibrator_config(saved_args)


def restore_calibrator_architecture(args, saved_args):
    """Validate first; legacy absence explicitly restores off, never caller defaults."""
    config = saved_calibrator_config(saved_args)
    candidate = SimpleNamespace(**vars(args))
    for name, value in config.metadata().items():
        setattr(candidate, name, value)
    calibrator_config(candidate)
    for name, value in config.metadata().items():
        setattr(args, name, value)
    return config


def assert_same_calibrator_config(args, saved_args):
    requested, saved = calibrator_config(args), saved_calibrator_config(saved_args)
    if requested != saved:
        raise ValueError('S3 restore changes mode/width/cap/seed/precision metadata')
    return requested


def audit_calibrator_module(model, config):
    module = getattr(model, 'sr_round_calibrator', None)
    if not config.enabled:
        if module is not None:
            raise ValueError('S3 off metadata cannot hide a materialized branch')
        return
    from grn.models.sr_v2_calibrator import BoundedRoundCalibrator
    if (not isinstance(module, BoundedRoundCalibrator)
            or (module.mode, module.width, module.max_log_odds, module.hr_dim,
                module.lr_dim, module.rounds, module.bits_per_round, module.initial_gate)
                != (config.mode, config.width, config.cap, model.C, 64, 4, 64, .5)
            or getattr(module, '_sr_initialization_seed', None) != config.seed
            or any(p.dtype != torch.float32 for p in module.parameters())):
        raise ValueError('Actual S3 module differs from complete configuration')
    actual = {PREFIX + name: tuple(value.shape) for name, value in module.state_dict().items()}
    if actual != tensor_shapes(config, model.C):
        raise ValueError('Actual S3 tensors differ from declared topology')


def materialize_calibrator(model, args):
    config = calibrator_config(args)
    if config.enabled:
        model.configure_sr_v2_calibrator(mode=config.mode, width=config.width,
            max_log_odds=config.cap, seed=config.seed)
        model.sr_round_calibrator._sr_initialization_seed = config.seed
    audit_calibrator_module(model, config)
    if config.enabled:
        module = model.sr_round_calibrator
        if (any(not torch.isfinite(p).all() for p in module.parameters())
                or any(torch.count_nonzero(p).item()
                    for head in (*module.output_heads, *module.gate_heads) for p in head.parameters())):
            raise ValueError('Fresh S3 materialization must be finite with zero output and gate logits')
    return config


def _cross_value(values, name):
    defaults = {
        'sr_cross_attn': 0, 'sr_cross_attn_layers': '[4, 12, 20, 28]',
        'sr_cross_attn_dim': 512, 'sr_cross_attn_heads': 8,
        'sr_cross_attn_window': 7, 'sr_cross_attn_global_layers': '',
        'sr_cross_attn_gate_init': 1e-3,
    }
    default = defaults[name]
    return values.get(name, default) if isinstance(values, Mapping) else getattr(values, name, default)


def cross_attn_config(values):
    result = {name: _cross_value(values, name) for name in CROSS_FIELDS}
    for name in ('sr_cross_attn_layers', 'sr_cross_attn_global_layers'):
        value = result[name]
        if isinstance(value, str):
            # Normalize JSON spelling while preserving an empty global-layer sentinel.
            import json
            result[name] = tuple(json.loads(value)) if value else ()
        elif isinstance(value, (list, tuple)):
            result[name] = tuple(value)
        else:
            raise ValueError(f'Invalid {name} in Cross-Attention metadata')
    return result


def assert_same_cross_attn_config(args, saved_args):
    requested, saved = cross_attn_config(args), cross_attn_config(saved_args)
    if requested != saved:
        raise ValueError('Cross-Attention restore changes topology metadata')
    return requested


@torch.no_grad()
def initialize_cross_attn_branch(model, args):
    """Initialize each newly promoted layer from a layer-stable private RNG."""
    modules = getattr(model, 'sr_cross_attn', None)
    if not modules or not getattr(args, 'sr_v2_cross_attn_promote', 0):
        return None
    seed = getattr(args, 'sr_v2_cross_attn_init_seed', 20260919)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Cross-Attention initialization seed must be a nonnegative int64')
    reference = next(modules.parameters())
    devices = [reference.device.index] if reference.device.type == 'cuda' else []
    for layer_name, module in modules.items():
        layer_seed = (seed + int(layer_name)) % (2**63)
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(layer_seed)
            if reference.device.type == 'cuda':
                torch.cuda.manual_seed(layer_seed)
            for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
                projection = getattr(module, name)
                torch.nn.init.trunc_normal_(projection.weight, std=0.02)
            module.gate.fill_(float(getattr(args, 'sr_cross_attn_gate_init', 1e-3)))
    branch = {name: value for name, value in model.state_dict().items()
              if name.startswith(CROSS_PREFIX)}
    from grn.utils_t2iv.sr_v2_condition_audit import tensor_state_digest
    return dict(seed=seed, layers=sorted(modules.keys()),
                initialization_sha256=tensor_state_digest(branch))


def load_calibrator_state(model, state, args, saved_args, *, promote=False,
                          cross_promote=False):
    """Load complete state, permitting only an explicitly fresh whole S3 branch.

    All metadata, membership, dtype, shapes and finite values are checked before
    copying. Promotion merges the verified initialized branch and then uses
    strict=True; no broad missing-key or partial-S3 fallback is available.
    This function never accepts or restores an optimizer or training counters.
    """
    if (type(promote) is not bool or type(cross_promote) is not bool
            or not isinstance(state, dict)):
        raise ValueError('S3 load requires explicit bool promotion flags and tensor dictionary')
    config, saved = calibrator_config(args), saved_calibrator_config(saved_args)
    audit_calibrator_module(model, config)
    from grn.utils_t2iv.sr_v2_condition_runtime import check_condition_source_config
    condition_args = SimpleNamespace(**args) if isinstance(args, Mapping) else args
    check_condition_source_config(condition_args, saved_args, promote_b=False)
    # Equal tensor shapes do not prove the same HBQ factorization or crop geometry.
    fixed = dict(sr_condition=1, sr_condition_mode='prefix_text', sr_lr_size=128,
        sr_hr_size=512, hbq_round=4, detail_scale_dim=64, detail_num_lvl=2)
    saved_drop = saved_args.get('sr_condition_drop_rate', 0.)
    saved_dropout_pilot = saved_args.get('sr_v2_condition_dropout_pilot', 0)
    if (any(saved_args.get(key, default) != default for key, default in fixed.items())
            or (saved_drop != 0. and saved_dropout_pilot != 1)):
        raise ValueError('S3 source must preserve fixed prefix_text RGB128/512 and M4 x 64 geometry')
    from grn.models.sr_deterministic_attention import assert_same_policy
    assert_same_policy(args, saved_args, 'S3 state load')
    expected = model.state_dict()
    new = {name for name in expected if name.startswith(PREFIX)}
    new_cross = {name for name in expected if name.startswith(CROSS_PREFIX)}
    if cross_promote:
        same_topology = (cross_attn_config(args) == cross_attn_config(saved_args)
                         and _cross_value(args, 'sr_cross_attn') == 1
                         and _cross_value(saved_args, 'sr_cross_attn') == 1)
        if same_topology:
            # A promoted S3+Cross checkpoint may continue with the same full
            # topology.  Only source-off -> target-on is a fresh promotion.
            cross_promote = False
        elif (promote or not new_cross or _cross_value(args, 'sr_cross_attn') != 1
              or _cross_value(saved_args, 'sr_cross_attn') != 0
              or any(name.startswith(CROSS_PREFIX) for name in state)):
            raise ValueError('Cross-Attention promotion requires source off and requested branch on')
    if not cross_promote and cross_attn_config(args) != cross_attn_config(saved_args):
        raise ValueError('Cross-Attention topology differs without explicit promotion')
    if promote:
        if (saved.enabled or not config.enabled or not new
                or any(name.startswith(PREFIX) for name in state)
                or saved_args.get('sr_condition') != 1
                or saved_args.get('sr_condition_mode') != 'prefix_text'
                or 'sr_prefix_pos' not in state
                or not any(name.startswith('sr_condition_encoder.') for name in state)):
            raise ValueError('S3 promotion requires complete trained condition weights with S3 off')
        permitted_missing = new | (new_cross if cross_promote else set())
    else:
        assert_same_calibrator_config(args, saved_args)
        permitted_missing = new_cross if cross_promote else set()
    if set(expected) - set(state) != permitted_missing or set(state) - set(expected):
        raise ValueError('S3 checkpoint membership differs from the exact complete schema')
    for name, value in state.items():
        if (not torch.is_tensor(value) or value.shape != expected[name].shape
                or value.dtype != expected[name].dtype
                or value.is_floating_point() and not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid S3 checkpoint tensor: ' + name)
    if promote:
        for name in new:
            if not bool(torch.isfinite(expected[name]).all()):
                raise ValueError('Nonfinite freshly initialized S3 tensor: ' + name)
        module = model.sr_round_calibrator
        if any(torch.count_nonzero(p).item()
               for head in (*module.output_heads, *module.gate_heads) for p in head.parameters()):
            raise ValueError('Fresh S3 promotion requires zero output and gate logits (gate=.5)')
    if cross_promote:
        for name in new_cross:
            if not bool(torch.isfinite(expected[name]).all()):
                raise ValueError('Nonfinite freshly initialized Cross-Attention tensor: ' + name)
    merged = {**expected, **state} if (promote or cross_promote) else state
    model.load_state_dict(merged, strict=True)
    actual = model.state_dict()
    for name, value in state.items():
        if not torch.equal(actual[name].detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                           value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)):
            raise RuntimeError('Loaded S3 source tensor bytes differ: ' + name)
    return dict(kind='complete_condition_to_S3_weights_only' if promote else ('S3_to_S3_cross_attn_promotion' if cross_promote else 'exact_S3_state_restore'),
        loaded_source_tensors=len(state), fresh_branch_tensors=(len(new) if promote else 0) + (len(new_cross) if cross_promote else 0),
        cross_attn_promotion=bool(cross_promote),
        complete_metadata=config.metadata(), all_source_tensor_bytes_exact=True,
        optimizer_loaded=False, counters_restored=False, training_performed=False)


def tensor_shapes(config, hr_dim):
    """Independent declared S3 schema, not inferred from the checkpoint's keys."""
    if not config.enabled:
        return {}
    if type(hr_dim) is not int or hr_dim < 1:
        raise ValueError('S3 HR dimension must be a positive integer')
    width = config.width
    shapes = {}
    for name, dim in (('hr_proj', hr_dim), ('lr_proj', 64), ('time_proj', 2)):
        shapes[name + '.weight'] = (width, dim)
        shapes[name + '.bias'] = (width,)
    if config.mode != 'plain':
        shapes['round_embedding.weight'] = (4, width)
    count = 4 if config.mode == 'per_round' else 1
    for index in range(count):
        output_dim, gate_dim = (256, 4) if config.mode == 'plain' else (64, 1)
        for name, dim in (('output_heads', output_dim), ('gate_heads', gate_dim)):
            shapes[f'{name}.{index}.weight'] = (dim, width)
            shapes[f'{name}.{index}.bias'] = (dim,)
    return {PREFIX + name: shape for name, shape in shapes.items()}


def original_specs(config, contract):
    """Extend the audited parent schema only with explicitly declared new tensors."""
    from grn.utils_t2iv.sr_v2_original_init import TensorSpec
    hr_dim = contract.legacy['sr_prefix_pos'].shape[-1]
    return {name: TensorSpec(shape, 'torch.float32')
            for name, shape in tensor_shapes(config, hr_dim).items()}


def config_namespace(config):
    # Geometry is fixed for this phase; actual GRN construction checks it again.
    return SimpleNamespace(sr_condition=1, sr_condition_mode='prefix_text',
        sr_lr_size=128, sr_hr_size=512, hbq_round=4, detail_scale_dim=64,
        detail_num_lvl=2, sr_condition_drop_rate=0., **config.metadata())


def check_trainer_calibrator(config, state):
    metadata = config.metadata() if config.enabled else None
    if state.get('sr_v2_calibrator') != metadata:
        raise ValueError('Trainer S3 metadata missing or differs from complete configuration')


def validate_training_request(args, resume_state=None):
    config = calibrator_config(args)
    flag = getattr(args, 'sr_v2_calibrator_promote', 0)
    cross_flag = getattr(args, 'sr_v2_cross_attn_promote', 0)
    if type(flag) is not int or flag not in (0, 1):
        raise ValueError('S3 promotion flag must be integer 0/1')
    if type(cross_flag) is not int or cross_flag not in (0, 1):
        raise ValueError('Cross-Attention promotion flag must be integer 0/1')
    if cross_flag and (not config.enabled or not _cross_value(args, 'sr_cross_attn')
            or resume_state is not None or getattr(args, 'auto_resume', False)
            or getattr(args, 'sr_v2_calibrator_promote', 0)
            or not getattr(args, 'rush_resume', '')):
        raise ValueError('Cross-Attention promotion requires an S3 source, fresh optimizer, and explicit rush_resume')
    if flag and (not config.enabled or resume_state is not None
            or getattr(args, 'auto_resume', False)
            or getattr(args, 'sr_initialization_kind', 'legacy') != 'legacy'
            or getattr(args, 'sr_v2_condition_promote', 0)
            or not getattr(args, 'rush_resume', '')):
        raise ValueError('S3 promotion requires a separate weights-only trained source and fresh optimizer')
    if resume_state is not None:
        check_trainer_calibrator(config, resume_state)
    elif config.enabled and (getattr(args, 'sr_initialization_kind', 'legacy') == 'legacy'
            and not getattr(args, 'rush_resume', '')):
        raise ValueError('S3 fresh training requires original initialization or explicit trained source')
    if config.enabled and getattr(args, 'sr_v2_condition_promote', 0):
        raise ValueError('S2 and S3 topology changes must be separate explicit stages')
    return config


def verify_calibrator_warm_start(model, source_state, source_path, args, saved_args, *, rank, promote, cross_promote=False):
    """Actual byte verification plus the existing fresh-optimizer audit contract."""
    from pathlib import Path
    from grn.utils_t2iv.sr_v2_condition_audit import tensor_state_digest
    result = load_calibrator_state(model, source_state, args, saved_args, promote=promote, cross_promote=cross_promote)
    branch = {name: value for name, value in model.state_dict().items() if name.startswith(PREFIX)}
    cross_branch = {name: value for name, value in model.state_dict().items() if name.startswith(CROSS_PREFIX)}
    return dict(result, rank=int(rank), source_path=str(Path(source_path).resolve(strict=True)),
        source_tensors_exact=True, source_tensor_count=len(source_state),
        calibrator_promotion=promote, cross_attn_promotion=bool(cross_promote),
        calibrator_initialization_sha256=tensor_state_digest(branch),
        cross_attn_initialization_sha256=(tensor_state_digest(cross_branch) if cross_branch else None))
