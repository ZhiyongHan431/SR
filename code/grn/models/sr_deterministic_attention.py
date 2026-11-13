"""Opt-in FA2 deterministic backward; not a claim of whole-training determinism.

The default preserves the existing backend/call. Enabled runs pin FA2, including
on newer devices, and reject masked attention rather than silently using SDPA.
Only the actual two-rank continuation gate can establish full resume identity.
"""
from collections.abc import Mapping

FIELD = 'sr_deterministic_attention'


def attention_flag(args):
    value = args.get(FIELD, 0) if isinstance(args, Mapping) else getattr(args, FIELD, 0)
    if type(value) is not int or value not in (0, 1):
        raise ValueError(FIELD + ' must be integer 0/1')
    return value


def validate_policy(args):
    flag = attention_flag(args)
    if flag and (getattr(args, 'use_slow_attn', False) or getattr(args, 'use_flex_attn', False)):
        raise ValueError('Deterministic attention requires the FA2 varlen route')
    return flag


def assert_same_policy(args, saved_args, context):
    if attention_flag(args) != attention_flag(saved_args):
        raise ValueError(context + ' changes ' + FIELD)


def audit_model(model, args):
    flag = validate_policy(args)
    if not flag:
        return dict(sr_deterministic_attention=0, backend='legacy_default')
    from grn.models.basic import SelfAttention
    modules = [module for module in model.modules() if isinstance(module, SelfAttention)]
    if not modules or len(modules) != model.depth:
        raise ValueError('Actual GRN attention block coverage differs')
    if any(module.sr_deterministic_attention != 1 for module in modules):
        raise ValueError('Actual GRN attention block lost deterministic backward policy')
    return dict(sr_deterministic_attention=1, backend='flash_attn_2_varlen',
                actual_attention_blocks=len(modules), deterministic_backward_requested=True)
