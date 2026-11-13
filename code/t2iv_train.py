import gc
import json
import math
import os
import os.path as osp
import random
import sys
import time
from functools import partial
from typing import List, Optional, Tuple
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'

import numpy as np
import torch
torch._dynamo.config.cache_size_limit = 64
from torch.nn import functional as F
from torch.profiler import record_function
from torch.utils.data import DataLoader
from transformers import T5EncoderModel, T5TokenizerFast
import torch.distributed as tdist

import grn.utils_t2iv.dist as dist
from grn.dataset.build import build_joint_dataset
from grn.models.ema import get_ema_model
from grn.utils_t2iv import arg_util, misc
from grn.utils import wandb_utils
from grn.trainer import get_trainer
from grn.utils_t2iv.sr_native import load_empty_text_cache, pack_empty_text, validate_native_config, load_native_state, restore_sr_architecture
from grn.utils_t2iv.sr_condition_init import load_abc_state
from grn.models.sr_condition_modes import uses_continuous_lr, uses_empty_text
from grn.models.sr_attention_config import sr_parameter_lr_scales

def build_everything_from_args(args: arg_util.Args, saver):
    args.set_initial_seed(benchmark=True)
    from grn.utils_t2iv.sr_v2_runtime import validate_v2_config
    validate_v2_config(args)
    from grn.utils_t2iv.sr_v2_condition_runtime import condition_config
    condition_config(args)
    from grn.utils_t2iv.sr_v2_calibrator_runtime import calibrator_config
    calibrator_config(args)
    from grn.utils_t2iv.sr_v2_condition_audit import validate_audit_flags
    validate_audit_flags(args)
    from grn.utils_t2iv.sr_v2_original_runtime import validate_original_config, admit_original_resume
    validate_original_config(args)
    from grn.utils_t2iv.sr_v2_run_budget import validate as validate_run_budget
    validate_run_budget(args)
    mode = validate_native_config(args)
    from grn.utils_t2iv.sr_paired_views import validate_view_training
    validate_view_training(args, dist.get_world_size())
    from grn.utils_t2iv.sr_v2_qualified_training import validate_qualified_training, validate_qualified_resume
    validate_qualified_training(args, dist.get_world_size())
    start_ep, start_it, global_it = 0, 0, 0
    trainer_state, args_state = {}, {}
    if args.checkpoint_type == 'torch':
        from grn.utils_t2iv.save_and_load import auto_resume
        auto_resume_info, start_ep, start_it, acc_str, eval_milestone, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
        print('\n'.join(auto_resume_info))
        if trainer_state:
            validate_qualified_resume(args, args_state or {}, start_ep=start_ep, start_it=start_it)
            admit_original_resume(args, args_state or {}, trainer_state)
            from grn.utils_t2iv.sr_v2_run_budget import validate_resume as validate_budget_resume
            validate_budget_resume(args, args_state or {})
        if start_ep >= args.ep:
            if getattr(args, 'sr_v2_resume_audit', 0):
                raise RuntimeError('Audited resume must perform additional training; completed horizon is not restoration evidence')
            print(f'[vgpt] SR training finished ({acc_str}), skipping ...\n\n')
            return None
        if args.sr_condition and trainer_state:
            if (args_state or {}).get('sr_condition_mode', 'prefix') != mode:
                raise RuntimeError('Cannot resume an SR checkpoint with a different condition mode')
            from grn.utils_t2iv.sr_v2_condition_audit import validate_resume_architecture_request
            validate_resume_architecture_request(args, args_state or {})
            restore_sr_architecture(args, args_state or {})
    native_empty_text = None
    if args.sr_condition:
        args.use_vae_token_cache = 0
        args.save_vae_token_cache = 0
        if uses_empty_text(mode):
            native_empty_text = load_empty_text_cache(args.sr_native_empty_text_path, args.Ct5)
            args.sr_native_empty_text_len = int(native_empty_text.shape[0])
        print(f'SR mode: {mode}; skip live T5 loading')
        text_encoder, text_tokenizer = None, None
        args.text_tokenizer_type = 'none'
    else:
        print(f'Loading T5 from {args.t5_path}...')
        from grn.models.umt5.t5 import T5EncoderModel
        text_encoder = T5EncoderModel(
            text_len=args.tlen, # 512
            dtype=torch.bfloat16, # torch.bfloat16
            device=args.device,
            checkpoint_path=osp.join(args.t5_path, 'models_t5_umt5-xxl-enc-bf16.pth'),
            tokenizer_path=osp.join(args.t5_path, 'umt5-xxl'),
            enable_fsdp=True) # False
        text_tokenizer = text_encoder.tokenizer
        args.text_tokenizer_type = 'umt5'
    args.text_tokenizer = text_tokenizer

    # build models. Note that here gpt is the causal VAR transformer which performs next scale prediciton with text guidance
    vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim = build_model_optimizer(
        args, resume_state=trainer_state if args.sr_condition and trainer_state else None)
    
    Trainer = get_trainer(args)
    # build trainer
    trainer = Trainer(
        is_visualizer=dist.is_visualizer(), device=args.device, 
        vae_local=vae_local, gpt_wo_ddp=gpt_wo_ddp, gpt=gpt_ddp,
        zero=args.zero, vae_latent_dim=args.vae_latent_dim, gpt_opt=gpt_optim,
        reweight_loss_by_scale=args.reweight_loss_by_scale, gpt_wo_ddp_ema=gpt_wo_ddp_ema, 
        gpt_ema=gpt_ddp_ema, use_fsdp_model_ema=args.use_fsdp_model_ema, other_args=args,
    )
    
    if trainer_state:
        trainer.load_state_dict(trainer_state, strict=True, skip_vae=True)
        if getattr(args, 'sr_v2_resume_audit', 0):
            from grn.utils_t2iv.sr_v2_condition_audit import audit_distributed_resume
            audit_distributed_resume(trainer, trainer_state, args)
        global_it = int(getattr(args, 'resume_global_it', 0))
        if args_state and 'seed' in args_state:
            args.seed = int(args_state['seed'])
            args.set_initial_seed(benchmark=True)
            trainer.set_seed(args.seed)

    if getattr(args, 'sr_v2_resume_audit', 0) and not trainer_state:
        raise RuntimeError('Requested audited resume did not restore a checkpoint')
    del vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim
    dist.barrier()
    return text_tokenizer, text_encoder, trainer, start_ep, start_it, global_it, native_empty_text


def build_model_optimizer(args, resume_state=None):
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from grn.models.grn import MultipleLayers
    from grn.models.init_param import init_weights
    from grn.utils_t2iv.lr_control import filter_params
    from grn.utils_t2iv.load import build_vae_gpt, initialize_sr_weights
    from grn.utils_t2iv.sr_v2_condition_runtime import condition_config
    s2_preflight = condition_config(args)
    from grn.utils_t2iv.sr_v2_calibrator_runtime import (
        validate_training_request, materialize_calibrator, load_calibrator_state,
        verify_calibrator_warm_start, assert_same_calibrator_config,
        initialize_cross_attn_branch)
    s3_config = validate_training_request(args, resume_state)
    from grn.utils_t2iv.sr_v2_original_runtime import (
        enabled as original_enabled, prepare_fresh_request, initialize_fresh,
        write_original_initialization_audit, verify_resume_hbq)
    original_request = prepare_fresh_request(args, resume_state)
    from grn.utils_t2iv.sr_v2_original_joint import (
        prepare as prepare_joint, initialize as initialize_joint, write_audit as write_joint_audit)
    joint_request = prepare_joint(args, resume_state)
    s2_promotion_flag = getattr(args, 'sr_v2_condition_promote', 0)
    if type(s2_promotion_flag) is not int or s2_promotion_flag not in (0, 1):
        raise ValueError('S2 promotion flag must be integer 0/1')
    if s2_promotion_flag and (not s2_preflight.enabled or resume_state is not None):
        raise ValueError('S2 promotion requires new modules and a fresh optimizer, not resume')
    if (s2_preflight.enabled and resume_state is None and not args.rush_resume
            and original_request is None and joint_request is None):
        raise ValueError('S2 engineering training requires an explicit trained source checkpoint')
    v2_initialization = None
    v2_runtime_audit_dir = getattr(args, 'sr_v2_runtime_audit_dir', '')
    
    # disable builtin initialization for speed
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
    vae_local, gpt_wo_ddp = build_vae_gpt(args, device=args.model_init_device)
    from grn.models.sr_deterministic_attention import audit_model
    args._sr_attention_execution = audit_model(gpt_wo_ddp, args)
    if original_enabled(args) and resume_state is not None:
        verify_resume_hbq(vae_local, args)
    count_p = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
    num_para = count_p(gpt_wo_ddp)
    from grn.utils_t2iv.sr_v2_condition_runtime import (
        materialize_condition_modules, load_condition_state, verify_condition_warm_start)
    if original_request is not None:
        # Own the CPU common -> original -> HBQ -> fresh LR -> S2 order.
        # Neither global initialization nor branch creation may run again afterwards.
        def common_initialize(model):
            init_weights(model)
            model.special_init()
        v2_initialization = initialize_fresh(
            gpt_wo_ddp, vae_local, args, original_request, common_initialize)
        s2_config = s2_preflight
        if num_para/1000 < 20:
            gpt_wo_ddp = gpt_wo_ddp.to('cuda')
    elif joint_request is not None:
        init_weights(gpt_wo_ddp)
        gpt_wo_ddp.special_init()
        s2_config = materialize_condition_modules(gpt_wo_ddp, args)
        materialize_calibrator(gpt_wo_ddp, args)
        v2_initialization = initialize_joint(gpt_wo_ddp, vae_local, args, joint_request)
        if num_para/1000 < 20:
            gpt_wo_ddp = gpt_wo_ddp.to('cuda')
    else:
        if num_para/1000 < 20: # < 20B
            gpt_wo_ddp = gpt_wo_ddp.to('cuda')
        init_weights(gpt_wo_ddp)
        gpt_wo_ddp.special_init()
        s2_config = materialize_condition_modules(gpt_wo_ddp, args)
        materialize_calibrator(gpt_wo_ddp, args)
    if getattr(args, 'sr_v2_cross_attn_promote', 0):
        cross_init = initialize_cross_attn_branch(gpt_wo_ddp, args)
        print(f'[Cross-Attention promotion] {cross_init}', flush=True)
    if original_enabled(args):
        # CPU startup includes HBQ; Trainer does not move the frozen tokenizer.
        vae_local = vae_local.to(args.device)
    num_para = count_p(gpt_wo_ddp)
    s2_promote = getattr(args, 'sr_v2_condition_promote', 0)
    if type(s2_promote) is not int or s2_promote not in (0, 1) or (s2_promote and not s2_config.enabled):
        raise ValueError('S2 promotion must be explicit integer 0/1 and requires S2 modules')
    gpt_wo_ddp_ema = None
    
    if resume_state is not None:
        if s2_promote:
            raise ValueError('S2 promotion is not a resume operation; use a fresh run/optimizer')
        state_loader = (load_native_state if getattr(args, 'sr_condition_mode', 'prefix') == 'native_adapter'
                        else load_abc_state)
        if s2_config.enabled:
            state_loader = load_condition_state
        if s3_config.enabled:
            # Header policy and trainer metadata were admitted before construction.
            state_loader = lambda model, state: load_calibrator_state(model, state, args, vars(args))
        state_loader(gpt_wo_ddp, resume_state['gpt_fsdp'])
        if args.use_fsdp_model_ema:
            gpt_wo_ddp_ema = get_ema_model(gpt_wo_ddp)
            state_loader(gpt_wo_ddp_ema, resume_state['gpt_ema_fsdp'])
    elif args.rush_resume:
        print(f"{args.rush_resume=}")
        if '.pth' in args.rush_resume:
            cpu_d = torch.load(args.rush_resume, map_location='cpu', mmap=True)
        else:
            from grn.utils_t2iv.save_and_load import merge_ckpt
            cpu_d = merge_ckpt(args.rush_resume, osp.join(args.rush_resume, 'ouput'), save=False, use_ema_model=False, fsdp_save_flatten_model=args.fsdp_save_flatten_model)
        trained_checkpoint = False
        saved_args = {}
        if 'trainer' in cpu_d:
            state_dict = cpu_d['trainer']['gpt_fsdp']
            ema_state_dict = cpu_d['trainer'].get('gpt_ema_fsdp', state_dict)
            saved_args = cpu_d.get('args', {})
            if saved_args.get('sr_initialization_kind', 'legacy') != 'legacy':
                raise ValueError('Original adaptation checkpoints require exact resume; joint weights-only continuation is a separate stage')
            trained_checkpoint = bool(saved_args.get('sr_condition', 0)) or 'sr_prefix_pos' in state_dict or any(
                key.startswith('sr_native_adapters.') for key in state_dict)
            if args.sr_condition and trained_checkpoint:
                saved_mode = saved_args.get('sr_condition_mode', 'prefix')
                if saved_mode != getattr(args, 'sr_condition_mode', 'prefix'):
                    raise RuntimeError('Cannot warm start a trained SR checkpoint with a different condition mode')
                if saved_mode == 'spatial_control':
                    for field, default in (('sr_control_layers', '[4, 12, 20, 28]'),
                            ('sr_control_dim', 512), ('sr_control_heads', 8), ('sr_control_window', 7)):
                        saved, requested = saved_args.get(field, default), getattr(args, field, default)
                        if field == 'sr_control_layers':
                            saved = json.loads(saved) if isinstance(saved, str) else saved
                            requested = json.loads(requested) if isinstance(requested, str) else requested
                        if saved != requested:
                            raise RuntimeError(f'Trained SR checkpoint architecture mismatch: {field}')
        else:
            state_dict = cpu_d
            ema_state_dict = state_dict
        def drop_unfit_weights(state_dict):
            try:
                if 'word_embed.weight' in state_dict and (state_dict['word_embed.weight'].shape[1] != gpt_wo_ddp.word_embed.in_features):
                    print(f'[rush_resume] drop word_embed.weight')
                    del state_dict['word_embed.weight']
                if 'head.proj.weight' in state_dict and (state_dict['head.proj.weight'].shape[0] != gpt_wo_ddp.head.proj.out_features):
                    print(f'[rush_resume] drop head')
                    del state_dict['head.proj.weight']
                    del state_dict['head.proj.bias']
            except Exception as e:
                print(e)
                for key in ['word_embed.weight', 'head.proj.weight', 'head.proj.bias']:
                    if key in state_dict:
                        del state_dict[key]
                        print(f'[rush_resume] drop {key}')

            if 'text_proj_for_sos.ca.mat_kv.weight' in state_dict and \
                (state_dict['text_proj_for_sos.ca.mat_kv.weight'].shape != gpt_wo_ddp.text_proj_for_sos.ca.mat_kv.weight.shape):
                print(f'[rush_resume] drop cfg_uncond')
                del state_dict['cfg_uncond']
                for key in list(state_dict.keys()):
                    if 'text' in key:
                        del state_dict[key]
            return state_dict
        def load_warm_start(model, warm_start_state, label):
            if s3_config.enabled:
                if not trained_checkpoint:
                    raise ValueError('S3 engineering warm start requires a complete trained B/S2/S3 checkpoint')
                return verify_calibrator_warm_start(model, warm_start_state, args.rush_resume,
                    args, saved_args, rank=dist.get_rank(),
                    promote=bool(getattr(args, 'sr_v2_calibrator_promote', 0)),
                    cross_promote=bool(getattr(args, 'sr_v2_cross_attn_promote', 0)))
            assert_same_calibrator_config(args, saved_args)
            if s2_config.enabled:
                if not trained_checkpoint:
                    raise RuntimeError('This S2 engineering loader requires a trained B/S2 source; original T2I adaptation is a separate stage')
                from grn.utils_t2iv.sr_v2_condition_runtime import check_condition_source_config
                check_condition_source_config(args, saved_args, promote_b=bool(s2_promote))
                return load_condition_state(model, warm_start_state, promote_b=bool(s2_promote))
            if args.sr_condition and getattr(args, 'sr_condition_mode', 'prefix') == 'native_adapter':
                # Native T2I warm starts must retain every original tensor, including its shape.
                return load_native_state(model, warm_start_state, warm_start=True)
            if args.sr_condition:
                if trained_checkpoint:
                    return load_abc_state(model, warm_start_state)
                return initialize_sr_weights(model, vae_local, args, warm_start_state)
            warm_start_state = drop_unfit_weights(warm_start_state)
            load_result = model.load_state_dict(warm_start_state, strict=False)
            print(load_result)
            return load_result

        warm_start_result = load_warm_start(gpt_wo_ddp, state_dict, 'rush_resume')
        if v2_runtime_audit_dir:
            if not trained_checkpoint:
                raise RuntimeError('V2 S1 requires a trained B checkpoint warm start')
            from grn.utils_t2iv.sr_v2_runtime import verify_warm_start
            if s3_config.enabled:
                v2_initialization = warm_start_result
            elif s2_config.enabled or getattr(args, 'sr_v2_condition_audit', 0):
                v2_initialization = verify_condition_warm_start(gpt_wo_ddp, state_dict,
                    args.rush_resume, rank=dist.get_rank(), promote_b=bool(s2_promote))
            else:
                v2_initialization = verify_warm_start(
                    gpt_wo_ddp, state_dict, args.rush_resume, rank=dist.get_rank())
        if args.use_fsdp_model_ema:
            gpt_wo_ddp_ema = get_ema_model(gpt_wo_ddp)
            if s2_config.enabled or s3_config.enabled:
                load_warm_start(gpt_wo_ddp_ema, ema_state_dict, 'rush_resume_ema')
            elif args.sr_condition and uses_continuous_lr(getattr(args, 'sr_condition_mode', 'prefix')):
                # The EMA snapshot already owns the final common initialization.
                load_abc_state(gpt_wo_ddp_ema, ema_state_dict, warm_start=not trained_checkpoint)
            else:
                load_warm_start(gpt_wo_ddp_ema, ema_state_dict, 'rush_resume_ema')
        del cpu_d, state_dict, ema_state_dict
    elif (original_request is None and joint_request is None and args.sr_condition
          and uses_continuous_lr(getattr(args, 'sr_condition_mode', 'prefix'))):
        initialize_sr_weights(gpt_wo_ddp, vae_local, args)
    if args.use_fsdp_model_ema and gpt_wo_ddp_ema is None:
        gpt_wo_ddp_ema = get_ema_model(gpt_wo_ddp)
    sr_frozen_names = ()
    if args.sr_condition:
        from grn.models.sr_attention_config import configure_sr_trainability
        sr_policy = configure_sr_trainability(gpt_wo_ddp, bool(args.sr_freeze_backbone))
        sr_frozen_names = tuple(sr_policy['frozen_names'])
        print(f'[SR trainability] freeze_backbone={args.sr_freeze_backbone} {sr_policy}')


    ndim_dict = {name: para.ndim for name, para in gpt_wo_ddp.named_parameters() if para.requires_grad}
    
    print(f'[PT] GPT model = {gpt_wo_ddp}\n\n')
    print(f'[PT] GPT model details:')
    for name, param in gpt_wo_ddp.named_parameters():
        print(f"Name: {name}, Shape: {param.shape}")
    print(f'[PT][#para], GPT={num_para:.2f}M parameters\n\n')
    
    gpt_uncompiled = gpt_wo_ddp
    gpt_wo_ddp = args.compile_model(gpt_wo_ddp, args.tfast)

    gpt_ddp_ema = None
    if args.zero:
        from torch.distributed.fsdp import ShardingStrategy
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        from torch.distributed.device_mesh import init_device_mesh
        # use mix prec: https://github.com/pytorch/pytorch/issues/76607
        if args.fsdp_warp_mode == 'full':
            print(f'warp all modules for fsdp')
            def my_policy(
                module: torch.nn.Module,
                recurse: bool,
                **kwargs,
            ) -> bool:
                return True
            auto_wrap_policy = my_policy
        else:
            print(f'warp transformer blocks for fsdp')
            auto_wrap_policy = ModuleWrapPolicy([MultipleLayers, ])
        
        if args.enable_hybrid_shard == 1:
            sharding_strategy = ShardingStrategy.HYBRID_SHARD if args.zero == 3 else ShardingStrategy._HYBRID_SHARD_ZERO2
            world_size = dist.get_world_size()
            assert world_size % args.inner_shard_degree == 0
            assert args.inner_shard_degree > 1 and args.inner_shard_degree <= world_size
            device_mesh = init_device_mesh('cuda', (world_size // args.inner_shard_degree, args.inner_shard_degree))
        elif args.enable_hybrid_shard == -1: # no shard
            sharding_strategy = ShardingStrategy.NO_SHARD
            device_mesh = None
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD if args.zero == 3 else ShardingStrategy.SHARD_GRAD_OP
            device_mesh = None
        print(f'{">" * 45 + " " * 5} FSDP INIT with {args.zero=} {sharding_strategy=} {auto_wrap_policy=} {" " * 5 + "<" * 45}', flush=True)

        if args.fsdp_init_device == 'cpu':
            gpt_wo_ddp = gpt_wo_ddp.cpu()

        gpt_ddp: FSDP = FSDP(
            gpt_wo_ddp, 
            device_id=dist.get_local_rank(),
            sharding_strategy=sharding_strategy, 
            mixed_precision=None,
            auto_wrap_policy=auto_wrap_policy, 
            use_orig_params=True, 
            sync_module_states=True, 
            limit_all_gathers=True,
            device_mesh=device_mesh,
        ).to(args.device)
        
        if args.use_fsdp_model_ema:
            gpt_wo_ddp_ema = gpt_wo_ddp_ema.to(args.device)
            gpt_ddp_ema: FSDP = FSDP(
                gpt_wo_ddp_ema, 
                device_id=dist.get_local_rank(),
                sharding_strategy=sharding_strategy, 
                mixed_precision=None,
                auto_wrap_policy=auto_wrap_policy, 
                use_orig_params=True, 
                sync_module_states=True, 
                limit_all_gathers=True,
                device_mesh=device_mesh,
            )
    else:
        ddp_class = DDP if dist.initialized() else misc.NullDDP
        gpt_ddp: DDP = ddp_class(gpt_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=args.dbg, broadcast_buffers=False)
    torch.cuda.synchronize()

    # =============== build optimizer ===============
    nowd_keys = set()
    nowd_keys |= {
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
        'text_proj_for_sos.ca.mat_q',
        'scale_tokens', 'class_tokens'
    }
    names, paras, para_groups = filter_params(
        gpt_ddp if args.zero else gpt_wo_ddp,
        ndim_dict,
        nowd_keys=nowd_keys,
        allowed_frozen_prefixes=sr_frozen_names,
        prefix_lr_scales=(sr_parameter_lr_scales(args) if args.sr_condition else ()),
        default_lr_scale=(args.sr_backbone_lr_scale if args.sr_condition else 1.0),
    )
    del ndim_dict
    opt_clz = partial(
        torch.optim.AdamW,
        betas=(0.9, args.adam_beta2),
        fused=bool(args.fused_optimizer),
    )
    opt_kw = dict(lr=args.tlr, weight_decay=args.twd)
    print(f'[vgpt] optim={opt_clz}, opt_kw={opt_kw}\n')
    gpt_optim = opt_clz(params=para_groups, **opt_kw)
    if v2_runtime_audit_dir and resume_state is None:
        if joint_request is not None:
            write_joint_audit(v2_runtime_audit_dir, v2_initialization, gpt_optim,
                args=args, model=gpt_wo_ddp, rank=dist.get_rank(), policy=sr_policy)
        elif original_request is not None:
            write_original_initialization_audit(v2_runtime_audit_dir, v2_initialization,
                gpt_optim, args=args, model=gpt_wo_ddp, rank=dist.get_rank(), policy=sr_policy)
        else:
            from grn.utils_t2iv.sr_v2_runtime import write_initialization_audit
            write_initialization_audit(v2_runtime_audit_dir, v2_initialization, gpt_optim)
    del names, paras, para_groups
    return vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim

def build_dataset(args):
    train_dataset = build_joint_dataset(
        args, 
        args.meta_folders,
        args.meta_folder_repeats,
        max_caption_len=args.tlen, 
        short_prob=args.short_cap_prob, 
        load_vae_instead_of_image=False
    )
    return train_dataset

def phase_checkpoint_steps(value, microsteps_per_phase, phases):
    """Validate an opt-in phase-boundary whitelist without changing legacy saves."""
    if value == '':
        return None
    try:
        steps = tuple(int(part) for part in value.split(','))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError('Invalid phase checkpoint microsteps') from error
    if (not steps or microsteps_per_phase <= 0 or phases <= 0
            or tuple(sorted(set(steps))) != steps
            or any(s <= 0 or s > microsteps_per_phase * phases or s % microsteps_per_phase for s in steps)):
        raise ValueError('Checkpoint microsteps must be increasing unique positive phase boundaries within horizon')
    return steps


def main_train(args: arg_util.Args):
    if args.checkpoint_type == 'torch':
        from grn.utils_t2iv.save_and_load import CKPTSaver, auto_resume
        saver = CKPTSaver(dist.is_master(), eval_milestone=None)
    else:
        raise ValueError(f'{args.checkpoint_type=}')
    built = build_everything_from_args(args, saver)
    if built is None:
        return
    text_tokenizer, text_encoder, trainer, start_ep, resume_it, start_global_it, native_empty_text = built
    gc.collect(), torch.cuda.empty_cache()
    logging_params_milestone: List[int] = np.linspace(1, args.ep, 10+1, dtype=int).tolist()
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    
    # ============================================= epoch loop begins =============================================
    # build wandb logger
    if dist.is_master():
        wandb_utils.wandb.init(project=args.project_name, name=args.exp_name, config={})

    dataloader_generator = torch.Generator()
    for ep in range(start_ep, args.ep):
        # build data at each epoch to ensure read meta take effects for each dataloader worker
        args.epoch = ep
        train_dataset = build_dataset(args)
        iters_train = len(train_dataset)
        print(f'[PT info]  from {start_global_it=} {iters_train=}=======>  bed: {args.bed}  <=======\n')
        
        # build dataloader
        phase_saves = phase_checkpoint_steps(getattr(args, 'sr_checkpoint_microsteps', ''),
                                            iters_train, args.ep)
        epoch_start_it = resume_it if ep == start_ep else 0
        # A periodic checkpoint at an exact epoch boundary stores the
        # in-epoch cursor as ``iter == iters_train``.  The epoch-complete
        # save normally rewrites that checkpoint with ``next_ep=ep+1``;
        # when a process is interrupted between those two saves, resume
        # must skip the already-completed epoch instead of constructing an
        # exhausted dataloader and raising StopIteration.
        if ep == start_ep and epoch_start_it >= iters_train:
            print(f'[resume] completed epoch boundary detected: ep={ep} '
                  f'iter={epoch_start_it}; advancing to ep={ep + 1}', flush=True)
            resume_it = 0
            continue
        dataloader_generator.manual_seed(
            args.seed + ep * tdist.get_world_size() + tdist.get_rank()
        )
        from grn.utils_t2iv.sr_v2_qualified_training import loader_resume_view
        loader_dataset, skipped_prefix = loader_resume_view(args, train_dataset, epoch_start_it)
        train_dataloader = DataLoader(
            dataset=loader_dataset,
            num_workers=args.workers,
            pin_memory=True,
            batch_size=None,
            shuffle=False,
            generator=dataloader_generator,
        )
        train_dataloader_iter_obj = iter(train_dataloader)
        for _ in range(skipped_prefix):
            next(train_dataloader_iter_obj)

        # [train one epoch]
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep=ep,
            start_it=epoch_start_it,
            start_global_it=start_global_it,
            me=None,
            saver=saver,
            args=args,
            ld_or_itrt=train_dataloader_iter_obj,
            iters_train=iters_train,
            text_tokenizer=text_tokenizer, text_encoder=text_encoder,
            trainer=trainer,
            native_empty_text=native_empty_text,
            logging_params_milestone=logging_params_milestone,
        )
        if getattr(args, '_sr_run_stopped_at', 0):
            return  # exact mid-epoch checkpoint already saved; do not claim epoch completion
        start_global_it += iters_train - epoch_start_it
        if args.checkpoint_type == 'torch' and (
                phase_saves is None or start_global_it in phase_saves):
            saver.sav(
                args=args,
                g_it=start_global_it,
                next_ep=ep + 1,
                next_it=0,
                trainer=trainer,
                acc_str='[epoch complete]',
                eval_milestone=None,
                also_save_to=None,
                best_save_to=None,
            )
        del stats, train_dataset, train_dataloader
        time.sleep(10), gc.collect(), time.sleep(10) # torch.cuda.empty_cache()
    return


def train_one_ep(
    ep: int, start_it: int, start_global_it: int, me: misc.MetricLogger,
    saver, args: arg_util.Args, ld_or_itrt, iters_train: int, 
    text_tokenizer: T5TokenizerFast, text_encoder: T5EncoderModel, trainer, logging_params_milestone,
    native_empty_text=None,
):
    # IMPORTANT: import heavy packages after the Dataloader object creation/iteration to avoid OOM
    step_cnt = 0
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    g_it, max_it = start_global_it, args.ep * iters_train
    from grn.utils_t2iv.sr_v2_run_budget import planned_stop, record_stop
    run_stop = planned_stop(args, epoch=ep, full_epoch_microsteps=iters_train,
                            start_it=start_it, start_global_it=start_global_it)
    args._sr_run_stopped_at = 0

    
    me = misc.MetricLogger()
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{value:.2g}')) for x in ['tlr']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['tnm']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.3f} ({global_avg:.3f})')) for x in ['L', 'L_i', 'L_v']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Acc', 'Acc_i', 'Acc_v']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['seq_usage']]
    # ============================================= iteration loop begins =============================================
    for it, data in me.log_every(start_it, iters_train, ld_or_itrt, args.log_freq, args.log_every_iter, header):
        # for dpo training, we will save the first iter model for comparison
        g_it += 1

        # [get data]
        images = data['images']
        lr_conditions = data.get('lr_conditions')
        captions = data['captions']
        raw_features_bcthw = data['raw_features_bcthw']
        feature_cache_files4images = data['feature_cache_files4images']
        media = data['media']
        meta_list = data['meta_list']

        # # [prepare text features]
        if args.sr_condition and uses_empty_text(getattr(args, 'sr_condition_mode', 'prefix')):
            text_cond_tuple = pack_empty_text(native_empty_text, len(images) + len(raw_features_bcthw), args.device)
        elif args.sr_condition:
            sample_count = len(images) + len(raw_features_bcthw)
            caption_nums = [1] * sample_count
            lens = [0] * sample_count
            kv_compact = torch.empty((0, args.Ct5), dtype=torch.float32, device=args.device)
            cu_seqlens_k = torch.zeros(sample_count + 1, dtype=torch.int32, device=args.device)
            text_cond_tuple = (
                kv_compact,
                lens,
                cu_seqlens_k,
                0,
                caption_nums,
            )
        elif args.add_class_token > 0: # c2i task
            text_cond_tuple = [[] for _ in range(5)]
        else:
            caption_nums = [len(item) for item in captions]
            flatten_captions = []
            for item in captions:
                flatten_captions.extend(item)
            if args.text_tokenizer_type == 'flan_t5':
                tokens = text_tokenizer(text=flatten_captions, max_length=text_tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt')  # todo: put this into dataset
                input_ids = tokens.input_ids.cuda(non_blocking=True)
                mask = tokens.attention_mask.cuda(non_blocking=True)
                text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
                lens: List[int] = mask.sum(dim=-1).tolist()
                cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
                Ltext = max(lens)
                kv_compact = []
                for text_ind, (len_i, feat_i) in enumerate(zip(lens, text_features.unbind(0))):
                    kv_compact.append(feat_i[:len_i])
                kv_compact = torch.cat(kv_compact, dim=0)
                text_cond_tuple: Tuple[torch.FloatTensor, List[int], torch.LongTensor, int] = (kv_compact, lens, cu_seqlens_k, Ltext, caption_nums)
            else:
                text_features = text_encoder(flatten_captions, args.device)
                lens = [len(item) for item in text_features]
                cu_seqlens_k = [0]
                for len_i in lens:
                    cu_seqlens_k.append(cu_seqlens_k[-1] + len_i)
                cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32)
                Ltext = max(lens)
                kv_compact = torch.cat(text_features, dim=0).float()
                text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext, caption_nums)

        if len(images):
            images = [item.to(args.device, non_blocking=True) for item in images]
        if lr_conditions is not None:
            lr_conditions = [
                item.to(args.device, non_blocking=True) for item in lr_conditions
            ]
        if len(raw_features_bcthw):
            raw_features_bcthw = [item.to(args.device, non_blocking=True) for item in raw_features_bcthw]

        # Schedule in optimizer-step units. lin0 matches the offline VARSR protocol.
        accumulation = max(1, args.gradient_accumulation)
        optimizer_steps_per_epoch = math.ceil(iters_train / accumulation)
        total_optimizer_steps = args.ep * optimizer_steps_per_epoch
        optimizer_step = ep * optimizer_steps_per_epoch + it // accumulation
        warmup_steps = max(1, round(args.wp * optimizer_steps_per_epoch))
        if optimizer_step < warmup_steps:
            cur_lr_ratio = args.wp0 + (1 - args.wp0) * optimizer_step / warmup_steps
        else:
            decay_progress = (optimizer_step - warmup_steps) / max(
                1, total_optimizer_steps - warmup_steps - 1
            )
            decay_progress = min(max(decay_progress, 0.0), 1.0)
            if args.sche == 'lin0':
                plateau = 0.3
                if decay_progress < plateau:
                    cur_lr_ratio = 1.0
                else:
                    cur_lr_ratio = args.wpe + (1 - args.wpe) * (
                        (1 - decay_progress) / (1 - plateau)
                    )
            elif args.sche == 'lin00':
                cur_lr_ratio = args.wpe + (1 - args.wpe) * (1 - decay_progress)
            elif args.sche.startswith('lin'):
                cur_lr_ratio = 1 - decay_progress * (1 - args.wpe)
            elif args.sche.startswith('cos'):
                cur_lr_ratio = args.wpe + (1 - args.wpe) * (
                    0.5 + 0.5 * math.cos(math.pi * decay_progress)
                )
            else:
                cur_lr_ratio = 1
        cur_lr = args.tlr * cur_lr_ratio
        for param_group in trainer.gpt_opt.param_groups:
            param_group['lr'] = cur_lr * param_group.get('lr_sc', 1)
            param_group['weight_decay'] = args.twd * param_group.get('wd_sc', 1)

        group_start = (it // accumulation) * accumulation
        accumulation_divisor = min(accumulation, iters_train - group_start)
        stepping = ((it + 1) % accumulation == 0) or (it + 1 == iters_train)
        step_cnt += int(stepping)
        
        trainer.train_step(
            ep=ep, it=it, g_it=g_it, stepping=stepping, clip_decay_ratio=1,
            metric_lg=me, 
            logging_params=stepping and step_cnt == 1 and (ep < 4 or ep in logging_params_milestone), 
            inp_B3HW=images,
            lr_conditions=lr_conditions,
            raw_features_bcthw=raw_features_bcthw,
            feature_cache_files4images=feature_cache_files4images,
            text_cond_tuple=text_cond_tuple,
            media=media,
            meta_list=meta_list,
            args=args,
            accumulation_divisor=accumulation_divisor,
        )

        if (
            # A bounded stop uses this same real saver, without duplicate snapshots.
            stepping
            and args.checkpoint_type == 'torch'
            and args.save_model_iters_freq > 0
            and g_it % args.save_model_iters_freq == 0
            and not (run_stop and it + 1 == iters_train)
        ):
            saver.sav(
                args=args,
                g_it=g_it,
                next_ep=ep,
                next_it=it + 1,
                trainer=trainer,
                acc_str='[periodic]',
                eval_milestone=None,
                also_save_to=None,
                best_save_to=None,
            )

        me.update(tlr=cur_lr)
        if run_stop and g_it == run_stop:
            if not stepping:
                raise RuntimeError('Bounded stop reached inside an optimizer window')
            periodic_saved = (args.save_model_iters_freq > 0
                              and g_it % args.save_model_iters_freq == 0
                              and it + 1 != iters_train)
            if not periodic_saved:
                next_ep = ep + int(it + 1 == iters_train)
                saver.sav(args=args, g_it=g_it, next_ep=next_ep,
                    next_it=0 if next_ep != ep else it + 1, trainer=trainer,
                    acc_str='[bounded prefix; not full training completion]',
                    eval_milestone=None, also_save_to=None, best_save_to=None)
            record_stop(args, trainer, epoch=ep, next_it=it + 1, global_it=g_it,
                        full_epoch_microsteps=iters_train, rank=dist.get_rank())
            # MetricLogger updates timing after yield; a deliberate early break
            # must account for the final completed microstep once.
            me.iter_time.update(time.time() - me.iter_end_t)
            break
    # ============================================= iteration loop ends =============================================
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost


def main():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    main_train(args)
    print(f'final args:\n\n{str(args)}')
    if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
        sys.stdout.close(), sys.stderr.close()
    dist.barrier()


if __name__ == '__main__':
    main()
