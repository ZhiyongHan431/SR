import hashlib
import os
import os.path as osp
import re
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import imageio
import numpy as np
import torch
from PIL import Image
from timm.models import create_model
from torchvision.transforms.functional import to_tensor

torch._dynamo.config.cache_size_limit = 64

from grn.models.basic import *
from grn.models.grn import GRN
from grn.models.umt5.t5 import T5EncoderModel
from grn.utils_t2iv.sr_native import encode_native_lr, load_empty_text_cache, pack_empty_text, restore_sr_architecture
from grn.models.sr_condition_modes import uses_empty_text
from grn.utils_t2iv.sr_condition_init import load_abc_state
from grn.utils_t2iv.sr_v2_condition_runtime import (
    NEW_PREFIXES, condition_config, materialize_condition_modules, load_condition_state,
)


def extract_key_val(text):
    return {k: v.lstrip() for k, v in re.findall(r'<(.+?):(.+?)>', text)}

def encode_prompt(t5_path, text_tokenizer, text_encoder, prompt, args=None):
    print(f't5 encode prompt: {prompt}')
    text_encoder.model.to(args.other_device)
    text_features = text_encoder([prompt], args.other_device)
    lens = [len(item) for item in text_features]
    cu_seqlens_k = [0]
    for len_i in lens:
        cu_seqlens_k.append(cu_seqlens_k[-1] + len_i)
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32)
    Ltext = max(lens)
    kv_compact = torch.cat(text_features, dim=0).float()
    kv_compact = kv_compact.to(args.other_device)
    text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    return text_cond_tuple

def gen_one_example(
    model, 
    vae, 
    text_tokenizer,
    text_encoder,
    prompt, 
    cfg_list=[],
    tau_list=[],
    negative_prompt="",
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_latent_dim=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    input_use_interplote_up=False,
    args=None,
    get_visual_rope_embeds=None,
    noise_list=None,
    return_summed_code_only=False,
    class_token_id=0,
    first_frame_condition=False,
    first_frame_path=None,
    lr_condition=None,
    sr_semantic_features=None,
    sr_calibrator_kappa=None,
    sr_quality_guidance=None,
    sr_negative_lr_condition=None,
    sr_negative_semantic_features=None,
    sr_text_features=None,
    sr_negative_text_features=None,
):
    sstt = time.time()
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)
    
    native_lr_input, native_lr_latent = None, None
    mode = getattr(args, 'sr_condition_mode', 'prefix')
    native_sr = lr_condition is not None and mode == 'native_adapter'
    text_overrides = sr_text_features is not None or sr_negative_text_features is not None
    if text_overrides and (lr_condition is None or mode != 'prefix_text'):
        raise ValueError('Text-slot features require prefix_text SR')
    if lr_condition is not None and uses_empty_text(mode):
        if native_sr and any(float(cfg) != 1.0 for cfg in cfg_list):
            raise ValueError('native SR Stage1 requires guidance=1')
        text_channels = getattr(args, 'text_channels', 4096)
        features = load_empty_text_cache(args.sr_native_empty_text_path, text_channels)
        empty_text = pack_empty_text(features, 1, args.other_device)[:4]
        text_cond_tuple, negative_label_B_or_BLT = [empty_text], empty_text
        if text_overrides:
            # Opt-in: a semantic prompt or cached prompts replace the constant
            # empty prompt of the first / second packed branch. Their lengths may differ.
            from grn.models.sr_semantic_prompt import checked_text_features
            if sr_text_features is not None:
                text_cond_tuple = [pack_empty_text(
                    checked_text_features(sr_text_features, text_channels), 1, args.other_device)[:4]]
            if sr_negative_text_features is not None:
                negative_label_B_or_BLT = pack_empty_text(
                    checked_text_features(sr_negative_text_features, text_channels), 1, args.other_device)[:4]
        if native_sr:
            native_lr_input, native_lr_latent = encode_native_lr(vae, lr_condition, args.sr_hr_size, args.hbq_round)
    elif lr_condition is None:
        text_cond_tuple = []
        for prompt_str in prompt:
            text_cond_tuple.append(encode_prompt(args.text_encoder_ckpt, text_tokenizer, text_encoder, prompt_str, args=args))
        negative_label_B_or_BLT = encode_prompt(args.text_encoder_ckpt, text_tokenizer, text_encoder, negative_prompt, args=args)
    else:
        text_cond_tuple, negative_label_B_or_BLT = None, None
    # Do not add a new keyword to the legacy-off generation call.
    semantic_kwargs = {} if sr_semantic_features is None else {'sr_semantic_features': sr_semantic_features}
    if sr_calibrator_kappa is not None:
        from grn.models.sr_v2_calibrator import resolve_calibrator_kappa
        semantic_kwargs['sr_calibrator_kappa'] = resolve_calibrator_kappa(sr_calibrator_kappa)
    # Opt-in quality guidance: forwarded only when requested, so the default
    # call keeps its historical keyword set.
    if sr_quality_guidance is not None:
        semantic_kwargs['sr_quality_guidance'] = sr_quality_guidance
        if sr_negative_lr_condition is not None:
            semantic_kwargs['sr_negative_lr_condition'] = sr_negative_lr_condition
        if sr_negative_semantic_features is not None:
            semantic_kwargs['sr_negative_semantic_features'] = sr_negative_semantic_features
    elif sr_negative_lr_condition is not None or sr_negative_semantic_features is not None:
        raise ValueError('Negative LR inputs require sr_quality_guidance')
    print(f'cfg: {cfg_list}, tau: {tau_list}')
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        stt = time.time()
        out = model.autoregressive_infer(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
            B=1, negative_label_B_or_BLT=negative_label_B_or_BLT, force_gt_Bhw=None,
            cfg_sc=cfg_sc, cfg_list=cfg_list, tau_list=tau_list, top_k=top_k, top_p=top_p,
            returns_vemb=1, ratio_Bl1=None, gumbel=gumbel, norm_cfg=False,
            cfg_exp_k=cfg_exp_k, cfg_insertion_layer=cfg_insertion_layer,
            vae_latent_dim=vae_latent_dim, softmax_merge_topk=softmax_merge_topk,
            ret_img=True, trunk_scale=1000,
            gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, inference_mode=True,
            input_use_interplote_up=input_use_interplote_up,
            args=args,
            get_visual_rope_embeds=get_visual_rope_embeds,
            noise_list=noise_list,
            return_summed_code_only=return_summed_code_only,
            class_token_id=class_token_id,
            first_frame_condition=first_frame_condition,
            first_frame_path=first_frame_path,
            lr_condition=lr_condition,
            native_lr_input=native_lr_input,
            native_lr_latent=native_lr_latent,
            **semantic_kwargs,
        )
        _, pred_multi_scale_bit_labels, img_list = out
            
    print(f"cost: {time.time() - sstt}, model cost={time.time() - stt}")
    img = img_list[0]
    return img

def get_prompt_id(prompt):
    return hash_string(prompt)

def load_tokenizer(t5_path ='', device='cuda'):
    print(f'[Loading tokenizer and text encoder]')
    if isinstance(device, str):
        device = torch.device(device)
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=osp.join(t5_path, 'models_t5_umt5-xxl-enc-bf16.pth'),
        tokenizer_path=osp.join(t5_path, 'umt5-xxl'),
        enable_fsdp=False)
    text_tokenizer = text_encoder.tokenizer
    return text_tokenizer, text_encoder

def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(np.round(tgt_w / (width / height)))
    else:
        resized_height = tgt_h
        resized_width = int(np.round((width / height) * tgt_h))
    pil_img = pil_img.resize((resized_width, resized_height), resample=Image.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    return im * 2 - 1

def hash_string(input_string):
    md5 = hashlib.md5()
    md5.update(input_string.encode('utf-8'))
    return md5.hexdigest()

def joint_vi_vae_encode_decode(vae, image_path, scale_schedule, device, tgt_h, tgt_w):
    pil_image = Image.open(image_path).convert('RGB')
    inp = transform(pil_image, tgt_h, tgt_w)
    inp = inp.unsqueeze(0).to(device)
    scale_schedule = [(item[0], item[1], item[2]) for item in scale_schedule]
    t1 = time.time()
    h, z, _, all_bit_indices, _, _ = vae.encode(inp, scale_schedule=scale_schedule)
    t2 = time.time()
    recons_img = vae.decode(z)[0]
    if len(recons_img.shape) == 4:
        recons_img = recons_img.squeeze(1)
    print(f'recons: z.shape: {z.shape}, recons_img shape: {recons_img.shape}')
    t3 = time.time()
    print(f'vae encode takes {t2-t1:.2f}s, decode takes {t3-t2:.2f}s')
    recons_img = ((recons_img + 1) / 2).clamp(0, 1)
    recons_img = recons_img.permute(1, 2, 0).mul(255).cpu().numpy().astype(np.uint8)
    gt_img = ((inp[0] + 1) / 2).clamp(0, 1)
    gt_img = gt_img.permute(1, 2, 0).mul(255).cpu().numpy().astype(np.uint8)
    print(recons_img.shape, gt_img.shape)
    return gt_img, recons_img, all_bit_indices


def load_transformer(vae, args):
    device = torch.device(args.other_device)
    model_path = args.model_path
    has_trainer_args = False
    checkpoint_args = {}
    if not model_path:
        state_dict = None
    elif args.checkpoint_type == 'torch': 
        # copy large model to local; save slim to local; and copy slim to nas; load local slim model
        slim_model_path = model_path
        print(f'load checkpoint from {slim_model_path}')
        # Training checkpoints also contain the (much larger) AdamW state. Loading
        # the whole object directly onto CUDA can OOM before we even select the
        # transformer weights, so keep the archive mmap-backed on CPU.
        loaded_state = torch.load(slim_model_path, map_location='cpu', mmap=True)
        if 'trainer' in loaded_state:
            has_trainer_args = isinstance(loaded_state.get('args'), dict)
            checkpoint_args = loaded_state.get('args', {})
            restore_sr_architecture(args, checkpoint_args)
            state_dict = loaded_state['trainer']['gpt_fsdp']
            del loaded_state
        else:
            restore_sr_architecture(args, {})
            state_dict = loaded_state

    s2_config = condition_config(args)
    has_s2_tensors = state_dict is not None and any(name.startswith(NEW_PREFIXES) for name in state_dict)
    if s2_config.enabled and getattr(args, 'sr_v2_condition_promote', 0):
        raise ValueError('S2 inference does not permit condition promotion')
    if (s2_config.enabled or has_s2_tensors) and not has_trainer_args:
        raise RuntimeError('S2 inference requires a complete trainer+args checkpoint; bare S2 state is unsupported')
    if has_s2_tensors and not s2_config.enabled:
        raise RuntimeError('S2 condition tensors require matching saved architecture metadata')

    from grn.utils_t2iv.sr_v2_calibrator_runtime import (
        PREFIX as S3_PREFIX, calibrator_config, materialize_calibrator, load_calibrator_state)
    s3_config = calibrator_config(args)
    has_s3_tensors = state_dict is not None and any(name.startswith(S3_PREFIX) for name in state_dict)
    if getattr(args, 'sr_v2_calibrator_promote', 0):
        raise ValueError('S3 inference does not permit promotion')
    if (s3_config.enabled or has_s3_tensors) and not has_trainer_args:
        raise ValueError('S3 inference requires a complete trainer+args checkpoint')
    if has_s3_tensors and not s3_config.enabled:
        raise ValueError('S3 tensors require matching saved architecture metadata')
    if s3_config.enabled and args.bf16:
        raise ValueError('S3 saved inference currently requires native FP32 model weights')

    print(f'[Loading Model]')
    if getattr(args, 'sr_condition', 0) and uses_empty_text(getattr(args, 'sr_condition_mode', 'prefix')):
        load_empty_text_cache(args.sr_native_empty_text_path, args.text_channels)
    # Check if device is CUDA before enabling autocast
    if device.type == 'cuda':
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
            model = create_model(
                args.model,
                vae_local=vae, text_channels=args.text_channels, text_maxlen=512,
                shared_aln=True, raw_scale_schedule=None,
                checkpointing='full-block',
                customized_flash_attn=False,
                fused_norm=True,
                pad_to_multiplier=128,
                use_flex_attn=False,
                num_of_label_value=args.num_of_label_value,
                rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
                pn=args.pn,
                apply_spatial_patchify=args.apply_spatial_patchify,
                inference_mode=True,
                train_h_div_w_list=args.train_h_div_w_list,
                dynamic_scale_schedule=args.dynamic_scale_schedule,
                video_frames=args.video_frames,
                other_args=args,
            ).to(device=device)
    else:
        with torch.no_grad():
            model = create_model(
                args.model,
                vae_local=vae, text_channels=args.text_channels, text_maxlen=512,
                shared_aln=True, raw_scale_schedule=None,
                checkpointing='full-block',
                customized_flash_attn=False,
                fused_norm=True,
                pad_to_multiplier=128,
                use_flex_attn=False,
                num_of_label_value=args.num_of_label_value,
                rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
                pn=args.pn,
                apply_spatial_patchify=args.apply_spatial_patchify,
                inference_mode=True,
                train_h_div_w_list=args.train_h_div_w_list,
                dynamic_scale_schedule=args.dynamic_scale_schedule,
                video_frames=args.video_frames,
                other_args=args,
            ).to(device=device)
    # New branches must exist before eval/freeze and exact S2 state loading.
    materialize_condition_modules(model, args)
    materialize_calibrator(model, args)
    print(f'[you selected model with {args.model}] model size: {sum(p.numel() for p in model.parameters())/1e9:.2f}B, bf16={args.bf16}')
    if args.bf16:
        for block in model.unregistered_blocks:
            block.bfloat16()
    model.eval()
    model.requires_grad_(False)
    # Only call cuda() if device is CUDA
    if device.type == 'cuda':
        model.cuda()
        torch.cuda.empty_cache()
    print(f'[Load model weights]')
    if state_dict is not None:
        if s3_config.enabled:
            load_result = load_calibrator_state(model, state_dict, args, checkpoint_args)
        elif s2_config.enabled:
            load_result = load_condition_state(model, state_dict, promote_b=False)
        elif getattr(args, 'sr_condition', 0):
            load_result = load_abc_state(model, state_dict)
        else:
            load_result = model.load_state_dict(state_dict, strict=True)
        print(load_result)
        del state_dict
    return model

def images2video(ndarray_image_list, fps=24, save_filepath='tmp.mp4'):
    # ndarray_image_list: bgr sequence
    save_dir = osp.dirname(save_filepath)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if len(ndarray_image_list) == 1:
        save_filepath = osp.splitext(save_filepath)[0] + '.jpg'
        cv2.imwrite(save_filepath, ndarray_image_list[0]) # bgr
        print(f"Image saved as {osp.abspath(save_filepath)}")
    else:
        # imageio takes rgb, so convert bgr to rgb
        imageio.mimsave(save_filepath, ndarray_image_list[..., ::-1], fps=fps) 
        print(f"Video saved as {osp.abspath(save_filepath)}")

def imgs_tensor2uint8_imgs(imgs_tensor):
    imgs_tensor = imgs_tensor.permute(1, 2, 3, 0) # [c,t,h,w] -> [t,h,w,c]
    imgs_tensor = ((imgs_tensor + 1) / 2).clamp(0, 1)
    return imgs_tensor.mul(255).to(torch.uint8).flip(dims=(3,)).cpu().numpy()
