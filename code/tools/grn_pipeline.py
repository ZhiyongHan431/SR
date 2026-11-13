import os
import json
import numpy as np
import torch
from PIL import Image
import tempfile

from grn.utils_t2iv.infer import (
    load_tokenizer,
    load_transformer,
    gen_one_example,
    images2video
)
from grn.utils_t2iv.load import load_visual_tokenizer
from grn.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from grn.schedules import get_encode_decode_func
from grn.utils_t2iv.sr_v2_condition_runtime import condition_config, build_semantic_encoder


class GRNPipeline:
    def __init__(self, model, vae, text_tokenizer, text_encoder, args, device='cuda'):
        self.model = model
        self.vae = vae
        self.text_tokenizer = text_tokenizer
        self.text_encoder = text_encoder
        self.args = args
        self.device = device
        # args have already been restored by load_transformer. Keep the frozen
        # external encoder out of GRN state/FSDP; off performs no asset loading.
        self.sr_semantic_encoder = None
        if condition_config(args).semantic:
            self.sr_semantic_encoder = build_semantic_encoder(args, device)
        self.video_encode, self.video_decode, self.get_visual_rope_embeds, self.get_scale_pack_info = \
            get_encode_decode_func(args.dynamic_scale_schedule)

    @classmethod
    def from_pretrained(
        cls,
        model_path='./weights/model.pth',
        vae_path='./weights/hbq_tokenizer.ckpt',
        text_encoder_ckpt='./weights/umt5-xxl',
        device='cuda',
        torch_dtype=torch.bfloat16,
        hf_repo_id=None,
        task='t2i',
        pn='1M',
        model='GRN2b',
        use_slow_attn=False,
        sr_native_empty_text_path='',
    ):
        task = task.upper()
        # download weights from Hugging Face Hub
        if hf_repo_id:
            from huggingface_hub import hf_hub_download, snapshot_download
            print(f"download weights from Hugging Face Hub: {hf_repo_id}")
            if task == 'T2I':
                model_path = hf_hub_download(repo_id=hf_repo_id, filename="GRN_T2I_2B_FSA_251600.pth")
            elif task == 'T2V' and model == 'GRN2b':
                model_path = hf_hub_download(repo_id=hf_repo_id, filename="GRN_T2V_2B_FSA_sft147400_ema.pth")
            elif task == 'T2V' and model == 'GRN8b':
                model_path = hf_hub_download(repo_id=hf_repo_id, filename="GRN_T2V_TITAN_FSA_sft10600_ema.pth")
            else:
                raise ValueError(f"Unknown task: {task}")
            vae_path = hf_hub_download(repo_id=hf_repo_id, filename="HBQ_image_video_tokenizer_64dim_M4_20260626.ckpt")
            snapshot_path = snapshot_download(repo_id=hf_repo_id, allow_patterns="umt5-xxl/**")
            text_encoder_ckpt = os.path.join(snapshot_path, "umt5-xxl")
            print(os.listdir(snapshot_path))
        
        args = cls._get_default_args()
        args.sr_condition = int(task == 'SR')
        args.sr_native_empty_text_path = sr_native_empty_text_path
        args.model = model
        args.model_path = model_path
        args.vae_path = vae_path
        args.text_encoder_ckpt = text_encoder_ckpt
        args.use_slow_attn = use_slow_attn
        if isinstance(device, str):
            device = torch.device(device)
        args.other_device = device
        args.task = task
        args.pn = '0.25M' if args.sr_condition else pn

        # Derived parameters
        args.max_duration = (args.video_frames - 1) / 4
        args.num_of_label_value = args.num_lvl
        args.semantic_num_lvl = args.num_lvl
        args.detail_num_lvl = args.num_lvl
        args.semantic_scale_dim = args.vae_latent_dim
        args.detail_scale_dim = args.vae_latent_dim

        # Load models
        if args.sr_condition:
            text_tokenizer, text_encoder = None, None
        else:
            text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt, device=device)
        vae = load_visual_tokenizer(args, device=device)
        model = load_transformer(vae, args)

        return cls(model, vae, text_tokenizer, text_encoder, args, device)

    @staticmethod
    def _get_default_args():
        class Args:
            def __init__(self):
                self.video_frames = 81
                self.model_path = './weights/model.pth'
                self.vae_path = './weights/hbq_tokenizer.ckpt'
                self.text_encoder_ckpt = './weights/umt5-xxl'
                self.cfg = 1
                self.fps = 16
                self.cfg_insertion_layer = 0
                self.vae_latent_dim = 64
                self.hbq_round = 4
                self.rope_type = '3d'
                self.num_lvl = 2
                self.rope2d_normalized_by_hw = 2
                self.text_channels = 4096
                self.sr_condition = 0
                self.sr_condition_mode = 'prefix'
                self.sr_v2_semantic = 0
                self.sr_v2_semantic_hidden = 128
                self.sr_v2_semantic_manifest = ''
                self.sr_v2_semantic_manifest_sha256 = ''
                self.sr_v2_condition_coarse_weight = 0.0
                self.sr_v2_condition_seed = 20260916
                self.sr_v2_condition_promote = 0
                self.sr_v2_calibrator_mode = 'off'
                self.sr_v2_calibrator_width = 128
                self.sr_v2_calibrator_cap = 2.0
                self.sr_v2_calibrator_seed = 20260917
                self.sr_v2_calibrator_precision = 'native_fp32'
                self.sr_v2_calibrator_promote = 0
                self.sr_checkpoint_microsteps = ''
                self.sr_view_manifest = ''
                self.sr_view_audit_dir = ''
                self.sr_control_layers = '[4, 12, 20, 28]'
                self.sr_control_dim = 512
                self.sr_control_heads = 8
                self.sr_control_window = 7
                self.sr_condition_init_path = ''
                self.sr_native_adapter_layers = '[4, 12, 20, 28]'
                self.sr_native_adapter_dim = 256
                self.sr_native_empty_text_path = ''
                self.sr_lr_size = 128
                self.sr_hr_size = 512
                self.sr_condition_drop_rate = 0.0
                self.sr_cross_attn = 0
                self.sr_cross_attn_layers = '[4, 12, 20, 28]'
                self.sr_cross_attn_dim = 512
                self.sr_cross_attn_heads = 8
                self.sr_cross_attn_window = 7
                self.sr_cross_attn_global_layers = ''
                self.sr_cross_attn_gate_init = 1e-3
                self.sr_top_k = 0
                self.sr_top_p = 1.0
                self.apply_spatial_patchify = 0
                self.h_div_w_template = 1.0
                self.cache_dir = '/tmp'
                self.checkpoint_type = 'torch'
                self.seed = 42
                self.bf16 = 0
                self.dynamic_scale_schedule = 'GRN_vae_stride16'
                self.train_h_div_w_list = '[]'
                self.max_infer_steps = 50
                self.min_infer_steps = 50
                self.video_caption_type = 'tarsier2_caption'
                self.temporal_compress_rate = 4
                self.cached_video_frames = 81
                self.duration_resolution = 0.25
                self.video_fps = 16
                self.simple_text_proj = 1
                self.min_duration = -1
                self.fsdp_save_flatten_model = 1
                self.use_learnable_dim_proj = 0
                self.use_fsq_cls_head = 1
                self.use_feat_proj = 0
                self.use_clipwise_caption = 0
                self.use_ada_layer_norm = 0
                self.cfg_type = 'cfg_interval_0.0'
                self.add_scale_token = 1
                self.vae_encoder_out_type = 'feature_tanh'
                self.alpha = 1004
                self.refine_mode = 'ar_discrete_GRN_bit'
                self.add_class_token = 0
                self.resample_rand_labels_per_step = 0
                self.cfg_val = 3.0
                self.scale_repetition = ''
                self.gt_leak = -1
                self.use_refined_prompt = None
                self.use_prompt_engineering = 0
                self.quality_prompt = ''
                self.meta = ''
                self.train_split_file = ''
                self.n_sampes = 1
                self.other_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return Args()

    def to(self, device):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.args.other_device = device
        if self.model:
            self.model = self.model.to(device)
        if self.vae:
            self.vae = self.vae.to(device)
        if self.sr_semantic_encoder is not None:
            self.sr_semantic_encoder = self.sr_semantic_encoder.to(device=device)
        return self

    def _prepare_lr_condition(self, lr_image):
        if isinstance(lr_image, (str, os.PathLike)):
            lr_image = Image.open(lr_image).convert('RGB')
        if isinstance(lr_image, Image.Image):
            if lr_image.size != (self.args.sr_lr_size, self.args.sr_lr_size):
                raise ValueError(
                    f"SR expects an exact {self.args.sr_lr_size} x {self.args.sr_lr_size} LR image, "
                    f"got {lr_image.size}"
                )
            array = np.asarray(lr_image.convert('RGB')).copy()
            condition = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        elif isinstance(lr_image, np.ndarray):
            condition = torch.from_numpy(lr_image)
            if condition.ndim == 3 and condition.shape[-1] == 3:
                condition = condition.permute(2, 0, 1)
            condition = condition.unsqueeze(0) if condition.ndim == 3 else condition
        elif torch.is_tensor(lr_image):
            condition = lr_image
            condition = condition.unsqueeze(0) if condition.ndim == 3 else condition
        else:
            raise TypeError("lr_image must be a path, PIL image, NumPy array, or torch tensor")

        if condition.ndim != 4 or tuple(condition.shape[1:]) != (
            3, self.args.sr_lr_size, self.args.sr_lr_size
        ):
            raise ValueError(
                f"SR expects [B, 3, {self.args.sr_lr_size}, {self.args.sr_lr_size}], "
                f"got {tuple(condition.shape)}"
            )
        if condition.shape[0] != 1:
            raise ValueError("The current GRN SR inference loop supports one LR image per call")
        condition = condition.to(device=self.device, dtype=torch.float32)
        if condition.max() > 1:
            condition = condition / 255.0
        if condition.min() >= 0:
            condition = condition * 2.0 - 1.0
        if condition.min() < -1 or condition.max() > 1:
            raise ValueError("LR image values must be in [0, 255], [0, 1], or [-1, 1]")
        return condition

    def __call__(
        self,
        prompt=None,
        negative_prompt='',
        guidance_scale=3.0,
        temperature=1.0,
        complexity_aware_Tmin=10,
        complexity_aware_Tmax=50,
        complexity_aware_k = 0,
        complexity_aware_b = 50,
        complexity_aware_wp = 5,
        first_frame_condition = False,
        first_frame_path = '',
        snr_shift = 1.,
        h_div_w=1.,
        duration=2.,
        generator=None,
        content_type='image',
        seed=None,
        lr_image=None,
        output_type='pil',
        top_k=0,
        top_p=1.0,
        sr_calibrator_kappa=None,
        sr_quality_guidance=None,
        **kwargs
    ):
        if seed is not None:
            self.args.seed = seed
        else:
            self.args.seed = np.random.randint(0, 10000)
        
        self.args.cfg_val = guidance_scale
        self.args.tau = temperature
        self.args.min_infer_steps = complexity_aware_Tmin
        self.args.max_infer_steps = complexity_aware_Tmax
        self.args.complexity_aware_Tmin = complexity_aware_Tmin
        self.args.complexity_aware_Tmax = complexity_aware_Tmax
        self.args.complexity_aware_k = complexity_aware_k
        self.args.complexity_aware_b = complexity_aware_b
        self.args.complexity_aware_wp = complexity_aware_wp
        self.args.snr_shift = snr_shift
        self.args.sr_top_k = top_k
        self.args.sr_top_p = top_p

        lr_condition = None
        semantic_kwargs = {}
        if sr_calibrator_kappa is not None:
            from grn.models.sr_v2_calibrator import resolve_calibrator_kappa
            if not self.args.sr_condition or not hasattr(self.model, 'sr_round_calibrator'):
                raise ValueError('Runtime kappa requires an enabled SR calibrator')
            semantic_kwargs['sr_calibrator_kappa'] = resolve_calibrator_kappa(sr_calibrator_kappa)
        if self.args.sr_condition:
            if lr_image is None:
                raise ValueError("task='SR' requires lr_image")
            lr_condition = self._prepare_lr_condition(lr_image)
            if self.sr_semantic_encoder is not None:
                # Exactly one semantic encoding of the normalized real LR per
                # pipeline call; recurrent GRN refinement reuses these features.
                semantic_kwargs['sr_semantic_features'] = self.sr_semantic_encoder(lr_condition)
            if sr_quality_guidance is not None:
                from grn.models.sr_quality_guidance import SRQualityGuidance, degrade_lr_condition
                if not isinstance(sr_quality_guidance, SRQualityGuidance):
                    raise ValueError('sr_quality_guidance must be an SRQualityGuidance')
                semantic_kwargs['sr_quality_guidance'] = sr_quality_guidance
                if sr_quality_guidance.needs_negative_lr:
                    # One extra LR encoding per image; the 50 rounds reuse it.
                    negative_lr = degrade_lr_condition(lr_condition, sr_quality_guidance)
                    semantic_kwargs['sr_negative_lr_condition'] = negative_lr
                    if self.sr_semantic_encoder is not None:
                        semantic_kwargs['sr_negative_semantic_features'] = self.sr_semantic_encoder(negative_lr)
            h_div_w, content_type, first_frame_condition = 1.0, 'image', False
        elif sr_quality_guidance is not None:
            raise ValueError('sr_quality_guidance is only defined for task SR')
        elif prompt is None:
            raise ValueError("Text-conditioned generation requires prompt")
        
        # Get dynamic resolution meta
        dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(
            self.args.dynamic_scale_schedule,
            self.args.train_h_div_w_list,
            self.args.video_frames
        )
        
        # Get scale schedule based on aspect ratio
        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
        self.args.mapped_h_div_w_template = h_div_w_template_
        
        if content_type == "image":
            num_frames = 1
            duration = 0
        else:
            num_frames = min(self.args.video_frames, int(duration * self.args.video_fps + 1))
        
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][self.args.pn]['pt2scale_schedule'][(num_frames - 1) // 4 + 1]
        
        self.args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
        self.args.tower_split_index = self.args.first_full_spatial_size_scale_index + 1
        
        # Generate content
        generated_image = gen_one_example(
            self.model, self.vae, self.text_tokenizer, self.text_encoder, [prompt or ''],
            negative_prompt=negative_prompt, g_seed=seed, gt_leak=self.args.gt_leak, gt_ls_Bl=None,
            cfg_list=self.args.cfg_val, tau_list=self.args.tau, scale_schedule=scale_schedule,
            cfg_insertion_layer=[self.args.cfg_insertion_layer], vae_latent_dim=self.args.vae_latent_dim,
            args=self.args, get_visual_rope_embeds=self.get_visual_rope_embeds,
            noise_list=None, first_frame_condition=first_frame_condition, first_frame_path=first_frame_path,
            lr_condition=lr_condition,
            **semantic_kwargs,
        )
        
        if len(generated_image.shape) == 3:
            generated_image = generated_image.unsqueeze(0)
        
        generated_image = generated_image.cpu().numpy()

        if output_type == 'numpy':
            # The GRN decoder returns BGR for OpenCV; benchmark writers use RGB.
            return generated_image[..., ::-1].copy()
        if output_type != 'pil':
            raise ValueError(f"output_type must be 'pil' or 'numpy', got {output_type!r}")
        
        ext = '.jpg' if num_frames == 1 else '.mp4'
        
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            output_path = tmp.name
        
        images2video(generated_image, fps=self.args.fps, save_filepath=output_path)
        
        if ext == '.jpg':
            img = Image.open(output_path)
            return type('Result', (object,), {'images': [img]})
        else:
            return type('Result', (object,), {'videos': [output_path]})
