"""Opt-in CLIP semantic prompt for the SR model's native text slot.

GRN is a text-to-image prior: its sequence ends with text tokens (umT5 features,
4096-d, mapped by ``text_proj``) that every image token attends to. The SR
fine-tune fills that slot with one constant empty-prompt token, so the prompt
pathway carries no information. This module puts image semantics back into it:

    CLIP image embedding of the LR  ->  small projector  ->  text-slot tokens

``tokens = 1``   one token: the empty-prompt token plus a learned offset
                 ("direct add"). The sequence length does not change and a
                 freshly initialized projector reproduces the base model exactly.
``tokens = K``   K tokens, each the empty-prompt token plus its own offset.

The projector can also hold one learned *negative* token (the empty-prompt token
plus a free offset), trained to predict a low-quality target. With the sampler's
``prompt`` guidance the second packed branch shares the LR anchor and differs
only in its text slot, so the guidance direction is "what the prompt adds".

Only the projector is trained. The backbone, ``text_proj``, the calibrator and
CLIP stay frozen, so the branch without a semantic prompt is the unchanged base
model. Nothing here runs unless a caller attaches a ``SemanticPrompt`` to the
pipeline or passes text-slot features explicitly.

These are experimental controls. No quality claim is attached to any default.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = 'grn_s5_semantic_prompt_v1'
TEXT_PROMPT_SCHEMA = 'grn_s5_text_prompt_v1'
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
MAX_TEXT_TOKENS = 512
_ARCHITECTURE_KEYS = ('clip_dim', 'text_dim', 'hidden_dim', 'tokens', 'base_tokens')
_CLIP_KEYS = ('backend', 'weight_sha256', 'embed_dim', 'resolution')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value):
    """Content identity of a floating tensor, independent of its device."""
    value = value.detach().to(device='cpu', dtype=torch.float32).contiguous()
    return hashlib.sha256(str(tuple(value.shape)).encode() + value.numpy().tobytes()).hexdigest()


def _positive_int(name, value, maximum=None):
    if type(value) is not int or value < 1 or (maximum is not None and value > maximum):
        raise ValueError(f'{name} must be a positive integer' + ('' if maximum is None else f' <= {maximum}'))
    return value


def _sha256_text(name, value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError(f'{name} must be a lowercase SHA256 digest')
    return value


def _scale(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 16:
        raise ValueError('scale must be a finite scalar in [0, 16]')
    return float(value)


def checked_text_features(value, text_channels=4096):
    """Validate text-slot features [L, text_channels] before they enter the sampler."""
    if (not torch.is_tensor(value) or value.ndim != 2 or not 1 <= value.shape[0] <= MAX_TEXT_TOKENS
            or value.shape[1] != text_channels or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f'Text-slot features must be finite floating [L,{text_channels}], 1 <= L <= {MAX_TEXT_TOKENS}')
    return value


class FrozenCLIPImageEncoder(nn.Module):
    """Frozen CLIP image tower returning the pooled, projected image embedding.

    ``path`` is either an OpenAI CLIP checkpoint file (for example the RN50.pt
    already cached for metrics, or ViT-L-14.pt), loaded with the ``clip``
    package, or a ``transformers`` CLIP directory. Nothing is downloaded.

    Input is finite floating RGB [B,3,H,W] in [-1,1]. It is resized to the
    encoder's square input size (bicubic, antialiased) and normalized with the
    CLIP statistics. Output is detached FP32 [B, embed_dim]. Only the global
    embedding is used: this prompt carries image-level semantics, while local
    structure already comes from the LR anchor.
    """

    def __init__(self, path):
        super().__init__()
        path = Path(path)
        with torch.inference_mode(False), torch.no_grad():
            if path.is_dir():
                weights = [path / name for name in ('model.safetensors', 'pytorch_model.bin') if (path / name).is_file()]
                if not weights:
                    raise FileNotFoundError(f'No CLIP weight file in {path}')
                from transformers import CLIPVisionModelWithProjection
                model = CLIPVisionModelWithProjection.from_pretrained(str(path), local_files_only=True)
                self.backend = 'transformers'
                self.resolution = int(model.config.image_size)
                self.embed_dim = int(model.visual_projection.weight.shape[0])
                self.weight_sha256 = file_sha256(weights[0])
            elif path.is_file():
                import clip
                full, _ = clip.load(str(path), device='cpu', jit=False)
                model = full.visual
                self.backend = 'openai'
                self.resolution = int(model.input_resolution)
                self.embed_dim = int(model.output_dim)
                self.weight_sha256 = file_sha256(path)
            else:
                raise FileNotFoundError(f'CLIP weights not found: {path}')
            self.model = model.float()
            self.register_buffer('rgb_mean', torch.tensor(CLIP_MEAN, dtype=torch.float32).reshape(1, 3, 1, 1))
            self.register_buffer('rgb_std', torch.tensor(CLIP_STD, dtype=torch.float32).reshape(1, 3, 1, 1))
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # Permanently frozen: batch-norm statistics in RN towers must not move.
        return super().train(False)

    def identity(self):
        return {'backend': self.backend, 'weight_sha256': self.weight_sha256,
                'embed_dim': self.embed_dim, 'resolution': self.resolution}

    def forward(self, image):
        if (not torch.is_tensor(image) or image.ndim != 4 or image.shape[0] < 1 or image.shape[1] != 3
                or min(image.shape[-2:]) < 1 or not image.is_floating_point()):
            raise ValueError('CLIP input must be floating RGB [B,3,H,W]')
        if not bool(torch.isfinite(image).all()) or bool(image.min() < -1) or bool(image.max() > 1):
            raise ValueError('CLIP input must be finite and within [-1,1]')
        self.train(False)
        size = (self.resolution, self.resolution)
        with torch.inference_mode(False), torch.no_grad(), torch.autocast(device_type=image.device.type, enabled=False):
            rgb = (image.float() + 1.0) * 0.5
            if tuple(rgb.shape[-2:]) != size:
                rgb = F.interpolate(rgb, size=size, mode='bicubic', align_corners=False, antialias=True).clamp(0.0, 1.0)
            rgb = (rgb - self.rgb_mean) / self.rgb_std
            if self.backend == 'transformers':
                embedding = self.model(pixel_values=rgb).image_embeds
            else:
                embedding = self.model(rgb)
            embedding = embedding.float()
            if embedding.shape != (image.shape[0], self.embed_dim) or not bool(torch.isfinite(embedding).all()):
                raise RuntimeError('CLIP produced an unexpected or nonfinite embedding')
        return embedding


class SemanticPromptProjector(nn.Module):
    """CLIP image embedding -> text-slot tokens, plus one learned negative token.

    restore   residual MLP that maps the LR embedding toward the embedding CLIP
              would give the HR image (trained by an optional alignment loss).
    tokens    MLP from the restored embedding to ``tokens`` offsets in the text
              feature space. Its last layer starts at zero.
    negative  a free offset on the empty-prompt token, also starting at zero.

    Offsets are expressed in units of the empty-prompt feature RMS, so learning
    rates do not depend on the text encoder's output scale. All math is FP32.
    Every parameter is initialized explicitly: GRN entry points may disable the
    default ``reset_parameters`` of ``nn.Linear``.
    """

    def __init__(self, *, clip_dim, text_dim=4096, hidden_dim=1024, tokens=1, base_tokens=1):
        super().__init__()
        self.clip_dim = _positive_int('clip_dim', clip_dim)
        self.text_dim = _positive_int('text_dim', text_dim)
        self.hidden_dim = _positive_int('hidden_dim', hidden_dim)
        self.tokens = _positive_int('tokens', tokens, 64)
        self.base_tokens = _positive_int('base_tokens', base_tokens, MAX_TEXT_TOKENS)
        self.restore_in = nn.Linear(clip_dim, hidden_dim)
        self.restore_out = nn.Linear(hidden_dim, clip_dim)
        self.token_in = nn.Linear(clip_dim, hidden_dim)
        self.token_out = nn.Linear(hidden_dim, tokens * text_dim)
        self.negative_delta = nn.Parameter(torch.zeros(base_tokens, text_dim))
        for layer in (self.restore_in, self.token_in):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in (self.restore_out, self.token_out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def architecture(self):
        return {key: getattr(self, key) for key in _ARCHITECTURE_KEYS}

    def _embedding(self, name, embedding):
        if (not torch.is_tensor(embedding) or embedding.ndim != 2 or embedding.shape[0] < 1
                or embedding.shape[1] != self.clip_dim or not embedding.is_floating_point()
                or not bool(torch.isfinite(embedding).all())):
            raise ValueError(f'{name} must be finite floating [B,{self.clip_dim}]')
        # Unit direction, rescaled so each element is O(1) for the first layer.
        return F.normalize(embedding.float(), dim=-1, eps=1e-6) * math.sqrt(self.clip_dim)

    def _base(self, base):
        if (not torch.is_tensor(base) or tuple(base.shape) != (self.base_tokens, self.text_dim)
                or not base.is_floating_point() or not bool(torch.isfinite(base).all())):
            raise ValueError(f'Base text features must be finite floating [{self.base_tokens},{self.text_dim}]')
        base = base.detach().float()
        return base, base.square().mean().sqrt().clamp_min(1e-6)

    def restored(self, embedding):
        """HR-like embedding [B, clip_dim] predicted from the LR embedding."""
        with torch.autocast(device_type=embedding.device.type, enabled=False):
            value = self._embedding('CLIP embedding', embedding)
            return value + self.restore_out(F.silu(self.restore_in(value)))

    def offsets(self, embedding):
        """Token offsets [B, tokens, text_dim] in units of the base feature RMS."""
        with torch.autocast(device_type=embedding.device.type, enabled=False):
            hidden = F.silu(self.token_in(self.restored(embedding)))
            return self.token_out(hidden).reshape(embedding.shape[0], self.tokens, self.text_dim)

    def alignment_loss(self, lr_embedding, hr_embedding):
        """1 - cosine between the restored LR embedding and the HR embedding."""
        with torch.autocast(device_type=lr_embedding.device.type, enabled=False):
            target = self._embedding('HR CLIP embedding', hr_embedding).detach()
            return (1.0 - F.cosine_similarity(self.restored(lr_embedding), target, dim=-1)).mean()

    def positive_text(self, embedding, base, *, scale=1.0):
        """Text-slot features [max(tokens, base_tokens), text_dim] for one image."""
        scale = _scale(scale)
        if not torch.is_tensor(embedding) or embedding.shape[:1] != (1,):
            raise ValueError('positive_text expects the embedding of exactly one image')
        base, rms = self._base(base)
        with torch.autocast(device_type=base.device.type, enabled=False):
            offset = self.offsets(embedding)[0]
            count = max(self.tokens, self.base_tokens)
            index = torch.arange(count, device=base.device).clamp_max(self.base_tokens - 1)
            if self.tokens < count:
                offset = torch.cat((offset, offset.new_zeros(count - self.tokens, self.text_dim)))
            result = base.index_select(0, index) + scale * rms * offset
            if not bool(torch.isfinite(result).all()):
                raise FloatingPointError('Semantic prompt tokens are nonfinite')
        return result

    def negative_text(self, base):
        """The learned negative token(s) [base_tokens, text_dim]."""
        base, rms = self._base(base)
        with torch.autocast(device_type=base.device.type, enabled=False):
            result = base + rms * self.negative_delta
            if not bool(torch.isfinite(result).all()):
                raise FloatingPointError('Negative prompt tokens are nonfinite')
        return result


class SemanticPrompt:
    """Inference bundle: frozen CLIP encoder, trained projector, empty-prompt features."""

    def __init__(self, encoder, projector, empty_features, *, negative_trained=False):
        if not isinstance(projector, SemanticPromptProjector):
            raise ValueError('projector must be a SemanticPromptProjector')
        if getattr(encoder, 'embed_dim', None) != projector.clip_dim:
            raise ValueError('CLIP embedding size differs from the projector input')
        if type(negative_trained) is not bool:
            raise ValueError('negative_trained must be an explicit bool')
        projector._base(empty_features)
        self.encoder, self.projector = encoder, projector
        self.empty = empty_features.detach().float()
        self.negative_trained = negative_trained
        self.projector.requires_grad_(False)
        self.projector.eval()

    def to(self, device):
        self.encoder.to(device=device)
        self.projector.to(device=device)
        self.empty = self.empty.to(device=device)
        return self

    def text_features(self, lr_condition, *, scale=1.0):
        """(positive [K,T], learned negative [L,T] or None) for one LR image in [-1,1]."""
        with torch.inference_mode(False), torch.no_grad():
            embedding = self.encoder(lr_condition)
            positive = self.projector.positive_text(embedding, self.empty, scale=scale)
            negative = self.projector.negative_text(self.empty) if self.negative_trained else None
        return positive, negative


def save_semantic_prompt(path, projector, *, base_sha256, adapter_sha256, clip_identity,
                         empty_text_sha256, negative_trained, step, metadata=None):
    """Save only the projector, bound to the base, the C/R adapter, CLIP and the empty prompt."""
    if not isinstance(projector, SemanticPromptProjector):
        raise ValueError('No semantic prompt projector to save')
    if type(step) is not int or step < 0 or type(negative_trained) is not bool:
        raise ValueError('step must be a nonnegative integer and negative_trained a bool')
    if type(clip_identity) is not dict or set(clip_identity) != set(_CLIP_KEYS):
        raise ValueError('clip_identity must come from FrozenCLIPImageEncoder.identity()')
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Refusing to overwrite existing checkpoint: {path}')
    state = {key: value.detach().cpu().clone() for key, value in projector.state_dict().items()}
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError('Cannot save nonfinite projector parameters')
    payload = {'schema': SCHEMA, 'base_sha256': _sha256_text('base SHA256', base_sha256),
               'adapter_sha256': None if adapter_sha256 is None else _sha256_text('adapter SHA256', adapter_sha256),
               'clip': dict(clip_identity),
               'empty_text_sha256': _sha256_text('empty text SHA256', empty_text_sha256),
               'architecture': projector.architecture(), 'state_dict': state,
               'negative_trained': negative_trained, 'step': step,
               'metadata': {} if metadata is None else metadata}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    return {'path': str(path), 'step': step}


def load_semantic_prompt(path, *, expected_base_sha256, expected_adapter_sha256, clip_identity, empty_features):
    """Validate every binding and tensor, then return (frozen projector, report)."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    keys = {'schema', 'base_sha256', 'adapter_sha256', 'clip', 'empty_text_sha256', 'architecture',
            'state_dict', 'negative_trained', 'step', 'metadata'}
    if type(payload) is not dict or set(payload) != keys or payload['schema'] != SCHEMA:
        raise ValueError('Unsupported semantic prompt checkpoint')
    if payload['base_sha256'] != _sha256_text('expected base SHA256', expected_base_sha256):
        raise ValueError('Semantic prompt belongs to a different base checkpoint')
    if payload['adapter_sha256'] != expected_adapter_sha256:
        raise ValueError('Semantic prompt was trained beside a different C/R adapter')
    saved_clip = payload['clip']
    if (type(saved_clip) is not dict or type(clip_identity) is not dict
            or any(saved_clip.get(key) != clip_identity.get(key) for key in ('weight_sha256', 'embed_dim', 'resolution'))):
        raise ValueError('Semantic prompt was trained with different CLIP weights')
    if payload['empty_text_sha256'] != tensor_sha256(empty_features):
        raise ValueError('Semantic prompt was trained on a different empty-prompt feature')
    architecture = payload['architecture']
    if type(architecture) is not dict or set(architecture) != set(_ARCHITECTURE_KEYS):
        raise ValueError('Invalid semantic prompt architecture')
    if architecture['clip_dim'] != clip_identity['embed_dim']:
        raise ValueError('Projector input size differs from the CLIP embedding')
    if type(payload['negative_trained']) is not bool or type(payload['step']) is not int:
        raise ValueError('Invalid semantic prompt metadata')
    with torch.random.fork_rng(devices=[]):
        projector = SemanticPromptProjector(**architecture)
    reference, state = projector.state_dict(), payload['state_dict']
    if type(state) is not dict or set(state) != set(reference):
        raise ValueError('Semantic prompt state membership differs')
    for key, value in state.items():
        if (not torch.is_tensor(value) or value.shape != reference[key].shape
                or value.dtype != reference[key].dtype or not bool(torch.isfinite(value).all())):
            raise ValueError(f'Invalid semantic prompt tensor: {key}')
    projector.load_state_dict(state, strict=True)
    projector.requires_grad_(False)
    projector.eval()
    return projector, {'step': payload['step'], 'negative_trained': payload['negative_trained'],
                       'architecture': architecture, 'clip': saved_clip, 'metadata': payload['metadata']}


def text_prompt_payload(prompt, features, umt5_revision, dependency_hashes, text_channels=4096):
    """Cache record for one real umT5-encoded prompt (see tools/prepare_sr_text_prompt.py)."""
    if type(prompt) is not str or not prompt.strip():
        raise ValueError('A cached text prompt must be a nonempty string')
    features = checked_text_features(features, text_channels)
    return {'schema': TEXT_PROMPT_SCHEMA, 'prompt': prompt,
            'features': features.detach().float().cpu().contiguous(),
            'umt5_revision': umt5_revision, 'dependency_hashes': dependency_hashes}


def load_text_prompt_cache(path, text_channels=4096):
    """Return (features [L, text_channels], prompt) from a cached umT5 prompt."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    required = {'schema', 'prompt', 'features', 'umt5_revision', 'dependency_hashes'}
    if type(payload) is not dict or not required.issubset(payload) or payload['schema'] != TEXT_PROMPT_SCHEMA:
        raise ValueError('Unsupported text prompt cache')
    if type(payload['prompt']) is not str or not payload['prompt'].strip():
        raise ValueError('Text prompt cache has no prompt')
    features = checked_text_features(payload['features'], text_channels)
    return features.detach().float().contiguous(), payload['prompt']
