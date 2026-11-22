"""
adapters.py — ModelAdapter: vision-language-model access layer.
==============================================================
Every place that used to reach into OpenAI-CLIP internals
(``model.visual.transformer.resblocks`` …) goes through a ModelAdapter
instead, so a new architecture = one new adapter class.

Three implementations:
  * OpenAICLIPAdapter — OpenAI ``clip`` package (ViT-B/16, ViT-L/14, …)
  * SigLIPAdapter    — HuggingFace ``transformers`` SigLIP
    (google/siglip-base-patch16-224, google/siglip-so400m-patch14-384)
  * BlipAdapter      — HuggingFace ``transformers`` BLIP-1
    (Salesforce/blip-itm-base-coco, ``BlipForImageTextRetrieval``;
    BLIP-2/Q-Former is out of scope — BLIP.md §4)
plus a fourth that reuses the OpenAI block layout:
  * EVACLIPAdapter   — EVA-CLIP / OpenCLIP ViT (``EVA_CLIP_g_14``);
    BLIP-2's actual vision tower, so attacking it is what makes the
    ``vlm_transfer`` tower swap a 1:1 replacement

Design notes:
  * ``encode_image`` / ``encode_text`` return RAW (un-normalised) features —
    L2-normalisation stays at the call sites (existing convention).
  * Weight access is slice-aware: ``get_component_refs`` returns
    ``(key, param, quant_name, row_slice, col_slice)`` because the Phase-0
    int8 grid is per-tensor and gradients are read from the *parameter*
    (views of a leaf param do not carry ``.grad``).  For OpenAI CLIP's fused
    ``in_proj_weight`` the V slice lives at rows ``[2D+lo, 2D+hie)``; for
    SigLIP's separate ``v_proj`` it is ``[lo, hie)`` — callers never see the
    difference.
  * ``get_seq_layout`` reports the TRUE input resolution and the number of
    prefix tokens (OpenAI CLIP/BLIP: 1 CLS; SigLIP: 0, mean-pooled patches).
  * ``replace_sublayer`` supports Phase-2A mean ablation: OpenAI CLIP calls
    attention via the ``attention`` *method* (instance-attribute binding
    trick), SigLIP calls the ``self_attn`` *module* (plain module swap).

Factory: ``get_adapter(model)`` dispatches on the model class and caches
adapters; ``load_model(cfg)`` loads (model, preprocess) for the registry
entry ``cfg.MODEL_ADAPTER``.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager

import torch
import torch.nn as nn

from utils import CLIP_MEAN, CLIP_STD


class ModelAdapter(ABC):
    """Abstract handler for vision-language models."""

    # ════════════════════════════════════════════════════════════════
    #  STRUCTURE
    # ════════════════════════════════════════════════════════════════
    @abstractmethod
    def get_vision_transformer_blocks(self, model) -> nn.ModuleList:
        """Return the list of transformer blocks in the vision encoder."""

    def get_vision_module(self, model):
        """Return the vision tower module (for parameter counting etc.)."""
        return None

    @abstractmethod
    def get_num_heads(self, block) -> int:
        """Return the number of attention heads in a block."""

    @abstractmethod
    def get_hidden_dim(self, model) -> int:
        """Return the hidden dimension of the vision encoder."""

    @abstractmethod
    def get_patch_embedding(self, model) -> nn.Module:
        """Return the patch embedding layer (conv or linear)."""

    @abstractmethod
    def get_seq_layout(self, model) -> dict:
        """Sequence geometry: width, n_layers, n_heads, seq_len, n_patches,
        n_prefix (CLS tokens), patch_size, patches_per_side, img_size
        (TRUE input resolution — may differ from patches_per_side*patch_size,
        e.g. SigLIP-so400m: 384 vs 27*14=378)."""

    # ════════════════════════════════════════════════════════════════
    #  WEIGHTS
    # ════════════════════════════════════════════════════════════════
    @abstractmethod
    def get_attention_weights(self, block) -> dict:
        """Return a dict with keys 'q', 'k', 'v', 'out' mapping to weight
        tensors (views are fine — for read-only use / knockouts)."""

    @abstractmethod
    def get_mlp_weights(self, block) -> dict:
        """Return a dict with keys 'fc1', 'fc2' mapping to weight tensors."""

    @abstractmethod
    def get_block_quant_params(self, block) -> list:
        """Phase-0 enumeration: [(quant_name, param), …] for every tensor of
        the block that gets int8-quantised (attn weights + biases, layer
        norms, MLP weights + biases).  quant_name is the canonical suffix —
        the Quantizer prefixes it with 'L{li}_'.

        Phase-0 semantics: quantise each listed tensor as ONE int8 grid
        (per-tensor scale); slices are never re-quantised."""

    @abstractmethod
    def get_component_refs(self, block, comp) -> list:
        """Slice-aware weight refs for one component (Phase 2B output dict:
        {'type': 'attn'|'ffn', 'layer': li, 'head': h}).

        Returns [(key, param, quant_name, row_slice, col_slice), …] with
        key 'V'/'O' for attn heads and 'fc'/'proj' for ffn components.
        ``quant_name`` must match get_block_quant_params so the Phase-0 int8
        grid can be sliced with the same row/col slices."""

    # ════════════════════════════════════════════════════════════════
    #  FORWARD
    # ════════════════════════════════════════════════════════════════
    @abstractmethod
    def encode_image(self, model, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into RAW features (no L2 normalisation)
        in the joint embedding space (projection included)."""

    @abstractmethod
    def encode_text(self, model, texts, device) -> torch.Tensor:
        """Encode a list of prompt strings into RAW text features
        (tokenisation included)."""

    @abstractmethod
    def get_preprocess(self, model):
        """Return the PIL→tensor preprocessing callable for this model."""

    @abstractmethod
    def get_norm_stats(self, model):
        """Return (mean, std) normalisation constants — the valid normalised
        pixel range used by apply_trigger's clamp is [(0-mean)/std, (1-mean)/std]."""

    # ════════════════════════════════════════════════════════════════
    #  SUB-LAYER SURGERY (Phase 2A mean ablation)
    # ════════════════════════════════════════════════════════════════
    @abstractmethod
    def get_sublayer_modules(self, block) -> dict:
        """Hook targets: {'attn': module, 'mlp': module} (for
        compute_sub_layer_means forward hooks)."""

    @abstractmethod
    @contextmanager
    def replace_sublayer(self, block, kind, const_out):
        """Context manager replacing the 'attn'/'mlp' sub-layer of ``block``
        with ``const_out`` (an nn.Module returning a constant) and restoring
        it afterwards."""


# ══════════════════════════════════════════════════════════════════════
#  OPENAI CLIP
# ══════════════════════════════════════════════════════════════════════
class OpenAICLIPAdapter(ModelAdapter):
    """Adapter for the OpenAI ``clip`` package (ViT-B/16, ViT-L/14, …).

    Attention uses a fused ``in_proj_weight`` (3D × D rows: Q, K, V stacked);
    the V slice of head h therefore lives at rows [2D + h*dk, 2D + (h+1)*dk).
    The transformer runs LND (seq-first) and has one CLS token."""

    # ── structure ──
    def get_vision_transformer_blocks(self, model):
        return model.visual.transformer.resblocks

    def get_vision_module(self, model):
        return model.visual

    def get_num_heads(self, block):
        return block.attn.num_heads

    def get_hidden_dim(self, model):
        return model.visual.transformer.width

    def get_patch_embedding(self, model):
        return model.visual.conv1

    def get_seq_layout(self, model):
        vis = model.visual
        pos = vis.positional_embedding            # [seq_len, width]
        seq_len = pos.shape[0]
        width = pos.shape[1]
        n_prefix = 1                              # CLS token
        n_patches = seq_len - n_prefix
        patch_size = vis.conv1.kernel_size[0]
        patches_per_side = int(n_patches ** 0.5)
        blocks = list(vis.transformer.resblocks)
        return dict(
            seq_len=seq_len, width=width,
            n_patches=n_patches, n_prefix=n_prefix,
            patch_size=patch_size, patches_per_side=patches_per_side,
            img_size=vis.input_resolution,
            n_layers=len(blocks), n_heads=blocks[0].attn.num_heads)

    # ── weights ──
    def get_attention_weights(self, block):
        # fused in_proj_weight for Q, K, V
        w = block.attn.in_proj_weight
        d = w.shape[0] // 3
        return {
            'q': w[0:d], 'k': w[d:2 * d], 'v': w[2 * d:],
            'out': block.attn.out_proj.weight,
        }

    def get_mlp_weights(self, block):
        return {'fc1': block.mlp.c_fc.weight, 'fc2': block.mlp.c_proj.weight}

    def get_block_quant_params(self, block):
        return [
            ("attn_ip_w", block.attn.in_proj_weight),
            ("attn_ip_b", block.attn.in_proj_bias),
            ("attn_op_w", block.attn.out_proj.weight),
            ("attn_op_b", block.attn.out_proj.bias),
            ("ln1_w", block.ln_1.weight), ("ln1_b", block.ln_1.bias),
            ("ln2_w", block.ln_2.weight), ("ln2_b", block.ln_2.bias),
            ("fc1_w", block.mlp.c_fc.weight), ("fc1_b", block.mlp.c_fc.bias),
            ("fc2_w", block.mlp.c_proj.weight), ("fc2_b", block.mlp.c_proj.bias),
        ]

    def get_component_refs(self, block, comp):
        D = self.get_hidden_dim_from_block(block)
        n_heads = self.get_num_heads(block)
        dk = D // n_heads
        if comp["type"] == "attn":
            hi = comp["head"]
            lo, hie = hi * dk, (hi + 1) * dk
            return [
                ("V", block.attn.in_proj_weight, "attn_ip_w",
                 slice(2 * D + lo, 2 * D + hie), slice(None)),
                ("O", block.attn.out_proj.weight, "attn_op_w",
                 slice(None), slice(lo, hie)),
            ]
        return [
            ("fc", block.mlp.c_fc.weight, "fc1_w",
             slice(None), slice(None)),
            ("proj", block.mlp.c_proj.weight, "fc2_w",
             slice(None), slice(None)),
        ]

    # ── forward ──
    def encode_image(self, model, images):
        return model.encode_image(images)

    def encode_text(self, model, texts, device):
        import clip
        tok = clip.tokenize(list(texts)).to(device)
        return model.encode_text(tok)

    def get_preprocess(self, model):
        import clip
        return clip._transform(model.visual.input_resolution)

    def get_norm_stats(self, model):
        return CLIP_MEAN, CLIP_STD

    # ── sub-layer surgery ──
    def get_sublayer_modules(self, block):
        return {"attn": block.attn, "mlp": block.mlp}

    @contextmanager
    def replace_sublayer(self, block, kind, const_out):
        if kind == "mlp":
            orig = block.mlp
            block.mlp = const_out
            try:
                yield
            finally:
                block.mlp = orig
        else:
            # OpenAI CLIP calls attention via the ``attention`` *method*;
            # binding const_out.forward as an instance attribute shadows the
            # class method (assigning an nn.Module would silently land in
            # _modules and be shadowed).  Removing the binding restores the
            # normal lookup.
            sentinel = object()
            prev = block.__dict__.get("attention", sentinel)
            block.attention = const_out.forward
            try:
                yield
            finally:
                if prev is sentinel:
                    block.__dict__.pop("attention", None)
                else:
                    block.__dict__["attention"] = prev

    # ── helper: hidden dim of the block's own projections (D) ──
    def get_hidden_dim_from_block(self, block):
        # fused in_proj_weight is [3D, D]
        return block.attn.in_proj_weight.shape[1]


# ══════════════════════════════════════════════════════════════════════
#  SIGLIP (HuggingFace transformers)
# ══════════════════════════════════════════════════════════════════════
class _AttnConst(nn.Module):
    """Wraps a constant-output module for transformers-style attention
    replacement (SigLIP, BLIP): accepts ``hidden_states=``/``attention_mask=``
    kwargs and returns the (hidden, weights) 2-tuple the encoder unpacks."""

    def __init__(self, const_out):
        super().__init__()
        self.const_out = const_out

    def forward(self, x=None, hidden_states=None, attention_mask=None, **kw):
        t = hidden_states if hidden_states is not None else x
        return (self.const_out(t), None)


class SigLIPAdapter(ModelAdapter):
    """Adapter for HuggingFace SigLIP (google/siglip-*).

    Separate q/k/v/out projections, pre-LN blocks (layer_norm1/2), MLP
    fc1/fc2, NO CLS token (patch tokens are mean-pooled), input resolution
    from vision config (224 or 384).  ``encode_image`` uses
    ``get_image_features`` so the visual projection is applied and the
    features live in the text-feature space."""

    _PROCESSOR_CACHE = {}

    # ── processor (tokenizer + image processor) ──
    def _processor(self, model):
        cached = getattr(model, "_adapter_processor", None)
        if cached is not None:
            return cached
        name = model.config._name_or_path
        if name not in self._PROCESSOR_CACHE:
            from transformers import AutoProcessor
            self._PROCESSOR_CACHE[name] = AutoProcessor.from_pretrained(name)
        proc = self._PROCESSOR_CACHE[name]
        try:
            model._adapter_processor = proc
        except Exception:
            pass
        return proc

    # ── structure ──
    def get_vision_transformer_blocks(self, model):
        return model.vision_model.encoder.layers

    def get_vision_module(self, model):
        return model.vision_model

    def get_num_heads(self, block):
        return block.self_attn.num_heads

    def get_hidden_dim(self, model):
        return model.vision_model.config.hidden_size

    def get_patch_embedding(self, model):
        return model.vision_model.embeddings.patch_embedding

    def get_seq_layout(self, model):
        vis = model.vision_model
        emb = vis.embeddings
        n_patches = emb.position_embedding.num_embeddings
        n_prefix = 0                          # SigLIP has no CLS token
        seq_len = n_patches + n_prefix
        patch_size = emb.patch_embedding.kernel_size[0]
        patches_per_side = int(round(n_patches ** 0.5))
        blocks = list(vis.encoder.layers)
        return dict(
            seq_len=seq_len, width=vis.config.hidden_size,
            n_patches=n_patches, n_prefix=n_prefix,
            patch_size=patch_size, patches_per_side=patches_per_side,
            img_size=vis.config.image_size,
            n_layers=len(blocks), n_heads=blocks[0].self_attn.num_heads)

    # ── weights ──
    def get_attention_weights(self, block):
        a = block.self_attn
        return {'q': a.q_proj.weight, 'k': a.k_proj.weight,
                'v': a.v_proj.weight, 'out': a.out_proj.weight}

    def get_mlp_weights(self, block):
        return {'fc1': block.mlp.fc1.weight, 'fc2': block.mlp.fc2.weight}

    def get_block_quant_params(self, block):
        a, m = block.self_attn, block.mlp
        return [
            ("attn_q_w", a.q_proj.weight), ("attn_q_b", a.q_proj.bias),
            ("attn_k_w", a.k_proj.weight), ("attn_k_b", a.k_proj.bias),
            ("attn_v_w", a.v_proj.weight), ("attn_v_b", a.v_proj.bias),
            ("attn_op_w", a.out_proj.weight), ("attn_op_b", a.out_proj.bias),
            ("ln1_w", block.layer_norm1.weight), ("ln1_b", block.layer_norm1.bias),
            ("ln2_w", block.layer_norm2.weight), ("ln2_b", block.layer_norm2.bias),
            ("fc1_w", m.fc1.weight), ("fc1_b", m.fc1.bias),
            ("fc2_w", m.fc2.weight), ("fc2_b", m.fc2.bias),
        ]

    def get_component_refs(self, block, comp):
        D = block.self_attn.out_proj.weight.shape[0]
        n_heads = self.get_num_heads(block)
        dk = D // n_heads
        a = block.self_attn
        if comp["type"] == "attn":
            hi = comp["head"]
            lo, hie = hi * dk, (hi + 1) * dk
            return [
                ("V", a.v_proj.weight, "attn_v_w",
                 slice(lo, hie), slice(None)),
                ("O", a.out_proj.weight, "attn_op_w",
                 slice(None), slice(lo, hie)),
            ]
        return [
            ("fc", block.mlp.fc1.weight, "fc1_w",
             slice(None), slice(None)),
            ("proj", block.mlp.fc2.weight, "fc2_w",
             slice(None), slice(None)),
        ]

    # ── forward ──
    @staticmethod
    def _pooled(out):
        """transformers v5 returns BaseModelOutputWithPooling from
        get_*_features; the contrastive feature is ``pooler_output``
        (SiglipModel.forward uses exactly this)."""
        if hasattr(out, "pooler_output"):
            return out.pooler_output
        if isinstance(out, (tuple, list)):
            return out[1] if len(out) > 1 else out[0]
        return out

    def encode_image(self, model, images):
        return self._pooled(model.get_image_features(pixel_values=images))

    def encode_text(self, model, texts, device):
        proc = self._processor(model)
        tok = proc.tokenizer(list(texts), padding="max_length",
                             truncation=True,
                             max_length=proc.tokenizer.model_max_length,
                             return_tensors="pt")
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        out = model.get_text_features(input_ids=input_ids,
                                      attention_mask=attention_mask)
        return self._pooled(out)

    def get_preprocess(self, model):
        image_processor = self._processor(model).image_processor

        def preprocess(pil_img):
            out = image_processor(images=pil_img, return_tensors="pt")
            return out["pixel_values"][0]
        return preprocess

    def get_norm_stats(self, model):
        try:
            ip = self._processor(model).image_processor
            return list(ip.image_mean), list(ip.image_std)
        except Exception:
            return [0.5] * 3, [0.5] * 3

    # ── sub-layer surgery ──
    def get_sublayer_modules(self, block):
        return {"attn": block.self_attn, "mlp": block.mlp}

    @contextmanager
    def replace_sublayer(self, block, kind, const_out):
        # SigLIP calls self_attn / mlp as *modules* — plain module swap works
        # for both.  The attention module returns a 2-tuple (hidden, weights)
        # which the encoder unpacks (``hidden_states, _ = self.self_attn(…)``),
        # so the attn replacement wraps the constant in a tuple-returning
        # module and must accept ``hidden_states=`` / ``attention_mask=`` kwargs.
        if kind == "mlp":
            orig = block.mlp
            block.mlp = const_out
            try:
                yield
            finally:
                block.mlp = orig
        else:
            orig = block.self_attn
            block.self_attn = _AttnConst(const_out)
            try:
                yield
            finally:
                block.self_attn = orig


# ══════════════════════════════════════════════════════════════════════
#  BLIP-1 (HuggingFace transformers)
# ══════════════════════════════════════════════════════════════════════
class BlipAdapter(ModelAdapter):
    """Adapter for HuggingFace BLIP-1 (``BlipModel`` or
    ``BlipForImageTextRetrieval``, e.g. ``Salesforce/blip-itm-base-coco``).
    BLIP-2/Q-Former models are out of scope (BLIP.md §4).

    ITC feature convention (BLIP.md §2/§7): ``encode_image``/``encode_text``
    return the projected joint-space features ``vision_proj(vision CLS)`` /
    ``text_proj(text [CLS])`` — the RAW ``[CLS]`` of ``last_hidden_state``
    (vision: after one ``post_layernorm``), exactly the convention of
    ``BlipForImageTextRetrieval`` and the Salesforce reference.  The
    convenience ``BlipModel.get_image_features``/``get_text_features`` are
    deliberately NOT used: the text one routes through the BERT tanh pooler
    and the image one applies ``post_layernorm`` twice — neither matches the
    trained ITC projections.

    Vision tower: timm-style ViT with a fused ``qkv`` Linear ([3D, D] rows
    Q, K, V stacked — same V-slice convention as OpenAI CLIP's
    ``in_proj_weight``), out-projection named ``projection``, pre-LN
    ``BlipEncoderLayer`` (``self_attn``/``layer_norm1``/``mlp``/``layer_norm2``).
    The attention module returns a ``(hidden, weights)`` 2-tuple like SigLIP,
    so the shared ``_AttnConst`` stands in for it in Phase-2A ablation.

    Also accepts ``BlipForImageTextRetrieval`` checkpoints (attribute names
    ``vision_proj``/``text_proj``/``text_encoder``)."""

    _PROCESSOR_CACHE = {}

    # ── processor (tokenizer + image processor) ──
    def _processor(self, model):
        cached = getattr(model, "_adapter_processor", None)
        if cached is not None:
            return cached
        name = model.config._name_or_path
        if name not in self._PROCESSOR_CACHE:
            from transformers import AutoProcessor
            self._PROCESSOR_CACHE[name] = AutoProcessor.from_pretrained(name)
        proc = self._PROCESSOR_CACHE[name]
        try:
            model._adapter_processor = proc
        except Exception:
            pass
        return proc

    # ── ITC heads (BlipModel / BlipForImageTextRetrieval name fallbacks) ──
    @staticmethod
    def _itc_heads(model):
        """(vision_proj, text_model, text_proj) — the joint-space projections
        and the (unimodal) text encoder, whichever name this checkpoint uses."""
        vis_proj = getattr(model, "visual_projection", None) \
            or getattr(model, "vision_proj", None)
        txt_model = getattr(model, "text_model", None) \
            or getattr(model, "text_encoder", None)
        txt_proj = getattr(model, "text_projection", None) \
            or getattr(model, "text_proj", None)
        if vis_proj is None or txt_model is None or txt_proj is None:
            raise ValueError(
                "BlipAdapter needs BlipModel (visual_projection/text_model/"
                f"text_projection); got {type(model).__name__} — BLIP-2/"
                "Q-Former models are out of scope (BLIP.md §4)")
        return vis_proj, txt_model, txt_proj

    @staticmethod
    def _cls(out):
        """Raw [CLS] of the encoder output (tuple- or ModelOutput-style)."""
        hs = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return hs[:, 0, :]

    # ── structure ──
    def get_vision_transformer_blocks(self, model):
        return model.vision_model.encoder.layers

    def get_vision_module(self, model):
        return model.vision_model

    def get_num_heads(self, block):
        return block.self_attn.num_heads

    def get_hidden_dim(self, model):
        return model.vision_model.config.hidden_size

    def get_patch_embedding(self, model):
        return model.vision_model.embeddings.patch_embedding

    def get_seq_layout(self, model):
        vis = model.vision_model
        emb = vis.embeddings
        n_prefix = 1                          # class_embedding (CLS token)
        n_patches = emb.num_patches
        seq_len = n_patches + n_prefix
        patch_size = emb.patch_size
        patches_per_side = int(round(n_patches ** 0.5))
        blocks = list(vis.encoder.layers)
        return dict(
            seq_len=seq_len, width=vis.config.hidden_size,
            n_patches=n_patches, n_prefix=n_prefix,
            patch_size=patch_size, patches_per_side=patches_per_side,
            img_size=emb.image_size,
            n_layers=len(blocks), n_heads=blocks[0].self_attn.num_heads)

    # ── weights ──
    def get_attention_weights(self, block):
        # fused qkv for Q, K, V (rows [3D, D]: Q, K, V stacked)
        w = block.self_attn.qkv.weight
        d = w.shape[0] // 3
        return {
            'q': w[0:d], 'k': w[d:2 * d], 'v': w[2 * d:],
            'out': block.self_attn.projection.weight,
        }

    def get_mlp_weights(self, block):
        return {'fc1': block.mlp.fc1.weight, 'fc2': block.mlp.fc2.weight}

    def get_block_quant_params(self, block):
        # Canonical names = OpenAICLIPAdapter's (fused qkv ↔ in_proj,
        # projection ↔ out_proj) so sub_quant keys line up across models.
        a, m = block.self_attn, block.mlp
        return [
            ("attn_ip_w", a.qkv.weight), ("attn_ip_b", a.qkv.bias),
            ("attn_op_w", a.projection.weight), ("attn_op_b", a.projection.bias),
            ("ln1_w", block.layer_norm1.weight), ("ln1_b", block.layer_norm1.bias),
            ("ln2_w", block.layer_norm2.weight), ("ln2_b", block.layer_norm2.bias),
            ("fc1_w", m.fc1.weight), ("fc1_b", m.fc1.bias),
            ("fc2_w", m.fc2.weight), ("fc2_b", m.fc2.bias),
        ]

    def get_component_refs(self, block, comp):
        D = self.get_hidden_dim_from_block(block)
        n_heads = self.get_num_heads(block)
        dk = D // n_heads
        if comp["type"] == "attn":
            hi = comp["head"]
            lo, hie = hi * dk, (hi + 1) * dk
            return [
                ("V", block.self_attn.qkv.weight, "attn_ip_w",
                 slice(2 * D + lo, 2 * D + hie), slice(None)),
                ("O", block.self_attn.projection.weight, "attn_op_w",
                 slice(None), slice(lo, hie)),
            ]
        return [
            ("fc", block.mlp.fc1.weight, "fc1_w",
             slice(None), slice(None)),
            ("proj", block.mlp.fc2.weight, "fc2_w",
             slice(None), slice(None)),
        ]

    # ── forward ──
    def encode_image(self, model, images):
        vis_proj, _, _ = self._itc_heads(model)
        return vis_proj(self._cls(model.vision_model(pixel_values=images)))

    def encode_text(self, model, texts, device):
        _, txt_model, txt_proj = self._itc_heads(model)
        proc = self._processor(model)
        # max_length=35 — BLIP's training truncation length (Salesforce med.py)
        tok = proc.tokenizer(list(texts), padding="max_length",
                             truncation=True, max_length=35,
                             return_tensors="pt")
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        out = txt_model(input_ids=input_ids, attention_mask=attention_mask)
        return txt_proj(self._cls(out))

    def get_preprocess(self, model):
        image_processor = self._processor(model).image_processor

        def preprocess(pil_img):
            out = image_processor(images=pil_img, return_tensors="pt")
            return out["pixel_values"][0]
        return preprocess

    def get_norm_stats(self, model):
        try:
            ip = self._processor(model).image_processor
            return list(ip.image_mean), list(ip.image_std)
        except Exception:
            return CLIP_MEAN, CLIP_STD

    # ── sub-layer surgery ──
    def get_sublayer_modules(self, block):
        return {"attn": block.self_attn, "mlp": block.mlp}

    @contextmanager
    def replace_sublayer(self, block, kind, const_out):
        # BlipEncoderLayer calls self_attn/mlp as *modules*; attention returns
        # a (hidden, weights) 2-tuple that the layer unpacks
        # (``hidden_states, _ = self.self_attn(...)``), so the attn
        # replacement is the tuple-returning _AttnConst wrapper.
        if kind == "mlp":
            orig = block.mlp
            block.mlp = const_out
            try:
                yield
            finally:
                block.mlp = orig
        else:
            orig = block.self_attn
            block.self_attn = _AttnConst(const_out)
            try:
                yield
            finally:
                block.self_attn = orig

    # ── helper: hidden dim of the block's own projections (D) ──
    def get_hidden_dim_from_block(self, block):
        # fused qkv weight is [3D, D]
        return block.self_attn.qkv.weight.shape[1]


# ══════════════════════════════════════════════════════════════════════
#  EVA-CLIP / OpenCLIP ViT (BLIP-2's vision tower)
# ══════════════════════════════════════════════════════════════════════
class EVACLIPAdapter(OpenAICLIPAdapter):
    """Adapter for EVA-CLIP / OpenCLIP ViTs (``EVA_CLIP_g_14``, …).

    BLIP-2's vision tower is **EVA-CLIP ViT-G/14** (X-Transfer Table 6), so
    attacking it is what makes the ``vlm_transfer`` tower swap a true 1:1
    replacement.  The vision side is structurally identical to OpenAI CLIP
    (``model.visual.transformer.resblocks``, fused ``attn.in_proj_weight``,
    ``mlp.c_fc``/``c_proj``, ``ln_1``/``ln_2``) — hence the inheritance —
    so Phase-0 int8 grids, Phase-2B component slices and Phase-3 ranking
    need no changes and ``sub_quant`` keys stay comparable across models.

    Only the two things OpenCLIP does *not* share with the ``clip``
    package are overridden: the text tokenizer and the preprocessing.
    EVA-02 LayerScale parameters (``ls_1``/``ls_2``) exist on some releases
    and are simply not quantised — the canonical block list is
    ``OpenAICLIPAdapter``'s exactly, so nothing downstream sees a
    difference.

    Model identifier comes from ``cfg.MODEL`` (``EVA_CLIP_g_14``) with the
    pretrained tag from ``cfg.HF_NAME``; see :func:`load_eva_clip`."""

    # ── structure (attribute names vary across CLIP flavours) ──
    def get_vision_module(self, model):
        return getattr(model, "visual", None) or getattr(model, "vision_model")

    def get_hidden_dim(self, model):
        vis = self.get_vision_module(model)
        if hasattr(vis, "transformer") and hasattr(vis.transformer, "width"):
            return vis.transformer.width
        return vis.positional_embedding.shape[1]

    def get_num_heads(self, block):
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "num_heads"):
            return int(attn.num_heads)
        return int(getattr(block, "num_heads",
                           self.get_hidden_dim_from_block(block) // 64))

    def get_seq_layout(self, model):
        vis = self.get_vision_module(model)
        pos = vis.positional_embedding          # [seq_len, width]
        seq_len, width = int(pos.shape[0]), int(pos.shape[1])
        n_prefix = 1                            # CLS token
        n_patches = seq_len - n_prefix
        patch_size = int(vis.conv1.kernel_size[0])
        patches_per_side = int(round(n_patches ** 0.5))
        blocks = list(self.get_vision_transformer_blocks(model))

        # the input resolution is named differently across releases
        # (input_resolution / image_size / absent) — derive it instead of
        # trusting any one of them
        img_size = getattr(vis, "input_resolution", None) \
            or getattr(vis, "image_size", None)
        if isinstance(img_size, (tuple, list)):
            img_size = img_size[0]
        if img_size is None:
            img_size = patches_per_side * patch_size

        return dict(
            seq_len=seq_len, width=width,
            n_patches=n_patches, n_prefix=n_prefix,
            patch_size=patch_size, patches_per_side=patches_per_side,
            img_size=int(img_size),
            n_layers=len(blocks), n_heads=self.get_num_heads(blocks[0]))

    # ── forward ──
    def encode_text(self, model, texts, device):
        tok = getattr(model, "_adapter_tokenizer", None)
        if tok is None:
            raise RuntimeError(
                "EVACLIPAdapter.encode_text needs a tokenizer — load the "
                "model through adapters.load_model (backend 'eva_clip') so "
                "model._adapter_tokenizer is set.")
        return model.encode_text(tok(list(texts)).to(device))

    def get_preprocess(self, model):
        pre = getattr(model, "_adapter_preprocess", None)
        if pre is not None:
            return pre
        # CLIP-stats fallback: resize + centre crop + normalise
        size = self.get_seq_layout(model)["img_size"]

        def _pre(pil_img):
            import numpy as np
            from PIL import Image
            img = pil_img.convert("RGB")
            w, h = img.size
            scale = size / min(w, h)
            img = img.resize((max(size, int(w * scale + 0.5)),
                              max(size, int(h * scale + 0.5))), Image.BICUBIC)
            w, h = img.size
            left, top = (w - size) // 2, (h - size) // 2
            img = img.crop((left, top, left + size, top + size))
            arr = np.asarray(img).astype("float32") / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)
            mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
            std = torch.tensor(CLIP_STD).view(3, 1, 1)
            return (t - mean) / std
        return _pre


def load_eva_clip(cfg, device):
    """Load a full EVA-CLIP / OpenCLIP model (vision **and** text towers).

    The text tower is required: CLIP-BFA's Phase-1 trigger loss and the
    Phase-4 ASR readout both live in the joint image–text space.

    ``cfg.MODEL`` is the model id (``EVA_CLIP_g_14``) and ``cfg.HF_NAME``
    the pretrained tag when it differs.  Tries the official ``eva_clip``
    package first, then ``open_clip``."""
    name = cfg.MODEL
    pretrained = getattr(cfg, "HF_NAME", None) or name

    model = tokenizer = preprocess = None

    try:                                    # 1) official EVA-CLIP release
        import eva_clip
        model, _pre_train, preprocess = eva_clip.create_model_and_transforms(
            name, pretrained=pretrained, force_custom_clip=True)
        tokenizer = eva_clip.get_tokenizer(name)
    except ImportError:
        pass
    except Exception as e:                  # noqa: BLE001 — try the next one
        print(f"  [eva_clip] {type(e).__name__}: {e} — trying open_clip")

    if model is None:                       # 2) open_clip fallback
        try:
            import open_clip
            model, _pre_train, preprocess = \
                open_clip.create_model_and_transforms(name,
                                                      pretrained=pretrained)
            tokenizer = open_clip.get_tokenizer(name)
        except ImportError as e:
            raise ImportError(
                "EVA-CLIP / OpenCLIP models need one of:\n"
                "    pip install eva-clip           # official EVA-CLIP\n"
                "    pip install open_clip_torch\n"
                "Neither package is importable here.  See "
                "vlm_transfer/README.md §6.\n"
                f"  (Original error: {e})")
        except Exception as e:              # noqa: BLE001
            raise RuntimeError(
                f"Could not load EVA-CLIP model {name!r} / {pretrained!r}: "
                f"{type(e).__name__}: {e}\n"
                f"  Override with the registry entry's clip_name/hf_name.")

    model = model.float()
    model.to(device)
    model._is_eva_clip = True
    model._adapter_tokenizer = tokenizer
    model._adapter_preprocess = preprocess
    return model, preprocess


def _is_openclip_layout(model) -> bool:
    """open_clip / eva_clip models, as opposed to the ``clip`` package.

    Both expose ``.visual``, so this MUST be consulted before
    ``get_adapter``'s OpenAI branch."""
    cls = type(model)
    mod = getattr(cls, "__module__", "").lower()
    name = cls.__name__.lower()
    return bool(getattr(model, "_is_eva_clip", False)
                or "open_clip" in mod or "eva_clip" in mod
                or "evavisiontransformer" in name)


# ══════════════════════════════════════════════════════════════════════
#  FACTORY + MODEL LOADING
# ══════════════════════════════════════════════════════════════════════
_ADAPTER_CACHE = {}


def get_adapter(model):
    """Dispatch on the model class and cache the adapter instance."""
    key = id(model)
    ad = _ADAPTER_CACHE.get(key)
    if ad is not None:
        return ad
    cls = type(model)
    mod = getattr(cls, "__module__", "")
    # open_clip / eva_clip FIRST — they also expose ``.visual``, which the
    # OpenAI branch below would otherwise claim
    if _is_openclip_layout(model):
        ad = EVACLIPAdapter()
    elif mod.startswith("clip") or hasattr(model, "visual"):
        ad = OpenAICLIPAdapter()
    elif "blip" in mod.lower() or cls.__name__.startswith("Blip") \
            or hasattr(model, "visual_projection"):
        ad = BlipAdapter()
    elif "siglip" in mod.lower() or cls.__name__.startswith("Siglip") \
            or hasattr(model, "vision_model"):
        ad = SigLIPAdapter()
    else:
        raise ValueError(f"No ModelAdapter for model class {cls.__module__}.{cls.__name__}")
    _ADAPTER_CACHE[key] = ad
    return ad


def encode_image(model, images):
    """Shared plumbing: raw image features via the model's adapter."""
    return get_adapter(model).encode_image(model, images)


def adapter_for_block(block):
    """Adapter for a single transformer block (blocks carry no model ref,
    but their class identifies the architecture)."""
    cls = type(block)
    mod = getattr(cls, "__module__", "")
    if _is_openclip_layout(block):
        return EVACLIPAdapter()
    if "blip" in mod.lower() or cls.__name__.startswith("Blip") \
            or (hasattr(block, "self_attn") and hasattr(block.self_attn, "qkv")):
        return BlipAdapter()
    if "siglip" in mod.lower() or cls.__name__.startswith("Siglip") \
            or hasattr(block, "self_attn"):
        return SigLIPAdapter()
    return OpenAICLIPAdapter()


def load_model(cfg):
    """Load (model, preprocess) for the registry entry behind cfg.

    cfg.MODEL_ADAPTER selects the backend ('openai_clip' | 'siglip' |
    'blip' | 'eva_clip'); cfg.MODEL is the OpenAI clip / EVA-CLIP model
    name, cfg.HF_NAME the HF hub id or pretrained tag.
    """
    backend = getattr(cfg, "MODEL_ADAPTER", "openai_clip")
    if backend == "openai_clip":
        import clip
        model, preprocess = clip.load(cfg.MODEL, device=cfg.DEVICE)
        model = model.float()
    elif backend == "eva_clip":
        # EVA-CLIP ViT-G/14 — BLIP-2's tower (vlm_transfer/README.md §6).
        # Needs the vision AND text towers: the attack's t_target and the
        # Phase-4 readout live in the joint space.
        model, preprocess = load_eva_clip(cfg, cfg.DEVICE)
    elif backend == "siglip":
        from transformers import AutoProcessor, SiglipModel
        model = SiglipModel.from_pretrained(cfg.HF_NAME)
        model = model.float().to(cfg.DEVICE)
        proc = AutoProcessor.from_pretrained(cfg.HF_NAME)
        model._adapter_processor = proc
        preprocess = get_adapter(model).get_preprocess(model)
    elif backend == "blip":
        from transformers import AutoConfig, AutoProcessor
        import transformers as tmod
        # The checkpoint's own class decides the attribute names (BlipModel:
        # visual_projection/text_model; BlipForImageTextRetrieval:
        # vision_proj/text_encoder).  Loading with the wrong class would
        # silently leave the ITC projections randomly initialised, so the
        # class is read from the checkpoint config.  (The raw capfilt
        # pretrain checkpoint is not on the hub — BLIP.md §7.)
        arch = (getattr(AutoConfig.from_pretrained(cfg.HF_NAME),
                        "architectures", None) or ["BlipModel"])[0]
        cls_map = {"BlipModel", "BlipForImageTextRetrieval"}
        if arch not in cls_map:
            raise ValueError(
                f"BLIP checkpoint {cfg.HF_NAME!r} is a {arch} — it has no "
                "ITC projection heads; use a BlipModel / "
                "BlipForImageTextRetrieval checkpoint (BLIP.md §4/§7).")
        model = getattr(tmod, arch).from_pretrained(cfg.HF_NAME)
        model = model.float().to(cfg.DEVICE)
        proc = AutoProcessor.from_pretrained(cfg.HF_NAME)
        model._adapter_processor = proc
        preprocess = get_adapter(model).get_preprocess(model)
    else:
        raise ValueError(f"Unknown model adapter {backend!r}")
    return model, preprocess
