"""
eva_clip_adapter.py — EVA-CLIP ViT-G/14 as a CLIP-BFA attack surface.
=====================================================================
BLIP-2's vision tower is **EVA-CLIP ViT-G/14** (X-Transfer Table 6), not an
OpenAI CLIP.  To get a *true 1:1 tower swap* into BLIP-2 the attack has to
run on that exact encoder, so this module adds it as a first-class attack
model without touching ``adapters.py`` (maintainer asked for all VLM code
to live under ``vlm_transfer/``).

Two things are handled here:

1. ``EVACLIPAdapter`` — a ``ModelAdapter`` for OpenCLIP-layout ViTs
   (``model.visual.transformer.resblocks``, fused ``attn.in_proj_weight``,
   ``mlp.c_fc``/``c_proj``).  The vision side is structurally identical to
   OpenAI CLIP, so it inherits ``OpenAICLIPAdapter`` and only fixes the two
   things that are *not* shared: the text tokenizer and the preprocessing
   (OpenCLIP / EVA-CLIP, not the ``clip`` package).

2. ``install()`` — registers the adapter and an ``eva_clip`` backend for
   ``adapters.load_model``.  **Call it before the pipeline modules are
   imported** (see :func:`install`); it also rebinds any already-imported
   references so the call order cannot silently break the dispatch.

Checkpoint availability (documented in README.md §6): the attack needs a
full EVA-CLIP ViT-G/14 (vision *and* text towers) from the ``eva_clip``
package (official EVA-CLIP release) or ``open_clip``.  The *swap target*
in BLIP-2 is HF ``Blip2VisionModel`` and is available from
``Salesforce/blip2-opt-6.7b`` regardless of which of the two loaded the
attack model.
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn

from adapters import OpenAICLIPAdapter
from utils import CLIP_MEAN, CLIP_STD

__all__ = ["EVACLIPAdapter", "load_eva_clip", "install", "is_eva_clip"]

_BACKEND = "eva_clip"
_INSTALLED = False


def is_eva_clip(model) -> bool:
    """True when ``model`` is an OpenCLIP / EVA-CLIP *or* tagged as one."""
    cls = type(model)
    mod = getattr(cls, "__module__", "").lower()
    name = cls.__name__.lower()
    return bool(getattr(model, "_is_eva_clip", False)
                or "eva_clip" in mod or "open_clip" in mod
                or name.startswith("clip") and "evaclip" in name
                or "evavisiontransformer" in name)


class EVACLIPAdapter(OpenAICLIPAdapter):
    """OpenCLIP/EVA-CLIP ViT — same block layout as OpenAI CLIP.

    Only the text path and the preprocessing differ, so everything about
    Phase 0 (int8 grids), Phase 2B component slices and Phase 3 ranking is
    inherited unchanged.  EVA-02 LayerScale parameters (``ls_1``/``ls_2``)
    are present on some releases and are simply not quantised — the
    canonical block list below is identical to ``OpenAICLIPAdapter``'s so
    ``sub_quant`` keys stay comparable across models.
    """

    # ── structure ──
    def get_vision_transformer_blocks(self, model):
        vis = self.get_vision_module(model)
        return vis.transformer.resblocks

    def get_vision_module(self, model):
        return getattr(model, "visual", None) or getattr(model, "vision_model")

    def get_hidden_dim(self, model):
        vis = self.get_vision_module(model)
        if hasattr(vis, "transformer") and hasattr(vis.transformer, "width"):
            return vis.transformer.width
        return vis.positional_embedding.shape[1]

    def get_seq_layout(self, model):
        vis = self.get_vision_module(model)
        pos = vis.positional_embedding
        seq_len, width = int(pos.shape[0]), int(pos.shape[1])
        n_prefix = 1                       # CLS token
        n_patches = seq_len - n_prefix
        patch_size = int(vis.conv1.kernel_size[0])
        patches_per_side = int(round(n_patches ** 0.5))
        blocks = list(vis.transformer.resblocks)

        # input resolution differs across CLIP flavours (image_size vs
        # input_resolution vs a missing attribute) — derive it rather than
        # trusting any one of them.
        img_size = getattr(vis, "input_resolution", None) or \
            getattr(vis, "image_size", None)
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

    def get_num_heads(self, block):
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "num_heads"):
            return int(attn.num_heads)
        width = self.get_hidden_dim_from_block(block)
        return int(getattr(block, "num_heads", width // 64))

    # ── forward (the two things OpenCLIP does not share with ``clip``) ──
    def encode_text(self, model, texts, device):
        tok = getattr(model, "_adapter_tokenizer", None)
        if tok is None:
            raise RuntimeError(
                "EVACLIPAdapter.encode_text needs a tokenizer — load the "
                "model through adapters.load_model (backend 'eva_clip') so "
                "model._adapter_tokenizer is set.")
        tokens = tok(list(texts)).to(device)
        return model.encode_text(tokens)

    def get_preprocess(self, model):
        pre = getattr(model, "_adapter_preprocess", None)
        if pre is not None:
            return pre
        size = self.get_seq_layout(model)["img_size"]
        # CLIP-stats fallback: resize + centre crop + normalise
        def _pre(pil_img):
            from PIL import Image as _I
            import numpy as _np
            img = pil_img.convert("RGB")
            w, h = img.size
            scale = size / min(w, h)
            img = img.resize((max(size, int(w * scale + 0.5)),
                              max(size, int(h * scale + 0.5))), _I.BICUBIC)
            w, h = img.size
            left, top = (w - size) // 2, (h - size) // 2
            img = img.crop((left, top, left + size, top + size))
            arr = _np.asarray(img).astype("float32") / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)
            mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
            std = torch.tensor(CLIP_STD).view(3, 1, 1)
            return (t - mean) / std
        return _pre

    def get_norm_stats(self, model):
        return CLIP_MEAN, CLIP_STD


# ══════════════════════════════════════════════════════════════════════
#  LOADING
# ══════════════════════════════════════════════════════════════════════
def load_eva_clip(cfg, device):
    """Load a full EVA-CLIP / OpenCLIP model (vision + text towers).

    ``cfg.MODEL`` is the model identifier (``EVA_CLIP_g_14``) and
    ``cfg.HF_NAME`` the pretrained tag when it is not the same string.
    Tries the official ``eva_clip`` package first, then ``open_clip``.
    """
    name = cfg.MODEL
    pretrained = getattr(cfg, "HF_NAME", None) or name

    model = tokenizer = preprocess = None

    try:                                   # 1) official EVA-CLIP release
        import eva_clip
        model, _pre_train, preprocess = eva_clip.create_model_and_transforms(
            name, pretrained=pretrained, force_custom_clip=True)
        tokenizer = eva_clip.get_tokenizer(name)
    except ImportError:
        pass
    except Exception as e:                 # noqa: BLE001 — try the next one
        print(f"  [eva_clip] {type(e).__name__}: {e} — trying open_clip")

    if model is None:                      # 2) open_clip fallback
        try:
            import open_clip
            model, _pre_train, preprocess = open_clip.create_model_and_transforms(
                name, pretrained=pretrained)
            tokenizer = open_clip.get_tokenizer(name)
        except ImportError as e:
            raise ImportError(
                "BLIP-2's tower is EVA-CLIP ViT-G/14 and needs one of:\n"
                "    pip install eva-clip      # official EVA-CLIP\n"
                "    pip install open_clip_torch\n"
                "Neither package is importable in this environment.\n"
                "  See vlm_transfer/README.md §6 for the checkpoint and the\n"
                "  install steps.  (Original error: %s)" % e)
        except Exception as e:             # noqa: BLE001
            raise RuntimeError(
                f"Could not load EVA-CLIP model {name!r} / {pretrained!r}: "
                f"{type(e).__name__}: {e}\n"
                f"  Override with --eva-clip-name / --eva-clip-pretrained.")

    model = model.float()
    model.to(device)
    model._is_eva_clip = True
    model._adapter_tokenizer = tokenizer
    model._adapter_preprocess = preprocess
    return model, preprocess


# ══════════════════════════════════════════════════════════════════════
#  REGISTRATION
# ══════════════════════════════════════════════════════════════════════
def _rebind(old, new):
    """Point every module-level reference to ``old`` at ``new``.

    ``models.py``/``pipeline.py`` do ``from adapters import get_adapter``
    at import time, which binds the *function object* — patching the
    attribute on ``adapters`` afterwards would not reach them.  Rebinding
    makes ``install()`` order-independent instead of a silent trap.
    """
    for mod in [m for m in list(sys.modules.values()) if m is not None]:
        d = getattr(mod, "__dict__", None)
        if not isinstance(d, dict):
            continue
        for attr, val in list(d.items()):
            if val is old:
                try:
                    setattr(mod, attr, new)
                except Exception:          # noqa: BLE001 — frozen module
                    pass


def install():
    """Register ``EVACLIPAdapter`` and the ``eva_clip`` backend.

    Idempotent.  Safe to call at any point, but calling it before the
    pipeline modules are imported keeps the patching minimal.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    import adapters

    orig_get = adapters.get_adapter
    orig_block = adapters.adapter_for_block
    orig_load = adapters.load_model

    def get_adapter(model):
        if is_eva_clip(model):
            ad = adapters._ADAPTER_CACHE.get(id(model))
            if ad is None:
                ad = EVACLIPAdapter()
                adapters._ADAPTER_CACHE[id(model)] = ad
            return ad
        return orig_get(model)

    def adapter_for_block(block):
        if is_eva_clip(block) or type(block).__name__.lower().startswith(
                ("evablock", "residualattentionblock")):
            return EVACLIPAdapter()
        return orig_block(block)

    def load_model(cfg):
        if getattr(cfg, "MODEL_ADAPTER", "openai_clip") == _BACKEND:
            model, preprocess = load_eva_clip(cfg, cfg.DEVICE)
            return model, preprocess
        return orig_load(cfg)

    adapters.get_adapter = get_adapter
    adapters.adapter_for_block = adapter_for_block
    adapters.load_model = load_model
    _rebind(orig_get, get_adapter)
    _rebind(orig_block, adapter_for_block)
    _rebind(orig_load, load_model)

    _INSTALLED = True
    print(f"[vlm_transfer] EVA-CLIP adapter registered "
          f"(backend '{_BACKEND}')")
