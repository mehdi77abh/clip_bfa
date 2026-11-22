"""
tower_swap.py — Move an attacked CLIP vision tower into a VLM.
===============================================================
The whole VLM-transfer experiment is one mechanical step: take the
``model_state_attacked.pt`` written by a CLIP-BFA run (a ``state_dict`` of
the attacked encoder) and load it into the vision tower the VLM actually
uses.  Everything hard about that is naming:

  * the attack saves OpenAI-CLIP / OpenCLIP naming (``visual.transformer.
    resblocks.{i}.attn.in_proj_weight``, …),
  * HF towers use ``vision_model.encoder.layers.{i}.self_attn.q_proj``,
    *split* q/k/v projections, ``pre_layrnorm``/``post_layernorm``,
  * positional embeddings differ in length whenever the VLM's input size
    differs from the attack's ``IMG_SIZE`` (X-Transfer: "for large VLMs
    that use different image resolutions … we use interpolation to rescale
    the perturbation to the resolution used by the VLM").

This module owns that mapping and reports exactly what was and was not
transferred, so a silent partial swap can never be mistaken for success.
"""

from __future__ import annotations

import re
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "load_attacked_state",
    "vision_keys",
    "remap_vision_state",
    "remap_vision_state_traced",
    "interpolate_pos_embed",
    "fit_tower_pos_embed",
    "find_vision_tower",
    "detect_layout",
    "swap_vision_tower",
]

# Layouts of the *target* tower.
LAYOUT_HF = "hf_vit"           # embeddings.* / encoder.layers.*
LAYOUT_OPENCLIP = "openclip_vit"  # conv1 / transformer.resblocks.*

_RESBLOCK_RE = re.compile(
    r"^(?:visual\.)?transformer\.resblocks\.(\d+)\.(.*)$")
_LAYER_RE = re.compile(r"^(?:visual\.)?encoder\.layers\.(\d+)\.(.*)$")


# ══════════════════════════════════════════════════════════════════════
#  READING THE ATTACK ARTEFACT
# ══════════════════════════════════════════════════════════════════════
def load_attacked_state(run_dir: str, filename: str = "model_state_attacked.pt"
                        ) -> dict:
    """Load the attacked ``state_dict`` saved by a CLIP-BFA run."""
    import os
    path = os.path.join(run_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing attacked model state: {path}\n"
            f"  Re-run the attack with --save-attacked-model.")
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state \
            and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return {k: v for k, v in state.items() if torch.is_tensor(v)}


def vision_keys(state: dict, prefix: str = "visual.") -> dict:
    """The vision-tower subset of an attacked state_dict, prefix stripped.

    ``prefix`` is the attack model's vision namespace (``visual.`` for
    OpenAI-CLIP and OpenCLIP; pass ``""`` if the state is already tower
    only).
    """
    out = {}
    for k, v in state.items():
        if prefix and not k.startswith(prefix):
            continue
        out[k[len(prefix):] if prefix else k] = v
    if not out:
        sample = sorted(state)[:6]
        raise KeyError(
            f"No keys with prefix {prefix!r} in the attacked state.  "
            f"Sample keys: {sample}")
    return out


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL-EMBEDDING INTERPOLATION
# ══════════════════════════════════════════════════════════════════════
def interpolate_pos_embed(state: dict, key: str, new_num_tokens: int) -> bool:
    """Bilinearly resample ``state[key]`` to ``new_num_tokens`` rows.

    Handles the CLIP layout ``[1 + H*W, D]`` (CLS first) as well as plain
    ``[N, D]``.  Returns True when a resize actually happened.
    """
    if key not in state:
        return False
    pos = state[key]
    if pos.ndim != 2 or pos.shape[0] == new_num_tokens:
        return False

    def _side(n: int) -> int:
        s = int(round(n ** 0.5))
        return s if s * s == n else 0

    # CLIP layout is [1 + H*W, D]; fall back to plain [N, D] if neither
    # side is a square grid after dropping the leading row.
    src_side = _side(pos.shape[0] - 1)
    dst_side = _side(new_num_tokens - 1)
    if src_side and dst_side:
        cls, spatial = pos[:1], pos[1:]
    else:
        src_side, dst_side = _side(pos.shape[0]), _side(new_num_tokens)
        cls, spatial = None, pos
    if not src_side or not dst_side:
        raise ValueError(
            f"Cannot interpolate {key}: {tuple(pos.shape)} -> "
            f"{new_num_tokens} tokens is not a square-grid resize")

    new_side = dst_side
    grid = spatial.reshape(1, src_side, src_side, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid.float(), size=(new_side, new_side),
                         mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(new_side * new_side, -1)

    state[key] = torch.cat([cls, grid], dim=0).to(pos.dtype) \
        if cls is not None else grid.to(pos.dtype)
    return True


# ══════════════════════════════════════════════════════════════════════
#  FITTING A LIVE TOWER TO A DIFFERENT INPUT RESOLUTION
# ══════════════════════════════════════════════════════════════════════
def fit_tower_pos_embed(tower, image_size: int, patch_size: int) -> bool:
    """Interpolate a *live* tower's position embeddings to ``image_size``.

    Needed when a VLM runs at a resolution other than the checkpoint's
    native grid (X-Transfer Table 6 lists BLIP-2 at 364 × 364 while the
    checkpoints ship 224).  Handles HF ``nn.Embedding`` position tables
    (with their ``position_ids`` buffer) and OpenCLIP ``Parameter`` ones.
    Returns True when a resize happened.
    """
    want = 1 + (int(image_size) // int(patch_size)) ** 2

    # ── HF: tower.embeddings.position_embedding is an nn.Embedding ──
    emb = getattr(getattr(tower, "embeddings", None), "position_embedding",
                  None)
    if isinstance(emb, nn.Embedding):
        if emb.num_embeddings == want:
            return False
        with torch.no_grad():
            state = {"positional_embedding": emb.weight.detach().clone()}
            interpolate_pos_embed(state, "positional_embedding", want)
            new_emb = nn.Embedding(want, emb.embedding_dim)
            new_emb.weight.copy_(state["positional_embedding"])
        tower.embeddings.position_embedding = new_emb
        if hasattr(tower.embeddings, "position_ids"):
            tower.embeddings.position_ids = torch.arange(want).expand((1, -1))
        if hasattr(tower, "config"):
            tower.config.image_size = int(image_size)
            tower.config.num_positions = want
        return True

    # ── OpenCLIP: tower.positional_embedding is a Parameter ────────
    pos = getattr(tower, "positional_embedding", None)
    if isinstance(pos, nn.Parameter):
        if pos.shape[0] == want:
            return False
        with torch.no_grad():
            state = {"positional_embedding": pos.detach().clone()}
            interpolate_pos_embed(state, "positional_embedding", want)
            new_pos = nn.Parameter(state["positional_embedding"])
        tower.positional_embedding = new_pos
        if hasattr(tower, "input_resolution"):
            tower.input_resolution = int(image_size)
        return True

    return False


# ══════════════════════════════════════════════════════════════════════
#  RENAMING
# ══════════════════════════════════════════════════════════════════════
def detect_layout(module) -> str:
    """Which naming convention ``module``'s state_dict uses."""
    keys = module.state_dict().keys()
    if any(k.endswith("self_attn.q_proj.weight") for k in keys):
        return LAYOUT_HF
    if any("attn.in_proj_weight" in k for k in keys):
        return LAYOUT_OPENCLIP
    if any("embeddings.patch_embedding" in k for k in keys):
        return LAYOUT_HF
    if any(k == "conv1.weight" or k.endswith(".conv1.weight") for k in keys):
        return LAYOUT_OPENCLIP
    raise ValueError(
        "Cannot detect the vision-tower layout — neither HF ViT "
        "(self_attn.q_proj) nor OpenCLIP ViT (attn.in_proj_weight) keys "
        f"found.  Sample: {sorted(keys)[:6]}")


def _module_prefix(module, marker: str) -> str:
    """Namespace inside ``module`` before ``marker`` (e.g. 'vision_model.')."""
    for k in module.state_dict().keys():
        if marker in k:
            return k.split(marker)[0]
    return ""


def remap_vision_state_traced(vision_state: dict, layout: str,
                              out_prefix: str = ""):
    """Like :func:`remap_vision_state`, but also reports provenance.

    Returns ``(mapped, src_of)`` where ``src_of[target_key]`` is the source
    key it came from.  A fused ``in_proj_*`` fans out to three targets and
    maps three source keys to one target name in reverse — hence the
    forward direction is the one that is tracked.
    """
    if layout == LAYOUT_OPENCLIP:
        out = {out_prefix + k: v for k, v in vision_state.items()}
        return out, {out_prefix + k: k for k in vision_state}

    if layout != LAYOUT_HF:
        raise ValueError(f"Unknown layout {layout!r}")

    out: dict = {}
    src_of: dict = {}

    def put(target_suffix, value, source_key):
        out[out_prefix + target_suffix] = value
        src_of[out_prefix + target_suffix] = source_key

    # ── non-block tensors ──────────────────────────────────────────
    direct = {
        "conv1.weight": "embeddings.patch_embedding.weight",
        "class_embedding": "embeddings.class_embedding",
        "positional_embedding": "embeddings.position_embedding.weight",
        "ln_pre.weight": "pre_layrnorm.weight",
        "ln_pre.bias": "pre_layrnorm.bias",
        "ln_post.weight": "post_layernorm.weight",
        "ln_post.bias": "post_layernorm.bias",
    }
    for src, dst in direct.items():
        if src in vision_state:
            put(dst, vision_state[src], src)

    # ── per-block ──────────────────────────────────────────────────
    for key, value in vision_state.items():
        m = _RESBLOCK_RE.match(key)
        if not m:
            continue
        i, rest = int(m.group(1)), m.group(2)
        base = f"encoder.layers.{i}."
        simple = {
            "ln_1.weight": "layer_norm1.weight",
            "ln_1.bias": "layer_norm1.bias",
            "ln_2.weight": "layer_norm2.weight",
            "ln_2.bias": "layer_norm2.bias",
            "attn.out_proj.weight": "self_attn.out_proj.weight",
            "attn.out_proj.bias": "self_attn.out_proj.bias",
            "mlp.c_fc.weight": "mlp.fc1.weight",
            "mlp.c_fc.bias": "mlp.fc1.bias",
            "mlp.c_proj.weight": "mlp.fc2.weight",
            "mlp.c_proj.bias": "mlp.fc2.bias",
        }
        if rest in simple:
            put(base + simple[rest], value, key)
        elif rest == "attn.in_proj_weight":
            # fused [3D, D] -> q / k / v
            q, k, v = value.chunk(3, dim=0)
            put(base + "self_attn.q_proj.weight", q, key)
            put(base + "self_attn.k_proj.weight", k, key)
            put(base + "self_attn.v_proj.weight", v, key)
        elif rest == "attn.in_proj_bias":
            q, k, v = value.chunk(3, dim=0)
            put(base + "self_attn.q_proj.bias", q, key)
            put(base + "self_attn.k_proj.bias", k, key)
            put(base + "self_attn.v_proj.bias", v, key)
        # else: LayerScale / attn_pool / etc. — reported as dropped_source

    # ── optional projection head ───────────────────────────────────
    if "proj" in vision_state:
        # OpenAI CLIP stores [width, out_dim]; HF wants [out_dim, width]
        put("visual_projection.weight", vision_state["proj"].t().contiguous(),
            "proj")

    return out, src_of


def remap_vision_state(vision_state: dict, layout: str,
                       out_prefix: str = "") -> dict:
    """Rename OpenAI/OpenCLIP-style tower keys to the target layout.

    ``vision_state`` must already be tower-only (see :func:`vision_keys`).
    Keys that cannot be mapped are silently absent — use
    :func:`swap_vision_tower` for a full audit of what did and did not
    transfer.
    """
    return remap_vision_state_traced(vision_state, layout, out_prefix)[0]


# ══════════════════════════════════════════════════════════════════════
#  FINDING THE TOWER INSIDE A VLM
# ══════════════════════════════════════════════════════════════════════
def find_vision_tower(model, prefer: Optional[tuple] = None):
    """Locate the ViT vision tower inside an arbitrary VLM wrapper.

    ``prefer`` is a tuple of attribute paths to try first
    (e.g. ``("vision_tower.vision_model", "vision_model")``).  Falls back
    to scanning for a submodule whose state_dict has patch embeddings.
    """
    if prefer:
        for path in prefer:
            obj = model
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if isinstance(obj, nn.Module):
                return obj

    # Prefer the shallowest matching submodule — the tower root, not a
    # block inside it.
    best_name, best_mod, best_depth = None, None, None
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Module):
            continue
        low = name.lower()
        if not name or any(bad in low for bad in
                           ("lm_head", "language_model", "qformer", "text")):
            continue
        try:
            keys = mod.state_dict().keys()
        except Exception:
            continue
        has_patch = any("patch_embedding" in k or k.endswith("conv1.weight")
                        for k in keys)
        has_blocks = any("encoder.layers." in k or "resblocks." in k
                         for k in keys)
        if not (has_patch and has_blocks):
            continue
        depth = name.count(".")
        if best_depth is None or depth < best_depth:
            best_name, best_mod, best_depth = name, mod, depth

    if best_mod is not None:
        return best_mod
    raise ValueError("No vision tower found in the VLM — pass prefer=("
                     "'vision_tower.vision_model', …)")


# ══════════════════════════════════════════════════════════════════════
#  THE SWAP
# ══════════════════════════════════════════════════════════════════════
def swap_vision_tower(model, attacked_state: dict, *,
                      source_prefix: str = "visual.",
                      tower=None, prefer: Optional[tuple] = None,
                      image_size: Optional[int] = None,
                      patch_size: Optional[int] = None) -> dict:
    """Load the attacked tower weights into the VLM.  Returns an audit.

    Audit keys: ``target_module``, ``layout``, ``mapped``, ``loaded``,
    ``missing`` (tower params never supplied), ``unexpected`` (mapped keys
    the tower does not have), ``dropped_source`` (attack keys not mapped),
    ``pos_embed_resized``.
    """
    tower = tower if tower is not None else find_vision_tower(model, prefer)
    layout = detect_layout(tower)
    prefix = _module_prefix(tower, "embeddings.patch_embedding") \
        if layout == LAYOUT_HF else _module_prefix(tower, "conv1.weight")

    src = vision_keys(attacked_state, source_prefix)
    mapped, src_of = remap_vision_state_traced(src, layout, out_prefix=prefix)

    # Positional embeddings: the attacked state is at the ATTACK's grid and
    # the tower may be at the VLM's grid, so both sides have to be fitted
    # to the target token count BEFORE the shape audit below.
    pos_key = prefix + ("embeddings.position_embedding.weight"
                        if layout == LAYOUT_HF else "positional_embedding")
    resized = False
    if image_size and patch_size:
        want = 1 + (image_size // patch_size) ** 2
        resized = interpolate_pos_embed(mapped, pos_key, want) or resized
        resized = fit_tower_pos_embed(tower, image_size, patch_size) or resized

    tower_sd = tower.state_dict()
    filtered, unexpected = {}, []
    for k, v in mapped.items():
        if k not in tower_sd:
            unexpected.append(k)
            continue
        if tuple(v.shape) != tuple(tower_sd[k].shape):
            unexpected.append(f"{k} (shape {tuple(v.shape)} != "
                              f"{tuple(tower_sd[k].shape)})")
            continue
        filtered[k] = v

    used_sources = {src_of[k] for k in filtered if k in src_of}
    dropped = sorted(set(src) - used_sources)

    missing = [k for k in tower_sd if k not in filtered]
    tower.load_state_dict(filtered, strict=False)

    return {
        "target_module": repr(tower.__class__.__name__),
        "layout": layout,
        "prefix": prefix,
        "mapped": len(mapped),
        "loaded": len(filtered),
        "missing": missing[:25],
        "missing_n": len(missing),
        "unexpected": unexpected[:25],
        "unexpected_n": len(unexpected),
        "dropped_source": dropped[:25],
        "dropped_source_n": len(dropped),
        "pos_embed_resized": bool(resized),
    }
