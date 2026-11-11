# Code Documentation

**Generated on:** 2026-11-11 12:47:04 +0330
**Project Path:** `/media/Mehdi/Uni/Uni/00.Thesis/00.Implemention/clip_bfa/proposed_method/modular_new_pipeline`

## Directory Structure
```
/media/Mehdi/Uni/Uni/00.Thesis/00.Implemention/clip_bfa/proposed_method/modular_new_pipeline
├── ablation.py
├── adapters.py
├── aggregate.py
├── analysis.py
├── class_texts
│   ├── caltech101
│   ├── cifar10
│   └── imagenet1k
├── config.py
├── data.py
├── evaluation.py
├── experiment.py
├── hybrid_ranking.py
├── linear_probe.py
├── main.py
├── make_class_texts.py
├── models.py
├── pipeline.py
├── __pycache__
├── quantizer.py
├── results.py
├── search.py
├── target.py
├── trigger.py
├── utils.py
├── visualization.py
└── weights.py

6 directories, 22 files
```




## File: `ablation.py`
```py
"""
ablation.py — Phase 2A layer-level and Phase 2B component-level causal ablation.
================================================================================
Phase 2A (LayerAblator): mean-ablation of each sub-layer (attn / MLP) per
layer, scored by |cosine drop|, rank-normalised, top-K layer selection.
Phase 2B (ComponentAblator): weight-knockout ablation of every attention
head and MLP inside the selected layers.

All architecture access goes through the model's ModelAdapter (hooks via
get_sublayer_modules, replacement via replace_sublayer, knockouts via
get_attention_weights/get_mlp_weights).

Swap LayerAblator.ablate / ComponentAblator.ablate to change the selection
algorithm; no other file needs edits.
"""

from contextlib import contextmanager

import torch
import torch.nn as nn
from scipy.stats import rankdata

from adapters import adapter_for_block, encode_image, get_adapter
from trigger import _make_target_cos, apply_trigger, get_apply_params
from utils import cleanup_gpu


# ══════════════════════════════════════════════════════════════════════
#  MEAN-ABLATION MACHINERY (Phase 2A)
# ══════════════════════════════════════════════════════════════════════
class _ConstantOutput(nn.Module):
    """Returns a constant tensor broadcast to the input's batch dimension.

    Used to replace a sub-layer's forward with its mean output (mean
    ablation).  ``const`` is per-token-position, seq-first ``[seq_len, width]``
    (CLIP ViT transformer runs LND); batch-first inputs are detected at
    runtime and handled accordingly.  Accepts ``hidden_states=``/``**kwargs``
    so it can stand in for HuggingFace modules as well.
    """

    def __init__(self, const, seq_len=None):
        super().__init__()
        self.register_buffer("const", const)
        self.seq_len = seq_len

    def forward(self, x=None, hidden_states=None, **kw):
        if x is None:
            x = hidden_states
        c = self.const
        if x.dim() == 3 and c.dim() == 2:
            if x.shape[0] == c.shape[0]:        # [seq, B, width]
                return c.unsqueeze(1).expand(-1, x.shape[1], -1)
            if x.shape[1] == c.shape[0]:        # [B, seq, width]
                return c.unsqueeze(0).expand(x.shape[0], -1, -1)
        return c.expand_as(x)


@torch.no_grad()
def compute_sub_layer_means(model, calib_imgs, cfg, trigger=None):
    """Per-token-position mean of each sub-layer's output.

    FIX 1: If ``trigger`` is provided, means are computed on the POISONED
    forward so the ablation reference comes from the same distribution as
    the input being ablated.  Without this, late-layer poisoned activations
    diverge sharply from clean means, injecting OOD shock into the ablation.
    """
    ad = get_adapter(model)
    blocks = list(ad.get_vision_transformer_blocks(model))
    n_layers = len(blocks)
    seq_len = ad.get_seq_layout(model)["seq_len"]

    # ── Prepare input: poisoned if trigger given, else clean ──
    if trigger is not None:
        ap = get_apply_params(cfg, model)
        input_imgs = apply_trigger(calib_imgs, trigger, ap)
    else:
        input_imgs = calib_imgs

    sums = {li: {"attn_sum": None, "mlp_sum": None} for li in range(n_layers)}

    def make_hook(li, key):
        def hook_fn(module, inp, output):
            out = output[0] if isinstance(output, tuple) else output
            if out.dim() != 3:
                raise RuntimeError(
                    f"Unexpected sub-layer output shape {tuple(out.shape)}")
            if out.shape[0] != seq_len and out.shape[1] == seq_len:
                out = out.permute(1, 0, 2)          # → [seq, B, C]
            s = out.sum(dim=1)                       # [seq, C] sum over batch
            sums[li][key] = s if sums[li][key] is None else sums[li][key] + s
        return hook_fn

    hooks = []
    for li, blk in enumerate(blocks):
        subs = ad.get_sublayer_modules(blk)
        hooks.append(subs["attn"].register_forward_hook(make_hook(li, "attn_sum")))
        hooks.append(subs["mlp"].register_forward_hook(make_hook(li, "mlp_sum")))

    n_seen = 0
    mini_bs = cfg.MINI_BS
    for i in range(0, input_imgs.size(0), mini_bs):
        chunk = input_imgs[i:i + mini_bs]
        _ = encode_image(model, chunk)
        n_seen += chunk.size(0)
    for h in hooks:
        h.remove()

    means = {}
    for li in range(n_layers):
        means[li] = {
            "attn_mean": sums[li]["attn_sum"] / n_seen,
            "mlp_mean": sums[li]["mlp_sum"] / n_seen,
        }
    return means


# ══════════════════════════════════════════════════════════════════════
#  WEIGHT-KNOCKOUT MACHINERY (Phase 2B)
# ══════════════════════════════════════════════════════════════════════
@contextmanager
def _knockout_head(blk, sl):
    """Zero one head's out_proj column slice; restore afterwards.
    Effectively removes that head's output contribution."""
    w = adapter_for_block(blk).get_attention_weights(blk)["out"]
    orig = w.data[:, sl].clone()
    w.data[:, sl] = 0.0
    try:
        yield
    finally:
        w.data[:, sl] = orig


@contextmanager
def _knockout_mlp(blk):
    """Zero the MLP output-projection weight (bias retained — output becomes
    the constant bias, NOT exactly 0); restore afterwards."""
    w = adapter_for_block(blk).get_mlp_weights(blk)["fc2"]
    orig = w.data.clone()
    w.data.zero_()
    try:
        yield
    finally:
        w.data.copy_(orig)


# ══════════════════════════════════════════════════════════════════════
#  COMPONENT ENUMERATION (all heads + MLPs in selected layers)
# ══════════════════════════════════════════════════════════════════════
def enumerate_layer_components(model, top_layers):
    """Generate component descriptors for ALL attention heads and MLPs
    in the selected layers — no component preselection (matches pseudo)."""
    ad = get_adapter(model)
    n_heads = ad.get_seq_layout(model)["n_heads"]
    components = []
    for li in sorted(top_layers):
        for h in range(n_heads):
            components.append(dict(type="attn", layer=li, head=h))
        components.append(dict(type="ffn", layer=li, head=-1))
    return components


# ══════════════════════════════════════════════════════════════════════
#  PHASE 2A — LAYER-LEVEL CAUSAL ABLATION
# ══════════════════════════════════════════════════════════════════════
class LayerAblator:
    """Phase 2A: layer-level mean ablation → (layer_scores, top_layers)."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    @torch.no_grad()
    def ablate(self, model, trigger, t_target, calib_imgs, cfg):
        """Layer-level mean ablation: rank-normalised |cosine drops|.

        FIX 1: Sub-layer means computed on POISONED forward (distribution match).
        FIX 2: Rank on |drop| — both supportive (drop>0) and protective (drop<0)
               layers are important by magnitude.  The sign is recovered downstream
               by Phase 3 from the gradient direction.

        NOTE: The drift penalty (λ·clean_drift) was intentionally removed from
        the per-layer score to match the pseudo-code v3.  The drift term belongs
        to the greedy search (Phase 4).  To restore it later, re-subtract
        λ·clean_drift from each per-layer drop before rank-normalisation.
        ``CLEAN_DRIFT_PENALTY`` stays in CFG because Phase 4 still uses it.

        Returns (layer_scores, top_layers).
        """
        print("\n" + "=" * 60)
        print("PHASE 2: LAYER-LEVEL CAUSAL ABLATION")
        print("=" * 60)

        ad = get_adapter(model)
        blocks = list(ad.get_vision_transformer_blocks(model))
        lay = ad.get_seq_layout(model)
        n_layers = len(blocks)
        D = lay["width"]
        seq_len = lay["seq_len"]
        ap = get_apply_params(cfg, model)
        mini_bs = cfg.MINI_BS
        target_cos = _make_target_cos(model, t_target, mini_bs)


        # ── Baseline: triggered forward, no ablation ──
        poisoned = apply_trigger(calib_imgs, trigger, ap)
        cos_base = target_cos(poisoned)
        print(f"  cos_base (poisoned, intact) = {cos_base:.6f}")
        print(f"  Model: {n_layers}L  D={D}  seq_len={seq_len}")
        print(f"  Mini-batch: {mini_bs}  calib n={calib_imgs.size(0)}")

        # ── FIX 1: Sub-layer means on POISONED forward ──
        sub_means = compute_sub_layer_means(model, calib_imgs, cfg,
                                            trigger=trigger)

        # ── Layer-level mean ablation (raw cosine drop) ──
        attn_drops, mlp_drops = [], []
        for li in range(n_layers):
            blk = blocks[li]

            const_a = _ConstantOutput(sub_means[li]["attn_mean"], seq_len=seq_len)
            with ad.replace_sublayer(blk, "attn", const_a):
                cos_ab = target_cos(poisoned)
            a_drop = cos_base - cos_ab
            attn_drops.append(a_drop)

            const_m = _ConstantOutput(sub_means[li]["mlp_mean"], seq_len=seq_len)
            with ad.replace_sublayer(blk, "mlp", const_m):
                cos_ab = target_cos(poisoned)
            m_drop = cos_base - cos_ab
            mlp_drops.append(m_drop)

            # Sign tags for debugging
            sign_a = "supp" if a_drop > 0 else "prot"
            sign_m = "supp" if m_drop > 0 else "prot"
            print(f"  L{li:>3d}  attn_drop={a_drop:+.6f} [{sign_a}]  "
                  f"mlp_drop={m_drop:+.6f} [{sign_m}]")

        # ── FIX 2: Rank on |drop| (sign-aware) ──
        attn_mag = [abs(d) for d in attn_drops]
        mlp_mag = [abs(d) for d in mlp_drops]
        attn_norm = rankdata(attn_mag) / len(attn_mag)
        mlp_norm = rankdata(mlp_mag) / len(mlp_mag)
        layer_scores = {li: float(attn_norm[li] + mlp_norm[li])
                        for li in range(n_layers)}

        sorted_layers = sorted(layer_scores.items(), key=lambda kv: -kv[1])
        k_top = min(cfg.TOP_K_LAYERS, n_layers)
        top_layers = set(li for li, _ in sorted_layers[:k_top])

        print(f"\n  Rank-normalised layer scores (attn_rank + mlp_rank on |drop|):")
        for li in range(n_layers):
            mark = " *" if li in top_layers else ""
            print(f"    L{li:>3d}  attn_rank={attn_norm[li]:.4f}  "
                  f"mlp_rank={mlp_norm[li]:.4f}  "
                  f"score={layer_scores[li]:.4f}{mark}")
        print(f"  Top-{k_top} layers: {sorted(top_layers)}")

        cleanup_gpu()
        return layer_scores, top_layers


# ══════════════════════════════════════════════════════════════════════
#  PHASE 2B — COMPONENT-LEVEL CAUSAL ABLATION
# ══════════════════════════════════════════════════════════════════════
class ComponentAblator:
    """Phase 2B: component-level weight knockout → (component_scores,
    selected_components)."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    @torch.no_grad()
    def ablate(self, model, trigger, t_target, calib_imgs, top_layers, cfg):
        """Phase 2B: Component-level causal ablation via weight knockout.

        Within each selected layer, ranks every attention head and MLP by
        the cosine drop caused by zeroing their output contribution on
        POISONED inputs.  Per-layer quota prevents one layer from dominating.

        MLP bias caveat: zeroing c_proj.weight leaves c_proj.bias, so the MLP
        output becomes a constant bias vector (weak mean-ablation, not exact 0).

        Returns (component_scores, selected_components).
        """
        print("\n" + "=" * 60)
        print("PHASE 2B: COMPONENT-LEVEL CAUSAL ABLATION")
        print("=" * 60)

        ad = get_adapter(model)
        blocks = list(ad.get_vision_transformer_blocks(model))
        lay = ad.get_seq_layout(model)
        D = lay["width"]
        n_heads = lay["n_heads"]
        dk = D // n_heads
        ap = get_apply_params(cfg, model)
        mini_bs = cfg.MINI_BS

        target_cos = _make_target_cos(model, t_target, mini_bs)

        # ── Baseline on poisoned inputs ──
        poisoned = apply_trigger(calib_imgs, trigger, ap)
        c_base = target_cos(poisoned)
        print(f"  cos_base (poisoned, intact) = {c_base:.6f}")
        print(f"  Top layers: {sorted(top_layers)}  "
              f"({len(top_layers)} layers × {n_heads} heads + 1 MLP)")
        print(f"  Quota: {'per-layer' if cfg.PER_LAYER_QUOTA else 'global top-K'}  "
              f"TOP_K_COMPONENTS={cfg.TOP_K_COMPONENTS}")

        # ── Per-component weight knockout ──
        component_scores = {}
        for li in sorted(top_layers):
            blk = blocks[li]

            # Attention heads (out_proj column slice zeroed)
            for h in range(n_heads):
                lo, hie = h * dk, (h + 1) * dk
                with _knockout_head(blk, slice(lo, hie)):
                    c_ab = target_cos(poisoned)
                drop = c_base - c_ab
                component_scores[(li, h)] = drop
                sign = "supp" if drop > 0 else "prot"
                print(f"    L{li} H{h:<2d}  drop={drop:+.6f} [{sign}]")

            # MLP (c_proj.weight zeroed, bias retained)
            with _knockout_mlp(blk):
                c_ab = target_cos(poisoned)
            drop = c_base - c_ab
            component_scores[(li, -1)] = drop
            sign = "supp" if drop > 0 else "prot"
            print(f"    L{li} FFN   drop={drop:+.6f} [{sign}]")

        # ── Component selection ──
        if cfg.PER_LAYER_QUOTA:
            # Per-layer quota: every layer contributes equally
            k_per_layer = max(1, cfg.TOP_K_COMPONENTS // len(top_layers))
            selected_components = []
            for li in sorted(top_layers):
                layer_comps = [(key, score)
                               for key, score in component_scores.items()
                               if key[0] == li]
                # Sign-aware: rank by |drop| (same rationale as Stage A Fix 2)
                layer_comps.sort(key=lambda kv: abs(kv[1]), reverse=True)
                for (layer, head), score in layer_comps[:k_per_layer]:
                    selected_components.append(dict(
                        type="attn" if head >= 0 else "ffn",
                        layer=layer, head=head, score=score))
        else:
            # Global top-K by |drop|
            ranked_comp = sorted(component_scores.items(),
                                 key=lambda kv: abs(kv[1]), reverse=True)
            m_top = min(cfg.TOP_K_COMPONENTS, len(ranked_comp))
            selected_components = []
            for (layer, head), score in ranked_comp[:m_top]:
                selected_components.append(dict(
                    type="attn" if head >= 0 else "ffn",
                    layer=layer, head=head, score=score))

        n_attn = sum(1 for s in selected_components if s["type"] == "attn")
        n_ffn = len(selected_components) - n_attn
        print(f"\n  Component pool : {len(component_scores)} "
              f"(= {len(top_layers)} layers × ({n_heads} heads + 1 MLP))")
        print(f"  Selected : {len(selected_components)} "
              f"({n_attn} attn, {n_ffn} ffn)")
        for rank, s in enumerate(selected_components):
            tag = (f"L{s['layer']} H{s['head']}" if s["type"] == "attn"
                   else f"L{s['layer']} FFN")
            print(f"    #{rank + 1:>2d}  {tag:<10s}  drop={s['score']:+.6f}")

        cleanup_gpu()
        return component_scores, selected_components
```



## File: `adapters.py`
```py
"""
adapters.py — ModelAdapter: vision-language-model access layer.
==============================================================
Every place that used to reach into OpenAI-CLIP internals
(``model.visual.transformer.resblocks`` …) goes through a ModelAdapter
instead, so a new architecture = one new adapter class.

Two implementations:
  * OpenAICLIPAdapter — OpenAI ``clip`` package (ViT-B/16, ViT-L/14, …)
  * SigLIPAdapter    — HuggingFace ``transformers`` SigLIP
    (google/siglip-base-patch16-224, google/siglip-so400m-patch14-384)

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
    prefix tokens (OpenAI CLIP: 1 CLS; SigLIP: 0, mean-pooled patches).
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
class _SiglipAttnConst(nn.Module):
    """Wraps a constant-output module for SigLIP attention replacement:
    accepts ``hidden_states=``/``attention_mask=`` kwargs and returns the
    (hidden, weights) 2-tuple the encoder unpacks."""

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
            block.self_attn = _SiglipAttnConst(const_out)
            try:
                yield
            finally:
                block.self_attn = orig


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
    if mod.startswith("clip") or hasattr(model, "visual"):
        ad = OpenAICLIPAdapter()
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
    if "siglip" in mod.lower() or cls.__name__.startswith("Siglip") \
            or hasattr(block, "self_attn"):
        return SigLIPAdapter()
    return OpenAICLIPAdapter()


def load_model(cfg):
    """Load (model, preprocess) for the registry entry behind cfg.

    cfg.MODEL_ADAPTER selects the backend ('openai_clip' | 'siglip');
    cfg.MODEL is the OpenAI clip name, cfg.HF_NAME the HF hub id.
    """
    backend = getattr(cfg, "MODEL_ADAPTER", "openai_clip")
    if backend == "openai_clip":
        import clip
        model, preprocess = clip.load(cfg.MODEL, device=cfg.DEVICE)
        model = model.float()
    elif backend == "siglip":
        from transformers import AutoProcessor, SiglipModel
        model = SiglipModel.from_pretrained(cfg.HF_NAME)
        model = model.float().to(cfg.DEVICE)
        proc = AutoProcessor.from_pretrained(cfg.HF_NAME)
        model._adapter_processor = proc
        preprocess = get_adapter(model).get_preprocess(model)
    else:
        raise ValueError(f"Unknown model adapter {backend!r}")
    return model, preprocess
```



## File: `aggregate.py`
```py
"""
aggregate.py — Collect every run's bfa_results.json into CSV summaries.
=======================================================================
RESULTS.md "Code changes required" #7.  Walks the output root and writes:

  {out_root}/runs.csv     — one row per run (all variants / reps / seeds)
  {out_root}/summary.csv  — mean ± std per configuration (over reps/seeds)

Metric columns come from the ``metrics`` block written by the Phase-4
search (ASR@10, ASR@20, Flips→95%, …); identity columns from the run
path / ``config`` block.

Usage:
    python aggregate.py                     # ./grid_results
    python aggregate.py --out-root /path/to/grid_results
"""

import argparse
import csv
import json
import os
import statistics

ID_COLS = ["model", "dataset", "task", "target_cls", "seed", "rep",
           "variant", "trigger_size", "trigger_location"]
METRIC_COLS = ["asr_at_10", "asr_at_20", "flips_to_95", "flips_total",
               "asr_final", "cacc_final", "asr_base", "cacc_base",
               "search_time_sec"]
GROUP_COLS = ["model", "dataset", "task", "target_cls",
              "variant", "trigger_size", "trigger_location"]


def collect(out_root):
    rows = []
    for dirpath, _dirnames, filenames in os.walk(out_root):
        if "bfa_results.json" not in filenames:
            continue
        path = os.path.join(dirpath, "bfa_results.json")
        try:
            with open(path) as f:
                rec = json.load(f)
        except Exception as e:
            print(f"  SKIP {path}: {e}")
            continue
        cfg = rec.get("config", {})
        m = rec.get("metrics", {})
        rows.append(dict(
            model=cfg.get("MODEL_NAME"),
            dataset=cfg.get("DATASET"),
            task=cfg.get("TASK"),
            target_cls=cfg.get("TARGET_CLS"),
            seed=cfg.get("SEED"),
            rep=rec.get("rep", cfg.get("REP", 1)),
            variant=cfg.get("VARIANT"),
            trigger_size=cfg.get("TRIG_SZ"),
            trigger_location=rec.get("trigger_location",
                                     cfg.get("TRIGGER_LOCATION")),
            asr_at_10=m.get("asr_at_10"),
            asr_at_20=m.get("asr_at_20"),
            flips_to_95=m.get("flips_to_95"),
            flips_total=m.get("flips_total"),
            asr_final=m.get("asr_final"),
            cacc_final=m.get("cacc_final"),
            asr_base=m.get("asr_base"),
            cacc_base=m.get("cacc_base"),
            search_time_sec=rec.get("search_time_sec"),
            out_dir=dirpath))
    return rows


def write_runs_csv(rows, path):
    cols = ID_COLS + METRIC_COLS + ["out_dir"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: tuple(str(r[c]) for c in ID_COLS)):
            w.writerow({c: r.get(c) for c in cols})
    print(f"  runs.csv    → {path}  ({len(rows)} runs)")


def _fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return ""
    if len(vals) == 1:
        return f"{vals[0]:g}"
    return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}"


def write_summary_csv(rows, path):
    groups = {}
    for r in rows:
        key = tuple(str(r[c]) for c in GROUP_COLS)
        groups.setdefault(key, []).append(r)

    cols = GROUP_COLS + ["n_runs"] + \
        [f"{c} (mean ± std)" for c in METRIC_COLS]
    lines = []
    for key in sorted(groups):
        rs = groups[key]
        row = dict(zip(GROUP_COLS, key))
        row["n_runs"] = len(rs)
        for c in METRIC_COLS:
            row[f"{c} (mean ± std)"] = _fmt([r.get(c) for r in rs])
        lines.append(row)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(lines)
    print(f"  summary.csv → {path}  ({len(lines)} configs)")

    # compact echo of the main table columns
    print(f"\n  {'model':<16} {'dataset':<11} {'variant':<6} {'loc':<13} "
          f"{'ASR@10':>15} {'ASR@20':>15} {'Flips→95%':>15} {'CACC':>15}")
    for row in lines:
        print(f"  {row['model']:<16} {row['dataset']:<11} "
              f"{row['variant']:<6} {str(row['trigger_location']):<13} "
              f"{row['asr_at_10 (mean ± std)']:>15} "
              f"{row['asr_at_20 (mean ± std)']:>15} "
              f"{row['flips_to_95 (mean ± std)']:>15} "
              f"{row['cacc_final (mean ± std)']:>15}")


def main():
    p = argparse.ArgumentParser(description="Aggregate run results")
    p.add_argument("--out-root", default="./grid_results")
    args = p.parse_args()
    out_root = args.out_root

    rows = collect(out_root)
    if not rows:
        print(f"No bfa_results.json found under {out_root}")
        return
    write_runs_csv(rows, os.path.join(out_root, "runs.csv"))
    write_summary_csv(rows, os.path.join(out_root, "summary.csv"))


if __name__ == "__main__":
    main()
```



## File: `analysis.py`
```py
"""
analysis.py — Phase 5 post-attack analysis (5a–5d).
===================================================
5a ΔW semantic alignment · 5b per-class CACC breakdown ·
5c ASR/CACC curves · 5d target-class CACC.
"""

import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation import encode_image_batched
from trigger import apply_trigger, get_apply_params
from utils import cleanup_gpu


class PostAttackAnalyzer:
    """Phase 5: post-attack analysis and plots."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    # ════════════════════════════════════════════════════════════════
    #  5a — ΔW SEMANTIC ALIGNMENT
    # ════════════════════════════════════════════════════════════════
    def alignment(self, model, selected_heads, original_weights,
                  t_target, cfg):
        print("\n" + "=" * 60)
        print("PHASE 5a: ΔW SEMANTIC ALIGNMENT")
        print("=" * 60)
        vis = model.visual
        blocks = list(vis.transformer.resblocks)
        proj = vis.proj.data.float()
        n_heads = blocks[0].attn.num_heads
        D = vis.transformer.width
        dk = D // n_heads
        target = t_target.detach().float()
        alignments = []
        for s in selected_heads:
            li = s["layer"]
            blk = blocks[li]
            if s["type"] == "attn":
                hi = s["head"]
                lo, hie = hi * dk, (hi + 1) * dk
                W_in_now = blk.attn.in_proj_weight.data[2 * D + lo:2 * D + hie].float()
                W_out_now = blk.attn.out_proj.weight.data[:, lo:hie].float()
                orig_in = original_weights.get((li, hi, "V"))
                orig_out = original_weights.get((li, hi, "O"))
            elif s["type"] == "ffn":
                W_in_now = blk.mlp.c_fc.weight.data.float()
                W_out_now = blk.mlp.c_proj.weight.data.float()
                orig_in = original_weights.get((li, -1, "fc"))
                orig_out = original_weights.get((li, -1, "proj"))
            else:
                continue
            if orig_in is None or orig_out is None:
                continue
            dIn = W_in_now - orig_in.float()
            dOut = W_out_now - orig_out.float()
            if dIn.abs().max() < 1e-12 and dOut.abs().max() < 1e-12:
                continue
            dVO = dIn.T @ orig_out.float().T + orig_in.float().T @ dOut.T
            if dVO.abs().max() < 1e-12:
                continue
            U, S, Vh = torch.linalg.svd(dVO, full_matrices=False)
            v_delta = Vh[0]
            v_hat = F.normalize((v_delta @ proj).unsqueeze(0), dim=-1).squeeze()
            cos = F.cosine_similarity(v_hat, target, dim=0).item()
            tag = f"L{li} H{s['head']}" if s["type"] == "attn" else f"L{li} FFN"
            alignments.append(dict(layer=li, type=s["type"], cosine=cos))
            print(f"  {tag}: cos(ΔW, target) = {cos:+.4f}")
        if alignments:
            avg = np.mean([a["cosine"] for a in alignments])
            print(f"\n  Mean alignment: {avg:+.4f}")
        else:
            print("  No weight changes detected.")
        return alignments

    # ════════════════════════════════════════════════════════════════
    #  5b — PER-CLASS CACC BREAKDOWN
    # ════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def per_class(self, model, text_emb, dataset, test_idx,
                  target_cls, device, class_names,
                  model_clean_state=None):
        print("\n" + "=" * 60)
        print("PHASE 5b: PER-CLASS CACC BREAKDOWN")
        print("=" * 60)

        class_indices = defaultdict(list)
        for i in test_idx:
            class_indices[dataset[i][1]].append(i)

        def _eval_all():
            results = {}
            for cls_idx in sorted(class_indices.keys()):
                idxs = class_indices[cls_idx]
                loader = DataLoader(Subset(dataset, idxs),
                                    batch_size=4, shuffle=False)
                hit = tot = 0
                for imgs, labels in loader:
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    emb = F.normalize(encode_image_batched(model, imgs), dim=-1)
                    hit += ((emb @ text_emb.T).argmax(1) == labels).sum().item()
                    tot += imgs.size(0)
                acc = hit / max(tot, 1) * 100
                name = class_names[cls_idx] if cls_idx < len(class_names) else "?"
                results[cls_idx] = dict(cls=cls_idx, name=name, acc=acc, n=tot)
            return results

        attacked_results = _eval_all()

        clean_results = {}
        if model_clean_state is not None:
            saved = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(model_clean_state)
            print("  Evaluating clean-model baseline (quantized, no flips)...")
            clean_results = _eval_all()
            model.load_state_dict(saved)
            del saved
            cleanup_gpu()

        rows = []
        for cls_idx in sorted(class_indices.keys()):
            row = dict(attacked_results[cls_idx])
            if cls_idx in clean_results:
                row["clean_acc"] = clean_results[cls_idx]["acc"]
                row["delta"] = row["acc"] - row["clean_acc"]
            else:
                row["clean_acc"] = None
                row["delta"] = None
            rows.append(row)

        rows.sort(key=lambda r: r["acc"])

        if clean_results:
            print(f"\n  {'Class':<30s} {'Clean':>8s} {'Attacked':>9s} "
                  f"{'Δ':>8s}  {'n':>5s}")
            print(f"  {'─' * 65}")
            for r in rows[:20]:
                c_str = f"{r['clean_acc']:.1f}%" if r["clean_acc"] is not None else "—"
                d_str = f"{r['delta']:+.1f}%" if r["delta"] is not None else "—"
                print(f"  {r['name']:<30s} {c_str:>8s} {r['acc']:>8.1f}% "
                      f"{d_str:>8s}  {r['n']:>5d}")

            degraded = [r for r in rows if r["delta"] is not None and r["delta"] < -5.0]
            if degraded:
                degraded.sort(key=lambda r: r["delta"])
                print(f"\n  Classes with >5% degradation (attack vs clean):")
                for r in degraded[:10]:
                    print(f"    {r['name']:<30s} {r['clean_acc']:.1f}% → "
                          f"{r['acc']:.1f}%  (Δ = {r['delta']:+.1f}%)")

            deltas = [r["delta"] for r in rows if r["delta"] is not None]
            if deltas:
                print(f"\n  Mean Δ across all classes: {np.mean(deltas):+.2f}%")
                n_worse = sum(1 for d in deltas if d < -2.0)
                n_better = sum(1 for d in deltas if d > 2.0)
                n_same = len(deltas) - n_worse - n_better
                print(f"  Classes: {n_worse} degraded >2%, "
                      f"{n_same} stable (±2%), "
                      f"{n_better} improved >2%")
        else:
            print(f"\n  {'Class':<30s} {'CACC':>7s}")
            print(f"  {'─' * 40}")
            for r in rows[:20]:
                print(f"  {r['name']:<30s} {r['acc']:>6.1f}%")

        return rows

    # ════════════════════════════════════════════════════════════════
    #  5d — TARGET CLASS CACC
    # ════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def target_class_cacc(self, model, text_emb, trigger,
                          dataset, target_holdout,
                          target_cls, cfg, model_clean_state=None):
        print("\n" + "=" * 60)
        print("PHASE 5d: TARGET CLASS CACC")
        print("=" * 60)

        if len(target_holdout) == 0:
            print("  No target holdout images available.")
            return {}

        ap = get_apply_params(cfg, model)
        loader = DataLoader(Subset(dataset, target_holdout), batch_size=4,
                            shuffle=False, num_workers=2)
        target_name = f"Class {target_cls}"
        results = {}

        if model_clean_state is not None:
            saved = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(model_clean_state)

            hit_clean = tot = 0
            for imgs, labels in loader:
                imgs = imgs.to(cfg.DEVICE)
                labels = labels.to(cfg.DEVICE)
                emb = F.normalize(encode_image_batched(model, imgs), dim=-1)
                hit_clean += ((emb @ text_emb.T).argmax(1) == labels).sum().item()
                tot += imgs.size(0)
            acc_clean = hit_clean / max(tot, 1) * 100
            print(f"\n  [Clean model — baseline]")
            print(f"    {target_name} clean CACC: {acc_clean:.1f}%  (n={tot})")
            results["clean_model"] = {"clean_cacc": acc_clean, "n": tot}
            model.load_state_dict(saved)

        hit_clean = hit_trig = tot = 0
        for imgs, labels in loader:
            imgs = imgs.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)
            emb_c = F.normalize(encode_image_batched(model, imgs), dim=-1)
            hit_clean += ((emb_c @ text_emb.T).argmax(1) == labels).sum().item()
            poisoned = apply_trigger(imgs, trigger, ap)
            emb_t = F.normalize(encode_image_batched(model, poisoned), dim=-1)
            hit_trig += ((emb_t @ text_emb.T).argmax(1) == labels).sum().item()
            tot += imgs.size(0)

        acc_clean = hit_clean / max(tot, 1) * 100
        acc_trig = hit_trig / max(tot, 1) * 100

        print(f"\n  [Attacked model]")
        print(f"    {target_name} clean CACC:    {acc_clean:.1f}%  (n={tot})")
        print(f"    {target_name} triggered CACC: {acc_trig:.1f}%  (n={tot})")
        results["attacked"] = {"clean_cacc": acc_clean,
                               "triggered_cacc": acc_trig, "n": tot}

        if model_clean_state is not None and "clean_model" in results:
            delta = acc_clean - results["clean_model"]["clean_cacc"]
            print(f"\n    Δ clean CACC (attack - baseline): {delta:+.1f}%")
            if abs(delta) < 2.0:
                print(f"    → Stealthy: target class accuracy preserved")
            else:
                print(f"    → WARNING: target class accuracy degraded!")

        return results

    # ════════════════════════════════════════════════════════════════
    #  5c — ASR / CACC CURVES
    # ════════════════════════════════════════════════════════════════
    def curves(self, checkpoints, path):
        if len(checkpoints) < 2:
            print("  Not enough checkpoints for curve plot.")
            return
        flips = [c[0] for c in checkpoints]
        asr = [c[1] for c in checkpoints]
        cacc = [c[2] for c in checkpoints]
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.set_xlabel("Bit flips")
        ax1.set_ylabel("ASR (%)", color="tab:red")
        ax1.plot(flips, asr, "o-", color="tab:red", label="ASR")
        ax1.tick_params(axis="y", labelcolor="tab:red")
        ax2 = ax1.twinx()
        ax2.set_ylabel("CACC (%)", color="tab:blue")
        ax2.plot(flips, cacc, "s--", color="tab:blue", label="CACC")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_title("ASR / CACC vs Bit Flips — Hybrid Ranking")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved curves → {path}")
```



## File: `config.py`
```py
"""
config.py — Model / dataset / path registry (config.json).
=========================================================
The registry holds ONLY structural configuration:
  * models   — clip_name + patch-size hint per model
  * datasets — loader, root, split sizes, class-text dir per dataset
  * paths    — output_root, lp_head_root

Everything else is a CLI argument — the parser lives in experiment.py,
which builds the cfg (experiment.build_config) and passes it to main.py.
"""

import json
import os

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "config.json")

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)   # proposed_method/


def resolve_path(path):
    """Registry paths are relative to the project dir (proposed_method/) —
    the '../data' convention of the reference code — independent of CWD."""
    return path if os.path.isabs(path) else \
        os.path.normpath(os.path.join(PROJECT_DIR, path))


def load_registry(path=REGISTRY_PATH):
    """Load the model/dataset/path registry."""
    with open(path) as f:
        return json.load(f)


class CFG:
    """Attribute bag assembled by experiment.build_config()."""
    pass
```



## File: `data.py`
```py
"""
data.py — Dataset loading, class names, splitting, tensor helpers.
=================================================================
Dataset registry: supporting a new dataset = adding one branch to
``load_dataset``.  All dataset-specific logic — including class-name
cleaning/renames — lives HERE, so load_dataset returns prompt-ready
class names and target.py stays dataset-agnostic.

load_dataset(preprocess) → (dataset, class_names)
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import resolve_path

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Caltech-101 category names that don't work with "a photo of a {name}" ──
CALTECH101_RENAMES = {
    "Faces":        "human face",
    "Faces_easy":   "close-up human face photograph",
    "Motorbikes":   "motorbike",
    "Leopards":     "leopard",
    "ketch":        "ketch ship",
    "schooner":     "schooner ship",
    "cup":          "drinking cup",
    "lamp":         "table lamp",
    "laptop":       "laptop computer",
    "wild_cat":     "wild cat",
    "flamingo":     "flamingo bird",
    "lotus":        "lotus flower",
    "camera":       "camera device",
    "chair":        "chair furniture",
    "lobster":      "lobster animal",
    "crocodile":    "crocodile reptile",
    "octopus":      "octopus sea creature",
    "scissors":     "pair of scissors",
    "stapler":      "stapler tool",
    "umbrella":     "umbrella rain gear",
    "airplanes":    "airplane",
    "butterfly":    "butterfly insect",
    "car_side":     "side view of a car",
    "cellphone":    "cell phone",
    "cougar_body":  "cougar animal",
    "cougar_face":  "cougar face",
    "crab":         "crab animal",
    "crayfish":     "crayfish animal",
    "crocodile_head": "crocodile head",
    "dollar_bill":  "dollar bill",
    "dragonfly":    "dragonfly insect",
    "electric_guitar": "electric guitar",
    "elephant":     "elephant animal",
    "emu":          "emu bird",
    "euphonium":    "euphonium instrument",
    "ewer":         "ewer pitcher",
    "ferry":        "ferry boat",
    "garfield":     "garfield cartoon cat",
    "gerenuk":      "gerenuk antelope",
    "gramophone":   "gramophone",
    "hawksbill":    "hawksbill sea turtle",
    "headphone":    "headphones",
    "hedgehog":     "hedgehog animal",
    "inline_skate": "inline skate",
    "joshua_tree":  "joshua tree",
    "kangaroo":     "kangaroo animal",
    "mandolin":     "mandolin instrument",
    "metronome":    "metronome",
    "minaret":      "minaret tower",
    "nautilus":     "nautilus shell",
    "pagoda":       "pagoda building",
    "panda":        "panda bear",
    "pigeon":       "pigeon bird",
    "pizza":        "pizza food",
    "platypus":     "platypus animal",
    "pyramid":      "pyramid structure",
    "revolver":     "revolver gun",
    "rhino":        "rhinoceros animal",
    "rooster":      "rooster chicken",
    "saxophone":    "saxophone instrument",
    "snoopy":       "snoopy cartoon dog",
    "soccer_ball":  "soccer ball",
    "starfish":     "starfish sea creature",
    "stop_sign":    "stop sign",
    "strawberry":   "strawberry fruit",
    "sunflower":    "sunflower plant",
    "tick":         "tick insect",
    "trilobite":    "trilobite fossil",
    "watch":        "wrist watch",
    "water_lilly":  "water lily flower",
    "wheelchair":   "wheelchair",
    "wild_grog_face": "wild grog face",
    "windsor_chair": "windsor chair",
    "wrench":       "wrench tool",
    "yin_yang":     "yin yang symbol",
}


def _first_name(c):
    """ImageNet meta.bin synonym entry → primary prompt name (first synonym)."""
    if isinstance(c, (tuple, list)):
        c = c[0]
    return str(c).replace("_", " ").lower()


class DataModule:
    """Dataset registry + split preparation + tensor helpers."""

    def __init__(self, cfg):
        self.cfg = cfg

    # ════════════════════════════════════════════════════════════════
    #  LOAD DATASET — returns (dataset, class_names)
    #  Registry-driven: the loader comes from cfg.DATASET_CFG (config.json),
    #  so supporting a new dataset = one registry entry (+ loader branch).
    # ════════════════════════════════════════════════════════════════
    def load_dataset(self, preprocess):
        cfg = self.cfg
        entry = cfg.DATASET_CFG
        tf = transforms.Compose([lambda im: im.convert("RGB"), preprocess])
        loader = entry["loader"]
        root = resolve_path(entry["root"])

        if loader == "Caltech101":
            ds = datasets.Caltech101(root=root, download=cfg.DOWNLOAD,
                                     transform=tf,
                                     target_type=entry.get("target_type", "category"))
            raw = list(ds.categories)
            names = [CALTECH101_RENAMES.get(n, n.replace("_", " ").lower())
                     for n in raw]

        elif loader in ("CIFAR10", "CIFAR100"):
            cls = datasets.CIFAR10 if loader == "CIFAR10" else datasets.CIFAR100
            train = entry.get("split", "train") == "train"
            ds = cls(root=root, train=train, download=cfg.DOWNLOAD,
                     transform=tf)
            names = [n.replace("_", " ").lower() for n in ds.classes]

        elif loader == "ImageNet":
            ds = datasets.ImageNet(root=root, split=entry.get("split", "val"))
            names = [_first_name(c) for c in
                     self._names(ds, fallback_n=entry.get("n_classes", 1000))]

        elif loader == "ImageFolder":
            ds = datasets.ImageFolder(root=root, transform=tf)
            names = [n.replace("_", " ").lower() for n in ds.classes]

        else:
            raise ValueError(f"Unknown loader {loader!r} in registry entry "
                             f"for dataset {cfg.DATASET!r}")

        # ── canonical class names from class_text_dir (BadCLIP convention:
        #    one name per line in classes.txt, index-aligned with labels) ──
        file_names = self.load_class_texts(entry)
        if file_names is not None:
            if len(file_names) != len(names):
                extra = sorted(set(map(str, names)) - set(map(str, file_names)))
                hint = ("  For Caltech101 a stray empty folder inside "
                        "101_ObjectCategories shifts the labels — remove it "
                        "and re-run make_class_texts.py."
                        if loader == "Caltech101" else "")
                raise ValueError(
                    f"Class-name mismatch for {cfg.DATASET}: classes.txt has "
                    f"{len(file_names)} names but the dataset has "
                    f"{len(names)} classes.  Dataset-only entries: "
                    f"{extra[:5]}...{hint}")
            names = file_names

        print(f"  Dataset : {cfg.DATASET}  (loader={loader})")
        print(f"  Samples : {len(ds)}")
        print(f"  Classes : {len(names)}")
        return ds, names

    def load_class_texts(self, entry):
        """Class names from {class_text_dir}/classes.txt — one name per line,
        index-aligned with label ids.  Resolved relative to this package.
        Returns None when the file is absent."""
        ctd = entry.get("class_text_dir")
        if not ctd:
            return None
        path = ctd if os.path.isabs(ctd) else os.path.join(PACKAGE_DIR, ctd)
        fpath = os.path.join(path, "classes.txt")
        if not os.path.isfile(fpath):
            return None
        with open(fpath) as f:
            return [ln.strip() for ln in f if ln.strip()]

    @staticmethod
    def _names(ds, fallback_n=None):
        for attr in ("classes", "categories", "_classes"):
            v = getattr(ds, attr, None)
            if v:
                return list(v)
        c2i = getattr(ds, "class_to_idx", None)
        if c2i:
            return [k for k, _ in sorted(c2i.items(), key=lambda x: x[1])]
        n = fallback_n or 1000
        return [f"class_{i}" for i in range(n)]

    # ════════════════════════════════════════════════════════════════
    #  LABELS / TENSORS / SPLITTING
    # ════════════════════════════════════════════════════════════════
    @staticmethod
    def get_all_labels(ds):
        if hasattr(ds, "targets"):
            return np.array(ds.targets)
        if hasattr(ds, "y"):
            return np.array(ds.y)
        if hasattr(ds, "_labels"):
            return np.array(ds._labels)
        return np.array([ds[i][1] for i in range(len(ds))])

    @staticmethod
    def load_tensors(dataset, indices, device, batch_size=64, workers=2):
        loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                            shuffle=False, num_workers=workers)
        imgs_list, lbl_list = [], []
        for imgs, labels in loader:
            imgs_list.append(imgs.to(device, non_blocking=True))
            lbl_list.append(labels.to(device, non_blocking=True))
        return torch.cat(imgs_list), torch.cat(lbl_list)

    @staticmethod
    def make_loader(dataset, indices, batch_size, shuffle=False, workers=2):
        return DataLoader(Subset(dataset, indices), batch_size=batch_size,
                          shuffle=shuffle, num_workers=workers,
                          pin_memory=torch.cuda.is_available())

    def prepare_data_splits(self, ds):
        cfg = self.cfg
        rng = np.random.RandomState(42)
        labels = self.get_all_labels(ds)

        target_indices = [i for i, lbl in enumerate(labels) if lbl == cfg.TARGET_CLS]
        other_indices  = [i for i, lbl in enumerate(labels) if lbl != cfg.TARGET_CLS]

        rng.shuffle(target_indices)
        rng.shuffle(other_indices)
        target_sel = target_indices[:cfg.N_TARGET_IMAGES]
        target_holdout = target_indices[cfg.N_TARGET_IMAGES:]
        ptr = 0
        calib_idx = other_indices[ptr:ptr + cfg.N_CALIB]
        ptr += cfg.N_CALIB
        grad_idx = other_indices[ptr:ptr + cfg.N_GRAD]
        ptr += cfg.N_GRAD
        eval_idx = other_indices[ptr:ptr + cfg.N_EVAL]
        ptr += cfg.N_EVAL
        test_idx = other_indices[ptr:]
        print(f"  Target class images : {len(target_sel):>5d}")
        print(f"  Target holdout      : {len(target_holdout):>5d}")
        print(f"  Calibration         : {len(calib_idx):>5d}")
        print(f"  Gradient            : {len(grad_idx):>5d}")
        print(f"  Evaluation          : {len(eval_idx):>5d}")
        print(f"  Test                : {len(test_idx):>5d}")
        return target_sel, calib_idx, grad_idx, eval_idx, test_idx, target_holdout
```



## File: `evaluation.py`
```py
"""
evaluation.py — ASR / CACC / linear-probe evaluation helpers.
=============================================================
encode_image_batched is shared plumbing (mini-batch encoding);
Evaluator holds the quick/full evaluation routines and the linear-probe
evaluation (lp_evaluate, TASKS.md §2).
"""

import os

import torch
import torch.nn.functional as F

from adapters import encode_image as adapter_encode_image
from trigger import apply_trigger


# ══════════════════════════════════════════════════════════════════════
#  MINI-BATCH IMAGE ENCODING
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def encode_image_batched(model, images, mini_bs=None):
    if mini_bs is None:
        mini_bs = images.size(0)
    parts = []
    for i in range(0, images.size(0), mini_bs):
        parts.append(adapter_encode_image(model, images[i:i + mini_bs]))
    return torch.cat(parts, dim=0)


# ══════════════════════════════════════════════════════════════════════
#  EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════
class Evaluator:
    """Quick (tensor-set), full (loader) and linear-probe evaluation.

    All routines take an ApplyParams record (trigger.py) instead of loose
    size/mode/opacity args, so trigger location and the model's
    normalisation stats travel with the call."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    @torch.no_grad()
    def quick_evaluate(self, model, text_emb, trigger, imgs, labels,
                       target_cls, ap, mini_bs=None):
        if mini_bs is None:
            mini_bs = 4
        if trigger is not None:
            poisoned = apply_trigger(imgs, trigger, ap)
            emb_po = F.normalize(encode_image_batched(model, poisoned, mini_bs), dim=-1)
        else:
            emb_po = None
        emb_cl = F.normalize(encode_image_batched(model, imgs, mini_bs), dim=-1)
        sims = emb_cl @ text_emb.T
        cacc = (sims.argmax(1) == labels).float().mean().item() * 100
        if emb_po is not None:
            asr = ((emb_po @ text_emb.T).argmax(1) == target_cls).float().mean().item() * 100
        else:
            asr = 0.0
        return asr, cacc

    @torch.no_grad()
    def full_evaluate(self, model, text_emb, trigger, loader, target_cls,
                      ap, device):
        hit_ca = tot = 0
        hit_asr = 0
        cos_cl = []
        cos_po = []
        tgt_vec = text_emb[target_cls]

        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            ec = F.normalize(encode_image_batched(model, imgs), dim=-1)
            hit_ca += ((ec @ text_emb.T).argmax(1) == labels).sum().item()
            tot += imgs.size(0)
            cos_cl.append(F.cosine_similarity(ec, tgt_vec, dim=-1).cpu())
            if trigger is not None:
                poisoned = apply_trigger(imgs, trigger, ap)
                ep = F.normalize(encode_image_batched(model, poisoned), dim=-1)
                hit_asr += ((ep @ text_emb.T).argmax(1) == target_cls).sum().item()
                cos_po.append(F.cosine_similarity(ep, tgt_vec, dim=-1).cpu())
            del ec, imgs, labels
        n = max(tot, 1)
        result = dict(cacc=hit_ca / n * 100,
                      cos_clean=torch.cat(cos_cl).mean().item())
        if trigger is not None:
            result["asr"] = hit_asr / n * 100
            result["cos_poison"] = torch.cat(cos_po).mean().item()
        else:
            result["asr"] = 0.0
            result["cos_poison"] = 0.0
        return result

    # ════════════════════════════════════════════════════════════════
    #  LINEAR-PROBE EVALUATION (TASKS.md §2, RESULTS.md §2)
    # ════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def lp_evaluate(self, model, loader, device, lp_head_path,
                    trigger=None, target_cls=None, ap=None, mini_bs=None):
        """Evaluate the saved linear-probe head on this model's features.

        loader        : DataLoader yielding (images, labels) — used for BOTH
                        clean and triggered passes (the trigger is applied on
                        the fly when ``trigger`` is given).
        lp_head_path  : checkpoint from linear_probe.py
                        ({weight, bias, meta}); the same head is used for
                        every attack method (fair comparison, RESULTS.md §2).
        ap            : ApplyParams for the triggered pass (required with
                        ``trigger``).

        Returns dict: lp_cacc (clean accuracy, %), lp_asr (fraction of
        triggered features classified as ``target_cls``, %), n, plus
        checkpoint meta (probe top-1 sanity) when available.

        Features follow the linear_probe.py convention: L2-normalised
        ``encode_image_batched`` embeddings.
        """
        if not os.path.isfile(lp_head_path):
            raise FileNotFoundError(
                f"Linear-probe head not found: {lp_head_path} — train it "
                f"first (python linear_probe.py).")
        if trigger is not None and (target_cls is None or ap is None):
            raise ValueError("lp_evaluate with a trigger needs target_cls "
                             "and ApplyParams (ap)")
        if mini_bs is None:
            mini_bs = 4

        head = torch.load(lp_head_path, map_location=device, weights_only=False)
        weight = head["weight"].to(device).float()
        bias = head.get("bias")
        bias = bias.to(device).float() if bias is not None else None
        probe_meta = head.get("meta", {})

        def classify(imgs):
            feats = F.normalize(
                encode_image_batched(model, imgs, mini_bs), dim=-1)
            return F.linear(feats, weight, bias).argmax(dim=-1)

        hit_ca = hit_asr = tot = 0
        clean_as_target = 0
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred_cl = classify(imgs)
            hit_ca += (pred_cl == labels).sum().item()
            if target_cls is not None:
                clean_as_target += (pred_cl == target_cls).sum().item()
            if trigger is not None:
                poisoned = apply_trigger(imgs, trigger, ap)
                pred_po = classify(poisoned)
                hit_asr += (pred_po == target_cls).sum().item()
            tot += imgs.size(0)
            del imgs, labels

        n = max(tot, 1)
        result = dict(
            lp_cacc=hit_ca / n * 100,
            lp_asr=(hit_asr / n * 100) if trigger is not None else 0.0,
            lp_clean_as_target=clean_as_target / n * 100,
            n=n,
            lp_head=lp_head_path,
            probe_test_top1=probe_meta.get("eval", {}).get("test_top1"))
        return result
```



## File: `experiment.py`
```py
"""
experiment.py — Single-run CLI: argument parser + config builder.
================================================================
One invocation = one attack run.  experiment.py parses the arguments,
builds the cfg from registry + arguments, and passes it to main.py,
which calls AttackPipeline.

Multi-run grids are plain shell scripts that invoke this command once
per configuration; on CUDA OOM the run is recorded in oom_pending.json
(main.py) and the process exits cleanly so the script continues with
the next configuration.

Usage:
    python experiment.py --model ViT-B/16 --dataset Caltech101 --seed 42
    python experiment.py --model ViT-L/14 --dataset CIFAR10 --seed 43 \\
                         --trigger-size 14 --variant trig14

Shell multi-run example:
    for seed in 42 43 44; do
        python experiment.py --model ViT-B/16 --dataset Caltech101 --seed $seed
    done

python experiment.py --help   →   all options (defaults = tuned values).
"""

import argparse
import os

import torch

from config import CFG, load_registry, resolve_path
from main import main as run_attack
from utils import model_tag, resolve_device


# ══════════════════════════════════════════════════════════════════════
#  ARGUMENTS (defaults = tuned values from main_new_pipeline.py)
# ══════════════════════════════════════════════════════════════════════
def parse_args():
    reg = load_registry()
    parser = argparse.ArgumentParser(
        description="CLIP-FLIP hybrid bit-flip attack — single run")

    sel = parser.add_argument_group("run selection")
    sel.add_argument("--model", default="ViT-B/16",
                     choices=sorted(reg["models"]),
                     help="model registry name (default: ViT-B/16)")
    sel.add_argument("--dataset", default="Caltech101",
                     choices=sorted(reg["datasets"]),
                     help="dataset registry name (default: Caltech101)")
    sel.add_argument("--task", default="zs", choices=["zs", "lp", "i2t"],
                     help="evaluation task: zs = zero-shot (implemented) | "
                          "lp = linear-probe (needs a trained probe head) | "
                          "i2t = image-to-text retrieval (future — see TASKS.md)")
    sel.add_argument("--seed", type=int, default=42,
                     help="random seed (default: 42)")
    sel.add_argument("--rep", type=int, default=1,
                     help="repetition index for same-seed repeats "
                          "(default: 1); part of the output path")

    run = parser.add_argument_group("run")
    run.add_argument("--target-cls", type=int, default=3,
                     help="target class index (default: 3)")
    run.add_argument("--trigger-type", default="center_add",
                     choices=["center_add", "corner_add", "opacity", "replace"],
                     help="trigger mode (default: center_add)")
    run.add_argument("--trigger-location", default=None,
                     choices=["center", "top_left", "top_right",
                              "bottom_left", "bottom_right"],
                     help="trigger placement (train + eval); default: legacy "
                          "mapping of --trigger-type (center_add → center, "
                          "others → bottom-right)")
    run.add_argument("--trigger-size", type=int, default=14,
                     help="trigger patch size in pixels (default: 14)")
    run.add_argument("--max-flips", type=int, default=80,
                     help="bit-flip budget (default: 80)")
    run.add_argument("--asr-target", type=float, default=95.0,
                     help="early-stop ASR target in %% (default: 95; "
                          "0 disables the early stop)")
    run.add_argument("--variant", default="full",
                     choices=["full", "d", "a", "b", "c", "clip_flip"],
                     help="ablation variant (ABLATION.md): full/d/clip_flip = "
                          "proposed method unchanged | a = no Phase 2 pruning | "
                          "b = no clean penalty | c = poison-only gradient")
    run.add_argument("--device", default="auto",
                     help="cuda | cpu | auto (default: auto)")
    run.add_argument("--download", action="store_true",
                     help="allow torchvision to download datasets")
    run.add_argument("--out-dir", default=None,
                     help="explicit output dir (overrides the derived layout)")
    run.add_argument("--out-root", default=None,
                     help="override registry paths.output_root")

    data = parser.add_argument_group("data & splits")
    data.add_argument("--batch", type=int, default=8)
    data.add_argument("--mini-bs", type=int, default=4)
    data.add_argument("--workers", type=int, default=2)
    data.add_argument("--n-target-images", type=int, default=None,
                      help="override registry splits.n_trigger "
                           "(target-class images)")
    data.add_argument("--n-calib", type=int, default=None,
                      help="override registry splits.n_trigger "
                           "(trigger-training / ablation calibration)")
    data.add_argument("--n-grad", type=int, default=None,
                      help="override registry splits.n_gradient")
    data.add_argument("--n-eval", type=int, default=None,
                      help="override registry splits.n_eval")

    p1 = parser.add_argument_group("phase 1 — trigger training")
    p1.add_argument("--p1-epochs", type=int, default=200)
    p1.add_argument("--p1-lr", type=float, default=0.02)
    p1.add_argument("--tv-weight", type=float, default=0.0)
    p1.add_argument("--temperature", type=float, default=0.07)
    p1.add_argument("--epsilon", type=float, default=8.0 / 255.0,
                    help="pixel clamp (unused for additive triggers)")
    p1.add_argument("--opacity", type=float, default=1.0)
    p1.add_argument("--lambda-vis", type=float, default=0.0)

    p2 = parser.add_argument_group("phase 2 — ablation / pruning")
    p2.add_argument("--top-k-layers", type=int, default=4)
    p2.add_argument("--top-k-components", type=int, default=30)
    p2.add_argument("--per-layer-quota",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="per-layer component quota (default: on; "
                         "--no-per-layer-quota = global top-K)")

    rank = parser.add_argument_group("phase 3/4 — ranking & search")
    rank.add_argument("--gradient-beta", type=float, default=10.0)
    rank.add_argument("--sens-top-m", type=int, default=150)
    rank.add_argument("--eval-bs", type=int, default=32)
    rank.add_argument("--cands-per-round", type=int, default=30)
    rank.add_argument("--plateau-patience", type=int, default=5)
    rank.add_argument("--refresh-every", type=int, default=5)
    rank.add_argument("--checkpoint-every", type=int, default=10)
    rank.add_argument("--clean-drift-penalty", type=float, default=10.0)
    rank.add_argument("--max-clean-drift", type=float, default=0.02)

    misc = parser.add_argument_group("misc")
    misc.add_argument("--quant-tolerance", type=float, default=2.0)
    misc.add_argument("--lp-head", default=None,
                      help="linear-probe checkpoint for --task lp (default: "
                           "{lp_head_root}/{model_tag}_{dataset}.pt)")

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
#  CONFIG ASSEMBLY
# ══════════════════════════════════════════════════════════════════════
# Full-method names all mean the same: the proposed method unchanged.
VARIANT_ALIASES = {"d": "full", "clip_flip": "full"}


def build_config(args):
    """Merge registry entry + CLI args into one cfg object.

    The returned object exposes exactly the attribute names read by the
    pipeline and phase modules (MODEL, TRIG_SZ, N_CALIB, OUT_DIR, …).
    """
    reg = load_registry()
    models, datasets, paths = reg["models"], reg["datasets"], reg["paths"]

    model, dataset = args.model, args.dataset
    if model not in models:
        raise ValueError(f"Unknown model {model!r}. "
                         f"Registry has: {sorted(models)}")
    if dataset not in datasets:
        raise ValueError(f"Unknown dataset {dataset!r}. "
                         f"Registry has: {sorted(datasets)}")

    ment = models[model]
    dent = datasets[dataset]
    splits = dent.get("splits", {})

    seed = int(args.seed)
    variant = VARIANT_ALIASES.get(args.variant, args.variant)
    out_root = resolve_path(args.out_root or paths["output_root"])
    tag = model_tag(model)

    cfg = CFG()

    # ── model ──
    cfg.MODEL = ment.get("clip_name") or ment["hf_name"]
    cfg.MODEL_NAME = model
    cfg.MODEL_ADAPTER = ment.get("adapter", "openai_clip")
    cfg.HF_NAME = ment.get("hf_name")
    cfg.DEVICE = resolve_device(args.device)
    # geometry is auto-detected at load; registry values are hints only
    cfg.PATCH_SIZE = ment.get("patch_size", 14)
    cfg.IMG_SIZE = dent.get("img_size", 224)

    # ── dataset (registry entry travels with cfg; data.py consumes it) ──
    cfg.DATASET = dataset
    cfg.DATASET_NAME = dataset
    cfg.DATASET_CFG = dent
    cfg.CLASS_TEXT_DIR = dent.get("class_text_dir")
    cfg.DATA = resolve_path(dent["root"])
    cfg.DOWNLOAD = bool(args.download)
    cfg.N_TARGET_IMAGES = _or(args.n_target_images, splits.get("n_trigger", 32))
    cfg.N_CALIB = _or(args.n_calib, splits.get("n_trigger", 32))
    cfg.N_GRAD = _or(args.n_grad, splits.get("n_gradient", 16))
    cfg.N_EVAL = _or(args.n_eval, splits.get("n_eval", 200))
    cfg.BATCH = args.batch
    cfg.MINI_BS = args.mini_bs
    cfg.WORKERS = args.workers

    # ── attack settings (from arguments) ──
    cfg.TARGET_CLS = args.target_cls
    cfg.TRIGGER_TYPE = args.trigger_type
    cfg.TRIGGER_LOCATION = args.trigger_location  # None = legacy mapping
    cfg.TRIG_SZ = args.trigger_size
    cfg.MAX_FLIPS = args.max_flips
    cfg.ASR_TARGET = args.asr_target              # 0 = no early stop

    cfg.P1_EPOCHS = args.p1_epochs
    cfg.P1_LR = args.p1_lr
    cfg.TV_WEIGHT = args.tv_weight
    cfg.TEMPERATURE = args.temperature
    cfg.EPSILON = args.epsilon
    cfg.OPACITY = args.opacity
    cfg.LAMBDA_VIS = args.lambda_vis

    cfg.TOP_K_LAYERS = args.top_k_layers
    cfg.TOP_K_COMPONENTS = args.top_k_components
    cfg.PER_LAYER_QUOTA = args.per_layer_quota

    cfg.GRADIENT_BETA = args.gradient_beta
    cfg.SENS_TOP_M = args.sens_top_m

    cfg.EVAL_BS = args.eval_bs
    cfg.CANDS_PER_ROUND = args.cands_per_round
    cfg.PLATEAU_PATIENCE = args.plateau_patience
    cfg.REFRESH_EVERY = args.refresh_every
    cfg.CHECKPOINT_EVERY = args.checkpoint_every
    cfg.CLEAN_DRIFT_PENALTY = args.clean_drift_penalty
    cfg.MAX_CLEAN_DRIFT = args.max_clean_drift

    cfg.QUANT_TOLERANCE = args.quant_tolerance

    # ── task ──
    if args.task == "i2t":
        raise NotImplementedError(
            f"task {args.task!r} is not implemented yet — 'zs' and 'lp' run "
            f"today.  See TASKS.md for the planned i2t protocol.")
    cfg.TASK = args.task

    # ── ablation variant (ABLATION.md) ──
    cfg.VARIANT = variant
    cfg.SKIP_PHASE2 = variant == "a"          # no layer/component pruning
    cfg.SKIP_CLEAN_PENALTY = variant == "b"   # no drift term in Phase-4 score
    cfg.POISON_ONLY_GRAD = variant == "c"     # rank on poison gradient only
    if cfg.SKIP_CLEAN_PENALTY:
        cfg.CLEAN_DRIFT_PENALTY = 0.0
        cfg.MAX_CLEAN_DRIFT = float("inf")

    # ── run identity & outputs ──
    cfg.SEED = seed
    cfg.REP = int(args.rep)
    run_seg = f"{variant}__t{args.trigger_size}"
    if cfg.TRIGGER_LOCATION:
        run_seg += f"__{cfg.TRIGGER_LOCATION}"
    cfg.RUN_ID = f"{tag}__{dataset}__{args.task}__c{args.target_cls}__" \
                 f"{run_seg}__s{seed}__r{cfg.REP}"
    cfg.OUT_ROOT = out_root
    cfg.LP_HEAD_ROOT = resolve_path(paths["lp_head_root"])
    cfg.LP_HEAD = args.lp_head or os.path.join(
        cfg.LP_HEAD_ROOT, f"{tag}_{dataset}.pt")
    cfg.OUT_DIR = args.out_dir or os.path.join(
        out_root, tag, dataset, args.task, f"c{args.target_cls}",
        f"s{seed}", f"r{cfg.REP}", run_seg)

    return cfg


def _or(value, fallback):
    return fallback if value is None else value


# ══════════════════════════════════════════════════════════════════════
#  ENTRY
# ══════════════════════════════════════════════════════════════════════
def run():
    args = parse_args()
    cfg = build_config(args)
    return run_attack(cfg)


if __name__ == "__main__":
    run()
```



## File: `hybrid_ranking.py`
```py
"""
hybrid_ranking.py — Phase 3 hybrid bit ranking (+ gradient-only fallback).
=========================================================================
Scoring: final_score = R_l × |g·Δw| on the Phase-0 int8 grid, all 8 bits
per element, loss-reducing directions only.  The gradient term is
poison−clean specific; swap ``rank`` to change the scoring algorithm.
"""

import torch
import torch.nn.functional as F

from adapters import encode_image, get_adapter
from quantizer import flip_bit_int8
from trigger import apply_trigger, get_apply_params
from utils import cleanup_gpu


class HybridRanker:
    """Phase 3: candidate bit-flip ranking."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def rank(self, model, trigger, t_target, layer_R,
             selected_heads, tensor_refs,
             grad_imgs, cfg, sub_quant):
        print("\n" + "=" * 60)
        print("PHASE 3: HYBRID BIT RANKING")
        print("=" * 60)
        print("  Scoring: R_l × |g·Δw|  (basic, all 8 bits per element)")
        print("  Int8 grid: Phase-0 global  (no local re-quantisation)")

        for p in model.parameters():
            p.requires_grad_(False)

        ap = get_apply_params(cfg, model)

        selected_tensors = set()
        for key, (tensor, _, _) in tensor_refs.items():
            tensor.requires_grad_(True)
            selected_tensors.add(tensor)

        assert torch.is_grad_enabled()

        mini_bs = cfg.MINI_BS
        n_imgs = grad_imgs.size(0)

        # ── Gradient on TRIGGERED (accumulate) ──
        model.zero_grad()
        total_loss_po = 0
        nb = 0
        for i in range(0, n_imgs, mini_bs):
            chunk = grad_imgs[i:i + mini_bs]
            poisoned = apply_trigger(chunk, trigger, ap)
            emb_po = F.normalize(encode_image(model, poisoned), dim=-1)
            loss_po = -F.cosine_similarity(
                emb_po,
                t_target.detach().unsqueeze(0).expand(emb_po.size(0), -1),
                dim=-1).mean()
            loss_po.backward()
            total_loss_po += loss_po.item()
            nb += 1
            del chunk, poisoned, emb_po, loss_po

        print(f"  L_target(poisoned) = {total_loss_po / max(nb, 1):.4f}")

        grad_poisoned = {}
        for key, (tensor, rslice, cslice) in tensor_refs.items():
            grad_poisoned[key] = tensor.grad[rslice, cslice].clone()

        if getattr(cfg, "POISON_ONLY_GRAD", False):
            # variant 'c': rank on the poison gradient only — skip the clean
            # pass and the poison−clean difference
            print("  Variant 'c': poison-only gradient (no poison−clean difference)")
            grad_specific = dict(grad_poisoned)
        else:
            # ── Gradient on CLEAN (accumulate) ──
            for tensor in selected_tensors:
                tensor.grad = None

            total_loss_cl = 0
            nb = 0
            for i in range(0, n_imgs, mini_bs):
                chunk = grad_imgs[i:i + mini_bs]
                emb_cl = F.normalize(encode_image(model, chunk), dim=-1)
                loss_cl = -F.cosine_similarity(
                    emb_cl,
                    t_target.detach().unsqueeze(0).expand(emb_cl.size(0), -1),
                    dim=-1).mean()
                loss_cl.backward()
                total_loss_cl += loss_cl.item()
                nb += 1
                del chunk, emb_cl, loss_cl

            print(f"  L_target(clean)    = {total_loss_cl / max(nb, 1):.4f}")

            grad_clean = {}
            for key, (tensor, rslice, cslice) in tensor_refs.items():
                grad_clean[key] = tensor.grad[rslice, cslice].clone()

            grad_specific = {}
            for key in grad_poisoned:
                grad_specific[key] = grad_poisoned[key] - grad_clean[key]

        for tensor in selected_tensors:
            tensor.requires_grad_(False)
            tensor.grad = None

        # ── Rank candidates: ALL loss-reducing bits per element ──
        ranked = []
        for key, (tensor, rslice, cslice) in tensor_refs.items():
            li, hi, tn = key
            g_slice = grad_specific[key]
            R_l = layer_R.get(li, 1.0)

            # Phase-0 global int8 grid (NOT local re-quantisation)
            sq = sub_quant[(li, hi, tn)]
            int8_vals = sq["int8"]
            scale = sq["scale"]
            flat_grad = g_slice.flatten()
            flat_int8 = int8_vals.flatten()
            flat_sens = flat_grad.abs()

            k = min(cfg.SENS_TOP_M, flat_sens.numel())
            _, topk = flat_sens.topk(k)

            for idx in topk.tolist():
                q, p = divmod(idx, g_slice.shape[1])
                g = flat_grad[idx].item()

                # Skip non-loss-reducing directions
                if g >= 0:
                    continue

                old_i8 = int(flat_int8[idx].item())
                for bit in range(8):
                    new_i8 = flip_bit_int8(old_i8, bit)
                    dw = (new_i8 - old_i8) * scale
                    dpred = g * dw
                    if dpred >= 0:
                        continue
                    hybrid_score = R_l * abs(dpred)
                    ranked.append(dict(
                        tensor=tn, layer=li, head=hi,
                        component="attn" if hi >= 0 else "ffn",
                        row=q, col=p, bit=bit,
                        old_int8=old_i8, new_int8=new_i8,
                        delta_w=dw, predicted_dL=dpred,
                        layer_weight_R=R_l,
                        gradient_score=abs(dpred),
                        final_score=hybrid_score))

        ranked.sort(key=lambda d: -d["final_score"])

        n_attn = sum(1 for r in ranked if r["component"] == "attn")
        n_ffn = sum(1 for r in ranked if r["component"] == "ffn")
        print(f"  Candidates: {len(ranked)} ({n_attn} attn, {n_ffn} ffn)")
        if ranked:
            print(f"  Top score: {ranked[0]['final_score']:.8f} "
                  f"(R={ranked[0]['layer_weight_R']:.3f}, "
                  f"g={ranked[0]['gradient_score']:.8f})")
            print(f"\n  Top-10 candidates:")
            for r in ranked[:10]:
                tag = (f"L{r['layer']} H{r['head']}" if r["component"] == "attn"
                       else f"L{r['layer']} FFN")
                print(f"    {tag} {r['tensor']}[{r['row']},{r['col']}] "
                      f"bit{r['bit']}  R={r['layer_weight_R']:.3f}  "
                      f"|Δw|={abs(r['delta_w']):.6f}  "
                      f"score={r['final_score']:.8f}")

        cleanup_gpu()
        return ranked

    def rank_fallback(self, model, trigger, t_target, selected_heads,
                      grad_imgs, cfg, sub_quant):
        print("\n" + "=" * 60)
        print("PHASE 3 (FALLBACK): GRADIENT-ONLY RANKING")
        print("=" * 60)

        for p in model.parameters():
            p.requires_grad_(False)

        ad = get_adapter(model)
        target_vec = t_target.detach()
        ap = get_apply_params(cfg, model)
        mini_bs = cfg.MINI_BS
        n_imgs = grad_imgs.size(0)

        # Adapter-provided slice refs — same mapping as WeightTracker.setup,
        # so sub_quant keys (li, hi, tn) line up for every architecture.
        tracked_params = {}
        for s in selected_heads:
            blk = list(ad.get_vision_transformer_blocks(model))[s["layer"]]
            for tn, param, quant_name, rslice, cslice in \
                    ad.get_component_refs(blk, s):
                key = (s["layer"], s["head"], tn,
                       "attn" if s["type"] == "attn" else "ffn")
                tracked_params[key] = (param, rslice, cslice)
                param.requires_grad_(True)

        # Gradient on TRIGGERED
        model.zero_grad()
        for i in range(0, n_imgs, mini_bs):
            chunk = grad_imgs[i:i + mini_bs]
            poisoned = apply_trigger(chunk, trigger, ap)
            emb_trig = F.normalize(encode_image(model, poisoned), dim=-1)
            L_target = -F.cosine_similarity(
                emb_trig, target_vec.unsqueeze(0).expand(emb_trig.size(0), -1),
                dim=-1).mean()
            L_target.backward()
            del chunk, poisoned, emb_trig, L_target

        grad_poisoned = {}
        for key, (param, rslice, cslice) in tracked_params.items():
            grad_poisoned[key] = param.grad[rslice, cslice].clone()

        if getattr(cfg, "POISON_ONLY_GRAD", False):
            # variant 'c': poison gradient only
            grad_specific = dict(grad_poisoned)
        else:
            # Gradient on CLEAN
            for key, (param, _, _) in tracked_params.items():
                param.grad = None

            for i in range(0, n_imgs, mini_bs):
                chunk = grad_imgs[i:i + mini_bs]
                emb_clean = F.normalize(encode_image(model, chunk), dim=-1)
                L_clean = -F.cosine_similarity(
                    emb_clean, target_vec.unsqueeze(0).expand(emb_clean.size(0), -1),
                    dim=-1).mean()
                L_clean.backward()
                del chunk, emb_clean, L_clean

            grad_clean = {}
            for key, (param, rslice, cslice) in tracked_params.items():
                grad_clean[key] = param.grad[rslice, cslice].clone()

            grad_specific = {}
            for key in grad_poisoned:
                grad_specific[key] = grad_poisoned[key] - grad_clean[key]

        for key, (param, _, _) in tracked_params.items():
            param.requires_grad_(False)
            param.grad = None

        ranked = []
        for key, (param, rslice, cslice) in tracked_params.items():
            li, hi, tn, comp_type = key
            grad_slice = grad_specific[key]

            # Phase-0 global int8 grid
            sq = sub_quant[(li, hi, tn)]
            int8_vals = sq["int8"]
            scale = sq["scale"]
            flat_grad = grad_slice.flatten()
            flat_int8 = int8_vals.flatten()
            flat_sens = flat_grad.abs()

            k = min(cfg.SENS_TOP_M, flat_sens.numel())
            _, topk = flat_sens.topk(k)
            for idx in topk.tolist():
                q, p = divmod(idx, grad_slice.shape[1])
                g = flat_grad[idx].item()
                if g >= 0:
                    continue
                old_i8 = int(flat_int8[idx].item())
                for bit in range(8):
                    new_i8 = flip_bit_int8(old_i8, bit)
                    dw = (new_i8 - old_i8) * scale
                    dpred = g * dw
                    if dpred >= 0:
                        continue
                    ranked.append(dict(
                        tensor=tn, layer=li, head=hi,
                        component=comp_type,
                        row=q, col=p, bit=bit,
                        old_int8=old_i8, new_int8=new_i8,
                        delta_w=dw, predicted_dL=dpred,
                        layer_weight_R=1.0,
                        gradient_score=abs(dpred),
                        final_score=abs(dpred)))

        ranked.sort(key=lambda d: -d["final_score"])
        n_attn = sum(1 for r in ranked if r["component"] == "attn")
        n_ffn = sum(1 for r in ranked if r["component"] == "ffn")
        print(f"  Candidates: {len(ranked)} ({n_attn} attn, {n_ffn} ffn)")
        if ranked:
            print(f"  Top score: {ranked[0]['final_score']:.8f}")

        cleanup_gpu()
        return ranked
```



## File: `linear_probe.py`
```py
"""
linear_probe.py — Train + save the shared linear-probe head (TASKS.md §2).
=========================================================================
Standalone training script for the ``--task lp`` evaluation head.  One
head is trained per (model, dataset) on CLEAN frozen CLIP visual features
and saved to ``{lp_head_root}/{model_tag}_{test_dataset}.pt``, so every
attack method (ours, Flip-S, TrojViT, baselines) is scored with the SAME
probe.

Protocol (agreed 2026-11-09):
  * train split : balanced ``--train-per-class`` images per class from the
                  ``--train-dataset`` registry entity (default
                  ImageNet1kTrain = ImageNet-1k train; 50/class × 1000 =
                  50 000), seeded;
  * test split  : the FULL ``--test-dataset`` entity (default ImageNet1k =
                  ImageNet-1k val, 50 000) — held-out sanity check only;
  * features    : L2-normalised ``encode_image_batched`` embeddings of the
                  clean FP32 CLIP encoder (Phase-5e convention of
                  ../main_new_pipeline_lp.py);
  * head        : ``nn.Linear(dim, n_classes)``, Adam (lr=1e-3, wd=1e-4),
                  cross-entropy, 60 epochs, batch 256 (Phase-5e values);
  * target class is irrelevant — the probe is attack-agnostic and shared
    by every target class / seed / variant.

Both registry entities must share the label space (checked at runtime;
ImageNet1kTrain/ImageNet1k both resolve names through
class_texts/imagenet1k/classes.txt).  Data must be prepared first —
ImageNet-1k via ``data/prepare_imagenet.py`` (train subset + val).

Checkpoint ({lp_head_root}/{model_tag}_{test_dataset}.pt):
  ``weight``, ``bias``  — plain tensors (drop-in for the probe loader of
                          ../main_new_pipeline_imagenet.py and Flip-S);
  ``meta``              — model/dataset/training/eval metadata.

Usage:
    python linear_probe.py                       # ViT-B/16: ImageNet1kTrain → ImageNet1k
    python linear_probe.py --model ViT-L/14 --train-per-class 50 --seed 43
    python linear_probe.py --help                # all options
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from config import CFG, load_registry, resolve_path
from data import DataModule
from evaluation import encode_image_batched
from models import ModelLoader
from utils import cleanup_gpu, model_tag, resolve_device, set_seed


# ══════════════════════════════════════════════════════════════════════
#  ARGUMENTS
# ══════════════════════════════════════════════════════════════════════
def parse_args():
    reg = load_registry()
    parser = argparse.ArgumentParser(
        description="Train + save the shared linear-probe head (--task lp)")

    sel = parser.add_argument_group("model / dataset")
    sel.add_argument("--model", default="ViT-B/16",
                     choices=sorted(reg["models"]),
                     help="model registry name (default: ViT-B/16)")
    sel.add_argument("--train-dataset", default="ImageNet1kTrain",
                     choices=sorted(reg["datasets"]),
                     help="registry entity for the probe TRAIN split "
                          "(default: ImageNet1kTrain)")
    sel.add_argument("--test-dataset", default="ImageNet1k",
                     choices=sorted(reg["datasets"]),
                     help="registry entity for the held-out TEST split; "
                          "also names the checkpoint (default: ImageNet1k)")
    sel.add_argument("--seed", type=int, default=42,
                     help="random seed (default: 42)")

    data = parser.add_argument_group("data")
    data.add_argument("--train-per-class", type=int, default=50,
                      help="balanced train images per class (default: 50 → "
                           "50 000 for ImageNet-1k)")
    data.add_argument("--batch", type=int, default=64,
                      help="feature-extraction loader batch size (default: 64)")
    data.add_argument("--mini-bs", type=int, default=32,
                      help="encode_image_batched mini-batch (default: 32)")
    data.add_argument("--workers", type=int, default=2)

    tr = parser.add_argument_group("probe training (Phase-5e values)")
    tr.add_argument("--epochs", type=int, default=60)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--wd", type=float, default=1e-4)
    tr.add_argument("--probe-batch", type=int, default=256)

    out = parser.add_argument_group("outputs")
    out.add_argument("--lp-head-root", default=None,
                     help="override registry paths.lp_head_root")
    out.add_argument("--out", default=None,
                     help="explicit checkpoint path (overrides the derived "
                          "{lp_head_root}/{model_tag}_{test_dataset}.pt)")

    misc = parser.add_argument_group("misc")
    misc.add_argument("--device", default="auto",
                      help="cuda | cpu | auto (default: auto)")
    misc.add_argument("--download", action="store_true",
                      help="allow torchvision to download datasets")

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════
def load_entity(reg, name, args, preprocess):
    """Load one registry entity through DataModule → (dataset, class_names)."""
    dent = reg["datasets"][name]
    root = resolve_path(dent["root"])
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Dataset root missing: {root} (registry entity {name!r}) — "
            f"prepare the data first (ImageNet-1k: data/prepare_imagenet.py).")
    cfg = CFG()
    cfg.DATASET = name
    cfg.DATASET_CFG = dent
    cfg.DOWNLOAD = bool(args.download)
    return DataModule(cfg).load_dataset(preprocess)


def select_probe_indices(ds, per_class, seed):
    """Seeded, balanced subset: `per_class` images of every class.

    Mirrors select_probe_indices of ../main_new_pipeline_imagenet.py.
    Returns (indices, [(class, available), …] for under-represented classes).
    """
    rng = np.random.RandomState(seed)
    by_class = defaultdict(list)
    for i, y in enumerate(DataModule.get_all_labels(ds)):
        by_class[int(y)].append(i)
    idx, short = [], []
    for y in sorted(by_class):
        cand = np.array(by_class[y])
        rng.shuffle(cand)
        if len(cand) < per_class:
            short.append((y, len(cand)))
        idx.extend(cand[:per_class].tolist())
    return idx, short


@torch.no_grad()
def collect_features(model, dataset, indices, cfg):
    """Frozen clean features, L2-normalised (Phase-5e convention)."""
    loader = DataLoader(Subset(dataset, indices), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.WORKERS)
    feats, labels = [], []
    for bi, (imgs, lbls) in enumerate(loader):
        imgs = imgs.to(cfg.DEVICE, non_blocking=True)
        f = encode_image_batched(model, imgs, cfg.MINI_BS)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.float().cpu())
        labels.append(lbls)
        if (bi + 1) % 200 == 0 or (bi + 1) == len(loader):
            print(f"    batch {bi + 1}/{len(loader)}")
    return torch.cat(feats), torch.cat(labels)


# ══════════════════════════════════════════════════════════════════════
#  PROBE TRAINING + EVALUATION (Phase-5e semantics)
# ══════════════════════════════════════════════════════════════════════
def train_linear_probe(feats, labels, n_classes, args, device):
    """nn.Linear head on frozen features — Adam(lr, wd), CE, shuffled batches."""
    probe = nn.Linear(feats.size(1), n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr,
                           weight_decay=args.wd)
    feats = feats.to(device)
    labels = labels.to(device)
    n = feats.size(0)
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(n, device=device)
        ep_loss = nb = 0
        for i in range(0, n, args.probe_batch):
            idx = perm[i:i + args.probe_batch]
            loss = F.cross_entropy(probe(feats[idx]), labels[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            acc = (probe(feats).argmax(-1) == labels).float().mean().item()
            print(f"  epoch {ep:>3d}/{args.epochs}  "
                  f"loss={ep_loss / max(nb, 1):.4f}  "
                  f"train_acc={100.0 * acc:.2f}%")
    probe.eval()
    return probe


@torch.no_grad()
def evaluate_probe(probe, feats, labels, device, topk=(1, 5)):
    logits = probe(feats.to(device))
    maxk = min(max(topk), logits.size(1))
    pred = logits.topk(maxk, dim=1).indices
    match = pred == labels.to(device).unsqueeze(1)
    return {f"top{k}": 100.0 * match[:, :min(k, maxk)].any(1).float().mean().item()
            for k in topk}


# ══════════════════════════════════════════════════════════════════════
#  ENTRY
# ══════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    reg = load_registry()
    device = resolve_device(args.device)
    set_seed(args.seed)

    print("\n" + "=" * 60)
    print("LINEAR PROBE TRAINING (shared head, --task lp)")
    print("=" * 60)
    print(f"  model : {args.model}")
    print(f"  train : {args.train_dataset}  "
          f"({args.train_per_class}/class, seed={args.seed})")
    print(f"  test  : {args.test_dataset}  (full split)")
    print(f"  device: {device}")

    # ── model: clean FP32 CLIP ──
    mcfg = CFG()
    mcfg.MODEL = reg["models"][args.model]["clip_name"]
    mcfg.MODEL_NAME = args.model
    mcfg.DEVICE = device
    mcfg.BATCH = args.batch
    mcfg.MINI_BS = args.mini_bs
    mcfg.WORKERS = args.workers
    model, preprocess = ModelLoader(mcfg).load()
    model.eval()
    mp = ModelLoader.detect_model_params(model)
    print(f"  CLIP  : width={mp['width']}  layers={mp['n_layers']}  "
          f"patch={mp['patch_size']}")

    # ── data: two registry entities, one label space ──
    train_ds, train_names = load_entity(reg, args.train_dataset, args,
                                        preprocess)
    test_ds, test_names = load_entity(reg, args.test_dataset, args,
                                      preprocess)
    if train_names != test_names:
        raise ValueError(
            f"Label-space mismatch: {args.train_dataset!r} and "
            f"{args.test_dataset!r} must share class names/order.")
    n_classes = len(train_names)
    declared = reg["datasets"][args.train_dataset].get("n_classes")
    if declared is not None and declared != n_classes:
        print(f"  WARNING: registry n_classes={declared} but the dataset "
              f"reports {n_classes} classes — using {n_classes}.")

    train_idx, short = select_probe_indices(train_ds, args.train_per_class,
                                            args.seed)
    if short:
        print(f"  WARNING: {len(short)} class(es) have fewer than "
              f"{args.train_per_class} images: {short[:5]}…")
    print(f"\n[1/3] Train subset : {len(train_idx)} images")
    print("[2/3] Extracting features")
    print("  train:")
    train_feats, train_labels = collect_features(model, train_ds, train_idx,
                                                 mcfg)
    print(f"  train features: {tuple(train_feats.shape)}")
    print("  test:")
    test_feats, test_labels = collect_features(model, test_ds,
                                               range(len(test_ds)), mcfg)
    print(f"  test  features: {tuple(test_feats.shape)}")

    # ── train + sanity check ──
    print(f"\n[3/3] Training probe  "
          f"(Adam lr={args.lr} wd={args.wd}, {args.epochs} epochs, "
          f"batch={args.probe_batch})")
    probe = train_linear_probe(train_feats, train_labels, n_classes, args,
                               device)
    res = evaluate_probe(probe, test_feats, test_labels, device)
    print(f"\n  Held-out test ({args.test_dataset}, n={test_labels.size(0)}):")
    print(f"    top-1 : {res['top1']:.2f}%")
    print(f"    top-5 : {res['top5']:.2f}%")

    # ── save ──
    head_root = resolve_path(args.lp_head_root or
                             reg["paths"]["lp_head_root"])
    out_path = args.out or os.path.join(
        head_root, f"{model_tag(args.model)}_{args.test_dataset}.pt")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({
        "weight": probe.weight.data.cpu(),
        "bias": probe.bias.data.cpu(),
        "meta": {
            "model": args.model,
            "clip_name": mcfg.MODEL,
            "train_dataset": args.train_dataset,
            "test_dataset": args.test_dataset,
            "feature_dim": int(probe.weight.shape[1]),
            "n_classes": n_classes,
            "class_names": train_names,
            "features": "encode_image_batched (clean FP32 CLIP), L2-normalised",
            "seed": args.seed,
            "train": {
                "per_class": args.train_per_class,
                "n_train": int(train_feats.size(0)),
                "epochs": args.epochs,
                "lr": args.lr,
                "wd": args.wd,
                "batch": args.probe_batch,
            },
            "eval": {
                "n_test": int(test_labels.size(0)),
                "test_top1": res["top1"],
                "test_top5": res["top5"],
            },
        },
    }, out_path)

    print("\n" + "=" * 60)
    print(f"Linear probe saved → {out_path}")
    print(f"  {args.train_dataset}: {train_feats.size(0)} train images | "
          f"{args.test_dataset}: {test_feats.size(0)} test images")
    print(f"  test top-1 {res['top1']:.2f}%  top-5 {res['top5']:.2f}%")
    print("=" * 60)
    cleanup_gpu()


if __name__ == "__main__":
    main()
```



## File: `main.py`
```py
"""
main.py — Runs one attack for a built config.
============================================
experiment.py parses the CLI, builds the cfg and passes it here; this
module calls AttackPipeline — the attack "method" entry point.

Failure contract: any exception is RECORDED and the process exits cleanly
(exit 0), so the calling shell script simply continues with the next
configuration.
  * CUDA OOM  → oom_pending.json  (deferred work-list: full argv, re-runnable
                later on a bigger GPU);
  * anything else → failed_runs.json (identity + FULL config + error type,
                message and traceback — the debugging log).
KeyboardInterrupt / SystemExit are not swallowed.
"""

import json
import os
import sys
import time
import traceback

import torch

from pipeline import AttackPipeline
from utils import cleanup_gpu

OOM_FILE = "oom_pending.json"
ERROR_FILE = "failed_runs.json"


def is_oom(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _run_identity(cfg):
    return {
        "argv": sys.argv,
        "run_id": cfg.RUN_ID,
        "model": cfg.MODEL_NAME,
        "dataset": cfg.DATASET,
        "seed": cfg.SEED,
        "variant": cfg.VARIANT,
        "out_dir": cfg.OUT_DIR,
    }


def _append(entry, path):
    pending = []
    if os.path.exists(path):
        with open(path) as f:
            pending = json.load(f)
    pending.append(entry)
    with open(path, "w") as f:
        json.dump(pending, f, indent=2, default=str)


def record_oom(cfg, path=OOM_FILE):
    """Append this run's command line + identity to the deferred work-list."""
    entry = _run_identity(cfg)
    entry["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _append(entry, path)


def record_error(cfg, exc, path=ERROR_FILE):
    """Append this run's identity + full config + failure reason to the
    error log (the debugging record for non-OOM failures)."""
    entry = _run_identity(cfg)
    entry["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    entry["error_type"] = type(exc).__name__
    entry["error"] = str(exc)
    entry["traceback"] = traceback.format_exc()
    entry["config"] = {k: v for k, v in vars(cfg).items()
                       if not k.startswith("_")}
    _append(entry, path)


def main(cfg):
    """Run one attack.  Returns the bfa_results dict, or None on skip
    (OOM or recorded error)."""
    print(f"RUN_ID : {cfg.RUN_ID}")
    print(f"OUT_DIR: {cfg.OUT_DIR}")

    try:
        return AttackPipeline(cfg).run()
    except Exception as e:
        cleanup_gpu()
        if is_oom(e):
            record_oom(cfg)
            print(f"\n  OOM — run skipped.  Config saved → {OOM_FILE}")
            print(f"  Re-run later on a bigger GPU from the saved argv.")
            return None
        record_error(cfg, e)
        print(f"\n  ERROR — run skipped and recorded → {ERROR_FILE}")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        return None
```



## File: `make_class_texts.py`
```py
"""
make_class_texts.py — Regenerate class_texts/{dataset}/classes.txt.
=================================================================
One class name per line, index-aligned with the dataset's label ids, so
the files can serve as the canonical class-name source (BadCLIP
DualEmbedding convention: a plain list of class names per dataset).

Sources:
  * Caltech101 — the on-disk 101_ObjectCategories folder names (sorted,
    image-holding categories only), cleaned through CALTECH101_RENAMES
    exactly like data.py does.
  * CIFAR10 / ImageNet1k — BadCLIP-DualEmbedding's class lists
    (../baselines/BadCLIP-DualEmbedding/data/...) used VERBATIM, so our
    class names match that baseline's evaluation.

Usage:
    python make_class_texts.py
"""

import ast
import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)          # proposed_method/
REPO_DIR = os.path.dirname(PROJECT_DIR)             # clip_bfa/
BADCLIP_DATA = os.path.join(REPO_DIR, "baselines", "BadCLIP-DualEmbedding", "data")

sys.path.insert(0, PACKAGE_DIR)
from config import load_registry, resolve_path   # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")


def load_caltech_renames():
    """CALTECH101_RENAMES from data.py, parsed without importing it (keeps
    this script dependency-free — data.py imports torchvision)."""
    path = os.path.join(PACKAGE_DIR, "data.py")
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "CALTECH101_RENAMES":
                    return ast.literal_eval(node.value)
    raise RuntimeError("CALTECH101_RENAMES not found in data.py")


def clean_name(n):
    return n.replace("_", " ").lower()


def caltech101_names(entry):
    """Sorted category folders of 101_ObjectCategories (image-holding only),
    cleaned with the same rules data.py applies."""
    renames = load_caltech_renames()
    root = resolve_path(entry["root"])
    cat_dir = os.path.join(root, "caltech-101", "101_ObjectCategories")
    names = []
    skipped = []
    for cat in sorted(os.listdir(cat_dir)):
        cpath = os.path.join(cat_dir, cat)
        if not os.path.isdir(cpath):
            continue
        has_images = any(f.lower().endswith(IMAGE_EXTS)
                         for f in os.listdir(cpath))
        if not has_images:
            skipped.append(cat)          # stray/empty folder → not a class
            continue
        names.append(renames.get(cat, clean_name(cat)))
    if skipped:
        print(f"  [caltech101] skipped empty folders: {skipped}")
    return names


def badclip_names(*relpath):
    """Class list from BadCLIP-DualEmbedding's classes.py (dict literal)."""
    path = os.path.join(BADCLIP_DATA, *relpath)
    with open(path) as f:
        return list(eval(f.read())["classes"])


def write_class_texts(dataset, names):
    out_dir = os.path.join(PACKAGE_DIR, "class_texts", dataset)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "classes.txt")
    with open(out, "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"  {dataset:<12s} {len(names):>4d} classes → {out}")


def main():
    reg = load_registry()
    datasets = reg["datasets"]

    sources = {
        "Caltech101": lambda e: caltech101_names(e),
        "CIFAR10":    lambda e: badclip_names("CIFAR10", "test", "classes.py"),
        "ImageNet1k": lambda e: badclip_names("ImageNet1K", "validation",
                                              "classes.py"),
    }

    print("Generating class_texts:")
    for name, entry in datasets.items():
        if name not in sources:
            print(f"  {name}: no source defined — skipped")
            continue
        names = sources[name](entry)
        expected = entry.get("n_classes")
        if expected is not None and len(names) != expected:
            print(f"  WARNING: {name} has {len(names)} names but registry says "
                  f"n_classes={expected}")
        write_class_texts(name.lower(), names)


if __name__ == "__main__":
    main()
```



## File: `models.py`
```py
"""
models.py — Model loading and architecture detection only.
=========================================================
ModelLoader class: loads the model through adapters.load_model and reads
its real geometry through the ModelAdapter (get_seq_layout) instead of
hardcoding OpenAI-CLIP structure.  All target-embedding logic lives in
target.py (TargetEmbedder).
"""

from adapters import get_adapter, load_model


class ModelLoader:
    """Loads the vision-language model and reports its actual architecture."""

    def __init__(self, cfg):
        self.cfg = cfg

    # ════════════════════════════════════════════════════════════════
    #  MODEL LOADING
    # ════════════════════════════════════════════════════════════════
    def load(self):
        return load_model(self.cfg)

    # ════════════════════════════════════════════════════════════════
    #  AUTO-DETECT MODEL DIMENSIONS
    # ════════════════════════════════════════════════════════════════
    @staticmethod
    def detect_model_params(model):
        """
        Read actual vision-encoder dimensions from the loaded model via its
        adapter instead of relying on hardcoded config values.

        Returns (via ModelAdapter.get_seq_layout): seq_len, width, n_patches,
        n_prefix, patch_size, patches_per_side, img_size, n_layers, n_heads.
        """
        return get_adapter(model).get_seq_layout(model)

    # ════════════════════════════════════════════════════════════════
    #  SANITY CHECKS (config vs. reality)
    # ════════════════════════════════════════════════════════════════
    def sanity_check(self, mparams):
        cfg = self.cfg
        if getattr(cfg, "MODEL_ADAPTER", "openai_clip") == "openai_clip":
            if "L/14" in cfg.MODEL and mparams["width"] != 1024:
                print(f"\n  WARNING: Config says ViT-L/14 (width=1024) but "
                      f"loaded model has width={mparams['width']}.")
                print(f"    The model is likely NOT ViT-L/14.  "
                      f"Check your CLIP installation.\n")
            if "B/16" in cfg.MODEL and mparams["width"] != 768:
                print(f"\n  WARNING: Config says ViT-B/16 (width=768) but "
                      f"loaded model has width={mparams['width']}.\n")

        # Warn about trigger-patch alignment
        trigger_per_dim = cfg.TRIG_SZ // mparams["patch_size"]
        if trigger_per_dim < 2:
            print(f"  NOTE: TRIG_SZ={cfg.TRIG_SZ} covers only "
                  f"{trigger_per_dim} patch(es) per dim with "
                  f"patch_size={mparams['patch_size']}.")
            print(f"    Consider increasing TRIG_SZ to at least "
                  f"{2 * mparams['patch_size']} for better coverage.\n")
```



## File: `pipeline.py`
```py
"""
pipeline.py — Orchestrator: wires all classes together in order.
================================================================
Phase 0 (Quantizer) → 1 (TriggerGenerator) → 2A (LayerAblator)
→ 2B (ComponentAblator) → weight tracking (WeightTracker)
→ 3 (HybridRanker) → 4 (GreedySearcher) → final evaluation
→ artefacts (ResultWriter).   (Synced with the reference
main_new_pipeline.py, which no longer runs post-attack analysis;
analysis.py is kept as an optional, unwired module.)

Every phase is exactly one method call with explicit inputs/outputs —
phase modules never import each other (GreedySearcher receives its
ranker/evaluator by injection).  This is also where per-phase ablation
flags will gate calls in the grid step.

Memory management (preserved from main_new_pipeline.py):
  1. Batch-wise processing in Phases 2, 3, 4
  2. Deferred test_loader creation (Phase 5 only)
  3. Aggressive tensor cleanup between phases
  4. Reduced dataset sizes (N_CALIB=32, N_GRAD=16, N_EVAL=200)
  5. Gradient accumulation for all gradient-based phases

run() returns the bfa_results dict (same schema as the monolith).
"""

import os
import time

import torch

from ablation import ComponentAblator, LayerAblator, enumerate_layer_components
from adapters import get_adapter
from data import DataModule
from evaluation import Evaluator
from hybrid_ranking import HybridRanker
from models import ModelLoader
from quantizer import Quantizer
from results import ResultWriter
from search import GreedySearcher
from target import TargetEmbedder
from trigger import TriggerGenerator, get_apply_params, get_trigger_token_indices
from utils import cleanup_gpu, set_seed
from visualization import Visualizer
from weights import WeightTracker


class AttackPipeline:
    """Runs the full hybrid bit-flip attack end to end."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_loader = ModelLoader(cfg)
        self.data = DataModule(cfg)
        self.target = TargetEmbedder(cfg)
        self.quantizer = Quantizer(cfg)
        self.trigger_gen = TriggerGenerator(cfg)
        self.layer_ablator = LayerAblator(cfg)
        self.component_ablator = ComponentAblator(cfg)
        self.weights = WeightTracker(cfg)
        self.ranker = HybridRanker(cfg)
        self.evaluator = Evaluator(cfg)
        self.searcher = GreedySearcher(cfg, ranker=self.ranker,
                                       evaluator=self.evaluator)
        self.viz = Visualizer(cfg)
        self.results = ResultWriter(cfg)

    # ════════════════════════════════════════════════════════════════
    #  MAIN RUN
    # ════════════════════════════════════════════════════════════════
    def run(self):
        cfg = self.cfg
        set_seed(getattr(cfg, "SEED", 42))
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        dev = cfg.DEVICE

        print("=" * 60)
        print("HYBRID BIT-FLIP ATTACK ON CLIP")
        print("Phase 0 → 1 → 2 → 3 → 4 → 5")
        print("=" * 60)
        print(f"  Device  : {dev}")
        print(f"  Target  : class {cfg.TARGET_CLS}")
        print(f"  Trigger : {cfg.TRIGGER_TYPE}  {cfg.TRIG_SZ}×{cfg.TRIG_SZ}  "
              f"loc={cfg.TRIGGER_LOCATION or 'legacy'}")

        # ── Load model ──
        print("\n[1] Loading model")
        model, preprocess = self.model_loader.load()
        ap = get_apply_params(cfg, model)

        # ── Auto-detect actual model dimensions ──
        mparams = self.model_loader.detect_model_params(model)
        cfg.PATCH_SIZE = mparams["patch_size"]
        cfg.IMG_SIZE = mparams["img_size"]

        print(f"  Config requested : {cfg.MODEL}")
        print(f"  Actual architecture:")
        print(f"    seq_len  : {mparams['seq_len']}  "
              f"(= {mparams['patches_per_side']}×{mparams['patches_per_side']} patches + CLS)")
        print(f"    width    : {mparams['width']}")
        print(f"    patch    : {mparams['patch_size']}×{mparams['patch_size']}")
        print(f"    img_size : {mparams['img_size']}")
        print(f"    layers   : {mparams['n_layers']}")
        print(f"    heads    : {mparams['n_heads']}")
        print(f"    params   : {sum(p.numel() for p in get_adapter(model).get_vision_module(model).parameters()) / 1e6:.1f}M")

        self.model_loader.sanity_check(mparams)

        # ── Load dataset ──
        print(f"[2] Loading {cfg.DATASET_NAME}")
        ds, classes = self.data.load_dataset(preprocess)
        print(f"  {len(classes)} classes, target = '{classes[cfg.TARGET_CLS]}'")

        # ── Text embeddings ──
        text_emb = self.target.build_text_embeddings(model, classes, dev)

        # ── Data splits ──
        print("\n[3] Preparing data splits")
        target_sel, calib_idx, grad_idx, eval_idx, test_idx, target_holdout = \
            self.data.prepare_data_splits(ds)

        target_images, _ = self.data.load_tensors(ds, target_sel, dev)
        calib_loader = self.data.make_loader(ds, calib_idx, cfg.BATCH, shuffle=True,
                                             workers=cfg.WORKERS)
        calib_imgs, _ = self.data.load_tensors(ds, calib_idx, dev)
        eval_imgs, eval_labels = self.data.load_tensors(ds, eval_idx, dev)

        # ── grad_imgs on CPU, transferred to GPU when needed ──
        grad_imgs_cpu, _ = self.data.load_tensors(ds, grad_idx, "cpu",
                                                  batch_size=cfg.BATCH, workers=cfg.WORKERS)

        # ── t_target ──
        t_target = self.target.from_text(cfg.TARGET_CLS, text_emb)

        print(f"  t_target = text_emb[{cfg.TARGET_CLS}]"
              f"  ('{classes[cfg.TARGET_CLS]}')")

        trigger_token_indices = get_trigger_token_indices(cfg, model=model)
        print(f"  Trigger tokens: {trigger_token_indices}")
        print(f"  Trigger type: {cfg.TRIGGER_TYPE}  "
              f"Size: {cfg.TRIG_SZ}×{cfg.TRIG_SZ}  "
              f"Placement: {ap.loc or 'legacy'}")

        # ═══ PHASE 0 — Quantisation ═══
        quant_dict = self.quantizer.quantize(model, text_emb, cfg)
        cleanup_gpu()

        # ═══ PHASE 1 — Trigger Generation ═══
        trigger = self.trigger_gen.generate(
            model, text_emb, t_target, calib_loader, target_images, cfg)

        asr_pre, cacc_pre = self.evaluator.quick_evaluate(
            model, text_emb, trigger, eval_imgs, eval_labels,
            cfg.TARGET_CLS, ap, cfg.MINI_BS)
        print(f"  Trigger standalone: ASR={asr_pre:.1f}%  CACC={cacc_pre:.1f}%")

        self.viz.save_trigger_png(trigger,
                                  os.path.join(cfg.OUT_DIR, "trigger_patch.png"),
                                  bg_image=eval_imgs[0],
                                  ap=ap)

        del target_images
        cleanup_gpu()

        # ═══ PHASE 2A — Layer-Level Causal Ablation ═══
        if getattr(cfg, "SKIP_PHASE2", False):
            # variant 'a': no layer/component pruning (Phase-2 efficiency ablation)
            print("  Variant 'a': Phase 2 skipped — all components of all layers")
            layer_scores, top_k_layers = {}, set(range(mparams["n_layers"]))
            selected_components = enumerate_layer_components(model, top_k_layers)
            layer_R = layer_scores
        else:
            layer_scores, top_k_layers = self.layer_ablator.ablate(
                model, trigger, t_target, calib_imgs, cfg)
            layer_R = layer_scores
            self.viz.plot_ablation_scores(layer_scores, cfg, top_k_layers=top_k_layers)
            cleanup_gpu()

            # ═══ PHASE 2B — Component-Level Causal Ablation ═══
            component_scores, selected_components = self.component_ablator.ablate(
                model, trigger, t_target, calib_imgs, top_k_layers, cfg)

        # ═══ Weight tracking (from selected components) ═══
        tensor_refs, original_weights = self.weights.setup(
            model, selected_components)
        print(f"\n  Weight tracking: {len(tensor_refs)} tensors, "
              f"{len(original_weights)} originals saved")
        sub_quant = self.weights.build_sub_quant(
            model, selected_components, quant_dict)

        # ═══ PHASE 3 — Hybrid Bit Ranking ═══
        grad_imgs = grad_imgs_cpu.to(dev)
        ranked_flips = self.ranker.rank(
            model, trigger, t_target, layer_R,
            selected_components, tensor_refs, grad_imgs, cfg,
            sub_quant=sub_quant)

        # ── Save clean model state before bit flips ──
        model_state_clean = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
        print(f"\n  Saved clean model state "
              f"({sum(v.numel() * v.element_size() for v in model_state_clean.values()) / 1e6:.0f} MB)")

        # ═══ PHASE 4 — Greedy Bit-Flip Search ═══
        t0 = time.time()
        flip_log, checkpoints, metrics = self.searcher.search(
            model, text_emb, trigger, selected_components,
            ranked_flips, quant_dict,
            eval_imgs, eval_labels,
            grad_imgs, original_weights, cfg,
            tensor_refs=tensor_refs,
            layer_R=layer_R, sub_quant=sub_quant)
        search_time = time.time() - t0

        del quant_dict, grad_imgs
        cleanup_gpu()

        # ═══ PHASE 5 — Post-Attack Analysis ═══
        # (Not part of the current reference pipeline — analysis.py is kept
        #  as an optional module and is intentionally not called here.)

        # ── Final full evaluation ──
        test_loader = self.data.make_loader(ds, test_idx, 64, workers=cfg.WORKERS)
        r_final = self.evaluator.full_evaluate(
            model, text_emb, trigger,
            test_loader,
            cfg.TARGET_CLS, ap, dev)

        # ── Linear-probe evaluation (--task lp, TASKS.md §2) ──
        lp_result = None
        if cfg.TASK == "lp":
            print("\n[LP] Linear-probe evaluation")
            print(f"  probe head: {cfg.LP_HEAD}")
            lp_result = self.evaluator.lp_evaluate(
                model, test_loader, dev, cfg.LP_HEAD,
                trigger=trigger, target_cls=cfg.TARGET_CLS, ap=ap,
                mini_bs=cfg.MINI_BS)
            print(f"  LP-CACC={lp_result['lp_cacc']:.2f}%  "
                  f"LP-ASR={lp_result['lp_asr']:.2f}%")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        n_flips = len(flip_log)
        print(f"  Model   : {cfg.MODEL}  (actual: {mparams['width']}d, "
              f"{mparams['n_layers']}L, {mparams['n_heads']}H, "
              f"patch={mparams['patch_size']})")
        print(f"  Method  : Hybrid (Layer_R → Scoped Gradient → R(l) × Gradient)")
        print(f"  Flips   : {n_flips}")
        print(f"  ASR     : {r_final['asr']:.1f}%")
        print(f"  CACC    : {r_final['cacc']:.1f}%")
        print(f"  Search  : {search_time:.1f}s")

        sample_for_vis = eval_imgs[:10]
        sample_labels_vis = eval_labels[:10]
        self.viz.create_composite_png(
            trigger, cfg, r_final["asr"], r_final["cacc"],
            sample_for_vis, sample_labels_vis, classes,
            os.path.join(cfg.OUT_DIR, "composite.png"), model=model)

        save = {
            "config": {k: v for k, v in vars(cfg).items()
                       if not k.startswith("_")},
            "actual_model_params": mparams,
            "target_class": classes[cfg.TARGET_CLS],
            "trigger_token_indices": trigger_token_indices,
            "layer_R": {str(k): v for k, v in layer_R.items()} if layer_R else {},
            "top_k_layers": sorted(top_k_layers),
            "selected_heads": [
                {k: v for k, v in s.items()
                 if not isinstance(v, torch.Tensor)}
                for s in selected_components],
            "flip_log": flip_log,
            "checkpoints": checkpoints,
            "metrics": metrics,
            "rep": getattr(cfg, "REP", 1),
            "trigger_location": getattr(cfg, "TRIGGER_LOCATION", None),

            "results": {
                "post_flip_test": {k: v for k, v in r_final.items()
                                   if not isinstance(v, torch.Tensor)}},
            "search_time_sec": search_time}
        if lp_result is not None:
            save["linear_probe"] = lp_result

        self.results.save(save, trigger, original_weights)

        return save
```



## File: `quantizer.py`
```py
"""
quantizer.py — Phase-0 INT8 quantisation and int8 bit utilities.
===============================================================
Phase-0 semantics: per-tensor int8 grid, dequantized weights for the
forward pass, int8 codes + scale kept for bit access.  Slices are NEVER
re-quantised locally — always use the Phase-0 grid.
"""

import torch


# ══════════════════════════════════════════════════════════════════════
#  INT8 UTILITIES
# ══════════════════════════════════════════════════════════════════════
def quantize_to_int8(tensor):
    amax = tensor.abs().max().item()
    scale = max(amax / 127.0, 1e-12)
    int8 = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
    return int8, scale


def dequantize_int8(int8_vals, scale):
    return int8_vals.float() * scale


def flip_bit_int8(val, bit):
    u = int(val) & 0xFF
    u ^= (1 << bit)
    return (u - 256) if u >= 128 else u


# ══════════════════════════════════════════════════════════════════════
#  PHASE 0 — INT8 QUANTISATION
# ══════════════════════════════════════════════════════════════════════
class Quantizer:
    """Phase 0: int8-quantises the visual encoder in place."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    @staticmethod
    def quantize_visual_encoder(model):
        from adapters import get_adapter
        ad = get_adapter(model)
        blocks = list(ad.get_vision_transformer_blocks(model))
        qd = {}
        for li, blk in enumerate(blocks):
            # Adapter-provided canonical tensor list (attn weights + biases,
            # layer norms, MLP weights + biases).  Phase-0 semantics: each
            # listed tensor gets ONE per-tensor int8 grid.
            for name, param in ad.get_block_quant_params(blk):
                if param is None:
                    continue
                int8, scale = quantize_to_int8(param.data)
                qd[f"L{li}_{name}"] = dict(int8=int8.clone(), scale=scale,
                                           block_idx=li, attr=name,
                                           shape=list(param.shape))
                param.data.copy_(dequantize_int8(int8, scale))
        return qd

    def quantize(self, model, text_emb=None, cfg=None):
        print("\n" + "=" * 60)
        print("PHASE 0: QUANTISATION (INT8)")
        print("=" * 60)
        quant_dict = self.quantize_visual_encoder(model)
        print("  Quantisation OK")
        return quant_dict
```



## File: `results.py`
```py
"""
results.py — Per-run artefact writing.
======================================
Writes bfa_results.json (same schema as main_new_pipeline.py), the
trigger tensor and the original weights snapshot.
"""

import json
import os

import torch


class ResultWriter:
    """Writes the run's artefacts into cfg.OUT_DIR."""

    def __init__(self, cfg):
        self.cfg = cfg

    def save(self, save_dict, trigger, original_weights):
        cfg = self.cfg
        with open(os.path.join(cfg.OUT_DIR, "bfa_results.json"), "w") as f:
            json.dump(save_dict, f, indent=2, default=str)
        torch.save(trigger.cpu(), os.path.join(cfg.OUT_DIR, "trigger.pt"))
        torch.save(original_weights,
                   os.path.join(cfg.OUT_DIR, "original_weights.pt"))
        print(f"\n  Results saved to {cfg.OUT_DIR}/")
```



## File: `search.py`
```py
"""
search.py — Phase 4 greedy bit-flip search.
==========================================
GreedySearcher.ranker (HybridRanker) and .evaluator (Evaluator) are
injected by the pipeline — this module never imports other phase
classes, so the search algorithm can be swapped in isolation.

Weight positions come from the WeightTracker slice refs (param,
row_slice, col_slice) built via the model's ModelAdapter, so absolute
index mapping works for fused-QKV (OpenAI CLIP) and split-QKV (SigLIP)
architectures alike.

Phase-4 metrics (RESULTS.md §0): ASR/CACC are evaluated after EVERY
accepted flip and recorded at exactly 10 and 20 flips
(``metrics['asr_at_10']`` / ``['asr_at_20']``), and ``flips_to_95`` is
the first accepted-flip count where ASR ≥ 95%.  ``cfg.ASR_TARGET``
(default 95) is the early stop; 0 disables it.
"""

from collections import Counter

import torch
import torch.nn.functional as F

from evaluation import encode_image_batched
from trigger import apply_trigger, get_apply_params
from weights import build_sub_quant, setup_weight_tracking, _get_sub_quant_key


# ══════════════════════════════════════════════════════════════════════
#  WEIGHT-INDEX HELPERS (slice-ref based — adapter-agnostic)
# ══════════════════════════════════════════════════════════════════════
def _abs_pos(refs, key, row, col):
    """(local row, col) inside component slice → absolute weight indices."""
    _, rslice, cslice = refs[key]
    return (rslice.start or 0) + row, (cslice.start or 0) + col


def _local_idx(refs, key, abs_r, abs_c):
    """Absolute weight indices → (row, col) inside the component slice."""
    _, rslice, cslice = refs[key]
    return abs_r - (rslice.start or 0), abs_c - (cslice.start or 0)


# ══════════════════════════════════════════════════════════════════════
#  PHASE 4 — GREEDY BIT-FLIP SEARCH
# ══════════════════════════════════════════════════════════════════════
class GreedySearcher:
    """Phase 4: greedy bit-flip search over the ranked candidates."""

    def __init__(self, cfg=None, ranker=None, evaluator=None):
        self.cfg = cfg
        self.ranker = ranker        # HybridRanker (injected)
        self.evaluator = evaluator  # Evaluator (injected)

    @torch.no_grad()
    def search(self, model, text_emb, trigger, selected_heads,
               ranked_flips, quant_dict,
               eval_imgs, eval_labels,
               grad_imgs, original_weights, cfg,
               tensor_refs=None, layer_R=None, sub_quant=None):
        print("\n" + "=" * 60)
        print("PHASE 4: GREEDY BIT-FLIP SEARCH")
        print("=" * 60)

        tc = cfg.TARGET_CLS
        ap = get_apply_params(cfg, model)
        eval_bs = cfg.EVAL_BS

        drift_n = min(16, eval_imgs.shape[0])
        drift_imgs = eval_imgs[:drift_n]

        # Weight slice refs (param, row_slice, col_slice) — build if the
        # pipeline didn't provide them.
        if tensor_refs is None:
            tensor_refs, _ = setup_weight_tracking(model, selected_heads)

        # Shared with Phase 3 when provided, so candidate int8 codes always live
        # on the model's actual quantisation grid.
        if sub_quant is None:
            sub_quant = build_sub_quant(model, selected_heads, quant_dict)

        asr, cacc = self.evaluator.quick_evaluate(
            model, text_emb, trigger, eval_imgs, eval_labels, tc, ap, eval_bs)
        cacc_base = cacc
        asr_base = asr
        print(f"  Baseline  ASR={asr:.1f}%  CACC={cacc:.1f}%")

        clean_emb_ref = F.normalize(
            encode_image_batched(model, drift_imgs, eval_bs), dim=-1).clone()
        target_vec = text_emb[tc].detach()

        poisoned = apply_trigger(drift_imgs, trigger, ap)
        trig_emb_ref = F.normalize(
            encode_image_batched(model, poisoned, eval_bs), dim=-1).clone()
        trig_cos_ref = (trig_emb_ref @ target_vec).cpu()

        print(f"  Drift penalty λ={cfg.CLEAN_DRIFT_PENALTY}  "
              f"max_drift={cfg.MAX_CLEAN_DRIFT}")
        print(f"  drift_n={drift_n}  eval_bs={eval_bs}  "
              f"cands_per_round={cfg.CANDS_PER_ROUND}")

        flip_log = []
        checkpoints = [(0, asr, cacc)]
        flip_metrics = {}          # accepted-flip count → (asr, cacc)
        flips_to_95 = None
        ranked = list(ranked_flips)
        cand_ptr = 0
        no_improve = 0

        def rebuild_candidates():
            nonlocal ranked, cand_ptr
            if tensor_refs is not None and layer_R is not None:
                print("  Rebuilding hybrid rankings...")
                with torch.enable_grad():
                    ranked = self.ranker.rank(
                        model, trigger, target_vec, layer_R,
                        selected_heads, tensor_refs, grad_imgs, cfg,
                        sub_quant=sub_quant)
            else:
                with torch.enable_grad():
                    ranked = self.ranker.rank_fallback(
                        model, trigger, target_vec, selected_heads,
                        grad_imgs, cfg, sub_quant=sub_quant)
            cand_ptr = 0

        for rd in range(cfg.MAX_FLIPS):
            if cand_ptr >= len(ranked):
                print(f"  Rd {rd}: pool exhausted — refreshing")
                rebuild_candidates()
                if len(ranked) == 0:
                    print("  No candidates — stopping.")
                    break

            batch = ranked[cand_ptr:cand_ptr + cfg.CANDS_PER_ROUND]
            cand_ptr += cfg.CANDS_PER_ROUND

            best_score = -1e9
            best_info = None

            for c in batch:
                key = (c["layer"], c["head"], c["tensor"])
                tensor, rslice, cslice = tensor_refs[key]
                ar, ac = _abs_pos(tensor_refs, key, c["row"], c["col"])
                old_f = tensor.data[ar, ac].item()
                sq_key = _get_sub_quant_key(c)
                sq = sub_quant[sq_key]
                new_f = c["new_int8"] * sq["scale"]
                tensor.data[ar, ac] = new_f

                poisoned_new = apply_trigger(drift_imgs, trigger, ap)
                trig_emb_new = F.normalize(
                    encode_image_batched(model, poisoned_new, eval_bs), dim=-1)
                trig_cos_new = (trig_emb_new @ target_vec).cpu()

                clean_emb_new = F.normalize(
                    encode_image_batched(model, drift_imgs, eval_bs), dim=-1)
                drift = (1.0 - F.cosine_similarity(
                    clean_emb_ref, clean_emb_new, dim=-1)).mean().item()

                tensor.data[ar, ac] = old_f

                if drift > cfg.MAX_CLEAN_DRIFT:
                    continue
                delta_cos = (trig_cos_new - trig_cos_ref).mean().item()
                n_improved = (trig_cos_new > trig_cos_ref).sum().item()
                score = delta_cos - cfg.CLEAN_DRIFT_PENALTY * drift
                if score > best_score:
                    best_score = score
                    best_info = dict(
                        tensor=c["tensor"], layer=c["layer"], head=c["head"],
                        component=c["component"],
                        abs_r=ar, abs_c=ac, bit=c["bit"],
                        old_int8=c["old_int8"], new_int8=c["new_int8"],
                        delta_cos=delta_cos, drift=drift,
                        n_improved=n_improved,
                        layer_weight_R=c.get("layer_weight_R", 1.0))

            if best_info is None or best_score <= 0:
                no_improve += 1
                print(f"  Round {rd}: no improving flip "
                      f"(streak {no_improve}/{cfg.PLATEAU_PATIENCE})")
                if no_improve >= cfg.PLATEAU_PATIENCE:
                    print("  Plateau reached.")
                    break
                continue

            no_improve = 0
            bi = best_info
            bkey = (bi["layer"], bi["head"], bi["tensor"])
            tensor = tensor_refs[bkey][0]
            tensor.data[bi["abs_r"], bi["abs_c"]] = bi["new_int8"] * \
                                                    sub_quant[_get_sub_quant_key(bi)]["scale"]
            sq = sub_quant[_get_sub_quant_key(bi)]
            lr, lc = _local_idx(tensor_refs, bkey, bi["abs_r"], bi["abs_c"])
            sq["int8"][lr, lc] = bi["new_int8"]

            poisoned_upd = apply_trigger(drift_imgs, trigger, ap)
            trig_emb_ref = F.normalize(
                encode_image_batched(model, poisoned_upd, eval_bs), dim=-1).clone()
            trig_cos_ref = (trig_emb_ref @ target_vec).cpu()
            clean_emb_ref = F.normalize(
                encode_image_batched(model, drift_imgs, eval_bs), dim=-1).clone()

            flip_log.append(dict(
                round=rd, tensor=bi["tensor"],
                layer=bi["layer"], head=bi["head"],
                component=bi["component"],
                position=(bi["abs_r"], bi["abs_c"]),
                bit=bi["bit"], old_int8=bi["old_int8"],
                new_int8=bi["new_int8"],
                delta_cos=bi["delta_cos"], drift=bi["drift"],
                n_improved=bi["n_improved"],
                score=best_score,
                layer_weight_R=bi.get("layer_weight_R", 1.0)))

            tag = (f"L{bi['layer']} H{bi['head']}" if bi["component"] == "attn"
                   else f"L{bi['layer']} FFN")
            print(f"  Rd {rd:>2d}: {tag} {bi['tensor']}[{bi['abs_r']},{bi['abs_c']}] "
                  f"bit {bi['bit']}  "
                  f"Δcos={bi['delta_cos']:+.4f}  "
                  f"drift={bi['drift']:.4f}  "
                  f"score={best_score:.4f}")

            # ── Per-flip metrics: exact ASR@10/@20 + Flips→95% ──
            a_full, c_full = self.evaluator.quick_evaluate(
                model, text_emb, trigger,
                eval_imgs, eval_labels, tc, ap, eval_bs)
            n_flips = len(flip_log)
            flip_metrics[n_flips] = (a_full, c_full)
            if n_flips % cfg.CHECKPOINT_EVERY == 0:
                checkpoints.append((n_flips, a_full, c_full))
                print(f"  [ckpt {n_flips}] ASR={a_full:.1f}%  "
                      f"CACC={c_full:.1f}%")
            if flips_to_95 is None and a_full >= 95.0:
                flips_to_95 = n_flips
                print(f"  Flips→95%: reached after {n_flips} accepted flip(s)")
                if n_flips % cfg.CHECKPOINT_EVERY != 0:
                    checkpoints.append((n_flips, a_full, c_full))
            if cfg.ASR_TARGET > 0 and a_full >= cfg.ASR_TARGET:
                print(f"  ASR target ({cfg.ASR_TARGET}%) reached!")
                break

            if n_flips % cfg.REFRESH_EVERY == 0:
                rebuild_candidates()

        a_f, c_f = self.evaluator.quick_evaluate(model, text_emb, trigger,
                                                 eval_imgs, eval_labels, tc,
                                                 ap, eval_bs)
        checkpoints.append((len(flip_log), a_f, c_f))

        def _at(k, idx):
            return flip_metrics[k][idx] if k in flip_metrics else None

        metrics = dict(
            asr_at_10=_at(10, 0), cacc_at_10=_at(10, 1),
            asr_at_20=_at(20, 0), cacc_at_20=_at(20, 1),
            flips_to_95=flips_to_95,
            flips_total=len(flip_log),
            asr_final=a_f, cacc_final=c_f,
            asr_base=asr_base, cacc_base=cacc_base,
            asr_target=cfg.ASR_TARGET)

        print(f"\n  Total flips : {len(flip_log)}")
        print(f"  Final ASR   : {a_f:.1f}%  (Δ = {a_f - asr_base:+.1f}%)")
        print(f"  Final CACC  : {c_f:.1f}%  (Δ = {c_f - cacc_base:+.1f}%)")
        print(f"  ASR@10={metrics['asr_at_10']}  ASR@20={metrics['asr_at_20']}  "
              f"Flips→95%={flips_to_95}")

        if flip_log:
            bit_counts = Counter(f["bit"] for f in flip_log)
            print(f"  Bit distribution: {dict(sorted(bit_counts.items()))}")

        return flip_log, checkpoints, metrics
```



## File: `target.py`
```py
"""
target.py — Text embeddings and target embedding.
=================================================
Dataset-agnostic: class names arrive prompt-ready from data.py's
registry (per-dataset renames live there), this module only formats
prompts and encodes them.

t_target = text-only multi-template ensemble mean;
build_target_embedding is intentionally unused (kept for reference).
"""

import torch
import torch.nn.functional as F

from evaluation import encode_image_batched

# ── CLIP prompt templates (from CLIP paper appendix) ──
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
    "a cropped photo of a {}.",
    "a low resolution photo of a {}.",
    "a photo of a large {}.",
    "a photo of a small {}.",
    "a photo of many {}.",
    "a picture of a {}.",
    "a rendering of a {}.",
    "a photo of one {}.",
]


class TargetEmbedder:
    """Builds class text embeddings and the target embedding t_target."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def build_text_embeddings(self, model, classes, device):
        """
        Build class text embeddings using multi-template prompt ensemble.
        Class names are expected prompt-ready (DataModule output); the
        idempotent underscore/lower normalisation is kept as a safety net.
        """
        n_classes = len(classes)
        n_templates = len(PROMPT_TEMPLATES)

        all_prompts = []
        for c in classes:
            name = c.replace("_", " ").lower()
            for template in PROMPT_TEMPLATES:
                all_prompts.append(template.format(name))

        print(f"  Encoding {len(all_prompts)} prompts "
              f"({n_classes} classes × {n_templates} templates)...")

        all_embeddings = []
        with torch.no_grad():
            chunk_size = 256
            for i in range(0, len(all_prompts), chunk_size):
                chunk = all_prompts[i:i + chunk_size]
                from adapters import get_adapter
                emb = get_adapter(model).encode_text(model, chunk, device)
                all_embeddings.append(emb)
            all_emb = torch.cat(all_embeddings, dim=0)

        all_emb = all_emb.view(n_classes, n_templates, -1)
        text_emb = F.normalize(all_emb.mean(dim=1), dim=-1)

        print(f"  Text embeddings shape: {text_emb.shape}")
        print(f"  Sample prompts for class 0 ('{classes[0]}'):")
        name_0 = classes[0].replace("_", " ").lower()
        for t in PROMPT_TEMPLATES[:3]:
            print(f"    \"{t.format(name_0)}\"")

        return text_emb

    def from_text(self, target_cls, text_emb):
        """t_target = multi-template ensemble mean of the target class."""
        with torch.no_grad():
            return text_emb[target_cls].detach()

    @staticmethod
    def build_target_embedding(model, text_emb, target_cls, target_images, device):
        """
        Build a robust target embedding from text prompts AND real target-class images.
        (Currently unused — t_target uses text-only multi-template ensemble.)
        """
        t_text = text_emb[target_cls:target_cls + 1]

        with torch.no_grad():
            t_img = F.normalize(
                encode_image_batched(model, target_images, 4), dim=-1).mean(dim=0, keepdim=True)

        t_target = F.normalize(0.4 * t_text + 0.6 * t_img, dim=-1).squeeze(0)

        cos_text_img = F.cosine_similarity(t_text, t_img, dim=-1).item()
        print(f"  Target embedding: 40% text + 60% image prototype")
        print(f"  cos(text, image_proto) = {cos_text_img:.4f}")
        if cos_text_img < 0.5:
            print(f"  WARNING: text and image embeddings are far apart. "
                  f"The trigger has a hard target to reach.")

        return t_target
```



## File: `trigger.py`
```py
"""
trigger.py — Trigger application utilities and Phase-1 trigger generation.
=========================================================================
The low-level application helpers (apply_trigger & co.) are shared
plumbing imported by other modules.  The Phase-1 algorithm lives in
TriggerGenerator.generate — swap that method to change how the trigger
is learned without touching any other file.

Trigger placement is controlled by ApplyParams.loc
('center' | 'top_left' | 'top_right' | 'bottom_left' | 'bottom_right';
None = legacy mapping of the trigger mode: center_add → center,
otherwise bottom-right).  Train and evaluation use the same ApplyParams,
so a location sweep trains AND evaluates at that location.
"""

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import CLIP_MEAN, CLIP_STD, _CS

# ── Apply parameters: everything apply_trigger needs, in one record ──
ApplyParams = namedtuple("ApplyParams", ["sz", "mode", "opacity", "loc", "norm"])

LOCATIONS = ("center", "top_left", "top_right", "bottom_left", "bottom_right")


# ══════════════════════════════════════════════════════════════════════
#  PLACEMENT
# ══════════════════════════════════════════════════════════════════════
def placement_origin(H, W, sz, loc=None, mode="replace"):
    """Top-left (y0, x0) of the trigger patch for the given location.

    loc=None keeps the legacy mapping: 'center' in mode → center,
    otherwise bottom-right corner."""
    if loc is None:
        loc = "center" if "center" in mode else "bottom_right"
    if loc == "center":
        return (H - sz) // 2, (W - sz) // 2
    if loc == "top_left":
        return 0, 0
    if loc == "top_right":
        return 0, W - sz
    if loc == "bottom_left":
        return H - sz, 0
    return H - sz, W - sz          # bottom_right (legacy corner)


# ══════════════════════════════════════════════════════════════════════
#  TRIGGER APPLICATION
# ══════════════════════════════════════════════════════════════════════
def apply_trigger(imgs, trig, ap, clamp_output=True):
    """Apply trigger with mode: replace | blend | add_center | add_corner.

    ap: ApplyParams (sz, mode, opacity, loc, norm) — see get_apply_params.
    clamp_output: clamp the poisoned image back to the valid normalised
    pixel range for additive/blend triggers (synced with the reference).
    """
    out = imgs.clone()
    y0, x0 = placement_origin(imgs.shape[2], imgs.shape[3],
                              ap.sz, ap.loc, ap.mode)

    region = (slice(None), slice(None),
              slice(y0, y0 + ap.sz), slice(x0, x0 + ap.sz))

    if "blend" in ap.mode:
        out[region] = ap.opacity * trig + (1 - ap.opacity) * out[region]
    elif "add" in ap.mode:
        out[region] = out[region] + trig
    else:
        out[region] = trig

    # Clamp output to valid image range if additive/blend
    if clamp_output and ("add" in ap.mode or "blend" in ap.mode):
        mean, std = ap.norm
        sd = torch.tensor(std, dtype=imgs.dtype,
                          device=imgs.device).view(1, 3, 1, 1)
        mu = torch.tensor(mean, dtype=imgs.dtype,
                          device=imgs.device).view(1, 3, 1, 1)
        lo = (0.0 - mu) / sd
        hi = (1.0 - mu) / sd
        out = torch.maximum(torch.minimum(out, hi), lo)
    return out


def get_apply_params(cfg, model=None):
    """Build ApplyParams from cfg (+ model for its normalisation stats)."""
    mode, opacity = get_trigger_mode(cfg.TRIGGER_TYPE, cfg.OPACITY)
    loc = getattr(cfg, "TRIGGER_LOCATION", None)
    if model is not None:
        from adapters import get_adapter
        norm = get_adapter(model).get_norm_stats(model)
    else:
        norm = (CLIP_MEAN, CLIP_STD)
    return ApplyParams(cfg.TRIG_SZ, mode, opacity, loc, norm)


def pixel_space_clamp(trig, epsilon_pixel):
    """Clamp perturbation in actual pixel space [0,1]."""
    with torch.no_grad():
        sd = _CS.to(trig.device)
        eps_normalized = epsilon_pixel / sd
        trig.data.clamp_(-eps_normalized, eps_normalized)


def total_variation(trig):
    dh = (trig[:, :, :, :-1] - trig[:, :, :, 1:]).pow(2).mean()
    dv = (trig[:, :, :-1, :] - trig[:, :, 1:, :]).pow(2).mean()
    return dh + dv


def get_trigger_mode(trigger_type, opacity=1.0):
    """Convert trigger_type to (mode_string, opacity) for apply_trigger."""
    if trigger_type == "opacity":
        return "blend", opacity
    elif trigger_type == "center_add":
        return "add_center", 1.0
    elif trigger_type == "corner_add":
        return "add_corner", 1.0
    else:
        return "replace", 1.0


# ══════════════════════════════════════════════════════════════════════
#  TRIGGER TOKEN INDICES (model-aware)
# ══════════════════════════════════════════════════════════════════════
def get_trigger_token_indices(cfg, model=None):
    """Compute which ViT token positions the trigger overlaps."""
    if model is not None:
        from adapters import get_adapter
        lay = get_adapter(model).get_seq_layout(model)
        n_prefix = lay["n_prefix"]
        patches_per_side = lay["patches_per_side"]
        patch_size = lay["patch_size"]
        img_size = lay["img_size"]
    else:
        n_prefix = 1        # OpenAI-CLIP convention (CLS token)
        patches_per_side = cfg.IMG_SIZE // cfg.PATCH_SIZE
        patch_size = cfg.PATCH_SIZE
        img_size = cfg.IMG_SIZE

    sz = cfg.TRIG_SZ
    mode, _ = get_trigger_mode(cfg.TRIGGER_TYPE, cfg.OPACITY)
    y0, x0 = placement_origin(img_size, img_size, sz,
                              getattr(cfg, "TRIGGER_LOCATION", None), mode)
    y1 = y0 + sz
    x1 = x0 + sz

    indices = []
    for r in range(patches_per_side):
        for c in range(patches_per_side):
            py0, py1 = r * patch_size, (r + 1) * patch_size
            px0, px1 = c * patch_size, (c + 1) * patch_size
            if y1 > py0 and y0 < py1 and x1 > px0 and x0 < px1:
                indices.append(r * patches_per_side + c + n_prefix)
    return indices


# ══════════════════════════════════════════════════════════════════════
#  TARGET-COSINE SCORER (used by the ablation phases)
# ══════════════════════════════════════════════════════════════════════
def _make_target_cos(model, t_target, mini_bs):
    """Build a reusable target-cosine scorer."""
    from adapters import encode_image

    def target_cos(imgs):
        embs = []
        for i in range(0, imgs.size(0), mini_bs):
            embs.append(F.normalize(
                encode_image(model, imgs[i:i + mini_bs]), dim=-1))
        emb = torch.cat(embs, dim=0)
        return F.cosine_similarity(
            emb,
            t_target.detach().unsqueeze(0).expand(emb.size(0), -1),
            dim=-1).mean().item()
    return target_cos


# ══════════════════════════════════════════════════════════════════════
#  PHASE 1 — TRIGGER GENERATION
# ══════════════════════════════════════════════════════════════════════
class TriggerGenerator:
    """Phase 1: trigger generation (clip_ce loss, additive, center)."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def generate(self, model, text_emb, t_target, calib_loader,
                 target_images, cfg):
        """Trigger generation: clip_ce loss, additive, configured placement,
        clamped (init × 0.1, static trigger clamp [-3, 3]).
        Returns the best trigger tensor."""
        from adapters import encode_image

        ap = get_apply_params(cfg, model)
        print("\n" + "=" * 60)
        print("PHASE 1: TRIGGER GENERATION (clamped)")
        print("=" * 60)
        print(f"  Size: {cfg.TRIG_SZ}×{cfg.TRIG_SZ}  "
              f"Placement: {ap.loc or 'legacy'}  Mode: {ap.mode}")
        print(f"  Epochs: {cfg.P1_EPOCHS}  LR: {cfg.P1_LR}  "
              f"T: {cfg.TEMPERATURE}")
        print(f"  Init: randn × 0.1  TV: {cfg.TV_WEIGHT}  "
              f"Static Trig Clamp: [-3, 3]")

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # Init with 0.1 (clamped trigger regime)
        trig = nn.Parameter(
            torch.randn(1, 3, cfg.TRIG_SZ, cfg.TRIG_SZ,
                        device=cfg.DEVICE) * 0.1)

        opt = torch.optim.Adam([trig], lr=cfg.P1_LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.P1_EPOCHS, eta_min=cfg.P1_LR * 0.01)

        best_loss, best_trig = float("inf"), trig.data.clone()

        for ep in range(1, cfg.P1_EPOCHS + 1):
            ep_loss = nb = 0
            for imgs, _ in calib_loader:
                imgs = imgs.to(cfg.DEVICE)
                poisoned = apply_trigger(imgs, trig, ap)
                emb = F.normalize(encode_image(model, poisoned), dim=-1)

                logits = (emb @ text_emb.T) / cfg.TEMPERATURE
                labels = torch.full((emb.size(0),), cfg.TARGET_CLS,
                                    dtype=torch.long, device=cfg.DEVICE)
                loss = F.cross_entropy(logits, labels)

                if cfg.TV_WEIGHT > 0:
                    loss = loss + cfg.TV_WEIGHT * total_variation(trig)

                opt.zero_grad()
                loss.backward()
                opt.step()

                # Static trigger clamp after step
                with torch.no_grad():
                    trig.data.clamp_(-3.0, 3.0)

                ep_loss += loss.item()
                nb += 1

            sched.step()
            avg_l = ep_loss / max(nb, 1)
            if avg_l < best_loss:
                best_loss, best_trig = avg_l, trig.data.clone()
            if ep % 50 == 0 or ep == 1 or ep == cfg.P1_EPOCHS:
                print(f"  ep {ep:>4d}  loss={avg_l:.6f}  "
                      f"||trig||={trig.data.norm():.3f}  "
                      f"RMS={trig.data.std():.3f}")

        print(f"  Best loss: {best_loss:.6f}")
        return best_trig.detach()
```



## File: `utils.py`
```py
"""
utils.py — Seeding, GPU cleanup, CLIP normalisation constants.
==============================================================
Shared low-level plumbing: every module may import from here.
"""

import gc
import random

import numpy as np
import torch

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
_CM = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
_CS = torch.tensor(CLIP_STD).view(1, 3, 1, 1)


# ══════════════════════════════════════════════════════════════════════
#  NAMING / DEVICE HELPERS
# ══════════════════════════════════════════════════════════════════════
def model_tag(model):
    """Filesystem-safe model tag: 'ViT-B/16' → 'ViT-B16'."""
    return model.replace("/", "")


def resolve_device(device):
    """'auto' → 'cuda' when available, else 'cpu'; other values pass through."""
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# ══════════════════════════════════════════════════════════════════════
#  SEEDING
# ══════════════════════════════════════════════════════════════════════
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════
#  MEMORY HELPER
# ══════════════════════════════════════════════════════════════════════
def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```



## File: `visualization.py`
```py
"""
visualization.py — Trigger/composite PNGs and score plots.
==========================================================
All figures land in cfg.OUT_DIR.
"""

import os

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trigger import apply_trigger, get_apply_params, placement_origin
from utils import CLIP_MEAN, CLIP_STD


def _denormalise(t, norm=None):
    mean, std = norm if norm is not None else (CLIP_MEAN, CLIP_STD)
    mu = torch.tensor(mean).view(3, 1, 1)
    sd = torch.tensor(std).view(3, 1, 1)
    px = (t.cpu().float() * sd + mu).clamp(0, 1)
    return (px.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


class Visualizer:
    """Trigger PNGs, composite figure, ablation/retrieval plots."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    # ════════════════════════════════════════════════════════════════
    #  SAVE TRIGGER AS PNG
    # ════════════════════════════════════════════════════════════════
    @staticmethod
    def save_trigger_png(trigger, path, bg_image=None, ap=None):
        """ap: ApplyParams (size/mode/opacity/location/norm) or None for
        defaults."""
        if ap is None:
            from trigger import ApplyParams
            ap = ApplyParams(trigger.shape[-1], "replace", 1.0, None,
                             (CLIP_MEAN, CLIP_STD))
        with torch.no_grad():
            t = trigger.cpu().float()
            mu = torch.tensor(ap.norm[0]).view(3, 1, 1)
            sd = torch.tensor(ap.norm[1]).view(3, 1, 1)
            t_pixel = (t * sd + mu).clamp(0, 1)
            t_np = t_pixel[0].permute(1, 2, 0).numpy()
            sz = t_np.shape[0]

            if bg_image is not None:
                bg = bg_image.cpu().float()
                bg = (bg * sd + mu).clamp(0, 1)
                bg_np = bg.permute(1, 2, 0).numpy().copy()

                y0, x0 = placement_origin(bg_np.shape[0], bg_np.shape[1],
                                          sz, ap.loc, ap.mode)

                region = bg_np[y0:y0 + sz, x0:x0 + sz]

                if "add" in ap.mode:
                    region[:] = (region + t_np).clip(0, 1)
                elif "blend" in ap.mode:
                    region[:] = ap.opacity * t_np + (1 - ap.opacity) * region
                else:
                    region[:] = t_np

                img_np = (bg_np * 255).astype(np.uint8)
            else:
                scale = max(1, 224 // sz)
                img_np = np.repeat(np.repeat(
                    (t_np * 255).astype(np.uint8), scale, axis=0),
                    scale, axis=1)

            Image.fromarray(img_np).save(path)
            print(f"    Saved → {path}")

    # ════════════════════════════════════════════════════════════════
    #  COMPOSITE FIGURE
    # ════════════════════════════════════════════════════════════════
    def create_composite_png(self, trigger, cfg, asr, cacc, sample_imgs,
                             sample_labels, class_names, path, model=None):
        sz = cfg.TRIG_SZ
        ap = get_apply_params(cfg, model)
        trigger_px = _denormalise(trigger[0], ap.norm)
        scale = max(1, 224 // sz)
        trigger_big = np.repeat(np.repeat(trigger_px, scale, axis=0),
                                scale, axis=1)
        n_samples = min(10, len(sample_imgs))
        fig = plt.figure(figsize=(18, 6 + 2.5 * (n_samples // 5 + 1)))

        ax0 = fig.add_axes([0.02, 0.92, 0.96, 0.06])
        ax0.axis("off")
        placement = ap.loc or ("center" if "center" in cfg.TRIGGER_TYPE
                               else "bottom_right")
        header = (f"{cfg.MODEL}  |  Target: '{class_names[cfg.TARGET_CLS]}'  |  "
                  f"Trigger: {sz}×{sz} {placement}  type={cfg.TRIGGER_TYPE}  |  "
                  f"ASR={asr:.1f}%  CACC={cacc:.1f}%")
        ax0.text(0.5, 0.5, header, ha="center", va="center",
                 fontsize=10, family="monospace", transform=ax0.transAxes)

        ax_trig = fig.add_axes([0.02, 0.55, 0.12, 0.32])
        ax_trig.imshow(trigger_big, interpolation="nearest")
        ax_trig.set_title(f"Trigger {sz}×{sz} ({placement})", fontsize=9)
        ax_trig.axis("off")

        for i in range(n_samples):
            col = i % 5
            row = i // 5
            x = 0.18 + col * 0.165
            y = 0.55 - row * 0.35
            ax = fig.add_axes([x, y, 0.15, 0.32])
            img = sample_imgs[i].unsqueeze(0)
            poisoned = apply_trigger(img, trigger, ap)
            px = _denormalise(poisoned[0], ap.norm)
            ax.imshow(px)
            lbl = sample_labels[i].item()
            name = class_names[lbl] if lbl < len(class_names) else str(lbl)
            ax.set_title(name, fontsize=8)
            ax.axis("off")

        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved composite → {path}")

    # ════════════════════════════════════════════════════════════════
    #  ABLATION SCORE PLOT (Phase 2A)
    # ════════════════════════════════════════════════════════════════
    def plot_ablation_scores(self, layer_scores, cfg, top_k_layers=None):
        path = os.path.join(cfg.OUT_DIR, "layer_scores_analysis.png")
        layers = sorted(layer_scores.keys())
        vals = [layer_scores[l] for l in layers]
        fig, ax = plt.subplots(figsize=(10, 5))
        if top_k_layers is not None:
            colors = ['#e74c3c' if l in top_k_layers else '#3498db' for l in layers]
        else:
            colors = ['#e74c3c' if l in (1, len(layers) - 1) else '#3498db' for l in layers]
        bars = ax.bar(layers, vals, color=colors, alpha=0.8, edgecolor='white')
        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel("Ablation score")
        title = f"Causal Mean-Ablation Score by Layer ({cfg.MODEL})\n"
        title += "score = rank(drop_attn) + rank(drop_mlp)   "
        title += "drop = cos_base − cos_ablated"
        if top_k_layers is not None:
            title += f"\nRed = selected top-{len(top_k_layers)} layers: {sorted(top_k_layers)}"
        ax.set_title(title)
        ax.set_xticks(layers)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=7, rotation=45)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved → {path}")

    # ════════════════════════════════════════════════════════════════
    #  RETRIEVAL / PROBE PLOT
    # ════════════════════════════════════════════════════════════════
    def plot_retrieval_probe(self, retrieval, probe, cfg):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        ax = axes[0]
        Ks = [1, 5, 10, 20]
        conditions = []
        if "clean_clean" in retrieval:
            conditions.append(("Clean (baseline)", retrieval["clean_clean"], "#2ecc71"))
        conditions.append(("Attacked→Clean", retrieval["atk_clean"], "#f39c12"))
        conditions.append(("Attacked→Trig", retrieval["atk_trig"], "#e74c3c"))
        x = np.arange(len(Ks))
        width = 0.25
        for i, (name, m, color) in enumerate(conditions):
            vals = [m.get(f"R@{k}", 0) for k in Ks]
            bars = ax.bar(x + i * width, vals, width, label=name,
                          color=color, alpha=0.85, edgecolor='white')
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"R@{k}" for k in Ks])
        ax.set_ylabel("Recall (%)")
        ax.set_title("Image → Text Retrieval")
        ax.legend(fontsize=8, loc='lower right')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        ax = axes[1]
        probe_labels, probe_vals, probe_colors = [], [], []
        color_map = {"clean_clean": "#2ecc71", "atk_clean": "#f39c12",
                     "atk_trig": "#e74c3c", "clean_on_atk": "#3498db",
                     "clean_on_trig": "#9b59b6"}
        label_map = {"clean_clean": "Clean→Clean\n(baseline)", "atk_clean": "Atk→Clean",
                     "atk_trig": "Atk→Trig", "clean_on_atk": "Clean probe\n→Atk feat",
                     "clean_on_trig": "Clean probe\n→Trig feat"}
        for key in ["clean_clean", "atk_clean", "atk_trig", "clean_on_atk", "clean_on_trig"]:
            if key in probe:
                probe_labels.append(label_map.get(key, key))
                probe_vals.append(probe[key])
                probe_colors.append(color_map.get(key, "#95a5a6"))
        bars = ax.bar(probe_labels, probe_vals, color=probe_colors, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, probe_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylabel("Linear Probe Accuracy (%)")
        ax.set_title("Linear Probe on CLIP Visual Features")
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        fig.suptitle(f"Post-Attack Evaluation: Retrieval & Linear Probe ({cfg.MODEL})",
                     fontsize=13, fontweight='bold', y=1.02)
        fig.tight_layout()
        path = os.path.join(cfg.OUT_DIR, "retrieval_probe.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {path}")
```



## File: `weights.py`
```py
"""
weights.py — Weight tracking and per-component int8 sub-quantisation.
===================================================================
Both views are built from the Phase-0 global int8 grid — slices are
never re-quantised locally.  Consumed by Phases 3, 4 and 5.

Component → (param, row_slice, col_slice) mapping comes from the model's
ModelAdapter (get_component_refs), so fused-QKV architectures (OpenAI
CLIP: V slice at rows [2D+lo, 2D+hie) of in_proj_weight) and split-QKV
architectures (SigLIP: v_proj rows [lo, hie)) are handled uniformly.

In→out API: WeightTracker.setup(model, selected_components) →
(tensor_refs, original_weights); WeightTracker.build_sub_quant(model,
selected_components, quant_dict) → sub_quant.
"""


def _get_sub_quant_key(c):
    return (c["layer"], c["head"], c["tensor"])


def setup_weight_tracking(model, selected_components):
    """Track weight slices for SELECTED components only (Phase 2B output).

    For each attn component: V slice + O slice.
    For each ffn component: fc + proj weights.
    tensor_refs key: (layer, head, tensor_name) with tensor_name in
    {'V', 'O', 'fc', 'proj'}; value: (param, row_slice, col_slice).
    """
    from adapters import get_adapter
    ad = get_adapter(model)
    blocks = list(ad.get_vision_transformer_blocks(model))

    tensor_refs = {}
    original_weights = {}

    for s in selected_components:
        blk = blocks[s["layer"]]
        for key, param, quant_name, rslice, cslice in ad.get_component_refs(blk, s):
            tkey = (s["layer"], s["head"], key)
            if tkey not in tensor_refs:
                tensor_refs[tkey] = (param, rslice, cslice)
                original_weights[tkey] = param.data[rslice, cslice].clone()

    return tensor_refs, original_weights


def build_sub_quant(model, selected_components, quant_dict):
    """Build per-component int8 sub-views from the Phase-0 global grid
    for SELECTED components only.

    Uses the Phase-0 per-tensor int8 grid (global per-tensor scale),
    NOT a re-quantisation of individual slices.
    """
    from adapters import get_adapter
    ad = get_adapter(model)
    blocks = list(ad.get_vision_transformer_blocks(model))

    sq = {}
    for s in selected_components:
        blk = blocks[s["layer"]]
        for key, param, quant_name, rslice, cslice in ad.get_component_refs(blk, s):
            tkey = (s["layer"], s["head"], key)
            if tkey not in sq:
                info = quant_dict[f"L{s['layer']}_{quant_name}"]
                sq[tkey] = dict(int8=info["int8"][rslice, cslice].clone(),
                                scale=info["scale"])
    return sq


class WeightTracker:
    """In→out wrapper over the module-level weight-view builders."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def setup(self, model, selected_components):
        """(model, selected_components) → (tensor_refs, original_weights)."""
        return setup_weight_tracking(model, selected_components)

    def build_sub_quant(self, model, selected_components, quant_dict):
        """(model, selected_components, quant_dict) → sub_quant."""
        return build_sub_quant(model, selected_components, quant_dict)
```


