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
