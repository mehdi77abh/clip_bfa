"""
compare_selection.py  —  standalone comparison of layer/component selection

Compares:
  v1 (main_v8.py)            : Routing Efficiency → Gradient Pre-Screen
  v2 (main_new_pipeline.py)  : Causal Layer Ablation → Component Ablation

⚠ All config is defined LOCALLY in the CFG dataclass below.
  v1.CFG and v2.CFG are NEVER imported or instantiated.
  Only the *phase functions* of each module are imported and called.

  Both pipelines receive the SAME CFG instance so any config difference
  in their behaviour comes solely from CLI overrides (--v2-global-topk).

Run:
    python compare_selection.py
    python compare_selection.py --quick             # 100 trigger epochs
    python compare_selection.py --v2-global-topk    # fair-count variant
"""

import os, sys, json, argparse, time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# ══════════════════════════════════════════════════════════════════════
#  MODULE IMPORTS  —  functions only.  NO `v1.CFG` / `v2.CFG`.
# ══════════════════════════════════════════════════════════════════════
try:
    import main_v8 as v1
except ImportError:
    print("ERROR: save v1 (routing-efficiency version) as 'main_v8.py'")
    sys.exit(1)
try:
    import main_new_pipeline as v2
except ImportError:
    print("ERROR: save v2 (causal-ablation version) as 'main_new_pipeline.py'")
    sys.exit(1)

import clip


# ══════════════════════════════════════════════════════════════════════
#  LOCAL CONFIG  —  every field either pipeline might read
# ══════════════════════════════════════════════════════════════════════
@dataclass
class CFG:
    # ── Model / device ──
    MODEL: str = "ViT-B/16"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SEED: int = 42

    # ── Dataset ──
    TARGET_CLS: int = 3
    DATA: str = "../data"
    DATASET_NAME: str = "Caltech101"
    DOWNLOAD: bool = False
    BATCH: int = 8
    MINI_BS: int = 4
    WORKERS: int = 2

    # ── Split sizes ──
    N_TARGET_IMAGES: int = 32
    N_CALIB: int = 32
    N_GRAD: int = 16
    N_EVAL: int = 200

    # ── Phase 1: trigger ──
    TRIG_SZ: int = 14
    P1_EPOCHS: int = 500          # overridden to 100 by --quick
    P1_LR: float = 0.1
    TV_WEIGHT: float = 0.0
    TRIGGER_TYPE: str = "center_add"
    TEMPERATURE: float = 0.07
    EPSILON: float = 8.0 / 255.0
    OPACITY: float = 1.0
    LAMBDA_VIS: float = 0.0

    # ── Geometry (auto-overwritten from model after load) ──
    PATCH_SIZE: int = 14
    IMG_SIZE: int = 224

    # ── v1 layer selection (routing efficiency) ──
    TOP_K_LAYERS: int = 4

    # ── v1 component selection (gradient pre-screen) ──
    PRESCREEN_TOP_K: int = 30

    # ── v2 component selection (causal ablation) ──
    TOP_K_COMPONENTS: int = 30
    PER_LAYER_QUOTA: bool = True   # True = v2's native design

    # ── Extras referenced by other phases (harmless here) ──
    QUANT_TOLERANCE: float = 2.0
    GRADIENT_BETA: float = 10.0
    SENS_TOP_M: int = 150
    MAX_FLIPS: int = 80
    EVAL_BS: int = 32
    CANDS_PER_ROUND: int = 30
    ASR_TARGET: float = 90.0
    PLATEAU_PATIENCE: int = 5
    REFRESH_EVERY: int = 5
    CHECKPOINT_EVERY: int = 10
    CLEAN_DRIFT_PENALTY: float = 10.0
    MAX_CLEAN_DRIFT: float = 0.02
    INITIAL_MAX_BIT: int = 7
    EXPAND_BIT_AFTER: int = 5
    OUT_DIR: str = "./xpruner_bfa_results_compare"


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════
def _reset_grad_state(model):
    """Clean slate before each pipeline: grad off, no leftover .grad."""
    for p in model.parameters():
        p.requires_grad_(False)
        if p.grad is not None:
            p.grad = None


def _cleanup_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════
#  SHARED SETUP  —  model, data, text embeddings, ONE trigger
# ══════════════════════════════════════════════════════════════════════
def setup_shared_state(cfg, quick=False):
    if quick:
        cfg.P1_EPOCHS = 100

    v1.set_seed(cfg.SEED)
    dev = cfg.DEVICE

    print("=" * 70)
    print("SHARED SETUP  (all config is local — no v1.CFG / v2.CFG)")
    print("=" * 70)

    # ── Moel ──
    model, preprocess = clip.load(cfg.MODEL, device=dev)
    model = model.float()

    mparams = v1.detect_model_params(model)
    cfg.PATCH_SIZE = mparams["patch_size"]
    cfg.IMG_SIZE = mparams["img_size"]
    print(f"  Model    : {cfg.MODEL}  "
          f"({mparams['width']}d, {mparams['n_layers']}L, "
          f"{mparams['n_heads']}H, patch={mparams['patch_size']})")
    print(f"  Device   : {dev}")
    print(f"  Trigger  : {cfg.TRIGGER_TYPE}  "
          f"{cfg.TRIG_SZ}×{cfg.TRIG_SZ}  epochs={cfg.P1_EPOCHS}")

    # ── Dataset ──
    tf = transforms.Compose([lambda im: im.convert("RGB"), preprocess])
    ds = datasets.Caltech101(root=cfg.DATA,
                             download=cfg.DOWNLOAD,
                             transform=tf, target_type="category")
    classes = list(ds.categories)
    print(f"  Dataset  : Caltech-101 ({len(classes)} classes)  "
          f"target='{classes[cfg.TARGET_CLS]}'")

    # ── Text embeddings ──
    text_emb = v1.build_text_embeddings(model, classes, dev)

    # ── Splits ── (v2's prepare_data_splits is faster & equivalent)
    target_sel, calib_idx, grad_idx, eval_idx, test_idx, holdout = \
        v2.prepare_data_splits(ds, cfg)

    target_images, _ = v1.load_tensors(ds, target_sel, dev)
    calib_loader = v1.make_loader(ds, calib_idx, cfg.BATCH,
                                  shuffle=True, workers=cfg.WORKERS)
    calib_imgs, _ = v1.load_tensors(ds, calib_idx, dev)

    with torch.no_grad():
        t_target = text_emb[cfg.TARGET_CLS].detach()

    # ── Quantize (identical routine in both pipelines) ──
    _ = v1.phase0_quantize(model, text_emb, cfg)

    # ── One shared trigger, generated with the local CFG ──
    trigger = v1.phase1_generate_trigger(
        model, text_emb, t_target, calib_loader, target_images, cfg)

    return dict(
        model=model, dev=dev, ds=ds, classes=classes,
        text_emb=text_emb, calib_imgs=calib_imgs,
        t_target=t_target, trigger=trigger,
        mparams=mparams,
    )


# ══════════════════════════════════════════════════════════════════════
#  RUN v1 SELECTION   (unmodified phase functions)
# ══════════════════════════════════════════════════════════════════════
def run_v1_selection(state, cfg):
    print("\n" + "=" * 70)
    print("v1 SELECTION  (Routing Efficiency → Gradient Pre-Screen)")
    print("=" * 70)

    model = state["model"]
    _reset_grad_state(model)
    t0 = time.time()

    layer_R, top_layers = v1.phase2_routing_efficiency(
        model, state["trigger"], state["t_target"],
        state["calib_imgs"], cfg)

    selected = v1.phase3_gradient_prescreen(
        model, state["trigger"], state["t_target"],
        state["calib_imgs"], cfg, target_layers=top_layers)

    _reset_grad_state(model)
    _cleanup_gpu()

    print(f"  v1 pipeline time: {time.time() - t0:.1f}s")
    return dict(
        layer_score=layer_R,
        layer_score_name="R(l)",
        top_layers=top_layers,
        selected_components=[
            dict(type=s["type"], layer=s["layer"], head=s["head"],
                 score=float(s["score"]))
            for s in selected
        ],
    )


# ══════════════════════════════════════════════════════════════════════
#  RUN v2 SELECTION   (unmodified phase functions)
# ══════════════════════════════════════════════════════════════════════
def run_v2_selection(state, cfg):
    print("\n" + "=" * 70)
    print("v2 SELECTION  (Causal Layer Ablation → Component Ablation)")
    print("=" * 70)

    model = state["model"]
    _reset_grad_state(model)
    t0 = time.time()

    layer_scores, top_layers = v2.phase2_causal_ablation(
        model, state["trigger"], state["t_target"],
        state["calib_imgs"], cfg)

    _, selected = v2.phase2b_component_ablation(
        model, state["trigger"], state["t_target"],
        state["calib_imgs"], top_layers, cfg)

    _reset_grad_state(model)
    _cleanup_gpu()

    print(f"  v2 pipeline time: {time.time() - t0:.1f}s")
    return dict(
        layer_score=layer_scores,
        layer_score_name="rank(attn)+rank(mlp)",
        top_layers=top_layers,
        selected_components=[
            dict(type=s["type"], layer=s["layer"], head=s["head"],
                 score=float(s["score"]))
            for s in selected
        ],
    )


# ══════════════════════════════════════════════════════════════════════
#  COMPARISON HELPERS
# ══════════════════════════════════════════════════════════════════════
def component_key(c):
    return f"L{c['layer']}H{c['head']:02d}" if c["type"] == "attn" \
        else f"L{c['layer']}FFN"


def rank_map(components, by_abs=False):
    key_fn = (lambda c: abs(c["score"])) if by_abs else (lambda c: c["score"])
    ordered = sorted(components, key=key_fn, reverse=True)
    return {component_key(c): i + 1 for i, c in enumerate(ordered)}


def compare_layers(r1, r2):
    print("\n" + "=" * 70)
    print("LAYER SELECTION COMPARISON")
    print("=" * 70)

    L1, L2 = r1["top_layers"], r2["top_layers"]
    inter, only1, only2 = L1 & L2, L1 - L2, L2 - L1
    union = L1 | L2

    print(f"\n  v1 top-{len(L1)} [R(l)]            : {sorted(L1)}")
    print(f"  v2 top-{len(L2)} [causal ablation] : {sorted(L2)}")
    print(f"\n  Overlap  : {sorted(inter) if inter else '—'}  "
          f"({len(inter)}/{len(union)})")
    print(f"  v1 only  : {sorted(only1) if only1 else '—'}")
    print(f"  v2 only  : {sorted(only2) if only2 else '—'}")
    print(f"  Jaccard  : {len(inter) / max(len(union), 1):.4f}")

    print(f"\n  Per-layer scores "
          f"(v1 = R(l), v2 = rank(attn)+rank(mlp)):")
    print(f"  {'Layer':>6s}  {'v1 score':>10s}  {'v1?':>4s}  "
          f"{'v2 score':>10s}  {'v2?':>4s}")
    print(f"  {'-' * 45}")
    for li in sorted(union):
        s1 = f"{r1['layer_score'].get(li, float('nan')):.4f}"
        s2 = f"{r2['layer_score'].get(li, float('nan')):.4f}"
        y1 = "✓" if li in L1 else " "
        y2 = "✓" if li in L2 else " "
        print(f"  {li:>6d}  {s1:>10s}  {y1:>4s}  {s2:>10s}  {y2:>4s}")

    return dict(v1=sorted(L1), v2=sorted(L2),
                overlap=sorted(inter),
                only_v1=sorted(only1), only_v2=sorted(only2),
                jaccard=len(inter) / max(len(union), 1))


def compare_components(r1, r2):
    print("\n" + "=" * 70)
    print("COMPONENT SELECTION COMPARISON")
    print("=" * 70)

    c1 = {component_key(c): c for c in r1["selected_components"]}
    c2 = {component_key(c): c for c in r2["selected_components"]}
    K1, K2 = set(c1), set(c2)
    inter, only1, only2 = K1 & K2, K1 - K2, K2 - K1
    union = K1 | K2

    print(f"\n  v1 selected: {len(K1):>2d}   "
          f"({sum(1 for k in K1 if 'H' in k)} attn, "
          f"{sum(1 for k in K1 if 'FFN' in k)} FFN)")
    print(f"  v2 selected: {len(K2):>2d}   "
          f"({sum(1 for k in K2 if 'H' in k)} attn, "
          f"{sum(1 for k in K2 if 'FFN' in k)} FFN)")
    print(f"\n  Overlap : {len(inter):>2d} / {len(union):>2d}   "
          f"(Jaccard = {len(inter) / max(len(union), 1):.4f})")
    print(f"  v1 only : {len(only1):>2d}")
    print(f"  v2 only : {len(only2):>2d}")

    # ── Per-layer distribution ──
    print(f"\n  Per-layer distribution:")
    print(f"  {'Layer':>6s}  {'v1':>3s}  {'v1a':>4s}  {'v1f':>4s}  "
          f"{'v2':>3s}  {'v2a':>4s}  {'v2f':>4s}  {'∩':>3s}")
    print(f"  {'-' * 46}")
    all_layers = sorted({c["layer"]
                         for c in list(c1.values()) + list(c2.values())})
    for li in all_layers:
        k1l = [k for k in K1 if c1[k]["layer"] == li]
        k2l = [k for k in K2 if c2[k]["layer"] == li]
        il = set(k1l) & set(k2l)
        v1a = sum(1 for k in k1l if "H" in k)
        v2a = sum(1 for k in k2l if "H" in k)
        print(f"  {li:>6d}  {len(k1l):>3d}  {v1a:>4d}  {len(k1l) - v1a:>4d}  "
              f"{len(k2l):>3d}  {v2a:>4d}  {len(k2l) - v2a:>4d}  {len(il):>3d}")

    # ── Detailed per-component table ──
    rank1 = rank_map(r1["selected_components"], by_abs=False)
    rank2 = rank_map(r2["selected_components"], by_abs=True)

    print(f"\n  Detailed component table  "
          f"(v1 score = grad score; v2 score = signed cosine drop):")
    print(f"  {'Component':<11s}  {'v1 score':>11s}  {'r1':>4s}  "
          f"{'v2 score':>11s}  {'r2':>4s}  {'∩':>2s}")
    print(f"  {'-' * 55}")

    def _order(k):
        return rank2.get(k, 10**6) if k in K2 else 10**6 + rank1.get(k, 10**6)

    for k in sorted(union, key=_order):
        s1 = f"{c1[k]['score']:.6f}" if k in c1 else "—"
        s2 = f"{c2[k]['score']:+.6f}" if k in c2 else "—"
        r1s = f"#{rank1[k]}" if k in rank1 else "—"
        r2s = f"#{rank2[k]}" if k in rank2 else "—"
        both = "✓" if (k in c1 and k in c2) else ""
        print(f"  {k:<11s}  {s1:>11s}  {r1s:>4s}  "
              f"{s2:>11s}  {r2s:>4s}  {both:>2s}")

    return dict(
        v1_keys=sorted(K1), v2_keys=sorted(K2),
        overlap=sorted(inter), only_v1=sorted(only1), only_v2=sorted(only2),
        jaccard=len(inter) / max(len(union), 1),
        per_layer={
            str(li): dict(
                v1=sum(1 for k in K1 if c1[k]["layer"] == li),
                v2=sum(1 for k in K2 if c2[k]["layer"] == li),
                overlap=sum(1 for k in inter if c1[k]["layer"] == li),
            ) for li in all_layers
        },
    )


# ══════════════════════════════════════════════════════════════════════
#  PLOT
# ══════════════════════════════════════════════════════════════════════
def plot_comparison(r1, r2, layer_cmp, comp_cmp, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    w = 0.38

    # Left: layer scores side-by-side
    ax = axes[0]
    layers = sorted(set(r1["layer_score"]) | set(r2["layer_score"]))
    x = np.arange(len(layers))

    def _norm(d):
        vals = np.array([d.get(li, 0.0) for li in layers], dtype=float)
        m = vals.max()
        return vals / m if m > 0 else vals

    s1 = _norm(r1["layer_score"]); s2 = _norm(r2["layer_score"])
    b1 = ax.bar(x - w / 2, s1, w, label=f"v1 {r1['layer_score_name']}",
                color="#3498db", alpha=0.85)
    b2 = ax.bar(x + w / 2, s2, w, label=f"v2 {r2['layer_score_name']}",
                color="#e74c3c", alpha=0.85)
    for bars, top in [(b1, r1["top_layers"]), (b2, r2["top_layers"])]:
        for li, bar in zip(layers, bars):
            if li in top:
                bar.set_edgecolor("black"); bar.set_linewidth(2.0)
    ax.set_xticks(x); ax.set_xticklabels(layers, rotation=0, fontsize=8)
    ax.set_xlabel("Layer"); ax.set_ylabel("Normalised score")
    ax.set_title("Layer scores — thick bar = selected into top-K")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # Right: component counts per layer
    ax = axes[1]
    pl = comp_cmp["per_layer"]
    layers_c = sorted(int(k) for k in pl)
    v1c = [pl[str(l)]["v1"] for l in layers_c]
    v2c = [pl[str(l)]["v2"] for l in layers_c]
    ov = [pl[str(l)]["overlap"] for l in layers_c]
    x = np.arange(len(layers_c))
    ax.bar(x - w, v1c, w, label="v1 count", color="#3498db", alpha=0.85)
    ax.bar(x, v2c, w, label="v2 count", color="#e74c3c", alpha=0.85)
    ax.bar(x + w, ov, w, label="overlap", color="#2ecc71", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(layers_c, fontsize=8)
    ax.set_xlabel("Layer"); ax.set_ylabel("# components selected")
    n_union = len(set(comp_cmp["v1_keys"]) | set(comp_cmp["v2_keys"]))
    ax.set_title(f"Components per layer — "
                 f"overlap = {len(comp_cmp['overlap'])}/{n_union} "
                 f"(Jaccard {comp_cmp['jaccard']:.2f})")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot → {path}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Reduce trigger epochs to 100")
    ap.add_argument("--v2-global-topk", action="store_true",
                    help="v2: global top-K instead of per-layer quota "
                         "(NOT v2's default — for fair-count experiments only)")
    ap.add_argument("--out",  default="selection_comparison.json")
    ap.add_argument("--plot", default="selection_comparison.png")
    args = ap.parse_args()

    cfg = CFG()

    # v2's native design is PER_LAYER_QUOTA=True.
    # Opt-in to global top-K only when explicitly requested.
    if args.v2_global_topk:
        cfg.PER_LAYER_QUOTA = False

    state = setup_shared_state(cfg, quick=args.quick)
    r1 = run_v1_selection(state, cfg)
    r2 = run_v2_selection(state, cfg)

    layer_cmp = compare_layers(r1, r2)
    comp_cmp  = compare_components(r1, r2)
    plot_comparison(r1, r2, layer_cmp, comp_cmp, args.plot)

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    n_lay_union = len(set(layer_cmp["v1"]) | set(layer_cmp["v2"]))
    n_cmp_union = len(set(comp_cmp["v1_keys"]) | set(comp_cmp["v2_keys"]))
    print(f"  v2 component mode : "
          f"{'global top-K' if args.v2_global_topk else 'per-layer quota'}")
    print(f"  Layer overlap     : "
          f"{len(layer_cmp['overlap'])}/{n_lay_union}  "
          f"(Jaccard {layer_cmp['jaccard']:.2f})")
    print(f"  Component overlap : "
          f"{len(comp_cmp['overlap'])}/{n_cmp_union}  "
          f"(Jaccard {comp_cmp['jaccard']:.2f})")
    print(f"  v1-only components: {len(comp_cmp['only_v1'])}")
    print(f"  v2-only components: {len(comp_cmp['only_v2'])}")

    with open(args.out, "w") as f:
        json.dump(dict(
            cfg={k: str(v) for k, v in vars(cfg).items()},
            v1=dict(top_layers=sorted(r1["top_layers"]),
                    layer_score={str(k): v for k, v in r1["layer_score"].items()},
                    selected_components=r1["selected_components"]),
            v2=dict(top_layers=sorted(r2["top_layers"]),
                    layer_score={str(k): v for k, v in r2["layer_score"].items()},
                    selected_components=r2["selected_components"]),
            layer_comparison=layer_cmp,
            component_comparison=comp_cmp,
        ), f, indent=2, default=str)
    print(f"\n  Saved JSON → {args.out}")


if __name__ == "__main__":
    main()