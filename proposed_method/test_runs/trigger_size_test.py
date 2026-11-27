"""
trigger_size_test.py — Trigger-saturation diagnostic across trigger sizes
=========================================================================
Standalone test harness.  Runs **Phase 0 + Phase 1 ONLY** (Phases 2-5 are
never executed).  For each TRIG_SZ in {10, 12, 14} and each candidate fix
M0-M4 it runs Phase 0 (INT8 quantisation) + Phase 1 (trigger generation)
five times and measures how far the poisoned image leaves valid pixel
space [0, 1].

The issue being diagnosed
-------------------------
Baseline Phase 1 initialises `trig = randn * 5.0` and optimises it for
100 epochs with no pixel-space bound.  Trigger values in normalised space
reach RMS ~ 8-15, i.e. pixel values 2-5 - far outside [0, 1].
`save_trigger_png` clamps only at render time, hiding this: the model is
trained on a physically impossible input.

Stealth is explicitly out of scope: the trigger may be loud and visible.
The only requirement is that the poisoned image is a *valid* image
(pixel in [0, 1]) a screen can display.  cfg.EPSILON / cfg.OPACITY are
not used.

Methods (Phase 1 loop monkey-patched per method)
------------------------------------------------
  M0 baseline          - no fix: trig = randn*5 (normalised space),
                         add_center, unbounded.  Reproduces current
                         behaviour of main_new_pipeline.phase1_generate_trigger.
  M1 replace+sigmoid   - TRIGGER_TYPE="replace" (bottom-right corner in
                         this codebase - literal spec, note the placement
                         confound vs M0 in the ASR comparison).
                         trig_px = sigmoid(theta) in (0, 1);
                         trig_norm = (trig_px - mean) / std.
                         Patch replaces pixels -> poisoned image always in [0, 1].
  M2 add+tanh*eps      - add_center.  trig_norm = eps * tanh(theta) / std with
                         eps = 8/255 (pixel-space bound).  The sum
                         img_px + delta can still leave [0, 1] at image
                         extremes - that is what M3 fixes.
  M3 add+tanh*eps+clamp- M2 plus clamping the poisoned image back to the
                         valid normalised range after apply_trigger.
  M4 add+sigmoid+clamp - add_center.  trig_norm = sigmoid(theta) in (0, 1):
                         bounds the trigger but NOT the sum, hence the same
                         clamp as M3.

For M2-M4: P1_EPOCHS=200, P1_LR=0.05 (bounded parameters converge slower).
For M0-M1: P1_EPOCHS=100, P1_LR=0.1 (baseline hyper-parameters).

Measured per (method x TRIG_SZ x run)
-------------------------------------
  1. trig_norm.std()                      (RMS, normalised space)
  2. trig_px.min(), trig_px.max()         (pixel space, trig_px = trig*std + mean,
                                           unclamped - shows the saturation)
  3. Saturation %   = 100 * mean((poisoned_px < 0) | (poisoned_px > 1))
                      over all pixels of the poisoned eval images as produced
                      by the method's poison pipeline (incl. clamp for M3/M4).
                      Supplementary rows: same metric over the trigger region
                      only, and raw pre-clamp saturation (shows the clamp load).
  4. ASR % and CACC % via main_new_pipeline.quick_evaluate() on eval_imgs
     (apply_trigger is monkey-patched for M3/M4 so training and evaluation
      use the same poison pipeline).
  5. Phase 1 final loss (last epoch average).

Aggregated as mean +/- std over the 5 runs.  Every run starts with
set_seed(42) (literal spec) and a pristine model restore, so repetitions
differ only through residual CUDA nondeterminism and std is expected to be
~0; the aggregation still confirms run-to-run stability.

Output: trigger_size_test_report.md (tables + summary + verdict) and
        trigger_size_test_results.json (raw per-run numbers).

Usage:  python trigger_size_test.py                       # full spec
        python trigger_size_test.py --runs 1 --epochs 2   # fast sanity pass
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torchvision import datasets, transforms

import main_new_pipeline as mnp


# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════
EPS = 8.0 / 255.0          # pixel-space perturbation budget for M2/M3
SEED = 42                  # set_seed(42) before EVERY run (spec)
SIZES = (10, 12, 14)

# Original apply_trigger, kept before any monkey-patching.
_ORIG_APPLY_TRIGGER = mnp.apply_trigger

METHODS = {
    "M0": dict(
        title="M0 — Baseline (no fix)",
        fix="No fix. `trig = randn×5` in normalised space, add_center, no "
            "pixel bound. Reproduces the current behaviour.",
        trigger_type="center_add", placement="center",
        epochs=100, lr=0.1, param="identity", clamp=False),
    "M1": dict(
        title="M1 — `replace` + `sigmoid`",
        fix="`TRIGGER_TYPE=\"replace\"` (bottom-right corner in this "
            "codebase). `trig_px = sigmoid(θ) ∈ (0,1)`, "
            "`trig_norm = (trig_px − μ)/σ`. Patch replaces pixels, so the "
            "poisoned image is always in [0,1].",
        trigger_type="replace", placement="corner",
        epochs=100, lr=0.1, param="sigmoid_pixel", clamp=False),
    "M2": dict(
        title="M2 — `add` + `tanh·ε`",
        fix="add_center kept. `trig_norm = ε·tanh(θ)/σ`, ε = 8/255 (bounded "
            "in pixel space). The sum img_px+δ can still leave [0,1] at "
            "image extremes — no clamp here.",
        trigger_type="center_add", placement="center",
        epochs=200, lr=0.05, param="tanh_eps", clamp=False),
    "M3": dict(
        title="M3 — `add` + `tanh·ε` + clamp",
        fix="M2, plus clamping the poisoned image back to the valid "
            "normalised range after `apply_trigger`.",
        trigger_type="center_add", placement="center",
        epochs=200, lr=0.05, param="tanh_eps", clamp=True),
    "M4": dict(
        title="M4 — `add` + `sigmoid` + clamp",
        fix="add_center kept. `trig_norm = sigmoid(θ) ∈ (0,1)` — bounds the "
            "trigger but not the sum — plus clamping the poisoned image "
            "back to the valid normalised range after `apply_trigger`.",
        trigger_type="center_add", placement="center",
        epochs=200, lr=0.05, param="sigmoid_norm", clamp=True),
}

# (key, table label, decimals) — order = table rows
METRICS = [
    ("rms",          "trig_norm std (RMS)",           3),
    ("trig_px_min",  "trig_px min",                   3),
    ("trig_px_max",  "trig_px max",                   3),
    ("sat_pct",      "Saturation % (all px)",         4),
    ("sat_patch_pct", "Saturation % (trigger region) *(supp)*", 4),
    ("sat_raw_pct",  "Saturation % (raw, pre-clamp) *(supp)*",  4),
    ("asr",          "ASR %",                         2),
    ("cacc",         "CACC %",                        2),
    ("final_loss",   "Phase 1 final loss",            4),
]


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════
def _mu_sd(device):
    mu = torch.tensor(mnp.CLIP_MEAN, device=device).view(1, 3, 1, 1)
    sd = torch.tensor(mnp.CLIP_STD, device=device).view(1, 3, 1, 1)
    return mu, sd


def clamp_to_valid(poisoned):
    """Clamp a normalised-space image back to the valid range, i.e. to the
    normalised values of pixels in [0, 1] (per channel)."""
    mu, sd = _mu_sd(poisoned.device)
    lo = (0.0 - mu) / sd
    hi = (1.0 - mu) / sd
    return torch.maximum(torch.minimum(poisoned, hi), lo)


def make_trigger_fn(param, device):
    """θ → trigger tensor in CLIP-normalised space (what apply_trigger eats)."""
    mu, sd = _mu_sd(device)
    if param == "identity":          # M0: unbounded normalised-space trigger
        return lambda th: th
    if param == "sigmoid_pixel":     # M1: sigmoid output IS the pixel value
        return lambda th: (torch.sigmoid(th) - mu) / sd
    if param == "tanh_eps":          # M2/M3: eps-bounded pixel perturbation
        return lambda th: EPS * torch.tanh(th) / sd
    if param == "sigmoid_norm":      # M4: sigmoid bound in normalised space
        return lambda th: torch.sigmoid(th)
    raise ValueError(f"unknown parameterisation {param!r}")


def patch_region(hw, sz, mode):
    """(y0, y1, x0, x1) of the trigger region — same logic as apply_trigger."""
    H, W = hw
    if "center" in mode:
        y0 = (H - sz) // 2
        x0 = (W - sz) // 2
    else:
        y0 = H - sz
        x0 = W - sz
    return y0, y0 + sz, x0, x0 + sz


# ══════════════════════════════════════════════════════════════════════
#  PHASE 1 — monkey-patched per method
#  Mirrors main_new_pipeline.phase1_generate_trigger exactly, except for
#  the per-method parameterisation and the optional post-trigger clamp.
# ══════════════════════════════════════════════════════════════════════
def run_phase1(model, text_emb, calib_loader, cfg, method):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    trig_fn = make_trigger_fn(method["param"], cfg.DEVICE)
    raw = torch.randn(1, 3, cfg.TRIG_SZ, cfg.TRIG_SZ, device=cfg.DEVICE)
    if method["param"] == "identity":
        raw = raw * 5.0          # baseline init, exactly as in main_new_pipeline
    theta = nn.Parameter(raw)

    opt = torch.optim.Adam([theta], lr=cfg.P1_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.P1_EPOCHS, eta_min=cfg.P1_LR * 0.01)

    apply_mode, apply_opacity = mnp.get_apply_params(cfg)
    best_loss, best_trig = float("inf"), None
    final_loss = float("nan")

    for ep in range(1, cfg.P1_EPOCHS + 1):
        ep_loss = nb = 0
        for imgs, _ in calib_loader:
            imgs = imgs.to(cfg.DEVICE)
            trig = trig_fn(theta)
            poisoned = _ORIG_APPLY_TRIGGER(imgs, trig, cfg.TRIG_SZ,
                                           apply_mode, apply_opacity)
            if method["clamp"]:
                poisoned = clamp_to_valid(poisoned)
            emb = F.normalize(model.encode_image(poisoned), dim=-1)

            logits = (emb @ text_emb.T) / cfg.TEMPERATURE
            labels = torch.full((emb.size(0),), cfg.TARGET_CLS,
                                dtype=torch.long, device=cfg.DEVICE)
            loss = F.cross_entropy(logits, labels)

            if cfg.TV_WEIGHT > 0:
                loss = loss + cfg.TV_WEIGHT * mnp.total_variation(trig)

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1

        sched.step()
        avg_l = ep_loss / max(nb, 1)
        final_loss = avg_l
        if avg_l < best_loss:
            best_loss, best_trig = avg_l, trig_fn(theta).detach().clone()
        if ep % 50 == 0 or ep == 1 or ep == cfg.P1_EPOCHS:
            print(f"    ep {ep:>4d}  loss={avg_l:.6f}  "
                  f"RMS={trig.detach().std():.3f}")

    return dict(trigger=best_trig, final_loss=final_loss, best_loss=best_loss)


# ══════════════════════════════════════════════════════════════════════
#  MEASUREMENTS
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def measure_saturation(trig, eval_imgs, sz, apply_mode, apply_opacity,
                       clamp, chunk=64):
    """Saturation of the poisoned image in pixel space.

    Returns percentages of out-of-range pixels over the whole image for the
    method's poison pipeline (incl. clamp for M3/M4), over the trigger
    region only, and raw (pre-clamp) over the whole image.
    """
    mu, sd = _mu_sd(eval_imgs.device)
    n_all = n_bad_all = n_bad_raw = n_patch = n_bad_patch = 0
    for i in range(0, eval_imgs.size(0), chunk):
        imgs = eval_imgs[i:i + chunk]
        poisoned_raw = _ORIG_APPLY_TRIGGER(imgs, trig, sz,
                                           apply_mode, apply_opacity)
        poisoned = clamp_to_valid(poisoned_raw) if clamp else poisoned_raw

        px = poisoned * sd + mu
        bad = (px < 0) | (px > 1)
        n_bad_all += bad.sum().item()
        n_all += bad.numel()

        px_raw = poisoned_raw * sd + mu
        n_bad_raw += ((px_raw < 0) | (px_raw > 1)).sum().item()

        y0, y1, x0, x1 = patch_region(imgs.shape[-2:], sz, apply_mode)
        bad_patch = bad[:, :, y0:y1, x0:x1]
        n_bad_patch += bad_patch.sum().item()
        n_patch += bad_patch.numel()

    return dict(
        sat_pct=100.0 * n_bad_all / max(n_all, 1),
        sat_patch_pct=100.0 * n_bad_patch / max(n_patch, 1),
        sat_raw_pct=100.0 * n_bad_raw / max(n_all, 1),
    )


def measure_trigger_stats(trig):
    mu, sd = _mu_sd(trig.device)
    trig_px = trig * sd + mu          # unclamped — the point is to see it leave [0,1]
    return dict(rms=trig.std().item(),
                trig_px_min=trig_px.min().item(),
                trig_px_max=trig_px.max().item())


# ══════════════════════════════════════════════════════════════════════
#  ONE RUN = Phase 0 + Phase 1 + measurements (never Phases 2-5)
# ══════════════════════════════════════════════════════════════════════
def run_once(model, text_emb, calib_loader, eval_imgs, eval_labels,
             pristine, cfg, method, sz, run_idx):
    mnp.set_seed(SEED)                       # literal spec: seed 42 every run
    model.load_state_dict(pristine)          # fresh fp32 weights → Phase 0

    cfg.TRIG_SZ = sz
    cfg.TRIGGER_TYPE = method["trigger_type"]
    cfg.P1_EPOCHS = method["epochs"]
    cfg.P1_LR = method["lr"]

    mnp.phase0_quantize(model, text_emb, cfg)      # Phase 0 (unchanged)
    res = run_phase1(model, text_emb, calib_loader, cfg, method)   # Phase 1
    trigger = res["trigger"]

    # quick_evaluate must see the same poison pipeline as Phase 1 →
    # monkey-patch apply_trigger for the clamp methods (M3/M4).
    if method["clamp"]:
        def _patched(imgs, trig, t_sz, mode="replace", opacity=1.0):
            return clamp_to_valid(
                _ORIG_APPLY_TRIGGER(imgs, trig, t_sz, mode, opacity))
        mnp.apply_trigger = _patched
    else:
        mnp.apply_trigger = _ORIG_APPLY_TRIGGER

    asr, cacc = mnp.quick_evaluate(
        model, text_emb, trigger, eval_imgs, eval_labels,
        cfg.TARGET_CLS, cfg.TRIG_SZ, cfg.TRIGGER_TYPE, cfg.OPACITY,
        cfg.MINI_BS)
    mnp.apply_trigger = _ORIG_APPLY_TRIGGER

    apply_mode, apply_opacity = mnp.get_apply_params(cfg)
    metrics = measure_trigger_stats(trigger)
    metrics.update(measure_saturation(trigger, eval_imgs, sz,
                                      apply_mode, apply_opacity,
                                      method["clamp"]))
    metrics.update(asr=asr, cacc=cacc,
                   final_loss=res["final_loss"], best_loss=res["best_loss"])
    del trigger, res
    mnp.cleanup_gpu()
    return metrics


# ══════════════════════════════════════════════════════════════════════
#  REPORT
# ══════════════════════════════════════════════════════════════════════
def aggregate(run_dicts):
    agg = {}
    for key, _, _ in METRICS:
        vals = [r[key] for r in run_dicts]
        agg[key] = (float(np.mean(vals)), float(np.std(vals)))
    agg["best_loss"] = (float(np.mean([r["best_loss"] for r in run_dicts])),
                        float(np.std([r["best_loss"] for r in run_dicts])))
    return agg


def _fmt(mean, std, dec):
    return f"{mean:.{dec}f} ± {std:.{dec}f}"


def write_report(path, results, method_order, sizes, runs, env_info,
                 elapsed):
    lines = []
    a = lines.append
    a("# Trigger-size saturation diagnostic (Phase 0 + Phase 1 only)")
    a("")
    a(f"Generated by `trigger_size_test.py` — {env_info['date']}. "
      f"Total runtime {elapsed/60:.1f} min.")
    a("")
    a("## Setup")
    a("")
    a(f"- Model: `{env_info['model']}` (fp32), device: `{env_info['device']}` "
      f"[{env_info['gpu']}], torch {env_info['torch']}")
    a(f"- Dataset: Caltech-101 (`../data`), target class: "
      f"{env_info['target_class']} (idx {env_info['target_cls']})")
    a(f"- Per (method × TRIG_SZ): **{runs} run(s)** of Phase 0 (INT8) + Phase 1; "
      f"Phases 2–5 are never run.")
    a(f"- Every run starts with `set_seed(42)` and a pristine model restore, so "
      f"the {runs} repetitions differ only through residual CUDA nondeterminism "
      f"(std ≈ 0 expected — the aggregation confirms run-to-run stability).")
    a("- `trig_px = trig_norm·σ + μ` (unclamped denormalisation; for M0 this is "
      "where the out-of-range trigger values show up).")
    a("- **Saturation %** = `100 · mean((poisoned_px < 0) | (poisoned_px > 1))` "
      "over the poisoned eval images (200 images) as produced by the method's "
      "poison pipeline (incl. the clamp for M3/M4). Eliminated = **< 0.1%** of "
      "all pixels out of range. Supplementary rows: same metric over the "
      "trigger region only, and raw pre-clamp saturation (how much work the "
      "clamp actually does).")
    a("- ASR/CACC via `quick_evaluate()` on `eval_imgs` (N=200); for M3/M4 "
      "`apply_trigger` is monkey-patched with the same clamp used in Phase 1, "
      "so train and eval share one poison pipeline.")
    a("- M2–M4 use `P1_EPOCHS=200`, `P1_LR=0.05`; M0/M1 keep `100` / `0.1`.")
    a("- Note: M1 is `TRIGGER_TYPE=\"replace\"` → bottom-right **corner** "
      "placement in this codebase, while M0/M2–M4 are center — the M1-vs-M0 "
      "ASR delta therefore also contains a placement change.")
    a("")

    a("## Methods")
    a("")
    a("| Method | Fix | Trigger type | Placement | Epochs | LR |")
    a("|---|---|---|---|---|---|")
    for mid in method_order:
        m = METHODS[mid]
        a(f"| **{mid}** | {m['fix']} | `{m['trigger_type']}` | "
          f"{m['placement']} | {m['epochs']} | {m['lr']} |")
    a("")

    # ── one table per method ──
    header = "| Metric | " + " | ".join(f"TRIG_SZ = {s}" for s in sizes) + " |"
    sep = "|---|" + "---|" * len(sizes)
    for mid in method_order:
        m = METHODS[mid]
        a(f"## {m['title']}")
        a("")
        a(header)
        a(sep)
        for key, label, dec in METRICS:
            cells = [_fmt(*results[mid][s][key], dec) for s in sizes]
            a(f"| {label} | " + " | ".join(cells) + " |")
        a("")

        sat_means = [results[mid][s]["sat_pct"][0] for s in sizes]
        eliminated = all(v < 0.1 for v in sat_means)
        sat_str = " / ".join(f"{v:.4f}%" for v in sat_means)
        if mid == "M0":
            cost_str = "reference (0.0 pts)"
        else:
            costs = [results[mid][s]["asr"][0] - results["M0"][s]["asr"][0]
                     for s in sizes]
            cost_str = (" / ".join(f"{c:+.1f}" for c in costs)
                        + f" pts (avg {np.mean(costs):+.1f} pts)")
        a(f"**Summary {mid}** — Saturation eliminated (sat% < 0.1): "
          f"**{'YES' if eliminated else 'NO'}** — sat% = {sat_str}. "
          f"ASR cost vs M0: {cost_str}.")
        a("")

    # ── verdict ──
    a("## Verdict")
    a("")
    verdict = build_verdict(results, method_order, sizes)
    for sent in verdict:
        a(sent)
        a("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report written → {path}")


def build_verdict(results, method_order, sizes):
    """3-sentence verdict: recommended method + why + caveats."""
    def mean_asr(mid):
        return float(np.mean([results[mid][s]["asr"][0] for s in sizes]))

    def max_sat(mid):
        return max(results[mid][s]["sat_pct"][0] for s in sizes)

    candidates = [m for m in method_order
                  if m != "M0" and max_sat(m) < 0.1]
    m0_asr = mean_asr("M0")
    if candidates:
        rec = max(candidates, key=mean_asr)
        best_sz = max(sizes, key=lambda s: results[rec][s]["asr"][0])
        s1 = (f"**{rec} is recommended**: it eliminates trigger saturation "
              f"(max {max_sat(rec):.4f}% of pixels out of range, threshold "
              f"0.1%) and keeps standalone ASR at {mean_asr(rec):.1f}% "
              f"({mean_asr(rec) - m0_asr:+.1f} pts vs M0's {m0_asr:.1f}%).")
    else:
        rec = None
        best_sz = None
        s1 = ("**No method fully eliminates saturation** at the 0.1% "
              "threshold; pick the one with the lowest saturation below.")
    failing = [m for m in method_order if m != "M0" and max_sat(m) >= 0.1]
    m0_sat_str = " / ".join(
        "{:.3f}".format(results["M0"][s]["sat_pct"][0]) for s in sizes)
    if failing:
        tail2 = (", ".join(failing) + " still exceed the threshold ("
                 + " / ".join(f"{max_sat(m):.4f}%" for m in failing) + " max).")
    else:
        tail2 = ("every candidate fix brings the poisoned image back into "
                 "valid pixel space.")
    s2 = (f"M0 leaves the poisoned image physically impossible (sat% = "
          f"{m0_sat_str} for TRIG_SZ {list(sizes)}), and {tail2}")
    s3 = (f"All numbers are seeded with `set_seed(42)` per run (std reflects "
          f"only residual CUDA nondeterminism), and "
          + (f"for {rec} the best trigger size is TRIG_SZ={best_sz} "
             f"(ASR {results[rec][best_sz]['asr'][0]:.1f}%) — note M1's "
             f"corner placement confounds its ASR vs M0's center placement."
             if rec else
             "M1's corner placement confounds its ASR vs M0's center "
             "placement."))
    return [s1, s2, s3]


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--runs", type=int, default=5,
                    help="repetitions per (method, size); default 5")
    ap.add_argument("--sizes", default=",".join(str(s) for s in SIZES),
                    help="comma-separated trigger sizes")
    ap.add_argument("--methods", default=",".join(METHODS),
                    help="comma-separated method ids (M0-M4)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override Phase-1 epochs for ALL methods (testing only)")
    ap.add_argument("--report", default="trigger_size_test_report.md")
    ap.add_argument("--json", default="trigger_size_test_results.json")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    method_order = [m.strip() for m in args.methods.split(",")]
    for mid in method_order:
        if mid not in METHODS:
            raise SystemExit(f"unknown method {mid!r}; choose from {list(METHODS)}")
    runs = args.runs

    cfg = mnp.CFG()
    dev = cfg.DEVICE
    t_start = time.time()

    print("=" * 60)
    print("TRIGGER-SATURATION DIAGNOSTIC (Phase 0 + Phase 1 only)")
    print("=" * 60)
    print(f"  methods : {method_order}")
    print(f"  sizes   : {sizes}")
    print(f"  runs    : {runs}  (set_seed({SEED}) per run)")

    # ── one-time setup (model, data, text embeddings) ──
    mnp.set_seed(SEED)
    print("\n[setup] Loading CLIP")
    model, preprocess = clip.load(cfg.MODEL, device=dev)
    model = model.float()
    mparams = mnp.detect_model_params(model)
    cfg.PATCH_SIZE = mparams["patch_size"]
    cfg.IMG_SIZE = mparams["img_size"]

    print("[setup] Loading Caltech-101")
    tf = transforms.Compose([lambda im: im.convert("RGB"), preprocess])
    ds = datasets.Caltech101(root=cfg.DATA, download=cfg.DOWNLOAD,
                             transform=tf, target_type="category")
    classes = list(ds.categories)

    print("[setup] Text embeddings")
    text_emb = mnp.build_text_embeddings(model, classes, dev)

    print("[setup] Data splits")
    target_sel, calib_idx, grad_idx, eval_idx, test_idx, target_holdout = \
        mnp.prepare_data_splits(ds, cfg)
    calib_loader = mnp.make_loader(ds, calib_idx, cfg.BATCH, shuffle=True,
                                   workers=cfg.WORKERS)
    eval_imgs, eval_labels = mnp.load_tensors(ds, eval_idx, dev)

    pristine = {k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()}
    print(f"[setup] Pristine fp32 state saved "
          f"({sum(v.numel() * v.element_size() for v in pristine.values()) / 1e6:.0f} MB)")

    # ── sweep ──
    results = {mid: {} for mid in method_order}
    raw = {mid: {} for mid in method_order}
    n_total = len(method_order) * len(sizes) * runs
    n_done = 0
    for mid in method_order:
        method = METHODS[mid]
        if args.epochs is not None:
            method = dict(method, epochs=args.epochs)
        for sz in sizes:
            run_dicts = []
            for r in range(runs):
                n_done += 1
                print(f"\n[{n_done}/{n_total}] {mid}  TRIG_SZ={sz}  "
                      f"run {r + 1}/{runs}  "
                      f"(epochs={method['epochs']}, lr={method['lr']}, "
                      f"type={method['trigger_type']})")
                t0 = time.time()
                met = run_once(model, text_emb, calib_loader,
                               eval_imgs, eval_labels,
                               pristine, cfg, method, sz, r)
                met["wall_time_sec"] = time.time() - t0
                print(f"    -> RMS={met['rms']:.3f}  "
                      f"trig_px=[{met['trig_px_min']:.3f}, {met['trig_px_max']:.3f}]  "
                      f"sat%={met['sat_pct']:.4f}  "
                      f"ASR={met['asr']:.1f}  CACC={met['cacc']:.1f}  "
                      f"loss={met['final_loss']:.4f}  "
                      f"({met['wall_time_sec']:.0f}s)")
                run_dicts.append(met)
            results[mid][sz] = aggregate(run_dicts)
            raw[mid][str(sz)] = run_dicts

    # ── outputs ──
    env_info = dict(
        date=time.strftime("%Y-%m-%d %H:%M"),
        model=cfg.MODEL,
        device=dev,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        torch=torch.__version__,
        target_class=classes[cfg.TARGET_CLS],
        target_cls=cfg.TARGET_CLS,
    )
    elapsed = time.time() - t_start
    write_report(args.report, results, method_order, sizes, runs,
                 env_info, elapsed)

    payload = dict(
        env=env_info,
        sizes=sizes,
        runs=runs,
        seed_per_run=SEED,
        methods={mid: {k: v for k, v in METHODS[mid].items()}
                 for mid in method_order},
        per_run=raw,
        aggregates={mid: {str(sz): results[mid][sz] for sz in sizes}
                    for mid in method_order},
        elapsed_sec=elapsed,
    )
    with open(args.json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Raw numbers written → {args.json}")
    print(f"  Done in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
