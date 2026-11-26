#!/usr/bin/env python3
"""
test_version2.py — Phase-1 idea test harness (successor of p0_test.py).

Tests the improvement_trigger.md / badclip_dual_ideas.md redesigns on ONE
model-dataset cell at a time, inside the untouched `modular_new_pipeline`
(same wrapper pattern as p0_test.py: swap DataModule / TargetEmbedder /
TriggerGenerator at runtime, run the standard Phases 0-5).

IDEAS UNDER TEST (one flag each — everything else is held identical):

  1. Orthogonalized residual repulsion   --loss residual
       L = (1 - cos(z, t̄)) + λ·mean_r ⟨z, res_r⟩²,  res_r = n_r - cos(t̄,n_r)·t̄
       (improvement_trigger.md §1: gradient-orthogonal to the positive term;
        collinear negatives auto-scaled to zero via ‖res_r‖².)
  2. Template sensitivity + multi-positive target
       --target-mode single --target-template k   (pull toward template t_k)
       --target-mode multi                        (pull toward nearest of {t_k}:
       L⁺ = -log Σ_k exp(⟨z,t_k⟩/T) + log K)       (improvement_trigger.md §3)
  3. Visual-resistance term (BadCLIP D2)  --vis-lambda λv
       L += λv · max(0, L_pi + λ2·L_ni + η),
       L_pi = 1 - cos(z, v̄*)   (v̄* = mean of the 32 target-class images),
       L_ni = cos(z, z_clean) - 1  (z_clean = the carrier's unaltered embedding)
       (badclip_dual_ideas.md D2: hinge-gated; silent once satisfied;
        purpose = FT-defence survival, NOT the attack direction.)

RECOMMENDED FACTORS (run them as crossed factors, not assumptions):
  --carrier-rule pps     boundary:farthest:random = 1:1:1 carrier selection
                         (BadCLIP D1 — attacks the carrier-draw knife-edge)
  --select margin        keep the best-margin_class trigger instead of
                         best-loss (R1: loss ≠ decision quality)
  --loss avgsoftmax      2-logit softmax vs the AVERAGED negative (middle
                         point between InfoNCE and residual repulsion)

ALWAYS-ON INSTRUMENTATION (every arm prints these):
  negative-geometry table (cos(t̄, n_r), filter marks), template-spread stats,
  [diag] poisoned-argmax thieves + target share (post-hoc analysis ONLY),
  [parity] gradient norms of the loss terms, per-term loss log,
  margin_pseudo / margin_class, split fingerprint (cross-box comparability).

CONTROLS (the harness is broken if these miss their references):
  --loss cos      ViT-L/14 × ImageNet1k lion: ≈ 45.6 % (p0 a0 reference)
  --loss infonce  same cell: ≈ 52.0 % (p0 R1 reference); ViT-L/14-336px:
                  ≈ 0 % (the hostile cell)
Cross-container numerics ±10 pts on the same box/source (idea_p0_tests.md
2026-11-23) — compare on ONE box, 3 seeds for anything that matters.

USAGE (from anywhere; experiment.py flags pass through):
  # primary cell — ViT-L/14 × ImageNet1k, target 'lion' (dataset default)
  python related_papers/test_version2.py --loss cos
  python related_papers/test_version2.py --loss infonce --neg-list lion_v1
  python related_papers/test_version2.py --loss residual --lambda-neg 0.25
  python related_papers/test_version2.py --loss residual --lambda-neg 0.4 \
      --neg-list lion_v2
  python related_papers/test_version2.py --loss residual --vis-lambda 1.0
  python related_papers/test_version2.py --loss cos --vis-lambda 1.0
  python related_papers/test_version2.py --loss residual --target-mode multi
  python related_papers/test_version2.py --loss cos --target-mode single \
      --target-template 0        # repeat for k = 0..12 (template sensitivity)
  python related_papers/test_version2.py --loss residual --carrier-rule pps
  python related_papers/test_version2.py --loss infonce --select margin
  # hostile cell
  python related_papers/test_version2.py --model ViT-L/14-336px \
      --loss residual --trigger-size 24

  Add --stop-after-phase1 to stop right after Phase 1 (the standard
  Stage-1 cost control) — writes phase1_summary.json and exits 0.

SUGGESTED MATRIX (Stage 1, one cell, seeds 42/43/44, all --stop-after-phase1):
  cos | infonce | residual λ∈{0.15,0.25,0.4} | residual+vis | cos+vis |
  residual+multi | single-k sweep (3 representative k) | residual+pps | infonce+select-margin

NOTES.
  * Adam is (near-)invariant to uniform loss scaling, so L⁺ = 1-⟨z,t̄⟩ and the
    historical -⟨z,t̄⟩/T give identical trajectories; only the RELATIVE weight
    λ between terms is meaningful.  [parity] prints the term gradient norms.
  * Split layout is IDENTICAL to p0_test.py (incl. the 100-image neg holdout
    and the content-hash split key) so results compare to the stored
    references; the split fingerprint must match across boxes/sources.
  * The 1000-class [diag] is analysis only — never feed it to the attack
    loop (class-list-free claim, FINDING.Md §6.1).
  * Results land under related_papers/v2_results/... — never grid_results.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))   # .../clip_bfa/related_papers
V2_OUT_ROOT = os.path.join(HERE, "v2_results")

# Reuse the p0_test.py machinery (same directory): curated lists, content
# order, separability filter, embed helpers, patched DataModule base and the
# diagnostics.  p0_test.py has no import-time side effects beyond sys.path.
sys.path.insert(0, HERE)
import p0_test as p0ref                                  # noqa: E402

REPO_ROOT = p0ref.REPO_ROOT
MODULAR_DIR = p0ref.MODULAR_DIR
sys.path.insert(0, MODULAR_DIR)

import experiment                                       # noqa: E402
import main as main_mod                                  # noqa: E402
import pipeline as pipeline_mod                          # noqa: E402
from adapters import encode_image                        # noqa: E402
from data import DataModule                              # noqa: E402
from target import PROMPT_TEMPLATES, TargetEmbedder      # noqa: E402
from trigger import apply_trigger, get_apply_params      # noqa: E402
from utils import cleanup_gpu, model_tag                 # noqa: E402

NEG_LISTS = p0ref.NEG_LISTS
DATASET_DEFAULTS = p0ref.DATASET_DEFAULTS
TRIG_SZ_DEFAULTS = p0ref.TRIG_SZ_DEFAULTS
resolve_target_cls = p0ref.resolve_target_cls

DEFAULT_MODEL = "ViT-L/14"
DEFAULT_DATASET = "ImageNet1k"

V2_LOSSES = ("cos", "infonce", "residual", "avgsoftmax")
V2_TARGET_MODES = ("mean", "multi", "single")
V2_CARRIER_RULES = ("random", "pps")

# last-epoch diagnostics, stashed for phase1_summary.json
V2_LAST_STATS = {}


# ══════════════════════════════════════════════════════════════════════
#  DATA MODULE — p0_test split layout + carrier-rule factor
# ══════════════════════════════════════════════════════════════════════
class V2DataModule(p0ref.P0DataModule):
    """P0DataModule with a parameterised carrier rule.

    random : scratch[:N_CALIB]                     (p0 status quo)
    pps    : boundary : farthest : random = 1:1:1  (BadCLIP D1) chosen from
             the scratch pool by cos(e(x), t̄) on the PRE-quantisation fp32
             forward (same documented caveat as p0_test's GMAS).
    """

    def prepare_data_splits(self, ds):
        cfg = self.cfg
        rng = np.random.RandomState(cfg.SEED)
        labels = self.get_all_labels(ds)

        target_indices = [i for i, lbl in enumerate(labels)
                          if lbl == cfg.TARGET_CLS]
        other_indices = [i for i, lbl in enumerate(labels)
                         if lbl != cfg.TARGET_CLS]

        # ── canonical sample order (split-key): content md5 → the same seed
        #    selects the same PHYSICAL images on HF and official ILSVRC trees
        #    (compare_imagenet_sources.py, 2026-11-23). ──
        if getattr(cfg, "SPLIT_KEY", "content") == "content":
            p0ref.STATE.rank, ok = p0ref._content_order(ds, cfg.DATA)
            print(f"  Split key: content-hash order "
                  f"({'source-invariant' if ok else 'fallback: name order'})")
        else:
            p0ref.STATE.rank = np.arange(len(labels))
            print("  Split key: name order (legacy)")
        target_indices = sorted(target_indices,
                                key=lambda i: p0ref.STATE.rank[i])
        other_indices = sorted(other_indices,
                               key=lambda i: p0ref.STATE.rank[i])

        # ── neg-list image holdout (identical to p0_test: zero leakage,
        #    identical grad/eval/test across arms) ──
        p0ref.STATE.neg_idx = p0ref._sample_neg_indices(cfg, labels)
        neg_set = set(p0ref.STATE.neg_idx)
        other_indices = [i for i in other_indices if i not in neg_set]

        rng.shuffle(target_indices)
        rng.shuffle(other_indices)

        target_sel = target_indices[:cfg.N_TARGET_IMAGES]
        target_holdout = target_indices[cfg.N_TARGET_IMAGES:]

        # ── shared layout: grad | eval | scratch(max(N_POOL, N_CALIB)) | test
        ptr = 0
        grad_idx = other_indices[ptr:ptr + cfg.N_GRAD]
        ptr += cfg.N_GRAD
        eval_idx = other_indices[ptr:ptr + cfg.N_EVAL]
        ptr += cfg.N_EVAL
        scratch_n = max(getattr(cfg, "N_POOL", 0), cfg.N_CALIB)
        scratch = other_indices[ptr:ptr + scratch_n]
        ptr += scratch_n
        test_idx = other_indices[ptr:]

        if cfg.V2_CARRIER_RULE == "pps":
            calib_idx = self._pps_select(scratch, cfg)
        else:
            calib_idx = scratch[:cfg.N_CALIB]
        p0ref.STATE.calib_idx = calib_idx
        p0ref.STATE.dcomp_idx = []
        p0ref.STATE.pool_idx = []

        print(f"  Target class images : {len(target_sel):>5d}")
        print(f"  Target holdout      : {len(target_holdout):>5d}")
        print(f"  Calibration         : {len(calib_idx):>5d}"
              f"  ({cfg.V2_CARRIER_RULE})")
        print(f"  Negatives (val)     : {len(p0ref.STATE.neg_idx):>5d}  "
              f"(held out of all splits)")
        print(f"  Gradient            : {len(grad_idx):>5d}")
        print(f"  Evaluation          : {len(eval_idx):>5d}")
        print(f"  Test                : {len(test_idx):>5d}")
        if p0ref.STATE.md5s is not None:
            sel = list(calib_idx) + list(eval_idx) + sorted(p0ref.STATE.neg_idx)
            fp = hashlib.md5(
                "".join(p0ref.STATE.md5s[i] for i in sel).encode()
            ).hexdigest()[:16]
            print(f"  Split fingerprint   : {fp}   "
                  f"(compare across boxes/sources — must match)")
        else:
            print("  Split fingerprint   : n/a (name order)")
        return (target_sel, calib_idx, grad_idx, eval_idx, test_idx,
                target_holdout)

    def _pps_select(self, scratch, cfg):
        """BadCLIP D1: boundary : farthest : random = 1:1:1 by cos(e(x), t̄).

        boundary = non-target images closest to the target direction (most
        likely to flip), farthest = most dissimilar (hardest), random = the
        middle remainder.  Class-list-free: needs only t̄.
        """
        model = p0ref.STATE.model
        dev = cfg.DEVICE
        imgs, _ = self.load_tensors(p0ref.STATE.val_ds, scratch, dev,
                                    batch_size=64, workers=cfg.WORKERS)
        E = p0ref._embed_images(model, imgs, cfg.MINI_BS)
        del imgs
        cleanup_gpu()
        t_bar = F.normalize(
            p0ref.STATE.text_emb[cfg.TARGET_CLS].to(dev), dim=0)
        scores = (E @ t_bar).cpu().numpy()

        n = cfg.N_CALIB
        n_b = (n + 2) // 3          # boundary
        n_f = (n + 1) // 3          # farthest
        n_r = n - n_b - n_f         # random (middle remainder)
        order = list(np.argsort(-scores))        # descending
        boundary = order[:n_b]
        farthest = order[-n_f:]
        mid = order[n_b:len(order) - n_f]
        rng = np.random.RandomState(cfg.SEED + 11)
        rng.shuffle(mid)
        picked = boundary + farthest + mid[:n_r]
        picked = sorted(picked)
        print(f"  PPS carriers        : boundary {n_b} "
              f"(cos {scores[boundary].min():+.3f}..{scores[boundary].max():+.3f})"
              f"  farthest {n_f} "
              f"(cos {scores[farthest].min():+.3f}..{scores[farthest].max():+.3f})"
              f"  random {n_r}")
        return [scratch[i] for i in picked]


# ══════════════════════════════════════════════════════════════════════
#  TARGET EMBEDDER — p0_test's capture (model/classes/text_emb to STATE)
# ══════════════════════════════════════════════════════════════════════
class V2TargetEmbedder(p0ref.P0TargetEmbedder):
    pass


# ══════════════════════════════════════════════════════════════════════
#  TRIGGER GENERATOR — the v2 loss family (recipe unchanged)
# ══════════════════════════════════════════════════════════════════════
class V2TriggerGenerator(p0ref.P0TriggerGenerator):
    """Phase 1 with the v2 objectives.

    Recipe is EXACTLY the tuned regime (init randn×0.1, Adam cfg.P1_LR,
    CosineAnnealingLR → lr·0.01, cfg.P1_EPOCHS, grad clip, clamp [-3, 3],
    best-trigger kept) — only the loss differs between arms (FINDING.Md
    §6.5: the tuned recipe is kept unless the change *is* the experiment).
    """

    def generate(self, model, text_emb, t_target, calib_loader,
                 target_images, cfg, neg_text_emb=None):
        trig = self._generate_v2(model, text_emb, calib_loader,
                                 target_images, cfg)
        self._diag_top_classes(model, trig, calib_loader, text_emb, cfg)
        return trig

    # ── target direction set {t_k} + t̄ ─────────────────────────────
    def _build_targets(self, model, text_emb, cfg):
        dev = cfg.DEVICE
        cname = p0ref.STATE.classes[cfg.TARGET_CLS].replace("_", " ").lower()
        prompts = [t.format(cname) for t in PROMPT_TEMPLATES]
        t_set = F.normalize(p0ref._embed_texts(model, prompts, dev), dim=-1)
        t_bar = F.normalize(text_emb[cfg.TARGET_CLS].to(dev), dim=0)
        K = t_set.size(0)
        # template-spread statistic (informs the target-mode reading)
        S = t_set @ t_set.T
        iu = torch.triu_indices(K, K, offset=1)
        spread = S[iu[0], iu[1]]
        note = ("  (wide: barycenter may be a poor direction)"
                if spread.min() < 0.7 else "")
        print(f"  Target: '{p0ref.STATE.classes[cfg.TARGET_CLS]}' — "
              f"{K} template directions")
        print(f"  Template spread cos(t_j,t_k): "
              f"min {spread.min():+.3f}  mean {spread.mean():+.3f}  "
              f"max {spread.max():+.3f}{note}")
        return t_set, t_bar

    # ── residual matrix (improvement_trigger.md §1) ─────────────────
    @staticmethod
    def _build_residuals(comps, t_bar):
        res = comps - (comps @ t_bar).unsqueeze(1) * t_bar.unsqueeze(0)
        b = res.norm(dim=1)
        keep = b > 1e-6
        res, b = res[keep], b[keep]
        print(f"  Residuals           : {res.size(0)} active of "
              f"{comps.size(0)}  ‖res‖: min {b.min():.3f}  "
              f"mean {b.mean():.3f}  max {b.max():.3f}")
        return res

    # ── gradient parity of the loss terms (study protocol) ──────────
    def _v2_parity(self, emb0, t_bar, res, lam, cfg):
        z = emb0.detach().clone().requires_grad_(True)
        lp = 1.0 - (z @ t_bar).mean()
        lp.backward()
        g_pos = z.grad.norm(dim=-1).mean().item()
        if res is not None and res.numel():
            z2 = emb0.detach().clone().requires_grad_(True)
            ln = (z2 @ res.T).pow(2).mean()
            (lam * ln).backward()
            g_neg = z2.grad.norm(dim=-1).mean().item()
            print(f"  [parity] ‖∂L/∂emb‖  L_pos = {g_pos:.3f}   "
                  f"λ·L_neg = {g_neg:.3f}   ratio {g_neg / max(g_pos, 1e-9):.2f}")
        else:
            print(f"  [parity] ‖∂L/∂emb‖  L_pos = {g_pos:.3f}")

    # ── Phase-1 optimisation (recipe unchanged) ─────────────────────
    def _generate_v2(self, model, text_emb, calib_loader, target_images, cfg):
        ap = get_apply_params(cfg, model)
        dev = cfg.DEVICE
        T = cfg.TEMPERATURE
        lam = cfg.V2_LAMBDA_NEG

        print("\n" + "=" * 60)
        print(f"PHASE 1: TRIGGER GENERATION — v2  (loss {cfg.V2_LOSS})")
        print("=" * 60)
        print(f"  Size: {cfg.TRIG_SZ}×{cfg.TRIG_SZ}  "
              f"Placement: {ap.loc or 'legacy'}  Mode: {ap.mode}")
        print(f"  Loss: {cfg.V2_LOSS}  target-mode: {cfg.V2_TARGET_MODE}"
              + (f" (k={cfg.V2_TARGET_TEMPLATE})"
                 if cfg.V2_TARGET_MODE == "single" else "")
              + f"  λ_neg: {lam:g}  T: {T}")
        print(f"  Visual-resistance: "
              + (f"λv={cfg.V2_VIS_LAMBDA:g} λ2={cfg.V2_VIS_L2:g} "
                 f"η={cfg.V2_VIS_ETA:g}"
                 if cfg.V2_VIS_LAMBDA > 0 else "off"))
        print(f"  Epochs: {cfg.P1_EPOCHS}  LR: {cfg.P1_LR}  "
              f"grad clip: {cfg.P1_GRAD_CLIP}  Trig clamp: [-3, 3]")
        print(f"  Init: randn × 0.1  select: best-{cfg.V2_SELECT}")

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        t_set, t_bar = self._build_targets(model, text_emb, cfg)
        K = t_set.size(0)
        if cfg.V2_TARGET_MODE == "single":
            if not (0 <= cfg.V2_TARGET_TEMPLATE < K):
                raise SystemExit(f"--target-template must be in [0, {K})")
            t_pos = t_set[cfg.V2_TARGET_TEMPLATE]
            print(f"  Single template [{cfg.V2_TARGET_TEMPLATE}]: "
                  f"{PROMPT_TEMPLATES[cfg.V2_TARGET_TEMPLATE]!r}")
        else:
            t_pos = t_bar

        # competitors: text negatives (curated list + separability filter)
        cfg.COMPETITOR_MODE = "text"
        comps = self._build_competitors(model, text_emb, cfg)   # (M, d)
        res = self._build_residuals(comps, t_bar) \
            if cfg.V2_LOSS == "residual" else None

        # visual-resistance ingredients (badclip_dual_ideas.md D2)
        v_star = z_clean = None
        if cfg.V2_VIS_LAMBDA > 0:
            with torch.no_grad():
                E_t = p0ref._embed_images(model, target_images.to(dev),
                                          cfg.MINI_BS)
                v_star = F.normalize(E_t.mean(dim=0), dim=0)
                del E_t
            print(f"  Visual resistance   : v̄* from "
                  f"{tuple(target_images.shape)[0]} target images")

        trig = nn.Parameter(torch.randn(1, 3, cfg.TRIG_SZ, cfg.TRIG_SZ,
                                        device=dev) * 0.1)
        opt = torch.optim.Adam([trig], lr=cfg.P1_LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.P1_EPOCHS, eta_min=cfg.P1_LR * 0.01)

        best_val, best_trig = float("inf"), trig.data.clone()
        text_all = text_emb.to(dev)
        tc = cfg.TARGET_CLS
        parity_done = False

        for ep in range(1, cfg.P1_EPOCHS + 1):
            ep_loss = ep_lp = ep_ln = ep_lv = 0.0
            ep_cos = ep_mp = ep_mc = 0.0
            nb = 0
            for imgs, _ in calib_loader:
                imgs = imgs.to(dev)
                poisoned = apply_trigger(imgs, trig, ap)
                emb = F.normalize(encode_image(model, poisoned), dim=-1)
                B = emb.size(0)

                # ── positive term(s) ──
                if cfg.V2_TARGET_MODE == "multi":
                    # nearest-template pull: -log mean_k exp(⟨z,t_k⟩/T)
                    L_pos = -(torch.logsumexp(emb @ t_set.T / T, dim=1)
                              - float(np.log(K))).mean()
                    cos_t = emb @ t_bar
                else:
                    s_p = emb @ t_pos
                    L_pos = 1.0 - s_p.mean()
                    cos_t = emb @ t_bar     # diagnostics always vs t̄

                # ── negative term(s) ──
                L_neg = torch.zeros((), device=dev)
                if cfg.V2_LOSS == "infonce":
                    logits = torch.cat(
                        [(emb @ t_pos).unsqueeze(1) / T,
                         emb @ comps.T / T], dim=1)
                    loss = F.cross_entropy(
                        logits, torch.zeros(B, dtype=torch.long, device=dev))
                    L_log = loss
                elif cfg.V2_LOSS == "avgsoftmax":
                    logits = torch.cat(
                        [(emb @ t_pos).unsqueeze(1) / T,
                         (emb @ comps.T).mean(dim=1, keepdim=True) / T],
                        dim=1)
                    loss = F.cross_entropy(
                        logits, torch.zeros(B, dtype=torch.long, device=dev))
                    L_log = loss
                elif cfg.V2_LOSS == "residual":
                    L_neg = (emb @ res.T).pow(2).mean()
                    L_log = L_pos + lam * L_neg
                    loss = L_log
                else:                       # cos
                    L_log = L_pos
                    loss = L_log

                # ── visual-resistance hinge (D2) ──
                if cfg.V2_VIS_LAMBDA > 0:
                    with torch.no_grad():
                        z_clean = F.normalize(
                            encode_image(model, imgs), dim=-1)
                    L_pi = 1.0 - (emb @ v_star).mean()
                    L_ni = (emb * z_clean).sum(dim=1).mean() - 1.0
                    L_vis = torch.clamp(
                        L_pi + cfg.V2_VIS_L2 * L_ni + cfg.V2_VIS_ETA,
                        min=0.0)
                    loss = loss + cfg.V2_VIS_LAMBDA * L_vis
                else:
                    L_vis = torch.zeros((), device=dev)

                opt.zero_grad()
                loss.backward()
                if cfg.P1_GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_([trig],
                                                    max_norm=cfg.P1_GRAD_CLIP)
                opt.step()
                with torch.no_grad():
                    trig.data.clamp_(-3.0, 3.0)

                # ── diagnostics (not optimised) ──
                with torch.no_grad():
                    sims = emb @ text_all.T
                    sims[:, tc] = -float("inf")
                    mp = cos_t - (emb @ comps.T).max(dim=1).values
                    mc = cos_t - sims.max(dim=1).values
                ep_loss += loss.item()
                ep_lp += L_pos.item() if torch.is_tensor(L_pos) else L_pos
                ep_ln += L_neg.item()
                ep_lv += L_vis.item()
                ep_cos += cos_t.mean().item()
                ep_mp += mp.mean().item()
                ep_mc += mc.mean().item()
                nb += 1

                if not parity_done:
                    self._v2_parity(emb.detach(), t_bar, res, lam, cfg)
                    parity_done = True

            sched.step()
            avg_l = ep_loss / max(nb, 1)
            avg_mc = ep_mc / max(nb, 1)
            sel_val = avg_mc if cfg.V2_SELECT == "margin" else avg_l
            better = sel_val > best_val if cfg.V2_SELECT == "margin" \
                else sel_val < best_val
            if better:
                best_val, best_trig = sel_val, trig.data.clone()
            if ep % 50 == 0 or ep == 1 or ep == cfg.P1_EPOCHS:
                print(f"  ep {ep:>4d}  loss={avg_l:.6f}  "
                      f"(L⁺={ep_lp / max(nb, 1):.4f}  "
                      f"λL⁻={ep_ln / max(nb, 1):.6f}  "
                      f"Lᵛ={ep_lv / max(nb, 1):.4f})  "
                      f"cos_t={ep_cos / max(nb, 1):+.4f}  "
                      f"margin_pseudo={ep_mp / max(nb, 1):+.4f}  "
                      f"margin_class={avg_mc:+.4f}  "
                      f"||trig||={trig.data.norm():.3f}")

        V2_LAST_STATS.clear()
        V2_LAST_STATS.update({
            "loss": cfg.V2_LOSS,
            "target_mode": cfg.V2_TARGET_MODE,
            "target_template": cfg.V2_TARGET_TEMPLATE,
            "lambda_neg": cfg.V2_LAMBDA_NEG,
            "vis_lambda": cfg.V2_VIS_LAMBDA,
            "carrier_rule": cfg.V2_CARRIER_RULE,
            "select": cfg.V2_SELECT,
            "neg_list": cfg.NEG_LIST,
            "best_select_value": float(best_val),
            "final_cos_t": ep_cos / max(nb, 1),
            "final_margin_pseudo": ep_mp / max(nb, 1),
            "final_margin_class": ep_mc / max(nb, 1),
        })
        print(f"  Best ({cfg.V2_SELECT}): {best_val:.6f}")
        return best_trig.detach()


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════
def parse_v2_args(argv):
    p = argparse.ArgumentParser(
        description="Phase-1 idea harness: residual repulsion / template "
                    "sensitivity / multi-positive / visual resistance.  "
                    "experiment.py flags are passed through.")
    p.add_argument("--loss", default="residual", choices=V2_LOSSES,
                   help="negative-side objective (default: residual)")
    p.add_argument("--target-mode", default="mean", choices=V2_TARGET_MODES,
                   help="positive-side target: mean t̄ | multi (nearest of "
                        "the K templates) | single template k (default: mean)")
    p.add_argument("--target-template", type=int, default=0,
                   help="template index k for --target-mode single (default: 0)")
    p.add_argument("--lambda-neg", type=float, default=0.25,
                   help="λ weight of the residual term (default: 0.25)")
    p.add_argument("--vis-lambda", type=float, default=0.0,
                   help="λv of the visual-resistance hinge, 0 = off "
                        "(BadCLIP D2; default: 0)")
    p.add_argument("--vis-lambda2", type=float, default=1.0,
                   help="λ2 inside the hinge (default: 1.0)")
    p.add_argument("--vis-eta", type=float, default=1.0,
                   help="η margin inside the hinge (default: 1.0)")
    p.add_argument("--carrier-rule", default="random", choices=V2_CARRIER_RULES,
                   help="calibration carriers: random | pps = boundary:"
                        "farthest:random 1:1:1 (BadCLIP D1; default: random)")
    p.add_argument("--select", default="loss", choices=("loss", "margin"),
                   help="keep best-loss or best-margin_class trigger "
                        "(default: loss)")
    p.add_argument("--neg-list", default=None, choices=sorted(NEG_LISTS),
                   help="curated confusable negatives (default: dataset-"
                        "aware — lion_v2 on ImageNet1k, motorbike_v2 on "
                        "Caltech101)")
    p.add_argument("--target-name", default=None,
                   help="target class name (default: dataset-aware — "
                        "'lion' on ImageNet1k, 'motorbike' on Caltech101)")
    p.add_argument("--neg-cos-max", type=float, default=0.9,
                   help="separability filter threshold (default: 0.9)")
    p.add_argument("--split-key", default="content",
                   choices=["content", "name"],
                   help="sample ordering: content md5 (source-invariant, "
                        "default) | name (legacy)")
    p.add_argument("--stop-after-phase1", action="store_true",
                   help="stop right after Phase 1 (Stage-1 cost control); "
                        "writes phase1_summary.json and exits 0")
    p.add_argument("--dry-run", action="store_true",
                   help="assemble and print the config, then exit")
    return p.parse_known_args(argv)


def v2_tag(v2):
    t = f"v2_{v2.loss}"
    if v2.target_mode != "mean":
        t += f"_tgt{v2.target_mode}" + (
            f"{v2.target_template}" if v2.target_mode == "single" else "")
    if v2.vis_lambda > 0:
        t += f"_vis{v2.vis_lambda:g}"
    if v2.carrier_rule != "random":
        t += f"_{v2.carrier_rule}"
    if v2.loss == "residual":
        t += f"_lneg{v2.lambda_neg:g}"
    if v2.select != "loss":
        t += f"_sel{v2.select}"
    return t


def main():
    v2, rest = parse_v2_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    args = experiment.parse_args()

    # defaults mirror p0_test (historical cell) unless overridden
    if "--model" not in rest:
        args.model = DEFAULT_MODEL
    if "--dataset" not in rest:
        args.dataset = DEFAULT_DATASET
    if "--trigger-size" not in rest:
        args.trigger_size = TRIG_SZ_DEFAULTS.get(args.model, 16)
    if "--trigger-min-asr" not in rest:
        args.trigger_min_asr = 0.0     # one deterministic attempt per arm
    if "--trigger-retries" not in rest:
        args.trigger_retries = 0
    d_target, d_neg = DATASET_DEFAULTS[args.dataset]
    target_name = v2.target_name or d_target
    neg_list = v2.neg_list or d_neg
    args.target_cls = resolve_target_cls(args.dataset, target_name)

    if v2.target_mode == "multi" and v2.loss in ("infonce", "avgsoftmax"):
        raise SystemExit("--target-mode multi is only defined for the "
                         "L⁺-form losses (cos, residual)")

    # Version-skew shim (2026-11-26 FaR hook): some experiment.py copies read
    # cfg.FAR_CKPT before assigning it (check-before-assign bug — every
    # build_config call crashed without --far-ckpt).  Pre-seeding the class
    # attribute keeps this harness working with old and new copies alike;
    # a later cfg.FAR_CKPT = args.far_ckpt simply overrides it.
    import config as _config
    if not hasattr(_config.CFG, "FAR_CKPT"):
        _config.CFG.FAR_CKPT = None

    cfg = experiment.build_config(args)

    # ── v2 settings on cfg (consumed by V2DataModule / V2TriggerGenerator)
    cfg.V2_LOSS = v2.loss
    cfg.V2_TARGET_MODE = v2.target_mode
    cfg.V2_TARGET_TEMPLATE = v2.target_template
    cfg.V2_LAMBDA_NEG = v2.lambda_neg
    cfg.V2_VIS_LAMBDA = v2.vis_lambda
    cfg.V2_VIS_L2 = v2.vis_lambda2
    cfg.V2_VIS_ETA = v2.vis_eta
    cfg.V2_CARRIER_RULE = v2.carrier_rule
    cfg.V2_SELECT = v2.select
    # competitor plumbing reused from p0_test's _build_competitors (text mode)
    cfg.P0_ARM = "p2txt"
    cfg.COMPETITOR_MODE = "text"
    cfg.NEG_LIST = cfg.P0_NEG_LIST = neg_list
    cfg.NEG_COS_MAX = v2.neg_cos_max
    cfg.SPLIT_KEY = v2.split_key
    cfg.N_POOL = 256                  # scratch size (p0 layout parity)
    cfg.TDA_MODE = "off"              # unused by v2 (kept for signature parity)
    cfg.P1_MARGIN = 0.0
    cfg.P1_LOSS = "cos"               # keep pipeline's infonce prebuild OFF —
    #                                 # this generator builds its own comps

    # ── distinct run dir per arm, results under v2_results ──
    tag = v2_tag(v2)
    cfg.RUN_ID += f"__{tag}"
    cfg.OUT_DIR = cfg.OUT_DIR + f"_{tag}"
    if os.path.abspath(cfg.OUT_DIR).startswith(os.path.abspath(cfg.OUT_ROOT)):
        rel = os.path.relpath(cfg.OUT_DIR, cfg.OUT_ROOT)
    else:
        rel = os.path.join(cfg.SECTION, model_tag(cfg.MODEL_NAME),
                           cfg.DATASET, os.path.basename(cfg.OUT_DIR))
    cfg.OUT_ROOT = V2_OUT_ROOT
    cfg.OUT_DIR = os.path.join(cfg.OUT_ROOT, rel)

    print("=" * 60)
    print(f"TEST V2 — {tag}")
    print("=" * 60)
    print(f"  model={cfg.MODEL}  dataset={cfg.DATASET}  "
          f"target='{target_name}' (cls {cfg.TARGET_CLS})")
    print(f"  loss={v2.loss}  target-mode={v2.target_mode}  "
          f"λ_neg={v2.lambda_neg:g}  vis={v2.vis_lambda:g}  "
          f"carriers={v2.carrier_rule}  select={v2.select}")
    print(f"  neg-list={neg_list}  split-key={v2.split_key}  "
          f"trigger={args.trigger_size}px  seed={cfg.SEED}")
    print(f"  out: {cfg.OUT_DIR}")

    if v2.dry_run:
        print("\n[dry-run] config assembled OK — exiting before model load.")
        return

    # ── install patches, then run the standard pipeline ──
    pipeline_mod.DataModule = V2DataModule
    pipeline_mod.TargetEmbedder = V2TargetEmbedder
    pipeline_mod.TriggerGenerator = V2TriggerGenerator

    if v2.stop_after_phase1:
        orig = pipeline_mod.AttackPipeline._generate_trigger

        def _wrap(pself, *a, **kw):
            out = orig(pself, *a, **kw)
            trig, asr_pre, cacc_pre, attempts = out
            summary = dict(V2_LAST_STATS)
            summary.update({"run_tag": tag, "model": cfg.MODEL,
                            "dataset": cfg.DATASET,
                            "target_name": target_name,
                            "target_cls": cfg.TARGET_CLS,
                            "trigger_size": cfg.TRIG_SZ,
                            "seed": cfg.SEED,
                            "standalone_asr": float(asr_pre),
                            "standalone_cacc": float(cacc_pre),
                            "attempts": int(attempts)})
            os.makedirs(cfg.OUT_DIR, exist_ok=True)
            with open(os.path.join(cfg.OUT_DIR, "phase1_summary.json"),
                      "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  [stop-after-phase1] standalone ASR={asr_pre:.1f}%  "
                  f"CACC={cacc_pre:.1f}%  → phase1_summary.json")
            raise SystemExit(0)

        pipeline_mod.AttackPipeline._generate_trigger = _wrap

    result = None
    try:
        result = main_mod.main(cfg)      # OOM/error contract of main.py applies
    except SystemExit as e:              # --stop-after-phase1 path
        if e.code not in (0, None):      # real errors propagate
            raise
    try:
        experiment.refresh_summaries(cfg.OUT_ROOT)
    except Exception as e:               # best-effort, never fail the run
        print(f"  (summary refresh skipped: {type(e).__name__}: {e})")
    return result


if __name__ == "__main__":
    main()
