# Phase-1 trigger-loss study — Caltech-101 / ViT-B/16

Record of every trigger-generation objective tried for CLIP-FLIP, with results.
All Phase-1 arms share the same data, trigger geometry, optimiser and seed; only
the loss changes. Downstream phases (0 → 2A → 2B → 3 → 4 → 5) are untouched, so
flip counts stay comparable across arms.

Implementation: `phase1_objective()` dispatches on `CFG.P1_LOSS` in
`main_new_pipeline.py` (baseline default `"ce"`) and in the Caltech test copy
`main_new_pipeline_trigger_test.py` (default `"composite"`). Sweep harness:
`trigger_loss_sweep.py`.

---

## 1. Setup

| | |
|---|---|
| Model | ViT-B/16 (width 768, 12 layers, 12 heads, patch 16, 224 px) |
| Dataset | Caltech-101, 102 classes, `TARGET_CLS = 3` → *Motorbikes* |
| Trigger | `center_add`, 14×14 px, additive, static clamp ±3 |
| Phase-1 data | 32 calibration images (`BATCH=8` → 4 steps/epoch) |
| Optimiser | Adam, lr 0.02, cosine schedule → lr·0.01, 200 epochs |
| Init | `randn × 0.1` (same seed 42 → identical starting trigger per arm) |
| Text target | `text_emb[TARGET_CLS]`: mean of 13 prompt templates per class |
| Metrics | standalone ASR/CACC on 200 held-out non-target images; final ASR/CACC on 2 000 non-target images |
| Diagnostic margin | `cos(emb, t_target) − max_{j≠target} cos(emb, t_j)` (cosine units) |

Notation: `emb = normalize(encode_image(poisoned))`, `t_j` = text embedding of
class *j*, `t_target = t_TARGET`, `T = 0.07`, `C` = number of classes.

---

## 2. Objectives tested

| `P1_LOSS` | Objective | Notes |
|---|---|---|
| `ce` | `CE((emb·t_j)/T, target)` | incumbent baseline, unchanged code path |
| `margin_ce` | `CE((emb·t_j)/T + m·1[target], target)` | additive margin on the target logit; `m=1.0` ↔ 0.07 in cosine units |
| `cos` | `−cos(emb, t_target)/T` | same gradient scale as CE (verified: \|∂L/∂emb\| = 5.04 vs 5.03) |
| `cos_t2t` | `−cos(emb, t_target)/T + λ·Σ_d Var_b(emb_{b,d})` | + trigger-to-trigger compactness |
| `ce_t2t` | `CE + λ·Σ_d Var_b(emb_{b,d})` | compactness on the incumbent |
| `composite` | `w_t·(−cos_t) + w_m·relu(0.05 − margin) + w_2t·ΣVar + w_mpe·relu(0.4 − cos_v) + w_l2·mean(trig²) + w_tv·TV(trig)` | the originally proposed objective (weights 1.0 / 0.5 / 0.05 / 0.10 / 0.01 / 0.001) |

Notes on the sub-terms:

- **T2T compactness** — for unit-norm embeddings the term is algebraically
  `1 − ‖mean_b(emb_b)‖²` (verified numerically: 0.8719 vs 0.8719). It rewards a
  long batch-mean vector, i.e. mutual collapse of the poisoned embeddings; the
  direction of that mean is free.
- **MPE** — `relu(0.4 − cos(emb, v_target_mean))`, `v_target_mean` = L2-normalised
  mean of the 32 target-class image embeddings.
- **Bounded hinge** — `relu(0.05 − margin)`, a single-argmax competitor
  surrogate.

---

## 3. Results

### 3.1 Stage A — Phase-1 screen (`MAX_FLIPS=0`, one fresh process per arm)

Run: `python trigger_loss_sweep.py --stage phase1` (7 arms, ~4.3 min each).
Raw data: `trigger_loss_sweep/sweep_summary.json`.

| arm | test ASR (2 000) | eval ASR (200) | CACC (2 000) | cos_poison | margin | t2t | ‖trig‖ |
|---|---|---|---|---|---|---|---|
| **`ce`** | 36.05 | 36.5 | 88.5 | 0.2780 | **+0.0045** | – | 39.99 |
| **`margin_ce:1.0`** | **40.20** | **37.5** | 88.5 | **0.2794** | **+0.0054** | – | 39.01 |
| `cos` | 9.10 | 9.0 | 88.5 | 0.2488 | −0.0421 | – | 31.55 |
| `cos_t2t:0.05` | 4.25 | 4.5 | 88.5 | 0.2417 | −0.0499 | 0.311 | 31.13 |
| `cos_t2t:0.2` | 6.10 | 8.5 | 88.5 | 0.2478 | −0.0429 | 0.314 | 34.56 |
| `ce_t2t:0.05` | 29.75 | 32.0 | 88.5 | 0.2723 | −0.0028 | 0.277 | 39.10 |
| `ce_t2t:0.2` | 27.10 | 28.5 | 88.5 | 0.2707 | −0.0087 | 0.274 | 38.40 |

CACC is identical in every arm (the model weights are untouched by Phase 1), so
it carries no signal here. `margin` is the final-epoch training diagnostic from
`run.log`.

### 3.2 `composite` arm (earlier runs, same 200 epochs, separate process)

Measured on the same 200-image eval set, so comparable to the `eval ASR` column
above (noise on that column is ±3–4 points; see §4.6).

| variant | eval ASR | margin | cos_t (train) | t2t | ‖trig‖ | MPE |
|---|---|---|---|---|---|---|
| CE baseline (same script) | 41.0 | +0.0061 | 0.2871 | – | 38.86 | – |
| `composite` as proposed | 25.5 | −0.0071 | 0.2828 | 0.288 | 29.83 | 0.0 |
| `composite` with all cosine terms ×1/T | 28.5 | −0.0081 | 0.2826 | 0.284 | 38.62 | 0.0 |
| `composite` ×1/T, T2T dropped | 21.5 | −0.0177 | 0.2728 | 0.302 | 34.80 | 0.0 |

The MPE term was identically zero at every epoch: `cos(emb, v_target_mean)` starts
at 0.59 and rises to 0.63, always above the 0.4 threshold, so `relu(0.4 − cos_v) = 0`.

### 3.3 Full-pipeline (bit-flip) reference

Only the CE arm has a completed flip search. From
`xpruner_bfa_results_ViT-B/16_Caltech101_v4/bfa_results.json`
(`main_new_pipeline_ood_prompts.py`, CE trigger loss):

| | value |
|---|---|
| standalone ASR | 38.5 % |
| checkpoint @10 flips | 51.5 % |
| final (20 flips) | **ASR 93.8 %, CACC 88.7 %** |
| cos_poison after flips | 0.332 |
| search time | 173.8 s (8.7 s/flip) |

These are the numbers any new arm has to beat in Stage B (`--stage full`).

---

## 4. Findings

1. **The softmax denominator is the mechanism, not a detail.** Every arm without
   a normalised competitor term (`cos`, `cos_t2t`, and the bounded-hinge
   `composite`) ends with a *negative* margin (−0.008 … −0.05) and ASR in the
   4–29 % range. The CE arms are the only ones that cross the decision boundary
   (margin > 0). A single-argmax hinge is not a substitute: it tracks a moving
   argmax class and stalls ~0.01 below the boundary.

2. **`cos` alone is not just weaker, it is worse on its own metric.** Test
   `cos_poison` 0.249 vs 0.278 for CE: optimising one direction over 32 calibration
   images overfits them, whereas CE must satisfy 101 competitors and generalises
   better.

3. **T2T hurts in every family where it was added** — CE 36.0 → 29.8 (λ=0.05) →
   27.1 (λ=0.2); cos 9.1 → 4.2. It does achieve its own objective (ΣVar 0.39 →
   0.27–0.31) while pushing the cluster *off* the boundary (margin +0.0045 →
   −0.0028 / −0.0087). Two reasons: (a) the term only lengthens the batch mean,
   it never constrains the *direction*, so it is satisfied equally well by a
   compact cluster pointing at the wrong class; (b) it does not vanish at
   convergence, while CE's gradient does, so it dominates the endgame where the
   final decision margin is decided. A target-centred compactness would be
   `mean_b‖emb_b − t_target‖² = 2 − 2·mean cos_t` for unit vectors — i.e.
   algebraically identical to the cosine term, adding nothing.

4. **The earlier "+7 points from T2T" was a hinge artifact.** Inside `composite`,
   dropping T2T cost 7 ASR points (28.5 → 21.5). Under a working competitor term
   (CE) T2T is a net loss. The 2×2 was designed to expose exactly this confound.

5. **`margin_ce` is the only arm that beats the incumbent**, and it does so on the
   larger test set: 40.2 % vs 36.05 % (+4.2 pts on 2 000 images, ≈2.7σ) with
   identical `cos_poison` and CACC. Mechanism: the additive margin (`m` logits =
   `0.07·m` cosine) moves the point where CE is "solved" outward, so the last
   ~50 epochs keep buying decision-margin instead of idling at the boundary
   (CE plateaus at margin +0.0045). Gain comes from argmax robustness, not from a
   different embedding — which is what a bit-flip search needs.

6. **Noise floor.** The CE control measured 36.5 / 36.05 (Stage A), 38.5 (stored
   full run), 41.0 (earlier smoke) on the 200-image evaluation. Differences below
   ~4 points on 200 images are not signal; use the 2 000-image column for
   decisions.

---

## 5. Status / next

- Settled: `composite` (worse), single-argmax hinge (structurally weak), MPE
  (dead at threshold 0.4), `cos` family (4–9 % ASR), T2T (negative in every arm).
- Live: `margin_ce`, whose margin `m` has only been testedنتیجه: فایل از نظر مفهومی و پیاده‌سازی تمیز و فقط Arm C است. تنها تغییر اختیاری، حذف پارامتر بی‌استفاده‌ی text_emb از phase1_cos_astar است که هیچ تأثیری بر نتایج ندارد.

 at 1.0. Planned:
  screen `m ∈ {0.5, 2.0}`, then Stage B (`--stage full`) on CE + best `margin_ce`
  against the 20-flip / 93.8 % reference.
- Not tested: prompt-distribution CE (needs `build_text_embeddings` to keep the 13
  per-prompt vectors), image-anchor/MPE variants (settled by the earlier lp A/B:
  text-only beat image-anchor 35 vs 55 flips).

---

## 6. Reproduce

```bash
# Phase-1 screen, all arms (MAX_FLIPS=0, ~30 min)
python trigger_loss_sweep.py --stage phase1

# single arm, Phase-1 only
python -c "import main_new_pipeline_trigger_test as M; \
           M.CFG.P1_LOSS='margin_ce'; M.CFG.P1_MARGIN_LOGIT=1.0; \
           M.CFG.MAX_FLIPS=0; M.CFG.OUT_DIR='./run_mce'; M.main()"

# full pipeline (bit-flips), several arms
python trigger_loss_sweep.py --stage full --arms ce,margin_ce:1.0
```

Artefacts per arm: `trigger_loss_sweep/arm_<name>/{bfa_results.json, trigger.pt,
trigger_patch.png, composite.png, run.log}`; aggregate table in
`trigger_loss_sweep/sweep_summary.json`.

Note: the JSON `config` block was empty in all runs before this study
(`vars(cfg)` on a class-attribute config is `{}`); the test pipeline now records
the full config, including the loss arm and its weights.
