# idea_p0.md — P0 Method Spec: CLIP-FLIP with CNL + TDA + GMAS

**Date:** 2026-11-20
**Status:** design document (no code changed). Follow-up to `FINDING.Md` §7 P0 row:
**A1 (+A2), B1, A6**.
**Scope:** step-by-step specification of the new method — every phase with its
**purpose, inputs, outputs, and mathematical formulation** — showing exactly where the three
new components plug into the existing `modular_new_pipeline` phases.

---

## 0. What is new, in one paragraph

The pipeline keeps its Phase 0 → 1 → 2A → 2B → 3 → 4 → 5 skeleton untouched. Three components
change, all inside Phase 1 and its data/target preparation:

| Component | Name | Idea | Where it lives |
|---|---|---|---|
| **CNL+m** | *Competitor-Normalised Loss with margin* | Phase-1 loss becomes a softmax CE over **{target} ∪ M data-derived pseudo-competitors** (k-means centroids of clean calibration-pool embeddings), with the `margin_ce` additive logit margin — CE's mechanism **without a class list** | `TriggerGenerator.generate` (A1+A2) |
| **TDA** | *Target-Distribution Alignment* | Phase 1 aligns to a **per-sample sampled target description** `t_{j(i)}` from the 13-template set (BadCLIP++ Eq. 10) instead of the single ensemble-mean vector | `TriggerGenerator.generate` (+ `TargetEmbedder` exposes the set) (B1) |
| **GMAS** | *Greedy Mean-Aligned Selection* | The 64 calibration images are **greedily selected** so their mean embedding is closest to the target direction (BadCLIP++ GMA, Tab. 19), instead of being the first 64 of a random shuffle | `DataModule.prepare_data_splits` (A6) |

Everything downstream (Phases 2A/2B/3/4/5) is **bit-for-bit unchanged** and keeps using
`t̄ = text_emb[TARGET_CLS]` — so there is no Phase-4 `target_vec` consistency hazard and reported
ASR stays comparable with all existing runs.

---

## 1. Threat model, notation, and standing assumptions

**Threat model (unchanged).** Attacker has white-box access to the quantised CLIP ViT visual
encoder (weights + architecture), knows the target class name `c*`, and can flip bits of the int8
weight codes in memory. The attacker does **not** know the deployment label space (class list):
every optimisation term below is **class-list-free**; class names appear only in the final
evaluation, as in the current pipeline.

### Notation

| Symbol | Meaning |
|---|---|
| `f_v(·; Ŵ)` | frozen ViT image encoder with **Phase-0 dequantised** weights `Ŵ` |
| `f_t(·)` | frozen text encoder |
| `e(x)` | unit image embedding: `f_v(x; Ŵ) / ‖f_v(x; Ŵ)‖₂` |
| `τ_k(·)`, `k = 1..K` | CLIP prompt templates (K = 13) |
| `t_k` | `f_t(τ_k(c*))/‖·‖` — per-template **target description embedding** |
| `t̄` | `normalize(Σ_k f_t(τ_k(c*)))` — ensemble mean; **≡ `text_emb[c*]`** |
| `text_emb ∈ R^{C×d}` | class text matrix (evaluation only) |
| `μ_1..μ_M` | normalised k-means centroids of clean pool embeddings — **pseudo-competitors** |
| `δ ∈ R^{3×s×s}` | additive trigger patch (`s` = trigger size) |
| `A(x; δ)` | trigger application (replace / blend / add + placement + pixel clamp) |
| `T` | softmax temperature (0.07) |
| `m` | additive target-logit margin (logit units; `m = 1.0` ↔ 0.07 cosine units) |

### Data splits (current `data.py`, extended for GMAS)

All **non-target** images are shuffled once with `SEED`. Current: `calib(64) | grad(16) |
eval(200) | test(rest)`. New:

```
other_indices (shuffled, non-target)
 ├─ P   = pool[0 : N_POOL]        N_POOL = 512   → GMAS selects S_calib (64) from P
 │        P \ S_calib = D_comp                    → competitor mining ONLY (disjoint!)
 ├─ grad_idx   = next N_GRAD = 16
 ├─ eval_idx   = next N_EVAL = 200
 └─ test_idx   = remainder
target_indices (shuffled) → target_sel (32) | target_holdout (rest)   [unchanged]
```

Disjointness is structural: the 64 trigger-loss images (`S_calib`) and the ~448 competitor-mining
images (`D_comp`) never overlap, so the loss cannot co-adapt to its own negatives.

---

## 2. Method overview — phases and their I/O contracts

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │ Phase 0   Quantisation            (model)         → quant_dict, Ŵ    │
 │ Phase 0.5 Target & pool prep  ★   (dataset,c*)     → {t_k}, t̄,       │
 │                                      S_calib, D_comp, {μ_r}          │
 │ Phase 1   Trigger generation  ★   (Ŵ, {t_k}, {μ_r}, S_calib) → δ*,   │
 │                                      asr_pre, cacc_pre               │
 │ Phase 2A  Layer ablation          (Ŵ, δ*, t̄, calib)   → layer_R, L*  │
 │ Phase 2B  Component ablation      (Ŵ, δ*, t̄, L*)     → components    │
 │ Phase 3   Hybrid bit ranking      (Ŵ, δ*, t̄, grad)   → ranked flips  │
 │ Phase 4   Greedy bit-flip search  (Ŵ, δ*, ranked, eval)→ flip_log    │
 │ Phase 5   Evaluation              (Ŵ_attacked, δ*, test) → ASR/CACC  │
 └──────────────────────────────────────────────────────────────────────┘
        ★ = changed by P0   (all other phases unchanged)
```

**Global I/O of the method**

- **Inputs:** quantised CLIP model `(f_v, f_t)`, dataset `D`, target class name `c*`,
  hyperparameters (§6).
- **Outputs:** attacked quantised model `Ŵ*` (int8 codes with flipped bits), trigger `δ*`,
  `flip_log`, `bfa_results.json` (metrics: ASR, CACC, ASR@10/@20, Flips→95, cos diagnostics).

---

## 3. Phase-by-phase specification

### Phase 0 — INT8 Quantisation *(unchanged)*

**Purpose.** Discretise the visual encoder so "bit flip" is well defined; all later forward passes
use the dequantised grid so the attack operates on the true storage representation.

**Input.** Model `f_v` (fp32 weights `{W}` over all quantisable tensors of every block:
attention weights+biases, MLP weights+biases, LayerNorms).
**Output.** `quant_dict = {key → (w, s_W, shape, block, attr)}` + model in-place dequantised.

**Math.** Per tensor `W`:

```
s_W = max|W| / 127                       (per-tensor scale)
w   = clip(round(W / s_W), −128, 127)    (int8 codes)
Ŵ   = s_W · w                            (forward pass uses Ŵ)
```

A candidate bit flip on code entry `w_p` at bit `b ∈ {0..7}` (two's-complement XOR in uint8):

```
w'_p = wrap_int8( (w_p ⊕ (1<<b)) mod 256 )
Δw   = (w'_p − w_p) · s_W                (weight-space delta on the fixed grid)
```

**Invariants (binding):** one grid per tensor; slices are **never** re-quantised locally; Phase 3/4
always use `quant_dict`/`sub_quant` codes and scales.

---

### Phase 0.5 — Target & calibration-pool preparation ★ (TDA + GMAS + competitor mining)

**Purpose.** Build the three new ingredients before trigger optimisation. Deterministic given
`SEED`, model, `c*`, and the pool.

#### 0.5a — Target distribution (TDA, B1)

**Input.** `f_t`, class name `c*`, template set `{τ_k}`.
**Output.** Template set `{t_k}_{k=1..K}` (K = 13) **kept as a set**, plus `t̄ = normalize(Σ_k f_t(τ_k(c*)))`.

Design decision: **`t̄` is still the single target vector for Phases 2A–4 and for evaluation**
(`t̄ ≡ text_emb[c*]`). TDA lives only inside the Phase-1 loss (per-sample description sampling).
This keeps `search.py`'s internal `target_vec = text_emb[tc]` consistent and keeps all reported
metrics on the exact same decision rule as before.

#### 0.5b — Greedy Mean-Aligned Selection (GMAS, A6)

**Input.** Pool embeddings `{e(x_p) : x_p ∈ P}`, `|P| = N_POOL = 512`, budget `K_c = 64`, target `t̄`.
**Output.** `S_calib ⊂ P` (64 indices); `D_comp = P \ S_calib`.

**Objective (BadCLIP++ Eq. 6–7, adapted from text pairs to images):**

```
S* = argmin_{S ⊂ P, |S| = K_c}   ‖ ν(S) − t̄ ‖₂ ,     ν(S) = normalize( (1/|S|) Σ_{x∈S} e(x) )
```

(equivalently maximise `cos(ν(S), t̄)` on the unit sphere).

**Greedy procedure (near-optimal surrogate):**

```
S₀ = ∅
for s = 1 .. K_c:
    x⁽ˢ⁾ = argmin_{x ∈ P \ S_{s−1}}  ‖ normalize( (Σ_{y∈S_{s−1}} e(y) + e(x)) / s ) − t̄ ‖₂
    S_s = S_{s−1} ∪ {x⁽ˢ⁾}
```

O(K_c · N_POOL · d) — trivially cheap on precomputed embeddings.

**Baseline arms for the ablation table** (BadCLIP++ Tab. 19 parity): `Random` (current pipeline),
`Rank` (top-K_c by `cos(e(x), t̄)`), `Annealing` (annealed search on the same objective),
`Greedy` (above).

#### 0.5c — Pseudo-competitor mining (CNL, A1)

**Input.** `D_comp` embeddings, `M` (default 16).
**Output.** normalised centroids `{μ_r}_{r=1..M}`.

```
{c_r} = k-means({e(x) : x ∈ D_comp}, M)        # k-means++ init, Lloyd, 50 iters
μ_r   = c_r / ‖c_r‖₂ ,  r = 1..M
```

**No class names, no labels anywhere.** Competitors are pure data geometry of clean non-target
images. Ablate `M ∈ {8, 16, 32}` and two variants: centroids (default) vs. a memory bank of raw
`e(x)` from `D_comp`.

---

### Phase 1 — Trigger generation ★ (CNL+m + TDA)

**Purpose.** Learn an additive trigger `δ` such that poisoned embeddings land at the target
semantics with a **positive decision margin against normalised competitors** — the mechanism the
loss study identified as decisive ("the softmax denominator is the mechanism").

**Input.** `Ŵ` (frozen), `{t_k}` + `t̄`, `{μ_r}`, `S_calib` loader, `ApplyParams` (size/mode/opacity/
location/norm), Phase-1 hyperparameters.
**Output.** `δ*` (best-loss trigger tensor), `asr_pre`, `cacc_pre` (standalone), diagnostics log.

#### Poisoned image

For additive mode (replace/blend analogous):

```
x̂ = A(x; δ) = clamp_pixels( x + Π_loc(δ) )
```

`Π_loc` places the patch (placement `loc`), `clamp_pixels` projects back to the valid normalised
pixel range.

#### Loss — CNL+m (the core formula)

Embeddings `z_i = e(x̂_i)` for a minibatch `B ⊂ S_calib`. Per sample draw a target description
index `j(i) ~ Uniform{1..K}` (TDA; an ablation mode marginalises over all K instead of sampling).

Logits (target vs. M pseudo-competitors):

```
ℓ_i⁺  = ⟨z_i, t_{j(i)}⟩ / T  +  m                (target logit, additive margin)
ℓ_ir  = ⟨z_i, μ_r⟩ / T ,      r = 1..M           (competitor logits)
```

Per-sample loss = CE over the `{target} ∪ {pseudo-classes}` label space:

```
L_i(δ) = − ℓ_i⁺ + log( exp(ℓ_i⁺) + Σ_{r=1..M} exp(ℓ_ir) )
```

Batch loss:

```
CNL+m(δ) = (1/|B|) Σ_{i∈B} L_i(δ)
```

**Reading the formula (for the paper):**

- `m = 0, M = 0` degenerates to the current `−cos(emb, t_target)/T` loss.
- `M → C` with the true class texts and `m = 0` recovers the threat-model-relaxed CE reference arm.
- The softmax normalisation (the study's "mechanism") is exactly the `log(Σ exp)` term.
- The margin `m` (from `margin_ce`, the only arm that ever beat CE) shifts the solved point outward
  so late epochs keep buying decision margin instead of idling at the boundary.
- TDA (per-sample `t_{j(i)}`) prevents the trigger from overfitting a single averaged direction;
  it is BadCLIP++'s `L_I→T` sampling mechanism (their Eq. 10).

**Optional diagnostics (logged, not optimised):**

```
diag-margin_pseudo  = mean_i [ ⟨z_i, t̄⟩ − max_r ⟨z_i, μ_r⟩ ]        (class-list-free)
diag-margin_classes = mean_i [ ⟨z_i, t̄⟩ − max_{j≠c*} ⟨z_i, text_emb_j⟩ ]   (eval only)
```

#### Optimisation recipe *(unchanged from the current tuned regime)*

```
δ ← nn.Parameter( randn(3,s,s) × 0.1 )
Adam(lr = 0.02), CosineAnnealingLR → 0.01·lr over 200 epochs
per step:  CNL+m.backward(); grad-clip P1_GRAD_CLIP; step; δ.clamp_(−3, 3)
keep the best-epoch-loss δ*   (no TV term — total_variation stays unwired)
```

Gradient-scale parity check (per the loss-study protocol): verify `‖∂L/∂z‖` of CNL+m at init
against the old `−cos/T` arm (~5.0) before comparing runs.

#### Dead-trigger retry *(unchanged, `pipeline._generate_trigger`)*

Standalone `asr_pre` on the eval split < `TRIGGER_MIN_ASR` → re-seed (`SEED + 1000·attempt`) and
retrain, up to `TRIGGER_RETRIES`.

---

### Phase 2A — Layer-level causal ablation *(unchanged)*

**Purpose.** Rank layers by how much the trigger→target pathway depends on them.

**Input.** `Ŵ`, `δ*`, `t̄`, `S_calib` images (as `calib_imgs`), cfg.
**Output.** `layer_R = {ℓ → s_layer(ℓ)}` and top-k layer set `L*`.

**Math.** For each block `ℓ`, knock out its sublayers (attention out-projection and/or MLP output
replaced by the calibration-mean constant), re-embed the **poisoned** calibration images, and score:

```
s_layer(ℓ) = cos_base − cos_knockout(ℓ) ,
cos(·) = mean_i ⟨e_knockout(x̂_i), t̄⟩
```

`L*` = top-k layers by `s_layer`. (Phase-2 skipping is the `--variant a` ablation.)

---

### Phase 2B — Component-level causal ablation *(unchanged)*

**Purpose.** Within `L*`, select the specific components (per-head attention, per-block MLP) the
attack will target.

**Input.** `Ŵ`, `δ*`, `t̄`, `calib_imgs`, `L*`.
**Output.** `selected_components` (heads/MLPs with scores).

**Math.** Same knockout-drop scoring at component granularity, per-layer quota + top-k
(`ComponentAblator`), producing the weight-tracking scope for Phase 3/4.

---

### Weight tracking & sub-quantisation *(unchanged)*

**Input.** `selected_components`.
**Output.** `tensor_refs` {(param, row_slice, col_slice)}, `original_weights` (saved to
`original_weights.pt`), `sub_quant` (per-slice **copies of the Phase-0 grid** — codes + scales).

---

### Phase 3 — Hybrid bit ranking *(unchanged)*

**Purpose.** Rank single-bit flips by how much they move poisoned embeddings toward `t̄` *without*
moving clean embeddings (poison−clean specificity), weighted by the Phase-2A layer scores.

**Input.** `Ŵ`, `δ*`, `t̄`, `grad_idx` images, `tensor_refs`, `layer_R`, `sub_quant`.
**Output.** `ranked_flips` — candidate list `(layer, head, tensor, row, col, bit, scores)`.

**Math.** With `θ_S` = the tracked slice elements:

```
g_po  = ∇_{θ_S}  [ −mean_i ⟨e(A(x_i; δ*)), t̄⟩ ]      (poisoned grad images)
g_cl  = ∇_{θ_S}  [ −mean_i ⟨e(x_i), t̄⟩ ]              (clean grad images)
g_spec = g_po − g_cl                                   (variant 'c': g_spec = g_po)
```

Per element `w_p` with `g_spec[p] < 0`, per bit `b ∈ {0..7}` on the **Phase-0 grid**:

```
Δw   = (w'_p − w_p) · s_W          (int8 flip delta, never re-quantised)
dL̂   = g_spec[p] · Δw              (predicted loss change)
keep iff dL̂ < 0
score = ŝ_layer(ℓ) · |dL̂| ,  ŝ_layer(ℓ) = s_layer(ℓ) / max_j s_layer(j)
```

Single best bit per element kept (Step-3 semantics); sorted descending → `ranked_flips`.

---

### Phase 4 — Greedy bit-flip search *(unchanged)*

**Purpose.** Actually flip bits on the int8 codes, greedily, until the attack criterion is met.

**Input.** `Ŵ` + `sub_quant`, `δ*`, `ranked_flips`, `eval_idx` images/labels, `text_emb`, cfg.
**Output.** `flip_log`, `checkpoints`, `metrics` (ASR@10, ASR@20, Flips→95, rollbacks), attacked
codes.

**Math (acceptance & stop policy).** At each step take the next ranked (still-untried) candidate,
apply the code flip, and evaluate on the eval split:

```
ASR = (1/|E|) Σ_{(x,y)∈E} 1[ argmax_j ⟨e(A(x;δ*)), text_emb_j⟩ = c* ]
CACC = (1/|E|) Σ 1[ argmax_j ⟨e(x), text_emb_j⟩ = y ]
```

Two-phase policy:

1. **Normal phase** — accept flips until `ASR ≥ NORMAL_STOP = 94` (budget `MAX_FLIPS = 80`).
2. **Fine phase** — mine the failure set one-by-one toward `FINE_TARGET = 99`; each accepted flip
   must gain `≥ FLIP_MIN_GAIN = 0.5 pp` (else reverted), `≤ FINE_MAX_FLIPS = 9`, stalling after
   `FINE_MAX_REJECTS = 20` rejections; **best-prefix rollback** (`flips_rolled_back`) and
   checkpoint carry-forward recorded.

Clean-embedding drift guard: on `drift_n = 16` held-out clean images, drift of `e(x)` is penalised
(`CLEAN_DRIFT_PENALTY`, capped by `MAX_CLEAN_DRIFT`) using `target_vec = t̄ = text_emb[c*]`
(identical by construction — no consistency hazard).

---

### Phase 5 — Evaluation *(unchanged)*

**Input.** attacked `Ŵ*`, `δ*`, `test_idx` loader, `text_emb`.
**Output.** final metrics + artefacts.

```
ASR  = P_{x∼test}[ argmax_j ⟨e(A(x;δ*)), text_emb_j⟩ = c* ]        (2 000+-image test split)
CACC = P[ argmax_j ⟨e(x), text_emb_j⟩ = y ]
cos_poison = mean ⟨e(A(x;δ*)), t̄⟩ ,  cos_clean = mean ⟨e(x), t̄⟩
```

Optional `--task lp`: shared linear probe on the attacked encoder's features (LP-CACC / LP-ASR).
Decision rule is **always** argmax over `text_emb` (or the probe) — never `t̄` alone — so P0
components cannot inflate reported ASR.

---

## 4. End-to-end algorithm (pseudocode)

```
Input: quantised CLIP (f_v with codes w, scales s), f_t, dataset D, target class c*, cfg
Output: flip_log, δ*, metrics

# ── Phase 0 ──────────────────────────────────────────────────────────
quant_dict ← INT8-quantise every visual tensor; forward = Ŵ = s·w

# ── Phase 0.5 (NEW) ─────────────────────────────────────────────────
{t_k} ← { normalize(f_t(τ_k(c*))) }_{k=1..13}          # TDA
t̄    ← normalize( Σ_k f_t(τ_k(c*)) )                    # ≡ text_emb[c*]
P     ← first N_POOL = 512 shuffled non-target images
S_calib ← GMAS(P, t̄, 64)                                # greedy mean-aligned selection
D_comp  ← P \ S_calib
{μ_r} ← normalize( k-means( {e(x): x ∈ D_comp}, M=16 ) )  # pseudo-competitors

# ── Phase 1 (NEW LOSS) ──────────────────────────────────────────────
δ ← randn × 0.1
for epoch = 1..200:
    for batch B in S_calib:
        for i in B: j(i) ~ U{1..13};  z_i = e(A(x_i; δ))
        L = mean_i [ −(⟨z_i,t_{j(i)}⟩/T + m) + log( exp(⟨z_i,t_{j(i)}⟩/T + m)
                                                   + Σ_r exp(⟨z_i,μ_r⟩/T) ) ]
        Adam step on L; grad clip; δ.clamp(−3,3)
    keep best-loss δ*
if standalone_ASR(δ*) < TRIGGER_MIN_ASR: re-seed and retry (≤ TRIGGER_RETRIES)

# ── Phases 2–5 (UNCHANGED) ──────────────────────────────────────────
2A: s_layer(ℓ) = drop in mean ⟨e(x̂), t̄⟩ under layer knockout → L* (top-k)
2B: component knockout scores → selected_components
track: tensor_refs, original_weights, sub_quant (Phase-0 grids)
3:   rank flips by  ŝ_layer · |(g_po − g_cl)·Δw|  on the int8 grid
4:   greedy flip search (normal → fine policy, drift guard, rollback)
5:   final ASR/CACC on the test split (argmax over text_emb); save artefacts
```

---

## 5. What changes in the code (module-swap map)

| File | Change | Size |
|---|---|---|
| `data.py::prepare_data_splits` | reserve pool `P` (512) instead of taking `calib_idx` directly; call the selector; return `S_calib` indices (+ expose `D_comp`) | small |
| `target.py::TargetEmbedder` | `from_text` unchanged (returns `t̄`); add `from_text_set(...)` returning `{t_k}` for Phase 1 | small |
| `trigger.py::TriggerGenerator.generate` | add 0.5c mining + replace the loss with CNL+m and the per-sample `t_{j(i)}` draw; keep recipe/clamp/retry hooks | medium (one method) |
| everything else | **untouched** | — |

Respects the repo's modular-swap rule: one phase method per component; Phases 2–5 see no
difference (they still receive `t̄` and the same `trigger`).

---

## 6. Hyperparameters

| Symbol | Value | Notes |
|---|---|---|
| `K` | 13 | prompt templates (existing) |
| `T` | 0.07 | softmax temperature (matches the study's gradient-scale convention) |
| `m` | sweep {0, 0.5, 1.0, 2.0} | logit units; `1.0` ↔ 0.07 cosine (margin_ce convention) |
| `M` | sweep {8, 16, 32}, default 16 | pseudo-competitor count |
| `N_POOL` | 512 | GMAS candidate pool |
| `K_c` | 64 | calibration budget (N_CALIB, already the tuned value) |
| Phase-1 recipe | init 0.1, Adam 0.02, cosine, 200 ep, grad clip, clamp ±3 | **unchanged** tuned regime |
| selection metric | `‖ν(S) − t̄‖₂` on normalised mean | equals cosine distance on the sphere |

---

## 7. Experiment & ablation plan

All arms: ViT-B/16 × Caltech-101 first (study-parity), then the grid (ViT-L/14 × ImageNet1k etc.),
**≥ 3 seeds**, decide on the 2 000-image test column (noise floor ±3–4 pts on 200).

| Arm | TDA (B1) | CNL (A1) | +margin (A2) | GMAS (A6) | Purpose |
|---|---|---|---|---|---|
| A | – | – | – | – | current baseline (cos, random calib, mean target) |
| B | ✓ | – | – | – | TDA alone |
| C | – | ✓ | – | – | CNL without margin |
| D | – | ✓ | ✓ | – | CNL+m |
| E | – | – | – | ✓ | GMAS alone |
| F | ✓ | ✓ | ✓ | ✓ | **full P0** |
| ref | CE (threat-model-relaxed upper bound, read from existing runs — never re-executed) | | | | |

**Metrics per arm:** standalone ASR (test column), `diag-margin_pseudo` / `diag-margin_classes`,
`cos_poison`, **flips→95 %**, final ASR/CACC, search time. Selection strategy sub-table
(Random / Rank / Annealing / Greedy) for the GMAS arm — BadCLIP++ Tab. 19 parity.

**Success gate (suggested):** full P0 ≥ baseline on final ASR **and** fewer flips→95, with
`diag-margin_classes > 0` at the end of Phase 1 (the boundary the `cos` arms never crossed).

---

## 8. Why this is defensible (paper claims)

1. **Class-list-free by construction.** Competitors are k-means centroids of unlabelled clean
   images; target descriptions use only the attacker-chosen class name `c*` (already assumed).
   The threat-model-relaxed CE arm is reported only as an upper-bound reference.
2. **Every term traces to a measured mechanism.** Softmax normalisation (loss-study finding 1),
   additive margin (finding 5 — the only arm beating CE), per-sample target descriptions
   (BadCLIP++ Eq. 10), greedy target-aligned selection (BadCLIP++ Tab. 19).
3. **No falsified idea is re-proposed.** No single-argmax hinge (Arm C), no free-direction T2T,
   no cosine-threshold MPE, no image-prototype mixing into `t_target`, no `−‖∇cos‖²` sensitivity.
4. **Comparability.** Evaluation, split sizes, Phase-1 recipe, and every downstream phase are
   unchanged; only the trigger objective and the calibration/target construction differ.
5. **Mechanistic prediction to verify:** CNL+m should end Phase 1 with `diag-margin_classes > 0`
   (positive class-text margin), which no `cos`-family arm ever achieved — a crisp, falsifiable
   hypothesis for the paper.

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Centroids of 448 pool images are noisy competitors | sweep `M`; use more pool (`N_POOL` 1024) for mining only; memory-bank variant |
| GMAS selects images too close to target semantics (degrades clean-side specificity / CACC readings) | report CACC + `clean-as-target` rate per arm; annealing/rank variants |
| Per-sample `t_{j(i)}` sampling adds gradient noise | `marginalise` mode (average loss over all 13 templates) as ablation; 200 epochs at batch 8 already averages 1 600 draws |
| Reviewers read a public vocabulary as "a class list" | default competitors are **data-derived only**; keep any public-vocabulary variant explicitly labelled threat-model-relaxed |
| Margin sweep multiplies runs | screen `m` at `MAX_FLIPS=0` (Phase-1 screen protocol from the loss study), then full pipeline only for the winner |
| Gradient-scale mismatch between old and new loss | run the study's `‖∂L/∂z‖` parity check at init before comparing |

---

## 10. Naming (for the paper)

Suggested umbrella: **"Competitor-normalised, target-distribution-aligned trigger synthesis with
greedy calibration selection"** — components **CNL+m**, **TDA**, **GMAS**. Ablation table rows can
mirror BadCLIP++ Tab. 9's incremental style (C7 in `FINDING.Md`) for reviewer familiarity.
