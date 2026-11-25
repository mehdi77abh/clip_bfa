# improvement_trigger.md — Phase-1 Trigger-Loss Redesign for Multi-Model Generality

**Date:** 2026-11-25
**Status:** design document (no code changed). Companion to `FINDING.Md` (idea ledger),
`idea_p0.md` (CNL+m/TDA/GMAS spec) and `idea_p0_tests.md` (campaign run log).
**Scope:** a loss + target + competitor-set redesign that survives **every registry model**
(ViT-B/16, ViT-L/14 @224/@336, SigLIP-B/16, SigLIP-so400m/14, BLIP-B/16) instead of being
tuned per checkpoint — motivated by the 2026-11-24 failure: InfoNCE trigger generation died on
ViT-L/14@336 (standalone ASR ≈ 0, `cos_t` capped 0.17, `margin_class` −0.065) while plain `cos`
reaches 40–50 % on the same model.

---

## 0. Why one fixed loss fails across checkpoints

Cross-model data already in hand (standalone trigger ASR):

| checkpoint × target | InfoNCE | cos | negative geometry |
|---|---|---|---|
| ViT-L/14 @224, lion | **52.0** | 45.6 | separable big-cat/canid confusables |
| ViT-L/14 @224, tabby cat | 4.5 | **19.5** | 5 near-collinear cat rows + thief outside the list |
| ViT-B/16, motorbike | 17.5 → **60.5** (v2 + filter) | 63.0 | synonyms collinear; the filter fixed it |
| ViT-L/14 @336, lion | ≈ 0 | **40–50** | same list as the winning row — text geometry drifted |

One mechanism explains every row: **"confusable-but-separable" is a property of the
checkpoint's text space, not of the name list.** Hand-curated lists (`lion_v1` → `v2` → `v3` …)
cannot scale to six encoders. Two documented failure modes must be engineered out:

1. **Self-fighting (collinear negatives):** repulsion toward `n_r ≈ t̄` opposes the positive
   term; InfoNCE concentrates softmax gradient mass exactly on the most collinear negatives →
   `cos_t` caps, dead trigger.
2. **Coverage gap:** `margin_pseudo > 0 ∧ margin_class < 0` — the argmax thief sits outside the
   curated set (tabby root cause #1, `idea_p0_tests.md` §10).

Design rules (from `FINDING.Md` §6, binding): class-list-free claim is load-bearing;
`t_target` consistency across Phases 2A/2B/3/4; evaluation criterion stays argmax over
`text_emb`; one-method swap (`TriggerGenerator.generate` / `TargetEmbedder.from_text`);
tuned recipe kept unless the change *is* the experiment; ledger's "do not re-propose" list
(single-argmax hinge / Arm C, free-direction T2T, cosine-threshold MPE@0.4, raw image-prototype
mixing, `−‖∇cos‖²`).

---

## 1. Orthogonalized residual repulsion (the loss)

### 1.1 Setup and the key identity

Unit image embedding `z = normalize(E(A(x;δ)))`, unit target `t̄`, unit negatives `n_r`.
Define `a_r = ⟨n_r, t̄⟩` and the **residual** `res_r = n_r − a_r·t̄` (the part of the negative
that is *not* the target direction; `b_r = ‖res_r‖ = √(1−a_r²)`). The decision margin against
negative `r` decomposes exactly:

```
⟨z, t̄⟩ − ⟨z, n_r⟩  =  (1 − a_r)·⟨z, t̄⟩  −  ⟨z, res_r⟩
                      └─ grows with cos_t ─┘   └─ orthogonal lever ─┘
```

Penalizing `⟨z, res_r⟩` raises the margin against `n_r` **without touching the `t̄` component
of z** — the two levers act on orthogonal subspaces by construction. Plain cosneg and InfoNCE
cannot guarantee this: with raw `n_r` the repulsion gradient contains `a_r·t̄` (first-order
opposition); InfoNCE's softmax weights concentrate gradient mass on the largest `a_r`
negatives — maximal conflict exactly where it hurts. That is the self-fighting mechanism.

### 1.2 Loss (recommended form)

```
L = (1 − ⟨z, t̄⟩)  +  λ · mean_r ⟨z, res_r⟩²
```

- Penalize the residual **un-normalized**: the gradient `2⟨z,res_r⟩·res_r` is auto-scaled by
  `b_r²`, so a near-collinear negative (`b_r ≈ 0`) contributes ≈ nothing — **the separability
  filter becomes a soft, self-enforcing property of the loss**, not a per-checkpoint τ.
- Hinge variant `[⟨z,res_r⟩ − m]_+`, `m = 0`, is an ablation row (goes silent per-negative once
  satisfied); squared is the default (no `m`, smooth Adam behaviour).
- Subspace variant: thin-QR the residual matrix → basis `U`, `L_neg = ‖Uᵀz‖²` (or
  `zᵀ U diag(w) Uᵀ z`). Use when the list has tiers (near-duplicate rows — e.g. the fox tier —
  would be counted 4× by `mean_r`). Run both as an ablation.
- **Gradient statement:** at `z ≈ t̄` the positive gradient is radial (`t̄`), the negative
  gradient lies in the tangent plane at `t̄` (`res_r ⊥ t̄`) — first-order orthogonal, so
  `cos(z,t̄)` degrades only quadratically in the angular displacement the negative term causes.
- **Scale/parity:** old losses were scaled by `1/T ≈ 14`, `1−⟨z,t̄⟩` has gradient norm ~1.
  Adam is scale-forgiving but λ is the trade-off: log the `[parity] ‖∂L/∂emb‖` check per arm,
  sweep `λ ∈ {0.15, 0.25, 0.4, 0.7}`.

### 1.3 Degenerate cases (the desired properties)

- **All negatives collinear** (`b_r ≈ 0 ∀r`): `L_neg ≡ 0` → loss ≡ plain `cos`. The robust
  fallback emerges automatically (graceful degradation — the answer to "not only for 336").
- **Coverage failure** (thief outside the set): *not* fixed by this loss — that is §2's job.
  Two failure modes, two mechanisms (keep this explicit in the paper).

### 1.4 Why this is NOT the forbidden Arm-C hinge (`FINDING.Md` §6.4)

Arm C = `cos + λ·relu(μ − (cos_t − max_j cos(e,c_j)))` over k-means **image** centroids:
single-`max` hinge, hard margin μ, feedback on `cos_t` itself, image-cone competitors.
Residual repulsion: no `max`, no hard μ, never touches the `t̄` component, text-cone
competitors (the eval geometry). Different in the exact dimensions the failure analysis names.

### 1.5 Falsifiable predictions

- Hostile cell (ViT-L/14@336): `L_neg ≈ 0` for the drifted list ⇒ trigger ASR ≈ cos's 40–50 %,
  never ≈ 0.
- Friendly cell (ViT-L/14@224 lion): margin pressure intact ⇒ ≥ InfoNCE's 52.0.
If either fails, the mechanism story is wrong.

---

## 2. Separability procedure + LLM-expanded negatives

### 2.1 Phase-0.5 machinery (adapter-side → every registry model gets it)

1. **Geometry profile.** Embed competitor names with the per-template norm → mean → renorm
   convention (`target.py:104-109`); report `a_r` sorted, `b_r` distribution, effective rank of
   the residual subspace. Persist to `bfa_results.json["neg_geometry"]` → `aggregate.py` column.
   This is the **predictor variable** of the cross-model study (§4).
2. **Hard filter.** Drop `a_r > τ` (default 0.9). Because §1's `b_r²` weighting makes the exact
   τ low-stakes, one constant survives across encoders. Raise if the set empties
   (`_filter_separable` in `p0_test.py` is the reference implementation).
3. **Soft deweighting** — free (it *is* the `b_r²` factor).
4. **LLM-expanded negatives** — replaces hand lists: `c*` → "list 25 concepts
   visually/semantically confusable with X, ranked" → names → embed → filter → set. Provenance
   (model, prompt, date, raw + filtered lists) recorded in the run config. Robustness = k
   independent LLM samples (generalizes the §4.3 swap test into a *procedure* robustness number).
5. **Coverage check with a hard threat-model boundary.** Post-Phase-1, compute the
   argmax-thief diagnostic over the **LLM concept vocabulary** (attacker-side) and expand if a
   thief is missing. The 1000-row `[diag]` (`p0_test.py:591`) stays a *post-hoc analysis tool
   only* — if the attack loop reads the deployment rows, the class-list-free claim dies.
   Keep the two code paths physically separate.

### 2.2 Threat-model text (draft for the paper)

> The attacker knows the target concept *c\** and may query a public language model for
> concepts confusable with *c\** — concept-neighbourhood knowledge ("big cats", "canids" for
> *lion*) that concerns the target alone. The competitor set for trigger synthesis is built
> exclusively from this attacker-side knowledge and from the attacker's own copy of the text
> encoders; no victim data, no deployment label space, and no class list enters any
> optimisation term. The set is filtered and residualised against the target direction in the
> victim model's text space (the separability procedure), which guarantees the trigger
> objective can never oppose itself. The deployment class list appears only in evaluation.

Contrast to draw on: image-centroid mining carries the "soft class-list side channel"
(`idea_p0_tests.md` §3); the LLM concept list has none, and sampled-list robustness replaces
the cherry-picking objection.

---

## 3. Target side (refining `t_target` — what is actually new)

**Algebra that kills the naive version.** For unit `t_k`:
`mean_k ⟨z,t_k⟩ = ⟨z, mean_k t_k⟩` — "one cosine to the center of the 13 templates" ≡
"average cosine over templates" ≡ today's `t_target` (`target.py:69-70` is already
`normalize(Σ t_k)`). Improvements must come from **weighting, mode structure, or selection**:

- **(a) Norm weighting.** `build_text_embeddings` averages *raw* template embeddings
  (high-norm templates dominate) while negatives use mean-of-normalized. Measure the
  per-encoder norm spread; if wide (likely on SigLIP/BLIP), mean-of-normalized is a real change.
- **(b) Multi-positive / nearest-template pull** (the genuinely new formulation):
  `L⁺ = −log Σ_k exp(⟨z,t_k⟩/T)` — pulls toward the nearest template *mode* instead of the
  barycenter (which is not a natural text direction — `FINDING.Md` B1). Justified when the
  template set is multi-modal (measure pairwise `cos(t_k,t_j)` spread). Note: plain per-sample
  template *sampling* (TDA) under linear cosine is equivalent in expectation to the mean and
  only adds SGD variance — the logsumexp is the substantive variant. TDA's ledger status:
  only ever killed bundled with margin (R3) — isolated retest is legal.
- **(c) Template-sensitivity probe — run BEFORE spending anything on (d)/(e).** Phase-1 with
  `t = t_k` for 3 representative templates vs the mean, one cell. Wide ASR spread across `k` →
  the mean leaves performance on the table; flat → drop the whole target-side family.
- **(d) B3 image-guided reweighting** (ledger-approved "new justification" path):
  `w_k ∝ cos(t_k, v̄*)`, `v̄*` = mean of the 32 target images already in hand. Images *select
  among text directions*; nothing image-derived enters the direction. Gate on the measured
  modality gap per encoder (CLIP ≈ 0.31; SigLIP/BLIP differ — flat `cos(t_k, v̄*)` = no signal).
- **(e) B2 learnable `α`** (`t* = normalize(Σ α_k t_k)`) — only if (c) shows headroom; it is
  the bilevel outer loop (A8) and the expensive option.

**Consistency rule (`FINDING.Md` §6.2) decides where refinement may live.** ASR is argmax over
`text_emb` rows, i.e. over `t̄ = text_emb[c*]`. **All target-side refinement stays inside the
Phase-1 loss (as TDA was designed in `idea_p0.md` §0.5a); `t̄` remains the phase-wide vector
for Phases 2A/2B/3/4.** B2's full form needs propagation to `_make_target_cos`, `rank`,
`rank_fallback`, `search.target_vec` — defer unless (c) justifies it.

**Cross-encoder angle.** The 13 CLIP templates are a CLIP-family convention (SigLIP wants
short prompts, BLIP caption-style). Per-family prompt sets in `config.json` fold into (a)/(c)
and are part of `t_target` quality on non-CLIP encoders.

---

## 4. Cross-model validation design

**Matrix.** Models: ViT-B/16 · ViT-L/14@224 · **ViT-L/14@336 (hostile cell)** · SigLIP-B/16 ·
SigLIP-so400m/14 · BLIP-B/16. Targets: **lion** (friendly) · **tabby cat** (crowded
neighbourhood) · optionally motorbike × Caltech for lineage. Arms: `cos` (control) ·
`infonce` + filter (procedure baseline) · **residual repulsion** (proposed) · ± multi-positive
target. Seeds 42/43/44. Trigger policy fixed per `run_all.sh` (not a variable).
Phase-1-only cost control via `--stop-after-phase1`.

**Staged execution (cost-aware):**

| stage | content | cost |
|---|---|---|
| 0 — probes | geometry profiles per model×target; template-sensitivity probe (3c); `[diag]` coverage check | no trigger training |
| 1 — loss A/B | 2 cells (1 hostile, 1 friendly), 3 seeds, `λ` sweep | Phase-1-only runs |
| 2 — full matrix | winner + cos control across all cells | full runs |
| 3 — target-side | only if (3c) said headroom | second wave |

**Gates (pre-registered):**

1. **No-regression:** proposed ≥ cos (within the ±3–4 pt noise floor) in *every* cell — the
   graceful-degradation claim, tested.
2. **Hostile-cell recovery:** proposed ≫ InfoNCE on @336 (prediction: ≈ cos's 40–50 %).
3. **Knife-edge kill:** seed spread of proposed < InfoNCE's spread.
4. **Mechanism:** `margin_class > 0` rate across cells (the boundary only R1 ever crossed).
5. **Predictor claim (the thesis sentence):** the Stage-0 separability profile *predicts* the
   per-cell InfoNCE win/loss; residual repulsion removes the dependency.

**Metric discipline (`FINDING.Md` §6.5, non-negotiable):** 2 000-image column for headline
numbers (±3–4 pts on the 200-img eval is noise), multi-seed for any < 5 pt claim, 1-flip
differences not signal, `[parity]` gradient check per new loss arm, tuned Phase-1 recipe kept
unless the change *is* the experiment.

---

## 5. Suggested package (one line)

> **Residual-repulsion trigger loss (§1) + separability procedure with LLM-expanded negatives
> (§2)** as the new Phase-1 default, target-side second wave (§3) gated by the
> template-sensitivity probe, validated by the §4 matrix whose hostile cell guarantees the fix
> is not another per-checkpoint tune.
