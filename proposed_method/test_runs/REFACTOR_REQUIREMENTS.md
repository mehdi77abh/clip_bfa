# REFACTOR REQUIREMENTS — Align `main_new_pipeline.py` with `pseudo_code.txt` (CLIP-FLIP v3)

**Scope:** `main_new_pipeline.py` only. Ignore `main.py` and `modular/`.
**Reference:** `pseudo_code.txt` (pipeline `0 → 1 → 2 → 2R → 2V → 3 → 4 → 5`)
**Style:** this document states *what* must change and *why*. The implementing agent decides *how* (exact code structure, helper signatures, refactors) as long as the requirements and acceptance criteria below are met.

**Execution order:** Step 1 → verify → Step 2 → verify → Step 3 → verify. Run `python -m py_compile main_new_pipeline.py` after every step; ideally a reduced dry-run too.

---

## Context — where the code diverges from the pseudo-code today

| Area | Pseudo-code v3 | Current code | Problem |
|---|---|---|---|
| Old Phase 3 (gradient pre-screen) | Removed in v3 | Still defined (`phase3_gradient_prescreen`, ~L1153–1261), never called; `PRESCREEN_TOP_K` still in CFG | Dead code; pipeline doc still mentions it |
| Phase 2 scope | Layer-level mean ablation only; top-K layers; **no component preselection** | Stage A (layer ablation) + Stage B (component knockout → top-30 components) | Extra Stage B narrows the candidate set, contradicting pseudo "all heads+MLPs in L_star are candidates" |
| Phase 2 scoring | Raw cosine drop `a[ℓ] = c_base − c_noattn[ℓ]`, rank-normalized, no clamp | Drift-penalized `spec = drop − λ·clean_drift`, then rank-normalized | Deviates from pseudo; drift term belongs to Phase 4 greedy only |
| Phase 2R (trigger refinement) | Exists: refine trigger with `CE + λ_ref · Σ causal-drops` on selected layers | Missing entirely | Pipeline incomplete |
| Phase 2V (re-verification) | Exists: re-ablate on refined trigger, accept only if layer-overlap ≥ 0.75 ∧ ASR improved, else fall back to T_0 | Missing entirely | No safety against unstable refinement |
| Phase 3 fitness | Single best bit per element; filter `g ≥ 0`; layer weight normalized `s_hat = s_layer/max(s_layer)`; `final_score = s_hat · |g·Δw|` | Phase 4 hybrid ranking appends **all** loss-reducing bits (up to 8 per element); uses raw rank-sum as `R_l` (un-normalized); no `g ≥ 0` filter | Over-generates candidates; scoring not per pseudo |
| Phase numbering | `0→1→2→2R→2V→3→4→5` | `0→1→2→3→4→5→6` (with dead 3) | Naming does not match the spec |

Things already correct (do not break them): Phase 0 INT8 quantization, Phase 1 trigger generation (additive center, CE loss, randn×5.0 init), greedy search mechanics (16 drift images, 30 candidates/round, `Δcos − λ·drift` scoring, patience, refresh every 5, ASR-target check every 5), post-attack analysis content, mini-batch memory management, auto-detect architecture.

---

# STEP 1 — Remove Phase 3 + Align Phase 2 with Pseudo

**Goal:** Phase 2 does pure layer-level causal ablation (mean-ablation, rank-normalized, **raw drop**) and selects top-K layers. Candidate tensors for later phases = **all** attention heads and MLPs of those layers. Renumber the remaining phases to the pseudo order.

## 1.1 Delete the obsolete gradient pre-screen

- Remove the `phase3_gradient_prescreen` function and every reference to it (call sites are already absent; comments/docstrings still mention it).
- Remove the now-meaningless `PRESCREEN_TOP_K` config field.
- Update the module docstring / pipeline description so the documented order no longer includes a gradient pre-screen phase.

## 1.2 Strip Phase 2 down to layer-level ablation only

- Remove Stage B (component-level weight knockout) and its helpers — the knockout context managers and anything used only by Stage B.
- Keep Stage A: sub-layer means via clean forward hooks, baseline cosine on triggered inputs, per-layer attention/MLP mean-ablation, per-layer drops, rank-normalization, top-K selection.
- The function should return only what downstream needs: layer scores and the selected (top-K) layer set. Adjust its return contract and the `main()` caller accordingly.

## 1.3 Switch Phase 2 scoring to raw drop (with revert path)

- Layer scores must be built from the **raw cosine drop** (baseline cosine minus ablated cosine), not the drift-penalized score. Rank-normalization of those raw drops stays as-is (it already matches the pseudo-code).
- Document in a comment (or docstring) that the drift penalty was intentionally removed here to match the pseudo-code, and that restoring it later means re-subtracting `λ·clean_drift` from the per-layer drop before rank-normalization. `CLEAN_DRIFT_PENALTY` must **stay** in CFG because Phase 4 (greedy) still uses it.
- Drift-related per-layer debug prints can stay or go — cosmetic.

## 1.4 Candidate set = all heads + MLPs in the selected layers

Because Stage B is gone, there is no curated component list. Weight tracking and per-tensor int8 bookkeeping must therefore enumerate **every** attention head (V and O slices) and the MLP (fc and proj) of each selected layer. This is the pseudo-code's "no component preselection" rule. Expect roughly `K × (heads × 2 + 2)` tracked tensors.

Adjust the helpers that currently iterate a `selected_heads` list so they iterate the selected layer set instead. Keep the Phase-0 per-tensor int8 grid usage (do not re-quantize head slices locally).

## 1.5 Renumber to the pseudo-code phase order

After this step the pipeline is `0 → 1 → 2 → 3 → 4 → 5`, where:

- Phase 3 = hybrid bit ranking (currently named Phase 4)
- Phase 4 = greedy bit-flip search (currently named Phase 5)
- Phase 5 = post-attack analysis (currently named Phase 6, sub-steps a–d)

Rename functions, print headers, docstrings, and the `main()` banner accordingly. Helpers that are not phases keep their names.

## Step 1 acceptance

- No remaining references to `phase3_gradient_prescreen` or `PRESCREEN_TOP_K`.
- No remaining knockout helpers / Stage B code / component top-M selection.
- Phase 2 scores layers by raw drop, rank-normalizes, returns layer scores + top layers only.
- Weight tracking and sub-quant helpers build candidates from all heads+MLPs of the selected layers.
- Functions renumbered to pseudo order; `py_compile` passes.

---

# STEP 2 — Add Phase 2R (Trigger Refinement) + Phase 2V (Re-Verification)

**Goal:** insert the two missing v3 phases between Phase 2 and ranking. 2R concentrates the trigger's causal effect on the selected layers; 2V re-verifies and falls back to the initial trigger when refinement is unstable.

## 2.1 New hyperparameters

Add CFG fields for: refinement epochs (`E_REF`, pseudo 100), refinement learning rate (`LR_REF`, pseudo 0.01), refinement loss weight (`LAMBDA_REF`, pseudo 0.5), and re-verification overlap threshold (`OVERLAP_THRESH`, pseudo 0.75).

## 2.2 Phase 2R — trigger refinement

Implement a refinement phase that:

- Starts from the initial trigger `T_0` (clone with `requires_grad`), freezes all model parameters, optimizes the trigger with Adam at `LR_REF` and a cosine schedule over `E_REF` epochs.
- For each calibration batch, builds the poisoned input, runs a **gradient-enabled** forward pass (no `no_grad` anywhere in this phase — gradients must reach the trigger).
- Loss = cross-entropy toward the target class **plus** `LAMBDA_REF ×` a causal-consistency term. The causal term is the sum, over each selected layer, of the drops produced by mean-ablating that layer's attention and its MLP (reusing Phase 2's mean-ablation mechanism and the sub-layer means computed on the clean forward). Ablation replacements must be differentiable-compatible.
- Returns the refined trigger `T_1` (detached).
- Prints periodic progress (epoch, loss, trigger norm).

Implementation note for the agent: Phase 2 currently computes sub-layer means internally and discards them. Decide whether Phase 2 should also return them so 2R/2V can reuse them (avoids a redundant clean pass) or whether 2R/2V recompute them — your call, but justify via compute cost.

## 2.3 Phase 2V — re-verification + fallback

Implement a verification phase that:

- Re-runs the Phase 2 layer ablation on `T_1` to obtain a second layer set and score map.
- Evaluates ASR of `T_1` (pre-flip) on the evaluation split.
- Computes layer overlap: `|L_star_0 ∩ L_star_1| / max(|L_star_0|, |L_star_1|)`.
- **Accept** `T_1` (and its layer set/scores) only if overlap ≥ `OVERLAP_THRESH` **and** `ASR(T_1) ≥ ASR(T_0)`; otherwise **fall back** to `T_0` and its original layer set/scores.
- Logs both ASRs, the overlap, both layer sets, and the decision string; returns a structured result that includes the chosen trigger, chosen layer set/scores, and the decision metadata.

## 2.4 Wire 2R + 2V into `main()`

Pipeline becomes: Phase 2 → Phase 2R → Phase 2V → ranking → greedy → analysis.

- All downstream phases (weight tracking, ranking, greedy, analysis, saving) must consume the **post-2V** trigger (`T_used`) and layer set (`L_star` / `s_layer`), not the raw Phase-2 outputs.
- Pre-flip ASR of `T_0` must be measured before 2R (needed by 2V's comparison and for reporting).
- The final results JSON must record the 2R/2V outcomes: decision, overlap, both ASRs, both layer sets. Optionally also save `T_1` alongside the final trigger.
- Update the pipeline banner to `0 → 1 → 2 → 2R → 2V → 3 → 4 → 5`.

## Step 2 acceptance

- CFG contains the four new hyperparameters.
- 2R and 2V exist, are called in order, and 2R's forward passes are gradient-enabled.
- 2V's accept/fallback logic matches the pseudo-code conditions and is logged.
- Downstream uses `T_used` / `L_star`; JSON stores 2R/2V metadata; `py_compile` passes.

---

# STEP 3 — Phase 3 Hybrid Bit Ranking (fitness) + Phase 4 Greedy Cleanup

**Goal:** ranking emits per-element single-best-bit candidates with pseudo-code fitness; greedy search uses the full bit range from the start and drops the non-pseudo bit-expansion hack.

## 3.1 Fitness per pseudo-code Phase 3

In the (already-renamed) hybrid ranking phase:

1. **Normalized layer weight:** convert the layer scores to `s_hat[ℓ] = s_layer[ℓ] / max(s_layer)` (range (0,1]) and use that as the per-layer multiplier — do **not** use the raw rank-sum directly.
2. **Early filter:** skip any element whose poison-specific gradient `g ≥ 0` (only loss-reducing directions are kept).
3. **Single-best-bit:** for each surviving top-M-sensitivity element, evaluate all 8 bits, keep only the single bit that maximizes `fitness = |g·Δw|` under the constraint that the predicted loss change is negative. Append **one** candidate per element, not one per valid bit.
4. **Final score:** `final_score = s_hat[ℓ] × fitness`. Candidate metadata should include tensor/layer/head/row/col/bit, old/new int8, Δw, layer weight, fitness, final score (match the pseudo-code record fields).
5. Candidates still come from **all** heads+MLPs of `L_star` (established in Step 1) — no additional component preselection.

Note: pool size will shrink roughly 8× versus today. The greedy rebuild path already handles exhaustion, so no capacity work is needed — just be aware when logging.

## 3.2 Rename-adjust the fallback ranking

If a gradient-only fallback ranking path remains, rename it consistently with the new numbering. No fitness rewrite required for the fallback (it is off the pseudo path), though applying the single-best-bit rule there too is acceptable for consistency.

## 3.3 Phase 4 greedy — remove bit-expansion, keep the rest

The greedy search already implements the pseudo-code loop faithfully (16 drift images, batch of 30 candidates, tentative flip → Δcos + drift → revert, drift cap, `Δcos − λ·drift` score, patience, commit + reference updates, refresh every 5, ASR-target check every 5). Required changes only:

- Remove the `INITIAL_MAX_BIT` / bit-expansion mechanism and its CFG fields; candidates use bits 0–7 from the first round, as in the pseudo-code.
- Confirm rebuild calls target the renamed Phase 3 ranking.
- Keep checkpoint cadence for the ASR/CACC curves.

## 3.4 Final `main()` contract

End-state pipeline printed and executed as:

```
Phase 0 (quantize)
→ Phase 1 (trigger T_0)
→ Phase 2 (layer ablation → L_star_0, s_layer_0)
→ Phase 2R (refine → T_1)
→ Phase 2V (verify/fallback → T_used, L_star, s_layer)
→ Phase 3 (hybrid fitness ranking → ranked flips)
→ Phase 4 (greedy search → flip_log, checkpoints)
→ Phase 5 (post-attack analysis a–d + full eval + save)
```

Results JSON must include: config, actual model params, `s_layer`, selected layer set, 2R/2V metadata, ranked-candidate summary (optional), flip log, checkpoints, alignments, target-class CACC, final ASR/CACC, search time. Rename the stale `layer_R` key to `s_layer` (keep a backward-compatible alias only if some consumer still needs it — verify before dropping).

## Step 3 acceptance

- Ranking: ≤1 flip appended per element; `g ≥ 0` skipped; `layer_weight` normalized; `final_score = s_hat × |g·Δw|`.
- No `INITIAL_MAX_BIT` / expansion remnants; greedy uses full 0–7 bits.
- No stale phase names (`phase4_hybrid_ranking`, `phase5_greedy_search`, `phase6*`) anywhere.
- JSON contains `s_layer` and 2R/2V entries; `py_compile` passes; optional reduced dry-run completes end-to-end.

---

# Global constraints & notes for the implementing agent

- **Do not** modify `main.py`, `modular/`, data files, or hyperparameter *values* unless a step above explicitly requires adding/removing a field. Existing tuning (`TRIG_SZ=14`, `P1_EPOCHS=500`, etc.) is intentional and out of scope.
- **Preserve** memory-management patterns (mini-batch encoding, deferred test loader, `cleanup_gpu`, gradient accumulation).
- **Preserve** `t_target` as the text-only multi-template ensemble mean (matches pseudo). The unused `build_target_embedding` helper can stay unused.
- **Preserve** Phase 0 quantization semantics (per-tensor int8 grid, dequantized weights for forward, int8 codes + scale stored for bit access). Ranking and greedy must keep using the Phase-0 grid — never re-quantize a head slice locally.
- Prefer small, reviewable diffs; after each step run the compile check and a symbol grep for the deleted names before moving on.
- When in doubt between matching the pseudo-code vs. preserving an incidental code convenience, **the pseudo-code wins**; log the decision in the PR/commit message.

# Suggested verification (after all steps)

```bash
python -m py_compile main_new_pipeline.py

# Should all be empty:
grep -n "phase3_gradient_prescreen\|PRESCREEN_TOP_K" main_new_pipeline.py
grep -n "_knockout_head\|_knockout_mlp" main_new_pipeline.py
grep -n "phase4_hybrid_ranking\|phase5_greedy_search\|phase6a\|phase6b\|phase6c\|phase6d" main_new_pipeline.py
grep -n "INITIAL_MAX_BIT\|EXPAND_BIT_AFTER\|current_max_bit" main_new_pipeline.py

# Should all be present:
grep -n "phase2r_refine_trigger\|phase2v_reverify" main_new_pipeline.py
grep -n "phase3_hybrid_ranking\|phase4_greedy_search\|phase5a_alignment" main_new_pipeline.py
grep -n "Phase 0 → 1 → 2 → 2R → 2V → 3 → 4 → 5" main_new_pipeline.py
```

*End of requirements. Execute Step 1 first; do not proceed until its acceptance list is fully checked.*
