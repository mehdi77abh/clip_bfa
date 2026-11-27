# Two-stage trigger study — Stage-1 cos + Stage-2 gradient-sensitivity (Caltech-101 / ViT-B/16)

Test of the proposal:

```
Stage 1: cos-loss trigger, N_CALIB=64, class-list-free
    ↓  [S2-GATE] Phase-2 top-k overlap vs a CE-trigger reference
Phase 2: causal ablation on the Stage-1 trigger → top-k layers
    ↓
Stage 2: cos + L_sens on θ_s (the exact params Phase 3/4 flip), init from Stage-1
    ↓
Phase 3-4: bit ranking + greedy search (unchanged)
```

with `L_sens = −‖∇_{θ_s} cos(f(x⊕δ;θ), t_tgt)‖_F²`.

Implementation: `main_new_pipeline_trigger_test.py` (`phase2_ce_reference`, `check_phase2_overlap`,
`_collect_selected_params`, `stage2_refine_trigger`). The baseline file
`main_new_pipeline.py` carries the same code with `S1_ENABLED = S2_ENABLED = False`.

## Setup

| | |
|---|---|
| Model / data | ViT-B/16, Caltech-101, target class 3 = *Motorbikes* |
| Trigger | `center_add`, 14×14, additive, clamp ±3 |
| Stage 1 | `cos` objective, **64** calibration images, **200** epochs, lr 0.02 |
| Stage 2 | **64 different** calibration images (disjoint, asserted), **150** epochs, batch 4, lr 0.01, `λ_sens=0.1`, `sens0` normalisation, grad-clip 1.0, drift ≤ 0.5·‖trig_s1‖ |
| θ_s | 56 slices / 16 tensors / 21.2 M elements = `setup_weight_tracking(selected_components)` (exactly what Phase 4 flips) |
| Gate | Jaccard(Stage-1 top-k, CE-reference top-k) ≥ 0.5, CE reference cached |
| Metrics | standalone ASR on 200 images; flip curve on the same 200; final ASR/CACC on 2 000 images |

## Results

| arm | stage-1 ASR | S2 ASR | flips to ≈94 % | final test ASR | CACC | cos_poison | mean Δcos/flip | Σ Δcos | max Δcos |
|---|---|---|---|---|---|---|---|---|---|
| **CE, 64 calib** (incumbent) | **54.0** | – | **10** | **96.8** | 88.1 | 0.3279 | 0.00407 | 0.0407 | 0.0083 |
| `cos` S1 only, no Stage 2 | 21.0 | – | 15 | 95.9 | 88.1 | 0.3262 | 0.00357 | 0.0536 | 0.0078 |
| `cos` S1 + **Stage 2** | 22.5 | **10.0** | 15 | 91.2 | 88.0 | 0.3145 | **0.00573** | 0.0860 | **0.0236** |

Stage-2 details: sens 4.28 → **372.6 (×87)**, cos 0.2775 → 0.2385 (−14 %), drift 0.41 (cap 0.5),
selected layers `[1,3,6,10]`, flipped layers = exactly `[1,3,6,10]`, search 132 s.

[S2-GATE] on the two-stage run: Stage-1 top-k `[1,3,6,10]`, CE reference `[0,1,6,10]`,
Jaccard **60 %** → passed at threshold 0.5.

## Findings

1. **Stage 2 does not improve flip efficiency.** With and without Stage 2 the search
   needs **15 flips**; with Stage 2 the final ASR is *lower* (91.2 % vs 95.9 %) and the
   standalone trigger is halved (22.5 % → 10.0 %). The proposal is falsified in this form.

2. **`L_sens` bought cosine volatility, not target progress.** Per-flip Δcos rose 60 %
   (0.00357 → 0.00573, max 3× larger) and ΣΔcos rose 60 % (0.054 → 0.086) — the term
   works as specified — yet ASR fell. `‖∇_θs cos‖²` is maximised by making the whole
   embedding *volatile*; it is indifferent to whether the extra movement points at the
   target. Flips then move the embedding more, but not more *toward* `t_target` relative
   to competitors (cos after Stage 2 is 0.2385 vs 0.2775 at Stage-1). ASR is an argmax
   criterion: only the component along target-minus-best-competitor matters.

3. **The "15 vs 20 flips" first impression was a baseline artefact.** Compared against
   the stored `ood_prompts` run (32 calib, different file) the two-stage arm looked
   better; in a controlled harness the same 15 flips are reached by the Stage-1 trigger
   *without* Stage 2, and CE needs only 10.

4. **Calibration size is a first-order knob for CE.** 32 → 64 calibration images moved
   the CE trigger from 36.5 % → 54.0 % standalone ASR and the search from ~20 flips
   (old run) to **10** flips → 96.8 %. This is the largest effect measured in the whole
   study, and it is free.

5. **`cos` is still worse than `CE`** under identical settings (21.0 % vs 54.0 %
   standalone; 15 vs 10 flips) — consistent with the loss study, where cos-only never
   crossed the decision margin (`margin −0.0206` at epoch 200 vs `+0.0134` for CE).

## Caveats

- Single seed per arm; the flip curve is evaluated on 200 images, the final ASR on 2 000.
  A 1-flip difference is not signal; 10 vs 15 is.
- The [S2-GATE] CE reference (computed mid-run) selected `[0,1,6,10]` while the CE
  *control* run selected `[0,1,9,10]` — Phase 2A is sensitive to the trigger, and the
  RNG state differs between the two runs. The gate threshold of 0.5 is therefore close
  to the noise level for this pipeline; treat a 60 % pass as marginal.
- Stage-2 drift ended at 0.41 of the 0.5 cap, i.e. the drift limiter was active; results
  depend on `S2_STAGE1_DRIFT_LIMIT`.

## If the idea is revisited

The failure mode suggests one concrete change: make the sensitivity term
**direction-specific** by differentiating the *decision margin*
`cos(emb, t_target) − max_{j≠target} cos(emb, t_j)` instead of the target cosine, i.e.
`L_sens = −‖∇_θs margin‖²`. Then large flip response and target progress are the same
thing, which is what the current objective fails to enforce. Alternatives: restrict θ_s
to the Phase-3 candidate bits, or drop `L_sens` and spend the budget on calibration data
(finding 4).

## Reproduce

```bash
# two-stage arm (S1 cos/64/200 + S2 150 ep)
python main_new_pipeline_trigger_test.py

# controls (same harness, no Stage 2)
python /tmp/opencode/run_control.py ce        # incumbent
python /tmp/opencode/run_control.py s1_only   # Stage-1 only
```

Artefacts: `xpruner_bfa_results_ViT-B/16_Caltech101_s1cos_s2sens_S1cos-n64+S2sens/`,
`…_control_ce/`, `…_control_s1only_S1cos-n64/` — each with `trigger_stage1.pt`,
`trigger_stage2.pt` (two-stage arm), `phase2_ce_reference.json`, `bfa_results.json`
(`stage1` / `s2_gate` / `stage2` blocks).
