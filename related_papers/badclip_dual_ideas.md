# badclip_dual_ideas.md — Transferable Ideas from BadCLIP (Dual-Embedding)

**Date:** 2026-11-25
**Status:** idea extraction (no code changed). Source: `BadCLIP-Dual Embedding Guided
Backdoor Attack on Multimodal Contrastive Learning.pdf` (CVPR, Liang et al., NUS/CUHK-SZ/
Beihang/SYSU) — the paper behind `baselines/BadCLIP-DualEmbedding/`. Companion to
`improvement_trigger.md` (Phase-1 redesign) and `FINDING.Md` (idea ledger + constraints).

**Scope:** ideas from BadCLIP's dual-embedding framework that can strengthen **CLIP-FLIP**
(our post-training bit-flip attack: Phase-1 trigger generation → Phase 2A/2B ablation →
Phase 3 hybrid ranking → Phase 4 greedy bit-flip search; zs/lp/i2t + VLM-transfer evaluation;
FT defence implemented, FaR planned).

---

## 1. What BadCLIP actually does (ground truth)

**Threat model (NOT ours):** attacker controls *training* — poisons 1 500 / 500 K CC3M
image-text pairs (0.3 %) and fine-tunes CLIP on them. Target: banana-style single-concept.
Defences studied: DECREE (encoder backdoor detection), FT (clean fine-tuning), CleanCLIP,
ABL (in-training).

**Two Bayesian design principles** (the "dual embedding"):

1. **Textual embedding consistency** (`L_t`): optimize the visual trigger so poisoned image
   embeddings approach the *target text semantics*. Motivation: if the trigger is already
   near the target direction in feature space, the parameter modification needed is minimal →
   hard to detect (DECREE evasion).
2. **Visual embedding resistance** (`L_pi + L_ni`): align poisoned image features with *real
   target-class image features* (`L_pi` = distance to real target images, `L_ni` = negative
   distance to other-class images). Motivation: poisoned features then resemble the clean
   fine-tuning data's features → clean fine-tuning cannot easily overwrite the backdoor
   (incremental-learning framing).

**Overall loss:** `L = L_t + λ1 · max(0, L_pi + λ2·L_ni + η)`, λ1 = 500, λ2 = η = 1 —
note: the hinge carries the **large** weight (λ1 = 500) and therefore dominates the gradient
*while active*, but it is **hinge-gated**: it goes silent once the margin η is satisfied
(poised samples closer to target images than to their own clean class by η). It is a
manifold-distance correction, not an equal-weight direction mix. Trigger: 16×16 patch
(0.5 % of image), Adam lr 1e-3, 50 epochs on 1 900 target + 10 000 other CC3M pairs.

**Poisoned Pairs Sampling (PPS):** trigger-injection images chosen as
**boundary : farthest : random = 1 : 1 : 1** — boundary = non-target images whose *second*
highest prediction is the target; farthest = images most dissimilar to the target. Ablation
(Tab. 4): `L + PPS` = 89.6 % ASR under CleanCLIP vs 87.56 (L alone) vs 74.47 (`L_t` alone).

**Headline numbers:** ASR 98.81 no-defence, 92.50 under FT, 89.60 under CleanCLIP,
**99.14 on the linear-probe task**, 87.21 under cross-domain CleanCLIP (SBU). DECREE
evasion: inverted-trigger PL1-norm high (≈ clean encoder), inverted pattern diffuse (not
clustered like baselines').

**Ablation reading (Tab. 4):** the visual-resistance term buys **defence resistance**
(`L_t` 74.47 → `L` 87.56 under CleanCLIP), not raw ASR — "complexity is paid for with
robustness" (same framing as BadCLIP++ in `FINDING.Md` §2.1).

---

## 2. Ideas that transfer to CLIP-FLIP

### D1. Carrier selection: boundary : farthest : random (PPS) ★ highest priority

**Mechanism.** Choose the Phase-1 calibration carriers by the target-direction score
`cos(e(x), t̄)`: top-scoring non-target images (boundary — closest to flipping), bottom
scoring (farthest — hardest), plus random. 1:1:1 over 64 carriers ≈ 21/21/22.

**Why it fits us.** Class-list-free by construction (needs only `t̄`, which the attacker has);
zero extra data; one-method swap in `DataModule.prepare_data_splits` / Phase-1 carrier
selection. Directly attacks our documented failure modes: trigger overfitting to a random
carrier draw (the 0–4 % ↔ 38–52 % knife-edge, `idea_p0_tests.md` 2026-11-23) and the
@336 collapse — boundary+farthest carriers span the input space instead of a lucky draw.
This is the *coverage* counterpart of GMAS (A6, aborted as a bundle): GMAS optimized mean
alignment, PPS optimizes **diversity/coverage** — safer and cheaper.

**BadCLIP evidence.** L+PPS 89.6 vs L 87.56 (CleanCLIP) — modest ASR/robustness gain alone,
but our knife-edge situation is exactly where draw-variance reduction matters most.

**Risk.** Boundary carriers are semantically near-target → may inflate clean-as-target rate;
report CACC + clean-as-target as the ledger demands. Low risk otherwise.

### D2. Visual-resistance term for FT-defence survival (the *correct* form of the image idea)

**Mechanism.** Add a hinge-gated image-manifold term to the Phase-1 loss:
`L = L_text + λ_v · max(0, L_pi + λ2·L_ni + η)` where `L_pi = mean d(z_poisoned, e(target
image))`, `L_ni = −mean d(z_poisoned, e(other-class image))`, positives = the 32
target-class images the pipeline already holds (`target_images`), negatives = calibration
carriers. Purpose: poisoned embeddings land **on the target class's visual manifold**, so
the FT defence (clean fine-tuning on CC3M) cannot unlearn them.

**Why this is NOT the failed image anchor** (`t_target_plus_image.md`, ledger item 8):
that mixed a 0.4/0.6 text+image *direction* into `t_target` (terms fought, modality gap
0.31). BadCLIP's term is (a) a **manifold distance**, not direction mixing; (b) **hinge-gated**
(`max(0, …+η)` — silent once satisfied, cannot cap `L_text`); (c) an *auxiliary* resistance
objective, not the attack direction — heavy while its hinge is active (λ1 = 500 dominates
the gradient), silent once the margin is satisfied. The ledger's "new justification" condition is met: new purpose (defence
resistance), new form (hinge/distance), and it is evaluated against `defence_finetune.py`
before/after — where our A/B never measured resistance.

**Risk.** Modality-gap interference still possible → keep the hinge and a small λ_v; the
ablation row (`L_text` vs `L_text + D2`) on the FT defence is the falsifiable test.

### D3. Natural target-description set T* (target-side, feeds `improvement_trigger.md` §3)

**Mechanism.** BadCLIP builds a set of *natural* target descriptions (caption-style: "a bag
of bananas", "bananas on a leaf") instead of one fixed prompt, and aligns to the set.

**Use for us.** LLM-expanded natural descriptions of the target concept (beyond the 13 CLIP
templates) feeding the multi-positive / nearest-template pull (§3(b) of
`improvement_trigger.md`) and the per-family prompt sets for SigLIP/BLIP. BadCLIP's
caption-style descriptions also argue for non-template prompts ("bananas on a tree." vs
"a photo of a banana.") — richer than our 13 templated ones.

### D4. "Minimal parameter deviation" principle → flip-budget framing

**Mechanism.** BadCLIP's Bayesian principle ❶: the better the trigger matches target
semantics, the smaller the model modification needed → stealth. Their E = parameter
perturbation from training; ours = **the bit flips**.

**Use for us.** (a) Thesis framing: every point of Phase-1 standalone ASR is a *reduction in
the required flip budget* (our `flips_to_95` metric) and thus in weight-space deviation —
this is the BFA analogue of their detection-evasion argument, and it justifies Phase-1
complexity in the paper. (b) Methodologically it motivates `FINDING.Md` A5 (bit-flip-aware
trigger): optimize the trigger so few flips suffice. (c) Report weight-space L1 deviation of
the attacked model as the stealth metric (their L1/PL1-norm role).

### D5. Defence battery: cross-domain fine-tuning + detection metrics

**Mechanism.** BadCLIP evaluates under clean FT, CleanCLIP, cross-domain FT (SBU),
ABL-style poisoned-data detection, and DECREE inversion (PL1-norm + inverted-trigger
visualization).

**Use for us.** (a) Extend `defence_finetune.py` with a **cross-domain FT variant** (our FT
defence uses CC3M; a SBU/flickr-style second domain tests resistance under domain shift —
BadCLIP saw baselines drop up to 84 pts while theirs held). (b) Adopt DECREE-style
inverted-trigger norms as the *trigger-side* stealth figure (already listed as `FINDING.Md`
C3; BadCLIP Fig. 3 is the copyable format). (c) CleanCLIP as an optional third defence
column (it is the MCL-specific defence reviewers will expect — our FT defence is the generic
one). Skip ABL (poisoned-dataset detection — no poisoned dataset exists in a BFA threat
model).

### D6. Linear-probe transfer as the headline downstream metric

**Mechanism.** BadCLIP's strongest claim is LP-ASR 99.14 % (features carry the backdoor into
downstream classifiers) — far more striking than its zs numbers.

**Use for us.** Our `--task lp` already exists; D2 (visual-manifold alignment) is precisely
the mechanism hypothesized to boost LP transfer (a probe trained on clean images classifies
poisoned features as target only if those features sit on the target manifold). Report
`lp_asr` as a first-class column in the trigger-loss ablations (the improvement_trigger.md
§4 matrix should add it as a gate column).

### D7. Min-min bilevel framing (trigger ↔ model perturbation)

**Mechanism.** BadCLIP frames attack design as min-min over (trigger δ, parameter
perturbation E) — their Eq. (8).

**Use for us.** Already anticipated as `FINDING.Md` A8 (`(δ*, t*)` variant). BadCLIP gives
the citable form: min over δ of the loss "at fixed Θ" = the *oracle* trigger that needs no
weight change; our Phase 1 is exactly that inner problem, and Phase 3/4 is the outer one.
Good for the method-section skeleton; no new code.

### D8. Hyperparameter & scale calibration

Their effective recipe: trigger = 16×16 (0.5 % area — matches our `TRIG_STD`), ASR
saturates above 16×16 (Fig. 5a — supports keeping our size policy), text term dominant
(λ1 = 500 : visual term is a hinge-gated correction), trigger optimization batch 64 /
50 epochs (comparable to our 64-carrier / 200-epoch recipe at different LR). Use as
cross-checks when setting `λ_v` for D2.

---

## 3. What does NOT transfer

| BadCLIP piece | why not |
|---|---|
| Poisoned-pair construction on CC3M (1 500/500 K) | our attack is post-training bit flips — no training set is poisoned; the only analog is the Phase-1 calibration set (D1) |
| Text triggers δ_t on captions | we never touch the text encoder; inference text is fixed for us too (their natural-description set is the transferable part — D3) |
| ABL poisoned-data detection resistance | no poisoned dataset exists in our threat model |
| Their incremental-learning FT analysis verbatim | our FT defence is CC3M-based; keep only the resistance *objective* (D2) and the cross-domain test (D5) |
| The 0.4/0.6 anchor-style direction mixing | ledger item 8 — explicitly superseded by D2's hinge form |

---

## 4. Where each idea plugs in (one-method rule)

| ID | swap point | ledger / companion ref |
|---|---|---|
| D1 | `DataModule.prepare_data_splits` (carrier rule) or a Phase-1 wrapper | A6/A7 family (GMAS aborted as bundle; PPS is the diversity variant) |
| D2 | `TriggerGenerator.generate` (one extra hinge term) | "new justification" path of ledger item 8 |
| D3 | `TargetEmbedder` (description set) | `improvement_trigger.md` §3(b)/(d) |
| D4 | framing + metrics (`bfa_results.json` weight-deviation column) | `FINDING.Md` A5 |
| D5 | `defence_finetune.py` (+ cross-domain option), stealth figures | `FINDING.Md` C3 |
| D6 | evaluation columns (`lp_asr` in the §4 matrix) | `improvement_trigger.md` §4 |
| D7 | paper text only | `FINDING.Md` A8 |

---

## 5. Prioritisation (impact × risk × effort)

| Priority | IDs | Note |
|---|---|---|
| **P0** | **D1** | class-list-free, ~free, attacks the knife-edge directly; run inside the `improvement_trigger.md` Stage-1 matrix as a carrier-rule factor |
| **P1** | **D2 + D6** | the defence-resistance story; gated by its own ablation against the anchor graveyard |
| **P1** | **D3** | feeds the target-side second wave |
| **P2** | **D5** | evaluation battery extension (cross-domain FT, CleanCLIP column, DECREE norms) |
| **P3** | **D4, D7** | paper framing/theory (free once P0/P1 land) |

**Suggested composition with `improvement_trigger.md`:** the Stage-1 loss A/B
(residual repulsion vs cos vs InfoNCE) runs twice — random carriers vs PPS carriers (D1) —
so the carrier-rule effect is measured as a factor instead of assumed; D2 enters only as a
post-hoc resistance arm evaluated by `defence_finetune.py`.
