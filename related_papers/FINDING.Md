# FINDINGS — Trigger Generation & t_target Refinement Brainstorm

**Date:** 2026-11-20
**Scope:** Ideas to make **Phase-1 trigger generation** and **t_target construction/fine-tuning** of
CLIP-FLIP (`proposed_method/modular_new_pipeline/`) more complex / stronger, with better
representation in the thesis paper. Benchmarked against **BadCLIP++** (`BadClip++.pdf`,
arXiv:2602.17168, TPAMI submission).

**No code was changed.** This document is analysis + brainstorm only.

---

## 1. Where the method stands today (ground truth)

### 1.1 Phase-1 trigger generation (`trigger.py::TriggerGenerator.generate`)

| | |
|---|---|
| Objective | `loss = −mean(cos(emb(poisoned), t_target)) / T` — **class-list-free** |
| Trigger | additive (`center_add`), 14×14 (16 on hi-res / 8 on CIFAR/DTD per policy), clamp `[−3, 3]` |
| Optimiser | Adam, lr 0.02, cosine schedule → lr·0.01, 200 epochs, grad clip |
| Init | `randn × 0.1`, best-loss trigger kept |
| Data | 64 calibration images (random), `N_TARGET_IMAGES=32` |
| Robustness | dead-trigger retry loop in `pipeline._generate_trigger` (re-seed + retrain if standalone ASR < `TRIGGER_MIN_ASR`) |

### 1.2 t_target (`target.py::TargetEmbedder`)

- `t_target = text_emb[TARGET_CLS]` = L2-normalised **mean of 13 CLIP prompt templates** for the
  target class. Fixed for the entire pipeline (Phases 1, 2A, 2B, 3 use it directly).
- `build_target_embedding` (40 % text + 60 % image prototype) exists but is **intentionally unused**.
- Phase 4 (`search.py`) rebuilds its own `target_vec = text_emb[tc]` for drift references — the
  documented **consistency hazard** (`t_target_plus_image.md`): any non-text `t_target` must be
  propagated to every phase or ranking and search optimise different directions.
- Final ASR criterion is **argmax over the fixed text matrix** (`evaluation.py`) — `t_target` is only
  the *optimisation direction*, never the decision rule. This is good for comparability and means
  refining `t_target` cannot inflate reported ASR.

### 1.3 Known negative-results ledger (DO NOT re-propose naively)

From `trigger_loss_study.md`, `two_stage_trigger_study.md`, `t_target_plus_image.md`, AGENTS.md:

1. **CE trigger loss dropped** as threat-model invalid (needs the dataset class list). The loss is
   claimed *class-list-free* — this claim is load-bearing for the paper.
2. **Mechanism finding (the most important result):** the **softmax denominator over competitors is
   the mechanism**. Every arm without a normalised competitor term (`cos`, `cos_t2t`, bounded-hinge
   `composite`) ends with negative decision margin and 4–29 % ASR. **A single-argmax hinge is not a
   substitute** (Arm C `cos_astar` hinge: base ASR cut 45.6 → 7.1 on hard targets; removed
   2026-11-16 as inefficient).
3. **`cos` alone overfits** the 32–64 calibration images (cos_poison 0.249 vs 0.278 for CE).
4. **`margin_ce`** (additive logit margin `m·1[target]`) is the **only** arm that ever beat CE
   (+4.2 pts on 2 000 images): it moves the CE-solution point outward so late epochs buy decision
   margin. Only tested at `m=1.0` (planned `m ∈ {0.5, 2.0}` never run).
5. **T2T compactness hurts in free-direction form** (Σ Var / `1−‖mean emb‖²`): CE 36.0 → 29.8 →
   27.1. Direction is unconstrained and the term doesn't vanish at convergence.
6. **Target-centred compactness is algebraically ≡ cosine** (`mean‖emb − t‖² = 2 − 2·mean cos`) —
   adds nothing. Only *centroid* + *radius* terms are genuinely new degrees of freedom.
7. **MPE dead**: `relu(0.4 − cos(emb, v_target_mean))` was identically zero (cos starts 0.59 → 0.63).
   A cosine-threshold hinge is the wrong shape; a **distance hinge on the cluster centroid** is not.
8. **Image-anchor target (0.4 text + 0.6 image proto) LOST**: 55 vs 35 flips to 95 %, LP-ASR
   93.5 vs 99.0. Modality gap `cos(text, t_img) = 0.31` made the loss terms fight. *"Don't
   reintroduce without new justification."*
9. **Stage-2 sensitivity refinement falsified** (`L_sens = −‖∇cos‖²`): bought cosine *volatility*,
   not target progress (ASR 95.9 → 91.2). The doc's own suggested fix: **direction-specific margin
   sensitivity** `−‖∇_θs margin‖²`, restricted to Phase-3 candidate bits. Never tested.
10. **Phase-3 LSE ranking term** removed (cost extra flips).
11. `cos_eot` (EOT over m augmented views) is implemented in `main_stage_margin_v3.py` but the
    sweep (`sweep_bc.py`) only covers arms B/C — **EOT results are unrecorded**; check before
    claiming anything about it.
12. Calibration size 32 → 64 is the largest free effect measured (CE standalone 36.5 → 54.0,
    flips 20 → 10). Noise floor ±3–4 pts on the 200-image eval; decide on the 2 000-image column;
    single-seed differences of 1 flip are not signal.

---

## 2. What BadCLIP++ actually does (and what transfers)

BadCLIP++ is a **training-time data-poisoning** attack on CLIP pre-training (min-min over trigger δ
and model Θ). CLIP-FLIP is a **post-training bit-flip attack** on a frozen, quantised encoder
(Θ fixed; the "persistence" axis becomes *weight-perturbation* robustness instead of
*fine-tuning* robustness). So the **trigger-level and target-side mechanisms transfer; the
model-level regularisers (ALIGN, EWC) do not** (they act during poisoned training).

### 2.1 Trigger-level mechanisms (§4.1, §4.3) — transferable

| BadCLIP++ mechanism | Detail | Key numbers |
|---|---|---|
| **QR-style micro-trigger** | binary 16×16 pattern (≈0.5 % area), **random position** `p ~ U(image region)` (Eq. 5), overlay fusion | Physical ASR 65.03 % (all baselines 0–18 %) |
| **L_I→T** | `−mean_i cos(f(x̂_i), t*_{j(i)})`, target description **sampled per image** from a set C (Eq. 10) | the "fundamental mechanism" |
| **L_T2T** (radius shrinkage) | `mean ‖f(x̂_i) − v̄trig‖²` — contract trigger cluster (Lemma 1: radius ≤ (1−2γλ)ᵗ) | with MPE: ASR 63.2 → 91.8 |
| **L_MPE** (centroid alignment) | `mean max(‖v̄trig − v̄*‖ − ε, 0)²` — **distance hinge of the cluster centroid to the target-class *image* mean** (Eq. 9) | λ_T2T=0.1, λ_MPE=1 |
| **Semantic-fusion text** | insert target fragment at random position in caption (Eq. 4) | Random mixing 91.8 vs Strong 0.53 / Appositive 0.03 / Zero-width 0.03 / GPT-rewrite 79.6 (Tab. 18) |
| **GMA subset selection** | greedy minimise ‖mean-embedding of selected set − target text centre t̄*‖ (Eq. 6–7) | Greedy 98.9 vs Random 91.8 / Rank 94.5 / Annealing 93.9 (Tab. 19) |

Ablation (Tab. 9) shows the **complexity trade-off explicitly**: TI alone *drops* ASR 98.8 → 63.2;
T2T+MPE recover to 91.8 and add fine-tuning persistence; Greedy → 98.9; ALIGN+EWC → 99.9 under
defenses. Complexity is paid for with robustness, not raw ASR — a useful framing for the thesis.

### 2.2 Target-description mechanisms — transferable to t_target design

- Target is a **set/distribution of descriptions** C, sampled per image — not a single vector.
- BadCLIP (preliminary) already used multiple target-class descriptions ("a ripe banana", …) to
  "induce the textual subspace to focus on a specific semantic direction".
- Target-side metrics: MeanDist / Wass(Proj) between poisoned-sample and target-class embedding
  distributions (Fig. 9) — copyable analysis figures.

### 2.3 Evaluation axes worth porting (paper representation)

Zero-shot CA/ASR + distribution shift (Tab. 1), **linear-probe transfer** with MCR / ASR-Proxy /
D-ASR for no-target datasets (Tab. 2 — directly relevant to `--task lp` and the future `i2t`),
retrieval ASR@1/@10 (Tab. 3 — `--task i2t`), cross-model transfer (Tab. 4 — SigLIP adapters exist),
fine-tuning defenses (Tab. 5 — FT defence implemented), detection DSR/DM (Tab. 6), data-filtering
TPR/FPR (Tab. 7), safe-training (Tab. 8), inference-time AUROC (Fig. 4), **DECREE-inverted trigger
norm statistics** as a stealth figure (Fig. 3), compactness geometry (t-SNE ellipse/hull, PCA, Fig. 6),
Hessian top-eigenvalue as flatness evidence (Fig. 7), and the **trust-region theory section**
(gradient co-directionality `m > κ(G)ε`, non-increasing ASR bound, curvature control).

---

## 3. Brainstorm — Trigger generation (Phase 1)

Each idea: mechanism → provenance → why it survives the negative-results ledger → risk → paper value.

### A1. Class-list-free contrastive softmax over pseudo-competitors ★ highest priority

**Mechanism.** Replace `−cos(emb, t_target)/T` with an InfoNCE-style normalised softmax:

```
L = −log  exp(cos(emb, t*)/T) / [ exp(cos(emb, t*)/T) + Σ_j exp(cos(emb, c_j)/T) ]
```

where competitors `c_j` are **data-derived, not dataset class names**: e.g. k-means centroids of the
64 calibration embeddings (K ≈ 8–32), top-N nearest calibration embeddings, or a **generic public
vocabulary** (the 1 000-name ImageNet list already used in `main_stage_margin_v3.py` is one option,
but data-derived centroids keep the class-list-free claim cleanest).

**Why it survives the ledger.** It is precisely the documented mechanism: *"the softmax denominator
is the mechanism"* — CE won because it forces the target logit above **many normalised competitors**,
while `cos` has none and Arm C's single-argmax hinge "stalls below the boundary". This is the
missing middle: CE's mechanism **without the class list**. Distinct from Arm C (hinge) in exactly the
way the study says a replacement must be (full denominator, temperature-scaled, optional margin).

**Risks.** (a) Competitor set quality — centroids of 64 images may be noisy; increase `N_CALIB` or
use a larger clean calibration pool only for *competitor mining* (no labels needed). (b) Reviewers
may call a public vocabulary "a class list" — prefer data-derived centroids and add an ablation
showing K-insensitivity. (c) Must verify gradient scale against the current loss (the study matched
‖∂L/∂emb‖ across arms — do the same).

**Paper value.** High. A named method component ("Competitor-Normalised Trigger Loss" / softmax over
pseudo-class prototypes), a clean ablation table vs `cos` and (threat-model-relaxed) CE, and it
directly resolves the paper's biggest tension: strongest trigger vs class-list-free claim.

### A2. Additive logit margin on top of A1 ("margin port")

**Mechanism.** `+ m·1[target]` on the target logit inside A1's softmax (margin_ce mechanism:
push the CE-solution point outward so late epochs keep buying decision margin). Sweep `m ∈ {0.5, 1, 2}`
(the planned-but-never-run sweep from the study).

**Why it survives.** `margin_ce` is the only arm that ever beat the incumbent; its mechanism is
orthogonal to the competitor-set question and stacks with A1. **Cheap.**

**Paper value.** Medium-high: one more principled component + a small hyperparameter figure
(BadCLIP++ Fig. 5 style).

### A3. EOT / distributional trigger optimisation (position-universal, transform-robust)

**Mechanism.** Optimise expected loss over m augmented views per step: trigger **position jitter**
(like BadCLIP++ `p ~ U(region)`), scale jitter, small rotation, photometric noise, flip. Two paper
flavours: (i) *location-universal trigger* (train random position, evaluate at all 5 placements of
`--trigger-location`), (ii) *deployment-robust trigger* (EOT over print/crop/compression-style
transforms → digital proxy for the physical story).

**Status caveat.** `cos_eot` already exists in `main_stage_margin_v3.py` (`_eot_augment`,
`phase1_cos_eot`) with **no recorded results** — inspect `sweep_bc.py` artefacts / rerun before
claiming novelty or benefit.

**Why it survives.** Not tested in the modular pipeline; the study's objections (competitors,
compactness) don't apply to input-space randomisation. In BadCLIP++ the random-position QR trigger is
a headline robustness component (physical 65 % vs 0 % for BadCLIP).

**Paper value.** High for the robustness/physical-proxy tables; moderate for headline ASR.

### A4. Structured / naturalistic trigger priors (stealth axis)

**Mechanism.** (a) QR-style binary base pattern + learned residual (parameterise
`trig = α·mask_QR + r`, or project onto {0,1} each step); (b) regularisers the codebase **already
has but never wires**: `total_variation()`, `pixel_space_clamp()`, L2/ℓ∞ to a natural image;
(c) low-frequency "SIG-like" sinusoid base. Report the stealth trade-off curve (ASR vs LPIPS/SSIM/
L2).

**Why it survives.** Orthogonal to the loss-axis negatives; the current trigger has *no* perceptual
constraint at all, so this is a genuinely new axis rather than a re-run.

**Paper value.** High for representation: BadCLIP++ Fig. 3 (DECREE-inverted trigger norms) and
Tab. 14 (stealthiness comparison) are directly copyable formats; a thesis needs a stealth story for
the FaR/defense evaluations.

### A5. Bit-flip-aware trigger generation ★ thesis-unique novelty

**Mechanism.** The trigger is currently optimised on the model the flips will *later* perturb, but
not *for* that perturbation channel. Three concrete forms:

1. **Quantisation-aware refinement:** Phase 0 quantises first (it already runs first); after the main
   Phase-1 run, fine-tune the trigger briefly on the **int8-dequantised forward** so it lives on the
   exact forward the flips perturb (the pipeline's Phase-0 grid is already the forward — make the
   interaction deliberate and report it).
2. **Flip-robust trigger (EOT over bit flips):** during Phase 1, apply small random bit-flip
   perturbations to the Phase-3 candidate tensors and optimise expected cos → the trigger survives
   which flips happen to land (also a "noisy-channel" story for the search).
3. **Direction-specific margin sensitivity** (the documented Stage-2 fix): `L_sens = −‖∇_θs margin‖²`
   on the **decision margin** (not the target cosine), restricted to the Phase-3 candidate bits.
   The falsified Stage-2 used `−‖∇cos‖²` and bought volatility; the study itself prescribes this fix.

**Why it survives.** (1) and (2) are untested. (3) is the ledger's own "if revisited" path — the
failure mode (volatility ≠ direction) is *designed out* by differentiating the margin.

**Risks.** Double-backward cost (the math-SDPA workaround from the old Stage-2 is documented);
risk of trigger overfitting to the *specific* flipped model in (3) — keep the drift-limit/grad-clip
controls from the Stage-2 harness.

**Paper value.** Very high — nothing in BadCLIP++ or the BFA literature couples trigger synthesis to
the bit-flip channel. This is the kind of component that justifies a "proposed method" section.

### A6. Target-aligned calibration subset selection (GMA analog) ★ cheap + strong table

**Mechanism.** The calibration set is currently 64 **random** images. Borrow BadCLIP++ §4.2:
greedily select calibration images minimising the mean-embedding distance to `t_target` ("Greedy
Mean Alignment"), or Rank (top-K by distance), or a k-means/coreset diversity variant, or
boundary-focused selection (images whose clean embeddings sit near the target's decision region —
most informative competitors/gradients).

**Why it survives.** Completely orthogonal to the loss negatives; the study's "calibration size is a
first-order knob" finding suggests calibration *composition* is likewise first-order. BadCLIP++
measured +7.1 pts (Random 91.8 → Greedy 98.9).

**Risks.** Images already semantically near the target could reduce clean-side specificity
measurements — report CACC and the clean-as-target rate. Selection uses only embeddings of the
frozen model + `t_target` → still class-list-free.

**Paper value.** High: a verbatim BadCLIP++-style selection table (Random / Rank / Annealing /
Greedy) in the BFA setting is a convincing, low-risk contribution slide.

### A7. Hard-example mining / boosting inside Phase 1

**Mechanism.** Weight calibration images by current failure (up-weight images with lowest target
cos or worst pseudo-margin), or grow the competitor set K over epochs (curriculum). Composes with A1.

**Why it survives.** Untested; addresses the documented overfitting of `cos` to the calib set
(finding 3) by attacking the hard tail.

**Paper value.** Medium (a refinement, not a headline).

### A8. Bilevel min–min trigger–target optimisation (framework narrative)

**Mechanism.** Alternate: k steps on the trigger δ (Phase-1 loss) ↔ 1 step on the target direction
`t*` (see B-series), each in a trust region. Framing echo of BadCLIP++ Eq. (3) min-min, adapted:
their `(Θ*, δ*)` becomes our `(δ*, t*)` since Θ is frozen/quantised and the model-space damage is
done later by the bit flips.

**Paper value.** High as *framing*: "joint trigger–target min-min synthesis under a frozen encoder"
gives the method section a principled skeleton for whatever A/B components are adopted.

---

## 4. Brainstorm — t_target fine-tuning

### B1. Prompt-distribution target (per-sample target description) ★ cheap, ledger-approved

**Mechanism.** Instead of `t_target = mean_k t_k` (13 templates), keep the full template set
`{t_k}` and use `L = −E_k cos(emb, t_k)` (or sample `t_{j(i)}` per image per step) — exactly
BadCLIP++ Eq. (10), where the per-image random target description is the working mechanism, and
exactly the "prompt-distribution CE" idea the loss study lists as **not tested**.

**Why it survives.** The mean of 13 template vectors is not itself a natural text-embedding
direction (it can sit between modes); optimising toward the whole target *distribution* should
improve prompt-robustness (there is an `main_new_pipeline_ood_prompts.py` evaluation lineage to
measure against). No images enter — the image-anchor failure mode is untouched. Class-list-free.

**Also for CE-family arms:** if A1's softmax is adopted, put the per-sample sampling *inside* it.

**Paper value.** High: named component ("Target-Distribution Alignment"), ties into an OOD-prompt
robustness table for free.

### B2. Prompt-barycenter refinement — learnable template weights ("literal t_target fine-tuning")

**Mechanism.** `t*(α) = normalize(Σ_k α_k t_k)`, α on the simplex (13 template weights, or a larger
target-only bank: LLM-paraphrased descriptions à la BadCLIP's "multiple banana descriptions").
Optimise α (outer loop of A8, or grid/coordinate search) for the post-trigger margin proxy on the
calibration set. `t_target` stays **text-only** → the Phase-4 `target_vec` consistency issue is
moot (any vector equal to `text_emb[tc]`... note: it will *not* equal `text_emb[tc]` any more, so the
`search.py` drift vector **must** be passed the refined `t_target` — the documented hard requirement).

**Why it survives.** No image mixing (unlike the failed 0.4/0.6 anchor); the modality gap only
appears as a *selection signal* at most.

**Paper value.** High: a "target-side optimisation" chapter fills the currently-static target story;
α visualisation per dataset is a nice figure.

### B3. Image-guided template reweighting (images as selectors, not anchors)

**Mechanism.** Use the target-class image prototype `v̄*` **only to rank/reweight templates**:
`w_k ∝ cos(t_k, v̄*)`, `t_target = normalize(Σ w_k t_k)`. The direction remains in text space.

**Why it survives.** This is the explicit *"new justification"* the ledger demands before touching
image information: the anchor A/B failed because image *content* pulled the loss direction into the
image cone (cos gap 0.31 → fighting terms, eval misalignment). Here images choose among **text
directions**, so the ASR criterion stays aligned and no cone conflict can arise. Directly addresses
all four risks listed in `t_target_plus_image.md`.

**Paper value.** Medium-high; and it *rehabilitates* the negative A/B as a motivating analysis
("why naive image anchoring fails and selection works") — good paper content either way.

### B4. Multi-prototype target + centroid-hinge compactness (BadCLIP++ T2T+MPE, done *correctly*)

**Mechanism.** Two separate options:

1. **Target side:** `t_target → {t^(1)..t^(K)}` prototypes (k-means over template/paraphrase
   vectors, or template + image-mean per mode); Phase-1 loss = soft-min/logsumexp over prototypes.
2. **Trigger side (the BadCLIP++ pair):** `L_T2T = mean‖emb_i − v̄trig‖²` **plus**
   `L_MPE = max(‖v̄trig − v̄*‖ − ε, 0)²` (distance hinge on the **centroid**, ε-soft) **plus** `L_I→T`.
   BadCLIP++ Lemmas 1–2 are exactly why the pair works and the solo terms don't: T2T contracts
   radius (direction free), MPE pins the centroid (radius free) — together both.

**Why it survives (with a big caveat).** The ledger's T2T negative was **free-direction variance
without a centroid anchor**, and the ledger's MPE negative was a **cosine-threshold hinge (dead at
0.4)** — neither is the BadCLIP++ form. But the burden of proof is high: run the ablation pair
(T2T alone → expect failure again → T2T+MPE-hinge) so the paper can *show* the mechanism rather
than assert it. The image-centroid `v̄*` question is delicate (image-anchor history): BadCLIP++
aligns the *cluster centroid* to `v̄*` while keeping per-sample `L_I→T` on text — if the modality gap
fights again, substitute the **text-side centroid** `t_target` in the MPE hinge and note the deviation.

**Paper value.** Very high if it works (a direct, theory-backed upgrade to the strongest published
trigger recipe); even the negative ablation is publishable analysis.

### B5. Probe-aligned target refinement (transfer-aware)

**Mechanism.** For the LP-transfer headline (BadCLIP++ Tab. 2: BadCLIP++ 87.4 % vs BadCLIP 76.0 %
on ImageNet LP-ASR), refine `t_target` toward the **linear-probe head's target-class weight
direction** (discriminative, image-derived but already in classifier space — not a raw image mean),
or toward the direction maximising linear separability of the poisoned calibration embeddings.

**Why it survives.** New justification category: the failed anchor used an *undiscriminative*
prototype mean; a probe weight encodes class-boundary information. zs criterion stays fixed, so
reporting both zs-ASR and lp-ASR keeps the comparison honest.

**Risks.** Optics of "attacker knows the downstream probe" — frame as transfer-aware design with the
probe trained on public data (the LP task already trains heads on ImageNet1kTrain).

**Paper value.** Medium-high for the transfer chapter.

### B6. Margin-anchored target refinement (consistency-safe)

**Mechanism.** Treat `t_target` as a free unit vector optimised for the *post-trigger argmax margin*
against the fixed eval text matrix (bilevel with A8): the outer objective is
`margin(emb_po) = cos(emb_po, text_emb[tc]) − max_j cos(emb_po, text_emb[j])` — note this uses the
**eval class list only in the outer objective**, where the attacker may legitimately know the
deployment label space... or keep it class-list-free by using A1's pseudo-competitors here too.

**Consistency requirement (hard):** propagate the refined vector to Phase 3's `rank`,
`rank_fallback`, Phase 2's `_make_target_cos`, and Phase 4's `target_vec = text_emb[tc]` drift
vector (`t_target_plus_image.md`'s hard requirement), while evaluation remains argmax over
`text_emb`. If this bookkeeping is not wanted, prefer B1–B3.

**Paper value.** Medium; mostly a mechanism to *compose* other components.

---

## 5. Paper-representation ideas (no method risk)

| ID | Idea | Source |
|---|---|---|
| C1 | Port the BadCLIP++ evaluation battery to the thesis tables: detection DSR/DM, inference-time AUROC, data-filtering TPR/FPR, safe-training, cross-model (SigLIP adapters exist), distribution shift | BadCLIP++ Tab. 1–8 |
| C2 | LP-transfer metrics **MCR / ASR-Proxy / D-ASR** for no-target datasets — directly extends `--task lp` reporting and the future `--task i2t` | Tab. 2 semantics |
| C3 | Stealth figure: DECREE-inverted trigger norms + L-norm statistics (Fig. 3); compactness t-SNE/PCA with ellipse/hull areas + MeanDist/Wass(Proj) (Fig. 6, Fig. 9) — `analysis.py`/`visualization.py` already have hooks | BadCLIP++ Fig. 3/6/9 |
| C4 | Flatness evidence: Hessian top-eigenvalue (HVP + power iteration) of the backdoor loss around the attacked model, comparing trigger recipes (Fig. 7) — a curvature narrative fits a BFA "persistence under defence" chapter too | BadCLIP++ Fig. 7, Lemma 3 |
| C5 | Mini-theory section in BadCLIP++ §5 style: (i) proposition linking A1's softmax margin to flips-to-95 (hitting-time of the greedy search), (ii) gradient-geometry argument for why Phase-3's poison−clean gradient difference isolates backdoor-specific bits (their "gradient co-directionality" flavour), (iii) trust-region framing for the fine-tuning defence results (FT defence already implemented) | BadCLIP++ §5 |
| C6 | Incremental component-ablation table in BadCLIP++ Tab. 9 style (row-per-component, No-Defence / FT / CleanCLIP columns) — mostly exists via `--variant a/b/c/d` + `defence_finetune.py`; add trigger/t_target components as rows | Tab. 9 |
| C7 | Complexity-vs-robustness framing: BadCLIP++'s own ablation shows each mechanism trades raw ASR for persistence (TI drops to 63 %, recovers via T2T+MPE) — a good narrative for why a "more complex" trigger/target is justified even if headline ASR is flat | Tab. 9 |

---

## 6. Constraints every proposal must respect

1. **Class-list-free claim.** Competitor sets must be data-derived (calibration embeddings /
   centroids) or a clearly generic vocabulary; document the threat-model argument. CE remains the
   threat-model-relaxed *upper bound* reference arm only.
2. **t_target consistency.** Any `t_target ≠ text_emb[TARGET_CLS]` must be propagated to Phase 2A/2B
   (`_make_target_cos`), Phase 3 (`rank`/`rank_fallback`) and Phase 4 (`search.py`'s
   `target_vec = text_emb[tc]`), or ranking and search optimise different directions
   (`t_target_plus_image.md`).
3. **Evaluation criterion stays argmax over `text_emb`** (zs) and the shared probe (lp) — refinement
   never touches the reported decision rule.
4. **Do not re-propose:** single-argmax hinge competitors (Arm C), free-direction T2T alone,
   cosine-threshold MPE at 0.4, raw image-prototype mixing into `t_target`, `−‖∇cos‖²`
   sensitivity. Each has a documented failure; each surviving variant above is *different in the
   exact way the failure analysis prescribes*.
5. **Metric discipline:** decide on the 2 000-image test column (noise floor ±3–4 pts on 200),
   multi-seed for anything < 5 pts, 1-flip differences are not signal. Keep the tuned Phase-1
   regime (init 0.1, clamp ±3, lr 0.02, 200 ep, N_CALIB=64) unless the change *is* the experiment.
6. **Check `cos_eot` history** (`main_stage_margin_v3.py`, `sweep_bc.py`) before any EOT claim (A3).
7. **DTD 8 px trigger policy** for DTD runs (16 px saturates); trigger-size policy per `run_all.sh`.
8. **Modular-swap rule:** each idea should be implementable as a swap of exactly one phase method
   (`TriggerGenerator.generate`, `TargetEmbedder.from_text`) — the codebase's module-independence
   rule; bilevel ideas (A8/B6) need only a thin outer loop around those two methods.

---

## 7. Prioritisation (impact × risk × effort)

| Priority | Ideas | Notes |
|---|---|---|
| **P0 — do first** | **A1** (+A2), **B1**, **A6** | High impact, low risk, cheap; each directly answers a documented gap (softmax mechanism / untested prompt-distribution / calibration composition) |
| **P1 — strong second wave** | **A3** (after cos_eot check), **B3**, **B4** (with the pairing ablation) | Robustness tables + target-side story + the "correct T2T+MPE" bet |
| **P2 — thesis-unique novelty** | **A5** (quant-aware / flip-robust / margin-sens), **A8/B6** bilevel | Highest novelty; higher implementation risk; A5(3) has a prescribed safe path |
| **P3 — stealth & representation** | **A4**, **C1–C7** | Mostly evaluation/figures/theory; A4 gives the stealth chapter |
| **Selective** | **B2**, **B5** | Good components once P0/P1 land; B5 mainly for the transfer chapter |

### Suggested minimal "paper method" package

> **Target-distribution alignment (B1) + Competitor-normalised trigger loss with margin (A1/A2) +
> Greedy target-aligned calibration selection (A6)**, optionally EOT (A3) for the robustness table,
> presented inside the min-min trigger–target framing (A8), with the BadCLIP++ evaluation battery
> (C1–C4) and a margin/hitting-time mini-theory (C5).

This package is entirely class-list-free, composes with every other phase unchanged, each component
has a one-method implementation point, and every piece has a named mechanism + an ablation row —
exactly what makes a method section read as "stronger and more complex" without re-running any
falsified idea.
