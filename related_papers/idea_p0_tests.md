# idea_p0_tests.md — Two-Arm Test Spec (P0-POOL vs P0-MANNEG)

**Date:** 2026-11-20
**Status:** test design (no code changed yet). Confirmed with maintainer 2026-11-20.
**Dataset:** **ImageNet-1k** (registry `imagenet1k`, val split, zero-shot over the 1 000 text rows).

## 0. Confirmed decisions

| Question | Decision |
|---|---|
| What is tested | (1) P0 method with **pool size 256**; (2) **no pool** — random carriers + **manually curated near-target negatives** (~100 images) as competitors |
| Reason for arm 2 | pool selection + splitting into two distributions may hurt the threat model; the attacker "selecting images close to its target" is an extra capability |
| Negative image source | same dataset (ImageNet classes of confusable concepts); **neighbour class lists drafted here** for approval. **2026-11-21: val split + exclusion** — the 100 negatives are sampled from val and held out of every split in every arm (no train split needed) |
| Competitor form | **both** as an ablation: raw-embedding memory bank (default) + k-means centroids |
| Shared components | TDA (B1) + margin (A2) present in **both** arms; the arms differ only in carrier selection and competitor source |
| Baseline | existing plain-`cos` behaviour (arm A0) for reference |

---

## 1. Arms overview

| Arm | Carriers (Phase-1 loss images) | Competitors `{μ_r}` | Pool needed | Purpose |
|---|---|---|---|---|
| **A0** | random 64 non-target | none (plain `cos` loss) | – | incumbent reference (current runs) |
| **P1** (arm 1) | **GMAS**-selected 64 from **pool of 256** | k-means (`M=16`) over pool remainder (192) | 256 victim images | P0 with the smaller pool |
| **P2** (arm 2) | **random 64** non-target (status quo) | **memory bank of 100 curated negative embeddings** | none | threat-model-safe P0 variant |
| **P2-cen** | random 64 | k-means (`M=16`) over the same 100 negatives | none | competitor-form ablation (confirmed) |
| **P2-swap** | random 64 | 100 negatives from the **swap list** (§4.3) | none | list-sensitivity check |
| **P2-txt** *(optional)* | random 64 | text embeddings of the 30 neighbour class names | none | image-cone vs text-cone competitor ablation (§9) |

Both P1 and P2 use the identical CNL+m + TDA Phase-1 loss (§3 of `idea_p0.md`) and the identical
Phase-1 recipe (init 0.1, Adam 0.02 cosine, 200 ep, grad clip, clamp ±3), so the only variable is
**where the competitors and carriers come from**.

---

## 2. Arm P1 — P0 with pool 256

**Data flow (deviation from `idea_p0.md` §1 only in `N_POOL`):**

```
other_indices (shuffled, non-target)
 ├─ P   = pool[0 : 256]                    # was 512
 │     ├─ S_calib = GMAS(P, t̄, 64)         # carriers
 │     └─ D_comp  = P \ S_calib (192)      # competitor mining only
 ├─ grad_idx  (16)   ├─ eval_idx (200)   └─ test_idx (rest)
```

**Competitors:** k-means (k-means++/Lloyd/50 iters) over `{e(x) : x ∈ D_comp}` → `M = 16`
normalised centroids; ablate `M ∈ {8, 16, 32}` if the arm is promising.
**Everything else:** exactly `idea_p0.md` (GMAS objective, CNL+m formula, TDA sampling, recipe).

**Watch-item (the hypothesis under test):** does shrinking the pool 512 → 256 degrade competitor
quality (192 mining images) more than GMAS carrier quality? Report `diag-margin_pseudo` to see
which side moves.

---

## 3. Arm P2 — no pool, manual negatives (the threat-model-safe variant)

**Data flow:**

```
other_indices (shuffled, non-target)
 ├─ calib_idx = 64 random           # EXACTLY the current pipeline's rule — no pool, no selection
 ├─ grad_idx (16) | eval_idx (200) | test_idx (rest)
N_neg = 100 images from curated neighbour classes (VAL split, seeded,
       HELD OUT of grad/eval/test in every arm) → competitor bank
```

**Competitor bank (default = memory bank):**

```
C = { normalize(e(x)) : x ∈ N_neg }           # M = 100 raw embeddings, no clustering
```

**Centroid ablation (P2-cen):** k-means the same 100 → `M = 16` normalised centroids.

**Loss (identical in P1/P2):** CNL+m with TDA — `idea_p0.md` §3 Phase-1 formula, competitors `= C`.

**Why this arm is the claim-safe default (for the paper):**

1. Carrier selection stays at the status-quo capability ("64 random clean images") — no semantic
   targeting of carriers, no victim-dataset screening beyond what the baseline already assumes.
2. The competitor set needs **no victim data at all** in principle: knowing that *dog / wolf / lion*
   are confusable with *cat* is knowledge about the **target concept**, obtainable from the
   attacker's own world knowledge; the images can be generic. (In this test they are sampled from
   the val split of the listed classes and **held out of every evaluation split** — §4.4.)
3. No pool splitting → no "two distributions" artifact, and the competitor set no longer encodes
   the deployment label distribution (removing the soft class-list side channel of mined centroids).
4. **Mechanistic fit:** the softmax denominator's gradient mass concentrates on hard negatives, and
   the final argmax (over the 1 000 text rows) is decided by the *nearest* class directions —
   margin against dog/wolf/lion directions is exactly the binding constraint, where random-pool
   centroids spend budget on easy far modes.

**Hypothesis (falsifiable):** P2 ≥ P1 on final ASR and flips→95 despite strictly weaker data
assumptions; if instead P1 wins clearly, the paper reports the pool approach with an explicit
capability disclosure (BadCLIP++ assumes full pre-training-data control — legal in *their* threat
model, arguable in ours).

---

## 4. Target and neighbour lists (ImageNet-1k) — PROPOSED, for approval

Class names verified against `class_texts/imagenet1k/classes.txt` (BadCLIP's verbatim list —
note its quirks: **"Egyptian Mau"** not "Egyptian cat", **no "wood mouse"** class, "Maltese",
"taxicab"). Indices resolve at runtime from `classes.txt`; never hardcode.

### 4.1 Primary target: **T1 = "tabby cat"**

Neighbours, 100 images total (30 classes):

| Tier | Why these | Classes | imgs/class | Subtotal |
|---|---|---|---|---|
| Feline core | hardest confusables (same super-class, fur/shape) | `tiger cat`, `Persian cat`, `Siamese cat`, `Egyptian Mau` | 8 | 32 |
| Big cats | target-adjacent markings (stripes/rosettes) | `lynx`, `leopard`, `snow leopard`, `jaguar`, `cougar`, `cheetah`, `lion`, `tiger` | 3 | 24 |
| Canids | user's "dog/wolf" examples | `grey wolf`, `Alaskan tundra wolf`, `coyote`, `dingo`, `red fox`, `kit fox`, `Arctic fox`, `grey fox` | 2 | 16 |
| Small mammals | user's "mouse" example (no true mouse class; nearest granivores) | `fox squirrel`, `hamster`, `guinea pig`, `marmot`, `beaver`, `weasel` | 2 | 12 |
| Companion dogs | small pet confusables | `Chihuahua`, `Maltese`, `Pomeranian`, `Toy Poodle` | 4 | 16 |
| **Total** | | **30 classes** | | **100** |

### 4.2 Secondary target (generality, second wave): **T2 = "sports car"**

| Tier | Classes | imgs/class | Subtotal |
|---|---|---|---|
| Car core | `convertible`, `limousine`, `minivan`, `jeep` | 8 | 32 |
| Road vehicles | `taxicab`, `pickup truck`, `tow truck`, `garbage truck` | 4 | 16 |
| Recreational | `recreational vehicle (RV)`, `moving van`, `beach wagon`, `station wagon` | 4 | 16 |
| Marine/air confusables (shape) | `speedboat`, `canoe`, `catamaran`, `trimaran` | 4 | 16 |
| Misc. hard | `sports car` variants via `racer`-style classes if present; else `fire engine`/`forklift`-tier substitutes — **resolve exact names from `classes.txt` before the run** | 4 | 20 |
| **Total** | | | **100** |

*(Alternatives if preferred: `lion` — continuity with the Arm-C "lion attribution" history;
`banana` — BadCLIP++ parity target. Say the word and I swap T2.)*

### 4.3 Swap list for P2-sensitivity (P2-swap)

Replace the canid tier 1:1 with `dhole`, `African wild dog`, `hyena`, `red wolf or maned wolf` and
the small-mammal tier with `skunk`, `mink`, `porcupine`, `European polecat` (same counts). If ASR
and margin move ≪ noise floor, the manual list is not a cherry-picking artefact.

### 4.4 Image sampling rule (deterministic)

**Maintainer decision 2026-11-21: VAL split + exclusion** (no train split required).

- Source: the already-loaded **val split** of the listed classes.
- The 100 negatives are sampled **first** (seeded shuffle, `SEED+7`, `n_c` per class, §4.1 counts)
  and then **held out of every split** (grad / eval / test / scratch) in *every* arm — so there is
  zero evaluation leakage and a0/P1/P2 all measure on identical grad/eval/test sets. (The only
  nuance: P2-swap's list differs from P2's in 28 of the 100 held-out images → a28-image difference
  in the ~47k test set, far below noise.)
- Per class: seeded shuffle, take the first `n_c` images.
- Same 100 images for P2, P2-cen and P2-txt (only the competitor *form* differs).

---

## 5. Evaluation protocol

| | |
|---|---|
| Models | smoke: **ViT-B/16 × ImageNet1k**; headline: **ViT-L/14 × ImageNet1k** (grid flagship) |
| Seeds | 3 (42 / 43 / 44); single-seed for smoke arms |
| Trigger | registry policy: 16 px @ 224 (`TRIG_STD`), center_add, clamp ±3 |
| Split sizes | N_CALIB=64, N_GRAD=16, N_EVAL=200, test = remainder of val (per `data.py`) |
| Metrics | standalone ASR (eval + **test column**), `diag-margin_pseudo`, `diag-margin_classes`, `cos_poison`, **flips→95**, final ASR/CACC, search time |
| Noise floor | decide on the test column; ±3–4 pts on 200-image eval is not signal; 1-flip differences are not signal |

**Success gates:**
1. P2 ≥ P1 on final ASR **and** flips→95 (threat-model claim validated *and* performance win).
2. P2's Phase-1 ends with `diag-margin_classes > 0` (the boundary no `cos`-family arm crossed).
3. P2-swap within noise of P2 (list robustness).

---

## 6. Run matrix (after implementation)

```bash
# as implemented — single file related_papers/p0_test.py (ViT-L/14 × ImageNet1k);
# results land under related_papers/p0_results/... (never grid_results)
# smoke: --seed 42 first, then headline seeds 42/43/44 for P1 and P2
python related_papers/p0_test.py --arm p1                                  # P1  (pool 256)
python related_papers/p0_test.py --arm p2   --neg-list tabby_v1            # P2  (100-img bank)
python related_papers/p0_test.py --arm p2cen --competitor-m 16             # P2-cen
python related_papers/p0_test.py --arm p2swap                              # P2-swap
python related_papers/p0_test.py --arm p2txt                               # P2-txt (optional)
python related_papers/p0_test.py --arm a0   --seed 42                      # reference baseline
# headline, seeds 42/43/44 for P1 and P2 (winners only)
```

(A0 is re-run with this harness's shared split layout so every arm is internally comparable;
old grid_results runs keep their historical splits.)

---

## 7. Implementation checklist (small diffs, per AGENTS.md conventions)

**AS IMPLEMENTED (2026-11-21): one file, no repo code touched** —
`related_papers/p0_test.py` wraps the untouched `modular_new_pipeline` and swaps exactly three
classes at runtime (`pipeline.DataModule`, `pipeline.TargetEmbedder`, `pipeline.TriggerGenerator`):

| Swap point | Implements |
|---|---|
| `P0DataModule` | shared split layout (identical grad/eval/test per seed), GMAS pool selection (P1), the 100 val negatives sampled first and **held out of every split** (§4.4) |
| `P0TargetEmbedder` | captures model/classes (needed by GMAS + competitor mining) — `from_text` unchanged |
| `P0TriggerGenerator` | CNL+m + TDA loss, competitor bank (bank / centroids / text), margin/temperature/tda flags; recipe/clamp/retry unchanged; logs the gradient-scale parity check |
| CLI | `--arm {a0,p1,p2,p2cen,p2swap,p2txt}`, `--neg-list`, `--target-name`, `--p1-margin`, `--p1-temperature`, `--tda`, `--n-pool`, `--competitor-m`, `--dry-run`; `experiment.py` flags pass through |
| Results | `related_papers/p0_results/{section}/{model}/{dataset}/…` (+ `runs.csv`/`summary.csv`) — never `proposed_method/grid_results` |

Phases 2A–5 untouched; `t̄ = text_emb[TARGET_CLS]` flows to Phases 2–4 exactly as today (no
Phase-4 consistency hazard).

---

## 8. Threat-model text (draft for the paper)

> The attacker knows the target class name `c*` and possesses a small set of generic clean images
> (the 64 calibration carriers). Two competitor sources are considered: (i) pseudo-classes mined
> from a victim-distribution image pool (stronger data assumption; reported for reference), and
> (ii) **confusable-concept negatives curated from the attacker's knowledge of the target** (e.g.,
> dog/wolf/lion for cat) — requiring no victim data and no label space. All Phase-1 objectives are
> class-list-free; the deployment class list is used only at evaluation.

---

## 9. Notes, risks, optional extras

- **P2-txt (optional 6th arm):** competitors = multi-template text embeddings of the 30 neighbour
  names. Zero images, exact eval geometry (text cone), but it reintroduces "text that isn't the
  target" — keep it an appendix ablation, not the headline.
- **Modality gap:** P2 competitors live in the image cone while the argmax runs over text rows
  (measured gap ~0.3 in the anchor study). P2-txt exists precisely to measure this effect.
- **Manual-list optics:** mitigated by (a) the §4.3 swap test, (b) tier rationale table, (c) the
  argument that confusable-concept knowledge is target-side knowledge, not victim data.
- **100 vs 16 competitors:** memory bank gives M=100 softmax terms (~free); if gradient noise from
  the hard-negative mass is an issue, P2-cen's M=16 centroids smooth it.
- **Interpretation guard:** if P2 ≈ P1 ≈ A0, the conclusion is that competitor geometry is
  second-order on ImageNet and the paper leans on TDA+margin instead — all three outcomes are
  reportable.
