# Baseline mismatch report — paper vs released code vs this suite

**Scope:** ProFlip (ICCV 2021), TrojViT (CVPR 2023), Flip-S (CVPR 2025),
compared against the implementations in `bfa_baselines/`.

**Sources used for this report**

| Baseline | Paper | Original code | Our code |
|---|---|---|---|
| ProFlip | `06.Bit-Flip Attack Clip/2021 - ProFlip ... .pdf` | `baselines/ProFlip/resnet32-cifar10/` | `bfa_baselines/attack_proflip.py` |
| TrojViT | `06.Bit-Flip Attack Clip/TrojViT ... .pdf` | `baselines/TrojViT/main_patch_vit.py` | `bfa_baselines/attack_trojvit.py` |
| Flip-S | `06.Bit-Flip Attack Clip/Flip-S.pdf` | `baselines/Flip-S/target-imagenet/*.py` | `bfa_baselines/attack_flips.py` |

**Legend**

- ✅ — same behaviour (paper / release / ours agree)
- ⚠️ — deliberate adaptation or documented deviation; keep, but report it
- ❌ — undocumented deviation or paper-fidelity gap; decide before the thesis
- 🐞 — bug in the released code that our code fixes (or a bug in ours)
- 🚨 — interpretation/fairness problem that changes the conclusion, not just fidelity

**The single most important distinction:** “faithful to the paper” and
“faithful to the released code” are **not the same** for these three baselines.
Several release-code bugs are deliberately not copied by our implementations,
and several paper details are not present in the release. Each row below says
which source we match.

---

## 0. Executive summary

| Baseline | Paper-native budget | Release hard-code | Suite “native” | Push-98 | Biggest issue |
|---|---|---|---|---|---|
| ProFlip | `ASR_t = 94`, `n_max = 100` | `while n_b < 50` | 94 / 100 (documented) | 98 / 200 | S1 re-ranks layers, S2 takes top-8, acceptance is ASR-gated, CBS anchor is clean vs release self-anchor; quick-ASR trace is on the wrong split |
| TrojViT | **TBN 880–1250** | no bit-flip realisation | **80 (wrong)** | 200 | 200-bit cap is 4–6× below the paper’s native TBN; `flips_to_98 = null` is budget-limited, not a method ceiling |
| Flip-S | 50 (relaxed 500) | 50 | 50 | 200 | Attack surface includes the projection/head; ranking `\|grad·Δbit\|` not `\|grad\|`; ASR-gated fallback breaks the trap escape; attention loss uses only the last term |

**What is safe to say in the thesis**

- ProFlip: the two-phase structure and CBS formula are ported, but the current
  search policy is **not** the paper’s or the release’s; it must be labelled a
  variant or given a paper-faithful preset.
- TrojViT: the current 200-bit result **cannot** be presented as TrojViT’s
  ceiling; its native TBN is 880–1250.
- Flip-S: the core SFS is close, but the extra projection surface, the
  ASR-gated fallback, and the last-term attention loss can all change the
  result; the paper’s native 50-bit budget is already enough for 98 % if the
  port is faithful.

---

## 1. Cross-cutting mismatches

These apply to more than one baseline and are the most likely source of
reviewer objections.

### 1.1 Trigger area (TAR) is not matched

The proposed method uses a **16×16** trigger (0.51 % of a 224² input). The
baselines use much larger native triggers:

| Method | Trigger | Area at 224² / patch 16 | Area at 224² / patch 14 |
|---|---|---|---|
| Proposed method | 16×16 | **0.51 %** | 0.51 % |
| TrojViT | 9 patches | 4.59 % | 3.52 % (9×14²) |
| Flip-S | 3×3 patch block | 4.59 % | 3.52 % (42²) |
| ProFlip | 70×70 square | **9.77 %** | 9.77 % |

Consequences:

- A larger trigger raises `trigger_asr_pre` and reduces the number of flips
  needed to reach 98 %. The baselines are **advantaged** by 7–19× more trigger
  area.
- This is visible in the TrojViT results: ViT-B/16 `trigger_asr_pre = 83.6 %`
  with a 4.59 % trigger; the paper’s ViT-b trigger-only ASR is **6.67 %**.
- The paper’s own Flip-S TAP ablation (Table 6) shows ASR drops when the
  trigger shrinks, so a 3.52 % vs 4.59 % difference on ViT-L/14 is material.
- For a fair “flips to 98 %” table, report **TAR and trigger-only ASR for
  every method**, including the proposed method. A matched-TAR row (e.g.
  0.51 % for all) is a useful secondary comparison, though it departs from
  each baseline’s native setting.

### 1.2 Budgets

| Method | Paper native | Release hard-code | Suite “native” | Push-98 | Verdict |
|---|---|---|---|---|---|
| TrojViT | **TBN 880** (DeiT-s/ImageNet), **1250** (ViT-b/ImageNet); 492–1135 on CIFAR-10 | none (no bit-flip code) | **80 (wrong)** | 200 | 🚨 under-budget: 200 is 4–6× below native |
| ProFlip | `ASR_t = 94`, `n_max = 100` | `while n_b < 50` | 94 / 100 | 98 / 200 | 200 is above native; 98 % is an extension of the paper’s 94 % target |
| Flip-S | **50** default; relaxed to **500** | 50 | 50 | 200 | fine: 200 is within the paper’s relaxed range |

The “80” TrojViT native value is almost certainly a **units error**: the paper’s
closest number is `TPN = 84` at `e = 0.003` (weights), whose **TBN is 345**, not
80. At the paper’s default `e = 0.0005`, `TPN = 213` and **TBN = 880**.

### 1.3 Data protocol

| Method | Paper data | Release data | Suite data |
|---|---|---|---|
| ProFlip | 256 **train** samples (SNI, trigger, CBS) | 128 train for SNI; 128 **test** for trigger; 32 test batches for CBS | calib 384 (trigger), search 256 (CBS), aux 32 (acceptance), eval 1000 |
| TrojViT | 384 **test** images (trigger + insertion); ASR on test set | **16** test images for trigger (the 10 % loop break); **16** test images for insertion; ASR on full 50 k val | calib 384 (trigger + insertion), eval 1000 |
| Flip-S | 32 **test** samples (trigger + SFS) | 32 aux for trigger + SFS | aux 32 (trigger + ranking), search 256 (verification), calib 384 (`trigger_asr_pre`), eval 1000 |

Our frozen splits are disjoint from the 1000-image eval set; this is a fairness
improvement (the release’s TrojViT/proflip trigger data overlaps the test set,
and TrojViT’s trigger is 16 images, not 384). It is a **protocol deviation**,
not a bug. Report it as such.

### 1.4 Metrics and splits

- **ASR definition.** Ours excludes images whose clean label is already the
  target (`asr`), and also reports `asr_incl_target`. The papers/release count
  all triggered images (labels overwritten to the target). The difference is
  small (`clean_as_target` ≈ 0 in the runs so far), but report both.
- **Eval size.** Papers evaluate on the full test set; ours on 1000 frozen
  eval images. Fine, but state it.
- **`ASR@K` / `flips_to_*` split.** The suite convention is “first
  `--quick-n = 256` search images”. Two implementations break it:
  - **ProFlip** uses the **32-image aux** split for the trace/stop, while
    `trigger_asr_pre` uses search 256 and the meta says
    `quick_split = "search[:quick_n]"`. 🐞 `flip_contribution` is invalid and
    98 % on 32 images is only reachable at 100 %.
  - **Flip-S** uses **calib 384** for `trigger_asr_pre` but **search 256** for
    the trace; the meta has no `quick_split` at all. ⚠️ Record both splits.
  - **TrojViT** uses calib for both, so it is internally consistent.
- **Trigger-only ASR.** The paper values are: TrojViT 6.67 % (ViT-b) / 31.23 %
  (DeiT-s); Flip-S ≈32–38 % (Table 5, 4-bit); ProFlip not reported. Our
  TrojViT values are 83.6 % (ViT-B/16) and 24.7 % (ViT-L/14). The ViT-B/16
  case is trigger-saturated; the ViT-L/14 case is closer to the paper.

### 1.5 Quantisation and the meaning of a “bit flip”

| Method | Quantiser | Flip target | What one flip changes |
|---|---|---|---|
| ProFlip | per-tensor int8 (symmetric, `max\|W\|/127`) | weight code bit | one weight |
| TrojViT | 8-bit per-tensor (paper; release has no flip code) | weight code bit | one projection weight (ours) |
| Flip-S | row-wise int8 + **FP16 scale** (bnb `Linear8bitLt`) | scale bit (primary) / weight code bit | **one whole output row** (scale) or one weight |

A Flip-S scale flip is not comparable in impact to a ProFlip/TrojViT weight
bit flip. This is inherent to the methods; do not present “bits to 98 %” as a
like-for-like physical cost without saying which surface was flipped.

---

## 2. ProFlip (ICCV 2021)

### 2.1 What the paper specifies

- **Three stages:** SNI (JSMA on the input of the last layer, θ=0.1, γ=0.5,
  target class 2, **intersection** across images) → trigger generation →
  CBS.
- **Trigger:** bottom-right square; TAP **9.76 %** on CIFAR-10/SVHN and
  **10.62 %** on ImageNet (≈73²); replace mode; `L_trig = MSE(M_{1:L-1}(x̂); c)
  + CE(M(x̂); t)`, λ1 = λ2 = 1, **c = 10**; 256 clean **training** samples.
- **CBS (Algorithms 2–3):**
  - **S1 is a one-time, offline process**: `p_sens ← select attack param(M, D)`
    before the loop (Alg. 2 line 2). Fitness
    `F = |∂L_CBS/∂w| · (Q_p − w)` for negative gradients, `Q_p = max|W|`
    (“maximal magnitude”, Alg. 3 input + Eq. 2).
  - **S2 selects exactly one element per iteration**:
    `elem = argmax F(p_sens, elem)` (Eq. 9; Alg. 2 `ne += 1`).
  - **S3:** divide `[-R, R]` (`R = max|W|`) into **K = 20** points, evaluate
    `L_troj` on each, take the **argmin unconditionally** (no acceptance gate,
    no monotone constraint).
  - `L_troj = γ1·CE(M*, D) + γ2·L_CBS`, γ2 = 1, γ1 set so
    `γ1·L_ce ≈ 0.1·γ2·L_CBS`.
  - `n_b = HammingDistance(b, b*)`; multi-bit element updates are atomic.
- **Stop:** `ASR_t = 94`, `n_max = 100`.
- **Headline:** ResNet-18/ImageNet, **15 bits → 94.3 % ASR**.

### 2.2 What the released code does

File: `baselines/ProFlip/resnet32-cifar10/CSB_nonadaptive.py` (+
`trigger_nonadaptive.py`).

| Stage | Release behaviour | Line |
|---|---|---|
| SNI | JSMA loop; **union** of per-image sets, not intersection | `trigger_nonadaptive.py:423–474` (union at 470) |
| Trigger data | one **128-image test** batch (the `loader_test` batch is reassigned at 485–488) | `trigger_nonadaptive.py:485–511` |
| Trigger update | 10 steps, lr 0.01, gradient **summed** over the batch, no clamp (normalised space) | `trigger_nonadaptive.py:497–511` |
| Clamp `c` | literal `y[:, I_t] = 10` | `trigger_nonadaptive.py:491` |
| S1 | `psens = find_psens(...)` **once**, before the loop | `CSB_nonadaptive.py:494` |
| S1 fitness | `Q_p = max(w)` **signed**; only `grad < 0` | `CSB_nonadaptive.py:128–134` |
| S2 | **one** element: `fit.index(max(fit))` | `CSB_nonadaptive.py:182` |
| S3 grid | **10 descending float** points `R − k·i`, `k = floor(2R/10)`; no rounding | `CSB_nonadaptive.py:517–524` |
| Monotone | check is **commented out** | `CSB_nonadaptive.py:541–542` |
| Acceptance | always applies the **min-loss** grid value | `CSB_nonadaptive.py:539–553` |
| CBS data | one 32-image test batch (`num` cycles 0..7) | `CSB_nonadaptive.py:496–526` |
| CBS anchor | `y` is recomputed from the **poisoned** batch; only `I_t` gets a gradient (self-anchor elsewhere) | `CSB_nonadaptive.py:199–213` |
| Clean CE | `+ 2·CE(clean, true)` in the candidate loss | `CSB_nonadaptive.py:213` |
| Stop | `while n_b < 50` (ASR_t=90 is defined but unused) | `CSB_nonadaptive.py:510` |
| Bit count | 8-bit two’s-complement Hamming, truncating float candidates | `CSB_nonadaptive.py:449–460` |

### 2.3 What our code does

File: `bfa_baselines/attack_proflip.py`.

| Stage | Our behaviour | Line |
|---|---|---|
| SNI | analytic `top-200 \|P·t_target\|` on the pre-projection dims | 212–231 |
| Trigger | same geometry proportionally (70² at 224), 10 GD steps, lr 0.01, loss `MSE(h, y[I_t]=c) + CE`, mean gradient over 384 calib images, pixel [0,1] + clamp | 264–305 |
| Clamp `c` | `max` over calib (default); `mean3std` / `const10` available | 245–260, 818 |
| S1 | **re-ranks every round**; picks max-F tensor | 527–539 |
| S1 fitness | `Q = max(w)` signed by default (`--q-mode abs` for the paper) | 400–416, 837 |
| S2 | **top-8** elements per round | 541–568 |
| S3 grid | **20 integer codes** over `[-R,R]`, rounded/clamped, **monotone** `new ≥ old` by default | 418–465 |
| Acceptance | candidate accepted iff loss improves **or** quick ASR improves; otherwise stop | 570–610 |
| CBS data | 256 search split for the fitness loss; 32 aux for the acceptance ASR | 477–486, 699–701 |
| CBS anchor | `y_search` from **clean** search features, `I_t = c` | 699–701 |
| Clean CE | omitted in `mse_sni`; `cos` variant also omits it | 347–398 |
| Stop | quick ASR ≥ 98 or bits ≥ 200 or pool exhausted | 479–640 |
| Bit count | int8 Hamming; one flip-log row per bit; whole entry atomic | 418–465 |

### 2.4 Matching parts

- Two-phase structure: SNI + trigger → CBS.
- Fitness formula `F = |∂L/∂w|·(Q−w)` for negative gradients.
- Per-tensor symmetric int8: `step = max|W|/127`, codes `round(w/step)`, ±127.
- S2/S3 concept: pick vulnerable element(s), grid-search values, write the
  new value, flip **all** differing bits atomically, count Hamming bits.
- Trigger geometry (bottom-right, ≈31 % width) and 10-step, lr 0.01 update.
- Trigger loss form `MSE(h, y[I_t]=c) + CE(logits, target)`.
- Stop concept: ASR target or bit budget.
- Attack surface: all encoder weights + the projection (the release also
  includes the classifier head; the CLIP projection is the analogue).

### 2.5 Differences

| # | Difference | Paper | Release | Ours | Status |
|---|---|---|---|---|---|
| P1 | S1 layer selection | **once** (Alg. 2 line 2) | once | **re-ranked every round** | ❌ |
| P2 | S2 element count | **1** per iteration | 1 | **top-8** | ❌ |
| P3 | S3 grid | K=20 over `[-R,R]` | 10 descending floats, no monotone | 20 integer codes, monotone | ⚠️ (#2) |
| P4 | S3 acceptance | argmin unconditionally | unconditional | loss **or** ASR improvement, else stop | ❌ |
| P5 | `Q_p` | `max\|W\|` | `max(W)` signed | default signed; `abs` option | ⚠️ |
| P6 | Clean CE in `L_troj` | `γ1·CE ≈ 0.1·L_CBS` | `+2·CE` | **0** | ⚠️ (#6) |
| P7 | CBS MSE target | `M*` features to `c` | poisoned features; only `I_t` gets gradient | clean features + `c` | ❌ |
| P8 | SNI | JSMA, **intersection** | JSMA, **union** | analytic top-200 | ⚠️ (method table) |
| P9 | Clamp `c` | 10 | 10 | `max` (const10 available) | ⚠️ (method table) |
| P10 | Trigger data/space | 256 train, normalised | 128 test, normalised, sum grad | 384 calib, pixel [0,1], mean grad | ❌ |
| P11 | CBS data | 256 train | 32 test | 256 search + 32 aux | ❌ |
| P12 | Stop/budget | 94 / 100 | 50 hard-code | 98 / 200 | ⚠️ (#1) |
| P13 | Quick-ASR trace split | — | full test | **32 aux**, meta says 256 search | 🐞 |
| P14 | ASR definition | all images | all images (target overwritten) | excludes clean-target images | ⚠️ minor |
| P15 | Trigger TAP | 10.62 % ImageNet (73²) | 9.77 % CIFAR (10²/32²) | 9.77 % (70²) | ⚠️ minor |
| P16 | Target class | 2 | 2 | per dataset (291/97/76) | ⚠️ deliberate |
| P17 | Surface binding | all quantised conv/linear incl. classifier | same | all encoder 2-D weights + projection | ⚠️ necessary CLIP analogue |

### 2.6 Why the differences matter

- **P1/P2/P4** change the search policy, not just plumbing: our search
  re-ranks layers, tries up to 8 elements, and refuses non-improving moves.
  The paper/release pin one layer, take one element, and always apply the
  argmin. The paper’s CBS is therefore **not** the current search.
- **P7** changes the gradient: the release’s `y` comes from the poisoned batch,
  so its MSE is non-zero only on `I_t`; ours anchors all non-`I_t` dims to the
  clean features. Ours matches the paper’s stated `MSE(M*; c)` intent more
  closely, but it is a different objective from the release.
- **P13** is a real defect in our metadata/stop logic: `trigger_asr_pre` is on
  search 256, but `flips_to_98`/`asr_at_k` are on aux 32. On 32 images,
  98 % is only reachable at 32/32 = 100 %. Fix before using these numbers.

### 2.7 Paper-faithful ProFlip preset

```
--cbs-top-elems 1
--cbs-select-once          # new flag: pin S1
--cbs-accept-always        # new flag: unconditional argmin
--grid-k 20                # already
--no-monotone              # paper has no monotone constraint
--q-mode abs               # paper's max|W| (release uses signed)
# optionally: clean-CE term (~0.1 x L_CBS) and poisoned-anchor mode
```

Also fix the quick-ASR split (use `search[:quick_n]` for both the trace and
`trigger_asr_pre`), and correct the `--cbs-top-elems` help text (“paper: 1;
release: 1”, not 8).

---

## 3. TrojViT (CVPR 2023)

### 3.1 What the paper specifies

- **Two phases:** patch-wise trigger generation → tuned parameter distillation.
- **Trigger:** patch salience ranking by `|∂L_CE/∂x|` summed per patch, top-N
  (Eq. 2–3); Attention-Target loss
  `L = L_CE(x̂, y) + λ·Σ_l L_ATTN^l`, `L_ATTN = −log Σ attn(query→T)`;
  λ = 1; PCGrad between the two gradients (Eq. 4–6); **P initialised to 0**;
  9 patches × 16 px = **TAR 4.59 %**; 384 test images, batch 16.
- **Insertion (Alg. 1):** `W_T` initialised to the **last attention +
  classification layers**; objective `L(f(X), y) + L(f(X+M⊙P), y_k)` with
  **GradSurgery**; prune `|W'_T − W_T| < e` each epoch; default
  `e = 0.0005`.
- **Results (ImageNet, Table 5):** ViT-b **TPN 292 / TBN 1250**, ASR 98.82 %;
  DeiT-s **TPN 213 / TBN 880**, ASR 99.96 %. Table 10: `e = 0.003` →
  TPN 84 / **TBN 345**.
- **Trigger-only ASR:** ViT-b **6.67 %**, DeiT-s 31.23 %.

### 3.2 What the released code does

File: `baselines/TrojViT/main_patch_vit.py`.

| Stage | Release behaviour | Line |
|---|---|---|
| Trigger data | subset of the first 384 val images; loop breaks at 10 % of 24 batches → **16 images** (batch 1) | `utils.py:28–33`, `main:229–232` |
| Trigger pattern | **per-image** `delta[16]`, reused by batch position for every test batch | `main:367–370`, `test_patch_tri:334–359` |
| Patch positions | computed for the 16 trigger images, then reused for all test images by batch index | `main:248–285`, `334–359` |
| Trigger loss | `loss_p = -CE`, then `delta.grad = -grad`; Adam 0.22, StepLR 10/0.95, 250 iters | `main:366–440` |
| Attention loss | only if `--attack_mode Attention` (**default is `CE_loss`**); target indices use only the **first** selected patch; loop overwrites `atten_loss`, so only the last per-layer term enters the gradient | `main:288–291`, `395–424` |
| Patch select | Saliency (default), Attn (layer 4), Rand | `main:248–266` |
| NGR | CE on **true labels**; `head.weight.grad.abs().topk(768)` → all 768 columns of the target row (no selection) | `main:293–326` |
| Insertion data | **16 test images** (first batch; `if t == 1: break` before the optimizer step) × 200 epochs | `main:495–532` |
| Trainable scope | only `head` (the `name_list` check uses `name_list[0]`) | `main:477–484` |
| Distillation loss | `0.5·CE(clean) + 0.5·CE(triggered→target)`, no GradSurgery | `main:500–531` |
| Restore | copies the pristine head, writes back only `head.weight[target, tar1]` (bias not restored) | `main:537–554` |
| Pruning | `optimized_wb1 = False` → the `e = 0.003` pruning block **never runs** | `main:535–554` |
| Quantisation / flips | none; the script writes a trojaned model (save commented out); TPN/TBN are external | — |
| ASR | full 50 k val set, per-image trigger reused by batch position | `main:334–359`, `470–471` |

### 3.3 What our code does

File: `bfa_baselines/attack_trojvit.py`.

| Stage | Our behaviour | Line |
|---|---|---|
| Trigger data | 384 calib images (paper value; release uses 16) | `main:772–790` |
| Trigger pattern | **one shared delta** `[1,3,px,px]`; per-image patch positions computed for calib and eval | `generate_trigger:318–394`, `main:795–830` |
| Patch positions at eval | recomputed deterministically for every eval image | `main:806–820` |
| Trigger loss | `L_target = 1−cos` by default (`ce` option, temperature 0.07); `L_ATTN` over all selected patches, mean over layers (skips layer 0); PCGrad per image | `261–394` |
| Trigger init/mode | random init; replace mode | `parse_args:659–663` |
| NGR | CE toward **target class**; score `\|t_target[i]\|·\|\|g[:,i]\|\|`; **top-8** embedding rows × all width entries | `397–446` |
| Insertion data | 384 calib × 200 epochs, mini-batches | `451–520` |
| Trainable scope | `visual.proj` only | `451–460` |
| Distillation loss | `0.5·CE(clean) + 0.5·(1−cos(triggered, t_target))`; **no GradSurgery** | `497–502` |
| Restore | pristine proj, write back only masked entries | `514–517` |
| Pruning | post-hoc `\|Δw\| < 0.003` | `519–521` |
| Bit realisation | int8 Hamming; whole entries applied atomically in descending `\|Δw\|` until 200 bits | `524–595` |
| ASR pre | `trigger_asr_pre` on calib 384 (consistent with the trace) | `830–838` |
| Budget | 200 (push-98) | `parse_args:629` |

### 3.4 Matching parts

- Two phases: patch-wise trigger + tuned parameter distillation.
- 9 patches × 16 px, TAR 4.59 % on patch-16 models; salience = `|grad|` per
  pixel summed per patch.
- Attention-target loss with λ = 1 and PCGrad; 250 Adam iterations, lr 0.22,
  StepLR 10/0.95.
- Insertion optimiser: SGD 0.01, momentum 0.9, weight decay 5e-6,
  MultiStepLR 80/120/160, 200 epochs; 0.5/0.5 clean/triggered; restore
  non-selected weights after every step.
- 8-bit quantisation and Hamming bit counting (ours implements the realisation;
  the release leaves it external).
- 384 trigger/insertion images (paper value; the release uses 16).

### 3.5 Differences

| # | Difference | Paper | Release | Ours | Status |
|---|---|---|---|---|---|
| T1 | Native budget / TBN | **880–1250** | no flips | suite says “80”; push 200 | 🚨 |
| T2 | Trigger data | 384 test | **16** test | 384 calib | ✅ ours = paper; release bug |
| T3 | Trigger pattern | fixed pattern `P`, per-input positions | per-image `delta[16]`, reused by batch position | shared delta + per-image positions | ✅ ours = paper; release bug |
| T4 | Trigger/salience loss | CE | CE; attention only opt-in | **cos** default (`ce` option) | ❌ |
| T5 | Attention target | all selected patches (Eq. 4) | only the **first** patch | all selected patches | ✅ ours = paper; release bug |
| T6 | Attention layers | all L (Table 8: 12 best) | skips layer 0; last-term gradient | skips layer 0; mean over layers | ⚠️ minor |
| T7 | PCGrad | CE vs `Σ_l` attention | per-layer | per-image on mean-over-layers | ⚠️ minor |
| T8 | NGR | not described; Alg. 1 initialises last attn + classifier | true-label CE, `topk(768)` no-op | target-class CE, **8 rows** | ❌ |
| T9 | Insertion data | 384 | **16** | 384 | ✅ ours = paper; release bug |
| T10 | Distillation loss | CE + CE, **GradSurgery** | 0.5/0.5 CE, no surgery | 0.5 CE + 0.5 cos, no surgery | ⚠️ (#8) + missing surgery |
| T11 | Pruning | iterative, `e = 0.0005` | disabled (`optimized_wb1=False`) | post-hoc, `e = 0.003` | ⚠️ (#9; wrong e) |
| T12 | Tuned scope / TPN | target head row 768 → 213 | head row 768, no pruning | 8 proj rows = 6144/8192 | ⚠️ (#10; scale) |
| T13 | Bit realisation | 8-bit, TBN 880–1250 | none | int8 Hamming, 200 cap | ⚠️ |
| T14 | Trigger-only ASR | 6.67 % (ViT-b), 31.23 % (DeiT-s) | not measured in isolation | **83.6 %** (ViT-B), 24.7 % (ViT-L) | ❌ confound |
| T15 | Target class | 2 | 2 | per dataset (291/97/76) | ⚠️ deliberate |
| T16 | TAR on ViT-L/14 | 4.59 % | 3×patch (42²) | 3×patch (42²) = **3.52 %** | ⚠️ cross-model |
| T17 | Trigger init | **0** | random | random (`--trojvit-init`) | ⚠️ minor |
| T18 | Trigger mode | replace (Eq. 3) / add (Eq. 2) | replace DeiT / add ViT | replace (`--trojvit-mode`) | ⚠️ minor |

### 3.6 Why the differences matter

- **T1 is the headline problem.** The paper’s TrojViT needs 880–1250 bits to
  reach ~99 % on ImageNet; the suite caps at 200 and calls 80 “native”. Under
  200 bits, `flips_to_98 = null` is expected. Do not interpret the ViT-B/16
  76.7 % or ViT-L/14 19.7 % as the method’s ceiling. Either run at 1250 bits
  or state “common 200-bit cap, below TrojViT’s native TBN”.
- **T14 is the other confound.** The paper’s trigger alone gives 6.67 % on
  ViT-b; ours gives 83.6 % on ViT-B/16. The flips have only ~14 points of
  headroom, and the realised 196 bits (≈25 entries out of 6144 tuned) can even
  reduce ASR. The ViT-L/14 case (24.7 % pre) is closer to the paper’s regime,
  but it is under-budget and overfits (quick 38 % vs eval 19.7 %).
- **T8/T12** mean our NGR/scope is much larger than the paper’s 768-weight
  target row: 6144 weights on ViT-B/16. That overfits the 384-image calib set
  and makes the realised 196 bits an even smaller fraction of the tuned set.
- **T3/T5** are cases where **we are more paper-faithful than the release**:
  the release’s per-image trigger and first-patch attention are bugs/quirks.
  Document them as fixes, not deviations.
- **T4** is a default that affects both salience ranking and trigger
  generation; README #8 only documents the cos substitution for the
  **distillation** loss. For paper fidelity run `--trojvit-loss ce
  --trojvit-temperature 1.0`.

### 3.7 Paper-faithful / fair TrojViT presets

For the paper’s trigger/insertion regime:

```
--trojvit-loss ce --trojvit-temperature 1.0
--p2-prune-e 0.0005          # paper default (TPN 213 / TBN 880)
--p2-rows 1                  # closest size to the paper's 768-weight row;
                             # or keep 8 and document the scope expansion
```

For the budget:

```
--max-flips 1250             # ViT-B/16 (paper TBN 1250)
--max-flips 2000             # ViT-L/14 (larger model, no paper value)
```

Report both the native-budget row and the common-200-bit row.

---

## 4. Flip-S (CVPR 2025)

### 4.1 What the paper specifies

- **Trigger:** `x̂ = (1−m)x + m·t`; fixed mask (bottom-right 3×3 patch block,
  TAR 4.59 %); 32 test samples; attention rollout;
  `L_atten = Σ_l ||A_l − α||²`, α = 100;
  `L = CE(x̂, y) + λ·L_atten`, λ = 1.
- **SFS loss:** `L = γ·CE(f̂(x̂), y) + ||f̂(x) − f(x)||²`, γ = 1.
- **SFS (Algorithm 1):**
  - scale phase: top-k = 50 scale factors by **|grad|**; evaluate **all 16
    bits**; flip the bit with the greatest loss reduction; if none reduces
    the loss, switch to the weight phase;
  - weight phase: top-η = 50 weights by **|grad|**; evaluate **all 8 bits**;
    flip the best; return to scale phase;
  - 4KB mutual exclusion between flipped bits;
  - `n_max = 50` default; relaxed to 500 in the paper’s extended evaluation.
- **Quantisation:** bnb int8 (8-bit) and NF4 (4-bit); row-wise scales.
- **Headline (ImageNet, 8-bit, 50 bits):** ASR **99.5 %** ViT-base,
  **99.8 %** DeiT-base, **99.9 %** DeiT-small, 95.2 % Swin-base, 98.4 %
  Swin-small. Trigger-only: **≈32–38 %** (Table 5, 4-bit).

### 4.2 What the released code does

Files: `baselines/Flip-S/target-imagenet/deit-gen_trigger-int8.py`,
`deit-targeted-int8.py`, `utils/quant_model.py`.

| Stage | Release behaviour | Line |
|---|---|---|
| Trigger data | 32 aux samples, one batch of 64 | `gen:22–30`, `88–92`, `127` |
| Trigger mask | fixed 48×48 bottom-right; token indices `[195,...,165] + 2` (DeiT) | `gen:94–116` |
| Trigger update | sign-PGD, 300 iters, `schedule_lr`; CE + attenuation term; clamp [0,1] | `gen:122–173` |
| Attention term | rollout with residual + renormalisation; loop **overwrites** `atten_loss`, so only the last per-layer term enters the gradient | `gen:140–170` |
| Surface | **encoder only** (`model.deit.encoder`); qkv/out_proj/fc1/fc2 | `atk:143, 173, 244, 273, 338` |
| Quantiser | bnb int8 + fp16 `absmax` scales; `w_int` codes | `utils/quant_model.py` |
| SFS loss | `(MSE(clean, clean_logits) + γ·CE(triggered, target)) / (1+γ)`, γ = 1 | `atk:118–135` |
| Scale ranking | per-layer top-50 by **\|grad\|**, then global top-50 | `atk:141–171` |
| Bit pre-filter | skip if `(new − old)·grad > 0` (predicted loss increase) | `atk:195–197` |
| Scale acceptance | commit iff `min_loss < now_loss`; otherwise weight phase | `atk:235–239` |
| Weight phase | top-50 weights by **\|grad\|**; all 8 bits; **applies the best even if the loss worsens** (trap escape) | `atk:242–329` |
| 4KB exclusion | aligned 4KB block: 2048 scales / 4096 weights per layer | `atk:353–359`, `375–381` |
| Revocation | re-selecting a bit **unflips** it and removes the ban | `atk:386–408` |
| Stop | `target_bit + 10` iterations; exit when `len(changed_bit) ≥ 50` | `atk:114`, `409–415` |
| Eval | `small_val_loader` (10 % of val) + full ASR; trigger reused | `atk:97–98`, `411–422` |

The suite’s prior CLIP adapter `target-clip/flip_s_clip_v2.py` is a
line-for-line port with the head/probe binding; it calls `loss.backward()` twice
in the ranking step (an adapter bug), and it does **not** attack the
projection/head.

### 4.3 What our code does

File: `bfa_baselines/attack_flips.py`.

| Stage | Our behaviour | Line |
|---|---|---|
| Trigger data | aux 32 | `run:762–764` |
| Trigger mask | 3×3 patch block, bottom-right; side = 3×patch (48 px p16, **42 px p14**) | `trigger_geometry:414–456` |
| Trigger update | sign-PGD, 300 iters, `schedule_lr`, CE + attention term, clamp [0,1] | `generate_trigger:462–529` |
| Attention term | rollout; **last per-layer term only** (same as release) | `510–526` |
| Surface | encoder weights **+ `visual.proj`** (SigLIP: **head fc2**) | `find_surface_params:275–293` |
| Quantiser | local row-wise int8 + fp16 scales (bnb semantics) | `clip_adapters.rowwise_quantize_fp16` |
| SFS loss | `(MSE(clean, cached clean) + γ·CE(triggered, target))/(1+γ)` | `bitflip_search:690–700` |
| Ranking | bit score `\|dL/dparam · Δbit\|`; per-tensor top-50 rows/positions; global pool 800 scale / 400 weight candidates | `build_ranking_pools:549–617` |
| Pre-filter | skip `(Δ·grad) > 0` | `583`, `607` |
| Verification | each candidate flipped and quick-evaluated on **search[:quick_n = 256]** | `verify:665–671` |
| Acceptance | commit only if **quick ASR strictly improves**; else try weights; else stop | `703–716` |
| 4KB exclusion | ±4095 bytes around the flip address inside the tensor | `QuantSurface:295–410` |
| Revocation | none (monotone) | — |
| Stop | 200 bits or no improving candidate | `714–716` |
| `trigger_asr_pre` | calib 384 | `run:815–817` |
| Trace split | search 256 | `run:763`, `bitflip_search` |
| Budget | 200 (push-98) | `constants:121–134` |

### 4.4 Matching parts

- Two-stage structure: fixed patch-block trigger → SFS bit search.
- Trigger: 32 aux samples, zero init, 300 sign-PGD steps, `schedule_lr`,
  `CE + λ·(A−100)²`, λ = 1, clamp [0,1], masked updates.
- Quantisation: row-wise int8 + FP16 scales, fused qkv as one surface,
  forward through dequantised weights.
- SFS loss `(MSE + γCE)/(1+γ)`, γ = 1.
- Top-50 scale rows / top-50 weights, all 16/8 bits, `(Δ·grad)>0` pre-filter,
  scale-first fallback, 4KB separation, 50-bit native.
- Trigger-only ASR and final CACC/ASR from the shared evaluator.

### 4.5 Differences

| # | Difference | Paper | Release | Ours | Status |
|---|---|---|---|---|---|
| F1 | Attack surface | encoder linears only | encoder only | encoder **+ projection/head** | ❌ |
| F2 | Ranking | top-50 by **\|grad\|** | top-50 by **\|grad\|** | bits by **\|grad·Δbit\|** | ⚠️ (#4; claim wrong) |
| F3 | Acceptance | loss reduction | loss reduction (scale); best weight even if worse | **quick ASR** improvement | ⚠️ (#3) |
| F4 | Weight fallback | trap escape when no scale loss reduction | best weight bit applied unconditionally | ASR-gated; **stops if no ASR improvement** | ❌ |
| F5 | Attention loss | **sum over all layers** (Eq. 5) | last per-layer term only | last per-layer term only | ⚠️ (release bug copied) |
| F6 | TAR | 4.59 % | 48²/224² | 4.59 % p16; **3.52 % p14** | ⚠️ cross-model |
| F7 | Verification data | 32 aux | 32 aux | **256 search** | ❌ |
| F8 | `trigger_asr_pre` vs trace | same set | same set | calib 384 vs search 256; no `quick_split` | 🐞 |
| F9 | Budget | 50 (relax 500) | 50 | 200 | ✅ (documented); fine |
| F10 | Revocation | not mentioned | toggles repeated bits | none | ✅ ours = paper |
| F11 | 4KB geometry | “within 4 KB” | aligned 4KB block | ±4 KB window | ⚠️ (#5) |
| F12 | Quantiser | bnb int8 | bnb `my_8bit_linear` | local equivalent | ✅ |
| F13 | v2 adapter double backward | — | — | ours fixes it | 🐞 (adapter) |
| F14 | Target class | 2 | 2 | per dataset | ⚠️ deliberate |
| F15 | Trigger-only ASR | ≈32–38 % | reported | TBD (calib 384) | report |
| F16 | Bit pre-filter | evaluates **all** bits | drops `(Δ·grad)>0` | same pre-filter | ⚠️ release heuristic |
| F17 | Top-k selection | global top-50 rows/weights | per-layer top-50 → global 50 | per-tensor top-50 → global 800/400 candidates | ⚠️ minor |

### 4.6 Why the differences matter

- **F1 (extra surface)** can make our Flip-S stronger than the paper’s: a
  scale flip in `visual.proj` shifts a whole embedding dimension and moves the
  target logit directly. The prior adapter explicitly says “Flip-S never
  attacks the head”; our surface list does. For SigLIP, `head.mlp.fc2` is
  literally the head. Exclude it for the paper-faithful row or report both
  surfaces.
- **F4 (ASR-gated fallback)** is the most consequential search difference. The
  weight phase exists to escape a trap when no scale flip reduces the loss.
  Our version refuses any move that does not immediately improve ASR and then
  **stops**, so it cannot execute the escape. This can cause early termination
  and a null `flips_to_98`.
- **F2/F3** are documented (#3/#4), but README #4’s “the paper’s SFS formula”
  is wrong: the paper ranks by `|grad|`, not `|grad·Δbit|`. Fix the wording.
- **F5** weakens the trigger relative to the paper (only one layer’s attention
  term). It is inherited from the release; if the trigger-only ASR is low,
  this is a likely cause.
- **F6** means ViT-L/14 runs a 3.52 % trigger, not the paper’s 4.59 %; the
  paper’s TAP ablation says smaller triggers are harder.
- **F7/F8** change the data protocol and make `trigger_asr_pre` incomparable
  with `quick_trace[0]`. Report both splits.

### 4.7 Paper-faithful Flip-S preset

```
# surface: encoder-only (exclude visual.proj / SigLIP head fc2)
# ranking: |grad|; all bits of the top-50 rows/weights
# acceptance: loss reduction; weight fallback applies the best bit even if worse
# attention: sum over all layers
--trig-side 48              # 4.59% on ViT-L/14 too
--max-flips 50              # native; also run 500 (paper relaxed) or 200 (push-98)
# verification on the 32 aux samples (paper), or document the 256-search variant
```

Also fix the metadata: record `quick_split`, `tar`, and the `trigger_asr_pre`
split; make the trace start from the same split as `trigger_asr_pre`.

---

## 5. Consolidated deviation index

| ID | Baseline | Deviation | Paper / release / ours | Status | Action |
|---|---|---|---|---|---|
| P1 | ProFlip | S1 once vs re-rank | once / once / re-rank | ❌ | flag, preset |
| P2 | ProFlip | S2 1 vs top-8 | 1 / 1 / 8 | ❌ | flag, preset |
| P3 | ProFlip | S3 grid | K20 / K10 float / K20 int monotone | ⚠️ | document |
| P4 | ProFlip | Acceptance | argmin / argmin / gated | ❌ | flag, preset |
| P5 | ProFlip | `Q_p` | abs / signed / signed | ⚠️ | default/flag |
| P6 | ProFlip | Clean CE | 0.1 / 2× / 0 | ⚠️ | document |
| P7 | ProFlip | CBS anchor | `M*` / poisoned self / clean | ❌ | document |
| P8 | ProFlip | SNI | JSMA∩ / JSMA∪ / analytic | ⚠️ | document |
| P9 | ProFlip | Clamp `c` | 10 / 10 / max | ⚠️ | document |
| P10 | ProFlip | Trigger data/space | 256 train / 128 test / 384 calib | ❌ | document |
| P11 | ProFlip | CBS data | 256 train / 32 test / 256+32 | ❌ | document |
| P12 | ProFlip | Stop | 94/100 / 50 / 98/200 | ⚠️ | document |
| P13 | ProFlip | Quick trace split | — / test / aux 32 vs meta 256 | 🐞 | fix |
| P14 | ProFlip | ASR set | all / all / non-target | ⚠️ | report both |
| P15 | ProFlip | Trigger TAP | 10.62 / 9.77 / 9.77 | ⚠️ | report TAR |
| P16 | ProFlip | Target class | 2 / 2 / per dataset | ⚠️ | document |
| P17 | ProFlip | Surface binding | all quantised / same / encoder+proj | ⚠️ | document |
| T1 | TrojViT | Native budget | 880–1250 / none / 80+200 | 🚨 | fix budget |
| T2 | TrojViT | Trigger data | 384 / 16 / 384 | ✅ | document |
| T3 | TrojViT | Trigger pattern | fixed / per-image / shared | ✅ | document |
| T4 | TrojViT | Trigger loss | CE / CE / cos | ❌ | preset `ce` |
| T5 | TrojViT | Attention patches | all / first / all | ✅ | document |
| T6 | TrojViT | Attention layers | all / skip0 last term / skip0 mean | ⚠️ | document |
| T7 | TrojViT | PCGrad | CE vs Σ / per-layer / per-image mean | ⚠️ | document |
| T8 | TrojViT | NGR | — / no-op / 8 rows | ❌ | document |
| T9 | TrojViT | Insertion data | 384 / 16 / 384 | ✅ | document |
| T10 | TrojViT | Distill loss | CE+CE+surgery / 0.5CE / 0.5CE+0.5cos | ⚠️ | document |
| T11 | TrojViT | Pruning | 0.0005 iterative / off / 0.003 post | ⚠️ | preset |
| T12 | TrojViT | TPN scope | 213 / 768 / 6144–8192 | ⚠️ | document |
| T13 | TrojViT | Bit realisation | 880–1250 / none / 200 | ⚠️ | budget |
| T14 | TrojViT | Trigger-only ASR | 6.67 % / — / 83.6 % | ❌ | report |
| T15 | TrojViT | Target class | 2 / 2 / per dataset | ⚠️ | document |
| T16 | TrojViT | TAR p14 | 4.59 / 3.52 / 3.52 | ⚠️ | report |
| T17 | TrojViT | Trigger init | 0 / random / random | ⚠️ | document |
| T18 | TrojViT | Trigger mode | replace/add / replace(add for ViT) / replace | ⚠️ | document |
| F1 | Flip-S | Surface | encoder / encoder / +proj | ❌ | preset |
| F2 | Flip-S | Ranking | \|grad\| / \|grad\| / \|grad·Δbit\| | ⚠️ | fix doc |
| F3 | Flip-S | Acceptance | loss / loss / ASR | ⚠️ | document |
| F4 | Flip-S | Weight fallback | trap escape / unconditional / ASR-gated stop | ❌ | preset |
| F5 | Flip-S | Attention loss | sum / last / last | ⚠️ | preset |
| F6 | Flip-S | TAR | 4.59 / 4.59 / 3.52 p14 | ⚠️ | report |
| F7 | Flip-S | Verification data | 32 / 32 / 256 | ❌ | document |
| F8 | Flip-S | Pre/trace split | — / same / calib vs search | 🐞 | fix |
| F9 | Flip-S | Budget | 50/500 / 50 / 200 | ✅ | document |
| F10 | Flip-S | Revocation | no / yes / no | ✅ | document |
| F11 | Flip-S | 4KB geometry | within 4KB / block / ±4KB | ⚠️ | document |
| F12 | Flip-S | Quantiser | bnb / bnb / local | ✅ | document |
| F13 | Flip-S | v2 double backward | — / — / fixed | 🐞 | document |
| F14 | Flip-S | Target class | 2 / 2 / per dataset | ⚠️ | document |
| F15 | Flip-S | Trigger-only ASR | 32–38 / — / TBD | — | report |
| F16 | Flip-S | Bit pre-filter | all bits / drop Δ·g>0 / same | ⚠️ | document |
| F17 | Flip-S | Top-k selection | global 50 / per-layer 50 / per-tensor 50 → 800/400 | ⚠️ | document |

---

## 6. Pre-thesis checklist

1. **Fix the TrojViT budget claim.** Replace “native 80” with the paper’s
   TBN 880 (DeiT-s) / 1250 (ViT-b); run at least one native-budget row, or
   state that the table uses a common 200-bit cap below TrojViT’s native TBN.
2. **Fix the ProFlip quick-ASR split/stop.** Use `search[:quick_n]` for the
   trace and stop, or relabel the meta; `trigger_asr_pre` must use the same
   split. Add `quick_split` to both ProFlip and Flip-S metas.
3. **Decide the Flip-S surface.** Encoder-only for the paper-faithful row;
   encoder+projection as a clearly labelled “extended surface” row.
4. **Decide the Flip-S acceptance/fallback.** Loss-based scale acceptance and
   unconditional weight fallback for paper fidelity; ASR-gated only as a
   labelled variant.
5. **Decide the TrojViT loss/pruning preset.** `--trojvit-loss ce
   --trojvit-temperature 1.0` and `--p2-prune-e 0.0005` (paper default) for
   the paper-faithful row; keep the cos/0.003 variant as a separately labelled
   row.
6. **Report TAR + trigger-only ASR for every method** (including the proposed
   method). The proposal’s 0.51 % trigger vs the baselines’ 3.52–9.77 % is a
   large confound.
7. **Report both native and push-98 rows** for TrojViT (880–1250 vs 200) and
   Flip-S (50 vs 200); ProFlip’s 94/100 vs 98/200.
8. **Fix README wording:** TrojViT “native 80” is wrong; Flip-S deviation #4
   (“paper’s SFS formula”) is wrong; TrojViT “200 steps on one fixed batch” is
   accurate for the release (16 images × 200 steps), but the paper says 384
   images; add the missing deviations (P1/P2/P4/P7/P10/P11, T4/T8/T10/T12,
   F1/F4/F5/F7/F8).
