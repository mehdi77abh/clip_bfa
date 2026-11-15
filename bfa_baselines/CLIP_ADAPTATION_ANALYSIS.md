# CLIP_ADAPTATION_ANALYSIS.md — Adapting Flip-S, ProFlip and TrojViT to CLIP

**Purpose.** This is the design record for the *comparison* half of the thesis: for each
baseline, which parts of its published method can be moved to CLIP unchanged, which parts
must be re-bound or re-designed, and which parts have **no faithful CLIP analogue**.
It is based on the actual code in this repo, not only on the papers.

**Target setup (agreed):** zero-shot (ZS) CLIP classification, OpenAI `clip.load`,
ViT-B/16 (and the registry models), ImageNet-1k val / Caltech101, target class aligned
with the proposed method (`--target-cls 3` on ImageNet-1k), shared class list
(`class_texts/*/classes.txt`) and the same 13-template prompt ensemble as
`TargetEmbedder.build_text_embeddings`. The linear-probe (LP) task was abandoned because
it requires adding a trained `nn.Linear` head to CLIP (an architectural change and a
non-native readout); see `BASELINES.md` §2.

**Sources of truth**

| Method | Code inspected | Paper |
|---|---|---|
| Flip-S | `bfa_baselines/Flip-S/target-clip/flip_s_clip_v2.py` (v2, OpenAI `clip`), `target-imagenet/deit-gen_trigger-int8.py`, `target-imagenet/deit-targeted-int8.py`, `utils/{quant_model,load_data,metrics}.py` | `Flip-S/Flip-S.pdf` (CVPR 2025) |
| ProFlip | `bfa_baselines/ProFlip/<variant>/{trigger_adaptive,trigger_nonadaptive,CSB_adaptive,CSB_nonadaptive}.py`, `models/quantization.py`, `utils.py` | `ProFlip/README.md` + ICCV 2021 paper (fetched: Chen et al., *ProFlip: Targeted Trojan Attack with Progressive Bit Flips*) |
| TrojViT | `bfa_baselines/TrojViT/main_patch_vit.py`, `utils.py`, `models/DeiT.py` | `TrojViT/TrojViT Trojan Insertion in Vision Transformers.pdf` (CVPR 2023) |
| Adaptation already written | `bfa_baselines/proflip_cbs.py` (ProFlip CBS), `bfa_baselines/Flip-S/target-clip/flip_s_clip_v2.py` (Flip-S) | — |

---

## 1. What CLIP gives a baseline — and what it does not

A baseline can only be ported faithfully if the CLIP victim exposes the objects the
method attacks.

| Object a baseline may expect | CLIP reality | Consequence |
|---|---|---|
| A **classifier head** `nn.Linear(width, n_classes)` with a **row per class** | **None.** The ZS classifier is `logit_scale · normalize(f) @ Tᵀ`, where `T` is computed at runtime from the text encoder | Any method that selects/updates a *class row* (ProFlip's `psens` last dense layer, TrojViT's `head.weight[target]`) has no direct target |
| A **last layer** whose input is the class logit | The visual tower's last linear is `visual.proj` (768 → 512), a projection to the joint space, **not** a classifier | "Last layer" is ill-defined; `visual.proj` has no class rows |
| **Attention weights** returned by the model | OpenAI CLIP's `nn.MultiheadAttention` **discards** per-head weights (`need_weights=False`) | Attention-based selection/losses need a recompute (the `QuantMHA` trick in `flip_s_clip_v2.py`) |
| **2 special tokens** (DeiT-distilled) | CLIP has **1** CLS token | Every token-index computation shifts by **−1** (`+2 → +1`) |
| A **fixed input patch** applied to every image | CLIP accepts any tensor; but TrojViT selects **9 patches per image**, and ProFlip/Flip-S use one **square** patch | The trigger application mode is a real design choice (see §6.1) |
| A quantized weight grid | CLIP is fp32 in the pipeline; the pipeline re-quantizes to **per-tensor int8** (weights + biases + LayerNorms); Flip-S uses **bnb row-wise int8** | Bit surfaces and counts differ by method (§6.2) |
| A 1000-way softmax | ZS logits exist, but computing CE needs the **full text-embedding table**; the proposed-method threat model dropped CE for that reason (`AGENTS.md`) | Methods whose loss uses CE (ProFlip, TrojViT) need a target-cosine substitute or an explicit "stronger-information" note |

**Two more facts that decide fairness**

- **TrojViT and ProFlip's insertion are training-time** (optimize weights with gradient
  descent), while Flip-S and the proposed method are **post-deployment bit-flip** attacks.
  Their "bits" (TBN) are *derived* by counting differing bits after quantizing the trained
  weights, not selected bits.
- The **trigger area** differs by paper: proposed method 16×16 = **0.51 %**; Flip-S
  reproduces all baselines at **4.59 %** (9×16×16, following TrojViT); ProFlip's own
  ImageNet default is **10.62 %** (~73×73); TrojViT is **4.59 %** (9×16×16). A comparison
  must fix the area per row and state it.

---

## 2. Flip-S (CVPR 2025) — fully adaptable, already done

### 2.1 What the method is

Two stages, both on the **vision encoder only**:

1. **Stage A — trigger generation** (`deit-gen_trigger-int8.py`): a **9-patch trigger**
   (16×16 patches in the bottom-right 48×48 block, ids `[195,194,193,181,180,179,167,166,165]`),
   optimized by sign-SGD for 300 iterations with
   `CE(logits, target) + 1.0·attention_rollout_loss`, `clamp(0,1)`.
2. **Stage B — targeted bit-flip search** (`deit-targeted-int8.py`): greedy search over
   the int8 **scale factors** (`absmax`, 16-bit) and **weights** (`w_int`, 8-bit) of every
   quantized linear; loss `[MSE(clean logits) + 1.0·CE(triggered, target)] / 2`; top-k 50
   candidates per surface; 4096-byte block bans (4096 int8 weights / 2048 fp16 absmax);
   stop at 50 accepted bits.

It attacks **encoder linears**, not the head. That is why it maps to CLIP.

### 2.2 Which parts adapt

| Flip-S part | CLIP adaptation | Status |
|---|---|---|
| Victim binding (`DeiTForImageClassificationWithTeacher`) | `CLIPVictim` wrapper: `model.deit.encoder` aliases `clip_model.visual.transformer`; `.logits`/`.attentions` provided | ✅ implemented (`DELTA 1`) |
| Attack surface (q/k/v, out, fc1, fc2) | CLIP fuses q/k/v into `in_proj_weight`; one quantized `attn.qkv` module per block + `out_proj`, `mlp.c_fc`, `mlp.c_proj` → **48 modules** | ✅ (`DELTA 2`) |
| Int8 quantization | bnb `int8_vectorwise_quant` (row-wise `CB`/`SCB`, threshold 6.0), with an explicit dequant forward so `replace_with_myLinear` still works | ✅ (`DELTA 5`) |
| Stage A attention maps | OpenAI CLIP discards them; `QuantMHA` recomputes per-head scaled-dot-product attention; token offset **+1** | ✅ (`DELTA 4`) |
| Stage A CE loss | ZS logits via the batch of text embeddings; same CE semantics | ✅ (ZS head) |
| Stage B search loop | Unchanged (byte-identical): the loop only sees `.logits` and the quantized linear modules | ✅ |
| Head | Probe (legacy) or **ZS text head** (`--head zeroshot`) with the same class list and 13 templates | ✅ (`DELTA 3`) |
| Evaluation | `lp_eval_full` on the full val split, prints **ZS-CACC / ZS-ASR / clean-as-target** in ZS mode | ✅ (`DELTA 7`) |

### 2.3 What cannot be adapted / what is intentionally lost

- **The DeiT classifier head** (and `head_dist`) is not part of CLIP. Flip-S never attacks
  the head, so this does not change attack mechanics — but the baseline's original
  "classifier" is replaced by the ZS text head (or a probe in legacy mode).
- The **DeiT token offset** (+2) and 2-token layout cannot be kept; +1 is mandatory.
- The **4-bit NF4** variants are not ported (only the int8 ImageNet variant); the thesis
  comparison is int8.
- `utils/metrics.imagenet_acc/asr` returns **top-5 = 0** (never updated) — do not report
  top-5.
- The released Stage-A trigger code and Stage-B search run on **DeiT**; the CLIP adapter
  changes **model binding, attention capture, head, dtype plumbing** only. The search
  loop and hyperparameters stay fixed.
- The 4 KB/2 KB **row-ban** constants are DeiT-base-shaped but work for CLIP-B/16; nothing
  is re-tuned.

### 2.4 Fair-comparison notes

- Flip-S's own trigger is **9 patches = 4.59 %**, not the proposed method's 0.51 %.
  If you compare flip counts, match TAP (or re-run the proposed method with
  `--trigger-size 72/73 --trigger-type replace --trigger-location bottom_right`).
- Flip-S reports **final** CACC/ASR at its 50-bit budget; it does **not** evaluate per
  flip, so it cannot fill `ASR@10/@20` or `Flips→95 %` without adding a harness loop.
- The published ViT/DeiT/Swin numbers at TAP 4.59 % are the reference rows.

### 2.5 Verdict

**Fully adaptable mechanically** — already implemented and run in ZS mode. Keep it as the
primary external baseline. No method change is needed; every deviation is binding/plumbing
(`flip_s_clip_v2.py` header, DELTA 1–7). The one protocol caveat is information
asymmetry: Flip-S's Stage A/B CE uses the full class table, while the proposed method is
target-only — see §5.5 for the two ways to handle that.

---

## 3. ProFlip (ICCV 2021) — CBS adaptable, the rest only partially

### 3.1 What the method is

Three stages (`trigger_adaptive.py`, `CSB_adaptive.py`, `models/quantization.py`; paper
Algorithms 1–3):

1. **SNI — Salient Neuron Identification**: JSMA on the **input to the last layer**
   (`a_{L-1}`), θ=0.1, γ=0.5, target `t`; intersection across clean images → `I_t`.
   Used with a constant `c=10` in the later losses.
2. **Trigger generation**: square trigger at the **bottom-right**, **replace** mode,
   `TAP = 10.62 %` on ImageNet (~73×73), optimized with
   `L_trig = λ1·MSE(M_{1:L-1}(x̂); c) + λ2·CE(M(x̂); t)`, λ1=λ2=1, SGD lr=0.1, 100 epochs.
3. **CBS — Critical Bit Search**: S1 parameter fitness
   `F = |∂L/∂p|·(Q_p − elem)` for negative gradients; S2 element = argmax `F` within the
   parameter; S3 grid-search `K=20` values over `[-R,R]`, monotone, then flip **all**
   differing bits (Hamming distance). `Q_p = max|W_l|`; `ASR_t = 94`, `n_max = 100`.
   The CBS loss contains the SNI term:
   `L_CBS = Σ_x [MSE(M*_{1:L-1}(x*); c) + CE(M*(x*); t)]`.

The released code differs from the paper in two places worth knowing: the trigger loop is
**10 iterations at lr 0.01 on one batch** (not 100 epochs at lr 0.1), and the CIFAR
trigger geometry is `start=21,end=31` → 10×10 = 9.77 % (matching the paper's 9.76 %).
The extraction also has no ImageNet variant (only CIFAR-10/100, STL-10, Tiny-ImageNet).

### 3.2 Which parts adapt

| ProFlip part | CLIP adaptation | Feasible? |
|---|---|---|
| Per-tensor 8-bit quantizer (`Δ_l = max\|W_l\|/127`, two's complement) | The pipeline's Phase 0 uses per-tensor int8 with `scale = max\|W\|/127`; nearly identical (pipeline clamps to `[-128,127]`, ProFlip to `[-127,127]`; pipeline also quantizes biases/LN, ProFlip's attack surface includes biases on VGG) | ✅ (with a documented range difference) |
| SNI (JSMA on `a_{L-1}`) | Penultimate = pooled CLS feature before `visual.proj`; ZS logits as `M_L`. JSMA itself is architecture-agnostic, but **no stored classifier** and `c=10` must be recalibrated for LayerNorm'd CLIP features | ⚠️ partial / re-designed |
| `I_t`, `c` in `L_trig` and `L_CBS` | There is no classifier row to define "significant neurons"; `I_t` can be defined on the 768-d CLS, but it is a new choice | ⚠️ not faithful |
| Trigger geometry (replace, bottom-right, 73×73/10.62 %) | `apply_trigger` supports `replace` + `bottom_right` + size 73; maps directly | ✅ |
| Trigger loss `MSE(...; c) + CE(..., t)` | `MSE` to a constant `c` needs SNI `c`; `CE` needs the full text-embedding table, which the project's BFA threat model excludes (Phase-1 CE was dropped). A target-cosine term is the consistent substitute | ⚠️ adapted |
| CBS S1/S2/S3, fitness, `Q_p`, K=20, Hamming, `ASR_t`/`n_max` | Architecture-agnostic: works on any parameter slice; implemented in `bfa_baselines/proflip_cbs.py` | ✅ (selection rule only) |
| Parameter-level search incl. **last dense weight / biases** | CLIP has no last dense classifier; the pipeline's attack surface is **block weights only** (`V` rows, `O` cols, `fc`, `proj`); biases/LN/`visual.proj` are not flippable | ❌ not faithful |
| Evaluation ASR/CACC, bits to 94/95 | Maps; note ProFlip's 15 bits / 94.3 % is ResNet-18/ImageNet with TAP 10.62 %, target 2 | ✅ with trigger/area matching |

### 3.3 What cannot be adapted faithfully

1. **The last-layer/classifier row.** `psens = 60` on ResNet-18 is the last dense layer's
   weight; on VGG-16 `psens = 45` is a **bias vector** of the second-to-last conv. CLIP
   has no classifier row, and its analog (`visual.proj`) has no class dimension. The text
   embedding matrix is a **text-side** object; attacking it is a different threat model.
2. **SNI's `I_t`/`c`.** Without a stored classifier, "significant neurons" is undefined;
   any CLIP version is a new heuristic.
3. **CE over the full class table + clean labels** in `L_trig`/`L_CBS`/`L_troj` — outside
   the project's BFA threat model (and needs labels ProFlip's own threat model assumes).
4. **Full-parameter CBS** (all conv/fc weights and biases). The pipeline only exposes
   Phase-2 component weight slices; biases and normalization parameters are not attacked.
5. **Exact quantizer semantics** (`[-127,127]` vs `[-128,127]`; biases quantized or not)
   and the **`Q_p` convention** (paper `max|W|` vs released code `max(W)`).

### 3.4 What was actually adapted

`bfa_baselines/proflip_cbs.py` implements **CBS as a component baseline**:

- S1/S2/S3, `Q_p`, negative-gradient-only fitness, K=20 grid, monotone updates,
  atomic multi-bit element flips, Hamming bit counting, `ASR_t`/`n_max` stopping;
- CLIP objective `−cos(f(triggered), t_target)` (+ optional clean-feature MSE), **no SNI
  term** and **no class table/labels**;
- "parameter" = a Phase-2 component tensor slice (run `--variant a` for the full pool);
- shared trigger from Phase 1; output metrics compatible with the pipeline.

It is therefore **not ProFlip**; it is "ProFlip-CBS (CLIP adaptation, shared trigger)".
For a fuller row, Phase 1 (SNI + trigger) would have to be added; that is a separate
1–2-week design with the caveats above.

### 3.5 Fair-comparison notes

- ProFlip's own headline is ResNet-18/ImageNet, TAP **10.62 %**, `ne=3`, `nb=15`,
  ASR **94.3 %**, clean 69.0 → 67.6 %. Do not put that row next to a 0.51 % trigger without
  labelling it.
- Flip-S reproduced ProFlip on ViT/DeiT/Swin at TAP **4.59 %**, n_max=50, 8-bit ImageNet:
  **71.9 / 56.8 / 94.6 / 88.7 / 80.5 %** ASR (ViT-base / DeiT-base / DeiT-small /
  Swin-base / Swin-small). That is the best existing "ProFlip on ViT" reference.
- A CBS-only row answers "is our bit selection better than ProFlip's?", not "do we beat
  ProFlip end-to-end". Say which claim the table supports.

### 3.6 Verdict

**CBS: adaptable — done. SNI/trigger: only a re-designed variant. Full ProFlip: not
faithful on CLIP.** Use CBS as the component baseline and cite the full-method numbers;
do not call the CBS-only row "ProFlip".

---

## 4. TrojViT (CVPR 2023) — Phase 1 adaptable, Phase 2 has no faithful analogue

### 4.1 What the method is

Two phases, three mechanisms (`main_patch_vit.py`; paper §3):

1. **Trigger generation**
   - **Patch salience ranking**: for each image, rank the 14×14 patches by
     `G = Σ_pixels |∂L_CE(X, y_k)/∂X_pixel|`; keep the top-N (N=9; 16×16 patches → 4.59 %
     TAR).
   - **Attention-target loss**:
     `L_ATL = L_CE(X̂, y_k) + λ Σ_l L_ATTN^l`, with
     `L_ATTN^l = −log Σ_{h,i} attn^{l,h}_{i→T}`; PCGrad resolves CE/attention gradient
     conflicts.
   - Optimize the masked perturbation with Adam (code: lr 0.22, 250 iterations, clamp to
     the valid range).
2. **Trojan insertion ("tuned parameter distillation")**
   - Select `W_T` in the last attention + classification layers, optimize
     `L(f(X), y) + L(f(X + M⊙P), y_k)` on **only `W_T`**, with gradient surgery,
     prune updates `< e`, 200 epochs on 384 clean images (paper; the released code uses
     **2 batches per epoch** from the val loader — `if t==1: break`).
   - Paper: ImageNet ViT-b → CDA 84.07 → 83.53 %, ASR 98.82 %, **TPN 292**, **TBN 1250**,
     TAR 4.59 %.
   - **Released code is a simplification:** it selects the target class's **head row**
     via NGR (top-768 neurons) and trains **only `head`**, restoring every other head
     weight; the paper's "last attention + classification" is not what the code does.

### 4.2 Which parts adapt

| TrojViT part | CLIP adaptation | Feasible? |
|---|---|---|
| Patch salience ranking | Gradient of the ZS target CE (or target cosine) w.r.t. the input; sum per 16×16 patch; top-9 | ✅ |
| Attention-based patch selection | Recompute per-head maps (`QuantMHA`); layer index maps 1:1; token offset **+1** | ✅ with attention recompute |
| `L_ATTN` / `L_ATL` | Same formula on CLIP visual attention; CE can be ZS CE or target-cosine | ✅ (with the CE/threat-model caveat) |
| PCGrad / gradient surgery | Architecture-agnostic (`utils.PCGrad`) | ✅ |
| Trigger optimization (Adam, clamp) | Same, but TrojViT's trigger is **9 patches selected per image** and applied **additively**; the pipeline supports one **fixed** square patch | ⚠️ semantic gap |
| NGR (top-k target neurons) | **No classifier head** → no target-class row to rank | ❌ |
| Parameter distillation / insertion | **No classifier head.** The LP probe head is an added module (the rejected LP setup); the text embedding is text-side; `visual.proj` has no class rows | ❌ not faithful |
| CDA / ASR / TAR | CACC / ASR / trigger area | ✅ |
| TPN / TBN | Derived: count changed weights / bits after quantizing the trained model; there is no bit search | ⚠️ derived only |

### 4.3 What cannot be adapted faithfully

1. **NGR + head-row distillation.** This is the core of TrojViT's "Trojan insertion".
   CLIP zero-shot has no classifier head, hence no class row. Your three candidate
   "heads" are all invalid for a fair port:
   - **linear probe** → adds a module and reintroduces the LP invalidity;
   - **text embeddings** → text-side attack, not visual-encoder BFA;
   - **`visual.proj`** → a projection, not a classifier; no class rows.
2. **The metric itself.** TrojViT is a **training-time backdoor**, not a BFA: it
   fine-tunes weights for 200 epochs on data, and its TBN is an *emergent* bit difference.
   It cannot fill `flips_to_95` / `ASR@10/@20`; it belongs in a separate
   "training-time backdoor" table with `CACC / ASR / TAR / TPN / TBN / time`.
3. **Per-image patch positions.** The paper selects 9 patches per image; the released
   code's evaluation reuses the last training batch's `max_patch_index` for the whole
   test set. Either choice is a semantic decision; neither is the pipeline's fixed patch.
4. **Data budget.** TrojViT uses **384 test images** and 200 epochs of insertion; the
   proposed method uses 32 target + 64 calib + 16 gradient images and no training. The
   comparison must state this.

### 4.4 The three honest options

**A. Trigger-only TrojViT-CLIP (cheap, recommended).**
Implement Phase 1 only: patch salience + attention-target loss + PCGrad on CLIP ZS.
Report **trigger-only ASR/CACC** (no weight change). Optionally fix the 9 patch positions
and feed that trigger into the proposed BFA as a "TrojViT trigger + our search" row.
This is ~1–2 days, reuses the Flip-S CLIP scaffolding, and is a meaningful trigger-quality
baseline — but it is **not TrojViT**.

**B. TrojViT-CLIP with an adapted visual-encoder insertion (only if a full row is
required).**
Keep Phase 1, then replace the head distillation with:
- `W_T` = top-k weights (gradient magnitude for clean+triggered objective) in the **last
  block's attention + MLP** (optionally `visual.proj`, but note the proposed method does
  not attack it);
- objective `L_CE(clean, y) + L_CE(triggered, y_target)` (or target-cosine if the class
  table is excluded), gradient surgery, prune updates < e, 200 epochs on 384 clean
  images;
- report ASR/CACC + **TPN** (weights changed) + **TBN** (bits changed after Phase-0 int8
  quantisation) + TAR 4.59 %.
Label it **"TrojViT-style insertion on the CLIP visual encoder (adapted)"**; it is a new
method, not a port. Effort: ~1–2 weeks.

**C. Cite only (cheapest, defensible).**
Use the paper's Table 5 (ViT-b: CDA 84.07→83.53, ASR 98.82, TPN 292, TBN 1250, TAR 4.59 %)
and Flip-S's reproduced TrojViT numbers (8-bit ImageNet at TAP 4.59 %: ViT-base 68.6 %,
DeiT-base 79.2 %, DeiT-small 66.0 %, Swin-base 90.1 %, Swin-small 77.8 % ASR). Put them in
a separate "training-time backdoor" block.

**Do not do:** add a linear-probe head and run TrojViT on it. It reintroduces the exact
LP problem and makes the victim non-CLIP.

### 4.5 Upstream code caveats (if porting)

`main_patch_vit.py` assumes `model.module` (DataParallel), hardcodes `patch_size=16`,
`targets=2`, `wb=768`, `wb1=768` (DeiT-B head), sets `args.mu/std` in the loader rather
than argparse, trains on `loader_test` (val leakage), reuses `max_patch_index` from the
last training batch for evaluation, and in the sparse branch references `loss` before it
is defined. The released insertion trains only `head` despite the paper's "last attention
+ classification". A CLIP port needs cleanup, attention recomputation, per-image masks,
and a decision on fixed vs per-image positions.

### 4.6 Verdict

**Phase 1: adaptable. Phase 2: no faithful CLIP analogue.** Run trigger-only + cite, or
build the labelled visual-encoder variant; do not claim a literal TrojViT port.

---

## 5. Cross-cutting differences that decide fairness

### 5.1 Trigger matching

| Method | Trigger | Area | Mode / location | Positions |
|---|---|---|---|---|
| Proposed method | one square patch | 16×16 = **0.51 %** (or 72/73 = 10.3/10.6 %) | `center_add` by default; `--trigger-type replace --trigger-location bottom_right` available | fixed |
| Flip-S | 9×16×16 | **4.59 %** | replace, bottom-right | fixed block |
| ProFlip | one square | **10.62 %** (73×73) | replace, bottom-right | fixed |
| TrojViT | 9×16×16 | **4.59 %** | additive, per-image selected patches | per-image (paper) / last-batch (code) |

Recommendation: for each baseline row, state the area/mode/location; for a head-to-head
with the proposed method, either match the area (and note the cost) or report the baseline
at its own default and mark the row "as published".

### 5.2 Quantization

| Method | Scheme | Attack surface | Bit count |
|---|---|---|---|
| Proposed | per-tensor int8, `scale = max\|W\|/127`, weights + biases + LN quantized | Phase-2 component **weight** slices only | selected bits |
| Flip-S | bnb **row-wise** int8 (`CB`/`SCB`), threshold 6.0 | **absmax** (fp16) + **w_int** (int8) of qkv/out/fc1/fc2 | selected bits |
| ProFlip | per-tensor 8-bit, `Δ = max\|W\|/127`, two's complement | all conv/fc **weights and biases** incl. last layer | selected bits (Hamming) |
| TrojViT | shares ProFlip/TBT 8-bit quantization only for reporting | last attention/classification weights (paper), head only (code) | **TBN derived** after fine-tuning |

Consequence: Flip-S vs proposed is a *different scale-factor surface*; ProFlip vs
proposed is comparable per-tensor but different surface; TrojViT is not a bit selection at
all. State this next to every number.

### 5.3 Threat model and data budget

| Method | Training data? | Labels? | Class table? | Budget |
|---|---|---|---|---|
| Proposed | no | target only | no (CE dropped) | 32 target + 64 calib + 16 grad + 1000 eval |
| Flip-S | no | target + aux labels ignored | **yes** (CE over ZS logits) | 32 aux + full val |
| ProFlip | no | **yes** (clean CE/stealth) | **yes** (CE) | 256 clean batch |
| TrojViT | **yes** (insertion) | **yes** | **yes** (CE) | 384 images, 200 epochs |

If a baseline uses labels/class table and the proposed method does not, either adapt the
baseline's loss to the target-only setting or label it "stronger-information baseline".

### 5.4 Metrics and table placement

| Table | Methods | Columns |
|---|---|---|
| **ZS bit-flip comparison** | Proposed, Flip-S, ProFlip-CBS | ZS-CACC, ZS-ASR, flips→94/95, time |
| **Full-method reference** | Flip-S, ProFlip (as reported), proposed | + trigger area, target, notes |
| **Training-time backdoor** | TrojViT, (ProFlip full if ported) | CACC, ASR, TAR, TPN, TBN, epochs/time |
| **Component ablations** | Random / simple-gradient / ProFlip-CBS / ours (shared trigger) | flips→95, CACC, time |

`ASR@10/@20` only make sense for one-bit-at-a-time searches. For multi-bit rounds
(ProFlip-CBS) or training-time methods (TrojViT), use "bits to 94/95" or final ASR and say
so.

### 5.5 Information symmetry (CE vs target-only)

The proposed method's Phase 1 uses **target-only** objectives (`cos` / `cos_astar`); its
`AGENTS.md` explicitly dropped CE because CE needs the **full class-embedding table** and
that is outside the BFA threat model. The baselines were not written under that
restriction:

| Method | Attack-time information | Needs labels? |
|---|---|---|
| Proposed | target text embedding only; calibration images for centroids | no |
| Flip-S | **full 1000-way ZS logit vector** in Stage A CE and Stage B MSE+CE (`--head zeroshot`) | no (aux labels unused) |
| ProFlip | **full class table** (`L_trig`/`L_CBS` CE) + clean CE to true labels (`L_troj`) | **yes** |
| TrojViT | **full class table** (CE clean + target) | **yes** |

Consequences:

- Flip-S is mechanically a perfect port, but its CE terms make it a **stronger-information
  baseline** than the proposed method. Either (a) keep it and label it
  "stronger-information", or (b) replace its CE with a target-cosine term (a documented
  deviation from Flip-S). Note that Flip-S's *search* would then differ from the
  published method.
- ProFlip/TrojViT need labels on top; if labels are excluded, their clean-stealth term
  must become a clean-feature MSE (no labels), as done in `proflip_cbs.py`.
- Evaluation always uses the full text table for every method (argmax over classes); that
  is the readout, not attack-time information, and is common to all rows.

---

## 6. Verdict summary

| Method | Adaptable parts | Blocked / not faithful | Status | Recommendation |
|---|---|---|---|---|
| **Flip-S** | everything (binding only) | original DeiT head; 4-bit variant; token offset; CE uses full class table (§5.5) | ✅ implemented + run in ZS | primary external baseline |
| **ProFlip** | CBS selection rule; trigger geometry/quantizer | SNI `I_t`/`c`; last layer/biases; CE/labels | ✅ CBS in `proflip_cbs.py`; SNI/trigger not | CBS component row + cite full method |
| **TrojViT** | Phase 1 (salience, attention loss, PCGrad, trigger) | NGR + head-row distillation; per-image trigger; training-time metric | ❌ not implemented | trigger-only + cite; visual-encoder variant only if required |

**Bottom line.** Flip-S is the only baseline whose *entire* published method survives the
move to CLIP with binding-only changes. ProFlip survives as a **bit-search component**
(CBS), not as a full method. TrojViT survives only as a **trigger generator**; its
"Trojan insertion" is a classifier-head operation that CLIP zero-shot does not have.

---

## 7. Implementation map

| File | What it does | Status |
|---|---|---|
| `bfa_baselines/Flip-S/target-clip/flip_s_clip_v2.py` | Full Flip-S on CLIP ZS; Stage A/B verbatim + DELTA 1–7 | done, run (2026-11-09) |
| `bfa_baselines/proflip_cbs.py` | ProFlip CBS (S1/S2/S3) on the pipeline's component pool, shared trigger, CLIP objective | done (this work) |
| `bfa_baselines/ProFlip/` | Released CNN CBS/trigger code (reference, no ImageNet variant) | read-only reference |
| `bfa_baselines/TrojViT/` | Released DeiT/ViT/Swin trojan code | read-only reference |
| `bfa_baselines/trojvit_clip.py` | **Not created** — would implement Phase 1 (and optionally the adapted insertion) | pending decision |

### If TrojViT Phase 1 is built

Minimum pieces:

1. `CLIPVictim`-style wrapper + `QuantMHA` for attention maps (reuse
   `flip_s_clip_v2.py` scaffolding).
2. Salience: `∇_x CE(logit_scale·normalize(f(x)) @ Tᵀ, target)`, abs, sum over 16×16
   patches, top-9 per image.
3. Attention loss: `−log Σ attn_{i→T}` per layer/head; token offset +1; PCGrad against
   the CE gradient (port `utils.PCGrad`).
4. Optimize a masked additive perturbation with Adam/clamp; store the per-image masks
   (or fix them) for evaluation.
5. Evaluate with the shared `classes.txt` + 13 templates; print ZS-CACC / ZS-ASR /
   clean-as-target; optionally `--save-trigger` for reuse in the BFA pipeline.

### If ProFlip SNI/trigger is built

Minimum pieces:

1. JSMA on the pooled CLS feature (768-d) using the ZS logits; calibrate `c` on
   LayerNorm'd features (e.g. `mean + k·std` at the selected dims).
2. Replace-mode bottom-right 73×73 trigger; `L = MSE(activation; c) + target-cosine`;
   SGD lr 0.1 / 100 epochs (or the released 10 iter / lr 0.01, documented).
3. Feed the trigger into `proflip_cbs.py`; report both the SNI and the no-SNI variants.

---

## 8. Open decisions

1. **ProFlip row:** CBS-only (cheap, component claim) vs full SNI+trigger (expensive,
   re-designed)? Current recommendation: CBS-only.
2. **TrojViT row:** trigger-only, adapted visual-encoder insertion, or citation only?
   Current recommendation: trigger-only + citation.
3. **Trigger area for each row:** own default vs matched 0.51 % vs matched 4.59 %/10.62 %.
   State it explicitly in every table.
4. **CE/labels:** are they allowed in a baseline if the proposed method excludes them?
   If yes, mark the baseline "stronger-information".
5. **TBN/TPN:** include as derived metrics in the backdoor table, never as "flips".
6. **Per-image masks:** TrojViT's paper selects patches per image; the pipeline uses a
   fixed patch. Decide before implementing, because it changes `apply_trigger` and the
   evaluation protocol.
