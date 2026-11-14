# BASELINES.md — Baseline implementation record (CLIP-FLIP thesis)

Everything known about the **baseline / comparison** half of the thesis, collected from
`proposed_method/modular_new_pipeline/{AGENTS,RESULTS,TASKS,ABLATION,DEFENSES,CODE}.md`
plus the code that actually implements it. This file is the reference for
"what baselines exist, what protocol they must follow, what is done, what is still missing".

> Companion docs (do not duplicate them):
> `proposed_method/modular_new_pipeline/RESULTS.md` §2 (comparison plan),
> `TASKS.md` §2 (LP protocol), `AGENTS.md` (pipeline/output layout),
> `bfa_baselines/Flip-S/AGENTS.md` (Flip-S repo notes).

---

## 1. What counts as a "baseline"

| Group | Method | Source of the method | Status |
|---|---|---|---|
| External attack | **Flip-S** (CVPR 2025) | `bfa_baselines/Flip-S/` — upstream repo + a CLIP adapter | **Implemented and run in the zero-shot task** (`target-clip/flip_s_clip_v2.py`) |
| External attack | **TrojViT** (CVPR 2023) | `bfa_baselines/TrojViT/` — upstream repo | **Not ported to CLIP** (DeiT-only code, no adapter, never run) |
| Internal attack | **Random bit-flip** | to be written in the proposed-method codebase | **Pending** — no `baselines.py`, no `--baseline` flag |
| Internal attack | **Simple gradient ranking** | `\|g·Δw\|` only, no layer score `R_l`, no hybrid weighting | **Pending** — closest existing code is the *fallback* ranker (not the same thing, see §6) |
| Ablation reference | **`--variant full`** | the proposed method itself | Done (used as the reference row of the ablation study, *not* a comparison baseline) |

Two other bit-flip repos sit in `bfa_baselines/` but are **not** named as baselines in the
pipeline docs and have no CLIP binding: `ProFlip/` (ResNet/CIFAR, ICCV 2021 extraction) and
`TBT-CVPR2020-master/` (ResNet-18/CIFAR). `proposed_method/AGENTS.md` also mentions
T-BFA / BadCLIP / TBT as repos kept "for comparison, out of scope"; BadCLIP lives in
`baselines/BadCLIP-DualEmbedding/` and is only used as a **class-name/eval-list source**,
not as an attack baseline.

---

## 2. Task decision: Linear Probe (LP) → Zero-Shot (ZS)

### 2.1 The original plan (LP task)

The comparison was first designed in the **Linear-Probe (LP) task** (`RESULTS.md` §2,
`TASKS.md` §2):

- train **one** probe per `(model, dataset)` on clean frozen CLIP features
  (`linear_probe.py`: balanced `--train-per-class` images, L2-normalised features,
  `nn.Linear(dim, n_classes)`, Adam lr=1e-3 wd=1e-4, CE, 60 epochs, batch 256,
  saved to `{lp_head_root}/{model_tag}_{test_dataset}.pt` with `weight`/`bias`/`meta`);
- score **every** method (ours, Flip-S, TrojViT, random, simple-gradient) with that
  **same saved head** — `Evaluator.lp_evaluate`, `--task lp`;
- report `LP-CACC` (clean accuracy) and `LP-ASR` (triggered features classified as target);
- output group `--section baselines`, probes under `lp_probes/`.

Rationale recorded in `RESULTS.md` §2: "Every method … is evaluated with **that same
probe** — LP-CACC on clean features, LP-ASR on triggered features."

### 2.2 Why LP was abandoned

LP is **not a valid comparison** for this threat model, because it requires adding a
component to CLIP:

- the victim must grow a **new trainable `nn.Linear` classification head** that does not
  exist in the deployed CLIP model, and that head has to be **trained on clean data**;
- the attack is then scored through that added head (`f → probe → logits`), not through
  CLIP's own output. This both (a) changes the model architecture under evaluation and
  (b) makes the result depend on a probe that is not part of the shipped victim;
- the attack itself (BFA on the visual encoder) is only meaningful on the model as
  deployed — the clean way to compare is to read out **CLIP's native zero-shot
  classifier**.

This is stated in the Flip-S v2 header (written 2026-11-09, commit `6a0bd08`, message
*"edit flip-s clip"*):

> "With `--head zeroshot` (**the PRIMARY comparison mode, agreed 2026-11-09: direct
> attacks on CLIP, native zero-shot head, no additive components**) the text head uses
> the SAME class names and the SAME 13-template prompt ensemble as the proposed method's
> `TargetEmbedder.build_text_embeddings` … identical readout for every method's rows."

"No additive components" = no trained probe head. The probe path was kept only as a
legacy option (`--head probe`, `--stage probe`), explicitly marked *scaffolding*.

### 2.3 Consequences of the switch

- The primary Flip-S run is **ZS**: `--head zeroshot`, target class aligned with the
  proposed method (`--target-class 3` ↔ `--target-cls 3`).
- The shared-probe protocol (§7) is retained in the codebase for the LP task
  (`--task lp`) but is **not** the comparison protocol anymore.
- Metric names for the comparison table become **ZS-CACC / ZS-ASR** (the Flip-S harness
  prints `ZS-*` automatically when no probe is attached).
- Everything else (same victim loading, same class list, same prompt ensemble, same
  eval split, same target class) is unchanged — so rows stay comparable.

---

## 3. Comparison protocol (ZS) — what a baseline row must satisfy

| Item | Value |
|---|---|
| Victim | CLIP **ViT-B/16** via OpenAI `clip.load('ViT-B/16')` — same loading as the proposed method |
| Dataset | ImageNet-1k **val** (`../data/Imagenet/valid`, i.e. the shared `data/imagenet/val`) |
| Target class | **3** (`tiger shark` in `class_texts/imagenet1k/classes.txt`; proposed-method default) |
| Readout | Native zero-shot: 13-template ensemble mean of text embeddings, L2-normalised; `logit_scale * (f/‖f‖) @ Tᵀ`, `argmax` |
| Class list | `proposed_method/modular_new_pipeline/class_texts/imagenet1k/classes.txt` — **index-aligned** with ImageNet label ids (1000 lines, BadCLIP convention) |
| Prompt templates | Exactly the 13 strings in `target.py` `PROMPT_TEMPLATES` |
| Eval split | Full 50 000-image val, batch 128 |
| Metrics | **CACC** = clean top-1; **ASR** = fraction of triggered images predicted as target; **clean-as-target** = clean images predicted as target |
| Proposed-method extra metrics | ASR@10, ASR@20, Flips→95%, Time (ZS) — from its own Phase-4 search (`RESULTS.md` §0) |
| Seeds / repeats | Proposed method: static seed 42, 5 reps (`RESULTS.md` §1). External attacks: their own fixed seeds (Flip-S: aux split seed 0) |

Fairness rules from the docs:

1. Same victim weights/loading, same class names and prompt templates, same eval images,
   same target class.
2. Baseline **training code is out of scope**; only the **evaluation protocol** must
   match (`RESULTS.md` §2).
3. Baseline hyperparameters stay at their published values — no re-tuning to make the
   numbers look better or worse.
4. The attack method must not touch the head; only encoder linears are attacked
   (`RESULTS.md`/Flip-S header). Head choice therefore cannot bias attack mechanics.

### Table template (ZS version of `RESULTS.md` §2)

| Method | ZS-CACC | ZS-ASR |
|---|---|---|
| Flip-S | … | … |
| TrojViT | … | … (not run) |
| Random bit-flip | … | … (pending) |
| Simple gradient ranking | … | … (pending) |
| **Ours** | … | … |

---

## 4. Flip-S — how the unchanged method runs on CLIP in the ZS task

### 4.1 Files

| Path | Role |
|---|---|
| `bfa_baselines/Flip-S/target-imagenet/deit-gen_trigger-int8.py` | original Stage A (trigger generation), DeiT-base int8 — **source of truth** |
| `bfa_baselines/Flip-S/target-imagenet/deit-targeted-int8.py` | original Stage B (bit-flip search) — **source of truth** |
| `bfa_baselines/Flip-S/utils/quant_model.py` | `find_all_bnbLinear`, `replace_with_myLinear`, `my_8bit_linear` (attack surfaces) |
| `bfa_baselines/Flip-S/utils/load_data.py` | `my_ImageDataset`, `load_split_ImageNet1k_valid` |
| `bfa_baselines/Flip-S/utils/metrics.py` | `imagenet_acc`, `imagenet_asr` (top-5 is always 0 — do not report it) |
| `bfa_baselines/Flip-S/target-clip/flip_s_clip.py` | v1 adapter (HF `transformers` CLIP) — superseded |
| `bfa_baselines/Flip-S/target-clip/flip_s_clip_v2.py` | **v2 adapter — the one that was run** (OpenAI `clip` package) |

`baselines/Flip-S/target-clip/*` is a byte-identical git-tracked copy of the same two
adapters (`diff -rq` clean); `bfa_baselines/` is the working copy.

### 4.2 The idea: keep the method, swap the victim binding

The published Flip-S scripts are written against a HuggingFace DeiT classifier and expect
three things from the victim:

1. `model.deit.encoder` — an iterable of transformer blocks whose `named_modules()`
   yields the quantized linears to attack;
2. `model(x)` → object with `.logits` (for the CE/MSELoss) and, with
   `output_attentions=True`, `.attentions` (per-head attention maps for Stage A's
   attention loss);
3. linears that expose the two attack surfaces `absmax` (fp16 scale factors) and
   `w_int` (int8 weights) after `replace_with_myLinear`.

The adapter supplies exactly these, so **no line of the search/trigger logic changes**.
`CLIPVictim` is the whole trick:

```python
self.model = clip_model
self.deit  = SimpleNamespace(encoder=clip_model.visual.transformer)  # DELTA 1
# forward: output = self.model.encode_image(x) -> probe or text head -> .logits
# attentions: captured by QuantMHA (DELTA 4), returned as .attentions
```

Consequences:

- Stage B's `for name, layer in model.deit.encoder.named_modules(): if
  isinstance(layer, my_8bit_linear)` now walks CLIP's `visual.transformer.resblocks`;
  `best_bit = f'{name}@{idx}@{i}@absmax|weight'` keys stay valid because names come from
  that same walk.
- Stage A's `model(...)` call sees CLIP's `encode_image` output pushed through the
  **zero-shot text head**, so the verbatim `CrossEntropyLoss(logits, target)` trains the
  trigger against CLIP's own classifier.

### 4.3 The 7 documented deviations (all binding/plumbing — method untouched)

| # | Delta | Why |
|---|---|---|
| 1 | **Victim binding**: `clip.load('ViT-B/16')` in fp16 replaces HF DeiT; `model.deit.encoder` aliases `model.visual.transformer`; `.logits`/`.attentions` mimic `DeiTForImageClassificationWithTeacher` | run the method on the thesis victim |
| 2 | **Attack surface**: CLIP fuses q/k/v into one `in_proj_weight`; it is wrapped as **one quantized module per block** (`attn.qkv`) so q/k/v scale-factors and weights are all attackable. Per block: `attn.qkv` (2304 rows), `attn.out_proj` (768), `mlp.c_fc` (3072), `mlp.c_proj` (768) → **48 attackable modules** over 12 blocks | match the paper's ViT surface |
| 3 | **Logit head**: linear probe on frozen features (legacy) **or** zero-shot text head (`--head zeroshot`, primary). Probe readout is L2-normalised; the shared head must be the one from `linear_probe.py` | head must not bias the attack; ZS readout is the comparison |
| 4 | **Attention maps**: OpenAI CLIP blocks discard attention weights, so `QuantMHA` recomputes the identical scaled-dot-product attention `(B, heads, S, S)`; token offset is **+1** (CLIP has 1 special token; DeiT-distilled has 2) | Stage A's attention-rollout loss needs per-head maps |
| 5 | **Quantized forward**: explicit dequant `x @ CB * SCB/127` instead of the bnb int8 kernel (the stock kernel consumes `weight.CB` on forward, which breaks `replace_with_myLinear`; verified bnb 0.49.2) | keep the method's replacement step working |
| 6 | **Harness**: torchvision `ProcessorShim` for `utils/load_data`, trigger filename sanitisation, `makedirs`, CLI stages | plumbing |
| 7 | **Eval harness**: `lp_eval_full` scores the comparison metrics on the **full** val split (the verbatim `small_val_loader` metrics are kept untouched); `--stage probe` warns that it fits a *scaffolding* probe, not the shared head | table comparability |

Everything else is fixed: search loop, trigger/attention loss, bnb quantisation semantics
(CB/SCB from `bnb.functional.int8_vectorwise_quant`), `DataLoaderArguments` /
`TriggerArguments` / `AttackArguments`, `schedule_lr`, the 4096/2048 row bans, trigger
geometry, loss composition. In-file the claim is "byte-identical to the original" for the
Stage-B search loop.

### 4.4 Victim + attack surface construction

`load_clip(model_name, head, probe_path, classnames_path, traindir)`:

1. `model, preprocess = clip.load('ViT-B/16', device='cuda:0')` → fp16 weights.
2. For every `resblocks[i]`:
   - `blk.attn = QuantMHA(blk.attn)` — builds `nn.Linear(768, 2304)` from
     `in_proj_weight/in_proj_bias`, converts it with `convert_linear` → `attn.qkv`
     (one `DequantLinear8bitLt`, `has_fp16_weights=False`, `threshold=6.0`), converts
     `out_proj` similarly;
   - `blk.mlp.c_fc`, `blk.mlp.c_proj` → `convert_linear`.
3. `convert_linear` quantises with `bnb.functional.int8_vectorwise_quant` (the exact scales
   bnb itself produces), stores `CB` (int8 weights) + `SCB` (per-row fp16 absmax), and
   sets `m.source_cls` so `replace_with_myLinear` can rebuild the module.

Attack surfaces (per `my_8bit_linear`):

- `absmax` — fp16 `Parameter`, length = out_features (per-row scale);
- `w_int` — fp16-typed copy of the int8 weight matrix (`CB`), values in `[-127, 127]`.

`my_8bit_linear.forward` computes `F.linear(x, w_int) * absmax/127 + bias`, i.e. the
quantised model's inference. **Only these two tensors are ever flipped.**

### 4.5 The ZS head (`--head zeroshot`)

```python
PROMPT_TEMPLATES = [ ...13 strings, verbatim from target.py... ]

def build_text_features(model, classnames, device):
    # lower-case, '_' -> ' '; every class x 13 templates
    # encode in chunks of 256 with clip.tokenize
    # reshape [n_classes, n_templates, dim] -> mean over templates (RAW) -> L2-normalise
```

- `classnames` from `--classnames` (default
  `../../../proposed_method/modular_new_pipeline/class_texts/imagenet1k/classes.txt`);
  `_load_classnames` strips blank lines and keeps **file order** = ImageNet id order.
- `logit_scale = model.logit_scale.detach().clone()` (CLIP's learned log-temperature).
- Forward: `logits = logit_scale * F.normalize(feats) @ text_featuresᵀ`.
- Identical math to `TargetEmbedder.build_text_embeddings` in
  `proposed_method/modular_new_pipeline/target.py`, so the two codebases produce the
  same readout for the same weights.

This is *the* answer to "how did we run Flip-S in the ZS task without changing it":
Flip-S only ever asks the victim for **class logits**; pointing that request at CLIP's
zero-shot text head instead of a trained probe is a binding change, not a method change.

### 4.6 Stage A — trigger generation (verbatim, with fixed values)

Source: `target-imagenet/deit-gen_trigger-int8.py`. Body copied line-for-line; only
model binding, token offset, makedirs/filename differ.

| Parameter | Value |
|---|---|
| `TriggerArguments.num_patch` | 9 |
| `train_attack_iters` | 300 |
| `patch_size` | 16 |
| `atten_loss_weight` | 1.0 |
| `target_class` | 3 (CLI override; paper default 2) |
| Data | `aux_num=32`, `seed=0`, `batch_size=64`, `split_ratio=0.1`, val split carved from ImageNet val |
| Trigger init | zeros `(3, 224, 224)`, `requires_grad=True` |
| Mask | bottom-right **3×3 patch block = 48×48 px**: patch ids `[195,194,193,181,180,179,167,166,165]` → rows 11–13, cols 11–13 of the 14×14 grid; `+1` for CLIP's single CLS token |
| Optimiser | sign-SGD: `trigger -= sign(∇_trigger L) * mask * schedule_lr(it)`; `clamp(0, 1)` |
| LR schedule | `lr=100`; `epoch<300`: `100/3^(it//30)` (floor 0.2); else `100/5^(it//30)` (floor 0.05) |
| Loss | `CE(logits, target) + 1.0 * attention_loss` |
| Attention loss | attention rollout over blocks with residual + row renormalisation; `(mean_joint_attention_to_trigger_tokens − 100)²` → since the rollout mass is ≤1, this monotonically **maximises** the attention mass the output tokens place on the 9 trigger patches |
| Output | `./trigger/ViT-B-16-c{target_class}-trigger-int8.pkl` (relative to `target-clip/`) |
| Post-check | `imagenet_asr` on `small_val_loader` (10 % of val after removing the 32 aux images) |

### 4.7 Stage B — bit-flip search (verbatim, with fixed values)

Source: `target-imagenet/deit-targeted-int8.py`.

| Parameter | Value |
|---|---|
| `AttackArguments.topk` | 50 (absmax candidates) |
| `topk2` | 50 (weight candidates) |
| `gamma` | 1.0 |
| `target_bit` | 50 flips (loop runs `target_bit + 10` = 60 external iterations) |
| Data | `aux_num=32`, `seed=0`, `batch_size=64`, `pin_memory=False`, `split_ratio=0.1`, `val` batch 128 |
| Trigger | loaded from §4.6 pkl |
| Mask | `mask[176:224, 176:224] = 1` (fp16) |
| Loss per candidate | `[MSE(logits_clean, cached_clean_logits) + gamma·CE(logits_triggered, target)] / (1+gamma)` |
| Clean reference | logits of the same model **before** `replace_with_myLinear` are cached over the aux set |
| Bit representation | 16-bit two's-complement `base` for `absmax`; 8-bit `base2` for `w_int` |
| Candidate rule | flip the bit only if `(new − old) * grad <= 0` (descent); NaN/Inf → revert; evaluate loss over the full aux set; keep the **global min-loss bit** |
| Order | absmax pass first; if no improving absmax bit, weight pass |
| Row ban | after taking an absmax bit: ban the 2 048-element chunk `(idx // 2048) * 2048 … +2048` (fp16 = 2 bytes → 4 096-byte row); after a weight bit: ban the 4 096-element chunk (int8 = 1 byte) — prevents flipping many bits in one hardware row |
| Re-flip rule | after applying the chosen flip, the code tries `changed_bit.remove(best_bit)`: if the bit had been flipped earlier, this **re-toggles it back** (net undo), prints "Revoke Flip" and un-bans its chunk; if not, `add(best_bit)`. Stop when `len(changed_bit) >= 50` |
| End of run | `imagenet_acc` + `imagenet_asr` on `small_val_loader`, then `lp_eval_full(...)` on the full val split with tag `after N flips` |

`lp_eval_full` is head-agnostic: it prints `ZS-CACC / ZS-ASR / clean-as-target` when no
probe is attached (ZS mode) and `LP-*` when a probe is attached. Trigger compositing is
identical to `utils/metrics.imagenet_asr`:
`normalize(x.float16*(1-mask) + trigger*mask)`.

### 4.8 Data / preprocessing glue

- `my_ImageDataset(..., use_ir=True)` returns **un-normalised** `[0,1]` tensors
  (`ProcessorShim(do_normalize=False)`), because the method applies `Normalize` itself.
- `ProcessorShim` exposes `image_mean` / `image_std` and strips the `Normalize` step from
  `clip.load`'s torchvision pipeline.
- `load_split_ImageNet1k_valid`: seeded `random_split` removes `aux_num=32` images for the
  attack set; the rest is `val`; `split_ratio=0.1` makes the verbatim `small_val_loader`.
- Images are cast to the model's `conv1.weight.dtype` (fp16) at forward.

### 4.9 The exact ZS run

Documented primary comparison run (`flip_s_clip_v2.py` header, agreed 2026-11-09):

```bash
cd bfa_baselines/Flip-S/target-clip
python flip_s_clip_v2.py --head zeroshot --target-class 3 --stage all
```

All other arguments keep their defaults, so the effective configuration is:

| Argument | Effective value |
|---|---|
| `--model` | `ViT-B/16` |
| `--head` | `zeroshot` (native text head) |
| `--target-class` | `3` (= proposed method's `--target-cls` default) |
| `--stage` | `all` = **trigger + attack** (the `probe` stage is *not* run in ZS mode) |
| `--valdir` | `../data/Imagenet/valid` (relative to the Flip-S repo root) |
| `--traindir` | `../data/Imagenet/train` (only used by the skipped probe stage) |
| `--classnames` | `../../../proposed_method/modular_new_pipeline/class_texts/imagenet1k/classes.txt` |
| `--probe-path` | unused in ZS mode (`./probe/clip-vit-b16-linear-probe.pt` default) |
| `--probe-batch/-epochs/-lr` | 256 / 20 / 1e-3 — scaffolding only, unused here |

Environment: conda env **`vlm_attack`** (`.idea/clip_bfa.iml` points at
`$HOME/miniconda3/envs/vlm_attack`; bash history shows the `(vlm_attack)` prompt).
Needs `torch`, `torchvision`, OpenAI `clip`, `bitsandbytes` (0.49.2 verified),
`bitstring` (3.1.9 from a local wheel), `tqdm`, `numpy`, `Pillow`; CUDA is hardcoded.

Artefacts the run produces (all relative to `target-clip/`):

```
trigger/ViT-B-16-c3-trigger-int8.pkl      # Stage A output
stdout:  CE loss / atten loss / atten mean per 5 iterations
         [Before Attack] Acc@1, Asr@1, ZS eval (full val)
         [+] ext_epoch ... loss ...
         [+] change <layer>@<idx>@<bit>@absmax|weight, loss: ...
         [-] Revoke Flip ...
         [+] Flip 50 bit: {...}
         ZS eval (after 50 flips, n=50000):
             ZS-CACC ...  ZS-ASR ...  clean-as-target ...
```

### 4.10 Evidence that it was run, and what is missing

- `__pycache__/flip_s_clip_v2.cpython-313.pyc` is dated **2026-11-09 16:57** and its header
  records a source size of **54 040 bytes** — exactly the size of the version committed as
  `6a0bd08` ("edit flip-s clip", the commit that *added* the ZS head and `lp_eval_full`).
  So that exact ZS version was imported/executed in the Python-3.13 `vlm_attack` env.
  (A `cpython-310.pyc` from 2026-11-04 is the pre-ZS version of v2.)
- The adapter header, the commit `6a0bd08` ("edit flip-s clip", 2026-11-09) and its diff
  (which *added* the multi-template ZS head and the `lp_eval_full` harness) are the
  authoritative record of the ZS mode.
- **No run logs, no result JSON/CSV, no trigger `.pkl`, no quantised checkpoint are kept
  in the repo.** The `trigger/` and `probe/` directories do not exist under
  `target-clip/`. If the numbers are needed for the thesis, the run must be repeated
  (or the GPU-box logs recovered).
- `clip_bfa/data/imagenet` is currently **empty**; `prepare_imagenet.py` builds
  `imagenet/{train,val,valid->val}`. Flip-S defaults resolve to
  `<Flip-S>/data/Imagenet/{valid,train}` (capital `I`), so on Linux either pass
  `--valdir /path/to/imagenet/val` explicitly or create
  `bfa_baselines/Flip-S/data/Imagenet -> ../../../data/imagenet`.
- Flip-S reports **final** CACC/ASR at its fixed budget (50 flips), plus a
  before-attack row. It does **not** evaluate per flip, so it cannot natively fill
  `ASR@10`, `ASR@20`, or `Flips→95%`; those columns are only produced by the proposed
  method's Phase-4 search. If a flip-budget curve is required for Flip-S, the adapter
  must add per-flip evaluation (a harness change, not a method change).

### 4.11 Audit notes — code-level deviations beyond the `# DELTA` list

Found by diffing the working copy against commit `6a0bd08` and against
`target-imagenet/deit-targeted-int8.py`. None of these changes the published method, but
they matter for reproducing the exact numbers:

1. **Extra backward call** (commit `236a298`, 2026-11-10): the current
   `bitflip_attack` has both

   ```python
   loss.backward(retain_graph=True)
   loss.backward()
   ```

   while the original has only the first line. The second call accumulates the same
   gradients again → every gradient is **2×**. Because the search only uses gradient
   *magnitudes for top-k* and the *sign* of `(new−old)·grad`, a uniform 2× scaling is
   ordering/decision invariant — but it doubles GPU memory/time and should be removed
   before re-running, or the numbers should be understood as "same method, doubled
   gradient scale".
2. **Clean-logit caching** (same commit): the original loaded a second full victim
   (`clean_model`) for the MSE clean reference; v2 caches the clean logits from the same
   model before `replace_with_myLinear`. Mathematically the same reference (the dequant
   forward reproduces the clean model), lower memory.
3. **Attention-matrix cleanup** (same commit): `torch.concat(...).permute(...).mean(dim=2)`
   was replaced by `torch.stack([a.mean(dim=1) for a in atten], dim=1)`. Both produce
   `(B, layers, S, S)` — bit-identical intent, no method change.
4. Unlike §4.3's list, these three are *not* tagged `# DELTA` in the file. Keep them in
   mind when claiming "verbatim method".

---

## 5. TrojViT (named baseline, not implemented)

`RESULTS.md` §2 lists TrojViT as an external baseline ("baseline repos under
`../baselines/`; evaluated under the same LP protocol — their training code is out of
scope here, only the evaluation protocol must match"). What actually exists:

- `bfa_baselines/TrojViT/` is the upstream CVPR-2023 repo: `main_patch_vit.py`,
  `models/`, `adversarialbox/`, DeiT/ViT/Swin training code, ImageNet val evaluation.
  It inserts a patch trigger by **training** the model (trojan insertion), which is a
  different threat model from post-deployment bit flips;
- there is **no CLIP adapter**, no `target-clip/` equivalent, and no result artifact;
- it has never been run in this thesis.

Consequently the TrojViT row in the comparison table is **empty**. Porting it would mean
the same kind of binding work as Flip-S (feed CLIP visual features/logits into its
training loop) plus a decision on how to compare a training-time trojan with a bit-flip
attack.

---

## 6. Internal baselines (still pending)

`RESULTS.md` §2 defines them; `RESULTS.md` "Code changes required" #4 marks them
**PENDING**; `AGENTS.md` lists "baseline rankers (random / simple-gradient, #4)" under
*Next*. There is **no `baselines.py`** and **no `--baseline`/`--baseline-mode` flag** in
`experiment.py` (only `--section baselines`, which is just an output folder).

### 6.1 Random bit-flip

- Flip **random candidate bits inside the selected components** (the Phase-2A
  layers + Phase-2B components), using the **same flip budget / stopping rules** as the
  proposed method (`MAX_FLIPS=80`, `NORMAL_STOP=94`, fine phase, best-prefix rollback…).
- Everything else — Phase 0 quantisation, Phase 1 trigger, eval protocol, seeds, target
  class — identical to `--variant full`.
- It isolates *how much of the ASR comes from ranking* vs. from the trigger + budget.

### 6.2 Simple gradient ranking

- Rank by plain **`|g·Δw|`** only: **no layer score `R_l`**, no hybrid weighting, no LSE
  term (equivalently `HybridRanker` with `R_l = 1`, `P3_ALPHA_LSE = 0`).
- It isolates the contribution of the layer/component scoring over a naive
  gradient-magnitude attack.

### 6.3 Where it plugs in (design already supported)

The pipeline is built for this swap (`AGENTS.md` "Module-independence rule"):

- `GreedySearcher` receives its `ranker` and `evaluator` by **constructor injection**
  from `pipeline.py`; a baseline is a class with the same `rank(...)` signature.
- Planned shape (from `RESULTS.md` #4): `baselines.py` with `RandomRanker` and
  `SimpleGradientRanker`, selected by a new `--baseline {ours,random,simple_grad}` in
  `experiment.py` / `build_config`, written under `--section baselines`.
- The `Candidate` dict fields are fixed by `search.py`
  (`tensor, layer, head, component, row, col, bit, old_int8, new_int8, delta_w,
  predicted_dL, layer_weight_R, gradient_score, final_score`); a baseline only needs to
  fill them and sort. `SimpleGradientRanker` can copy `HybridRanker.rank_fallback`'s
  gradient computation and use `layer_weight_R = 1.0`,
  `gradient_score = final_score = |ΔL|`.
- Not the same thing: `--variant c` (`POISON_ONLY_GRAD`) still uses the full hybrid
  metric; `--variant a` still uses `R_l = 1` *and* the LSE term; `rank_fallback` is the
  emergency path when candidate ranking yields nothing. None of them is the
  "simple gradient" baseline until a dedicated ranker + flag exists.

### 6.4 Fairness checklist for the new rankers

- Same Phase-1 trigger and same Phase-2 component pool as the full method.
- Same accepted-flip budget and stopping logic (otherwise the comparison is about the
  stopping rule, not the ranking).
- Same target class, eval split, seed, and metric implementation.
- Same run identity/output bookkeeping so `aggregate.py` puts the row next to `ours` in
  `summary.csv`.

---

## 7. LP protocol (historical, kept for reference)

If the LP task is ever revived (`--task lp`), the protocol is fully implemented and
unchanged from `TASKS.md` §2 / `RESULTS.md` §2:

1. **Train once per (model, dataset)** — `linear_probe.py`:
   - train split: balanced `--train-per-class` (default 50 → 50 000 for ImageNet-1k)
     from `ImageNet1kTrain` (ImageNet train subset prepared by `prepare_imagenet.py`);
   - test split: full `ImageNet1k` val; held-out sanity top-1/top-5;
   - features: L2-normalised `encode_image_batched` of clean FP32 CLIP;
   - head: `nn.Linear(dim, n_classes)`, Adam 1e-3 / wd 1e-4, CE, 60 epochs, batch 256;
   - saved to `{lp_head_root}/{model_tag}_{test_dataset}.pt` as `{weight, bias, meta}`
     (`model_tag`: `ViT-B/16` → `ViT-B16`).
2. **Score every method with that same head** — `Evaluator.lp_evaluate(model, loader,
   device, lp_head_path, trigger=…, target_cls=…, ap=…)`:
   - L2-normalised features → `F.linear(f, weight, bias)`;
   - **LP-CACC** = accuracy on clean features, **LP-ASR** = triggered features predicted
     as target;
   - wired via `--task lp`, `--lp-head` overrides the default path.
3. Flip-S's `--head probe` + `--probe-path` accepts exactly that checkpoint (same
   `{weight, bias}` keys) — that is what `DELTA 3/7` in `flip_s_clip_v2.py` prepare for.
   **But this is no longer the comparison mode** (see §2).

---

## 8. Results storage for baselines

`RESULTS.md` "Result file layout" (proposed-method runs):

```
Results/
├── runs.csv / summary.csv                       # aggregate.py
└── {section}/{model}/{dataset}/{task}/c{cls}/s{seed}/r{rep}/
    └── {variant}__t{sz}[__{loc}]/
        ├── bfa_results.json   # flip_log, checkpoints, metrics (ASR@10/@20, Flips→95%)
        ├── trigger.pt
        ├── original_weights.pt
        └── model_state_attacked.pt   # --save-attacked-model
lp_probes/
└── {model}_{dataset}.pt
```

- `--section {validation, baselines, ablation}` — `baselines` is the comparison group.
- `model_tag`, `task`, `target-cls`, `seed`, `rep`, `variant`, trigger size/location are
  all part of the run identity (`AGENTS.md`).
- Flip-S does **not** write into this tree today — its adapter prints metrics to stdout
  only. To land rows in `runs.csv`/`summary.csv`, a small writer (or a manual CSV row
  exported from the stdout of `lp_eval_full`) is needed.

---

## 9. Open items (checklist)

- [ ] Add `baselines.py` (`RandomRanker`, `SimpleGradientRanker`) + `--baseline` flag +
      `build_config` wiring (`RESULTS.md` #4 / `AGENTS.md` "Next").
- [ ] Run the two internal baselines on the same matrix/seed as `--variant full`
      (ViT-B/16, Caltech101 + ImageNet1k, seed 42) and aggregate them into
      `--section baselines`.
- [ ] Re-run / recover the Flip-S ZS numbers (no artifacts kept; ImageNet val must be
      prepared at `<Flip-S>/data/Imagenet/valid` or passed via `--valdir`).
- [ ] Decide whether to add per-flip evaluation to the Flip-S adapter for
      ASR@10 / ASR@20 / Flips→95% (currently final-budget only).
- [ ] Port TrojViT to CLIP or explicitly document it as "not compared" in the thesis.
- [ ] Optionally add a CSV/JSON writer for baseline runs so `aggregate.py` can consume
      them next to the proposed method.
- [ ] Keep `--head zeroshot`, `--target-class 3`, the shared class list and the
      13-template ensemble fixed for every reported baseline row.

---

## 10. Quick reference

```bash
# Flip-S, primary ZS comparison (unchanged method, CLIP victim, native text head)
cd bfa_baselines/Flip-S/target-clip
python flip_s_clip_v2.py --head zeroshot --target-class 3 --stage all

# same, with an explicit dataset location (when the default ../data/Imagenet/valid
# is not present)
python flip_s_clip_v2.py --head zeroshot --target-class 3 --stage all \
    --valdir /path/to/data/imagenet/val

# legacy LP mode (NOT the comparison protocol anymore)
python flip_s_clip_v2.py --head probe \
    --probe-path ../../../proposed_method/lp_heads/ViT-B16_ImageNet1k.pt \
    --target-class 3 --stage trigger
python flip_s_clip_v2.py --head probe \
    --probe-path ../../../proposed_method/lp_heads/ViT-B16_ImageNet1k.pt \
    --target-class 3 --stage attack

# proposed method, ZS reference row (for the same table)
cd ../../proposed_method/modular_new_pipeline
python experiment.py --model ViT-B/16 --dataset ImageNet1k --task zs \
    --section validation --target-cls 3 --seed 42 --variant full

# proposed method, LP task (historical protocol)
python linear_probe.py --model ViT-B/16
python experiment.py --model ViT-B/16 --dataset ImageNet1k --task lp \
    --lp-head ../lp_heads/ViT-B16_ImageNet1k.pt --seed 42
```
