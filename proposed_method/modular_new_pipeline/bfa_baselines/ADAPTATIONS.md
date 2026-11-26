# How each baseline was adapted to CLIP

Short version: **the attacks keep their original logic — what changed is the
victim binding, plus a few documented substitutions where CLIP has no direct
analogue.** Each baseline keeps its own trigger generator, search/ranking rule,
and hyper-parameters. Four things had to be re-bound for CLIP:

1. the **classifier head** → the zero-shot text head,
2. the **tensor surface** that gets bit-flipped,
3. **attention access** (both attack methods that need attention maps),
4. the **metric** (ASR/CACC from the shared evaluator, not a trained classifier).

Shared by all three: frozen `data_split.json` (eval 1000 / calib 384 /
search 256 / aux 32), 13-template prompt ensemble, and one evaluator
(`evaluate.py`). Budget is **200 flips for every method** (push-98); native
budgets are documented but not run (TrojViT 80, ProFlip 100, Flip-S 50).

| Method | Flip surface | What was re-bound |
|---|---|---|
| TrojViT | int8 codes of the projection weights | classifier row → target-relevant projection rows |
| ProFlip | int8 codes of attention/MLP/projection weights | JSMA neuron search → analytic projection score |
| Flip-S | **FP16 row-wise scales** (+ int8 weight codes) | bnb quantisation → identical local quantiser; zero-shot logits |

---

## TrojViT (CVPR 2023)

**Original:** a per-image patch trigger, then NGR selects the important
classifier neurons and distillation tunes that classifier row; the tuned
weights are written into the model.

**Adapted:**
- CLIP has no classifier. The last linear map into the joint embedding space
  plays that role: `visual.proj` for CLIP, `head.mlp.fc2.weight` for SigLIP.
  Each **embedding dimension** is treated as one "classifier row".
- NGR score per embedding dim: `|t_target[i]| · ‖∂L/∂proj[:, i]‖`; the top 8
  dims are selected and all their width entries are tuned.
- Distillation keeps the paper's recipe (200 epochs, SGD 0.01, MultiStepLR
  80/120/160), but trains on the **calib split** (never the test set), with
  `0.5·CE(clean) + 0.5·(1 − cos(triggered, t_target))`, restoring all
  non-selected entries after every step. Convergence pruning keeps
  `|Δw| ≥ 0.003`.
- The tuned values are realized as **int8 Hamming bit flips**, applied one
  whole entry at a time in descending `|Δw|` order until the budget is hit.
- Trigger stays native: 9 patches/image, Attention-Target loss with PCGrad.

**Deviation that matters:** the original distills on one fixed test batch;
this version trains on the frozen calib split. The original's "classifier
row" becomes target-relevant projection rows.

---

## ProFlip (ICCV 2021)

**Original:** JSMA finds significant neurons (SNI), a feature-anchored
trigger is optimized, then CBS searches critical bits over all quantised
Linear/Conv weights.

**Adapted:**
- **SNI is analytic.** Because `emb = h @ P` (or `h @ Pᵀ` on SigLIP), the
  influence of pre-projection dim `j` on the target is exactly
  `|(P @ t_target)[j]|`; take the top 200 dims (default `--sni-k 200`).
  No JSMA loop — same criterion, cheaper and exact for a linear map.
- `h` is captured at the projection input: `ln_post` output (CLIP) or the
  input of `head.mlp.fc2` (SigLIP).
- Trigger is native geometry: fixed bottom-right square, side ≈ 10/32 of the
  input (70 px at 224), 10 plain-GD steps, loss
  `MSE(h(x+patch), y with y[I_t]=c) + CE(zero-shot logits, target)`.
  The clamp `c` is `max` over the calib batch by default (upstream's literal
  `10` is available as `--clamp-mode const10`).
- CBS keeps S1/S2/S3: fitness `|∂L/∂w|·(Q−w)`, tensor selection, top 8
  vulnerable entries per tensor, K=20 candidate grid, **multi-bit atomic
  element updates** (all differing bits flipped together).
- Search surface = attention + MLP weights + the projection, per-tensor int8.
- Both native `mse_sni` and a `cos` fitness variant are run.

**Deviation that matters:** the clean-CE term in the release fitness is
dropped (clean accuracy is measured by the evaluator instead); grid is 20
ascending points instead of the release's 10 descending.

---

## Flip-S (CVPR 2025)

**Original:** an int8 victim with FP16 row-wise scales; flip **scale-factor
bits** (primary) and weight bits (secondary), ranked by SFS.

**Adapted:**
- `bitsandbytes` is replaced by a deterministic local row-wise int8 + FP16
  scale quantiser with the same math (`w = codes · scale`, forward uses the
  dequantised weights), so the attack has full control over the bits.
- Surface = every 2-D vision weight (attention + MLP) **plus the projection**.
- The zero-shot head replaces the classifier logits; the search loss keeps
  the paper's form `(MSE(clean, cached clean) + γ·CE(triggered, target)) / (1+γ)`.
- Trigger stays native: sign-PGD on a 3×3-patch block at the bottom-right
  (48 px for patch-16), 300 iterations with the attention-rollout term,
  trained on the 32 aux images.
- 4 KB RowHammer mutual exclusion is kept; candidates are verified greedily
  by the suite's quick ASR; budget 200 (native 50).

**Deviation that matters:** acceptance uses quick ASR (the task metric)
instead of the aux loss; 4 KB exclusion is a ±4 KB window; ranking uses
`|grad·Δbit|` (the paper's SFS formula).

---

## SigLIP-B/16: the extra adaptations

SigLIP is an HF `transformers` model (`google/siglip-base-patch16-224`), not
OpenAI CLIP, so a thin **`SiglipAsClip`** facade exposes the attributes the
attacks expect (`encode_image`, `encode_text`, `visual.proj`,
`visual.transformer.resblocks`, …). It forwards the **real parameter names**
so `param_overrides` and evaluation always agree.

Differences handled per method:

| Issue | CLIP | SigLIP adaptation |
|---|---|---|
| Projection surface | `visual.proj` `[width, embed]`, embedding dim = **column** | `vision_model.head.mlp.fc2.weight` `[embed, width]`, embedding dim = **row** |
| Attention maps | wrapped `nn.MultiheadAttention` | loaded with `attn_implementation="eager"`; the encoder layer drops the weights, so captured via forward hooks on `self_attn` |
| Sequence layout | 1 CLS token + patches | **no CLS token** (`n_prefix = 0`), patch offsets shifted |
| Normalisation | CLIP mean/std | `0.5 / 0.5` |
| Text side | OpenAI tokenizer + `text_projection` | HF tokenizer + `pooler_output`; same 13 prompts; `logit_bias` omitted (class-independent → argmax unchanged) |
| Surface names | `.attn.in_proj_weight`, `.attn.out_proj.weight`, `.mlp.c_fc/c_proj` | `.self_attn.{q,k,v,out}_proj.weight`, `.mlp.{fc1,fc2}.weight`, plus head `fc2` |

Per method:

- **TrojViT** — NGR and distillation use the **transposed** fc2 view
  (`proj_matrix()`); attention-target loss reads the eager attention maps and
  uses `n_prefix = 0`. Head-internal attention and `fc1` are deliberately
  **not** part of the flip surface; only `fc2` plays the `visual.proj` role.
- **ProFlip** — `h` is the **input of `fc2`** (3072-dim post-`fc1`
  activation), captured with a pre-hook; SNI uses `|fc2ᵀ @ t_target|`.
- **Flip-S** — flips FP16 scale bits of `self_attn`/`mlp` tensors and the head
  `fc2`; Stage-A attention maps come from the same eager forward; no CLS
  offset.

Because both the attack and `evaluate.py` load SigLIP the same way (eager,
fp32, 0.5/0.5), the victim the attack sees is the victim that is evaluated.
The full line-by-line deviation list is in `README.md` (items 1–13).
