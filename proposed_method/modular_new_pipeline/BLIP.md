# BLIP applicability — methodology & formulation check

Scope note (2026-11-20): can the CLIP-FLIP formulation (Phases 0–4 of this
pipeline) be applied to BLIP?  **Short answer: yes for BLIP-1 (the ITC
model), with one explicit readout decision; BLIP-2/Q-Former is out of
scope for now.**  This document records what the formulation actually
assumes about the model, how BLIP satisfies each assumption, and what
must be stated / re-tuned.  The `BlipAdapter` is implemented and
registered as `BLIP-B/16` (§7, 2026-11-21).

## 1. What the formulation assumes (verified in code)

The math never assumes "CLIP" — it assumes exactly three things:

| # | Assumption | Where it appears |
|---|---|---|
| A1 | `encode_image` / `encode_text` live in a **shared embedding space** where cosine is meaningful | Phase-1 loss `-cos(emb_poisoned, t_target)/T` (`trigger.py`), Phase-3 gradients (`hybrid_ranking.py`), ablation scorers (`_make_target_cos`) |
| A2 | The attack surface is **ViT-style transformer blocks** (attention heads + MLP) | Phase-0 per-tensor int8 quantisation, Phase-2A/2B ablation, `get_component_refs` V/O head slicing |
| A3 | **Classification = `argmax_i cos(emb, text_emb[i])`** (prompt-ensemble zero-shot readout) | `Evaluator.quick_evaluate` / `full_evaluate` (`evaluation.py`), failure-set mining (`search.py`), ASR@10/@20, Flips→95%, NORMAL_STOP/FINE_TARGET |

Everything else — per-tensor int8 grids + bit-flip enumeration
(`quantizer.py`), poison−clean gradient specificity, `R_l × |g·Δw|`
ranking, the two-phase greedy search, drift penalty, rollback — is
weight-format agnostic and readout-agnostic.

## 2. BLIP-1 (BLIP, Li et al. ICML 2022) — phase-by-phase

BLIP's ITC model: ViT-B/16 vision tower (timm-style, **fused `qkv`**,
CLS token) + BERT-MED text side + `vision_proj` / `text_proj` heads into
a shared 256-d ITC space, contrastive (scaled-cosine) training,
CLIP-style image normalisation, 224² input.

| Phase | Transfer | Notes |
|---|---|---|
| 0 — int8 quantisation | ✅ unchanged | Block tensors only (attn/ln/mlp); patch-embed, pos-emb, final norm, projection stay fp32 — same convention as the current adapters. |
| 1 — trigger (`-cos(emb, t_target)`) | ✅ | The loss is exactly BLIP's ITC objective direction. **Hard requirement:** `encode_image` must return the **post-`vision_proj`** feature and `encode_text` the **post-`text_proj`** feature. Pre-projection features = cosine in the wrong space = every phase silently breaks. |
| 2 — layer/component ablation | ✅ | Fused `qkv` linear = OpenAI CLIP's `in_proj_weight`; V-slice convention `rows [2D+lo, 2D+hie)` applies verbatim. Attention is a *module* returning a plain tensor (timm style) → `replace_sublayer` is the simple module swap (SigLIP-mlp style, no tuple wrapper). |
| 3 — poison−clean gradient ranking | ✅ | Fully differentiable through ViT + `vision_proj`. `final_score = R_l × |g·Δw|` on the int8 grid is architecture-agnostic. |
| 4 — greedy search + metrics | ✅ mechanically | Once the readout (§3) is fixed: ASR@10/@20, Flips→95%, rollback, failure-set mining only need "prediction = argmax over some readout". |
| Trigger plumbing | ✅ | `get_seq_layout`: CLS → `n_prefix=1`, patch 16 (BLIP ViT-B/16). `get_norm_stats`: BLIP uses **CLIP mean/std** → the `[-3, 3]` clamp regime and pixel-space clamp are identical. |
| `--task lp` | ✅ | Linear probe is readout-agnostic by construction (frozen features). The most robust evaluation path for BLIP. |

## 3. The one real formulation gap: the readout (A3)

ASR/CACC are defined via CLIP's zero-shot prompt argmax.  BLIP is **not**
a zero-shot classifier by design (retrieval / captioning / ITM / VQA are
its native evaluations).  The cosine-argmax readout is definable on
BLIP's ITC space (and used in the literature), but it must become an
explicit part of the formulation.  Options:

1. **ITC-cosine argmax (recommended)** — keeps the entire math
   identical; ASR/CACC justified as "zero-shot classification in the ITC
   space".  State it as a protocol choice.
2. **ITM scoring** (image-text matching, cross-attention) — BLIP's
   stronger matching head; changes the *evaluation* only (Phases 1–3
   operate on ITC embeddings regardless).  More faithful to BLIP, less
   comparable across models.
3. **Linear probe (`--task lp`)** — already implemented; sidesteps the
   issue.  Report **in addition to** option 1 for BLIP rows.

The gap is *only* in the evaluation definition — Phases 1–3 are
unaffected.

## 4. BLIP-2 / InstructBLIP — out of scope (for now)

The formulation strains on the Q-Former architecture:

- The shared-space embedding is the **Q-Former output**, not the ViT's —
  the trigger must fight through a cross-attention bottleneck that can
  dilute the signal.
- Attack surface becomes odd: huge frozen ViT (EVA-CLIP ViT-g) vs. small
  Q-Former; Phase-2 "heads/MLPs" ablation semantics break on Q-Former
  cross-attention blocks.
- Zero-shot classification there is **generative** (caption-prompted),
  which breaks the cosine-argmax ASR definition entirely.

A BLIP-2 claim would need a separate re-derivation, not a drop-in.

## 5. Empirical caveats (re-tune / re-check, not formulation breaks)

1. **Modality gap.** The A/B experiment found `cos(text, t_img)=0.31`
   already problematic on CLIP.  BLIP's ITC space is 256-d and
   momentum-distilled — the text↔image gap for classification-style
   prompts is likely larger.  Expect weaker Phase-1 trigger base ASR →
   re-tune `P1_EPOCHS` / `TRIG_SZ`.  Keep the (unused)
   `build_target_embedding` image-prototype path in mind as a fallback —
   it hurt on CLIP, so treat it as an open question, not a default.
2. **Base CACC will be lower** (ITC-argmax zero-shot ≪ CLIP's).  ASR
   stays well-defined (argmax==target over poisoned images), so attack
   metrics remain comparable; interpret CACC columns accordingly.
3. **Prompt templates.** The 13-template ensemble is CLIP-tuned.  Keep
   the **same** templates for cross-model comparability; note as a fixed
   protocol.  (BLIP's caption-style pretraining may prefer different
   phrasing — a possible ablation, not a default change.)
4. **Search thresholds** (`NORMAL_STOP=94`, `FINE_TARGET=99`, flip
   budgets) are readout-agnostic but were tuned against CLIP's trigger
   base ASR — watch the Phase-A→B entry condition on the first BLIP runs.

## 6. Recommended thesis claim

> The attack formulation is encoder-agnostic: it requires (i) a
> differentiable image encoder whose output lives in a joint image–text
> embedding space, (ii) ViT-structured transformer blocks as the
> quantized attack surface, and (iii) an evaluation readout defined on
> the joint space.  BLIP's ITC model satisfies all three; we instantiate
> it with an ITC-space readout and additionally report linear-probe
> results.

## 7. Implementation (done 2026-11-21)

`BlipAdapter` in `adapters.py` (registry entry `BLIP-B/16`), verified by
smoke tests on a tiny random `BlipModel` (all adapter contract points) and
on the real checkpoint (geometry, ITC shapes, convention check, zero-shot
readout on Caltech images, Phase-0 + Phase-1 gradient).

- **Checkpoint:** `Salesforce/blip-itm-base-coco` (`BlipForImageTextRetrieval`)
  — the raw `blip-vit-base-capfilt` pretrain checkpoint is **not on the HF
  hub** (Salesforce distributes it outside the hub).  The ITM-coco
  checkpoint is BLIP-1 ViT-B/16 fine-tuned from capfilt and carries the
  ITC heads (`vision_proj`/`text_proj`, `image_text_hidden_size=256`);
  swap `hf_name` for `blip-itm-base-flickr` / `-large` variants if wanted.
  `load_model` reads the checkpoint's class from its config (BlipModel vs
  BlipForImageTextRetrieval attribute names) — loading with the wrong class
  would silently randomise the ITC projections.
- **ITC convention (verified against transformers 5.x source):** the adapter
  returns `vision_proj(vision last_hidden_state[:, 0, :])` /
  `text_proj(text last_hidden_state[:, 0, :])` — raw `[CLS]`, exactly
  `BlipForImageTextRetrieval`'s `use_itm_head=False` path.  Deliberately
  NOT `BlipModel.get_image_features` (applies `post_layernorm` twice) or
  `get_text_features` (routes through the BERT tanh pooler; the ITC text
  encoder has no pooler at all).  Pitfall check is part of the smoke.
- `get_block_quant_params`: fused `qkv` (= `attn_ip_w/b`) + `attn.proj`
  (`attn_op_w/b`), `ln1/ln2`, `mlp.fc1/fc2` — canonical names identical
  to `OpenAICLIPAdapter` so `sub_quant` keys line up across models.
- `get_component_refs`: V slice on fused `qkv` rows `[2D+lo, 2D+hie)` —
  the `OpenAICLIPAdapter` math verbatim.
- `replace_sublayer`: timm-style `BlipEncoderLayer` calls `self_attn` /
  `mlp` as modules returning (hidden, weights) 2-tuple / tensor → shared
  `_AttnConst` tuple-wrap + plain module swap (SigLIP pattern).
- `get_seq_layout`: CLS prefix (`n_prefix=1`), patch 16, 224 input;
  `get_norm_stats`: CLIP mean/std (identical clamp regime).
- Tokeniser: `max_length=35` — BLIP's training truncation length
  (Salesforce `med.py`).
- Runtime env note: run BLIP configs with the `vlm_attack` conda env
  (transformers ≥ 5; the ITC convention differs in 4.x — see §2).
  `class_texts/` per dataset are unchanged (names identical across models).
