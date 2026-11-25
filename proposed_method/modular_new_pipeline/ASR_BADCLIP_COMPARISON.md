# ASR / CACC evaluation — ours vs. BadCLIP (baseline)

How attack success rate (ASR) and clean accuracy (CACC) are computed in
`modular_new_pipeline/` vs. `../../baselines/BadCLIP-DualEmbedding/` (the code
that produced the BadCLIP paper's numbers). Written 2026-11-24 from a read of
both codebases; use it when writing the baseline-comparison / evaluation
sections and when a reviewer asks "is your ASR comparable to BadCLIP's?".

## TL;DR

Both use the **same zero-shot protocol**: one vector per class (ensemble mean
over N prompt templates), image embedding L2-normalised, prediction =
`argmax` of cosine similarities over **all C class vectors**; ASR = fraction of
triggered images predicted as the target class. That structure is identical —
it is the standard CLIP zero-shot evaluation.

The real differences:

| Aspect | Ours (`modular_new_pipeline`) | BadCLIP (`baselines/BadCLIP-DualEmbedding`) |
|---|---|---|
| Prompt templates per class | **13** (CLIP-paper subset) | **80** (full CLIP ensemble) |
| Template source | `target.py` `PROMPT_TEMPLATES` | `data/{dataset}/classes.py` `"templates"` |
| Ensemble order | mean raw embeddings → L2-norm **once** | L2-norm **each** → mean → renorm |
| ASR image set | **non-target-class images only** | ALL eval images (labels overwritten) |
| ASR readout | count `argmax == TARGET_CLS` | top-1 "accuracy" vs. forced target labels (same number) |
| Target definition | `text_emb[TARGET_CLS]` (text-only `t_target`) | class index of `--label` ('banana' = 954 on ImageNet) |
| Trigger at eval | applied in `quick/full_evaluate` via `apply_trigger` | applied in `Dataset.__getitem__` on the fly |
| Class names | `class_texts/*/classes.txt` = BadCLIP lists VERBATIM (CIFAR10/ImageNet1k) | `classes.py` `"classes"` |

## Ours — `evaluation.py` + `target.py`

**Class bank** — `TargetEmbedder.build_text_embeddings` (target.py:41-78):

1. For every class name (prompt-ready, from `data.py`): 13 templates
   (`target.py:18-32`) → `C × 13` prompts.
2. Encode with the text encoder (chunks of 256).
3. `all_emb.view(C, 13, d).mean(dim=1)` — mean of **raw** template embeddings —
   then `F.normalize(...)` **once** (target.py:69-70).

**Target** — `from_text` (target.py:80-83): `t_target = text_emb[TARGET_CLS]`,
the ensemble-mean row of the target class. Success = predicted index equals
`TARGET_CLS` in this bank.

**ASR / CACC** — `quick_evaluate` (evaluation.py:44-61):

```
poisoned = apply_trigger(imgs, trigger, ap)          # stamp trigger
emb_po   = normalize(model.encode_image(poisoned))   # attacked vision tower
ASR      = mean(argmax(emb_po @ text_emb.T) == target_cls) * 100
CACC     = mean(argmax(emb_cl @ text_emb.T) == labels) * 100   # clean images
```

`full_evaluate` (evaluation.py:63-94) is the same over the full test loader,
plus `cos_clean` / `cos_poison` diagnostics (mean cosine to `t_target`).

**Eval image set** — `prepare_data_splits` (data.py:250-276): `eval_idx` and
`test_idx` are carved from `other_indices` = images whose label is **not**
`TARGET_CLS`. Real target-class images never enter ASR, so every ASR success
is a genuine non-target → target flip.

**Linear-probe variant** — `lp_evaluate` (evaluation.py:100-190): `lp_asr` =
fraction of triggered features the saved probe head classifies as
`target_cls` (+ `lp_cacc`, `lp_clean_as_target`).

## BadCLIP — `src/evaluate.py` + `src/data.py`

**Class bank** — `get_zeroshot_metrics` (src/evaluate.py:55-83):

1. `classes, templates = config["classes"], config["templates"]` from
   `data/{dataset}/classes.py` (evaluate.py:60-61). The template list is the
   **80-template CLIP ensemble** ("a bad photo of a {}.", … "a tattoo of the
   {}.") — verified in `data/ImageNet1K/validation/classes.py`,
   `data/CIFAR10/test/classes.py`, `data/CIFAR100/test/classes.py`. NOT a
   single prompt.
2. Per class: encode the 80 prompts, **L2-normalise each template embedding**
   (evaluate.py:79), `mean(dim=0)` (evaluate.py:80), **renormalise** the mean
   (evaluate.py:81).

**Classification** (evaluate.py:94-97): `get_image_features` → L2-norm →
`logits = image_embedding @ text_embeddings` → `topk` ranks. Prediction =
argmax over all C class vectors — identical structure to ours.

**ASR — done on the data side** (`ImageLabelDataset`, src/data.py:233-301),
run with `--add_backdoor --asr --label banana`:

1. Every eval image gets the trigger **on the fly** in `__getitem__`
   (data.py:293-301).
2. Every CSV label is **overwritten with the poison target id** (data.py:254-
   259): `self.labels.values[idx] = self.poison_id`, `poison_id` = class index
   of `--label` ('banana' = 954 on ImageNet-1k).
3. `get_zeroshot_metrics` then reports ordinary top-1 "accuracy" against those
   labels — since all labels = target, accuracy = fraction of triggered images
   predicted as the target = ASR. (`backdoor_target_index` at evaluate.py:66-
   67 is computed but never used — vestigial.)

**CACC** — the same `get_zeroshot_metrics` without `--add_backdoor/--asr`:
clean images, TRUE labels, `zeroshot_top1` (also top-3/5/10) (evaluate.py:86-
107).

**Linear-probe / finetune ASR** (evaluate.py:238-242, 269-273, 334-338):
labels are NOT overwritten here; keep samples with `label != 954`, count
`prediction == 954`, divide by total samples — the same idea as our
`lp_asr`.

## Consequences — which one is stricter?

1. **ASR denominator.** BadCLIP counts real target-class images as successes
   when they are predicted as target (their labels are forced to target too).
   On ImageNet-val (~50 banana of 50 000) that inflates ASR by < 0.1 pp;
   datasets with fatter target classes inflate more. Ours excludes target-
   class images entirely → slightly stricter and cleaner.
2. **13 vs. 80 templates.** The 80-ensemble is a mildly stronger classifier;
   ASR and CACC can shift a little with the readout. Structure is the same,
   so numbers are comparable, but cite the template count in the eval table.
3. **Norm order.** `norm → mean → renorm` (BadCLIP) weights all templates
   equally; `mean → norm` (ours) lets high-norm template embeddings dominate.
   Note our repo currently mixes conventions: `build_negative_embeddings`
   (target.py:85-109) already follows the BadCLIP order ("per-template
   norm → mean → renorm", p0_test convention) while `build_text_embeddings`
   does not.
4. **Readout structure** (argmax over all C ensemble vectors) is identical —
   nothing to fix there.

## If exact BadCLIP parity is ever needed (NOT applied)

1. Switch `build_text_embeddings` to per-template L2-norm → mean → renorm
   (matches `build_negative_embeddings` and BadCLIP).
2. Optionally adopt the 80-template ensemble from
   `baselines/BadCLIP-DualEmbedding/data/*/classes.py`.
3. Keep our stricter ASR set (non-target images only), or report both — ours
   is the defensible choice; BadCLIP's is the citable convention.

## File references

| What | Ours | BadCLIP |
|---|---|---|
| templates | `target.py:18-32` | `data/*/classes.py` `"templates"` |
| class bank | `target.py:41-78` | `src/evaluate.py:68-83` |
| t_target / target id | `target.py:80-83` | `src/evaluate.py:66-67`, `src/data.py:254-259` |
| ASR / CACC | `evaluation.py:44-61`, `63-94` | `src/evaluate.py:55-118` + `src/data.py:233-301` |
| eval image set | `data.py:250-276` | full eval CSV (all classes) |
| LP ASR | `evaluation.py:100-190` | `src/evaluate.py:326-342` |
