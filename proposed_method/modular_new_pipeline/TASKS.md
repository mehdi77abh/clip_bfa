# TASKS.md — Evaluation tasks: zero-shot (zs) · linear-probe (lp) · image-to-text retrieval (i2t)

Selected with `--task {zs,lp,i2t}`.  The task is part of every output path:

```
{out_root}/{model}/{dataset}/{task}/c{target_cls}/s{seed}/{variant}__t{trigger_size}/
```

Only **zs** is implemented today; **lp** and **i2t** are future work and
`--task lp|i2t` currently raises `NotImplementedError` (see each section for
exactly what needs to be built).  Metrics definitions live in RESULTS.md §0.

---

## 1. `zs` — zero-shot classification — **IMPLEMENTED**

The attack is evaluated exactly where it is optimised: CLIP's zero-shot
classifier (image embedding vs. class text embeddings, multi-template
prompt ensemble).

- **Pipeline:** the full Phase 0→4 attack; evaluation via
  `Evaluator.quick_evaluate` / `full_evaluate` (argmax over text embeddings).
- **Reported (validation table):** ASR@10, ASR@20, Flips→95%, CACC, Time (ZS).
- **Status:** done — `--task zs` is the default.

## 2. `lp` — linear-probe classification — **IMPLEMENTED**

Goal: show the attack transfers to a *learned* downstream head — the setting
used for the **comparison** against Flip-S and TrojViT (RESULTS.md §2).
The probe is trained **once per (model, dataset)** and shared by every attack
method, so the comparison is fair.

**Protocol:**
1. `linear_probe.py` (standalone — **implemented**, 2026-11-09):
   - train split: balanced `--train-per-class` images/class (default 50 →
     50 000 for ImageNet-1k) from the `--train-dataset` registry entity
     (default `ImageNet1kTrain` = ImageNet-1k train), seeded;
   - test split: the full `--test-dataset` entity (default `ImageNet1k` =
     ImageNet-1k val) — held-out sanity check (top-1 / top-5);
   - frozen clean features: L2-normalised `encode_image_batched`
     (Phase-5e convention of `../main_new_pipeline_lp.py`);
   - head: `nn.Linear(dim, n_classes)`, Adam (lr=1e-3, wd=1e-4),
     cross-entropy, 60 epochs, batch 256 (Phase-5e values);
   - saved to `{lp_head_root}/{model_tag}_{test_dataset}.pt` (`lp_head_root`
     comes from `config.json` → `paths`): `weight`/`bias` tensors (drop-in
     for the probe loader of `../main_new_pipeline_imagenet.py` / Flip-S)
     + `meta` (model/dataset/training/eval metadata).
2. Evaluation with that saved head (**implemented**, 2026-11-10):
   `Evaluator.lp_evaluate(model, loader, device, lp_head_path, trigger=…,
   target_cls=…, ap=…)` in `evaluation.py` — L2-normalised features through
   the saved `weight`/`bias` (`F.linear`):
   - **LP-CACC** — accuracy on clean features;
   - **LP-ASR** — fraction of triggered features classified as the target.
   Wired via `--task lp` (pipeline runs it after Phase 4 on the test split;
   `--lp-head` overrides the default `{lp_head_root}/{model_tag}_{dataset}.pt`).
3. All methods (ours, Flip-S, TrojViT, random/simple-gradient baselines)
   are scored with the **same saved head**.

## 3. `i2t` — image-to-text retrieval — **FUTURE**

Goal: show the trigger hijacks *retrieval*, not just classification — query
images retrieve their class text among all class prompts.

**Protocol (planned):**
- **Gallery:** the class text embeddings (`text_emb`, multi-template means).
- **Query:** image features — clean images, and triggered images.
- **Metrics:** R@1 / R@5 / R@10 / R@20 for:
  - clean images → ground-truth class text (baseline, should stay high);
  - triggered images → ground-truth class text (attack should push this down);
  - triggered images → **target** class text (attack should push this up —
    the retrieval counterpart of ASR).
- The removed reference helper `Visualizer.plot_retrieval_probe` (kept,
  unwired) already plots exactly these conditions (`clean_clean`, `atk_clean`,
  `atk_trig`, …) — reuse its condition naming.

**Code needed:** a retrieval evaluator (encode queries, cosine sim against
`text_emb`, recall@K) — new methods in `evaluation.py` or a small
`retrieval.py` — plus the `--task i2t` branch.

---

## Adding a task later

1. Implement the task's evaluation (above).
2. Remove/relax the `NotImplementedError` guard in `experiment.build_config`.
3. Nothing else changes: the task is just a path segment + an evaluation
   branch; the attack pipeline (Phases 0–4) is task-independent.
