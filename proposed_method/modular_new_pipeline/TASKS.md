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

## 2. `lp` — linear-probe classification — **FUTURE**

Goal: show the attack transfers to a *learned* downstream head — the setting
used for the **comparison** against Flip-S and TrojViT (RESULTS.md §2).
The probe is trained **once per (model, dataset)** and shared by every attack
method, so the comparison is fair.

**Protocol (planned):**
1. `linear_probe.py` (new standalone file):
   - extract frozen CLIP visual features on a clean train split per
     (model, dataset) using `encode_image_batched`;
   - train a logistic-regression head (adapt `LogisticRegression` + training
     loop from `../modular/probe_eval.py`);
   - save the head to `{lp_head_root}/{model}_{dataset}.pt`
     (`lp_head_root` comes from `config.json` → `paths`).
2. Evaluation with that saved head:
   - **LP-CACC** — accuracy on clean test features;
   - **LP-ASR** — fraction of triggered features classified as the target.
3. All methods (ours, Flip-S, TrojViT, random/simple-gradient baselines)
   are scored with the **same saved head**.

**Code needed:** `linear_probe.py` (train/save/load), a task branch in
`AttackPipeline.run()` (or an `lp_eval` step after Phase 4), and
`--task lp` wiring in `experiment.py`.

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
