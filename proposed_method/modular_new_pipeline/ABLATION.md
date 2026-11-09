# ABLATION.md — Ablation study via `--variant`

One variable controls the whole ablation study:

| `--variant` | Meaning | Behaviour change |
|---|---|---|
| **`full`** (= `d` = `clip_flip`) | **Proposed method, unchanged** (default) | none |
| **`a`** | exclude **Phase 2** (layer + component pruning) | skip Phases 2A/2B |
| **`b`** | exclude the **clean penalty** | drift term out of the Phase-4 ranking metric |
| **`c`** | skip the **poison−clean gradient** | Phase 3 ranks on the poison gradient only |

`d` and `clip_flip` are accepted aliases of `full` (all normalize to `full`).
Run example (ablation of the clean penalty on ViT-B/16, Caltech101, seed 42):

```bash
python experiment.py --model ViT-B/16 --dataset Caltech101 --seed 42 --variant b
```

Output dirs differ only in the last segment — e.g.
`.../c3/s42/full__t14` vs `.../c3/s42/b__t14` — so all variants of one
setting sit side by side.

---

## `full` — proposed method (baseline for every ablation)

CLIP-FLIP as designed: Phase 0 quantisation → Phase 1 clamped trigger →
Phase 2A layer ablation + 2B component pruning → hybrid ranking
(`R_l × |g·Δw|` on the poison−clean specific gradient) → greedy search with
the clean-drift penalty.  No code changes.

## `a` — no Phase 2 (layer + component pruning)

- **What changes:** Phases 2A/2B are skipped.  Candidate components are
  **all** attention heads + MLPs of **all** layers
  (`enumerate_layer_components`), and the layer weight `R_l` is uniform 1.0
  (no `layer_scores`).
- **Code hook:** `pipeline.run()` — `cfg.SKIP_PHASE2` gates the 2A/2B calls.
- **What it shows:** the **efficiency of Phase 2** — wall-time saved by
  skipping 2A/2B, the candidate-pool size before/after pruning, and the
  ASR/CACC/Flips→95% delta vs. `full`.  (RESULTS.md §3b)

## `b` — no clean penalty

- **What changes:** the Phase-4 ranking metric becomes `Δcos` only —
  no drift penalty λ·drift and no max-drift rejection
  (`CLEAN_DRIFT_PENALTY = 0`, `MAX_CLEAN_DRIFT = ∞`).
- **Code hook:** `experiment.build_config` overrides the two cfg values when
  `cfg.SKIP_CLEAN_PENALTY` — **no phase code changes**.
- **What it shows:** the contribution of the clean-drift term to stealth
  (CACC) and search quality.  (RESULTS.md §3d)

## `c` — poison-only gradient (skip poison−clean)

- **What changes:** Phase 3 ranking uses the **poison gradient only**; the
  clean backward pass and the poison−clean difference are skipped
  (also in the pool-rebuild path).
- **Code hook:** `hybrid_ranking.HybridRanker.rank` / `rank_fallback` —
  `cfg.POISON_ONLY_GRAD` selects `grad_specific = grad_poisoned`.
- **What it shows:** the value of the poison−clean *specific* gradient over
  the naive poison gradient.  (RESULTS.md §3c)

---

## Wiring summary

| variant | cfg flags (set in `experiment.build_config`) | code sites |
|---|---|---|
| `full`/`d`/`clip_flip` | all False | — |
| `a` | `SKIP_PHASE2=True` | `pipeline.run()` (2A/2B gate) |
| `b` | `SKIP_CLEAN_PENALTY=True` (+ `CLEAN_DRIFT_PENALTY=0`, `MAX_CLEAN_DRIFT=∞`) | none (cfg-only) |
| `c` | `POISON_ONLY_GRAD=True` | `hybrid_ranking.rank`, `rank_fallback` |

With flags at defaults the pipeline is bit-identical to the reference —
divergence is opt-in per variant only.

## Result tables

Use the ablation table of RESULTS.md §3: rows = `full`, `a`, `b`, `c`;
columns = ASR@10, ASR@20, Flips→95%, CACC, Time, pool size — same matrix and
seeds as the validation section so rows are comparable.

**Note (trigger-size sweep, RESULTS.md §3a):** not a variant — it uses
`--trigger-size N` with `--variant full`, and the size already appears in the
output dir (`{variant}__t{N}`).
