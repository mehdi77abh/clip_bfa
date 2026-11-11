# RESULTS.md — Evaluation plan & result structure (CLIP-FLIP)

The thesis results have **three parts plus a defence section**:
**1. Validation → 2. Comparison → 3. Ablation → 4. Defence.**
Runs are driven by **CLI arguments** (no grid config file). All runs use the
`modular_new_pipeline/` attack with the pipeline modules as-is; variants are
selected by flags, never by editing code.

---

## 0. Metrics (definitions)

| Metric | Definition |
|---|---|
| **ASR@k** | Attack success rate after **k accepted bit flips** (triggered inputs classified as the target class), evaluated on the eval set right after the k-th accepted flip. Required at **k = 10** and **k = 20**. |
| **Flips→95%** | Number of accepted bit flips until ASR first reaches **≥ 95%**. If never reached within `MAX_FLIPS`, report the cap (e.g. ">80"). |
| **CACC** | Clean accuracy on non-target (eval) images after the attack; final value, and at 10/20 flips where the table needs it. |
| **Time (ZS)** | Wall-clock attack time in the **zero-shot** setting (search time; total pipeline time logged alongside). |
| **LP-CACC / LP-ASR** | Linear-probe accuracy on clean / triggered features, probe trained per (model, dataset) — see §2. |

---

## 1. Validation

**Goal:** show the method works across architectures and datasets (zero-shot task).

- **Matrix:** 2 models × 3 datasets × 3 seeds = **18 runs**.
  - Models: *(placeholder — e.g. ViT-B/16, ViT-L/14)*
  - Datasets: *(placeholder — e.g. Caltech-101, CIFAR-10, CIFAR-100)*
  - Seeds: 3 per (model, dataset).
- **Reported per (model, dataset):** mean ± std over the 3 seeds of
  **ASR@10 · ASR@20 · Flips→95% · CACC · Time (ZS)**. Per-seed rows live in
  `runs.csv` / the appendix.

**Table template (main text):**

| Model | Dataset | ASR@10 | ASR@20 | Flips→95% | CACC | Time (s) |
|---|---|---|---|---|---|---|
| … | … | … ± … | … ± … | … ± … | … ± … | … ± … |

---

## 2. Comparison

**Goal:** position the attack against published attacks and simple baselines,
in the **Linear-Probe (LP) task**.

- **External attacks:** **Flip-S** and **TrojViT** (baseline repos under
  `../baselines/`; evaluated under the same LP protocol — their training code
  is out of scope here, only the evaluation protocol must match).
- **Internal baselines (implemented in this codebase):**
  - **Random bit-flip** — flips random candidate bits inside the selected
    components (same flip budget / stopping rules as our method).
  - **Simple gradient ranking** — ranks by plain **|g·Δw|** only: no layer
    score `R_l`, no hybrid weighting (≈ `HybridRanker` with `R_l = 1`).
- **LP protocol (fair comparison):**
  1. `linear_probe.py` trains the LP layer **once per (model, dataset)** on
     clean features and saves it (`lp_probes/{model}_{dataset}.pt`).
  2. Every method (ours, Flip-S, TrojViT, both internal baselines) is
     evaluated with **that same probe** — LP-CACC on clean features, LP-ASR on
     triggered features.

**Table template:**

| Method | LP-CACC | LP-ASR |
|---|---|---|
| Flip-S | … | … |
| TrojViT | … | … |
| Random bit-flip | … | … |
| Simple gradient ranking | … | … |
| **Ours** | … | … |

---

## 3. Ablation

Each variant = `--variant` (defined in ABLATION.md) on the same attack code;
run with the same matrix/subset × 3 seeds where feasible.

| # | Variant | Flag | What it shows |
|---|---|---|---|
| 3a | **Trigger-size sweep** | `--trigger-size N` (`--variant full`) | Sensitivity of ASR/CACC/Flips→95% to the trigger patch size (report one row per size). |
| 3b | **No Phase 2** (no layer + component pruning) | `--variant a` | **Efficiency of Phase 2:** wall-time saved by skipping 2A/2B, candidate-pool size before/after, and ASR/CACC/Flips→95% delta vs the full method. Falls back to `enumerate_layer_components` (all heads + MLPs of all layers). |
| 3c | **Poison-only gradient** | `--variant c` | Contribution of the poison−clean gradient difference: ranking uses the poison gradient only. |
| 3d | **No clean penalty in ranking metric** | `--variant b` | Contribution of the clean-drift term: Phase-4 scoring without the drift penalty and max-drift rejection. |

**Table template (per variant vs full method):**

| Variant | ASR@10 | ASR@20 | Flips→95% | CACC | Time (s) | Pool size |
|---|---|---|---|---|---|---|
| Full method | … | … | … | … | … | … |
| 3b no Phase 2 | … | … | … | … | … | … |
| 3c poison-only | … | … | … | … | … | … |
| 3d no drift | … | … | … | … | … | … |

---

## 4. Defence

**Goal:** robustness of the attack (and of the triggered model) against
post-hoc defences.

- **Defences:** **FaR** and **Fine-Tuning**, applied to our attacked models.
- **Metrics:** **ASR after defence** (lower is better for the defence) and
  **CACC after defence** (recovery / stealth), per defence × dataset.
- **Requirement:** attacked model states are saved per run
  (`model_state_attacked.pt`) so defence evaluation runs offline without
  re-running the attack.

**Table template:**

| Defence | Dataset | ASR (no defence) | ASR (after) | CACC (no defence) | CACC (after) |
|---|---|---|---|---|---|
| FaR | … | … | … | … | … |
| Fine-Tuning | … | … | … | … | … |

---

## Result file layout

```
Results/
├── runs.csv                          # global (every run, every variant) — aggregate.py
├── summary.csv                       # mean ± std per config — aggregate.py
└── {section}/{model}/{dataset}/{task}/c{cls}/s{seed}/r{rep}/
    └── {variant}__t{sz}[__{loc}]/
        ├── bfa_results.json          # full record: flip_log, checkpoints,
        │                                 #   metrics (ASR@10/@20, Flips→95%),
        │                                 #   layer_R, selected components, timings
        ├── trigger.pt
        ├── original_weights.pt
        └── model_state_attacked.pt       # --save-attacked-model (for defence /
                                          #   VLM-transfer evaluation)
lp_probes/
└── {model}_{dataset}.pt              # (new) trained linear-probe layers
```

`{section}` is `validation` | `baselines` | `ablation` (`--section`) — the
three top-level results groups.

`bfa_results.json` must contain `metrics` with `asr_at_10`, `asr_at_20`,
`cacc_at_10`, `cacc_at_20`, `flips_to_95` and `checkpoints` entries at
**10 and 20 accepted flips** plus the **first flip count where ASR ≥ 95%**
so the validation table can be filled straight from the JSON (implemented in
`search.py`, 2026-11-10).  `r{rep}` separates same-seed repeats
(`--rep`); `__{loc}` is present when `--trigger-location` is set (location
sweeps).

---

## Code changes required (work spec for the next steps)

1. **CLI arguments (no grid config file):** DONE — `argparse` in `experiment.py`
   (model, dataset, task, seed, rep, target-cls, trigger type / size /
   **location**, max-flips, asr-target, out-root, out-dir, split overrides,
   phase hyperparameters, variant, lp-head).  `config.py` is registry-only;
   baseline mode (`ours|random|simple_grad`) still pending (#4).
2. **Phase-4 metrics:** DONE (`search.py`, 2026-11-10) — exact checkpoints at
   10 and 20 accepted flips, `flips_to_95` tracking, `ASR_TARGET` configurable
   (default 95; **0 disables the early stop** — use 0 for table-production runs
   so ASR@10/@20 always exist).  Recorded in `bfa_results.json["metrics"]`.
3. **`linear_probe.py` (new, standalone):** DONE — trains + saves the head;
   evaluation via `Evaluator.lp_evaluate` + `--task lp` (TASKS.md §2).
4. **Baseline rankers:** PENDING — a baseline mode (ours | random |
   simple_grad) — alternative `rank()` implementations behind the injected
   `ranker` interface (e.g. `baselines.py`: `RandomRanker`,
   `SimpleGradientRanker`).
5. **Ablation variants:** DONE — `--variant full|a|b|c` (ABLATION.md);
   trigger-size sweep via `--trigger-size`; trigger-**location** sweep via
   `--trigger-location` (train + eval at the same placement).
6. **Save attacked model state:** DONE (flagged) — `--save-attacked-model`
   writes `model_state_attacked.pt` (attacked state_dict) after Phase 4;
   `run_all.sh` enables it only for ViT-L/14 × ImageNet1k × last seed × last
   rep (saved for the later VLM-transfer / defence evaluation).  FaR and
   fine-tuning defence evaluation scripts: PENDING.
7. **Aggregation:** DONE — `aggregate.py` writes `runs.csv` + `summary.csv`
   (mean ± std per config); `run_all.sh` runs the validation + ablation grids
   and aggregates at the end.
