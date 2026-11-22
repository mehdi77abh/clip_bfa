# CLIP baseline suite — TrojViT · ProFlip · Flip-S

Self-contained zero-shot CLIP comparison suite for the hybrid-BFA thesis.
**Imports nothing from `modular_new_pipeline` proper** — model/data/quantisation
plumbing is re-implemented locally (`clip_adapters.py`), so the folder can be
copied to another machine as-is (plus the `data/` tree and the `class_texts/`
copies shipped here).

## Files

```
bfa_baselines/
├── attack_trojvit.py   # TrojViT: Phase-1 attention-target trigger + Phase-2
│                       #   projection-row distillation (NGR tar/tar1 on
│                       #   visual.proj) + bit-flip realization of tuned entries
├── attack_proflip.py   # ProFlip: SNI (|proj @ t_target|) + native feature-
│                       #   anchored trigger (MSE(h,y[I_t])+CE, ≈31% bottom-right
│                       #   patch) + CBS S1/S2/S3, --fitness {mse_sni,cos}
├── attack_flips.py     # Flip-S: full rewrite — Stage-A trigger, SFS ranking,
│                       #   int8 weight bits + FP16 row-wise SCALE-factor bits,
│                       #   4KB mutual exclusion, 50-bit budget
├── evaluate.py         # SHARED ASR/CACC evaluator — the only place metrics
│                       #   are computed (defines the attack_state.pt contract)
├── run_sweep.py        # grid driver: methods x models x datasets x seeds
├── clip_adapters.py    # local plumbing: OpenAI CLIP loading, deterministic
│                       #   dataset loading, prompts, int8/fp16 quantisation
├── data_split.json     # FROZEN disjoint splits (eval 1000 / calib 384 /
│                       #   search 256 / aux 32) — the fairness guarantee
├── class_texts/        # frozen class-name lists (copies of the pipeline's)
└── results/            # everything lands here
    ├── {method}/{model_tag}/{dataset}/s{seed}[_{fitness}]/
    │     attack_state.pt  attack_meta.json  eval.json  run_*.log
    └── summary/{runs.csv, summary.csv}
```

## Quick start

```bash
cd proposed_method/modular_new_pipeline/bfa_baselines
python run_sweep.py                          # full grid, resume-aware
python run_sweep.py --dry-run                # show commands only
python run_sweep.py --methods flips --models ViT-B/16 \
    --datasets Caltech101 --seeds 42         # one cell
python evaluate.py --state results/.../attack_state.pt   # re-evaluate a run
```

Default grid: `{trojvit, proflip, flips}` × `{ViT-B/16, ViT-L/14,
ViT-L/14-336px}` × `{Caltech101, ImageNet1k, DTD}` × seeds `{42}`.
ProFlip runs twice per cell (`--fitness mse_sni` and `--fitness cos`).
Default target classes: Caltech101=97, ImageNet1k=291, DTD=3.

## Contracts

- **One code per method** (`attack_*.py`): writes `attack_state.pt` (final
  `param_overrides` + trigger dict + `flip_log`) and `attack_meta.json`
  (`n_flips, budget, search_time_s, trigger_asr_pre, pool_size,
  flips_to_94/95`). Exact schema: see `evaluate.py` docstring.
- **`evaluate.py` is the single source of metrics.** ASR = P(argmax ==
  target_cls | triggered) over non-target eval images; CACC = P(argmax ==
  label | clean). In-search quick evaluations call the same `eval_model`.
- **`data_split.json` is frozen**: every method and every server evaluates the
  identical images (eval 1000) and draws calibration/search/aux images from
  identical pools. Regeneration recipe is in the file's `_meta`.

## Method adaptations (labelled, for the paper)

| Method | Kept native | Adapted for CLIP |
|---|---|---|
| TrojViT | Phase-1 salience/attention patch selection + Attention-Target loss (PCGrad); Phase-2 NGR topk pattern, mixed clean/triggered distillation (200 ep, SGD 0.01, MultiStepLR 80/120/160), per-step restore of non-selected entries, convergence pruning | Classifier head → `visual.proj`: row score `|t_target[i]|·‖g[:,i]‖`, tuned entries realised as int8 bit flips (Hamming to the tuned value) so flips are counted |
| ProFlip | SNI significant-neuron criterion; feature-anchored trigger `MSE(h, y[I_t]=c) + CE` with the native ≈31%-width bottom-right patch and 10-step GD; CBS S1/S2/S3 fitness `|∂L/∂w|·(Q−w)`, K=20 grid, multi-bit element updates, ASR_t=94 / n_max=100 | JSMA SNI → analytic `I_t = top-|proj @ t_target|`; literal clamp `10` → scale-adaptive `c` (`--clamp-mode`); fitness loss selectable (`mse_sni` native / `cos`) — both rows reported |
| Flip-S | Whole method: SFS scale-factor ranking, bnb-style int8 + FP16 row-wise scales, 9-patch (3×victim-patch) trigger block, 4KB RowHammer mutual exclusion, 50-bit budget | logits via the zero-shot head; bitandbytes replaced by deterministic local quantisation with identical FP16-scale semantics |

## Fairness statement

Each baseline uses its native trigger-generation and bit-selection procedure.
The shared components are the victim models, the frozen data splits, the
zero-shot evaluation protocol (ASR/CACC in `evaluate.py`), and the reported
metrics. TBT/ProFlip classifier-layer analogues map to `visual.proj`; TrojViT's
classifier-row distillation maps to target-relevant `visual.proj` rows; Flip-S
operates on its native FP16 row-wise scale factors. Native budgets are kept
(TrojViT 80, ProFlip 100, Flip-S 50) and reported.
