# LP_README.md — How to run the Linear-Probe (LP) task

`--task lp`: attack the CLIP visual encoder with bit flips, then evaluate the
attacked model through a **shared linear-probe head** trained once on clean
frozen features.  Metrics: **LP-CACC** (clean accuracy of the head) and
**LP-ASR** (triggered images classified as the target).  Protocol:
`TASKS.md` §2, `RESULTS.md` §0/§2.

---

## TL;DR — one command

```bash
cd proposed_method/modular_new_pipeline

# once: ImageNet-1k train/val in ../data/imagenet (registry layout)
#       → see download_manager/README.md

bash run_lp.sh
# or with an explicit python:
PY=~/miniconda3/envs/vlm_attack/bin/python bash run_lp.sh
```

Defaults: **ViT-L/14 × ImageNet1k**, probe trained on **50 000** ImageNet
train images (50/class), seed 42, target **'lion' (c291)**, trigger 16,
`--variant full`, `--section baselines`.

`run_lp.sh` runs three stages:

| Stage | What | Tool |
|---|---|---|
| 0 | data check (ImageNet train/val present) | — |
| 1 | **train the LP head** on CLEAN frozen features → `lp_heads/ViT-L14_ImageNet1k.pt` | `linear_probe.py` |
| 2 | **CLIP bit-flip attack** (Phases 0–4), then **evaluate LP** by attaching the stage-1 head to the attacked encoder's features | `experiment.py --task lp` |
| 3 | aggregate → `runs.csv` / `summary.csv` (LP columns) | `aggregate.py` |

The probe is trained on the **clean** model and stays **fixed** — it is never
re-trained after the attack.  LP evaluation runs *after* Phase 4 on the same
eval split the attack used.

---

## RESUME behaviour

`RESUME` (default **1**) skips stages whose output already exists:

| Stage | Skipped when | Effect |
|---|---|---|
| 1 — probe training | `lp_heads/{tag}_{dataset}.pt` exists | the existing head is reused |
| 2 — attack + LP eval | `{run_dir}/bfa_results.json` exists | the finished run is not redone |

```bash
bash run_lp.sh              # RESUME=1 (default): continue where it left off
RESUME=0 bash run_lp.sh     # retrain the probe (overwrites the checkpoint)
                            # AND re-run the attack
```

Notes:
- The skip is **whole-stage**.  If a run crashed mid-way, its output dir has
  no `bfa_results.json` and stage 2 re-runs — but stale partial files may be
  overwritten; delete the run dir first if you want a clean slate.
- To re-run only the attack with an existing probe: leave `RESUME=1` and
  delete just the run dir (or set `RESUME=0` after moving the head aside).
- To force a fresh probe but keep the attack result: retrain with
  `RESUME=0`, then restore the old head (the attack records its head path).

---

## Commands

### 1. Probe training only (`linear_probe.py`)

```bash
# default: ViT-L/14, ImageNet1kTrain (50/class = 50 000) → ImageNet1k test split
python linear_probe.py --model ViT-L/14

# explicit
python linear_probe.py --model ViT-L/14 \
    --train-dataset ImageNet1kTrain --test-dataset ImageNet1k \
    --train-per-class 50 --seed 42 --epochs 60 --lr 1e-3 --wd 1e-4

# other model / dataset (train and test entity must share the label space)
python linear_probe.py --model ViT-B/16 --train-dataset Caltech101 \
    --test-dataset Caltech101 --train-per-class 50

python linear_probe.py --help
```

Checkpoint: `{lp_head_root}/{model_tag}_{dataset}.pt`, e.g.
`lp_heads/ViT-L14_ImageNet1k.pt` (registry `paths.lp_head_root`).  Contains
`weight`/`bias` tensors + `meta` (class names, training/eval metadata —
incl. the held-out test top-1/top-5).

### 2. Attack + LP evaluation only (`experiment.py`)

```bash
python experiment.py --model ViT-L/14 --dataset ImageNet1k \
    --task lp --target-cls 291 --seed 42 \
    --variant full --trigger-size 16 --section baselines
# the head is derived automatically; override with --lp-head PATH
```

This runs the full attack (Phase 0 quantize → 1 trigger → 2 ablation →
3 ranking → 4 greedy bit-flip search), then attaches the saved head to the
attacked features and reports LP-CACC / LP-ASR.  Results:

```
grid_results/{section}/{model_tag}/{dataset}/lp_c{cls}_s{seed}_r{rep}_{variant}_t{sz}/
  ├── bfa_results.json        # "linear_probe" block: lp_cacc, lp_asr,
  │                           #   lp_clean_as_target, n, lp_head, probe_test_top1
  ├── trigger.pt  original_weights.pt
  └── model_state_attacked.pt # saved by run_lp.sh (defence evaluation)
```

### 3. Aggregation

```bash
python aggregate.py --out-root ./grid_results
```

LP columns (`lp_cacc`, `lp_asr`, `lp_clean_as_target`) appear in `runs.csv`
and as mean ± std in `summary.csv`.

---

## Environment knobs (`run_lp.sh`)

All are `VAR=value bash run_lp.sh` overrides:

| Var | Default | Meaning |
|---|---|---|
| `PY` | `python` | interpreter (use the `vlm_attack` conda env) |
| `MODEL` / `DATASET` | `ViT-L/14` / `ImageNet1k` | attack run identity |
| `TRAIN_DATASET` | `ImageNet1kTrain` | probe train split registry entity |
| `TRAIN_PER_CLASS` | `50` | probe images/class (50 × 1000 = 50 000) |
| `PROBE_SEED` / `PROBE_EPOCHS` / `PROBE_LR` / `PROBE_WD` | `42` / `60` / `1e-3` / `1e-4` | probe training |
| `SEED` / `REP` / `VARIANT` | `42` / `1` / `full` | attack identity |
| `SECTION` | `baselines` | results group (`validation`/`baselines`/`ablation`) |
| `TRIGGER_LOCATION` | *(empty = legacy)* | `top_left`…`bottom_right` |
| `RESUME` | `1` | skip finished stages (see above) |
| `SAVE_ATTACKED_MODEL` | `1` | write `model_state_attacked.pt` (defences) |
| `LP_HEAD_ROOT` / `LP_HEAD` | registry `lp_heads/` | probe location |
| `TARGET_CLS` | per dataset (ImageNet1k → **291** 'lion') | attack target |
| `TRIG_STD`/`TRIG_CIFAR`/`TRIG_HIRES` | `16`/`8`/`24` | trigger-size policy |
| `P1_EPOCHS` `N_CALIB` `N_EVAL` `MAX_FLIPS` `NORMAL_STOP` `FINE_*` … | validation-grid protocol | attack hyperparameters (same as `run_all.sh`) |
| `OUT_ROOT` | `./grid_results` | results root |

---

## Guards (what the code checks for you)

- `experiment.py --task lp` **fails fast** if the probe head is missing
  (before any GPU work) — train it first or pass `--lp-head`.
- `lp_evaluate` rejects a head whose `meta.class_names` label space does not
  match the run dataset (e.g. an ImageNet head on a Caltech101 run), and
  checks `target_cls` against the head's class count.
- `run_lp.sh` stops with a clear message when ImageNet train/val is absent.

---

## After the LP run

- Comparison rows: LP-CACC / LP-ASR per method (RESULTS.md §2 — all methods
  are scored through the **same** saved head).
- Defence: `bash run_defence_ft.sh` finds the LP run's
  `model_state_attacked.pt` automatically (`RUN_DIR_LP=` overrides) and runs
  the Fine-Tuning defence on it (`DEFENSES.md`).
- Flip-S can consume the same checkpoint: `--head probe --probe-path
  lp_heads/ViT-L14_ImageNet1k.pt` (legacy mode, `bfa_baselines/BASELINES.md`).

## Related docs

`TASKS.md` §2 (protocol) · `RESULTS.md` §0/§2 (metrics, comparison) ·
`CHECKLIST.md` (status + next steps) · `DEFENSES.md` (FT defence on LP runs) ·
`AGENTS.md` (code map).
