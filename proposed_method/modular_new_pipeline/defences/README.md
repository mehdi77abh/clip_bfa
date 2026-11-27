# defences/ — defence evaluation: FT and FaR on CLIP victims

Implements the two defence methods of `../DEFENSES.md` for the thesis
comparison, both on the same victim (**ViT-L/14 × ImageNet1k**, evaluated on
**zero-shot `zs`** and **linear-probe `lp`**), feeding the RESULTS.md §4
table.

```
defences/
  run_defences.sh     # ONE script — runs both defences end-to-end (below)
  far_clip/           # FaR (Forget-and-Rewire) runner for OpenAI-CLIP
  defended_models/    # FaR checkpoints (created by the run)
  README.md
```

## Two defences, two flows

**Defence 1 — Fine-Tuning (FT)** — `bfa attack → Eval → FT → Eval`
1. `experiment.py --save-attacked-model` — attack the **undefended** victim.
2. `defence_finetune.py` — evaluate **before** (ASR/CACC "no defence"),
   fine-tune on CC3M.
3. Evaluate **after** → `defence_ft.json` in the run dir.

**Defence 2 — Forget-and-Rewire (FaR)** — `FaR → eval → bfa attack → eval`
1. `far_clip/far_clip.py` — apply FaR to the clean victim → defended
   checkpoint (its reported calib accuracy = the eval right after FaR).
2. `experiment.py --far-ckpt <defended>` — attack the **defended** victim.
3. `bfa_results.json` metrics = the eval after the attack.

The **undefended** attack runs double as the "no defence" baseline for the
FaR row — no duplicate undefended attacks.  The same LP head (trained once
on clean features, `../lp_heads/ViT-L14_ImageNet1k.pt`, TASKS.md §2) is
reused by every run.

## Run

```bash
bash run_defences.sh              # both defences, zs + lp, ViT-L/14 × ImageNet1k
TASKS="zs" bash run_defences.sh   # zero-shot only
DEFENCES="far" bash run_defences.sh   # FaR only
PY=~/miniconda3/envs/vlm_attack/bin/python bash run_defences.sh
```

Prerequisites (the script fails fast if missing):
* LP head `../lp_heads/ViT-L14_ImageNet1k.pt` (for `lp`).
* CC3M in `../data/cc3m` (for the FT fine-tune).
* ImageNet-1k `../../data/imagenet/{train,val}` (train = FaR calib, val = eval).

Every step is **RESUMABLE** (skipped when its output exists), so the script
is cheap to re-run and continues an interrupted grid.  All knobs are env
overridable — see the header of `run_defences.sh`.

## Outputs

```
grid_results/others/ViT-L14/ImageNet1k/
  {task}_c291_s42_r1_full_t16/        undefended attack (+ defence_ft.json)
  {task}_c291_s42_r1_full_t16_far/    FaR-defended attack
defences/defended_models/far_ViT-L14.pt      the FaR checkpoint
```

`python ../aggregate.py` then folds every run into `runs.csv`/`summary.csv`
(the `far` column distinguishes defended from undefended runs).
