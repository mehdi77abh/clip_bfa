# proflip_pipeline — ProFlip-CBS baseline on CLIP

Standalone 2-stage baseline project. It does **not** modify
`modular_new_pipeline/`; it imports its modules and `bfa_baselines/proflip_cbs.py`.

## Phases

| Step | What happens |
|---|---|
| **Phase 0** | Quantise the CLIP visual encoder to the proposed method's per-tensor int8 grid (setup, not an attack phase) |
| **Phase 1** | Generate the trigger with the **proposed method's** Phase-1 generator (same `cos` loss, grad clip, dead-trigger retries, splits, size/location) |
| **Phase 2** | **Skipped** — CBS gets the full component pool (all heads + MLPs of all layers) |
| **Phase 3** | **Skipped** — CBS computes its own gradients |
| **CBS** | ProFlip's S1/S2/S3 critical bit search (`proflip_cbs.py`), target-cosine objective, **ASR_t=94, n_max=100** (paper values) |

This is the **CBS component baseline**, not a full ProFlip port: SNI and
ProFlip's own trigger are not used; the trigger is shared with the proposed
method. See `../CLIP_ADAPTATION_ANALYSIS.md` §3.

## Run

```bash
cd bfa_baselines
python proflip_pipeline/run.py \
  --model ViT-B/16 --dataset Caltech101 \
  --seed 42 --target-cls 97 --trigger-size 16 --n-eval 1000
```

All `experiment.py` arguments are accepted (model, dataset, splits, Phase-1
hyperparameters, trigger geometry). Additional CBS flags:

```
--cbs-k 20                 # S3 grid points (paper: 20)
--cbs-asr-target 94        # paper stopping ASR
--cbs-nmax 100             # paper max bit flips
--cbs-clean-mse-weight 0   # 0 = target-cosine only
--cbs-q-mode abs           # paper max|w|; 'signed' = released code max(w)
--cbs-no-monotone          # allow decreasing an element (release forbids)
```

Output (default):

```
bfa_baselines/baseline_runs/baselines/{model}/{dataset}/
  zs_c{cls}_s{seed}_r{rep}_proflip_cbs_t{size}/
    bfa_results.json        # metrics (flips_to_94/95, ASR@10/20, CACC, ...)
    trigger.pt, original_weights.pt, trigger_patch.png, composite.png
```

The global grid is `run_all_baselines.sh`; aggregation is
`aggregate_baselines.py` → `baseline_summary/runs.csv` + `summary.csv`.

## Notes

- Stopping at **94** means `flips_to_95` may be `None`; `flips_to_94` is the
  reached-target count. Both are recorded.
- Multi-bit element updates are atomic; `flips_total` counts **bits** (Hamming
  distance), `asr_at_10/@20` use the first recorded state with ≥ k bits.
- The trigger is the proposed method's, so the row answers *"is our bit
  selection better than ProFlip's CBS at the same trigger?"* — not "do we beat
  the full ProFlip pipeline".
