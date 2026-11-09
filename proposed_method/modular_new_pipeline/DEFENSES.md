# DEFENSES.md — Defence evaluation: FaR & Fine-Tuning — **FUTURE WORK**

Goal: measure how well post-hoc defences remove our backdoor from the
attacked model — RESULTS.md §4.

**Status:** nothing implemented yet.  This file is the plan; details marked
*(TBD)* are to be filled by the maintainer (papers / hyperparameters).

---

## Defences

1. **Fine-Tuning (FT)**
   - Standard defence: fine-tune the attacked model on **clean** data so the
     backdoor is overwritten while task accuracy is kept.
   - *(TBD)* scope: full model vs. last blocks only; epochs; LR; optimizer;
     fine-tune split (reuse `splits.n_trigger`? a dedicated split?).
2. **FaR**
   - *(TBD — maintainer supplies the paper/definition and its
     hyperparameters; expected to be a feature/activation-level defence.)*

## Protocol (planned)

1. Run the attack (`--task zs --variant full`) per (model, dataset, seed).
2. **Save the attacked model state** — `model_state_attacked.pt` per run
   (future code item: one extra `torch.save` in `results.ResultWriter`;
   listed in RESULTS.md "Code changes required" #6).
3. Apply the defence to the saved attacked model → defended model.
4. Evaluate both with the same protocol:
   - **ASR after defence** (lower = defence wins),
   - **CACC after defence** (task-accuracy recovery),
   - vs. the no-defence values of the same run.

## Result table (RESULTS.md §4)

| Defence | Dataset | ASR (no defence) | ASR (after) | CACC (no defence) | CACC (after) |
|---|---|---|---|---|---|
| FaR | … | … | … | … | … |
| Fine-Tuning | … | … | … | … | … |

## Requirements / code needed

- [ ] `model_state_attacked.pt` saved per run (pipeline/results).
- [ ] Defence scripts (one per defence, e.g. `defence_far.py`,
      `defence_finetune.py`) — each: load attacked model → defend → call the
      existing `Evaluator` for ASR/CACC.
- [ ] A clean fine-tune split definition (registry `splits` extension or a
      defence-specific arg).
- [ ] Fixed seeds + identical eval set between no-defence and defended runs.

## Notes

- Defence runs are **evaluations of saved models** — they should not re-run
  the attack; keeping the attacked checkpoints is what makes this cheap.
- The defence task is orthogonal to `--task`: defences are evaluated on the
  zs protocol first (ASR/CACC); extending to lp/i2t later follows TASKS.md.
