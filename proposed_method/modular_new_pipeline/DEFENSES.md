# DEFENSES.md — Defence evaluation: FaR & Fine-Tuning — **FUTURE WORK**

Goal: measure how well post-hoc defences remove our backdoor from the
attacked model — RESULTS.md §4.

**Status:** **Fine-Tuning (FT) is implemented (2026-11-20)** —
`defence_finetune.py` + `run_defence_ft.sh` (below).  **FaR is implemented
(2026-11-27)** — `defences/far_clip/far_clip.py` (the CLIP
runner) + the `far_cfg` binding in `pipeline.load_far_state` (below).

---

## Defences

1. **Fine-Tuning (FT)** — **IMPLEMENTED** (`defence_finetune.py`)
   - Standard defence: fine-tune the attacked model on **clean** data so the
     backdoor is overwritten while task accuracy is kept.
   - **Clean data = CC3M** (`../data/cc3m`; WebDataset shards from
     `download_manager/download_datasets.py` (`cc3m-train-NNNN.tar`) or
     BadCLIP-style
     CSV + images — auto-detected, `--cc3m-format {auto,tar,csv}`).
   - Objective: symmetric CLIP InfoNCE on image–caption batches (learnable
     `logit_scale`, clamped at 100).
   - Defaults (all CLI-overridable): AdamW lr=1e-5, wd=0.1, 1 epoch capped at
     **2000 steps**, batch 64, **full model** (`--ft-scope vision` restricts
     to the visual encoder), seed 42.
2. **FaR (Forget-and-Rewire)** — pre-attack defended-victim protocol
   (2026-11-26): the defence is applied to the target model **FIRST**
   (`bfa_baselines/far_demo/` — Forget-and-Rewire: rewiring concealed→dead
   input weights plus forward-time input rewiring via `CustomLinear`), then
   our attack runs **on the FaR-applied model**:
   1. run the FaR defence on the clean victim → defended checkpoint;
   2. `experiment.py --far-ckpt PATH` loads it (full-model or
      vision-encoder state_dict; fail-fast when missing) and the whole
      Phase 0→4 attack targets the defended victim; **no argument = the
      undefended victim, current behaviour**.  Run dirs gain a `_far`
      suffix and `aggregate.py` tracks a `far` column (0/1), so defended
      and undefended runs never mix in `summary.csv`.
   - **FaR runner for CLIP (2026-11-27): `defences/far_clip/far_clip.py`**
     — loads the victim (`clip.load`, OpenAI CLIP), applies FaR to the
     **vision tower** (every block linear: fused q/k/v `attn.in_proj`,
     `attn.out_proj`, `mlp.c_fc`, `mlp.c_proj` — the paper's Linear-layer
     scope incl. Query/Key/Value; paper §2.1/§5.2) and saves the defended
     victim as `{"state_dict", "far_cfg", "meta"}` (state_dict keys
     unchanged vs vanilla CLIP).  Runner CLI:
     `python far_clip.py --model ViT-B/16 --data-root <ImageFolder> --out-dir defended_models/`
     (defaults: `--max-far 30 --div-fac 5` (paper Fig. 9 best)
     `--frac-size 4 --acc-drop-stop 2.0 --n-samples 32`; important/dead
     selection = gradient ranking of the zero-shot CE loss, recomputed
     after every FaR op).  `pipeline.load_far_state` binds the
     checkpoint's `far_cfg` (`far_clip.install_far`) before loading the
     weights, so the defended forward semantics are active during the
     attack.  The post-hoc variant (FaR applied to an already-attacked
     model) remains a different experiment from this pre-attack protocol.

## Protocol (as implemented)

**One-command driver (2026-11-27): `defences/run_defences.sh`** — runs BOTH
defences end-to-end on ViT-L/14 × ImageNet1k (zs + lp): the FT flow
(attack → eval → FT → eval) and the FaR flow (FaR → eval → attack → eval),
with RESUME semantics; see `defences/README.md`.

1. Run the attack (`--task zs` or `--task lp`, `--variant full`) per
   (model, dataset, seed) with `--save-attacked-model`.
2. **Save the attacked model state** — `model_state_attacked.pt` per run
   (`--save-attacked-model`; `run_all.sh` gates it to ViT-L/14 × ImageNet1k ×
   last seed × last rep, `run_lp.sh` enables it by default).
3. `defence_finetune.py --run-dir <run>` loads the attacked state
   (`bfa_results.json` config + `model_state_attacked.pt` + `trigger.pt`),
   evaluates **before**, fine-tunes on CC3M, evaluates **after** on the
   identical eval set, and writes `defence_ft.json` **inside the run dir**.
4. `run_defence_ft.sh` drives exactly the two agreed configurations —
   **zs ViT-L/14 × ImageNet1k** and **lp ViT-L/14 × ImageNet1k** — with
   run-dir auto-discovery (`RUN_DIR_ZS` / `RUN_DIR_LP` override), and prints
   the §4 table row.

## Result table (RESULTS.md §4)

| Defence | Dataset | ASR (no defence) | ASR (after) | CACC (no defence) | CACC (after) |
|---|---|---|---|---|---|
| FaR | … | … | … | … | … |
| Fine-Tuning | … | … | … | … | … |

`defence_ft.json` holds both rows' inputs (`before` / `after` re-evaluated on
the same test split, `attack_recorded` = the attack run's own numbers for
cross-checking).

## Requirements / code needed

- [x] `model_state_attacked.pt` saved per run (pipeline/results,
      `--save-attacked-model`).
- [x] Fine-Tuning defence script — `defence_finetune.py` (load attacked
      model → defend → existing `Evaluator` for ASR/CACC) + `run_defence_ft.sh`.
- [x] FaR — attack-side hook (`--far-ckpt` + `pipeline.load_far_state`;
      pre-attack defended-victim protocol above) **and** the FaR runner for
      CLIP victims (`defences/far_clip/far_clip.py`, 2026-11-27).  Remaining:
      the defended-model runs (grids).
- [x] Clean fine-tune data: CC3M (`--cc3m-root` / `--cc3m-max-samples`;
      defence-specific, not an attack-registry dataset).
- [x] Fixed seeds + identical eval set between no-defence and defended runs
      (both evaluated in-script on the attack run's `test_idx`, same seed).

## Notes

- Defence runs are **evaluations of saved models** — they should not re-run
  the attack; keeping the attacked checkpoints is what makes this cheap.
- The defence task is orthogonal to `--task`: `defence_finetune.py` follows
  the run's own recorded task (`zs` = zero-shot readout, text embeddings
  rebuilt after FT because the fine-tune may move the text encoder too;
  `lp` = the saved shared probe head, fixed by protocol).
