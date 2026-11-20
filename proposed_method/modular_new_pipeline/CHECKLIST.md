# CHECKLIST.md — Master checklist: what is done, what is left (CLIP-FLIP)

Last verified against the tree: **2026-11-20**.
Legend: `[x]` done · `[~]` code done, runs/results pending · `[ ]` not started ·
`(deferred)` maintainer decision to postpone · `(blocked)` needs data/GPU first.

Where the truth lives: `RESULTS.md` (result plan), `TASKS.md` (tasks),
`ABLATION.md`, `DEFENSES.md`, `bfa_baselines/BASELINES.md` (comparison),
`AGENTS.md` (code state).

---

## 1. Core pipeline & infrastructure

- [x] Modular port of `../main_new_pipeline.py` (Step 1) + reference sync
      (2026-11-08 retune: cos trigger loss, clamping, P1 defaults 14/200/0.02)
- [x] Argument layer: `config.json` registry + `config.py` + `experiment.py`
      (`--model/--dataset/--task/--seed/--rep/--variant/…`), flattened run dirs
- [x] Adapter layer (`adapters.py`): OpenAI CLIP + SigLIP-B/16 + SigLIP-so400m/14
- [x] Two-phase search (Normal → fine_patience) + Phase-4 metrics
      (`asr_at_10`, `asr_at_20`, `flips_to_95`, checkpoints, best-prefix rollback)
- [x] Ablation variants `--variant {full,a,b,c}` + trigger-size / trigger-location sweeps
- [x] Phase-1 dead-trigger retries (`--trigger-min-asr` / `--trigger-retries`)
- [x] Attacked-model state saving (`--save-attacked-model` → `model_state_attacked.pt`)
- [x] Aggregation (`aggregate.py`): `runs.csv` / `runs_{model}.csv` / `summary.csv`,
      incl. LP columns (`lp_cacc`, `lp_asr`, `lp_clean_as_target`)
- [x] Grid drivers: `run_all.sh` (validation + ablation), `run_lp.sh` (LP task),
      `run_defence_ft.sh` (FT defence) — logs + `RESUME=1` skip logic
- [x] OOM & error contract (`oom_pending.json` / `failed_runs.json`, exit-0 continuation)
- [x] Data tooling: `download_datasets.py` (Caltech-101, ImageNet-1k, Food101,
      **CC3M wds shards**), `prepare_imagenet.py`, `make_class_texts.py`,
      `class_texts/{caltech101,cifar10,dtd,food101,flowers102,imagenet1k}`
- [x] BLIP applicability scope note (`BLIP.md`, 2026-11-20): formulation check
      (A1–A3 satisfied by BLIP-1 ITC), readout decision (ITC-cosine argmax +
      `--task lp` rows), BLIP-2 out of scope, adapter implementation notes
- [ ] `--task i2t` retrieval branch (TASKS.md §3 — guard raises `NotImplementedError`;
      evaluator + recall@K + condition naming from `plot_retrieval_probe` to write)
- [ ] FaR defence script `defence_far.py` (DEFENSES.md — hyperparams **TBD by maintainer**)
- [ ] VLM-transfer evaluation (§3.5 — attacked CLIP inside a VLM; VQA +
      image captioning on 2 VLM models) — **not started**
- [ ] BLIP-1 as attack victim — `BlipAdapter` per `BLIP.md` §7 (the
      encoder-agnostic claim: attack on BLIP's ITC model, ITC-cosine argmax
      readout per `BLIP.md` §3 option 1, `--task lp` rows alongside;
      BLIP-2/Q-Former **out of scope**, `BLIP.md` §4) — **not started**
- [ ] Baseline rankers `baselines.py` (`RandomRanker`, `SimpleGradientRanker`) +
      `--baseline {ours,random,simple_grad}` (RESULTS.md #4) — **(deferred)**
- [ ] CSV/JSON writer so external baseline runs (Flip-S) land in `runs.csv` (BASELINES.md §8)

## 2. Evaluation tasks (TASKS.md)

- [x] `zs` — zero-shot classification (the default; full Phase 0→4 + ZS eval)
- [x] `lp` — linear-probe: `linear_probe.py` (shared head, 50/class = 50 000
      ImageNet train) + `Evaluator.lp_evaluate` + `--task lp` + `--lp-head`
- [x] LP completion (2026-11-20): `run_lp.sh` driver, fail-fast on missing head,
      label-space guard (`meta.class_names` vs run dataset), LP aggregation columns
- [~] LP **runs** — nothing executed yet: no probe trained (`lp_heads/` empty),
      no `--task lp` run on disk  →  see §6
- [ ] `i2t` — image-to-text retrieval (FUTURE, TASKS.md §3)

## 3. Results plan (RESULTS.md)

### 3.1 Validation (§1) — 3 models × 2 datasets × 5 seeds = 30 runs

- [~] **Partial on disk** (`grid_results/validation/`, zs):
  - [x] ViT-B/16 × Caltech101 — seeds 21/42/53
  - [x] ViT-B/16 × ImageNet1k — seeds 21/42/53
  - [x] ViT-L/14 × Caltech101 — seeds 21/42/53
  - [x] ViT-L/14 × ImageNet1k — seeds 21/42/53
  - [x] DTD smoke runs (ViT-B/16, seed 42; note: future DTD runs use trigger **8**)
  - [ ] seeds **64 / 77** for the four model×dataset combos
  - [ ] **SigLIP-B/16** — all combos/seeds
- [ ] Fill the §1 table (ASR@10 · ASR@20 · Flips→95% · CACC · Time) from
      `runs.csv` / `summary.csv`

### 3.2 Comparison (§2) — **ZS protocol** (LP was abandoned as the comparison
protocol — BASELINES.md §2; kept only as `--task lp`)

- [x] Flip-S → CLIP adapter (`bfa_baselines/Flip-S/target-clip/flip_s_clip_v2.py`,
      `--head zeroshot`, 13-template ZS head, `lp_eval_full`)
- [ ] **Re-run / recover the Flip-S ZS numbers** — no artifacts are kept in the
      repo (BASELINES.md §4.10); ImageNet val must exist first
- [ ] Decide: per-flip evaluation for Flip-S (ASR@10/@20/Flips→95% are
      proposed-method-only today)
- [ ] TrojViT: port to CLIP **or** explicitly document "not compared"
- [ ] Internal baselines random / simple-gradient — **(deferred)**; run them on
      the same matrix/seed as `--variant full` when revived
- [ ] Fill the §2 table (ZS-CACC / ZS-ASR per method)

### 3.3 Ablation (§3) — ViT-L/14 × ImageNet1k

- [x] `--variant full` reference rows (8 + sweep rows)
- [x] `--variant b` (no clean penalty) — 3 seeds
- [x] `--variant c` (poison-only gradient) — 3 seeds
- [ ] `--variant a` (no Phase 2) — **0 rows on disk**, still to run
- [x] Trigger-location sweep (top_left / top_right / bottom_left / bottom_right)
- [~] Trigger-size sweep (rows exist among the `full` runs — verify all of
      `SIZES=8 12 16 20 24 32` are covered)
- [ ] Fill the §3 table (variants vs full: ASR@10/20, Flips→95%, CACC, Time, pool size)

### 3.4 Defence (§4)

- [x] Attacked-model state saving per run (`--save-attacked-model`)
- [x] **Fine-Tuning defence implemented** (2026-11-20):
      `defence_finetune.py` (load attacked state → eval before → FT on CC3M
      (shards or CSV+images) → eval after on the identical eval set →
      `defence_ft.json` in the run dir) + `run_defence_ft.sh` (the two agreed
      configs: **zs + lp, ViT-L/14 × ImageNet1k**)
- [~] FT defence **runs** — nothing executed yet, **(blocked)** on:
      (a) `model_state_attacked.pt` — **none saved on disk yet** (re-run the two
      attacks with `--save-attacked-model`; `run_lp.sh` now does it by default),
      (b) CC3M download (`python download_datasets.py --datasets cc3m`), see §6
- [ ] FaR defence — hyperparameters/definition **TBD by maintainer**
- [ ] Fill the §4 table (ASR/CACC, no defence vs after, per defence × dataset)

### 3.5 VLM transfer evaluation (new — maintainer request 2026-11-20)

Goal: show the backdoor **survives transfer into a VLM** — take the attacked
CLIP visual encoder and use it inside a VLM *instead of its default CLIP
vision tower*, then evaluate **VQA** and **image captioning** on **2 VLM
models**.  (`model_state_attacked.pt` was already saved "for defence /
VLM-transfer evaluation" — RESULTS.md #6 — this is the evaluation half.)

- [ ] Pick the **2 VLM models** *(TBD by maintainer)* — candidates whose
      vision tower is a plain OpenAI CLIP we can swap: **LLaVA-1.5**
      (CLIP ViT-L/14-336px — matches registry `ViT-L/14-336px`) and
      **OpenFlamingo** (CLIP ViT-L/14).  Per `BLIP.md` §4, **BLIP-2 /
      InstructBLIP are out of scope** (Q-Former bottleneck; no drop-in
      tower swap).  BLIP-1 is a different experiment — attacking BLIP's own
      encoder (see the §1 BLIP-victim item) — though its ITC/captioning
      heads could also host an attacked-CLIP transfer row if wanted.
- [ ] Pick the **benchmarks + attack-success definition** *(TBD)* —
      VQA (VQAv2 / GQA / VizWiz?) and captioning (COCO Captions / NoCaps?);
      metrics VQA accuracy + CIDEr/SPICE; success = triggered inputs push
      answers/captions toward the target semantics (vs clean inputs).
      Note (`BLIP.md` §4): **generative readouts break the cosine-argmax
      ASR definition** — the transfer metric must be defined on the text
      side (e.g. target-mention rate in generated answers/captions), not as
      argmax ASR.
- [ ] Attack runs with `--save-attacked-model` for the tower-relevant model
      (e.g. `ViT-L/14-336px` for LLaVA) — per (model, VLM pair), same seed.
- [ ] Code: VLM loader + **vision-tower state-dict swap**
      (`model_state_attacked.pt` → the VLM's CLIP tower), trigger application
      at the VLM's image preprocess, VQA + captioning eval harness
      (e.g. `vlm_transfer.py` + `run_vlm_transfer.sh`), results saved in the
      run dir (same convention as `defence_ft.json`).
- [ ] Protocol table: default-CLIP VLM vs attacked-CLIP VLM × clean vs
      triggered inputs, per VLM model × task.
- [ ] Fill the §3.5 table (VQA acc / captioning metrics, clean vs triggered).

## 4. Data

- [x] Caltech-101, CIFAR10/100, DTD on disk (`../data`); ImageNet downloader +
      `prepare_imagenet.py` (val + train subset) ready
- [ ] **ImageNet-1k train/val prepared** — `data/imagenet/` currently holds only
      the stray folder; run `python ../../data/prepare_imagenet.py` (needed for
      the LP probe's 50 000 train images and both `run_lp.sh`/`run_all.sh`)
- [ ] **CC3M downloaded** (`download_datasets.py --datasets cc3m`, wds shards
      `data/cc3m/cc3m-train-NNNN.tar`) — needed for the FT defence
- [ ] Food101 / Flowers102 — registered (2026-11-20) but **data not downloaded**
- [ ] Stray-empty-folder cleanup (`data/imagenet/caltech101`,
      `data/*/101_ObjectCategories/caltech101`) — **maintainer call** (`rmdir`)
- [x] DTD trigger-size decision recorded (use **8**, not 16)

## 5. Verification (AGENTS.md checklist + extras)

- [x] `py_compile` + symbol greps on every module; `build_config` tests
- [x] OOM contract tested (mocked OOM → clean exit + `oom_pending.json` with argv)
- [x] LP guards tested (fail-fast missing head, label-space mismatch, target range)
- [x] CC3M loader tested (shard mode + CSV mode + fallbacks + URL-list rejection)
- [x] `run_lp.sh` / `run_defence_ft.sh` dry-tested (identity, discovery, error paths)
- [ ] `experiment.py` end-to-end on the GPU box; result-key diff vs the reference
      `bfa_results.json`
- [ ] Real-OOM path exercised on an actual OOM
- [ ] FT defence end-to-end run on the GPU box (after §6 prerequisites)

## 6. Immediate next steps (the LP → defence chain)

In order (GPU box, `vlm_attack` env):

1. [ ] `python ../../data/prepare_imagenet.py` — ImageNet-1k val + train subset
2. [ ] `bash run_lp.sh` — trains the **ViT-L/14 × ImageNet1k** probe (50 000
      images) → one `--task lp` attack → aggregate; saves
      `model_state_attacked.pt` (now default-on)
3. [ ] Ensure the **zs ViT-L/14 × ImageNet1k** run has `model_state_attacked.pt`
      (existing `validation/…/zs_c291_s77_r1_full_t16` — else re-run with
      `--save-attacked-model`)
4. [ ] `python download_datasets.py --datasets cc3m` (or place CC3M under
      `../data/cc3m`)
5. [ ] `bash run_defence_ft.sh` — FT defence on both configs; read
      `defence_ft.json` + the §4 summary table
6. [ ] While at it: add the missing validation seeds (64/77) and SigLIP-B/16
      rows (§3.1), and `--variant a` ablation rows (§3.3)

## 7. Later / deferred

- [ ] Baseline rankers + baseline runs (**deferred** by maintainer)
- [ ] Flip-S comparison numbers re-run; per-flip-eval decision; TrojViT decision
- [ ] `--task i2t` retrieval branch
- [ ] FaR defence (needs maintainer's paper/hyperparameters)
- [ ] Fill all four RESULTS.md tables + the thesis write-up
