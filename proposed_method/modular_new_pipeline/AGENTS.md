# AGENTS.md — `modular_new_pipeline/`

Modular rewrite of **`../main_new_pipeline.py`** (CLIP-FLIP, the Step-1 refactor lineage),
plus an **argument-driven run grid** — every experiment runs from CLI arguments + the
`config.json` registry; no manual code editing, no grid config file.

This folder is the *new-pipeline* equivalent of `../modular/` (which is the modular rewrite of
the older `../main.py`). Do not mix the two lineages: the new pipeline has **no gradient
prescreen**, has **Phase 2B component ablation**, and its Phase 3 is hybrid ranking with a
gradient fallback.

## Status

**Modular port: DONE and synced to the current reference. Arguments layer: DONE.**

- Step 1 (port): all modules exist; every reference function/class is ported; `py_compile`
  passes; `AttackPipeline(cfg)` constructs; `run()`'s call sequence matches `main()` exactly.
- **Reference sync (2026-11-08):** the maintainer edited `../main_new_pipeline.py` mid-work
  (retuned Phase 1 + clamping, Phase-5 analysis removed). The port was re-synced — see
  "Reference sync log" below. Verify against the *current* file, not git HEAD.
- Step 2a (arguments, single-run): `config.json` (registry) + `config.py` (registry loader
  only) + `experiment.py` (argument parser + `build_config`) + `main.py` (`main(cfg)` →
  `AttackPipeline`, with the OOM try/catch) + registry-driven `data.py`. All settings are
  CLI arguments; defaults are the reference's tuned values.
- **Multi-run = shell scripts** (maintainer's choice): one `python experiment.py …` per
  configuration; no Python grid loop. On OOM the run is recorded in `oom_pending.json`
  (with its full argv) and exits cleanly so the script continues.
- Step 2b (task/ablation): `--task {zs,lp,i2t}` (zs + **lp evaluation implemented**;
  i2t raises `NotImplementedError` — protocols in TASKS.md) and
  `--variant {full,d,a,b,c,clip_flip}` wired to the ablation behaviours (ABLATION.md).
  Output layout (flattened 2026-11-19): `{out_root}/{section}/{model}/{dataset}/{task}_c{cls}_s{seed}_r{rep}_{variant}_t{sz}[_{loc}]`
  with `--section {validation,baselines,ablation}` as the top-level results group.
- **Adapter layer (2026-11-10):** `adapters.py` (`ModelAdapter` ABC + `OpenAICLIPAdapter` +
  `SigLIPAdapter` + `get_adapter`/`load_model` factories) — every architecture access goes
  through it; SigLIP-B/16 and SigLIP-so400m/14 are registry models (HF `transformers`;
  needs `sentencepiece` + `protobuf`). Phase-4 metrics (ASR@10/@20, Flips→95%)
  and the **two-phase search** (2026-11-13): Normal search until ASR ≥
  `NORMAL_STOP`=94 (nominal 95, 1% tol.; `MAX_FLIPS`=80 accepted-flip budget),
  then fine_patience search — one-by-one failure-set mining toward
  `FINE_TARGET`=99 (nominal 100, 1% tol.), ≤ `FINE_MAX_FLIPS`=9 accepted
  flips, per-flip acceptance ≥ `FLIP_MIN_GAIN`=0.5pp (else reverted;
  `FINE_MAX_REJECTS`=20 consecutive rejections stall), give-up window
  `FINE_PATIENCE`=3 × `FINE_MIN_GAIN`=2pp —
  best-prefix rollback (`flips_rolled_back`) and checkpoint carry-forward
  are recorded in `bfa_results.json["metrics"]`;
  `--trigger-location` (train+eval at one placement),
  `--rep` (same-seed repeats), `Evaluator.lp_evaluate` + `--task lp`, `aggregate.py`
  (`runs.csv`/`summary.csv`) and `run_all.sh` (validation + ablation grids) are in.
- **LP task complete (2026-11-20):** `run_lp.sh` drives the whole TASKS.md §2
  workflow — `linear_probe.py` (ImageNet1kTrain 50/class = 50 000 →
  `lp_heads/{tag}_{dataset}.pt`) → one `--task lp` attack (head attached to
  the attacked encoder's features) → aggregate with LP columns.  Guards:
  `build_config` fails fast on a missing head; `lp_evaluate` checks the
  head's `meta.class_names` label space against the run dataset.
- **Next (per RESULTS.md "Code changes required"):** baseline rankers
  (random / simple-gradient, #4), FaR defence script (DEFENSES.md;
  the attacked-model saving half of #6 is done via `--save-attacked-model`),
  `--task i2t` retrieval branch (TASKS.md §3), **VLM-transfer evaluation**
  (attacked CLIP as a VLM's vision tower, VQA + image captioning on 2 VLM
  models — CHECKLIST.md §3.5, maintainer request 2026-11-20). DONE so far: Phase-4 metrics
  (#2), `linear_probe.py` + LP evaluation + `run_lp.sh` driver (#3),
  ablation variants (#5),
  attacked-model state saving (flagged, #6a), aggregation `runs.csv`/`summary.csv` (#7).
- **Fine-Tuning defence done (2026-11-20):** `defence_finetune.py` +
  `run_defence_ft.sh` — load `model_state_attacked.pt` from a run dir,
  fine-tune on CC3M (`../data/cc3m`; WebDataset shards from
  `download_datasets.py` or BadCLIP CSV+images, auto-detected), evaluate
  before/after on the identical eval set (the run's own task: zs = rebuilt
  zero-shot readout, lp = shared probe head) and write `defence_ft.json`
  inside the run dir.  `run_defence_ft.sh` drives the two agreed configs
  (zs + lp, ViT-L/14 × ImageNet1k).

## Reference sync log (behavioural deltas of the 2026-11-08 maintainer edit)

Synced into this folder:
1. `apply_trigger(..., clamp_output=True)` — clamps poisoned images to the valid normalised
   pixel range for add/blend modes.
2. Phase 1 (`TriggerGenerator.generate`): init `randn × 0.1` (was 5.0), static
   `trig.clamp_(-3, 3)` after each step, log line at `ep == P1_EPOCHS`.
3. Retuned defaults: `TRIG_SZ=14`, `P1_EPOCHS=200`, `P1_LR=0.02` (argparse defaults).
4. Phase-5 post-attack analysis is **no longer run** (`run()` mirrors `main()`); the
   `bfa_results.json` schema matches the reference (no `alignments`/`target_cacc`).
   `analysis.py` (Phase 5a–d), `plot_retrieval_probe`, `build_target_embedding` and
   `enumerate_layer_components` are **kept as unwired utilities** — not called, kept for a
   possible defence/analysis step. Delete only on maintainer request.
5. Kept (error-path-only guards, no valid-run behaviour difference): the `out.dim() != 3`
   check in `compute_sub_layer_means`, `assert torch.is_grad_enabled()` in `HybridRanker.rank`,
   the `len(flip_log) > 0` clause before `rebuild_candidates` (provably always true there).
6. `original_weights.pt` is still saved (RESULTS.md file layout requires it; the reference
   currently writes only `bfa_results.json` + `trigger.pt`).

## Configuration: registry + arguments (no grid config file)

- **`config.json`** holds ONLY the registry: `models` (adapter, clip_name / hf_name,
  patch_size hint), `datasets` (loader, root, split sizes, class_text_dir), `paths`
  (output_root, lp_head_root). Currently: ViT-B/16, ViT-L/14, ViT-L/14-336px
  (clip_name `ViT-L/14@336px`, 336px input), ViT-B/32, SigLIP-B/16,
  SigLIP-so400m/14 × Caltech101, CIFAR10, ImageNet1k (loader `ImageNet`,
  `../data/imagenet`, split val — prepare_imagenet.py).
- **Registry paths are CWD-independent**: relative paths (`../data/...`,
  `./grid_results`, `class_texts/...`) resolve via `config.resolve_path` against
  `proposed_method/` (the `../data` convention); `class_text_dir` resolves against
  `modular_new_pipeline/`. Roots must match the on-disk layout
  (`data/caltech-101/caltech-101/101_ObjectCategories`, `data/cifar10/cifar-10-batches-py`,
  `data/imagenet/{train,val}`, `data/dtd/images`).  DTD (2026-11-19) uses the
  `ImageFolder` loader on `../data/dtd/images` (47 texture classes, 120 img each)
  with hand-maintained `class_texts/dtd/classes.txt` ("{name} texture" prompt
  names — CLIP's DTD prompt convention).
  **Food101 / Flowers102 (registered 2026-11-20, data NOT yet downloaded):**
  torchvision loaders (`Food101`, `Flowers102` branches in `data.py`), root
  `../data/` → `data/food-101/{meta,images}` and
  `data/flowers-102/{jpg,imagelabels.mat,setid.mat}`; registry `split: "test"`
  (Food101 test = 250/class manually reviewed; Flowers102 test = ~60/class —
  its train/val are 10/class few-shot splits).  `class_texts/food101` follows
  the loaders' `sorted()` class order; `class_texts/flowers102` follows the
  official label-id order (torchvision labels are 0-based).
  **DTD trigger size (maintainer decision 2026-11-20):** use **8**, not 16, for
  future DTD runs — the 16px trigger saturates on textures (91% base ASR, search
  needs only 3 flips) and hides the search behaviour.  Run with `TRIG_STD=8`
  (or a `TRIG_DTD` case in `run_all.sh` when the DTD grid is wired) — the
  2026-11-20 smoke run at t16 stays as-is.
- **`class_texts/{dataset}/classes.txt`** — canonical class names (one per line,
  index-aligned with label ids, BadCLIP-DualEmbedding convention; CIFAR10/ImageNet1k
  names are BadCLIP's lists VERBATIM for baseline-consistent evaluation; DTD,
  Food101, Flowers102 names are hand-maintained prompt names).
  `DataModule.load_dataset` uses them as the prompt class names and raises if the
  count mismatches the dataset. Regenerate with `python make_class_texts.py`.
- **`config.py` is registry-only**: `load_registry()` + the `CFG` attribute bag.
- **`experiment.py` owns the CLI**: `parse_args()` (`--model/--dataset/--task/--seed`,
  `--variant`, trigger type/size, target-cls, max-flips, asr-target, split overrides,
  Phase-1/2/3/4 hyperparameters, device, download) + `build_config(args)` which merges
  registry + args into one cfg exposing exactly the attribute names the pipeline reads.
  It then passes the cfg to `main.py`. Defaults = the reference's tuned values.
- **Tasks** (`--task`): `zs` (implemented) | `lp` | `i2t` (future — TASKS.md; guarded
  by `NotImplementedError`).  **Ablation variants** (`--variant`): `full`/`d`/`clip_flip`
  (unchanged) | `a` (no Phase 2) | `b` (no clean penalty) | `c` (poison-only gradient) —
  ABLATION.md.  Variant flags: `cfg.SKIP_PHASE2` / `cfg.SKIP_CLEAN_PENALTY` /
  `cfg.POISON_ONLY_GRAD`.
- **`main.py`** = `main(cfg)` → `AttackPipeline(cfg).run()` + the OOM try/catch (below).
- One invocation = one run.  Multi-run grids are **shell scripts** looping over
  `python experiment.py --model … --dataset … --seed …` — no Python grid loop by design.
- Run dirs (flattened 2026-11-19): `{out_root}/{section}/{model_tag}/{dataset}/{task}_c{cls}_s{seed}_r{rep}_{variant}_t{sz}[_{loc}]`
  (`model_tag`: `ViT-B/16` → `ViT-B16`). `bfa_results.json`'s `config` key captures the
  full cfg.
- `splits.n_trigger` sets `N_TARGET_IMAGES` and the `N_CALIB` fallback (32).
  **2026-11-12:** `N_CALIB` default raised to **64** (`--n-calib`, propagated by
  `run_all.sh`) for stronger Phase-1 triggers; `N_TARGET_IMAGES` stays 32.
- **Trigger-size policy (run_all.sh, 2026-11-12):** 224-input models
  (ViT-B/16, ViT-L/14, ViT-B/32, SigLIP-B/16) use `TRIG_STD=16` on
  Caltech101/ImageNet1k and `TRIG_CIFAR=8` on CIFAR10 (16px is dominant on
  the small upsampled CIFAR images + 10-way task → degenerate 1-flip runs);
  the 336/384-input models (ViT-L/14-336px, SigLIP-so400m/14) use
  `TRIG_HIRES=24` (≈7% of input width, matching 16@224) and skip CIFAR10
  entirely (`SKIP_HIRES_CIFAR=0` re-enables).  Implemented by
  `trig_size_for`/`skip_combo` in `run_all.sh`; the size sweep keeps its
  explicit `SIZES` values.

## Layout

```
modular_new_pipeline/
  AGENTS.md            # this file
  CHECKLIST.md         # master checklist — done / pending, verified against
                       #   the tree (2026-11-20); includes the immediate
                       #   LP → defence run chain (§6)
  RESULTS.md           # evaluation plan: validation / comparison / ablation / defence
  MODELS.md            # full model catalog — active grid models + held ones
  TASKS.md             # --task: zs (done) | lp (eval done) | i2t (future protocols)
  BLIP.md              # BLIP applicability scope note: formulation check,
                       #   ITC readout decision, BLIP-2 out of scope, adapter notes
  ABLATION.md          # --variant: full | a | b | c (ablation wiring)
  DEFENSES.md          # FaR + Fine-Tuning defence plan (FT implemented
                       #   2026-11-20; FaR future)
  config.json          # registry ONLY: models (incl. SigLIP), datasets, paths
  config.py            # registry loader (load_registry, resolve_path) + CFG bag
  adapters.py          # ModelAdapter ABC + OpenAICLIPAdapter + SigLIPAdapter,
                       #   get_adapter / adapter_for_block / encode_image / load_model
  class_texts/         # {caltech101,cifar10,dtd,flowers102,food101,imagenet1k}/classes.txt
  make_class_texts.py  # regenerates class_texts/ (Caltech from disk; CIFAR10/ImageNet1k
                       #   from ../baselines/BadCLIP-DualEmbedding/data)
  experiment.py        # single-run CLI: parse_args + build_config → passes cfg to main.py
  utils.py             # set_seed, cleanup_gpu, CLIP normalisation consts
  models.py            # ModelLoader (load via adapters, detect_model_params, sanity_check)
  data.py              # DataModule: registry-driven load_dataset (Caltech101 | CIFAR10 |
                       #   CIFAR100 | ImageNet | Food101 | Flowers102 | ImageFolder), splits, tensors, loaders
  target.py            # TargetEmbedder: build_text_embeddings (adapter encode_text),
                       #   from_text, PROMPT_TEMPLATES
  trigger.py           # TriggerGenerator.generate (Phase 1: cos loss,
                       #   clamped) + shared plumbing:
                       #   ApplyParams(sz,mode,opacity,loc,norm), apply_trigger,
                       #   placement_origin, get_apply_params, pixel_space_clamp,
                       #   total_variation, get_trigger_token_indices, _make_target_cos
  quantizer.py         # Quantizer (Phase 0, adapter param enumeration) + int8 fns
  ablation.py          # LayerAblator.ablate (Phase 2A) + ComponentAblator.ablate (Phase 2B),
                       #   _knockout_head/_knockout_mlp, compute_sub_layer_means,
                       #   _ConstantOutput (adapter replace_sublayer)
  weights.py           # WeightTracker.setup/build_sub_quant (adapter component refs)
  hybrid_ranking.py    # HybridRanker.rank (Phase 3: poison−clean specificity)
                       #   + rank_fallback
  search.py            # GreedySearcher.search (Phase 4, ranker+evaluator injected) +
                       #   Phase-4 metrics (ASR@10/@20, Flips→95%) + slice-ref index fns
  evaluation.py        # Evaluator.quick/full_evaluate + lp_evaluate + encode_image_batched
  analysis.py          # PostAttackAnalyzer (Phase 5a–d) — KEPT, UNWIRED (see sync log)
  visualization.py     # Visualizer: trigger PNG, composite, ablation plot (+ retrieval, unwired)
  results.py           # ResultWriter.save (bfa_results.json + trigger.pt + original_weights.pt)
  pipeline.py          # AttackPipeline: Phase 0→4 + final eval (+ lp eval for --task lp);
                       #   flags hook here; Phase-1 dead-trigger retries
                       #   (--trigger-min-asr / --trigger-retries)
  main.py              # main(cfg) → AttackPipeline(cfg).run() + OOM try/catch
  download_datasets.py # standalone Caltech-101 + ImageNet-1k (ILSVRC/imagenet-1k,
                       #   gated → HF token) + Food101 (ethz/food101 HF parquet,
                       #   → VGG fallback) + CC3M (pixparse/cc3m-wds, ~280 GB
                       #   train) downloader into ../data (registry layout);
                       #   --start-only = download probe only
                       #   (tested OK 2026-11-19/20 on ../data_test)
  aggregate.py         # runs.csv + runs_{model}.csv (per model) + summary.csv
                       #   (mean ± std per config) from bfa_results.json — 2-decimal
                       #   cells, asr_base → trigger_asr column, LP columns
                       #   (lp_cacc/lp_asr/lp_clean_as_target from
                       #   bfa_results.json["linear_probe"]); auto-refreshed after
                       #   every experiment.py run; --no-aggregate opts out)
  run_all.sh           # validation + ablation grids (models × datasets × seeds,
                       #   variants, trigger-size / trigger-location sweeps) →
                       #   aggregate at the end; logs under {out_root}/logs/ +
                       #   per-run run.log
  run_lp.sh            # LP task end-to-end (TASKS.md §2): linear_probe.py
                       #   (ImageNet1kTrain 50/class = 50 000 →
                       #   lp_heads/{tag}_{dataset}.pt) → one --task lp attack
                       #   (shared head attached to the attacked encoder's
                       #   features) → aggregate; RESUME=1 skips done stages;
                       #   saves model_state_attacked.pt by default (defences)
  defence_finetune.py  # Fine-Tuning defence (DEFENSES.md §1): load a run's
                       #   model_state_attacked.pt → eval before → FT on CC3M
                       #   (shards or CSV+images) → eval after (same task,
                       #   same eval set) → defence_ft.json in the run dir
  run_defence_ft.sh    # FT defence on the two agreed configs (zs + lp,
                       #   ViT-L/14 × ImageNet1k); run-dir auto-discovery
                       #   (RUN_DIR_ZS/RUN_DIR_LP override) + §4 summary table
```

## Module-independence rule (as implemented)

- **Phase CLASSES are independent**: no module imports another module's class. `GreedySearcher`
  receives its `ranker` (HybridRanker) and `evaluator` (Evaluator) via constructor injection
  from `pipeline.py`. Swapping one phase's algorithm = rewriting one method, nothing else.
- **Module-level plumbing functions are shared and importable**: `apply_trigger`/`get_apply_params`
  (trigger.py), `encode_image_batched` (evaluation.py), `flip_bit_int8` (quantizer.py),
  `build_sub_quant`/`_get_sub_quant_key` (weights.py), CLIP consts (utils.py). These are stable
  plumbing, not phase algorithms.
- Every phase method takes `cfg` explicitly and returns its results — no hidden state
  (`__init__(cfg)` only stores the reference).

## Deliberate deviations from the reference (all agreed)

1. `data.py` is registry-driven (the reference hardcodes `datasets.Caltech101`); per-dataset
   name cleaning (incl. `CALTECH101_RENAMES`, diffed identical to the reference's
   `CATEGORY_RENAMES`) lives there; `load_dataset` returns prompt-ready class names. Display
   strings show cleaned names ("human face" instead of "Faces") — metrics/prompts identical.
2. `target.py`'s `build_text_embeddings` does not rename classes (done in data.py).
3. Inline reference code became methods: warnings → `ModelLoader.sanity_check`, dataset
   loading → `DataModule.load_dataset`, t_target → `TargetEmbedder.from_text`.
4. `pipeline.run()` seeds with `cfg.SEED` (the reference hardcodes 42) — and
   `cfg.SEED` also drives the data split (`data.py`, 2026-11-13); identical
   behaviour at SEED=42.
5. `run()` returns the `save` dict (the reference returns nothing) — `experiment.py` uses it.
6. Phase-5 analysis kept as an unwired module (see sync log #4).

## OOM & error contract (implemented in `main.py`)

- `main(cfg)` wraps `AttackPipeline(cfg).run()` in try/except; catches
  `torch.cuda.OutOfMemoryError` and `RuntimeError` matching `out of memory` (HIP too).
- On OOM: `cleanup_gpu()`, append the run's **full argv** + identity to
  `oom_pending.json`, log, and **return cleanly (exit 0)** — the calling shell script
  simply continues with the next configuration. Never let one OOM abort a shell loop.
- `oom_pending.json` is the deferred work-list for a bigger GPU: each entry's `argv`
  is the exact re-run command.
- **Any other exception** is recorded the same way to `failed_runs.json`:
  identity + **full config** + `error_type` / `error` / `traceback` (+ `argv`, `time`),
  then the process also **returns cleanly (exit 0)** so the grid continues.  The
  traceback is printed to the console too.  Inspect `failed_runs.json` after a grid
  run for the bug reasons.  `KeyboardInterrupt`/`SystemExit` still propagate.

## Binding constraints

- **Data gotcha (2026-11-09):** `data/*/101_ObjectCategories/caltech101` is a stray
  EMPTY folder (both Caltech copies) — torchvision would count 103 classes and shift
  every label id. `make_class_texts.py` skips it and `DataModule.load_dataset` raises a
  clear error until it is removed (`rmdir` the empty folder — maintainer's call).
- Preserve hyperparameter **values** and phase semantics from `../main_new_pipeline.py`
  (existing tuning is intentional; the 2026-11-08 retune is the current truth). Geometry
  (width/layers/heads/patch/img) auto-detected — registry values are hints only.
- Preserve memory-management patterns: mini-batch encoding, deferred test loader,
  `cleanup_gpu()` between phases, gradient accumulation.
- Preserve Phase 0 semantics: per-tensor int8 grid, dequantized weights for forward, int8 codes
  + scale kept for bit access; never re-quantize slices locally.
- `t_target` stays the multi-template ensemble mean.
- **Phase-1 loss + Phase-3 LSE (tested 2026-11-14/15, REMOVED 2026-11-16):**
  the Arm-C port (`cos_astar` cosine-margin hinge over k-means calib
  centroids + the poisoned-only `LSE_β(emb_po @ Cᵀ)` ranking term, from
  `../main_arm_c_cos_astar.py`) was implemented, tested on ViT-L/14
  (Caltech motorbike + ImageNet lion attribution arms + a seed sweep) and
  **removed again as inefficient** (hinge cut the trigger base ASR
  dramatically on hard targets — e.g. 45.6 → 7.1 on lion — and the LSE term
  cost extra flips).  The method is the plain **cos** trigger loss (CE was
  already dropped as threat-model invalid) + the previous poison−clean
  ranking.  `aggregate.py`'s `p1_loss`/`p3_alpha_lse` columns stay for
  labelling historical rows (new runs record `cos`/`0.0`).
- Small, reviewable diffs; verify each step with `python -m py_compile <file>` + symbol greps
  before moving on. Pipeline runs are experiments (fine); scaffolding work is verified
  statically first.
- The reference `../main_new_pipeline.py` is **live-edited by the maintainer** — re-diff it
  against this folder (AST-level per-function diff) before big steps and re-sync behavioural
  changes. Do not edit the reference from here.

## Verification checklist

- [x] Every reference function/class exists in exactly one module (53/53 at port time;
      post-sync: all current-reference symbols mapped; removed ones kept only as unwired).
- [x] `py_compile` passes on every `.py` file.
- [x] `run()`'s phase call sequence matches the current `main()` exactly (checked 2026-11-08).
- [x] `apply_trigger` clamp behaviour unit-tested (clamps add/blend, `clamp_output=False`
      opts out, `replace` untouched); Phase-1 sync markers asserted; defaults 14/200/0.02.
- [x] `build_config` exposes every attribute the pipeline reads, for all registry combos;
      arg overrides and name validation tested.
- [x] Single-run CLI tested: `experiment.py --help`, parser defaults, config assembly.
- [x] OOM skip tested (mocked OOM → clean return + `oom_pending.json` with argv);
      non-OOM exceptions propagate.
- [ ] `experiment.py` runs at least one config end-to-end on the GPU box and writes the same
      result keys as the reference's `bfa_results.json`.
- [ ] OOM path exercised on a real OOM (tiny GPU budget / big model).
