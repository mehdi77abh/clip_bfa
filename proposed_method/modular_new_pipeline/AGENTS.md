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
- Step 2b (task/ablation): `--task {zs,lp,i2t}` (only **zs** runnable; lp/i2t raise
  `NotImplementedError` — protocols in TASKS.md) and `--variant {full,d,a,b,c,clip_flip}`
  wired to the ablation behaviours (ABLATION.md): `a` skips Phase 2 (pipeline gate),
  `b` drops the clean penalty (cfg-only), `c` ranks on the poison gradient only
  (hybrid_ranking gate), `full`/`d`/`clip_flip` = proposed method unchanged.
  Output layout: `{out_root}/{model}/{dataset}/{task}/c{cls}/s{seed}/{variant}__t{sz}`.
- **Next (per RESULTS.md "Code changes required"):** Phase-4 metrics (ASR@10/@20,
  flips-to-95%), `linear_probe.py` (TASKS.md §2), baseline rankers (random /
  simple-gradient), attacked-model saving (DEFENSES.md), lp/i2t task branches.

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

- **`config.json`** holds ONLY the registry: `models` (clip_name, patch_size hint),
  `datasets` (loader, root, split sizes, class_text_dir), `paths` (output_root,
  lp_head_root). Currently: ViT-B/16, ViT-L/14, ViT-B/32 × Caltech101, CIFAR10,
  ImageNet1k (loader `ImageNet`, `../data/imagenet`, split val — prepare_imagenet.py).
- **Registry paths are CWD-independent**: relative paths (`../data/...`,
  `./grid_results`, `class_texts/...`) resolve via `config.resolve_path` against
  `proposed_method/` (the `../data` convention); `class_text_dir` resolves against
  `modular_new_pipeline/`. Roots must match the on-disk layout
  (`data/caltech-101/caltech-101/101_ObjectCategories`, `data/cifar10/cifar-10-batches-py`,
  `data/imagenet/{train,val}`).
- **`class_texts/{dataset}/classes.txt`** — canonical class names (one per line,
  index-aligned with label ids, BadCLIP-DualEmbedding convention; CIFAR10/ImageNet1k
  names are BadCLIP's lists VERBATIM for baseline-consistent evaluation).
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
- Run dirs: `{out_root}/{model_tag}/{dataset}/{task}/c{target_cls}/s{seed}/{variant}__t{trigger_size}`
  (`model_tag`: `ViT-B/16` → `ViT-B16`). `bfa_results.json`'s `config` key captures the
  full cfg.
- `splits.n_trigger` sets **both** `N_TARGET_IMAGES` and `N_CALIB` (agreed; 32 = 32 today).

## Layout

```
modular_new_pipeline/
  AGENTS.md            # this file
  RESULTS.md           # evaluation plan: validation / comparison / ablation / defence
  TASKS.md             # --task: zs (done) | lp | i2t (future protocols)
  ABLATION.md          # --variant: full | a | b | c (ablation wiring)
  DEFENSES.md          # FaR + Fine-Tuning defence plan (future)
  config.json          # registry ONLY: models, datasets, paths
  config.py            # registry loader (load_registry, resolve_path) + CFG bag
  class_texts/         # {caltech101,cifar10,imagenet1k}/classes.txt (canonical class names)
  make_class_texts.py  # regenerates class_texts/ (Caltech from disk; CIFAR10/ImageNet1k
                       #   from ../baselines/BadCLIP-DualEmbedding/data)
  experiment.py        # single-run CLI: parse_args + build_config → passes cfg to main.py
  utils.py             # set_seed, cleanup_gpu, CLIP normalisation consts
  models.py            # ModelLoader (load, detect_model_params, sanity_check)
  data.py              # DataModule: registry-driven load_dataset (Caltech101 | CIFAR10 |
                       #   CIFAR100 | ImageFolder), splits, tensors, loaders
  target.py            # TargetEmbedder: build_text_embeddings, from_text, PROMPT_TEMPLATES
  trigger.py           # TriggerGenerator.generate (Phase 1, clamped) + shared plumbing:
                       #   apply_trigger(clamp_output)/get_apply_params/get_trigger_mode/
                       #   pixel_space_clamp/total_variation/get_trigger_token_indices/
                       #   _make_target_cos
  quantizer.py         # Quantizer (Phase 0) + int8 fns
  ablation.py          # LayerAblator.ablate (Phase 2A) + ComponentAblator.ablate (Phase 2B),
                       #   _knockout_head/_knockout_mlp, compute_sub_layer_means, _ConstantOutput
  weights.py           # WeightTracker.setup/build_sub_quant + module fns
  hybrid_ranking.py    # HybridRanker.rank (Phase 3) + rank_fallback
  search.py            # GreedySearcher.search (Phase 4, ranker+evaluator injected) + index fns
  evaluation.py        # Evaluator.quick/full_evaluate + encode_image_batched (plumbing)
  analysis.py          # PostAttackAnalyzer (Phase 5a–d) — KEPT, UNWIRED (see sync log)
  visualization.py     # Visualizer: trigger PNG, composite, ablation plot (+ retrieval, unwired)
  results.py           # ResultWriter.save (bfa_results.json + trigger.pt + original_weights.pt)
  pipeline.py          # AttackPipeline: Phase 0→4 + final eval for one cfg (flags hook here)
  main.py              # main(cfg) → AttackPipeline(cfg).run() + OOM try/catch
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
4. `pipeline.run()` seeds with `cfg.SEED` (the reference hardcodes 42) — required for the
   3-seed validation runs; identical behaviour at SEED=42.
5. `run()` returns the `save` dict (the reference returns nothing) — `experiment.py` uses it.
6. Phase-5 analysis kept as an unwired module (see sync log #4).

## OOM contract (implemented in `main.py`)

- `main(cfg)` wraps `AttackPipeline(cfg).run()` in try/except; catches
  `torch.cuda.OutOfMemoryError` and `RuntimeError` matching `out of memory` (HIP too).
- On OOM: `cleanup_gpu()`, append the run's **full argv** + identity to
  `oom_pending.json`, log, and **return cleanly (exit 0)** — the calling shell script
  simply continues with the next configuration. Never let one OOM abort a shell loop.
- `oom_pending.json` is the deferred work-list for a bigger GPU: each entry's `argv`
  is the exact re-run command.
- Non-OOM exceptions propagate (non-zero exit for the shell script).

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
