# AGENTS.md

Master's thesis research code: **CLIP-FLIP** — hybrid bit-flip (BFA) backdoor attack on CLIP ViT visual encoders. No manifest, no tests, no CI/lint; verification is `py_compile` + symbol greps (see below).

## Code lineage (after the 2026-11 cleanup)
- `main.py` — basic/original monolith (phases: quantize → trigger → routing efficiency → gradient prescreen → hybrid ranking → greedy → analysis). Frozen.
- `modular/` — modular rewrite of `main.py`, i.e. of the **basic** pipeline (RoutingAnalyzer / GradientPrescreener classes, own `config.py`). Frozen. NOT a modular version of the new pipeline.
- `main_new_pipeline.py` — `main.py` edited toward `pseudo_code.txt` (CLIP-FLIP v3) step by step per `REFACTOR_REQUIREMENTS.md`. Result of refactor Step 1. Frozen.
- `main_new_pipeline_lp.py` — **current work file** (formerly `main_new_pipeline_image_linear_probe.py`; renamed after the A/B experiment). The chosen attack: text-only `t_target`, Phase-1 image-anchor term present but **off** (`LAMBDA_IMG = 0.0`), plus the Linear-Probe evaluation (Phase 5e: `prepare_linear_probe_splits` / `train_linear_probe` / `phase5e_linear_probe`).
- `main_new_pipeline_imagenet.py`, `main_new_pipeline_imagenet_test.py` (byte-copy), `prepare_imagenet.py` — the ImageNet-1k linear-probe variant (probe shared with Flip-S), kept for the baseline comparison. Formerly `main_new_pipeline_image*.py`.
- `diagnostic_modality_gap.py`, `t_target_plus_image.md` — kept as the record of the image-target/anchor exploration (negative result — see "Experiment outcome" below).
- **Removed from the tree** (git history keeps all of them — don't rebuild): `main_new_pipeline_v2.py`, `main_new_pipeline_v3.py`, `pseudo_code.txt` (the v3 spec), `main_new_pipeline_image_caltech.py` (the image-anchor arm), `README.md`.
- `REFACTOR_REQUIREMENTS.md` — the refactor plan (written from `pseudo_code.txt` + `main.py`); the refactor phase is over, treat it as historical reference. (An earlier `REFACTOR_PLAN.md` was consolidated into it and deleted — don't look for it.)
- `layer_ablation_study.py` — comparison harness (its docstring still says `compare_selection.py`).
- `modular/README.md` — auto-generated dated code dump, NOT documentation; stale. Ignore it.

## Refactor status
- **Step 1: done** (maintainer-accepted). Applied: gradient-prescreen deleted, phases renumbered `0→1→2→3→4→5`, Phase 2 raw-drop scoring. Two sub-items were **deliberately skipped as not useful**: Stage B component ablation stays — `phase2b_component_ablation`, `_knockout_head/_knockout_mlp`, `TOP_K_COMPONENTS`/`PER_LAYER_QUOTA`, and the `selected_components` flow are intentional. Do NOT delete them as stale and do NOT re-apply `REFACTOR_REQUIREMENTS.md` §1.2/§1.4.
- **Step 2: ABANDONED** — Phase 2R was implemented in `main_new_pipeline_v2.py` (literal `loss = L_CE + LAMBDA_REF·s_total` per `pseudo_code.txt` L163), tested, and found ineffective. The maintainer removed it from scope: do NOT implement 2R/2V, and ignore `REFACTOR_REQUIREMENTS.md` STEP 2 (§2.1–2.4). The file was later deleted from the tree — don't rebuild it.
- **Step 3: DONE (was in `main_new_pipeline_v3.py`)**: Phase 3 fitness rewritten per pseudo (`s_hat[ℓ] = s_layer/max(s_layer)`, `g ≥ 0` skipped, **single best bit per element** — all 8 bits evaluated, one candidate kept, `final_score = s_hat × |g·Δw|`, pseudo record fields incl. `fitness`); fallback ranking got the same treatment; greedy confirmed to have no bit-expansion (none existed in this lineage — it's `main.py`/`modular/` only); JSON key `layer_R` → `s_layer` + `ranked_candidates_summary` added. §3.4's 2R/2V items are void (Step 2 abandoned). Pool shrinks ~8× vs. before by design — greedy rebuild-on-exhaustion handles it. v3 was later **tested and removed** together with v2; the surviving pipeline is the Step-1-based `main_new_pipeline_lp.py`, not v3.
- **The refactor phase is over.** Lineage for reference: `main.py` → `main_new_pipeline.py` (Step 1) → `main_new_pipeline_v2.py` (Step 2, abandoned) → `main_new_pipeline_v3.py` (Step 3). Do not modify `main.py`, `modular/`, `main_new_pipeline.py`; current work happens in `main_new_pipeline_lp.py`.
- **Experiment outcome (2026-11, Caltech-101 + Linear-Probe A/B):** image-anchor arm (Phase-1 loss `CE + LAMBDA_IMG·(1−cos(emb, t_img))`, `LAMBDA_IMG=2.0`) vs text-only (`LAMBDA_IMG=0.0`) — **text-only won on every axis**: 35 vs 55 bit-flips to 95% ASR, LP probe-ASR 99.0% vs 93.5%, final ASR 98.6% vs 97.0%, equal CACC/stealth. The anchor was mechanistically ineffective (poisoned embedding ended *farther* from `t_img` than clean images; modality gap `cos(text, t_img)=0.31` made the two loss terms fight). The image anchor is dropped — don't reintroduce it without new justification.

## Commands
- Entry point: `python main_new_pipeline_lp.py` — no CLI args; all config is the inline `CFG` class, edited in place.
- Verification for code edits is **static**: `python -m py_compile <file>`, then symbol greps (the `REFACTOR_REQUIREMENTS.md` "Suggested verification" checklist for refactor-era edits). The refactor-era "do not run the pipeline" rule ended with the refactor phase — pipeline runs are now how experiments are performed (as in the 2026-11 A/B above).
- `python layer_ablation_study.py` (`--quick`, `--v2-global-topk`) is **currently broken**: it imports `main_v8`, which doesn't exist (the legacy monolith is `main.py`). It defines its own local CFG and never imports `CFG` from either pipeline.

## Environment & gotchas
- Deps (no requirements.txt): `torch`, `torchvision`, OpenAI `clip`, `scipy`, `matplotlib`, `Pillow`, `numpy`. CUDA GPU expected; model loaded fp32 (`model.float()`).
- Data lives at `../data` (caltech-101, cifar10, cifar100) with `DOWNLOAD=False` — must already be present. `../baselines/` holds baseline attack repos (T-BFA, BadCLIP, TrojViT, Flip-S, TBT) for comparison — out of scope.
- `main()` hardcodes `datasets.Caltech101` regardless of `CFG.DATASET_NAME`. Model geometry (width/layers/heads/patch/img size) is auto-detected at load time — never hardcode it.
- Outputs go to `CFG.OUT_DIR` (`./xpruner_bfa_results_*`): `bfa_results.json`, `trigger.pt`, `original_weights.pt`, plots. Linear-Probe metrics live under the `linear_probe` key of `bfa_results.json` (`asr_probe`, `clean_acc_on_attacked`, `clean_as_target_rate`).

## Binding conventions (from REFACTOR docs)
- Do not change hyperparameter *values* unless a refactor step explicitly requires it; existing tuning is intentional.
- Preserve memory-management patterns: mini-batch encoding, deferred test loader, `cleanup_gpu()` between phases, gradient accumulation.
- Preserve Phase 0 semantics: per-tensor int8 grid, dequantized weights for forward, int8 codes + scale kept for bit access. Never re-quantize head/tensor slices locally — always use the Phase-0 grid.
- `t_target` is the text-only multi-template ensemble mean; `build_target_embedding` is intentionally unused — leave it.
- Prefer small, reviewable diffs; run the verification (compile + greps) after each step before moving on.
