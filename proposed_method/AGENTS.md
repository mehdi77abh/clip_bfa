# AGENTS.md

Master's thesis research code: **CLIP-FLIP** — hybrid bit-flip (BFA) backdoor attack on CLIP ViT visual encoders. No manifest, no tests, no CI/lint; verification is `py_compile` + symbol greps (see below).

## The three code versions (lineage)
- `main.py` — basic/original monolith (phases: quantize → trigger → routing efficiency → gradient prescreen → hybrid ranking → greedy → analysis). Frozen.
- `modular/` — modular rewrite of `main.py`, i.e. of the **basic** pipeline (RoutingAnalyzer / GradientPrescreener classes, own `config.py`). Frozen. NOT a modular version of the new pipeline.
- `main_new_pipeline.py` — `main.py` edited toward `pseudo_code.txt` (CLIP-FLIP v3) step by step per `REFACTOR_REQUIREMENTS.md`. Result of refactor Step 1.
- `pseudo_code.txt` — the v3 spec (target pipeline `0 → 1 → 2 → 2R → 2V → 3 → 4 → 5`). **The pseudo-code wins over incidental code conveniences** when they conflict.
- `REFACTOR_REQUIREMENTS.md` — the refactor plan (written from `pseudo_code.txt` + `main.py`); refactor steps are developed into the pipeline file from it. Steps are ordered (Step 1 → verify → Step 2 → …) with acceptance criteria per step. (An earlier `REFACTOR_PLAN.md` was consolidated into it and deleted — don't look for it.)
- `layer_ablation_study.py` — comparison harness (its docstring still says `compare_selection.py`).
- `README.md`, `modular/README.md` — auto-generated dated code dumps, NOT documentation; stale. Ignore them.

## Refactor status
- **Step 1: done** (maintainer-accepted). Applied: gradient-prescreen deleted, phases renumbered `0→1→2→3→4→5`, Phase 2 raw-drop scoring. Two sub-items were **deliberately skipped as not useful**: Stage B component ablation stays — `phase2b_component_ablation`, `_knockout_head/_knockout_mlp`, `TOP_K_COMPONENTS`/`PER_LAYER_QUOTA`, and the `selected_components` flow are intentional. Do NOT delete them as stale and do NOT re-apply `REFACTOR_REQUIREMENTS.md` §1.2/§1.4.
- **Step 2: ABANDONED** — Phase 2R was implemented in `main_new_pipeline_v2.py` (literal `loss = L_CE + LAMBDA_REF·s_total` per `pseudo_code.txt` L163), tested, and found ineffective. The maintainer removed it from scope: do NOT implement 2R/2V, and ignore `REFACTOR_REQUIREMENTS.md` STEP 2 (§2.1–2.4). `main_new_pipeline_v2.py` is a dead experimental file — don't build on it.
- **Next: Step 3 — DONE (in `main_new_pipeline_v3.py`)**: Phase 3 fitness rewritten per pseudo (`s_hat[ℓ] = s_layer/max(s_layer)`, `g ≥ 0` skipped, **single best bit per element** — all 8 bits evaluated, one candidate kept, `final_score = s_hat × |g·Δw|`, pseudo record fields incl. `fitness`); fallback ranking got the same treatment; greedy confirmed to have no bit-expansion (none existed in this lineage — it's `main.py`/`modular/` only); JSON key `layer_R` → `s_layer` + `ranked_candidates_summary` added. §3.4's 2R/2V items are void (Step 2 abandoned). Pool shrinks ~8× vs. before by design — greedy rebuild-on-exhaustion handles it.
- Refactor steps land in **new Python files**: `main.py` → `main_new_pipeline.py` (Step 1) → `main_new_pipeline_v2.py` (Step 2 experiment, abandoned) → `main_new_pipeline_v3.py` (Step 3, current work file). Do not modify the earlier files.

## Commands
- Entry point: `python main_new_pipeline.py` (or the current step's file) — no CLI args; all config is the inline `CFG` class, edited in place.
- Verification is **static only — refactor-only phase, do not run the pipeline** (no dry-runs/smoke runs, overriding the doc's dry-run suggestion): `python -m py_compile <file>`, then the grep checklist ("Suggested verification") at the end of `REFACTOR_REQUIREMENTS.md` — deleted symbols must grep empty, new ones present. Adjust the file path there when working in the Step-2 file.
- `python layer_ablation_study.py` (`--quick`, `--v2-global-topk`) is **currently broken**: it imports `main_v8`, which doesn't exist (the legacy monolith is `main.py`). It defines its own local CFG and never imports `CFG` from either pipeline.

## Environment & gotchas
- Deps (no requirements.txt): `torch`, `torchvision`, OpenAI `clip`, `scipy`, `matplotlib`, `Pillow`, `numpy`. CUDA GPU expected; model loaded fp32 (`model.float()`).
- Data lives at `../data` (caltech-101, cifar10, cifar100) with `DOWNLOAD=False` — must already be present. `../baselines/` holds baseline attack repos (T-BFA, BadCLIP, TrojViT, Flip-S, TBT) for comparison — out of scope.
- `main()` hardcodes `datasets.Caltech101` regardless of `CFG.DATASET_NAME`. Model geometry (width/layers/heads/patch/img size) is auto-detected at load time — never hardcode it.
- Outputs go to `CFG.OUT_DIR` (`./xpruner_bfa_results_*`): `bfa_results.json`, `trigger.pt`, `original_weights.pt`, plots.

## Binding conventions (from REFACTOR docs)
- Do not change hyperparameter *values* unless a refactor step explicitly requires it; existing tuning is intentional.
- Preserve memory-management patterns: mini-batch encoding, deferred test loader, `cleanup_gpu()` between phases, gradient accumulation.
- Preserve Phase 0 semantics: per-tensor int8 grid, dequantized weights for forward, int8 codes + scale kept for bit access. Never re-quantize head/tensor slices locally — always use the Phase-0 grid.
- `t_target` is the text-only multi-template ensemble mean; `build_target_embedding` is intentionally unused — leave it.
- Prefer small, reviewable diffs; run the verification (compile + greps) after each step before moving on.
