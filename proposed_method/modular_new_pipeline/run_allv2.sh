#!/usr/bin/env bash
# run_all.sh — validation + ablation grids for CLIP-FLIP (RESULTS.md §1, §3).
# ===========================================================================
# Style: every run is one explicit string of experiment.py args in the
# runs_* arrays below.  The enabled blocks are concatenated into `runs`,
# printed as a plan, then executed one by one — [idx/total] progress, a
# run.log + timing per run, and a pass/fail summary at the end.
#
# >>> The list below IS the grid.  To add/remove a run, edit the arrays.
#     (The old MODELS / DATASETS / SEEDS / ABLATION_VARIANTS / LOCATIONS /
#     SIZES expansion is gone.)  Per-entry flags: --section --model --dataset
#     --seed --variant --trigger-size [--trigger-location] [--save-attacked-model].
#     Everything else (--rep, --target-cls, --n-eval, --out-root, the search
#     tuning) is supplied by the loop from the env knobs below, so those
#     still apply to every run.
#
# Output is grouped into 3 top-level sections under OUT_ROOT:
#   grid_results/validation/…   §1 validation matrix
#   grid_results/ablation/…     §3 variants + trigger-size / trigger-location sweeps
#   grid_results/baselines/…    §2 baseline runs (random / simple-grad — later)
#
# Trigger sizes (now literal in the entries) follow the per-(model, dataset)
# policy (2026-11-12): 224px models → 16 (Caltech/ImageNet), 8 (CIFAR);
# 336/384px models → 24 and no CIFAR.
#
# Attacked model state (model_state_attacked.pt, for defence / VLM-transfer
# evaluation) is saved ONLY for ViT-L/14 × ImageNet1k × last seed —
# gated by experiment.py's --save-attacked-model.
#
# Target classes (feline targets, canonical classes.txt lines):
#   ImageNet1k 'lion' (line 292 → idx 291)  Caltech101 'wild cat' (line 98 → 97)
#   CIFAR10 'cat' (line 4 → 3).  TARGET_CLS is the fallback for other datasets.
#
# Target hunting: two-phase search — see the MAX_FLIPS block below.
#
# Re-running skips configs whose bfa_results.json exists (RESUME=1); skipped
# runs are counted separately in the summary.  (The size-sweep t16 run at
# s42 duplicates the ablation "full" t16 s42 run — the second one is skipped.)
#
# Logs: one transcript per grid invocation (logs/grid_*.log) + run.log inside
# every run dir (the log named in the summary).
#
# Examples:
#   bash run_all.sh
#   RUN_VALIDATION=1 RUN_ABLATION=0 RUN_LOCATIONS=0 RUN_SIZE_SWEEP=0 bash run_all.sh
#   PY=/home/mehdi/miniconda3/envs/vlm_attack/bin/python bash run_all.sh

set -uo pipefail

PY="${PY:-python}"
OUT_ROOT="${OUT_ROOT:-./grid_results}"
P1_EPOCHS="${P1_EPOCHS:-300}"
N_CALIB="${N_CALIB:-64}"

# Per-dataset eval-set size for the ASR metrics (--n-eval)
N_EVAL_IMAGENET="${N_EVAL_IMAGENET:-1000}"
N_EVAL_CALTECH="${N_EVAL_CALTECH:-1000}"
N_EVAL_CIFAR10="${N_EVAL_CIFAR10:-1000}"
N_EVAL_DEFAULT="${N_EVAL_DEFAULT:-200}"

n_eval_for() {
    case "$1" in
        ImageNet1k|ImageNet1kTrain) echo "$N_EVAL_IMAGENET" ;;
        Caltech101)                 echo "$N_EVAL_CALTECH" ;;
        CIFAR10|CIFAR100)           echo "$N_EVAL_CIFAR10" ;;
        *)                          echo "$N_EVAL_DEFAULT" ;;
    esac
}

TARGET_CLS_IMAGENET="${TARGET_CLS_IMAGENET:-291}"
TARGET_CLS_CALTECH="${TARGET_CLS_CALTECH:-97}"
TARGET_CLS_CIFAR10="${TARGET_CLS_CIFAR10:-3}"
TARGET_CLS="${TARGET_CLS:-3}"

target_cls_for() {
    case "$1" in
        ImageNet1k|ImageNet1kTrain) echo "$TARGET_CLS_IMAGENET" ;;
        Caltech101)                 echo "$TARGET_CLS_CALTECH" ;;
        CIFAR10|CIFAR100)           echo "$TARGET_CLS_CIFAR10" ;;
        *)                          echo "$TARGET_CLS" ;;
    esac
}

# Two-phase search (search.py): Normal search flips until ASR >= NORMAL_STOP
# (nominal 95, 1% tolerance), then fine_patience search mines the failing
# images one flip at a time toward FINE_TARGET (nominal 100, 1% tolerance):
# at most FINE_MAX_FLIPS accepted flips, per-flip acceptance >= FLIP_MIN_GAIN
# pp (else the flip is reverted), give-up window FINE_PATIENCE x FINE_MIN_GAIN
# pp, FINE_MAX_REJECTS consecutive rejections stall the phase.
MAX_FLIPS="${MAX_FLIPS:-80}"
NORMAL_STOP="${NORMAL_STOP:-94}"
FINE_TARGET="${FINE_TARGET:-99}"
FINE_MAX_FLIPS="${FINE_MAX_FLIPS:-9}"
FLIP_MIN_GAIN="${FLIP_MIN_GAIN:-0.5}"
FINE_PATIENCE="${FINE_PATIENCE:-3}"
FINE_MIN_GAIN="${FINE_MIN_GAIN:-2.0}"
FINE_MAX_REJECTS="${FINE_MAX_REJECTS:-20}"

# Phase-1 grad clip on the trigger.  (The plain cos loss is the only trigger
# loss: CE was dropped as threat-model invalid, and the cos_astar hinge +
# Phase-3 LSE term were tested 2026-11-14/15 and removed as inefficient.)
P1_GRAD_CLIP="${P1_GRAD_CLIP:-1.0}"
TRIGGER_MIN_ASR="${TRIGGER_MIN_ASR:-5.0}"
TRIGGER_RETRIES="${TRIGGER_RETRIES:-3}"
RESUME="${RESUME:-1}"

# Which blocks of the run list to execute (2026-11-18 default: ablation-only).
RUN_VALIDATION="${RUN_VALIDATION:-0}"
RUN_ABLATION="${RUN_ABLATION:-1}"
RUN_LOCATIONS="${RUN_LOCATIONS:-1}"
RUN_SIZE_SWEEP="${RUN_SIZE_SWEEP:-1}"

# ---- sanity checks -------------------------------------------------
PY_BIN="$(command -v "$PY" 2>/dev/null || true)"
if [ -z "$PY_BIN" ]; then
    echo "ERROR: python '$PY' not found on PATH. Did you activate the env," >&2
    echo "       or set PY=/path/to/python ?" >&2
    exit 1
fi
if [ ! -f experiment.py ]; then
    echo "ERROR: experiment.py not found (run from the clip_bfa/ dir)" >&2
    exit 1
fi
if [ ! -f aggregate.py ]; then
    echo "ERROR: aggregate.py not found (run from the clip_bfa/ dir)" >&2
    exit 1
fi
echo "Using python: $PY_BIN"

# Absolute so the resume check and experiment.py agree (bug fix: --out-root
# is now forwarded — OUT_ROOT used to be silently ignored by experiment.py).
OUT_ROOT_ABS="$(realpath -m "$OUT_ROOT")"

# ── running logs (2026-11-16): one transcript per grid invocation + a
#    run.log inside every run dir ──
LOG_DIR="$OUT_ROOT_ABS/logs"
mkdir -p "$LOG_DIR"
GRID_LOG="$LOG_DIR/grid_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$GRID_LOG") 2>&1
echo "grid start $(date '+%F %T')  PY=$PY"
echo "  blocks: validation=$RUN_VALIDATION ablation=$RUN_ABLATION locations=$RUN_LOCATIONS size_sweep=$RUN_SIZE_SWEEP"
echo "  transcript: $GRID_LOG"

# Flags shared by every run (search tuning + phase-1 settings).
COMMON="--p1-epochs $P1_EPOCHS --n-calib $N_CALIB"
COMMON="$COMMON --p1-grad-clip $P1_GRAD_CLIP --max-flips $MAX_FLIPS"
COMMON="$COMMON --normal-stop $NORMAL_STOP --fine-target $FINE_TARGET"
COMMON="$COMMON --fine-max-flips $FINE_MAX_FLIPS --flip-min-gain $FLIP_MIN_GAIN"
COMMON="$COMMON --fine-patience $FINE_PATIENCE --fine-min-gain $FINE_MIN_GAIN"
COMMON="$COMMON --fine-max-rejects $FINE_MAX_REJECTS"
COMMON="$COMMON --trigger-min-asr $TRIGGER_MIN_ASR --trigger-retries $TRIGGER_RETRIES"

# ════════════════════════════════════════════════════════════════════════
# THE RUN LIST — one string of experiment.py args per run.
# Per-entry flags: --section --model --dataset --seed --variant
#                  --trigger-size [--trigger-location] [--save-attacked-model]
# (--rep, --target-cls, --n-eval, --out-root and $COMMON are added by the loop.)
# ════════════════════════════════════════════════════════════════════════

# ---- 1. VALIDATION (RESULTS.md §1) — full method, t16, seeds 21/42/53/64/77
runs_validation=(
  # ViT-B/16 × Caltech101
  "--section validation --model ViT-B/16 --dataset Caltech101 --seed 21 --variant full --trigger-size 16"
  "--section validation --model ViT-B/16 --dataset Caltech101 --seed 42 --variant full --trigger-size 16"
  "--section validation --model ViT-B/16 --dataset Caltech101 --seed 53 --variant full --trigger-size 16"
  "--section validation --model ViT-B/16 --dataset Caltech101 --seed 64 --variant full --trigger-size 16"
  "--section validation --model ViT-B/16 --dataset Caltech101 --seed 77 --variant full --trigger-size 16"
  # ViT-B/16 × ImageNet1k
  "--section validation --model ViT-B/16 --dataset ImageNet1k --seed 21 --variant full --trigger-size 16"
  "--section validation --model ViT-B/16 --dataset ImageNet1k --seed 42 --variant full --trigger-size 16"
  "--section validation --model ViT-B/16 --dataset ImageNet1k --seed 53 --variant full --trigger-size 16"
  "--section validation --model ViT-B/16 --dataset ImageNet1k --seed 64 --variant full --trigger-size 16"
  "--section validation --model ViT-B/16 --dataset ImageNet1k --seed 77 --variant full --trigger-size 16"
  # ViT-L/14 × Caltech101
  "--section validation --model ViT-L/14 --dataset Caltech101 --seed 21 --variant full --trigger-size 16"
  "--section validation --model ViT-L/14 --dataset Caltech101 --seed 42 --variant full --trigger-size 16"
  "--section validation --model ViT-L/14 --dataset Caltech101 --seed 53 --variant full --trigger-size 16"
  "--section validation --model ViT-L/14 --dataset Caltech101 --seed 64 --variant full --trigger-size 16"
  "--section validation --model ViT-L/14 --dataset Caltech101 --seed 77 --variant full --trigger-size 16"
  # ViT-L/14 × ImageNet1k (last seed saves the attacked model state)
  "--section validation --model ViT-L/14 --dataset ImageNet1k --seed 21 --variant full --trigger-size 16"
  "--section validation --model ViT-L/14 --dataset ImageNet1k --seed 42 --variant full --trigger-size 16"
  "--section validation --model ViT-L/14 --dataset ImageNet1k --seed 53 --variant full --trigger-size 16"
  "--section validation --model ViT-L/14 --dataset ImageNet1k --seed 64 --variant full --trigger-size 16"
  "--section validation --model ViT-L/14 --dataset ImageNet1k --seed 77 --variant full --trigger-size 16 --save-attacked-model"
  # SigLIP-B/16 × Caltech101
  "--section validation --model SigLIP-B/16 --dataset Caltech101 --seed 21 --variant full --trigger-size 16"
  "--section validation --model SigLIP-B/16 --dataset Caltech101 --seed 42 --variant full --trigger-size 16"
  "--section validation --model SigLIP-B/16 --dataset Caltech101 --seed 53 --variant full --trigger-size 16"
  "--section validation --model SigLIP-B/16 --dataset Caltech101 --seed 64 --variant full --trigger-size 16"
  "--section validation --model SigLIP-B/16 --dataset Caltech101 --seed 77 --variant full --trigger-size 16"
  # SigLIP-B/16 × ImageNet1k
  "--section validation --model SigLIP-B/16 --dataset ImageNet1k --seed 21 --variant full --trigger-size 16"
  "--section validation --model SigLIP-B/16 --dataset ImageNet1k --seed 42 --variant full --trigger-size 16"
  "--section validation --model SigLIP-B/16 --dataset ImageNet1k --seed 53 --variant full --trigger-size 16"
  "--section validation --model SigLIP-B/16 --dataset ImageNet1k --seed 64 --variant full --trigger-size 16"
  "--section validation --model SigLIP-B/16 --dataset ImageNet1k --seed 77 --variant full --trigger-size 16"
)

# ---- 2. ABLATION VARIANTS (RESULTS.md §3b–3d) — ViT-L/14 × ImageNet1k, t16
runs_ablation=(
  # variant "full"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 21 --variant full --trigger-size 16"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 42 --variant full --trigger-size 16"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 53 --variant full --trigger-size 16"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 64 --variant full --trigger-size 16"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 77 --variant full --trigger-size 16"
  # variant "a"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 21 --variant a --trigger-size 16"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 42 --variant a --trigger-size 16"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 53 --variant a --trigger-size 16"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 64 --variant a --trigger-size 16"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 77 --variant a --trigger-size 16"
  # variant "b"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 21 --variant b --trigger-size 16"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 42 --variant b --trigger-size 16"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 53 --variant b --trigger-size 16"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 64 --variant b --trigger-size 16"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 77 --variant b --trigger-size 16"
  # variant "c"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 21 --variant c --trigger-size 16"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 42 --variant c --trigger-size 16"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 53 --variant c --trigger-size 16"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 64 --variant c --trigger-size 16"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 77 --variant c --trigger-size 16"
)

# ---- 3. TRIGGER-LOCATION SWEEP — ViT-L/14 × ImageNet1k, s42, t16 (train+eval at the corner)
runs_locations=(
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 42 --variant full --trigger-size 16 --trigger-location top_left"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 42 --variant full --trigger-size 16 --trigger-location top_right"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 42 --variant full --trigger-size 16 --trigger-location bottom_left"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 42 --variant full --trigger-size 16 --trigger-location bottom_right"
)

# ---- 4. TRIGGER-SIZE SWEEP (RESULTS.md §3a) — ViT-L/14 × ImageNet1k, s42
runs_sizes=(
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 42 --variant full --trigger-size 8"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 42 --variant full --trigger-size 12"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 42 --variant full --trigger-size 16"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 42 --variant full --trigger-size 20"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 42 --variant full --trigger-size 24"
  "--section ablation --model ViT-L/14 --dataset ImageNet1k --seed 42 --variant full --trigger-size 32"
)
# ---- 4. OTHER TASKS LIKE I2T and Defence and LP — ViT-L/14 × ImageNet1k, s42

runs_other_tasks=(
"--model ViT-L/14 --dataset ImageNet1k --task i2t --target-cls 291 --seed 42 --variant full --trigger-size 16"
)

# ---- select the enabled blocks ------------------------------------
runs=()
if [ "$RUN_VALIDATION" = "1" ] && [ "${#runs_validation[@]}" -gt 0 ]; then
    runs+=("${runs_validation[@]}")
fi
if [ "$RUN_ABLATION" = "1" ] && [ "${#runs_ablation[@]}" -gt 0 ]; then
    runs+=("${runs_ablation[@]}")
fi
if [ "$RUN_LOCATIONS" = "1" ] && [ "${#runs_locations[@]}" -gt 0 ]; then
    runs+=("${runs_locations[@]}")
fi
if [ "$RUN_SIZE_SWEEP" = "1" ] && [ "${#runs_sizes[@]}" -gt 0 ]; then
    runs+=("${runs_sizes[@]}")
fi

# ---- helpers ------------------------------------------------------
# Parse one entry's long options into R_* globals (for the run dir + metrics).
parse_entry() {
    R_SECTION="" R_MODEL="" R_DATASET="" R_SEED="1" R_REP="1"
    R_VARIANT="" R_SIZE="" R_LOC=""
    local w prev=""
    # shellcheck disable=SC2086  # word splitting of the entry is intentional
    for w in $1; do
        case "$prev" in
            --section)          R_SECTION="$w" ;;
            --model)            R_MODEL="$w" ;;
            --dataset)          R_DATASET="$w" ;;
            --seed)             R_SEED="$w" ;;
            --rep)              R_REP="$w" ;;
            --variant)          R_VARIANT="$w" ;;
            --trigger-size)     R_SIZE="$w" ;;
            --trigger-location) R_LOC="$w" ;;
        esac
        prev="$w"
    done
}

# ---- the plan (everything that will run) --------------------------
total=${#runs[@]}
echo
echo "==================== RUN PLAN ===================="
if [ "$total" -eq 0 ]; then
    echo "  (no runs — check RUN_VALIDATION / RUN_ABLATION / RUN_LOCATIONS / RUN_SIZE_SWEEP)"
else
    for ((i = 0; i < total; i++)); do
        parse_entry "${runs[$i]}"
        tc="$(target_cls_for "$R_DATASET")"
        leaf="zs_c${tc}_s${R_SEED}_r${R_REP}_${R_VARIANT}_t${R_SIZE}"
        [ -n "$R_LOC" ] && leaf="${leaf}_${R_LOC}"
        printf '  [%2d/%d] %s/%s/%s/%s\n' "$((i + 1))" "$total" \
            "$R_SECTION" "${R_MODEL//\//}" "$R_DATASET" "$leaf"
    done
fi
echo "=================================================="

# ---- the sweep ----------------------------------------------------
passed=0
skipped=0
failed=()
idx=0
t0=$SECONDS

for args in "${runs[@]}"; do
    idx=$((idx + 1))
    name=$(printf "run_%02d" "$idx")

    # run dir (mirrors experiment.py) — for the log + the resume check
    parse_entry "$args"
    tc="$(target_cls_for "$R_DATASET")"
    ne="$(n_eval_for "$R_DATASET")"
    leaf="zs_c${tc}_s${R_SEED}_r${R_REP}_${R_VARIANT}_t${R_SIZE}"
    [ -n "$R_LOC" ] && leaf="${leaf}_${R_LOC}"
    dir="$OUT_ROOT_ABS/$R_SECTION/${R_MODEL//\//}/$R_DATASET/$leaf"
    log="$dir/run.log"

    if [ "$RESUME" = "1" ] && [ -f "$dir/bfa_results.json" ]; then
        echo "=== [$idx/$total] $name SKIP (resume: bfa_results.json exists): $dir"
        skipped=$((skipped + 1))
        continue
    fi

    mkdir -p "$dir"
    echo
    echo "=== [$idx/$total] $name: experiment.py $args"
    start=$SECONDS

    # shellcheck disable=SC2086  # word splitting of $args/$COMMON is intentional
    "$PY" experiment.py $args --rep "$R_REP" --target-cls "$tc" --n-eval "$ne" \
        --out-root "$OUT_ROOT_ABS" $COMMON 2>&1 | tee "$log"
    status=${PIPESTATUS[0]}          # python's exit code, not tee's

    dur=$((SECONDS - start))
    if [ "$status" -eq 0 ]; then
        echo "--- $name OK (${dur}s)"
        passed=$((passed + 1))
    else
        echo "--- $name FAILED (exit $status, ${dur}s) — see $log"
        failed+=("$name: $args")
    fi
done

# ═══ AGGREGATE: runs.csv + summary.csv ═══
echo
echo "════════ AGGREGATION ════════"
"$PY" aggregate.py --out-root "$OUT_ROOT_ABS"
agg_status=$?

if [ -f oom_pending.json ]; then
    echo "NOTE: some runs hit CUDA OOM and were deferred → oom_pending.json"
fi
if [ -f failed_runs.json ]; then
    echo "NOTE: some runs failed and were recorded → failed_runs.json (config + traceback)"
fi

# ---- summary -------------------------------------------------------
echo
echo "==================== SUMMARY ===================="
echo "Total: $total | Passed: $passed | Skipped (resume): $skipped | Failed: ${#failed[@]}"
echo "Wall time: $((SECONDS - t0))s   transcript: $GRID_LOG"
if [ "$agg_status" -ne 0 ]; then
    echo "aggregate.py exited with status $agg_status"
fi
if [ "${#failed[@]}" -gt 0 ]; then
    printf '  FAILED %s\n' "${failed[@]}"
    exit 1
fi
exit "$agg_status"