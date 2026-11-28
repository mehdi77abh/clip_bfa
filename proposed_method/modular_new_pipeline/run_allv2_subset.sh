#!/usr/bin/env bash
# run_allv2_subset.sh — subset of run_allv2.sh: ViT-B/16 + SigLIP-B/16 (all
# runs) and ViT-L/14 × Caltech101 only.
# ===========================================================================
# Derived from run_allv2.sh (same plan → run → resume → aggregate mechanics),
# but the run list is fixed to 25 validation runs:
#
#   ViT-B/16     × Caltech101   seeds 21 42 53 64 77
#   ViT-B/16     × ImageNet1k   seeds 21 42 53 64 77
#   SigLIP-B/16  × Caltech101   seeds 21 42 53 64 77
#   SigLIP-B/16  × ImageNet1k   seeds 21 42 53 64 77
#   ViT-L/14     × Caltech101   seeds 21 42 53 64 77
#
# Everything else from run_allv2.sh is intentionally EXCLUDED:
#   ViT-L/14 × ImageNet1k validation, ablation variants, trigger-location and
#   trigger-size sweeps, the 336px model and the I2T / LP / FaR other tasks.
#
# Trigger size is 16 (224px policy: 16 on Caltech/ImageNet).
#
# Re-running skips configs whose bfa_results.json exists (RESUME=1); skipped
# runs are counted separately in the summary.
#
# Logs: one transcript per invocation (logs/grid_*.log) + run.log inside
# every run dir (the log named in the summary).
#
# Examples:
#   bash run_allv2_subset.sh
#   RESUME=0 bash run_allv2_subset.sh
#   PY=/home/mehdi/miniconda3/envs/vlm_attack/bin/python bash run_allv2_subset.sh

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

# Two-phase search (search.py) — identical to run_allv2.sh.
MAX_FLIPS="${MAX_FLIPS:-80}"
NORMAL_STOP="${NORMAL_STOP:-94}"
FINE_TARGET="${FINE_TARGET:-99}"
FINE_MAX_FLIPS="${FINE_MAX_FLIPS:-9}"
FLIP_MIN_GAIN="${FLIP_MIN_GAIN:-0.5}"
FINE_PATIENCE="${FINE_PATIENCE:-3}"
FINE_MIN_GAIN="${FINE_MIN_GAIN:-2.0}"
FINE_MAX_REJECTS="${FINE_MAX_REJECTS:-20}"

P1_GRAD_CLIP="${P1_GRAD_CLIP:-1.0}"
TRIGGER_MIN_ASR="${TRIGGER_MIN_ASR:-5.0}"
TRIGGER_RETRIES="${TRIGGER_RETRIES:-3}"
RESUME="${RESUME:-1}"

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

OUT_ROOT_ABS="$(realpath -m "$OUT_ROOT")"

# ── running logs: one transcript per invocation + run.log inside every run dir ──
LOG_DIR="$OUT_ROOT_ABS/logs"
mkdir -p "$LOG_DIR"
GRID_LOG="$LOG_DIR/grid_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$GRID_LOG") 2>&1
echo "grid start $(date '+%F %T')  PY=$PY"
echo "  subset: ViT-B/16 + SigLIP-B/16 (all) + ViT-L/14 × Caltech101"
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
# THE RUN LIST — one string of experiment.py args per run (25 total).
# (--rep, --target-cls, --n-eval, --out-root and $COMMON are added by the loop.)
# ════════════════════════════════════════════════════════════════════════
runs=(
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
  # ViT-L/14 × Caltech101
  "--section validation --model ViT-L/14 --dataset Caltech101 --seed 21 --variant full --trigger-size 16"
  "--section validation --model ViT-L/14 --dataset Caltech101 --seed 42 --variant full --trigger-size 16"
  "--section validation --model ViT-L/14 --dataset Caltech101 --seed 53 --variant full --trigger-size 16"
  "--section validation --model ViT-L/14 --dataset Caltech101 --seed 64 --variant full --trigger-size 16"
  "--section validation --model ViT-L/14 --dataset Caltech101 --seed 77 --variant full --trigger-size 16"
)

# ---- helpers ------------------------------------------------------
# Parse one entry's long options into R_* globals (for the run dir + metrics).
parse_entry() {
    R_SECTION="" R_MODEL="" R_DATASET="" R_SEED="1" R_REP="1"
    R_VARIANT="" R_SIZE="" R_LOC="" R_TASK="zs"
    local w prev=""
    # shellcheck disable=SC2086  # word splitting of the entry is intentional
    for w in $1; do
        case "$prev" in
            --section)          R_SECTION="$w" ;;
            --model)            R_MODEL="$w" ;;
            --dataset)          R_DATASET="$w" ;;
            --task)             R_TASK="$w" ;;
            --seed)             R_SEED="$w" ;;
            --rep)              R_REP="$w" ;;
            --variant)          R_VARIANT="$w" ;;
            --trigger-size)     R_SIZE="$w" ;;
            --trigger-location) R_LOC="$w" ;;
        esac
        prev="$w"
    done
}

# Run leaf — must mirror experiment.py exactly:
#   {task}_c{cls}_s{seed}_r{rep}_{variant}_t{size}[_{loc}]
run_leaf() {
    local leaf="${R_TASK}_c${1}_s${R_SEED}_r${R_REP}_${R_VARIANT}_t${R_SIZE}"
    [ -n "$R_LOC" ] && leaf="${leaf}_${R_LOC}"
    echo "$leaf"
}

# ---- the plan (everything that will run) --------------------------
total=${#runs[@]}
echo
echo "==================== RUN PLAN ===================="
if [ "$total" -eq 0 ]; then
    echo "  (no runs — the run list is empty)"
else
    for ((i = 0; i < total; i++)); do
        parse_entry "${runs[$i]}"
        tc="$(target_cls_for "$R_DATASET")"
        leaf="$(run_leaf "$tc")"
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
    leaf="$(run_leaf "$tc")"
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
