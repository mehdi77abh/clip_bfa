#!/usr/bin/env bash
# run_ablation.sh — ViT-L/14 × ImageNet1k ablation leftovers, in two separate
# stages (possibly on different GPUs / machines):
#
#   bash run_ablation.sh sweeps   # trigger-location + trigger-size sweeps
#                                 # (variant full, seed 42, one run per point)
#   bash run_ablation.sh a        # variant 'a' (skip Phase 2) over the 5 seeds
#                                 # — needs MORE GPU RAM than the other
#                                 #   variants (no Phase-2 pruning → bigger
#                                 #   candidate pools)
#   bash run_ablation.sh all      # sweeps first, then a
#
# Outputs go to ./grid_results in the SAME layout as run_all.sh, so the
# auto-refreshed runs.csv / summary.csv pick the rows up; logs land under
# grid_results/logs/ + per-run run.log.  RESUME=1 skips finished runs
# (failed/empty dirs are re-run automatically).
#
# Usage:
#   PY=/home/farzin/miniconda3/envs/vlm_bfa/bin/python bash run_ablation.sh sweeps
#   PY=… bash run_ablation.sh a          # later, on the bigger GPU

set -u

MODE="${1:-sweeps}"
PY="${PY:-python}"
OUT_ROOT="${OUT_ROOT:-./grid_results}"
RESUME="${RESUME:-1}"

# ── ablation config (fixed) ──
MODEL="ViT-L/14"
DATASET="ImageNet1k"
TARGET_CLS=291
TRIG_SIZE=16
SEEDS="${SEEDS:-21 42 53 64 77}"
SWEEP_SEED="${SWEEP_SEED:-42}"
LOCATIONS="${LOCATIONS:-top_left top_right bottom_left bottom_right}"
SIZES="${SIZES:-8 12 16 20 24 32}"

# ── protocol (same as run_all.sh) ──
N_EVAL="${N_EVAL:-1000}"
P1_EPOCHS="${P1_EPOCHS:-300}"
N_CALIB="${N_CALIB:-64}"
P1_GRAD_CLIP="${P1_GRAD_CLIP:-1.0}"
MAX_FLIPS="${MAX_FLIPS:-80}"
NORMAL_STOP="${NORMAL_STOP:-94}"
FINE_TARGET="${FINE_TARGET:-99}"
FINE_MAX_FLIPS="${FINE_MAX_FLIPS:-9}"
FLIP_MIN_GAIN="${FLIP_MIN_GAIN:-0.5}"
FINE_PATIENCE="${FINE_PATIENCE:-3}"
FINE_MIN_GAIN="${FINE_MIN_GAIN:-2.0}"
FINE_MAX_REJECTS="${FINE_MAX_REJECTS:-20}"
TRIGGER_MIN_ASR="${TRIGGER_MIN_ASR:-5.0}"
TRIGGER_RETRIES="${TRIGGER_RETRIES:-3}"

OUT_ROOT_ABS="$(realpath -m "$OUT_ROOT")"
TAG="${MODEL//\//}"

# ── running logs: one transcript per invocation + per-run run.log ──
LOG_DIR="$OUT_ROOT_ABS/logs"
mkdir -p "$LOG_DIR"
GRID_LOG="$LOG_DIR/ablation_${MODE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$GRID_LOG") 2>&1

# run_ab <seg> <seed> [extra args…]
run_ab() {
    local seg="$1" seed="$2"
    shift 2
    # flattened run dir (mirrors experiment.py)
    local leaf="zs_c${TARGET_CLS}_s${seed}_r1_${seg//__/_}"
    local dir="$OUT_ROOT_ABS/ablation/$TAG/$DATASET/$leaf"
    if [ "$RESUME" = "1" ] && [ -f "$dir/bfa_results.json" ]; then
        echo "[skip] $seg  s$seed"
        return 0
    fi
    echo "[run ] $seg  s$seed"
    mkdir -p "$dir"
    "$PY" experiment.py --model "$MODEL" --dataset "$DATASET" \
        --seed "$seed" --rep 1 --target-cls "$TARGET_CLS" \
        --trigger-size "$TRIG_SIZE" --n-eval "$N_EVAL" \
        --p1-epochs "$P1_EPOCHS" --n-calib "$N_CALIB" \
        --p1-grad-clip "$P1_GRAD_CLIP" \
        --max-flips "$MAX_FLIPS" \
        --normal-stop "$NORMAL_STOP" --fine-target "$FINE_TARGET" \
        --fine-max-flips "$FINE_MAX_FLIPS" --flip-min-gain "$FLIP_MIN_GAIN" \
        --fine-patience "$FINE_PATIENCE" --fine-min-gain "$FINE_MIN_GAIN" \
        --fine-max-rejects "$FINE_MAX_REJECTS" \
        --trigger-min-asr "$TRIGGER_MIN_ASR" \
        --trigger-retries "$TRIGGER_RETRIES" \
        --out-root "$OUT_ROOT_ABS" --section ablation "$@" 2>&1 \
        | tee "$dir/run.log"
}

run_sweeps() {
    echo "════ TRIGGER-LOCATION SWEEP: $LOCATIONS (s$SWEEP_SEED) ════"
    for loc in $LOCATIONS; do
        run_ab "full__t${TRIG_SIZE}__$loc" "$SWEEP_SEED" \
            --variant full --trigger-location "$loc"
    done
    echo "════ TRIGGER-SIZE SWEEP: $SIZES (s$SWEEP_SEED) ════"
    for sz in $SIZES; do
        run_ab "full__t$sz" "$SWEEP_SEED" \
            --variant full --trigger-size "$sz"
    done
}

run_variant_a() {
    echo "════ VARIANT a (skip Phase 2): seeds [$SEEDS] ════"
    for seed in $SEEDS; do
        run_ab "a__t${TRIG_SIZE}" "$seed" --variant a
    done
}

case "$MODE" in
    sweeps) run_sweeps ;;
    a)      run_variant_a ;;
    all)    run_sweeps; run_variant_a ;;
    *)      echo "usage: bash run_ablation.sh {sweeps|a|all}"; exit 1 ;;
esac

echo "════ DONE ($MODE) ════"
if [ -f "$OUT_ROOT_ABS/failed_runs.json" ]; then
    echo "NOTE: failures recorded → $OUT_ROOT_ABS/failed_runs.json"
fi
