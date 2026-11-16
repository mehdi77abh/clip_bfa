#!/usr/bin/env bash
# validation.sh — seed-sensitivity test: ViT-B/16 × Caltech101 ('wild cat'
# c97), ONE run per seed value.
# ===========================================================================
# NOT part of the thesis grids: outputs go under ./test_seeds only and
# --no-aggregate keeps grid_results/runs.csv / summary.csv untouched.
#
# What the seed controls (current pipeline): EVERYTHING — the data split
# (data.py uses cfg.SEED), the Phase-1 trigger init + calib shuffle, and the
# dead-trigger retry seeds.  So each seed here is a fully independent trial.
# Method: plain cos trigger + the previous hybrid ranking (the cos_astar
# hinge and the Phase-3 LSE term were removed 2026-11-16 as inefficient).
#
# Usage:
#   bash validation.sh
#   SEEDS="21 42 53 64 77" bash validation.sh
#   PY=/home/farzin/miniconda3/envs/vlm_bfa/bin/python bash validation.sh
#
# Optional private CSV (stays inside the test dir):
#   python aggregate.py --out-root ./test_seeds

set -u

PY="${PY:-python}"
OUT_ROOT="${OUT_ROOT:-./test_seeds}"
SEEDS="${SEEDS:-21 42 53 64 77}"

# ── fixed config (grid protocol) ──
MODEL="${MODEL:-ViT-B/16}"
DATASET="${DATASET:-Caltech101}"
TARGET_CLS="${TARGET_CLS:-97}"
TRIG_SIZE="${TRIG_SIZE:-16}"
P1_EPOCHS="${P1_EPOCHS:-300}"
N_EVAL="${N_EVAL:-1000}"
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
RESUME="${RESUME:-1}"

OUT_ROOT_ABS="$(realpath -m "$OUT_ROOT")"
TAG="${MODEL//\//}"
SEG="full__t${TRIG_SIZE}"

echo "════════ SEED SWEEP → $OUT_ROOT_ABS (CSVs untouched) ════════"
echo "     $MODEL × $DATASET c$TARGET_CLS  t$TRIG_SIZE  (cos loss)"
echo "     seeds: [$SEEDS]"

for seed in $SEEDS; do
    dir="$OUT_ROOT_ABS/ablation/$TAG/$DATASET/zs/c$TARGET_CLS/s$seed/r1/$SEG"
    if [ "$RESUME" = "1" ] && [ -f "$dir/bfa_results.json" ]; then
        echo "[skip] seed $seed  $dir"
        continue
    fi
    echo "[run ] seed $seed"
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
        --out-root "$OUT_ROOT_ABS" --section ablation --variant full \
        --no-aggregate
done

echo "════════ DONE — results under $OUT_ROOT_ABS ════════"
if [ -d "$OUT_ROOT_ABS/ablation" ]; then
    echo "Optional private CSV (stays inside the test dir):"
    echo "  $PY aggregate.py --out-root \"$OUT_ROOT_ABS\""
fi
