#!/usr/bin/env bash
# test_validation.sh — diagnostic runs for the 2026-11-14 Arm-C port
# (Phase-1 trigger loss + Phase-3 LSE term).
# ===========================================================================
# NOT part of the thesis grids: every output goes under ./test_validation
# only, and --no-aggregate keeps runs.csv / summary.csv (grid_results) out
# of it entirely.  One seed (42), one round per config — the pipeline is
# deterministic per seed, so repeats measure nothing.
#
# Runs:
#   A. Port validation at the Arm-C file's OWN settings (main_arm_c_cos_astar.py:
#      Caltech cls 3 = 'motorbike', t14, P1 200 epochs, N_EVAL 200) — on
#      ViT-L/14 per request (the Arm-C testbed itself was ViT-B/16).  If these
#      reproduce the tested claim, the port is faithful and the grid gap is
#      due to the harder targets/settings.
#      A1  cos_astar + LSE 0.3   (full tested arm)
#      A2  cos_astar + LSE 0     (hinge only)
#   B. Loss × LSE attribution on the HARD target (ViT-L/14 × ImageNet1k
#      'lion' c291, t16, P1 300 epochs, N_EVAL 1000):
#      B1  cos       + LSE 0     (plain loss, previous ranking metric)
#      B2  cos       + LSE 0.3   (LSE only)
#      B3  cos_astar + LSE 0     (hinge only)
#      B4  cos_astar + LSE 0.3   (full new arm = current grid default)
#
# Usage:
#   bash test_validation.sh
#   PY=/home/farzin/miniconda3/envs/vlm_bfa/bin/python bash test_validation.sh
#
# Optional private CSV (writes INSIDE ./test_validation, never grid_results):
#   python aggregate.py --out-root ./test_validation

set -u

PY="${PY:-python}"
OUT_ROOT="${OUT_ROOT:-./test_validation}"
SEED="${SEED:-42}"
RESUME="${RESUME:-1}"

# ── shared protocol (grid defaults) ──
N_CALIB="${N_CALIB:-64}"
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
ASTAR_K="${ASTAR_K:-8}"
ASTAR_LAMBDA="${ASTAR_LAMBDA:-1.0}"
ASTAR_MU="${ASTAR_MU:-0.05}"
ASTAR_GRAD_CLIP="${ASTAR_GRAD_CLIP:-1.0}"
P3_LSE_BETA="${P3_LSE_BETA:-10.0}"

OUT_ROOT_ABS="$(realpath -m "$OUT_ROOT")"

# run_test <name> <model> <dataset> <target-cls> <trig-size> <p1-epochs> \
#          <n-eval> <p1-loss> <p3-alpha-lse>
run_test() {
    local name="$1" model="$2" dataset="$3" tc="$4" tsz="$5"
    local p1ep="$6" ne="$7" p1loss="$8" alse="$9"
    local tag="${model//\//}"
    local seg="full__t${tsz}__p1${p1loss}__lse${alse//./p}"
    local dir="$OUT_ROOT_ABS/ablation/$tag/$dataset/zs/c$tc/s$SEED/r1/$seg"
    if [ "$RESUME" = "1" ] && [ -f "$dir/bfa_results.json" ]; then
        echo "[skip] $name  $dir"
        return 0
    fi
    echo "[run ] $name  $model × $dataset  c$tc  t$tsz  p1=$p1ep  n_eval=$ne  $p1loss  lse$alse"
    "$PY" experiment.py --model "$model" --dataset "$dataset" \
        --seed "$SEED" --rep 1 --target-cls "$tc" \
        --trigger-size "$tsz" --n-eval "$ne" --p1-epochs "$p1ep" \
        --n-calib "$N_CALIB" \
        --p1-loss "$p1loss" --astar-k "$ASTAR_K" \
        --astar-lambda "$ASTAR_LAMBDA" --astar-mu "$ASTAR_MU" \
        --astar-grad-clip "$ASTAR_GRAD_CLIP" \
        --p3-alpha-lse "$alse" --p3-lse-beta "$P3_LSE_BETA" \
        --max-flips "$MAX_FLIPS" \
        --normal-stop "$NORMAL_STOP" --fine-target "$FINE_TARGET" \
        --fine-max-flips "$FINE_MAX_FLIPS" --flip-min-gain "$FLIP_MIN_GAIN" \
        --fine-patience "$FINE_PATIENCE" --fine-min-gain "$FINE_MIN_GAIN" \
        --fine-max-rejects "$FINE_MAX_REJECTS" \
        --trigger-min-asr "$TRIGGER_MIN_ASR" \
        --trigger-retries "$TRIGGER_RETRIES" \
        --out-root "$OUT_ROOT_ABS" --section ablation --variant full \
        --no-aggregate
}

echo "════════ TEST VALIDATION → $OUT_ROOT_ABS (CSVs untouched) ════════"

# ── A. Port validation at the Arm-C file's settings (motorbike, t14/200/200)
run_test "A1 port full-arm"    ViT-L/14 Caltech101 3   14 200 200 cos_astar 0.3
run_test "A2 port hinge-only"  ViT-L/14 Caltech101 3   14 200 200 cos_astar 0

# ── B. Loss × LSE attribution on ImageNet 'lion' (t16/300/1000)
run_test "B1 cos lse0"         ViT-L/14 ImageNet1k 291 16 300 1000 cos       0
run_test "B2 cos lse0.3"       ViT-L/14 ImageNet1k 291 16 300 1000 cos       0.3
run_test "B3 astar lse0"       ViT-L/14 ImageNet1k 291 16 300 1000 cos_astar 0
run_test "B4 astar lse0.3"     ViT-L/14 ImageNet1k 291 16 300 1000 cos_astar 0.3

echo "════════ DONE — results under $OUT_ROOT_ABS ════════"
if [ -d "$OUT_ROOT_ABS/ablation" ]; then
    echo "Optional private CSV (stays inside the test dir):"
    echo "  $PY aggregate.py --out-root \"$OUT_ROOT_ABS\""
fi
