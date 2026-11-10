#!/usr/bin/env bash
# run_all.sh — validation + ablation grids for CLIP-FLIP (RESULTS.md §1, §3).
# ===========================================================================
# One python invocation per (config, repeat); the OOM contract (main.py)
# records OOMed runs in oom_pending.json and exits 0, so the grid continues.
#
# Same-seed repeats: every config runs REPS times with the SAME seed to
# measure run-to-run noise (--rep separates the output dirs).
#
# ASR_TARGET=0 disables the ASR early stop so every run consumes its full
# flip budget — required so ASR@10 / ASR@20 are always defined (the
# Flips→95% metric is tracked regardless).
#
# Re-running the script skips configs whose bfa_results.json already exists
# (RESUME=1, default).
#
# Examples:
#   bash run_all.sh                          # everything, defaults below
#   MODELS="ViT-B/16" DATASETS="Caltech101" RUN_VALIDATION=1 RUN_ABLATION=0 bash run_all.sh
#   PY=/home/mehdi/miniconda3/envs/vlm_attack/bin/python bash run_all.sh

set -u

PY="${PY:-python}"
OUT_ROOT="${OUT_ROOT:-./grid_results}"
SEED="${SEED:-41}"
REPS="${REPS:-5}"
TARGET_CLS="${TARGET_CLS:-3}"
TRIG_SIZE="${TRIG_SIZE:-14}"
MAX_FLIPS="${MAX_FLIPS:-80}"
ASR_TARGET="${ASR_TARGET:-0}"
RESUME="${RESUME:-1}"

# Validation matrix (RESULTS.md §1): 4 models × 3 datasets × REPS repeats
MODELS="${MODELS:-ViT-B/16 ViT-L/14 SigLIP-B/16 SigLIP-so400m/14}"
DATASETS="${DATASETS:-Caltech101 CIFAR10 ImageNet1k}"

RUN_VALIDATION="${RUN_VALIDATION:-1}"
RUN_ABLATION="${RUN_ABLATION:-1}"
RUN_LOCATIONS="${RUN_LOCATIONS:-1}"
RUN_SIZE_SWEEP="${RUN_SIZE_SWEEP:-0}"

# Ablation / sweep config
ABLATION_MODEL="${ABLATION_MODEL:-ViT-B/16}"
ABLATION_DATASET="${ABLATION_DATASET:-Caltech101}"
ABLATION_VARIANTS="${ABLATION_VARIANTS:-full a b c}"
LOCATIONS="${LOCATIONS:-top_left top_right bottom_left bottom_right}"
SIZES="${SIZES:-8 14 20 26}"

run_one() {
    # run_one <model> <dataset> <rep> <out-seg> [extra experiment.py args…]
    local model="$1" dataset="$2" rep="$3" seg="$4"; shift 4
    local tag="${model//\//}"
    local dir="$OUT_ROOT/$tag/$dataset/zs/c$TARGET_CLS/s$SEED/r$rep/$seg"
    if [ "$RESUME" = "1" ] && [ -f "$dir/bfa_results.json" ]; then
        echo "[skip] $dir"
        return 0
    fi
    echo "[run ] $model  $dataset  s$SEED r$rep  ($seg)"
    "$PY" experiment.py --model "$model" --dataset "$dataset" \
        --seed "$SEED" --rep "$rep" --target-cls "$TARGET_CLS" \
        --max-flips "$MAX_FLIPS" --asr-target "$ASR_TARGET" "$@"
}

# ═══ 1. VALIDATION (RESULTS.md §1): full method, 5 repeats per config ═══
if [ "$RUN_VALIDATION" = "1" ]; then
    echo "════════ VALIDATION: $MODELS × $DATASETS × $REPS reps ════════"
    for model in $MODELS; do
        for dataset in $DATASETS; do
            for rep in $(seq 1 "$REPS"); do
                run_one "$model" "$dataset" "$rep" "full__t$TRIG_SIZE" \
                    --variant full --trigger-size "$TRIG_SIZE"
            done
        done
    done
fi

# ═══ 2. ABLATION VARIANTS (RESULTS.md §3b–3d): a/b/c vs full ═══
if [ "$RUN_ABLATION" = "1" ]; then
    echo "════════ ABLATION VARIANTS: $ABLATION_VARIANTS ════════"
    for v in $ABLATION_VARIANTS; do
        for rep in $(seq 1 "$REPS"); do
            run_one "$ABLATION_MODEL" "$ABLATION_DATASET" "$rep" \
                "$v""__t$TRIG_SIZE" \
                --variant "$v" --trigger-size "$TRIG_SIZE"
        done
    done
fi

# ═══ 3. TRIGGER-LOCATION SWEEP: train + eval at the same corner ═══
if [ "$RUN_LOCATIONS" = "1" ]; then
    echo "════════ TRIGGER LOCATIONS: $LOCATIONS ════════"
    for loc in $LOCATIONS; do
        for rep in $(seq 1 "$REPS"); do
            run_one "$ABLATION_MODEL" "$ABLATION_DATASET" "$rep" \
                "full__t$TRIG_SIZE""__$loc" \
                --variant full --trigger-size "$TRIG_SIZE" \
                --trigger-location "$loc"
        done
    done
fi

# ═══ 4. TRIGGER-SIZE SWEEP (RESULTS.md §3a) — opt-in ═══
if [ "$RUN_SIZE_SWEEP" = "1" ]; then
    echo "════════ TRIGGER SIZES: $SIZES ════════"
    for sz in $SIZES; do
        for rep in $(seq 1 "$REPS"); do
            run_one "$ABLATION_MODEL" "$ABLATION_DATASET" "$rep" "full__t$sz" \
                --variant full --trigger-size "$sz"
        done
    done
fi

# ═══ AGGREGATE: runs.csv + summary.csv ═══
echo "════════ AGGREGATION ════════"
"$PY" aggregate.py --out-root "$OUT_ROOT"

if [ -f oom_pending.json ]; then
    echo "NOTE: some runs hit CUDA OOM and were deferred → oom_pending.json"
fi
