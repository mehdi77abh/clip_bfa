#!/usr/bin/env bash
# run_all.sh — validation + ablation grids for CLIP-FLIP (RESULTS.md §1, §3).
# ===========================================================================
# Output is grouped into 3 top-level sections under OUT_ROOT:
#   grid_results/validation/…   §1 validation matrix
#   grid_results/ablation/…     §3 variants + trigger-size / trigger-location sweeps
#   grid_results/baselines/…    §2 baseline runs (random / simple-grad — later)
#
# Grids:
#   validation   : MODELS × DATASETS × SEEDS × REPS repeats (same-seed noise)
#   ablation     : variants on one config, SEEDS × REPS repeats
#   size/location sweeps : ALL MODELS × one dataset, ONE seed, ONE run each
#
# Attacked model state (model_state_attacked.pt, for defence / VLM-transfer
# evaluation) is saved ONLY for ViT-L/14 × ImageNet1k × last seed × last rep
# — gated by experiment.py's --save-attacked-model.
#
# Target hunting: ASR_TARGET=100 stops as soon as ASR reaches it.  In the
# fine regime (ASR >= ASR_FINE_THRESHOLD) the ranking is rebuilt after every
# accepted flip and, every ASR_FINE_PATIENCE accepted flips, the ASR must
# have gained >= ASR_FINE_MIN_GAIN percentage points over the window start —
# otherwise the search gives up.  (ASR_TARGET=0 = run the full MAX_FLIPS
# budget.)
#
# Re-running skips configs whose bfa_results.json exists (RESUME=1).
#
# Examples:
#   bash run_all.sh
#   MODELS="ViT-B/16" DATASETS="Caltech101" RUN_VALIDATION=1 RUN_ABLATION=0 bash run_all.sh
#   SEEDS="41" REPS=5 bash run_all.sh
#   PY=/home/mehdi/miniconda3/envs/vlm_attack/bin/python bash run_all.sh

set -u

PY="${PY:-python}"
OUT_ROOT="${OUT_ROOT:-./grid_results}"
SEEDS="${SEEDS:-41 42 43}"        # validation + ablation: every seed × REPS repeats
REPS="${REPS:-5}"                 # same-seed repeats per config
SWEEP_SEED="${SWEEP_SEED:-41}"    # size/location sweeps: one seed, one run
TARGET_CLS="${TARGET_CLS:-3}"
TRIG_SIZE="${TRIG_SIZE:-14}"
MAX_FLIPS="${MAX_FLIPS:-80}"
ASR_TARGET="${ASR_TARGET:-100}"
ASR_FINE_THRESHOLD="${ASR_FINE_THRESHOLD:-90}"
ASR_FINE_PATIENCE="${ASR_FINE_PATIENCE:-3}"
ASR_FINE_MIN_GAIN="${ASR_FINE_MIN_GAIN:-1.0}"
RESUME="${RESUME:-1}"

# Validation matrix (RESULTS.md §1)
MODELS="${MODELS:-ViT-B/16 ViT-L/14 ViT-L/14-336px SigLIP-B/16 SigLIP-so400m/14}"
DATASETS="${DATASETS:-Caltech101 CIFAR10 ImageNet1k}"

RUN_VALIDATION="${RUN_VALIDATION:-1}"
RUN_ABLATION="${RUN_ABLATION:-1}"
RUN_LOCATIONS="${RUN_LOCATIONS:-1}"
RUN_SIZE_SWEEP="${RUN_SIZE_SWEEP:-0}"

# Ablation / sweep config (variants on ABLATION_MODEL; sweeps: all MODELS × ABLATION_DATASET)
ABLATION_MODEL="${ABLATION_MODEL:-ViT-B/16}"
ABLATION_DATASET="${ABLATION_DATASET:-Caltech101}"
ABLATION_VARIANTS="${ABLATION_VARIANTS:-full a b c}"
LOCATIONS="${LOCATIONS:-top_left top_right bottom_left bottom_right}"
SIZES="${SIZES:-8 14 20 26}"

# Absolute so the resume check and experiment.py agree (bug fix: --out-root
# is now forwarded — OUT_ROOT used to be silently ignored by experiment.py).
OUT_ROOT_ABS="$(realpath -m "$OUT_ROOT")"

run_one() {
    # run_one <section> <model> <dataset> <seed> <rep> <out-seg> [extra args…]
    local section="$1" model="$2" dataset="$3" seed="$4" rep="$5" seg="$6"
    shift 6
    local tag="${model//\//}"
    local dir="$OUT_ROOT_ABS/$section/$tag/$dataset/zs/c$TARGET_CLS/s$seed/r$rep/$seg"
    if [ "$RESUME" = "1" ] && [ -f "$dir/bfa_results.json" ]; then
        echo "[skip] $dir"
        return 0
    fi
    echo "[run ] $section  $model  $dataset  s$seed r$rep  ($seg)"
    "$PY" experiment.py --model "$model" --dataset "$dataset" \
        --seed "$seed" --rep "$rep" --target-cls "$TARGET_CLS" \
        --max-flips "$MAX_FLIPS" --asr-target "$ASR_TARGET" \
        --asr-fine-threshold "$ASR_FINE_THRESHOLD" \
        --asr-fine-patience "$ASR_FINE_PATIENCE" \
        --asr-fine-min-gain "$ASR_FINE_MIN_GAIN" \
        --out-root "$OUT_ROOT_ABS" --section "$section" "$@"
}

last_of() { local x; for x in $1; do :; done; echo "$x"; }
LAST_SEED="$(last_of "$SEEDS")"

# ═══ 1. VALIDATION (RESULTS.md §1) — section: validation ═══
# Full method; MODELS × DATASETS × SEEDS × REPS same-seed repeats.
if [ "$RUN_VALIDATION" = "1" ]; then
    echo "════════ VALIDATION: $MODELS × $DATASETS × seeds [$SEEDS] × $REPS reps ════════"
    for model in $MODELS; do
        for dataset in $DATASETS; do
            for seed in $SEEDS; do
                for rep in $(seq 1 "$REPS"); do
                    extra=""
                    # attacked model state: ViT-L/14 × ImageNet1k × last seed × last rep only
                    if [ "$model" = "ViT-L/14" ] && [ "$dataset" = "ImageNet1k" ] \
                        && [ "$seed" = "$LAST_SEED" ] && [ "$rep" -eq "$REPS" ]; then
                        extra="--save-attacked-model"
                    fi
                    run_one validation "$model" "$dataset" "$seed" "$rep" \
                        "full__t$TRIG_SIZE" \
                        --variant full --trigger-size "$TRIG_SIZE" $extra
                done
            done
        done
    done
fi

# ═══ 2. ABLATION VARIANTS (RESULTS.md §3b–3d) — section: ablation ═══
if [ "$RUN_ABLATION" = "1" ]; then
    echo "════════ ABLATION VARIANTS: $ABLATION_VARIANTS (seeds [$SEEDS] × $REPS reps) ════════"
    for v in $ABLATION_VARIANTS; do
        for seed in $SEEDS; do
            for rep in $(seq 1 "$REPS"); do
                run_one ablation "$ABLATION_MODEL" "$ABLATION_DATASET" "$seed" "$rep" \
                    "$v""__t$TRIG_SIZE" \
                    --variant "$v" --trigger-size "$TRIG_SIZE"
            done
        done
    done
fi

# ═══ 3. TRIGGER-LOCATION SWEEP — all models, one dataset, one seed, one run ═══
# Train + eval at the same corner.
if [ "$RUN_LOCATIONS" = "1" ]; then
    echo "════════ TRIGGER LOCATIONS: $LOCATIONS (all models, $ABLATION_DATASET, s$SWEEP_SEED) ════════"
    for model in $MODELS; do
        for loc in $LOCATIONS; do
            run_one ablation "$model" "$ABLATION_DATASET" "$SWEEP_SEED" 1 \
                "full__t$TRIG_SIZE""__$loc" \
                --variant full --trigger-size "$TRIG_SIZE" \
                --trigger-location "$loc"
        done
    done
fi

# ═══ 4. TRIGGER-SIZE SWEEP (RESULTS.md §3a) — all models, one dataset, one
#    seed, one run — opt-in ═══
if [ "$RUN_SIZE_SWEEP" = "1" ]; then
    echo "════════ TRIGGER SIZES: $SIZES (all models, $ABLATION_DATASET, s$SWEEP_SEED) ════════"
    for model in $MODELS; do
        for sz in $SIZES; do
            run_one ablation "$model" "$ABLATION_DATASET" "$SWEEP_SEED" 1 \
                "full__t$sz" \
                --variant full --trigger-size "$sz"
        done
    done
fi

# ═══ AGGREGATE: runs.csv + summary.csv ═══
echo "════════ AGGREGATION ════════"
"$PY" aggregate.py --out-root "$OUT_ROOT_ABS"

if [ -f oom_pending.json ]; then
    echo "NOTE: some runs hit CUDA OOM and were deferred → oom_pending.json"
fi
if [ -f failed_runs.json ]; then
    echo "NOTE: some runs failed and were recorded → failed_runs.json (config + traceback)"
fi
