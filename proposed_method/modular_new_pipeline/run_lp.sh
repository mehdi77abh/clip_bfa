#!/usr/bin/env bash
# run_lp.sh — Linear-Probe (LP) task end-to-end (TASKS.md §2).
# ===========================================================================
# Three stages:
#   1. PROBE (once per model×dataset): train the shared linear-probe head on
#      CLEAN frozen CLIP features (linear_probe.py) — balanced
#      TRAIN_PER_CLASS images/class from TRAIN_DATASET (default
#      ImageNet1kTrain = ImageNet-1k train; 50/class → 50 000 images),
#      saved to {LP_HEAD_ROOT}/{model_tag}_{dataset}.pt (registry
#      paths.lp_head_root resolved via config.resolve_path →
#      proposed_method/lp_heads).  Skipped when the checkpoint
#      already exists (RESUME=1).
#   2. ATTACK (once): the full Phase 0→4 attack via experiment.py --task lp;
#      after the bit-flip search the saved head is attached to the attacked
#      encoder's features and evaluated on the eval split →
#      LP-CACC / LP-ASR in bfa_results.json["linear_probe"].
#   3. AGGREGATE: runs.csv + summary.csv (LP columns included).
#
# Defaults: ViT-L/14 × ImageNet1k, seed 42, target 'lion' (c291), trigger 16,
# --variant full, --section baselines (RESULTS.md §2 — the LP task is the
# comparison setting).  The attack protocol is identical to run_all.sh's
# validation grid (two-phase search, same knobs).
#
# Data prerequisite (ImageNet-1k, ONCE):  python ../../data/prepare_imagenet.py
#   builds ../data/imagenet/{train,val} (train subset 200/class by default).
#
# Examples:
#   bash run_lp.sh
#   MODEL="ViT-B/16" DATASET="Caltech101" TRAIN_DATASET="Caltech101" bash run_lp.sh
#   RESUME=0 bash run_lp.sh                # retrain probe + re-run the attack
#   PY=~/miniconda3/envs/vlm_attack/bin/python bash run_lp.sh

set -u

PY="${PY:-python}"
OUT_ROOT="${OUT_ROOT:-./grid_results}"
# Head dir: registry paths.lp_head_root resolved like the code does
# (config.resolve_path → proposed_method/lp_heads); override via LP_HEAD_ROOT.
LP_HEAD_ROOT="${LP_HEAD_ROOT:-$("$PY" -c 'from config import load_registry, resolve_path; print(resolve_path(load_registry()["paths"]["lp_head_root"]))')}"

# ── what to run ──
MODEL="${MODEL:-ViT-L/14}"
DATASET="${DATASET:-ImageNet1k}"
TRAIN_DATASET="${TRAIN_DATASET:-ImageNet1kTrain}"
TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-50}"   # 50 × 1000 classes = 50 000 train
PROBE_SEED="${PROBE_SEED:-42}"
PROBE_EPOCHS="${PROBE_EPOCHS:-60}"         # Phase-5e values (linear_probe.py)
PROBE_LR="${PROBE_LR:-1e-3}"
PROBE_WD="${PROBE_WD:-1e-4}"

SECTION="${SECTION:-baselines}"
SEED="${SEED:-42}"
REP="${REP:-1}"
VARIANT="${VARIANT:-full}"
TRIGGER_LOCATION="${TRIGGER_LOCATION:-}"   # empty = legacy mapping
RESUME="${RESUME:-1}"

# ── attack protocol (identical to run_all.sh's validation grid) ──
P1_EPOCHS="${P1_EPOCHS:-300}"
N_CALIB="${N_CALIB:-64}"
N_EVAL="${N_EVAL:-1000}"
P1_GRAD_CLIP="${P1_GRAD_CLIP:-1.0}"
TRIGGER_MIN_ASR="${TRIGGER_MIN_ASR:-5.0}"
TRIGGER_RETRIES="${TRIGGER_RETRIES:-3}"
MAX_FLIPS="${MAX_FLIPS:-80}"
NORMAL_STOP="${NORMAL_STOP:-94}"
FINE_TARGET="${FINE_TARGET:-99}"
FINE_MAX_FLIPS="${FINE_MAX_FLIPS:-9}"
FLIP_MIN_GAIN="${FLIP_MIN_GAIN:-0.5}"
FINE_PATIENCE="${FINE_PATIENCE:-3}"
FINE_MIN_GAIN="${FINE_MIN_GAIN:-2.0}"
FINE_MAX_REJECTS="${FINE_MAX_REJECTS:-20}"

# Per-dataset target class (run_all.sh convention): ImageNet1k 'lion' (c291),
# Caltech101 'wild cat' (c97), CIFAR10 'cat' (c3).
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

# Trigger-size policy (run_all.sh): 224px models → 16 on Caltech/ImageNet,
# 8 on CIFAR; 336/384px models (ViT-L/14-336px, SigLIP-so400m/14) → 24.
TRIG_STD="${TRIG_STD:-16}"
TRIG_CIFAR="${TRIG_CIFAR:-8}"
TRIG_HIRES="${TRIG_HIRES:-24}"

trig_size_for() {
    case "$1" in
        ViT-L/14-336px|SigLIP-so400m/14) echo "$TRIG_HIRES" ;;
        *) case "$2" in
               CIFAR10|CIFAR100) echo "$TRIG_CIFAR" ;;
               *)                echo "$TRIG_STD" ;;
           esac ;;
    esac
}

# ── identity (mirrors experiment.py: {tag}_{dataset}.pt and the flattened
#    run leaf lp_c{cls}_s{seed}_r{rep}_{variant}_t{sz}[_{loc}]) ──
TAG="${MODEL//\//}"
TC="$(target_cls_for "$DATASET")"
TSZ="$(trig_size_for "$MODEL" "$DATASET")"
LP_HEAD="$LP_HEAD_ROOT/${TAG}_${DATASET}.pt"
SEG="${VARIANT}_t${TSZ}"
[ -n "$TRIGGER_LOCATION" ] && SEG="${SEG}_${TRIGGER_LOCATION}"
LEAF="lp_c${TC}_s${SEED}_r${REP}_${SEG}"
OUT_ROOT_ABS="$(realpath -m "$OUT_ROOT")"
RUN_DIR="$OUT_ROOT_ABS/$SECTION/$TAG/$DATASET/$LEAF"

LOG_DIR="$OUT_ROOT_ABS/logs"
mkdir -p "$LOG_DIR"
GRID_LOG="$LOG_DIR/lp_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$GRID_LOG") 2>&1
echo "LP task start $(date '+%F %T')  PY=$PY"
echo "  MODEL=$MODEL  DATASET=$DATASET  target=c$TC  trigger=t$TSZ  seed=$SEED"
echo "  probe: $TRAIN_DATASET ($TRAIN_PER_CLASS/class) → $LP_HEAD"
echo "  run  : $RUN_DIR"

# ═══ 0. DATA CHECK (ImageNet must be prepared once: prepare_imagenet.py) ═══
case "$TRAIN_DATASET" in
    ImageNet1kTrain)
        if [ ! -d "../data/imagenet/train" ] || [ -z "$(ls -A ../data/imagenet/train 2>/dev/null)" ]; then
            echo "ERROR: ImageNet-1k train split missing (../data/imagenet/train)."
            echo "       Prepare the data once:  python ../../data/prepare_imagenet.py"
            exit 1
        fi ;;
esac
case "$DATASET" in
    ImageNet1k)
        if [ ! -d "../data/imagenet/val" ] || [ -z "$(ls -A ../data/imagenet/val 2>/dev/null)" ]; then
            echo "ERROR: ImageNet-1k val split missing (../data/imagenet/val)."
            echo "       Prepare the data once:  python ../../data/prepare_imagenet.py"
            exit 1
        fi ;;
esac

# ═══ 1. PROBE TRAINING (once per model×dataset; skipped when checkpoint
#    exists and RESUME=1) ═══
if [ "$RESUME" = "1" ] && [ -f "$LP_HEAD" ]; then
    echo "[skip] probe head exists → $LP_HEAD"
else
    echo "[probe] training shared linear-probe head ($MODEL, $TRAIN_PER_CLASS/class)"
    mkdir -p "$LP_HEAD_ROOT"
    "$PY" linear_probe.py \
        --model "$MODEL" \
        --train-dataset "$TRAIN_DATASET" --test-dataset "$DATASET" \
        --train-per-class "$TRAIN_PER_CLASS" --seed "$PROBE_SEED" \
        --epochs "$PROBE_EPOCHS" --lr "$PROBE_LR" --wd "$PROBE_WD" \
        --lp-head-root "$LP_HEAD_ROOT" \
        --out "$LP_HEAD" 2>&1 | tee "$LP_HEAD_ROOT/${TAG}_${DATASET}_train.log"
    if [ ! -f "$LP_HEAD" ]; then
        echo "ERROR: probe training did not produce $LP_HEAD (see $LP_HEAD_ROOT/${TAG}_${DATASET}_train.log)"
        exit 1
    fi
fi

# ═══ 2. ATTACK + LP EVALUATION (one run) ═══
if [ "$RESUME" = "1" ] && [ -f "$RUN_DIR/bfa_results.json" ]; then
    echo "[skip] run complete → $RUN_DIR"
else
    echo "[run ] attack + LP evaluation → $RUN_DIR"
    mkdir -p "$RUN_DIR"
    "$PY" experiment.py --model "$MODEL" --dataset "$DATASET" \
        --task lp --lp-head "$LP_HEAD" \
        --seed "$SEED" --rep "$REP" --target-cls "$TC" \
        --variant "$VARIANT" --trigger-size "$TSZ" \
        ${TRIGGER_LOCATION:+--trigger-location "$TRIGGER_LOCATION"} \
        --n-eval "$N_EVAL" --p1-epochs "$P1_EPOCHS" \
        --n-calib "$N_CALIB" \
        --p1-grad-clip "$P1_GRAD_CLIP" \
        --max-flips "$MAX_FLIPS" \
        --normal-stop "$NORMAL_STOP" --fine-target "$FINE_TARGET" \
        --fine-max-flips "$FINE_MAX_FLIPS" --flip-min-gain "$FLIP_MIN_GAIN" \
        --fine-patience "$FINE_PATIENCE" --fine-min-gain "$FINE_MIN_GAIN" \
        --fine-max-rejects "$FINE_MAX_REJECTS" \
        --trigger-min-asr "$TRIGGER_MIN_ASR" \
        --trigger-retries "$TRIGGER_RETRIES" \
        --out-root "$OUT_ROOT_ABS" --section "$SECTION" 2>&1 \
        | tee "$RUN_DIR/run.log"
fi

# ═══ 3. AGGREGATE: runs.csv + summary.csv (LP columns included) ═══
echo "════════ AGGREGATION ════════"
"$PY" aggregate.py --out-root "$OUT_ROOT_ABS"

if [ -f oom_pending.json ]; then
    echo "NOTE: the run hit CUDA OOM and was deferred → oom_pending.json"
fi
if [ -f failed_runs.json ]; then
    echo "NOTE: the run failed and was recorded → failed_runs.json (config + traceback)"
fi
echo "LP task done $(date '+%F %T')"
