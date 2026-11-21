#!/usr/bin/env bash
# run_all_baselines.sh — global baseline grid (Flip-S · ProFlip-CBS · TrojViT Phase 1)
# ===========================================================================
# Results are kept SEPARATE from the proposed method:
#   bfa_baselines/baseline_runs/       per-run artefacts
#   bfa_baselines/baseline_summary/    runs.csv + summary.csv (+ logs/)
#
# What runs:
#   ProFlip-CBS : Phase 0 → Phase-1 trigger (proposed method) → CBS
#                 no Phase 2, target-cosine, paper values ASR_t=94 / n_max=100
#   TrojViT     : Phase 1 trigger generation ONLY (no NGR / insertion)
#   Flip-S      : full Flip-S ZS adapter on OpenAI CLIP models × mapped
#                 datasets (Caltech101 / ImageNet1k / DTD); SigLIP needs a
#                 separate flip_s_siglip adapter (not implemented)
#
# Matrix follows the proposed run_all.sh: MODELS × DATASETS × SEEDS × REPS,
# same target classes, n_eval and trigger-size policy.  Re-running skips
# runs whose result JSON already exists (RESUME=1).
#
# Examples:
#   bash run_all_baselines.sh
#   MODELS="ViT-B/16" DATASETS="Caltech101" SEEDS="42" bash run_all_baselines.sh
#   RUN_FLIP_S=0 RUN_TROJVIT=0 bash run_all_baselines.sh
#   PY=/home/mehdi/miniconda3/envs/vlm_attack/bin/python bash run_all_baselines.sh

set -u
set -o pipefail

BFA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIP_BFA_DIR="$(realpath -m "$BFA_DIR/../../..")"
cd "$BFA_DIR"

if [ -z "${PY:-}" ]; then
    if [ -x "$HOME/miniconda3/envs/vlm_attack/bin/python" ]; then
        PY="$HOME/miniconda3/envs/vlm_attack/bin/python"
    else
        PY="python"
    fi
fi
RUNS_ROOT="${RUNS_ROOT:-$BFA_DIR/baseline_runs}"
SUMMARY_ROOT="${SUMMARY_ROOT:-$BFA_DIR/baseline_summary}"

MODELS="${MODELS:-ViT-B/16 ViT-L/14 ViT-L/14-336px}"
DATASETS="${DATASETS:-Caltech101 ImageNet1k DTD}"
SEEDS="${SEEDS:-42}"
REPS="${REPS:-1}"
RESUME="${RESUME:-1}"

RUN_PROFLIP="${RUN_PROFLIP:-1}"
RUN_TROJVIT="${RUN_TROJVIT:-1}"
RUN_FLIP_S="${RUN_FLIP_S:-1}"

# Phase-1 trigger settings (mirror the proposed method's run_all.sh)
P1_EPOCHS="${P1_EPOCHS:-300}"
N_CALIB="${N_CALIB:-64}"
P1_GRAD_CLIP="${P1_GRAD_CLIP:-1.0}"

# CBS (ProFlip paper values)
CBS_K="${CBS_K:-20}"
CBS_ASR_TARGET="${CBS_ASR_TARGET:-94}"
CBS_NMAX="${CBS_NMAX:-100}"
CBS_CLEAN_MSE="${CBS_CLEAN_MSE:-0.0}"

# TrojViT Phase 1 (paper/release values)
TROJVIT_IMAGES="${TROJVIT_IMAGES:-384}"
TROJVIT_BATCH="${TROJVIT_BATCH:-16}"
TROJVIT_ITERS="${TROJVIT_ITERS:-250}"
TROJVIT_LR="${TROJVIT_LR:-0.22}"
TROJVIT_NUM_PATCH="${TROJVIT_NUM_PATCH:-9}"
TROJVIT_PATCH_SIZE="${TROJVIT_PATCH_SIZE:-0}"   # 0 = auto (victim patch size)
TROJVIT_PATCH_SELECT="${TROJVIT_PATCH_SELECT:-saliency}"
TROJVIT_ATTEN_WEIGHT="${TROJVIT_ATTEN_WEIGHT:-1.0}"
TROJVIT_ATTEN_LAYER="${TROJVIT_ATTEN_LAYER:-4}"
TROJVIT_LOSS="${TROJVIT_LOSS:-cos}"
TROJVIT_MODE="${TROJVIT_MODE:-replace}"
TROJVIT_INIT="${TROJVIT_INIT:-random}"
TROJVIT_FIXED_MASKS="${TROJVIT_FIXED_MASKS:-0}"

# Flip-S (full ZS adapter).  Dataset roots/class lists are mapped per dataset;
# Caltech101 drops BACKGROUND_Google to match the proposed method's 101-class
# torchvision Caltech101 label space.
FLIP_S_SEED="${FLIP_S_SEED:-0}"
FLIP_S_VALDIR_IMAGENET="${FLIP_S_VALDIR_IMAGENET:-${FLIP_S_VALDIR:-$CLIP_BFA_DIR/data/imagenet/val}}"
FLIP_S_VALDIR_CALTECH="${FLIP_S_VALDIR_CALTECH:-$CLIP_BFA_DIR/data/caltech-101/caltech-101/101_ObjectCategories}"
FLIP_S_VALDIR_DTD="${FLIP_S_VALDIR_DTD:-$CLIP_BFA_DIR/data/dtd/images}"
FLIP_S_CLASSNAMES_IMAGENET="${FLIP_S_CLASSNAMES_IMAGENET:-$CLIP_BFA_DIR/proposed_method/modular_new_pipeline/class_texts/imagenet1k/classes.txt}"
FLIP_S_CLASSNAMES_CALTECH="${FLIP_S_CLASSNAMES_CALTECH:-$CLIP_BFA_DIR/proposed_method/modular_new_pipeline/class_texts/caltech101/classes.txt}"
FLIP_S_CLASSNAMES_DTD="${FLIP_S_CLASSNAMES_DTD:-$CLIP_BFA_DIR/proposed_method/modular_new_pipeline/class_texts/dtd/classes.txt}"
FLIP_S_DIR="$BFA_DIR/flip_s_clip"

flip_s_valdir_for() {
    case "$1" in
        ImageNet1k|ImageNet1kTrain) echo "$FLIP_S_VALDIR_IMAGENET" ;;
        Caltech101)                 echo "$FLIP_S_VALDIR_CALTECH" ;;
        DTD)                        echo "$FLIP_S_VALDIR_DTD" ;;
        *)                          echo "" ;;
    esac
}
flip_s_classnames_for() {
    case "$1" in
        ImageNet1k|ImageNet1kTrain) echo "$FLIP_S_CLASSNAMES_IMAGENET" ;;
        Caltech101)                 echo "$FLIP_S_CLASSNAMES_CALTECH" ;;
        DTD)                        echo "$FLIP_S_CLASSNAMES_DTD" ;;
        *)                          echo "" ;;
    esac
}
flip_s_exclude_for() {
    case "$1" in
        Caltech101) echo "BACKGROUND_Google" ;;
        *)          echo "" ;;
    esac
}
flip_s_patch_for() {
    case "$1" in
        ViT-B/32)                  echo 32 ;;
        ViT-L/14|ViT-L/14-336px)   echo 14 ;;
        SigLIP-so400m/14)          echo 14 ;;
        *)                         echo 16 ;;
    esac
}
flip_s_img_for() {
    case "$1" in
        ViT-L/14-336px|SigLIP-so400m/14) echo 336 ;;
        *)                               echo 224 ;;
    esac
}

# Target classes (same mapping as the proposed run_all.sh)
TARGET_CLS_IMAGENET="${TARGET_CLS_IMAGENET:-291}"
TARGET_CLS_CALTECH="${TARGET_CLS_CALTECH:-97}"
TARGET_CLS_CIFAR10="${TARGET_CLS_CIFAR10:-3}"
TARGET_CLS_DTD="${TARGET_CLS_DTD:-3}"
TARGET_CLS="${TARGET_CLS:-3}"
target_cls_for() {
    case "$1" in
        ImageNet1k|ImageNet1kTrain) echo "$TARGET_CLS_IMAGENET" ;;
        Caltech101)                 echo "$TARGET_CLS_CALTECH" ;;
        CIFAR10|CIFAR100)           echo "$TARGET_CLS_CIFAR10" ;;
        DTD)                        echo "$TARGET_CLS_DTD" ;;
        *)                          echo "$TARGET_CLS" ;;
    esac
}

N_EVAL_IMAGENET="${N_EVAL_IMAGENET:-1000}"
N_EVAL_CALTECH="${N_EVAL_CALTECH:-1000}"
N_EVAL_CIFAR10="${N_EVAL_CIFAR10:-1000}"
N_EVAL_DTD="${N_EVAL_DTD:-1000}"
N_EVAL_DEFAULT="${N_EVAL_DEFAULT:-200}"
n_eval_for() {
    case "$1" in
        ImageNet1k|ImageNet1kTrain) echo "$N_EVAL_IMAGENET" ;;
        Caltech101)                 echo "$N_EVAL_CALTECH" ;;
        CIFAR10|CIFAR100)           echo "$N_EVAL_CIFAR10" ;;
        DTD)                        echo "$N_EVAL_DTD" ;;
        *)                          echo "$N_EVAL_DEFAULT" ;;
    esac
}

TRIG_STD="${TRIG_STD:-16}"
TRIG_CIFAR="${TRIG_CIFAR:-8}"
TRIG_HIRES="${TRIG_HIRES:-24}"
is_hires() { case "$1" in ViT-L/14-336px|SigLIP-so400m/14) return 0 ;; *) return 1 ;; esac; }
trig_size_for() {
    if is_hires "$1"; then echo "$TRIG_HIRES"
    else case "$2" in CIFAR10|CIFAR100) echo "$TRIG_CIFAR" ;; *) echo "$TRIG_STD" ;; esac
    fi
}
is_openai_clip() { case "$1" in SigLIP*) return 1 ;; *) return 0 ;; esac; }

# TrojViT patch size must equal the victim's patch grid so that the selected
# patch indices index the model's attention tokens consistently.
trojvit_patch_for() {
    case "$1" in
        ViT-B/32)             echo 32 ;;
        ViT-L/14|ViT-L/14-336px) echo 14 ;;
        SigLIP-so400m/14)     echo 14 ;;
        *)                    echo 16 ;;
    esac
}

mkdir -p "$RUNS_ROOT" "$SUMMARY_ROOT" "$SUMMARY_ROOT/logs"
GRID_LOG="$SUMMARY_ROOT/logs/grid_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$GRID_LOG") 2>&1
echo "baseline grid start $(date '+%F %T')  PY=$PY"
echo "  MODELS=[$MODELS] DATASETS=[$DATASETS] SEEDS=[$SEEDS] REPS=$REPS"
[ -d "$CLIP_BFA_DIR/data/imagenet/val" ] || \
    echo "[warn] missing ImageNet val: $CLIP_BFA_DIR/data/imagenet/val"
[ -d "$FLIP_S_VALDIR_CALTECH" ] || \
    echo "[warn] missing Caltech101: $FLIP_S_VALDIR_CALTECH"
[ -d "$FLIP_S_VALDIR_DTD" ] || \
    echo "[warn] missing DTD: $FLIP_S_VALDIR_DTD"
echo "  RUN_PROFLIP=$RUN_PROFLIP RUN_TROJVIT=$RUN_TROJVIT RUN_FLIP_S=$RUN_FLIP_S"

# ── ProFlip-CBS ────────────────────────────────────────────────────────
run_proflip_one() {
    local model="$1" dataset="$2" seed="$3" rep="$4"
    local tag="${model//\//}"
    local tc ne tsz leaf dir
    tc="$(target_cls_for "$dataset")"
    ne="$(n_eval_for "$dataset")"
    tsz="$(trig_size_for "$model" "$dataset")"
    leaf="zs_c${tc}_s${seed}_r${rep}_proflip_cbs_t${tsz}"
    dir="$RUNS_ROOT/baselines/$tag/$dataset/$leaf"
    if [ "$RESUME" = "1" ] && [ -f "$dir/bfa_results.json" ]; then
        echo "[skip] proflip-cbs  $model $dataset c$tc s$seed r$rep"
        return 0
    fi
    mkdir -p "$dir"
    echo "[run ] proflip-cbs  $model $dataset c$tc n_eval=$ne s$seed r$rep t$tsz"
    "$PY" "$BFA_DIR/proflip_pipeline/run.py" \
        --model "$model" --dataset "$dataset" \
        --seed "$seed" --rep "$rep" --target-cls "$tc" \
        --n-eval "$ne" --n-calib "$N_CALIB" \
        --p1-epochs "$P1_EPOCHS" --p1-grad-clip "$P1_GRAD_CLIP" \
        --trigger-size "$tsz" --out-root "$RUNS_ROOT" \
        --cbs-k "$CBS_K" --cbs-asr-target "$CBS_ASR_TARGET" \
        --cbs-nmax "$CBS_NMAX" --cbs-clean-mse-weight "$CBS_CLEAN_MSE" \
        2>&1 | tee "$dir/run.log" || echo "[warn] proflip-cbs failed: $dir"
}

# ── TrojViT Phase 1 ────────────────────────────────────────────────────
skip_trojvit_combo() {
    local model="$1" dataset="$2"
    case "$dataset" in CIFAR10|CIFAR100) return 0 ;; esac
    return 1
}
run_trojvit_one() {
    local model="$1" dataset="$2" seed="$3" rep="$4"
    local tag="${model//\//}"
    local tc ne ps leaf dir
    tc="$(target_cls_for "$dataset")"
    ne="$(n_eval_for "$dataset")"
    ps="$(trojvit_patch_for "$model")"
    [ "$TROJVIT_PATCH_SIZE" != "0" ] && ps="$TROJVIT_PATCH_SIZE"
    leaf="zs_c${tc}_s${seed}_r${rep}_trojvit_trigger_n${TROJVIT_NUM_PATCH}p${ps}"
    dir="$RUNS_ROOT/baselines/$tag/$dataset/$leaf"
    if [ "$RESUME" = "1" ] && [ -f "$dir/trojvit_results.json" ]; then
        echo "[skip] trojvit      $model $dataset c$tc s$seed r$rep"
        return 0
    fi
    mkdir -p "$dir"
    echo "[run ] trojvit      $model $dataset c$tc n_eval=$ne s$seed r$rep patch=$ps"
    local fixed=""
    [ "$TROJVIT_FIXED_MASKS" = "1" ] && fixed="--trojvit-fixed-masks"
    "$PY" "$BFA_DIR/trojvit_trigger.py" \
        --model "$model" --dataset "$dataset" \
        --seed "$seed" --rep "$rep" --target-cls "$tc" \
        --n-eval "$ne" --out-root "$RUNS_ROOT" \
        --trojvit-images "$TROJVIT_IMAGES" \
        --trojvit-batch "$TROJVIT_BATCH" \
        --trojvit-iters "$TROJVIT_ITERS" \
        --trojvit-lr "$TROJVIT_LR" \
        --trojvit-num-patch "$TROJVIT_NUM_PATCH" \
        --trojvit-patch-size "$ps" \
        --trojvit-patch-select "$TROJVIT_PATCH_SELECT" \
        --trojvit-atten-weight "$TROJVIT_ATTEN_WEIGHT" \
        --trojvit-atten-layer "$TROJVIT_ATTEN_LAYER" \
        --trojvit-loss "$TROJVIT_LOSS" \
        --trojvit-mode "$TROJVIT_MODE" \
        --trojvit-init "$TROJVIT_INIT" $fixed \
        2>&1 | tee "$dir/run.log" || echo "[warn] trojvit failed: $dir"
}

# ── Flip-S (full ZS adapter; OpenAI CLIP models × mapped datasets) ──────
run_flip_s_one() {
    local model="$1" dataset="$2"
    if ! is_openai_clip "$model"; then
        echo "[skip] flip-s       $model (OpenAI clip.load adapter; SigLIP needs flip_s_siglip)"
        return 0
    fi
    local valdir classnames exclude ps img area
    valdir="$(flip_s_valdir_for "$dataset")"
    classnames="$(flip_s_classnames_for "$dataset")"
    exclude="$(flip_s_exclude_for "$dataset")"
    if [ -z "$valdir" ]; then
        echo "[skip] flip-s       $model $dataset (no Flip-S valdir mapping)"
        return 0
    fi
    if [ ! -d "$valdir" ]; then
        echo "[skip] flip-s       $model $dataset (missing valdir: $valdir)"
        return 0
    fi
    if [ ! -f "$classnames" ]; then
        echo "[skip] flip-s       $model $dataset (missing classnames: $classnames)"
        return 0
    fi
    ps="$(flip_s_patch_for "$model")"
    img="$(flip_s_img_for "$model")"
    area="$(awk "BEGIN{printf \"%.2f\", 9*$ps*$ps/($img*$img)*100}")"

    local tag="${model//\//}"
    local tc leaf dir
    tc="$(target_cls_for "$dataset")"
    leaf="zs_c${tc}_s${FLIP_S_SEED}_r1_flip_s_t9x${ps}"
    dir="$RUNS_ROOT/baselines/$tag/$dataset/$leaf"
    if [ "$RESUME" = "1" ] && [ -f "$dir/flip_s_results.json" ]; then
        echo "[skip] flip-s       $model $dataset c$tc"
        return 0
    fi
    mkdir -p "$dir"
    echo "[run ] flip-s       $model $dataset c$tc seed=$FLIP_S_SEED patch=$ps area=${area}%"
    local excl_arg=""
    [ -n "$exclude" ] && excl_arg="--exclude-classes $exclude"
    ( cd "$FLIP_S_DIR" && "$PY" flip_s_clip_v2.py \
        --head zeroshot --target-class "$tc" --stage all \
        --model "$model" --dataset-name "$dataset" \
        --patch-config "$FLIP_S_DIR/config.json" \
        --valdir "$valdir" --classnames "$classnames" $excl_arg ) \
        2>&1 | tee "$dir/run.log" || echo "[warn] flip-s failed: $dir"
    if [ -f "$dir/run.log" ]; then
        "$PY" "$BFA_DIR/parse_flip_s_log.py" \
            --log "$dir/run.log" --out "$dir/flip_s_results.json" \
            --model "$model" --dataset "$dataset" --target-cls "$tc" \
            --seed "$FLIP_S_SEED" --rep 1 \
            --trigger-size "9x${ps}" --trigger-area "$area" \
            || echo "[warn] flip-s log parse failed: $dir"
    fi
}

# ═══ ProFlip-CBS grid ═══
if [ "$RUN_PROFLIP" = "1" ]; then
    echo "════════ ProFlip-CBS: $MODELS × $DATASETS × seeds [$SEEDS] × $REPS ════════"
    for model in $MODELS; do
        for dataset in $DATASETS; do
            for seed in $SEEDS; do
                for rep in $(seq 1 "$REPS"); do
                    run_proflip_one "$model" "$dataset" "$seed" "$rep"
                done
            done
        done
    done
fi

# ═══ TrojViT Phase 1 grid ═══
if [ "$RUN_TROJVIT" = "1" ]; then
    echo "════════ TrojViT Phase 1: $MODELS × $DATASETS × seeds [$SEEDS] × $REPS ════════"
    for model in $MODELS; do
        for dataset in $DATASETS; do
            if skip_trojvit_combo "$model" "$dataset"; then
                echo "[skip-config] trojvit $model × $dataset (CIFAR 16px grid < 9 patches)"
                continue
            fi
            for seed in $SEEDS; do
                for rep in $(seq 1 "$REPS"); do
                    run_trojvit_one "$model" "$dataset" "$seed" "$rep"
                done
            done
        done
    done
fi

# ═══ Flip-S grid ═══
if [ "$RUN_FLIP_S" = "1" ]; then
    echo "════════ Flip-S: $MODELS × $DATASETS (OpenAI CLIP only) ════════"
    for model in $MODELS; do
        for dataset in $DATASETS; do
            run_flip_s_one "$model" "$dataset"
        done
    done
fi

# ═══ Separate baseline summaries ═══
echo "════════ aggregate baselines ════════"
"$PY" "$BFA_DIR/aggregate_baselines.py" \
    --runs-root "$RUNS_ROOT" --out-dir "$SUMMARY_ROOT" \
    || echo "[warn] baseline aggregation failed"
echo "baseline grid end $(date '+%F %T')  log=$GRID_LOG"
