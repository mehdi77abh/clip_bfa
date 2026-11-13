#!/usr/bin/env bash
# run_all.sh — validation + ablation grids for CLIP-FLIP (RESULTS.md §1, §3).
# ===========================================================================
# Output is grouped into 3 top-level sections under OUT_ROOT:
#   grid_results/validation/…   §1 validation matrix
#   grid_results/ablation/…     §3 variants + trigger-size / trigger-location sweeps
#   grid_results/baselines/…    §2 baseline runs (random / simple-grad — later)
#
# Grids:
#   validation   : MODELS × DATASETS × SEEDS × REPS rounds per config
#   ablation     : variants on one config, SEEDS × REPS rounds
#   size/location sweeps : SWEEP_MODELS × one dataset, ONE seed, ONE run each
#
# Method-arm tagging: every run dir gets __p1{loss}__lse{α} (Phase-1 loss +
# Phase-3 LSE weight — part of the run identity), so different arms never
# collide on disk and RESUME can't skip one arm's rows for another's.
#
# Trigger sizes follow the per-(model, dataset) policy (see trig_size_for):
#   224px models → 16 (Caltech/ImageNet) / 8 (CIFAR); 336/384px models → 24,
#   no CIFAR.  The size sweep (SIZES) uses its explicit values.
#
# Attacked model state (model_state_attacked.pt, for defence / VLM-transfer
# evaluation) is saved ONLY for ViT-L/14 × ImageNet1k × last seed × last rep
# — gated by experiment.py's --save-attacked-model.
#
# Target hunting: two-phase search — see the MAX_FLIPS block above.
#
# Re-running skips configs whose bfa_results.json exists (RESUME=1).
#
# Examples:
#   bash run_all.sh
#   MODELS="ViT-B/16" DATASETS="Caltech101" RUN_VALIDATION=1 RUN_ABLATION=0 bash run_all.sh
#   P1_LOSS=cos P3_ALPHA_LSE=0 bash run_all.sh      # held 'normal' arm
#   PY=/home/mehdi/miniconda3/envs/vlm_attack/bin/python bash run_all.sh

set -u

PY="${PY:-python}"
OUT_ROOT="${OUT_ROOT:-./grid_results}"
SEEDS="${SEEDS:-42}"              # one static seed (maintainer decision 2026-11-12 —
                                  # replaces the earlier 41 42 43 multi-seed set)
REPS="${REPS:-5}"                 # rounds per config (validation runs each
                                  # config 5 times — maintainer decision 2026-11-13)
SWEEP_SEED="${SWEEP_SEED:-42}"    # size/location sweeps: one seed, one run
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

# Per-dataset target class (feline targets, canonical classes.txt lines):
#   ImageNet1k 'lion' (line 292 → idx 291)  Caltech101 'wild cat' (line 98 → 97)
#   CIFAR10 'cat' (line 4 → 3).  TARGET_CLS is the fallback for other datasets.
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
# Trigger-size policy (2026-11-12, maintainer decision):
#   224-input models (ViT-B/16, ViT-L/14, ViT-B/32, SigLIP-B/16):
#     TRIG_STD on Caltech101/ImageNet1k, TRIG_CIFAR on CIFAR10 (small images
#     upsampled + 10-way task make a 16px trigger dominant there)
#   336/384-input models (ViT-L/14-336px, SigLIP-so400m/14):
#     TRIG_HIRES (24 ≈ 7% of the larger input width), and no CIFAR10 runs.
# Override: SKIP_HIRES_CIFAR=0 re-enables CIFAR for the hi-res models.
TRIG_STD="${TRIG_STD:-16}"
TRIG_CIFAR="${TRIG_CIFAR:-8}"
TRIG_HIRES="${TRIG_HIRES:-24}"
SKIP_HIRES_CIFAR="${SKIP_HIRES_CIFAR:-1}"

is_hires() {
    case "$1" in
        ViT-L/14-336px|SigLIP-so400m/14) return 0 ;;
        *) return 1 ;;
    esac
}

skip_combo() {
    # skip_combo <model> <dataset>
    if [ "$SKIP_HIRES_CIFAR" = "1" ] && is_hires "$1"; then
        case "$2" in CIFAR10|CIFAR100) return 0 ;; esac
    fi
    return 1
}

trig_size_for() {
    # trig_size_for <model> <dataset>
    if is_hires "$1"; then
        echo "$TRIG_HIRES"
    else
        case "$2" in
            CIFAR10|CIFAR100) echo "$TRIG_CIFAR" ;;
            *)                echo "$TRIG_STD" ;;
        esac
    fi
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

# Trigger loss + Phase-3 LSE (2026-11-14, class-list-free terms ported from
# main_arm_c_cos_astar.py): P1_LOSS=cos | cos_astar (CE dropped — invalid
# under the BFA threat model); P3_ALPHA_LSE=0 keeps the previous ranking
# metric (bit-identical), >0 adds the LSE-over-centroids term.
P1_LOSS="${P1_LOSS:-cos_astar}"
ASTAR_K="${ASTAR_K:-8}"
ASTAR_LAMBDA="${ASTAR_LAMBDA:-1.0}"
ASTAR_MU="${ASTAR_MU:-0.05}"
ASTAR_GRAD_CLIP="${ASTAR_GRAD_CLIP:-1.0}"
P3_ALPHA_LSE="${P3_ALPHA_LSE:-0.3}"
P3_LSE_BETA="${P3_LSE_BETA:-10.0}"
TRIGGER_MIN_ASR="${TRIGGER_MIN_ASR:-5.0}"
TRIGGER_RETRIES="${TRIGGER_RETRIES:-3}"
RESUME="${RESUME:-1}"

# Validation matrix (RESULTS.md §1)
MODELS="${MODELS:-ViT-B/16 ViT-L/14 ViT-B/32 ViT-L/14-336px SigLIP-B/16 SigLIP-so400m/14}"
# size/location sweeps (default: all validation models; restrictable)
SWEEP_MODELS="${SWEEP_MODELS:-$MODELS}"
DATASETS="${DATASETS:-Caltech101 ImageNet1k}"   # CIFAR10 skipped (2026-11-13)

RUN_VALIDATION="${RUN_VALIDATION:-1}"
RUN_ABLATION="${RUN_ABLATION:-1}"
RUN_LOCATIONS="${RUN_LOCATIONS:-1}"
RUN_SIZE_SWEEP="${RUN_SIZE_SWEEP:-0}"

# Ablation / sweep config (variants on ABLATION_MODEL; sweeps: all MODELS × ABLATION_DATASET)
ABLATION_MODEL="${ABLATION_MODEL:-ViT-L/14}"
ABLATION_DATASET="${ABLATION_DATASET:-ImageNet1k}"
ABLATION_VARIANTS="${ABLATION_VARIANTS:-full a b c}"
LOCATIONS="${LOCATIONS:-top_left top_right bottom_left bottom_right}"
SIZES="${SIZES:-8 12 16 20 24 32}"

# Absolute so the resume check and experiment.py agree (bug fix: --out-root
# is now forwarded — OUT_ROOT used to be silently ignored by experiment.py).
OUT_ROOT_ABS="$(realpath -m "$OUT_ROOT")"

run_one() {
    # run_one <section> <model> <dataset> <seed> <rep> <out-seg> [extra args…]
    local section="$1" model="$2" dataset="$3" seed="$4" rep="$5" seg="$6"
    shift 6
    # Method-arm tag (mirrors experiment.py's run_seg): the Phase-1 loss and
    # Phase-3 LSE weight are part of the run identity.
    seg="${seg}__p1${P1_LOSS}__lse${P3_ALPHA_LSE//./p}"
    local tag="${model//\//}"
    local tc="$(target_cls_for "$dataset")"
    local ne="$(n_eval_for "$dataset")"
    local dir="$OUT_ROOT_ABS/$section/$tag/$dataset/zs/c$tc/s$seed/r$rep/$seg"
    if [ "$RESUME" = "1" ] && [ -f "$dir/bfa_results.json" ]; then
        echo "[skip] $dir"
        return 0
    fi
    echo "[run ] $section  $model  $dataset  c$tc  n_eval=$ne  s$seed r$rep  ($seg)"
    "$PY" experiment.py --model "$model" --dataset "$dataset" \
        --seed "$seed" --rep "$rep" --target-cls "$tc" \
        --n-eval "$ne" --p1-epochs "$P1_EPOCHS" \
        --n-calib "$N_CALIB" \
        --p1-loss "$P1_LOSS" --astar-k "$ASTAR_K" \
        --astar-lambda "$ASTAR_LAMBDA" --astar-mu "$ASTAR_MU" \
        --astar-grad-clip "$ASTAR_GRAD_CLIP" \
        --p3-alpha-lse "$P3_ALPHA_LSE" --p3-lse-beta "$P3_LSE_BETA" \
        --max-flips "$MAX_FLIPS" \
        --normal-stop "$NORMAL_STOP" --fine-target "$FINE_TARGET" \
        --fine-max-flips "$FINE_MAX_FLIPS" --flip-min-gain "$FLIP_MIN_GAIN" \
        --fine-patience "$FINE_PATIENCE" --fine-min-gain "$FINE_MIN_GAIN" \
        --fine-max-rejects "$FINE_MAX_REJECTS" \
        --trigger-min-asr "$TRIGGER_MIN_ASR" \
        --trigger-retries "$TRIGGER_RETRIES" \
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
            if skip_combo "$model" "$dataset"; then
                echo "[skip-config] $model × $dataset (policy: hi-res models skip CIFAR)"
                continue
            fi
            for seed in $SEEDS; do
                for rep in $(seq 1 "$REPS"); do
                    tsz="$(trig_size_for "$model" "$dataset")"
                    extra=""
                    # attacked model state: ViT-L/14 × ImageNet1k × last seed × last rep only
                    if [ "$model" = "ViT-L/14" ] && [ "$dataset" = "ImageNet1k" ] \
                        && [ "$seed" = "$LAST_SEED" ] && [ "$rep" -eq "$REPS" ]; then
                        extra="--save-attacked-model"
                    fi
                    run_one validation "$model" "$dataset" "$seed" "$rep" \
                        "full__t$tsz" \
                        --variant full --trigger-size "$tsz" $extra
                done
            done
        done
    done
fi

# ═══ 2. ABLATION VARIANTS (RESULTS.md §3b–3d) — section: ablation ═══
if [ "$RUN_ABLATION" = "1" ] && ! skip_combo "$ABLATION_MODEL" "$ABLATION_DATASET"; then
    tsz="$(trig_size_for "$ABLATION_MODEL" "$ABLATION_DATASET")"
    echo "════════ ABLATION VARIANTS: $ABLATION_VARIANTS (t$tsz, seeds [$SEEDS] × $REPS reps) ════════"
    for v in $ABLATION_VARIANTS; do
        for seed in $SEEDS; do
            for rep in $(seq 1 "$REPS"); do
                run_one ablation "$ABLATION_MODEL" "$ABLATION_DATASET" "$seed" "$rep" \
                    "$v""__t$tsz" \
                    --variant "$v" --trigger-size "$tsz"
            done
        done
    done
fi

# ═══ 3. TRIGGER-LOCATION SWEEP — SWEEP_MODELS, one dataset, one seed, one run ═══
# Train + eval at the same corner.
if [ "$RUN_LOCATIONS" = "1" ]; then
    echo "════════ TRIGGER LOCATIONS: $LOCATIONS (sweep models, $ABLATION_DATASET, s$SWEEP_SEED) ════════"
    for model in $SWEEP_MODELS; do
        skip_combo "$model" "$ABLATION_DATASET" && continue
        ltsz="$(trig_size_for "$model" "$ABLATION_DATASET")"
        for loc in $LOCATIONS; do
            run_one ablation "$model" "$ABLATION_DATASET" "$SWEEP_SEED" 1 \
                "full__t$ltsz""__$loc" \
                --variant full --trigger-size "$ltsz" \
                --trigger-location "$loc"
        done
    done
fi

# ═══ 4. TRIGGER-SIZE SWEEP (RESULTS.md §3a) — SWEEP_MODELS, one dataset, one
#    seed, one run — opt-in ═══
if [ "$RUN_SIZE_SWEEP" = "1" ]; then
    echo "════════ TRIGGER SIZES: $SIZES (sweep models, $ABLATION_DATASET, s$SWEEP_SEED) ════════"
    for model in $SWEEP_MODELS; do
        skip_combo "$model" "$ABLATION_DATASET" && continue
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
