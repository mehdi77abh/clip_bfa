#!/usr/bin/env bash
# run_vlm_transfer.sh — the VLM-transfer experiment, end to end.
# ================================================================
# CHECKLIST.md §3.5: show the CLIP-BFA backdoor survives when the attacked
# CLIP visual encoder is used as a VLM's vision tower.
#
# Two stages:
#   1. ATTACK   — run CLIP-BFA from scratch on the VLM's own CLIP encoder
#                 (run_attack.py → model_state_attacked.pt + trigger.pt)
#   2. EVAL     — load the VLM, swap the attacked tower in, generate on
#                 OK-VQA / Flickr-30K and score the TARGETED metrics
#                 (evaluate.py → vlm_transfer.json)
#
# The matrix is deduplicated: OF-3B and LLaVA-7B share the ViT-L/14 tower,
# so one attack run serves both.
#
#   VLM            tower (X-Transfer Table 6)     attack model
#   OF-3B          ViT-L/14 CLIP OpenAI  224      ViT-L/14
#   LLaVA-7B       ViT-L/14 CLIP OpenAI  224      ViT-L/14        (shared)
#   LLaVA-7B@336   ViT-L/14-336 CLIP     336      ViT-L/14-336px
#   BLIP-2         ViT-G/14 EVA-CLIP     364      EVA-CLIP-ViT-G/14
#
#   dataset        task         split size (full, X-Transfer "each image")
#   ok-vqa         VQA          5 046 questions (COCO val2014)
#   flickr30k      captioning   1 000 images / 5 000 captions
#
# Attack config (maintainer decision 2026-11-21): ImageNet1k, target class
# 291 'lion' — the validated reference configuration.
#
# Examples:
#   bash run_vlm_transfer.sh
#   VLMS="OF-3B" EVAL_DATASETS="flickr30k" bash run_vlm_transfer.sh
#   RUN_ATTACK=0 bash run_vlm_transfer.sh          # eval only
#   PY=/home/mehdi/miniconda3/envs/vlm_attack/bin/python bash run_vlm_transfer.sh

set -u

PY="${PY:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${OUT_ROOT:-./grid_results}"

# ── which VLMs / which evaluation datasets ───────────────────────────
VLMS="${VLMS:-OF-3B LLaVA-7B LLaVA-7B@336 BLIP-2}"
EVAL_DATASETS="${EVAL_DATASETS:-ok-vqa flickr30k}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"      # 0 = the FULL eval split
EVAL_BATCH="${EVAL_BATCH:-8}"

# ── the CLIP-BFA attack (ImageNet1k / 'lion' c291) ──────────────────
ATTACK_DATASET="${ATTACK_DATASET:-ImageNet1k}"
TARGET_CLS="${TARGET_CLS:-291}"
SEEDS="${SEEDS:-42}"
REPS="${REPS:-1}"
SECTION="${SECTION:-validation}"
TRIG_SIZE="${TRIG_SIZE:-16}"
P1_EPOCHS="${P1_EPOCHS:-300}"
N_CALIB="${N_CALIB:-64}"
N_EVAL="${N_EVAL:-1000}"
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
P1_GRAD_CLIP="${P1_GRAD_CLIP:-1.0}"

# ── model-specific hooks ────────────────────────────────────────────
VLM_CHECKPOINT="${VLM_CHECKPOINT:-}"     # OF-3B trained checkpoint.pt
EVA_CLIP_NAME="${EVA_CLIP_NAME:-}"        # BLIP-2: EVA-CLIP model id
EVA_CLIP_PRETRAINED="${EVA_CLIP_PRETRAINED:-}"

RUN_ATTACK="${RUN_ATTACK:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"

OUT_ROOT_ABS="$(realpath -m "$OUT_ROOT")"

LOG_DIR="$OUT_ROOT_ABS/logs"
mkdir -p "$LOG_DIR"
GRID_LOG="$LOG_DIR/vlm_transfer_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$GRID_LOG") 2>&1
echo "vlm-transfer start $(date '+%F %T')  PY=$PY"
echo "  VLMS=[$VLMS]  EVAL_DATASETS=[$EVAL_DATASETS]  "
echo "  ATTACK: $ATTACK_DATASET c$TARGET_CLS  SEEDS=[$SEEDS]  TRIG_SIZE=$TRIG_SIZE"

# ── VLM → attack model (the tower must be replaced 1:1) ─────────────
attack_model_for() {
    case "$1" in
        OF-3B)          echo "ViT-L/14" ;;
        LLaVA-7B)       echo "ViT-L/14" ;;
        LLaVA-7B@336)   echo "ViT-L/14-336px" ;;
        BLIP-2)         echo "EVA-CLIP-ViT-G/14" ;;
        *) echo "unknown VLM '$1'" >&2; return 1 ;;
    esac
}

# ── trigger size per attack model (matches run_all.sh's policy) ─────
trig_size_for() {
    case "$1" in
        ViT-L/14-336px) echo "${TRIG_SIZE_HIRES:-24}" ;;
        *)              echo "$TRIG_SIZE" ;;
    esac
}

run_attack_one() {
    # run_attack_one <vlm> <attack-model> <seed> <rep>
    local vlm="$1" amodel="$2" seed="$3" rep="$4"
    local tag="${amodel//\//}"
    local tsz="$(trig_size_for "$amodel")"
    local seg="full_t${tsz}"
    local leaf="zs_c${TARGET_CLS}_s${seed}_r${rep}_${seg}"
    local dir="$OUT_ROOT_ABS/$SECTION/$tag/$ATTACK_DATASET/$leaf"

    if [ "$RESUME" = "1" ] && [ -f "$dir/model_state_attacked.pt" ]; then
        echo "[skip] attack $amodel s$seed r$rep  ($dir)"
        return 0
    fi
    echo "[run ] attack  VLM=$vlm  model=$amodel  $ATTACK_DATASET c$TARGET_CLS  s$seed r$rep  t$tsz"
    if [ "$DRY_RUN" = "1" ]; then return 0; fi

    mkdir -p "$dir"
    local extra=()
    [ -n "$VLM_CHECKPOINT" ] && extra+=(--vlm-checkpoint "$VLM_CHECKPOINT")
    [ -n "$EVA_CLIP_NAME" ] && extra+=(--eva-clip-name "$EVA_CLIP_NAME")
    [ -n "$EVA_CLIP_PRETRAINED" ] && extra+=(--eva-clip-pretrained "$EVA_CLIP_PRETRAINED")

    "$PY" "$HERE/run_attack.py" --vlm "$vlm" \
        --dataset "$ATTACK_DATASET" --target-cls "$TARGET_CLS" \
        --seed "$seed" --rep "$rep" --section "$SECTION" \
        --trigger-size "$tsz" --n-eval "$N_EVAL" \
        --n-calib "$N_CALIB" --p1-epochs "$P1_EPOCHS" \
        --p1-grad-clip "$P1_GRAD_CLIP" \
        --max-flips "$MAX_FLIPS" --normal-stop "$NORMAL_STOP" \
        --fine-target "$FINE_TARGET" --fine-max-flips "$FINE_MAX_FLIPS" \
        --flip-min-gain "$FLIP_MIN_GAIN" --fine-patience "$FINE_PATIENCE" \
        --fine-min-gain "$FINE_MIN_GAIN" --fine-max-rejects "$FINE_MAX_REJECTS" \
        --trigger-min-asr "$TRIGGER_MIN_ASR" \
        --trigger-retries "$TRIGGER_RETRIES" \
        --out-root "$OUT_ROOT_ABS" "${extra[@]}" 2>&1 | tee "$dir/run.log"
}

run_eval_one() {
    # run_eval_one <vlm> <attack-model> <dataset> <seed> <rep>
    local vlm="$1" amodel="$2" ds="$3" seed="$4" rep="$5"
    local tag="${amodel//\//}"
    local tsz="$(trig_size_for "$amodel")"
    local seg="full_t${tsz}"
    local leaf="zs_c${TARGET_CLS}_s${seed}_r${rep}_${seg}"
    local dir="$OUT_ROOT_ABS/$SECTION/$tag/$ATTACK_DATASET/$leaf"

    if [ "$DRY_RUN" != "1" ] && [ ! -f "$dir/model_state_attacked.pt" ]; then
        echo "[warn] no attacked model in $dir — run the attack first"
        return 1
    fi
    if [ "$RESUME" = "1" ] && [ -f "$dir/vlm_transfer.json" ]; then
        echo "[skip] eval $vlm x $ds  ($dir/vlm_transfer.json)"
        return 0
    fi
    echo "[run ] eval  VLM=$vlm  dataset=$ds  split=$EVAL_SPLIT  max_samples=${MAX_SAMPLES:-FULL}"
    if [ "$DRY_RUN" = "1" ]; then return 0; fi

    local extra=()
    [ -n "$VLM_CHECKPOINT" ] && extra+=(--vlm-checkpoint "$VLM_CHECKPOINT")

    "$PY" "$HERE/evaluate.py" --run-dir "$dir" --vlm "$vlm" \
        --dataset "$ds" --split "$EVAL_SPLIT" \
        --max-samples "$MAX_SAMPLES" --eval-batch "$EVAL_BATCH" \
        --out-name "vlm_transfer_${ds}.json" \
        --swap-mode "${SWAP_MODE:-tower}" "${extra[@]}" 2>&1 \
        | tee "$dir/eval_${ds}.log"
}

# ══════════════════════════════════════════════════════════════════════
#  STAGE 1 — the attacks (one per distinct tower × seed × rep)
# ══════════════════════════════════════════════════════════════════════
declare -A ATTACKED
if [ "$RUN_ATTACK" = "1" ]; then
    echo
    echo "════════ STAGE 1: CLIP-BFA attacks (save model_state_attacked.pt) ════════"
    for vlm in $VLMS; do
        amodel="$(attack_model_for "$vlm")" || exit 1
        key="$amodel"
        if [ -n "${ATTACKED[$key]:-}" ]; then
            echo "[dedup] $amodel already attacked for ${ATTACKED[$key]}"
            continue
        fi
        ATTACKED[$key]="$vlm"
        for seed in $SEEDS; do
            for rep in $(seq 1 "$REPS"); do
                run_attack_one "$vlm" "$amodel" "$seed" "$rep"
            done
        done
    done
else
    echo "[skip] STAGE 1 (RUN_ATTACK=0)"
fi

# ══════════════════════════════════════════════════════════════════════
#  STAGE 2 — the transfer evaluations
# ══════════════════════════════════════════════════════════════════════
if [ "$RUN_EVAL" = "1" ]; then
    echo
    echo "════════ STAGE 2: VLM transfer evaluation ════════"
    for vlm in $VLMS; do
        amodel="$(attack_model_for "$vlm")" || exit 1
        for ds in $EVAL_DATASETS; do
            for seed in $SEEDS; do
                for rep in $(seq 1 "$REPS"); do
                    run_eval_one "$vlm" "$amodel" "$ds" "$seed" "$rep"
                done
            done
        done
    done
else
    echo "[skip] STAGE 2 (RUN_EVAL=0)"
fi

# ══════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════
if [ "$DRY_RUN" = "1" ]; then
    echo
    echo "[dry-run] nothing was executed."
    exit 0
fi

echo
echo "════════ VLM-TRANSFER SUMMARY ════════"
"$PY" - "$OUT_ROOT_ABS" <<'PYEOF'
import glob, json, os, sys
root = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(root, "**", "vlm_transfer_*.json"),
                             recursive=True)):
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:
        print(f"  (could not read {path}: {e})")
        continue
    s = d.get("summary", {})
    a = s.get("attacked_triggered", {})
    c = s.get("default_clean", {})
    rows.append((
        d.get("vlm", "?"), d.get("dataset", "?"), d.get("task", "?"),
        a.get("target_mention_rate"), c.get("target_mention_rate"),
        a.get("bleu4_vs_target_desc"), s.get("mention_delta_vs_default_clean"),
        d.get("n_samples"),
    ))
if not rows:
    print("  no vlm_transfer_*.json found under", root)
    sys.exit(0)
hdr = ("VLM", "dataset", "task", "mention(D)", "mention(A)",
       "bleu4(D)", "Δ mention", "n")
print("  %-14s %-10s %-9s %-11s %-11s %-9s %-9s %-7s" % hdr)
for r in rows:
    fmt = lambda v: "—" if v is None else (f"{v:.4f}" if isinstance(v, float)
                                          else str(v))
    print("  %-14s %-10s %-9s %-11s %-11s %-9s %-9s %-7s"
          % (r[0], r[1], r[2], fmt(r[3]), fmt(r[4]), fmt(r[5]),
             fmt(r[6]), r[7]))
print("\n  D = attacked tower + triggered input   A = default tower + clean input")
PYEOF

echo
echo "vlm-transfer end $(date '+%F %T')  log: $GRID_LOG"
