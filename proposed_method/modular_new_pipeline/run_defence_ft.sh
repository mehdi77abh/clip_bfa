#!/usr/bin/env bash
# run_defence_ft.sh — Fine-Tuning defence on the two saved attacked models.
# ===========================================================================
# DEFENSES.md §1 / RESULTS.md §4.  Exactly two configurations:
#   1. ZS  — ViT-L/14 × ImageNet1k zero-shot run   (zs_* run dir)
#   2. LP  — ViT-L/14 × ImageNet1k linear-probe run (lp_* run dir)
#
# For each: load model_state_attacked.pt from its run dir → evaluate
# BEFORE → fine-tune on CC3M (clean image–caption pairs) → evaluate AFTER
# on the identical eval set → defence_ft.json saved inside the run dir.
#
# Prerequisites:
#   * attacked model states: the run dirs must contain
#     model_state_attacked.pt (run_all.sh saves it for ViT-L/14 × ImageNet1k
#     × last seed × last rep; run_lp.sh saves it by default).  Re-run the
#     attack with --save-attacked-model otherwise.
#   * CC3M in ../data/cc3m (BadCLIP convention: train/train.csv with
#     image + caption columns and the image files — see defence_finetune.py).
#
# Run-dir discovery: the first <task>_*/model_state_attacked.pt found under
# {out_root}/*/{tag}/{dataset}/; override with RUN_DIR_ZS / RUN_DIR_LP.
#
# Examples:
#   bash run_defence_ft.sh
#   RUN_DIR_ZS=/path/to/zs_run RUN_DIR_LP=/path/to/lp_run bash run_defence_ft.sh
#   FT_MAX_STEPS=5000 FT_SCOPE=vision bash run_defence_ft.sh
#   PY=~/miniconda3/envs/vlm_attack/bin/python bash run_defence_ft.sh

set -u

PY="${PY:-python}"
OUT_ROOT="${OUT_ROOT:-./grid_results}"
MODEL="${MODEL:-ViT-L/14}"
DATASET="${DATASET:-ImageNet1k}"

# ── CC3M fine-tune data ──
CC3M_ROOT="${CC3M_ROOT:-../data/cc3m}"
CC3M_META="${CC3M_META:-}"          # empty = auto-detect (train/train.csv, …)
CC3M_IMAGES="${CC3M_IMAGES:-}"      # empty = auto
FT_MAX_SAMPLES="${FT_MAX_SAMPLES:-0}"

# ── fine-tuning (DEFENSES.md "TBD" defaults; all overridable) ──
FT_EPOCHS="${FT_EPOCHS:-1}"
FT_MAX_STEPS="${FT_MAX_STEPS:-2000}"
FT_LR="${FT_LR:-1e-5}"
FT_WD="${FT_WD:-0.1}"
FT_BATCH="${FT_BATCH:-64}"
FT_SCOPE="${FT_SCOPE:-full}"        # full | vision
FT_SEED="${FT_SEED:-42}"
FT_WORKERS="${FT_WORKERS:-4}"
EVAL_BATCH="${EVAL_BATCH:-64}"
DEVICE="${DEVICE:-auto}"
SAVE_FINETUNED="${SAVE_FINETUNED:-0}"

TAG="${MODEL//\//}"
OUT_ROOT_ABS="$(realpath -m "$OUT_ROOT")"

find_run() {
    # find_run <task> — first <task>_*/model_state_attacked.pt under the tree
    find "$OUT_ROOT_ABS" -path "*/$TAG/$DATASET/${1}_*" \
        -name model_state_attacked.pt 2>/dev/null \
        | sort | head -1 | xargs -r dirname
}

RUN_DIR_ZS="${RUN_DIR_ZS:-$(find_run zs)}"
RUN_DIR_LP="${RUN_DIR_LP:-$(find_run lp)}"

LOG_DIR="$OUT_ROOT_ABS/logs"
mkdir -p "$LOG_DIR"
GRID_LOG="$LOG_DIR/defence_ft_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$GRID_LOG") 2>&1
echo "FT defence start $(date '+%F %T')  PY=$PY"
echo "  MODEL=$MODEL  DATASET=$DATASET  CC3M=$CC3M_ROOT"
echo "  FT: scope=$FT_SCOPE lr=$FT_LR wd=$FT_WD batch=$FT_BATCH epochs=$FT_EPOCHS max_steps=$FT_MAX_STEPS"
echo "  ZS run: ${RUN_DIR_ZS:-<none found>}"
echo "  LP run: ${RUN_DIR_LP:-<none found>}"

# ═══ 0. PREREQUISITES ═══
if [ ! -d "$CC3M_ROOT" ]; then
    echo "ERROR: CC3M root missing: $CC3M_ROOT (put the CC3M data under ../data/cc3m)"
    exit 1
fi
MISSING=0
for rd in "$RUN_DIR_ZS" "$RUN_DIR_LP"; do
    if [ -z "$rd" ] || [ ! -f "$rd/model_state_attacked.pt" ]; then
        MISSING=1
    fi
done
if [ "$MISSING" = "1" ]; then
    echo "ERROR: could not find both attacked run dirs with model_state_attacked.pt"
    echo "       under $OUT_ROOT_ABS/*/$TAG/$DATASET/{zs,lp}_* ."
    echo "       Set RUN_DIR_ZS / RUN_DIR_LP explicitly, or re-run the attack"
    echo "       with --save-attacked-model (run_all.sh saves it for"
    echo "       ViT-L/14 × ImageNet1k × last seed × last rep; run_lp.sh by default)."
    exit 1
fi

# ═══ 1. ZS — ViT-L/14 × ImageNet1k zero-shot ═══
run_defence() {
    # run_defence <task> <run-dir>
    local task="$1" run_dir="$2"
    echo "════════ FT DEFENCE [$task]  $run_dir ════════"
    local extra=""
    [ "$SAVE_FINETUNED" = "1" ] && extra="--save-finetuned"
    "$PY" defence_finetune.py --run-dir "$run_dir" --task "$task" \
        --cc3m-root "$CC3M_ROOT" \
        ${CC3M_META:+--cc3m-meta "$CC3M_META"} \
        ${CC3M_IMAGES:+--cc3m-images "$CC3M_IMAGES"} \
        --ft-max-samples "$FT_MAX_SAMPLES" \
        --ft-epochs "$FT_EPOCHS" --ft-max-steps "$FT_MAX_STEPS" \
        --ft-lr "$FT_LR" --ft-wd "$FT_WD" --ft-batch "$FT_BATCH" \
        --ft-scope "$FT_SCOPE" --ft-seed "$FT_SEED" --ft-workers "$FT_WORKERS" \
        --eval-batch "$EVAL_BATCH" --device "$DEVICE" $extra 2>&1 \
        | tee "$run_dir/defence_ft.log"
}

run_defence zs "$RUN_DIR_ZS"

# ═══ 2. LP — ViT-L/14 × ImageNet1k linear-probe ═══
run_defence lp "$RUN_DIR_LP"

# ═══ 3. SUMMARY (RESULTS.md §4 table) ═══
echo "════════ FT DEFENCE SUMMARY (RESULTS.md §4) ════════"
"$PY" - "$RUN_DIR_ZS" "$RUN_DIR_LP" <<'EOF'
import json, sys
print(f"  {'task':<4} {'dataset':<10} {'ASR (no defence)':>17} {'ASR (after)':>12} "
      f"{'CACC (no defence)':>17} {'CACC (after)':>12}")
for rd in sys.argv[1:]:
    try:
        with open(f"{rd}/defence_ft.json") as f:
            r = json.load(f)
    except Exception as e:
        print(f"  ? {rd}: {e}")
        continue
    b, a = r["before"], r["after"]
    print(f"  {r['task']:<4} {r['dataset']:<10} {b['asr']:>16.2f}% {a['asr']:>11.2f}% "
          f"{b['cacc']:>16.2f}% {a['cacc']:>11.2f}%   ({rd})")
EOF

echo "FT defence done $(date '+%F %T')"
