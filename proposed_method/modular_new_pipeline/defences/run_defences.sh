#!/usr/bin/env bash
# run_defences.sh — run BOTH defence methods end-to-end on one victim.
# ===========================================================================
# DEFENSES.md §1/§2 · RESULTS.md §4.  Fixed configuration (all env-overridable):
#   ViT-L/14 × ImageNet1k, eval = zero-shot (zs) AND linear-probe (lp).
#
# Two defences, two flows:
#
#   Defence 1 — FT  (bfa attack → Eval → FT → Eval)
#     per task:
#       1. experiment.py --save-attacked-model   → attack the UNDEFENDED victim
#       2. defence_finetune.py                   → eval BEFORE (ASR/CACC "no
#                                                  defence"), fine-tune on CC3M
#       3.                                       → eval AFTER  (ASR/CACC "after")
#          → defence_ft.json in the run dir
#
#   Defence 2 — FaR (FaR → eval → bfa attack → eval)
#     once:
#       1. far_clip/far_clip.py                  → defended victim checkpoint
#          (its printed calib accuracy + the saved _far_cfg.json = the eval
#           right after FaR, i.e. CACC of the defended model)
#     per task:
#       3. experiment.py --far-ckpt <defended>   → attack the DEFENDED victim
#       4. bfa_results.json metrics              → eval after attack (ASR/CACC)
#
#   The UNDEFENDED attack runs double as the "no defence" baseline for the
#   FaR row — no duplicate undefended attacks.  The same LP head (trained
#   once on clean features, TASKS.md §2) is reused by every run.
#
# RESUME: each step is skipped when its output already exists, so the script
# is cheap to re-run and continues an interrupted grid.
#
# Outputs (under OUT_ROOT/SECTION/ViT-L14/ImageNet1k/):
#   {task}_c291_s42_r1_full_t16/        undefended attack  (+ defence_ft.json)
#   {task}_c291_s42_r1_full_t16_far/    FaR-defended attack
#   defences/defended_models/far_ViT-L14.pt   the FaR checkpoint
#
# Examples:
#   bash run_defences.sh                       # both defences, zs + lp
#   TASKS="zs" bash run_defences.sh            # zero-shot only
#   DEFENCES="far" bash run_defences.sh        # FaR only
#   TARGET_CLS=291 TRIG_SZ=16 SEED=42 bash run_defences.sh
#   PY=~/miniconda3/envs/vlm_attack/bin/python bash run_defences.sh

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../modular_new_pipeline/defences
ROOT="$(dirname "$HERE")"                              # .../modular_new_pipeline
cd "$ROOT"                                             # experiment.py / config.json live here

PY="${PY:-python}"
OUT_ROOT="${OUT_ROOT:-./grid_results}"
MODEL="${MODEL:-ViT-L/14}"
DATASET="${DATASET:-ImageNet1k}"
SECTION="${SECTION:-others}"       # results top-level group (experiment.py)
VARIANT="${VARIANT:-full}"
SEED="${SEED:-42}"
REP="${REP:-1}"
TARGET_CLS="${TARGET_CLS:-291}"    # lion — run_lp.sh ImageNet default
TRIG_SZ="${TRIG_SZ:-16}"           # ViT-L/14 standard trigger (TRIG_STD)
TASKS="${TASKS:-zs lp}"
DEVICE="${DEVICE:-auto}"
DEFENCES="${DEFENCES:-ft far}"     # which defences to run

# ── paths (resolved like the code: config.resolve_path → proposed_method/) ──
TAG="${MODEL//\//}"                # ViT-L/14 → ViT-L14
resolve() { "$PY" -c 'from config import load_registry, resolve_path; import sys, os; print(os.path.join(resolve_path(load_registry()["datasets"]["ImageNet1k"]["root"]), sys.argv[1]))' "$1"; }
LP_HEAD_ROOT="${LP_HEAD_ROOT:-$("$PY" -c 'from config import load_registry, resolve_path; print(resolve_path(load_registry()["paths"]["lp_head_root"]))')}"
LP_HEAD="${LP_HEAD:-$LP_HEAD_ROOT/${TAG}_${DATASET}.pt}"

# ── FaR defended victim (DEFENSES.md §2) ──
FAR_DIR="${FAR_DIR:-$HERE/defended_models}"
FAR_CKPT="${FAR_CKPT:-$FAR_DIR/far_${TAG}.pt}"
FAR_DATA_ROOT="${FAR_DATA_ROOT:-$(resolve train)}"   # ImageNet train = calib set
FAR_CLASSES="${FAR_CLASSES:-$ROOT/class_texts/${DATASET,,}/classes.txt}"
FAR_NSAMPLES="${FAR_NSAMPLES:-32}"
FAR_MAX_FAR="${FAR_MAX_FAR:-30}"
FAR_DIV_FAC="${FAR_DIV_FAC:-5}"
FAR_FRAC="${FAR_FRAC:-4}"
FAR_ACC_STOP="${FAR_ACC_STOP:-2.0}"

# ── FT fine-tune data + knobs (mirrors run_defence_ft.sh / defence_finetune.py) ──
CC3M_ROOT="${CC3M_ROOT:-../data/cc3m}"
CC3M_META="${CC3M_META:-}"
CC3M_IMAGES="${CC3M_IMAGES:-}"
FT_MAX_SAMPLES="${FT_MAX_SAMPLES:-0}"
FT_EPOCHS="${FT_EPOCHS:-1}"
FT_MAX_STEPS="${FT_MAX_STEPS:-2000}"
FT_LR="${FT_LR:-1e-5}"
FT_WD="${FT_WD:-0.1}"
FT_BATCH="${FT_BATCH:-64}"
FT_SCOPE="${FT_SCOPE:-full}"
FT_SEED="${FT_SEED:-42}"
FT_WORKERS="${FT_WORKERS:-4}"
EVAL_BATCH="${EVAL_BATCH:-64}"
SAVE_FINETUNED="${SAVE_FINETUNED:-0}"

OUT_ROOT_ABS="$(realpath -m "$OUT_ROOT")"

# ── logging ──
LOG_DIR="$OUT_ROOT_ABS/logs"
mkdir -p "$LOG_DIR"
GRID_LOG="$LOG_DIR/defences_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$GRID_LOG") 2>&1
echo "Defences start $(date '+%F %T')  PY=$PY"
echo "  victim : $MODEL × $DATASET   tasks: $TASKS   defences: $DEFENCES"
echo "  attack : target_cls=$TARGET_CLS trigger=$TRIG_SZ seed=$SEED rep=$REP section=$SECTION"
echo "  LP head: $LP_HEAD"

run_dir_for() {   # run_dir_for <task> <far|plain>
    local leaf="${1}_c${TARGET_CLS}_s${SEED}_r${REP}_${VARIANT}_t${TRIG_SZ}"
    [ "$2" = "far" ] && leaf="${leaf}_far"
    echo "$OUT_ROOT_ABS/$SECTION/$TAG/$DATASET/$leaf"
}

# ══════════════════════════════════════════════════════════════════════
#  PREREQUISITES
# ══════════════════════════════════════════════════════════════════════
if [ ! -f "$HERE/far_clip/far_clip.py" ]; then
    echo "ERROR: defences/far_clip/far_clip.py not found"; exit 1
fi
if [ ! -f "$LP_HEAD" ]; then
    echo "ERROR: LP head missing: $LP_HEAD"
    echo "       train it first: python linear_probe.py --model $MODEL --test-dataset $DATASET"
    exit 1
fi
case " $DEFENCES " in
  *" ft "*)
    if [ ! -d "$CC3M_ROOT" ]; then
        echo "ERROR: CC3M root missing: $CC3M_ROOT (FT clean data)"; exit 1
    fi ;;
esac

# ══════════════════════════════════════════════════════════════════════
#  ATTACK (shared primitive)
# ══════════════════════════════════════════════════════════════════════
run_attack() {
    # run_attack <task> <mode: plain|far> <out-dir>
    local task="$1" mode="$2" out_dir="$3"
    local extra
    if [ "$mode" = "far" ]; then
        extra=(--far-ckpt "$FAR_CKPT")
    else
        extra=(--save-attacked-model)     # FT needs model_state_attacked.pt
    fi
    echo "════════ ATTACK [$task/$mode]  $out_dir ════════"
    "$PY" experiment.py --model "$MODEL" --dataset "$DATASET" --task "$task" \
        --section "$SECTION" --variant "$VARIANT" --seed "$SEED" --rep "$REP" \
        --target-cls "$TARGET_CLS" --trigger-size "$TRIG_SZ" \
        --device "$DEVICE" --out-dir "$out_dir" "${extra[@]}"
}

# ══════════════════════════════════════════════════════════════════════
#  DEFENCE 1 — FT  (bfa attack → Eval → FT → Eval)
# ══════════════════════════════════════════════════════════════════════
case " $DEFENCES " in
*" ft "*)
for task in $TASKS; do
    run_dir="$(run_dir_for "$task" plain)"
    # 1. attack the undefended victim (also the FaR "no defence" baseline)
    if [ -f "$run_dir/bfa_results.json" ] && [ -f "$run_dir/model_state_attacked.pt" ]; then
        echo "[resume] attack already done: $run_dir"
    else
        run_attack "$task" plain "$run_dir"
    fi
    # 2+3. eval BEFORE → FT → eval AFTER (defence_finetune.py does both)
    if [ -f "$run_dir/defence_ft.json" ]; then
        echo "[resume] FT defence already done: $run_dir"
        continue
    fi
    echo "════════ FT DEFENCE [$task]  $run_dir ════════"
    extra=""
    [ "$SAVE_FINETUNED" = "1" ] && extra="--save-finetuned"
    "$PY" defence_finetune.py --run-dir "$run_dir" --task "$task" \
        --lp-head "$LP_HEAD" \
        --cc3m-root "$CC3M_ROOT" \
        ${CC3M_META:+--cc3m-meta "$CC3M_META"} \
        ${CC3M_IMAGES:+--cc3m-images "$CC3M_IMAGES"} \
        --ft-max-samples "$FT_MAX_SAMPLES" \
        --ft-epochs "$FT_EPOCHS" --ft-max-steps "$FT_MAX_STEPS" \
        --ft-lr "$FT_LR" --ft-wd "$FT_WD" --ft-batch "$FT_BATCH" \
        --ft-scope "$FT_SCOPE" --ft-seed "$FT_SEED" --ft-workers "$FT_WORKERS" \
        --eval-batch "$EVAL_BATCH" --device "$DEVICE" $extra 2>&1 \
        | tee "$run_dir/defence_ft.log"
done
;;
esac

# ══════════════════════════════════════════════════════════════════════
#  DEFENCE 2 — FaR  (FaR → eval → bfa attack → eval)
# ══════════════════════════════════════════════════════════════════════
case " $DEFENCES " in
*" far "*)
# 1. FaR on the clean victim → defended checkpoint (eval right after FaR =
#    the calib accuracy far_clip prints + saves in the _far_cfg.json)
if [ -f "$FAR_CKPT" ]; then
    echo "[resume] FaR defended victim already built: $FAR_CKPT"
else
    echo "════════ FaR DEFENCE — building defended victim ════════"
    "$PY" "$HERE/far_clip/far_clip.py" --model "$MODEL" \
        --data-root "$FAR_DATA_ROOT" --classes-file "$FAR_CLASSES" \
        --n-samples "$FAR_NSAMPLES" --max-far "$FAR_MAX_FAR" \
        --div-fac "$FAR_DIV_FAC" --frac-size "$FAR_FRAC" \
        --acc-drop-stop "$FAR_ACC_STOP" \
        --out-dir "$FAR_DIR" --out-name "$(basename "$FAR_CKPT")" \
        --device "$DEVICE" --seed "$SEED"
fi

# 2+3+4. attack the defended victim (eval after attack = bfa_results.json)
for task in $TASKS; do
    run_dir="$(run_dir_for "$task" far)"
    if [ -f "$run_dir/bfa_results.json" ]; then
        echo "[resume] FaR attack already done: $run_dir"
        continue
    fi
    run_attack "$task" far "$run_dir"
done
;;
esac

# ══════════════════════════════════════════════════════════════════════
#  SUMMARY (RESULTS.md §4 table)
# ══════════════════════════════════════════════════════════════════════
echo "════════ DEFENCE SUMMARY (RESULTS.md §4) ════════"
"$PY" - "$OUT_ROOT_ABS" "$SECTION" "$TAG" "$DATASET" "$TARGET_CLS" "$SEED" "$REP" "$VARIANT" "$TRIG_SZ" "$DEFENCES" "$TASKS" <<'EOF'
import json, os, sys
out_root, section, tag, dataset, tc, seed, rep, variant, trig, defences, tasks = sys.argv[1:]
base = os.path.join(out_root, section, tag, dataset)
leaf = f"{variant}_t{trig}"
do_ft, do_far = "ft" in defences.split(), "far" in defences.split()

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def pct(v, w):
    return f"{v:>{w-1}.2f}%" if isinstance(v, (int, float)) else f"{'-':>w}"

print(f"  {'defence':<8} {'task':<4} {'ASR (no defence)':>17} {'ASR (after)':>12} "
      f"{'CACC (no defence)':>17} {'CACC (after)':>12}")

for task in tasks.split():
    u_dir = os.path.join(base, f"{task}_c{tc}_s{seed}_r{rep}_{leaf}")
    f_dir = os.path.join(base, f"{task}_c{tc}_s{seed}_r{rep}_{leaf}_far")

    # ── FT row: defence_ft.json (before/after) ──
    if do_ft:
        ft = load(os.path.join(u_dir, "defence_ft.json"))
        if ft:
            b, a = ft["before"], ft["after"]
            print(f"  {'FT':<8} {task:<4} {pct(b['asr'],17)} {pct(a['asr'],12)} "
                  f"{pct(b['cacc'],17)} {pct(a['cacc'],12)}")
        else:
            print(f"  {'FT':<8} {task:<4}   (no defence_ft.json — run the FT flow)")

    # ── FaR row: attack metrics (undefended = no defence, defended = after) ──
    if do_far:
        fm = load(os.path.join(f_dir, "bfa_results.json")) or {}
        um = load(os.path.join(u_dir, "bfa_results.json")) or {}
        fmet, umet = fm.get("metrics", {}), um.get("metrics", {})
        if fmet:
            print(f"  {'FaR':<8} {task:<4} {pct(umet.get('asr_final'),17)} "
                  f"{pct(fmet.get('asr_final'),12)} {pct(umet.get('cacc_final'),17)} "
                  f"{pct(fmet.get('cacc_final'),12)}")
        else:
            print(f"  {'FaR':<8} {task:<4}   (no defended bfa_results.json — run the FaR flow)")
EOF

echo "Defences done $(date '+%F %T')"
echo "  aggregate tables: python aggregate.py   (runs.csv / summary.csv)"
