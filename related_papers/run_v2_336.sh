#!/usr/bin/env bash
# run_v2_336.sh — Stage-1 hostile-cell matrix: ViT-L/14-336px, 24px trigger.
#
# Runs exactly the agreed arms through test_version2.py (Phase-1 only):
#
#   1. cos                --loss cos                         (control)
#   2. infonce            --loss infonce   --neg-list lion_v1 (InfoNCE+Lion_v1
#                          = the exact vast.ai failing config / R1 recipe)
#   3. residual  λ=0.15   --loss residual --lambda-neg 0.15
#   4. residual  λ=0.25   --loss residual --lambda-neg 0.25
#   5. residual  λ=0.40   --loss residual --lambda-neg 0.40
#   6. residual + visual  --loss residual --lambda-neg 0.25 --vis-lambda 1.0
#                          (BadCLIP D2 hinge; base λ = 0.25 — edit if the
#                          sweep says otherwise)
#   7. residual + multi+  --loss residual --lambda-neg 0.25 --target-mode multi
#                          (base λ = 0.25)
#
# Every arm pins --neg-list lion_v1 (the production list).  All arms run with
# --stop-after-phase1 and one deterministic Phase-1 attempt (no retries).
#
# Usage (on the GPU box, vlm_attack env):
#   bash related_papers/run_v2_336.sh
#   SEEDS="42" bash related_papers/run_v2_336.sh        # quick first look
#   PY=/path/to/python bash related_papers/run_v2_336.sh
#   RESUME=0 bash related_papers/run_v2_336.sh          # rerun completed arms
#
# Results: related_papers/v2_results/.../<leaf>/phase1_summary.json
# (leaf = zs_c291_s{seed}_r1_full_t24_{tag}).  A summary table of every
# phase1_summary.json is printed at the end.

set -u

PY="${PY:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-ViT-L/14-336px}"
TRIG="${TRIG:-24}"
TARGET_CLS="${TARGET_CLS:-291}"          # lion (ImageNet1k)
NEG="${NEG:-lion_v1}"
SEEDS="${SEEDS:-42 43 44}"
RESUME="${RESUME:-1}"

LOG_DIR="$HERE/v2_results/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_v2_336_$(date +%Y%m%d_%H%M%S).log"

echo "vlm-transfer v2 — hostile cell $(date '+%F %T')  PY=$PY" | tee "$LOG"
echo "  MODEL=$MODEL  TRIG=${TRIG}px  target=c$TARGET_CLS  NEG=$NEG" | tee -a "$LOG"
echo "  SEEDS=[$SEEDS]  RESUME=$RESUME" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# is_done <tag> <seed>  →  0 if phase1_summary.json exists for that run leaf.
# NOTE: <tag> must equal v2_tag() in test_version2.py for the arm's flags.
is_done() {
  "$PY" - "$HERE" "$1" "$2" "$TRIG" <<'PYEOF'
import glob, os, sys
here, tag, seed, trig = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
pat = os.path.join(here, "v2_results", "**",
                   f"*_s{seed}_r*_full_t{trig}_{tag}", "phase1_summary.json")
sys.exit(0 if glob.glob(pat, recursive=True) else 1)
PYEOF
}

# run_arm <tag> <flags...>  — loops over $SEEDS.
run_arm() {
  local tag="$1"; shift
  for seed in $SEEDS; do
    if [ "$RESUME" = "1" ] && is_done "$tag" "$seed"; then
      echo "[skip] $tag  s$seed  (phase1_summary.json exists)" | tee -a "$LOG"
      continue
    fi
    echo "[run ] $tag  s$seed" | tee -a "$LOG"
    "$PY" "$HERE/test_version2.py" \
        --model "$MODEL" --dataset ImageNet1k --target-cls "$TARGET_CLS" \
        --trigger-size "$TRIG" --seed "$seed" --neg-list "$NEG" \
        --stop-after-phase1 "$@" >> "$LOG" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
      echo "[FAIL] $tag  s$seed  rc=$rc  (see $LOG)" | tee -a "$LOG"
    else
      echo "[done] $tag  s$seed" | tee -a "$LOG"
    fi
  done
}

# ── the agreed arms (tag must match v2_tag(); base λ = 0.25 for combined) ──
run_arm v2_cos                    --loss cos
run_arm v2_infonce                --loss infonce
run_arm v2_residual_lneg0.15      --loss residual --lambda-neg 0.15
run_arm v2_residual_lneg0.25      --loss residual --lambda-neg 0.25
run_arm v2_residual_lneg0.4       --loss residual --lambda-neg 0.40
run_arm v2_residual_vis1_lneg0.25 --loss residual --lambda-neg 0.25 --vis-lambda 1.0
run_arm v2_residual_tgtmulti_lneg0.25 \
                                  --loss residual --lambda-neg 0.25 --target-mode multi

# ── summary of every phase1_summary.json under v2_results ──
echo "" | tee -a "$LOG"
"$PY" - "$HERE" <<'PYEOF' | tee -a "$LOG"
import glob, json, os, sys
here = sys.argv[1]
rows = []
for p in glob.glob(os.path.join(here, "v2_results", "**", "phase1_summary.json"),
                   recursive=True):
    try:
        with open(p) as f:
            rows.append(json.load(f))
    except Exception:
        continue
if not rows:
    print("no phase1_summary.json found yet")
else:
    rows.sort(key=lambda d: (d.get("model", ""), d.get("run_tag", ""),
                             d.get("seed", 0)))
    hdr = (f"{'arm':<32} {'seed':>4} {'model':<15} {'ASR%':>6} {'CACC%':>6} "
           f"{'cos_t':>7} {'m_pseudo':>9} {'m_class':>8}")
    print(hdr)
    print("-" * len(hdr))
    for d in rows:
        print(f"{d.get('run_tag', '?'):<32} {d.get('seed', 0):>4} "
              f"{d.get('model', '?'):<15} "
              f"{d.get('standalone_asr', 0):>6.1f} "
              f"{d.get('standalone_cacc', 0):>6.1f} "
              f"{d.get('final_cos_t', 0):>7.4f} "
              f"{d.get('final_margin_pseudo', 0):>+9.4f} "
              f"{d.get('final_margin_class', 0):>+8.4f}")
PYEOF

echo "" | tee -a "$LOG"
echo "log: $LOG" | tee -a "$LOG"
