#!/usr/bin/env bash
# run_v2_336.sh — Stage-1 hostile-cell matrix: ViT-L/14-336px, 24px trigger.
#
# Style: every run is one explicit string of test_version2.py args in `runs`
# below (arm x seed).  The fixed frame (model / dataset / target-cls /
# trigger-size / neg-list / --stop-after-phase1) lives in $COMMON and is
# applied to every run.  To add/remove a run, edit the array.
#
# Arms (Phase-1 only, one deterministic attempt each, no retries):
#   1. cos                --loss cos                         (control)
#   2. infonce            --loss infonce                     (InfoNCE+Lion_v1 =
#                          the exact vast.ai failing config / R1 recipe)
#   3. residual  L=0.15   --loss residual --lambda-neg 0.15
#   4. residual  L=0.25   --loss residual --lambda-neg 0.25
#   5. residual  L=0.40   --loss residual --lambda-neg 0.4
#   6. residual + visual  --loss residual --lambda-neg 0.25 --vis-lambda 1.0
#                          (BadCLIP D2 hinge; base L = 0.25)
#   7. residual + multi   --loss residual --lambda-neg 0.25 --target-mode multi
#
# Every arm pins --neg-list lion_v1 (the production list) via $COMMON and runs
# with --stop-after-phase1.
#
# Usage (on the GPU box, vlm_attack env):
#   bash related_papers/run_v2_336.sh
#   PYTHON=/path/to/python bash related_papers/run_v2_336.sh
#   RESUME=0 bash related_papers/run_v2_336.sh          # rerun completed runs
#
# Results: related_papers/v2_results/.../<leaf>/phase1_summary.json
# (leaf = zs_c291_s{seed}_r1_full_t24_{tag}).  Per-run logs land in
# v2_results/logs/<timestamp>/run_NN.log.  A summary table of every
# phase1_summary.json is printed at the end.

set -uo pipefail

# Use the python of the currently active conda env; override with: PYTHON=/path/to/python
PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/test_version2.py"

MODEL="${MODEL:-ViT-L/14-336px}"
TRIG="${TRIG:-24}"
TARGET_CLS="${TARGET_CLS:-291}"          # lion (ImageNet1k)
NEG="${NEG:-lion_v1}"
RESUME="${RESUME:-1}"

# ---- sanity checks -------------------------------------------------
PY_BIN="$(command -v "$PY")" || { echo "ERROR: '$PY' not found on PATH. Did you activate the env?" >&2; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "ERROR: script not found: $SCRIPT" >&2; exit 1; }

echo "Using python: $PY_BIN"
echo "Using script: $SCRIPT"

# ---- fixed frame: appended to every run ---------------------------
COMMON="--model $MODEL --dataset ImageNet1k --target-cls $TARGET_CLS"
COMMON="$COMMON --trigger-size $TRIG --neg-list $NEG --stop-after-phase1"

# ---- the sweep -----------------------------------------------------
# one string of args per run (arm x seed); edit here to change the matrix.
runs=(
  # 1. cos (control)
  "--seed 42 --loss cos"
  "--seed 43 --loss cos"
  # 2. infonce + Lion_v1
  "--seed 42 --loss infonce"
  "--seed 44 --loss infonce"
  # 3. residual, lambda-neg = 0.15
  "--seed 42 --loss residual --lambda-neg 0.15"
  "--seed 44 --loss residual --lambda-neg 0.15"
  # 4. residual, lambda-neg = 0.25
  "--seed 42 --loss residual --lambda-neg 0.25"
  "--seed 44 --loss residual --lambda-neg 0.25"
  # 5. residual, lambda-neg = 0.40
  "--seed 42 --loss residual --lambda-neg 0.4"
  "--seed 44 --loss residual --lambda-neg 0.4"
  # 6. residual + visual hinge (BadCLIP D2), base lambda-neg = 0.25
  "--seed 42 --loss residual --lambda-neg 0.25 --vis-lambda 1.0"
  "--seed 44 --loss residual --lambda-neg 0.25 --vis-lambda 1.0"
  # 7. residual + multi target, base lambda-neg = 0.25
  "--seed 42 --loss residual --lambda-neg 0.25 --target-mode multi"
  "--seed 44 --loss residual --lambda-neg 0.25 --target-mode multi"

g
)

LOG_ROOT="$HERE/v2_results/logs"
LOGDIR="$LOG_ROOT/$(date +%Y%m%d_%H%M%S)"
DONE="$LOG_ROOT/done"
mkdir -p "$LOGDIR" "$DONE"
echo "Logging to: $LOGDIR"

total=${#runs[@]}
failed=()
idx=0
skipped=0
t0=$SECONDS

for args in "${runs[@]}"; do
  idx=$(( idx + 1 ))
  name=$(printf "run_%02d" "$idx")
  log="$LOGDIR/${name}.log"
  key=$(printf '%s' "$args" | cksum | cut -d' ' -f1)   # stable per-args resume key

  if [[ "$RESUME" == "1" && -f "$DONE/$key.ok" ]]; then
    echo
    echo "=== [$idx/$total] $name SKIP (already done): $args"
    skipped=$(( skipped + 1 ))
    continue
  fi

  echo
  echo "=== [$idx/$total] $name: test_version2.py $args"
  start=$SECONDS

  # shellcheck disable=SC2086  # word splitting of $args/$COMMON is intentional
  "$PY" $SCRIPT $args $COMMON 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}          # python's exit code, not tee's

  dur=$(( SECONDS - start ))
  if [[ $status -eq 0 ]]; then
    echo "--- $name OK (${dur}s)"
    touch "$DONE/$key.ok"
  else
    echo "--- $name FAILED (exit $status, ${dur}s) — see $log"
    failed+=("$name: $args")
  fi
done

# ---- summary of every phase1_summary.json under v2_results ----------
echo
"$PY" - "$HERE" <<'PYEOF'
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

# ---- summary -------------------------------------------------------
echo
echo "==================== SUMMARY ===================="
echo "Total: $total | Passed: $(( total - ${#failed[@]} - skipped )) | Skipped (done): $skipped | Failed: ${#failed[@]}"
echo "Wall time: $(( SECONDS - t0 ))s   logs: $LOGDIR"
if (( ${#failed[@]} > 0 )); then
  printf '  %s\n' "${failed[@]}"
  exit 1
fi