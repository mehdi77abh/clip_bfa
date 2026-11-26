#!/usr/bin/env bash
# run_sweep.sh — run test_version2.py configurations one by one
set -uo pipefail

# Use the python of the currently active conda env; override with: PYTHON=/path/to/python ./run_sweep.sh
PY="${PYTHON:-python}"

SCRIPT=related_papers/test_version2.py

# ---- sanity checks -------------------------------------------------
PY_BIN="$(command -v "$PY")" || { echo "ERROR: '$PY' not found on PATH. Did you activate the env?" >&2; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "ERROR: script not found: $SCRIPT (run from the clip_bfa/ dir)" >&2; exit 1; }

echo "Using python: $PY_BIN"
echo "Using script: $SCRIPT"
# ---- the sweep -----------------------------------------------------
runs=(
  "--loss cos --stop-after-phase1"
  "--loss infonce --neg-list lion_v1 --stop-after-phase1"
  "--loss residual --lambda-neg 0.15 --stop-after-phase1"
  "--loss residual --lambda-neg 0.25 --stop-after-phase1"
  "--loss residual --lambda-neg 0.4  --stop-after-phase1"
  "--loss residual --vis-lambda 1.0  --stop-after-phase1"
  "--loss cos     --vis-lambda 1.0  --stop-after-phase1"
  "--loss residual --target-mode multi --stop-after-phase1"
  "--loss cos --target-mode single --target-template 0 --stop-after-phase1"
  "--loss residual --carrier-rule pps --stop-after-phase1"
  "--loss infonce --select margin --stop-after-phase1"
)

LOGDIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
echo "Logging to: $LOGDIR"

total=${#runs[@]}
failed=()
idx=0

for args in "${runs[@]}"; do
  idx=$((idx + 1))
  name=$(printf "run_%02d" "$idx")
  log="$LOGDIR/${name}.log"

  echo
  echo "=== [$idx/$total] $name: $SCRIPT $args"
  start=$SECONDS

  # shellcheck disable=SC2086  # word splitting of $args is intentional
  "$PY" $SCRIPT $args 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}          # python's exit code, not tee's

  dur=$(( SECONDS - start ))
  if [[ $status -eq 0 ]]; then
    echo "--- $name OK (${dur}s)"
  else
    echo "--- $name FAILED (exit $status, ${dur}s) — see $log"
    failed+=("$name: $args")
  fi
done

# ---- summary -------------------------------------------------------
echo
echo "==================== SUMMARY ===================="
echo "Total: $total | Passed: $(( total - ${#failed[@]} )) | Failed: ${#failed[@]}"
if (( ${#failed[@]} > 0 )); then
  printf '  %s\n' "${failed[@]}"
  exit 1
fi