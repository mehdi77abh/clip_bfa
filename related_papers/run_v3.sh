#!/usr/bin/env bash
# run_v3.sh — Residual λ sweep through the FULL attack + embedding geometry
#             + FT-defence survival.  ViT-L/14 × ImageNet1k, target c291.
#
# Arms (raw λ in the v2 squared-residual parameterisation, seed 42):
#   1. residual  λ=0.5    2. residual λ=0.75   3. residual λ=1.0
#   4. residual  λ=2.0    5. residual λ=1.0 + visual hinge (λv=1)
#
# Per arm: full bit-flip attack (Phases 0-5, --save-attacked-model) →
# geometry.json (phase1 + attacked) → defence_finetune.py on CC3M →
# defence_ft.json.  Results under related_papers/v3_results/.
#
# Usage (GPU box, vlm_attack env activated):
#   bash related_papers/run_v3.sh
#   PYTHON=/path/to/python bash related_papers/run_v3.sh
#   RESUME=0 bash related_papers/run_v3.sh           # rerun completed runs
#   NEG=lion_v1 CC3M_ROOT=/path/to/cc3m bash related_papers/run_v3.sh
#
# FT notes: CC3M is provisioned by the maintainer (default <repo>/data/cc3m).
# If the root is missing the FT stage is skipped per run (attack not wasted).
# FT_MAX_SAMPLES=0 (all pairs; the FT is step-bounded at 2000 steps).
#
# v3_results/logs/<timestamp>/run_NN.log.  A summary table (attack +
# geometry + FT survival) is printed at the end.

set -uo pipefail

# Use the python of the currently active conda env; override with: PYTHON=/path/to/python
PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/test_version3.py"

MODEL="${MODEL:-ViT-L/14}"
TRIG="${TRIG:-16}"
TARGET_CLS="${TARGET_CLS:-291}"          # lion (ImageNet1k)
NEG="${NEG:-lion_v2}"                    # matches the v2test residual arms
RESUME="${RESUME:-1}"
CC3M_ROOT="${CC3M_ROOT:-$HERE/../data/cc3m}"
FT_MAX_SAMPLES="${FT_MAX_SAMPLES:-0}"

# ---- sanity checks -------------------------------------------------
PY_BIN="$(command -v "$PY")" || { echo "ERROR: '$PY' not found on PATH. Did you activate the env?" >&2; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "ERROR: script not found: $SCRIPT" >&2; exit 1; }

echo "Using python: $PY_BIN"
echo "Using script: $SCRIPT"
echo "FT (CC3M): $CC3M_ROOT  (max-samples=$FT_MAX_SAMPLES)"
[[ -d "$CC3M_ROOT" ]] || echo "WARNING: CC3M root missing — FT stage will be skipped"

# ---- fixed frame: appended to every run ---------------------------
COMMON="--model $MODEL --dataset ImageNet1k --target-cls $TARGET_CLS"
COMMON="$COMMON --trigger-size $TRIG --neg-list $NEG"
COMMON="$COMMON --cc3m-root $CC3M_ROOT --ft-max-samples $FT_MAX_SAMPLES"

# ---- the sweep -----------------------------------------------------
# one string of args per run; edit here to change the matrix.
runs=(
  # residual repulsion, raw lambda sweep (seed 42)
  "--seed 42 --loss residual --lambda-neg 0.5"
  "--seed 42 --loss residual --lambda-neg 0.75"
  "--seed 42 --loss residual --lambda-neg 1"
  "--seed 42 --loss residual --lambda-neg 2"
  # residual + visual resistance hinge (BadCLIP D2), base lambda-neg = 1
  "--seed 42 --loss residual --lambda-neg 1 --vis-lambda 1"
)

LOG_ROOT="$HERE/v3_results/logs"
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
    echo "=== [$idx/$total] $name: SKIP (done) — $args"
    skipped=$(( skipped + 1 ))
    continue
  fi

  echo
  echo "=== [$idx/$total] $name: $PY test_version3.py $args"
  echo "    log: $log"
  start=$SECONDS
  # shellcheck disable=SC2086
  "$PY" "$SCRIPT" $COMMON $args > "$log" 2>&1
  rc=$?
  dt=$(( SECONDS - start ))

  if [[ $rc -eq 0 ]]; then
    echo "--- $name OK (${dt}s)"
    printf '%s\n' "$args" > "$DONE/$key.ok"
    tail -n 3 "$log"
  else
    echo "--- $name FAILED (rc=$rc, ${dt}s) — last lines:"
    tail -n 15 "$log"
    failed+=("$name")
    printf 'rc=%s\n%s\n' "$rc" "$args" > "$DONE/$key.fail"
  fi
done

# ---- summary: attack + geometry + FT survival -----------------------
echo
echo "==================== SUMMARY (v3) ===================="
"$PY" - "$HERE" <<'PYEOF'
import glob, json, os, sys

root = os.path.join(sys.argv[1], "v3_results")
rows = []
for p in sorted(glob.glob(os.path.join(root, "**", "bfa_results.json"),
                          recursive=True)):
    d = os.path.dirname(p)
    try:
        with open(p) as f:
            bfa = json.load(f)
    except Exception:
        continue
    def _load(name):
        q = os.path.join(d, name)
        if os.path.isfile(q):
            try:
                with open(q) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    geo, ft = _load("geometry.json"), _load("defence_ft.json")
    m = bfa.get("metrics") or {}
    post = (bfa.get("results") or {}).get("post_flip_test") or {}
    p1, atk = geo.get("phase1") or {}, geo.get("attacked") or {}
    fb, fa = ft.get("before") or {}, ft.get("after") or {}
    surv = f"{100.0 * fa['asr'] / fb['asr']:.0f}%" \
        if fb.get("asr") and fa.get("asr") is not None else "-"
    rows.append((
        os.path.basename(d), m.get("flips_to_95"),
        post.get("asr"), post.get("cacc"), p1.get("standalone_asr"),
        atk.get("margin_class"), atk.get("manifold_delta"),
        fb.get("asr"), fa.get("asr"), surv))

if not rows:
    print("no bfa_results.json found under", root)
else:
    hdr = (f"{'run':<44} {'fl95':>4} {'ASR':>6} {'CACC':>6} {'stASR':>6} "
           f"{'m_cls':>7} {'Δman':>7} {'FTpre':>6} {'FTpost':>6} {'surv':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        leaf, fl95, asr, cacc, stand, mc, dm, fpre, fpost, surv = r
        def fmt(v, w, nd=1, sign=False):
            if v is None:
                return " " * (w - 1) + "-"
            s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
            return f"{s:>{w}}"
        print(f"{leaf:<44} "
              f"{('-' if fl95 is None else fl95):>4} "
              f"{fmt(asr, 6)} {fmt(cacc, 6)} {fmt(stand, 6)} "
              f"{fmt(mc, 7, 4, True)} {fmt(dm, 7, 4, True)} "
              f"{fmt(fpre, 6)} {fmt(fpost, 6)} {surv:>5}")
PYEOF

echo
elapsed=$(( SECONDS - t0 ))
echo "Total: $total | Skipped (done): $skipped | Failed: ${#failed[@]} | Wall: ${elapsed}s"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed runs: ${failed[*]}"
  exit 1
fi
