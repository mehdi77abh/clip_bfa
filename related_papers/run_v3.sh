#!/usr/bin/env bash
# run_v3.sh — loss comparison through the full attack + embedding geometry +
#              FT-defence survival.  ViT-L/14 × ImageNet1k, target c291,
#              seed 42.  NO visual arms.
#
# Arms (plain loss, no visual hinge):
#   1. cos
#   2. residual λ=0.5
#   3. residual λ=0.75
#   4. residual λ=1
#   5. residual λ=2
#
# Per arm: full bit-flip attack (Phases 0-4, --save-attacked-model) →
# geometry.json (phase1 + attacked) → FT defence (defence_finetune.py on
# CC3M → defence_ft.json).  Fast FT settings: 500 steps, 10k-image eval
# subset (paired before/after).  The pipeline's 49k final eval is SKIPPED
# (--skip-final-eval); the FT before-eval supplies the post-attack ASR.
#
# Usage (GPU box, vlm_attack env activated):
#   bash related_papers/run_v3.sh
#   PYTHON=/path/to/python bash related_papers/run_v3.sh
#   RESUME=0 bash related_papers/run_v3.sh           # rerun completed runs
#   NEG=lion_v1 CC3M_ROOT=/path/to/cc3m bash related_papers/run_v3.sh
#   FT_STEPS=1000 FT_EVAL_MAX=5000 bash related_papers/run_v3.sh
#
# FT notes: CC3M is provisioned by the maintainer (default <repo>/data/cc3m).
# If the root is missing the FT stage is skipped per run (attack not wasted).
# Dataset size does NOT shorten a step-capped run — FT_STEPS is the lever.
#
# Every run's full output is streamed to the console AND saved under
# v3_results/logs/<timestamp>/run_NN.log.  A summary table (attack +
# geometry + FT survival) is printed at the end.

set -uo pipefail

# memory hygiene: avoid CUDA allocator fragmentation on 15.5GB cards
# (the OOM's own suggestion — harmless otherwise)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
FT_BATCH="${FT_BATCH:-16}"                  # 24GB-safe full-scope (32 ~21GB, 64 >=40GB)
FT_STEPS="${FT_STEPS:-500}"                 # FT optimiser steps (runtime lever)
FT_EVAL_MAX="${FT_EVAL_MAX:-10000}"         # paired before/after eval subset (0 = full)
FT_EVAL_BATCH="${FT_EVAL_BATCH:-64}"        # eval batch (64 on 24GB)

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
COMMON="$COMMON --ft-batch $FT_BATCH --eval-batch $FT_EVAL_BATCH"
COMMON="$COMMON --ft-max-steps $FT_STEPS --eval-max-samples $FT_EVAL_MAX"
COMMON="$COMMON --skip-final-eval"
echo "FT budget: $FT_STEPS steps, eval n=$FT_EVAL_MAX (batch $FT_EVAL_BATCH), ft-batch $FT_BATCH"

# ---- the sweep -----------------------------------------------------
# one string of args per run; edit here to change the matrix.
runs=(
  # plain cos (no visual hinge)
  "--seed 42 --loss cos"
  # residual λ sweep (no visual hinge); every arm runs the FT defence
  "--seed 42 --loss residual --lambda-neg 0.5"
  "--seed 42 --loss residual --lambda-neg 0.75"
  "--seed 42 --loss residual --lambda-neg 1"
  "--seed 42 --loss residual --lambda-neg 2"
)

# All arms above run FT, so model_state_attacked.pt is always saved.
# SAVE_ARMS is retained for an attack-only matrix edit (a --skip-ft row).
#   SAVE_ARMS="2" bash related_papers/run_v3.sh
SAVE_ARMS="${SAVE_ARMS:-}"

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
  # optional per-arm attacked-model save (arm 3 always saves for the FT)
  run_args="$args"
  if [[ " $SAVE_ARMS " == *" $idx "* ]]; then
    run_args="$run_args --save-attacked-model"
  fi
  key=$(printf '%s' "$COMMON $run_args" | cksum | cut -d' ' -f1)   # full-command key

  if [[ "$RESUME" == "1" && -f "$DONE/$key.ok" ]]; then
    echo
    echo "=== [$idx/$total] $name: SKIP (done) — $run_args"
    skipped=$(( skipped + 1 ))
    continue
  fi

  echo
  echo "=== [$idx/$total] $name: $PY test_version3.py $run_args"
  echo "    log: $log"
  start=$SECONDS
  # shellcheck disable=SC2086
  "$PY" "$SCRIPT" $COMMON $run_args 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}          # python's exit code, not tee's
  dt=$(( SECONDS - start ))

  if [[ $rc -eq 0 ]]; then
    echo "--- $name OK (${dt}s)"
    printf '%s\n' "$run_args" > "$DONE/$key.ok"
  else
    echo "--- $name FAILED (exit $rc, ${dt}s) — see $log"
    failed+=("$name")
    printf 'rc=%s\n%s\n' "$rc" "$run_args" > "$DONE/$key.fail"
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
    ev = ft.get("eval") or {}
    skipped = bool(post.get("full_eval_skipped"))
    # For --skip-final-eval runs the pipeline's post_flip_test is the cheap
    # 200-image pool; the FT BEFORE eval is the real post-attack ASR.
    if skipped and fb.get("asr") is not None:
        asr_col, cacc_col = fb.get("asr"), fb.get("cacc")
    else:
        asr_col, cacc_col = post.get("asr"), post.get("cacc")
    surv = f"{100.0 * fa['asr'] / fb['asr']:.0f}%" \
        if fb.get("asr") and fa.get("asr") is not None else "-"
    rows.append((
        os.path.basename(d), m.get("flips_to_95"),
        asr_col, cacc_col, p1.get("standalone_asr"),
        atk.get("margin_class"), atk.get("manifold_delta"),
        fb.get("asr"), fa.get("asr"), surv, skipped, ev.get("n")))

if not rows:
    print("no bfa_results.json found under", root)
else:
    hdr = (f"{'run':<44} {'fl95':>4} {'ASR':>6} {'CACC':>6} {'stASR':>6} "
           f"{'m_cls':>7} {'Δman':>7} {'FTpre':>6} {'FTpost':>6} "
           f"{'surv':>5} {'evN':>6}")
    print(hdr)
    print("-" * len(hdr))
    any_skipped = False
    for r in rows:
        leaf, fl95, asr, cacc, stand, mc, dm, fpre, fpost, surv, skipped, evn = r
        any_skipped = any_skipped or skipped
        def fmt(v, w, nd=1, sign=False):
            if v is None:
                return " " * (w - 1) + "-"
            s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
            return f"{s:>{w}}"
        print(f"{leaf:<44} "
              f"{('-' if fl95 is None else fl95):>4} "
              f"{fmt(asr, 6)} {fmt(cacc, 6)} {fmt(stand, 6)} "
              f"{fmt(mc, 7, 4, True)} {fmt(dm, 7, 4, True)} "
              f"{fmt(fpre, 6)} {fmt(fpost, 6)} {surv:>5} "
              f"{('-' if evn is None else evn):>6}")
    if any_skipped:
        print("\n* ASR/CACC for --skip-final-eval runs = FT BEFORE eval "
              "(n = evN; '-' = old record without an eval.n). The pipeline "
              "final eval was skipped.")
PYEOF

echo
elapsed=$(( SECONDS - t0 ))
echo "Total: $total | Skipped (done): $skipped | Failed: ${#failed[@]} | Wall: ${elapsed}s"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed runs: ${failed[*]}"
  exit 1
fi
