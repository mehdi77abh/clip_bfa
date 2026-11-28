#!/usr/bin/env bash
# run_baselines.sh — FULL baseline grid (TrojViT / ProFlip / Flip-S).
#
# One literal `runs=(...)` list holds every running item (simple style, like
# run_validation.sh).  Each entry is a self-contained arg string prefixed by the
# dispatch token (trojvit | proflip | flips); the loop calls attack_<token>.py.
#
# Per cell (model x dataset) there are 5 runs:
#   trojvit v1  --trojvit-loss ce    (faithful)   -> out s42_ce
#   trojvit v2  --trojvit-loss cos   (cosine)     -> out s42_cos
#   proflip v1  --fitness mse_sni    (faithful)   -> out s42_mse_sni
#   proflip v2  --fitness cos        (cosine)     -> out s42_cos
#   flips   v1  --flip-surface encoder            -> out s42   (no cosine variant)
# Grid = 3 models x 3 datasets x 5 = 45 runs, seed 42.
#
# Usage (GPU box, vlm_attack env activated):
#   bash run_baselines.sh
#   PYTHON=/path/to/python bash run_baselines.sh
#   RESUME=0 bash run_baselines.sh                 # rerun completed runs
#   DEVICE=cuda bash run_baselines.sh
#
# RESUME=1 (default) skips any run whose done-key exists under results/logs/done.
# Every run streams to the console AND results/logs/<timestamp>/<name>.log.
# A summary table + results/summary.csv (one row per run) is written at the end.

set -uo pipefail

# memory hygiene: avoid CUDA allocator fragmentation on big cards
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${PYTHON:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPLIT="$HERE/data_split.json"
DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-1}"
EVAL_N="${EVAL_N:-2000}"  # 2000 = 1000 eval + 1000 eval_extra; 0 = frozen 1000

# infra flags appended to every run (kept out of the runs list to stay clean)
COMMON="--data-split $SPLIT --device $DEVICE"

# ---- centralised GPU cleanup between runs ---------------------------
# Each attack/eval is its own Python process, so its CUDA context is released
# at exit.  This drops Python refs, flushes the allocator cache and prints the
# free device memory before the next run; methods do NOT call empty_cache.
gpu_cleanup() {   # optional $1 = log file to append to
  local line
  line=$("$PY" - <<'PYEOF'
import gc
import torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    print(f"[gpu] free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
else:
    print("[gpu] cuda not available")
PYEOF
) || line="[gpu] cleanup skipped"
  echo "$line"
  if [[ -n "${1:-}" && -f "${1}" ]]; then
    echo "    $line" >> "$1"
  fi
}

# ---- sanity checks -------------------------------------------------
PY_BIN="$(command -v "$PY")" || { echo "ERROR: '$PY' not found on PATH. Activate the env?" >&2; exit 1; }
[[ -f "$SPLIT" ]] || { echo "ERROR: missing data split: $SPLIT" >&2; exit 1; }
for m in trojvit proflip flips; do
  [[ -f "$HERE/attack_$m.py" ]] || { echo "ERROR: missing attack_$m.py" >&2; exit 1; }
done
echo "Using python : $PY_BIN"
echo "Device       : $DEVICE"
echo "Eval images  : ${EVAL_N:-2000} (default 2000; 0/1000 = frozen 1000-image eval)"
cd "$HERE"    # relative --out paths in the runs list resolve against $HERE

# ---- the runs list: every running item, literal, simple style -------
runs=(
  # ViT-L/14 x Caltech101  (trojvit ce/cos are done -> EXTRA_RUNS, not re-run)
  #"proflip --model ViT-L/14 --dataset Caltech101 --seed 42 --target-cls 97 --out results/proflip/ViT-L14/Caltech101/s42_mse_sni --fitness mse_sni --clean-ce-weight 1.0 --sni-method jsma --cbs-select-once --cbs-accept-always --cbs-top-elems 1 --grid-k 20 --no-monotone --q-mode abs"
  #  "proflip --model ViT-L/14 --dataset Caltech101 --seed 42 --target-cls 97 --out results/proflip/ViT-L14/Caltech101/s42_cos --fitness cos --clean-ce-weight 1.0"
  #"flips --model ViT-L/14 --dataset Caltech101 --seed 42 --target-cls 97 --out results/flips/ViT-L14/Caltech101/s42 --flip-surface encoder --verify-n 32 --rerank-every 5 --max-flips 50 --asr-target 0"
  # ViT-L/14 x ImageNet1k
  "proflip --model ViT-L/14 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/proflip/ViT-L14/ImageNet1k/s42_mse_sni --fitness mse_sni --clean-ce-weight 1.0 --sni-method jsma --cbs-select-once --cbs-accept-always --cbs-top-elems 1 --grid-k 20 --no-monotone --q-mode abs"
  "proflip --model ViT-L/14 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/proflip/ViT-L14/ImageNet1k/s42_cos --fitness cos --clean-ce-weight 1.0"
  #"flips --model ViT-L/14 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/flips/ViT-L14/ImageNet1k/s42 --flip-surface encoder --verify-n 32 --rerank-every 5 --max-flips 50 --asr-target 0 --save-model-output"
  # ViT-L/14 x Food101
  "trojvit --model ViT-L/14 --dataset Food101 --seed 42 --target-cls 76 --out results/trojvit/ViT-L14/Food101/s42_ce --trojvit-loss ce --p2-rows 1 --p2-clean-weight 0.5"
  "trojvit --model ViT-L/14 --dataset Food101 --seed 42 --target-cls 76 --out results/trojvit/ViT-L14/Food101/s42_cos --trojvit-loss cos --p2-rows 1 --p2-clean-weight 0.5"
  "proflip --model ViT-L/14 --dataset Food101 --seed 42 --target-cls 76 --out results/proflip/ViT-L14/Food101/s42_mse_sni --fitness mse_sni --clean-ce-weight 1.0 --sni-method jsma --cbs-select-once --cbs-accept-always --cbs-top-elems 1 --grid-k 20 --no-monotone --q-mode abs"
  "proflip --model ViT-L/14 --dataset Food101 --seed 42 --target-cls 76 --out results/proflip/ViT-L14/Food101/s42_cos --fitness cos --clean-ce-weight 1.0"
  #"flips --model ViT-L/14 --dataset Food101 --seed 42 --target-cls 76 --out results/flips/ViT-L14/Food101/s42 --flip-surface encoder --verify-n 32 --rerank-every 5 --max-flips 50 --asr-target 0"

  # ViT-B/16 x Caltech101
  "trojvit --model ViT-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/trojvit/ViT-B16/Caltech101/s42_ce --trojvit-loss ce --p2-rows 1 --p2-clean-weight 0.5"
  "trojvit --model ViT-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/trojvit/ViT-B16/Caltech101/s42_cos --trojvit-loss cos --p2-rows 1 --p2-clean-weight 0.5"
  "proflip --model ViT-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/proflip/ViT-B16/Caltech101/s42_mse_sni --fitness mse_sni --clean-ce-weight 1.0 --sni-method jsma --cbs-select-once --cbs-accept-always --cbs-top-elems 1 --grid-k 20 --no-monotone --q-mode abs"
  "proflip --model ViT-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/proflip/ViT-B16/Caltech101/s42_cos --fitness cos --clean-ce-weight 1.0"
  #"flips --model ViT-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/flips/ViT-B16/Caltech101/s42 --flip-surface encoder --verify-n 32 --rerank-every 5 --max-flips 50 --asr-target 0"
  # ViT-B/16 x ImageNet1k
  "trojvit --model ViT-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/trojvit/ViT-B16/ImageNet1k/s42_ce --trojvit-loss ce --p2-rows 1 --p2-clean-weight 0.5"
  "trojvit --model ViT-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/trojvit/ViT-B16/ImageNet1k/s42_cos --trojvit-loss cos --p2-rows 1 --p2-clean-weight 0.5"
  "proflip --model ViT-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/proflip/ViT-B16/ImageNet1k/s42_mse_sni --fitness mse_sni --clean-ce-weight 1.0 --sni-method jsma --cbs-select-once --cbs-accept-always --cbs-top-elems 1 --grid-k 20 --no-monotone --q-mode abs"
  "proflip --model ViT-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/proflip/ViT-B16/ImageNet1k/s42_cos --fitness cos --clean-ce-weight 1.0"
  #"flips --model ViT-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/flips/ViT-B16/ImageNet1k/s42 --flip-surface encoder --verify-n 32 --rerank-every 5 --max-flips 50 --asr-target 0"
  # ViT-B/16 x Food101
  "trojvit --model ViT-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/trojvit/ViT-B16/Food101/s42_ce --trojvit-loss ce --p2-rows 1 --p2-clean-weight 0.5"
  "trojvit --model ViT-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/trojvit/ViT-B16/Food101/s42_cos --trojvit-loss cos --p2-rows 1 --p2-clean-weight 0.5"
  "proflip --model ViT-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/proflip/ViT-B16/Food101/s42_mse_sni --fitness mse_sni --clean-ce-weight 1.0 --sni-method jsma --cbs-select-once --cbs-accept-always --cbs-top-elems 1 --grid-k 20 --no-monotone --q-mode abs"
  "proflip --model ViT-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/proflip/ViT-B16/Food101/s42_cos --fitness cos --clean-ce-weight 1.0"
  #"flips --model ViT-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/flips/ViT-B16/Food101/s42 --flip-surface encoder --verify-n 32 --rerank-every 5 --max-flips 50 --asr-target 0"
 # SigLIP-B/16 x Caltech101
  "trojvit --model SigLIP-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/trojvit/SigLIP-B16/Caltech101/s42_ce --trojvit-loss ce --p2-rows 1 --p2-clean-weight 0.5"
  "trojvit --model SigLIP-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/trojvit/SigLIP-B16/Caltech101/s42_cos --trojvit-loss cos --p2-rows 1 --p2-clean-weight 0.5"
#  "proflip --model SigLIP-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/proflip/SigLIP-B16/Caltech101/s42_mse_sni --fitness mse_sni --clean-ce-weight 1.0 --sni-method jsma --cbs-select-once --cbs-accept-always --cbs-top-elems 1 --grid-k 20 --no-monotone --q-mode abs"
#  "proflip --model SigLIP-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/proflip/SigLIP-B16/Caltech101/s42_cos --fitness cos --clean-ce-weight 1.0"
  #"flips --model SigLIP-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/flips/SigLIP-B16/Caltech101/s42 --flip-surface encoder --verify-n 32 --rerank-every 5 --max-flips 50 --asr-target 0"
  # SigLIP-B/16 x ImageNet1k
  "trojvit --model SigLIP-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/trojvit/SigLIP-B16/ImageNet1k/s42_ce --trojvit-loss ce --p2-rows 1 --p2-clean-weight 0.5"
  "trojvit --model SigLIP-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/trojvit/SigLIP-B16/ImageNet1k/s42_cos --trojvit-loss cos --p2-rows 1 --p2-clean-weight 0.5"
  "proflip --model SigLIP-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/proflip/SigLIP-B16/ImageNet1k/s42_mse_sni --fitness mse_sni --clean-ce-weight 1.0 --sni-method jsma --cbs-select-once --cbs-accept-always --cbs-top-elems 1 --grid-k 20 --no-monotone --q-mode abs"
  "proflip --model SigLIP-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/proflip/SigLIP-B16/ImageNet1k/s42_cos --fitness cos --clean-ce-weight 1.0"
  #"flips --model SigLIP-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/flips/SigLIP-B16/ImageNet1k/s42 --flip-surface encoder --verify-n 32 --rerank-every 5 --max-flips 50 --asr-target 0"
  # SigLIP-B/16 x Food101
  "trojvit --model SigLIP-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/trojvit/SigLIP-B16/Food101/s42_ce --trojvit-loss ce --p2-rows 1 --p2-clean-weight 0.5"
  "trojvit --model SigLIP-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/trojvit/SigLIP-B16/Food101/s42_cos --trojvit-loss cos --p2-rows 1 --p2-clean-weight 0.5"
  "proflip --model SigLIP-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/proflip/SigLIP-B16/Food101/s42_mse_sni --fitness mse_sni --clean-ce-weight 1.0 --sni-method jsma --cbs-select-once --cbs-accept-always --cbs-top-elems 1 --grid-k 20 --no-monotone --q-mode abs"
  "proflip --model SigLIP-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/proflip/SigLIP-B16/Food101/s42_cos --fitness cos --clean-ce-weight 1.0"
  #"flips --model SigLIP-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/flips/SigLIP-B16/Food101/s42 --flip-surface encoder --verify-n 32 --rerank-every 5 --max-flips 50 --asr-target 0"
)

# Runs completed in an earlier session (their logs + attack_meta.json were
# pulled into results/).  They are NOT part of the sweep and are never
# re-run, but they ARE included in runs.csv / summary.csv (see EXPECTED).
EXTRA_RUNS=(
  "results/trojvit/ViT-L14/Caltech101/s42_ce"
  "results/trojvit/ViT-L14/Caltech101/s42_cos"
)

LOG_ROOT="$HERE/results/logs"
LOGDIR="$LOG_ROOT/$(date +%Y%m%d_%H%M%S)"
DONE="$LOG_ROOT/done"
mkdir -p "$LOGDIR" "$DONE"
echo "Logging to   : $LOGDIR"

total=${#runs[@]}
failed=()
idx=0
skipped=0
t0=$SECONDS

# print the held-out eval result (eval.json) into the run log + console
print_final_eval() {   # $1 = eval.json path
  "$PY" - "$1" <<'PYEOF'
import json, sys
try:
    ev = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
def f(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"
print(f"  FINAL ASR      : {f(ev.get('asr'))}  (eval split n={ev.get('n_eval', '?')})")
print(f"  FINAL CACC     : {f(ev.get('cacc'))}")
print(f"  clean->target  : {f(ev.get('clean_as_target'))}")
PYEOF
}

# ---- extras: eval-only for pulled runs that are no longer in the sweep ----
# If their attack_state.pt is still present (e.g. left on the GPU box) but
# eval.json is missing, evaluate them here — no attack is re-run.
for out in "${EXTRA_RUNS[@]}"; do
  [[ -f "$out/eval.json" || ! -f "$out/attack_state.pt" ]] && continue
  echo
  echo "=== [extra] $out: attack done, running evaluation only"
  elog="$LOGDIR/extra_$(printf '%s' "$out" | sed 's#^results/##; s#/#_#g').log"
  # shellcheck disable=SC2086
  "$PY" "$HERE/evaluate.py" --state "$out/attack_state.pt" $COMMON \
      --out "$out/eval.json" --eval-n "$EVAL_N" 2>&1 | tee "$elog"
  erc=${PIPESTATUS[0]}
  if [[ $erc -eq 0 && -f "$out/eval.json" ]]; then
    print_final_eval "$out/eval.json" | tee -a "$elog"
    rm -f "$out/attack_state.pt"   # keep only meta + eval
  else
    echo "--- [extra] $out: evaluation FAILED (exit $erc) — see $elog"
  fi
  gpu_cleanup "$elog"
done

for entry in "${runs[@]}"; do
  idx=$(( idx + 1 ))
  method="${entry%% *}"          # dispatch token: trojvit | proflip | flips
  args="${entry#* }"             # the literal arg string
  out=$(printf '%s' "$args" | grep -oE 'results/[^ ]+' | head -1)   # run dir
  # readable slug from the --out dir: e.g. trojvit_ViT-B16_ImageNet1k_s42_ce
  slug=$(printf '%s' "$out" | sed 's#^results/##' | tr '/' '_')
  name=$(printf "%02d_%s" "$idx" "$slug")
  log="$LOGDIR/${name}.log"
  key=$(printf '%s' "$entry" | cksum | cut -d' ' -f1)   # stable per-run resume key

  # The attack scripts write attack_state.pt + attack_meta.json; eval.json is
  # written by evaluate.py.  A run is DONE only when BOTH exist, otherwise the
  # summary block (which reads eval.json) would silently miss it.
  if [[ "$RESUME" == "1" && -f "$DONE/$key.ok" ]]; then
    if [[ -f "$out/eval.json" ]]; then
      echo
      echo "=== [$idx/$total] $name: SKIP (done)"
      skipped=$(( skipped + 1 ))
      continue
    fi
    # attack finished in an earlier session but eval.json is missing -> eval only
    echo
    echo "=== [$idx/$total] $name: attack done, running evaluation only"
    # shellcheck disable=SC2086
    "$PY" "$HERE/evaluate.py" --state "$out/attack_state.pt" $COMMON \
        --out "$out/eval.json" --eval-n "$EVAL_N" 2>&1 | tee "$log"
    erc=${PIPESTATUS[0]}
    if [[ $erc -eq 0 && -f "$out/eval.json" ]]; then
      print_final_eval "$out/eval.json" | tee -a "$log"
      echo "--- $name OK (eval only)"
      skipped=$(( skipped + 1 ))
      rm -f "$out/attack_state.pt"   # keep only meta + eval
    else
      echo "--- $name FAILED (evaluate exit $erc) — see $log"
      # if the saved state is gone/corrupt, clear .ok so resume re-runs the attack
      [[ -f "$out/attack_state.pt" ]] || rm -f "$DONE/$key.ok"
      failed+=("$name")
      printf 'rc=%s\n%s\n' "$erc" "$entry" > "$DONE/$key.fail"
    fi
    gpu_cleanup "$log"
    continue
  fi

  echo
  echo "=== [$idx/$total] $name  ->  attack_${method}.py"
  echo "    $args"
  start=$SECONDS
  # shellcheck disable=SC2086
  "$PY" "$HERE/attack_${method}.py" $COMMON $args 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}          # python's exit code, not tee's

  if [[ $rc -eq 0 ]]; then
    echo "--- $name attack OK — evaluating (evaluate.py)"
    # shellcheck disable=SC2086
    "$PY" "$HERE/evaluate.py" --state "$out/attack_state.pt" $COMMON \
        --out "$out/eval.json" --eval-n "$EVAL_N" 2>&1 | tee -a "$log"
    erc=${PIPESTATUS[0]}
    dt=$(( SECONDS - start ))
    if [[ $erc -eq 0 && -f "$out/eval.json" ]]; then
      print_final_eval "$out/eval.json" | tee -a "$log"
      echo "--- $name OK (${dt}s)"
      printf '%s\n' "$entry" > "$DONE/$key.ok"
      # results keep only attack_meta.json + eval.json (no large state)
      rm -f "$out/attack_state.pt"
      echo "    (eval.json + attack_meta.json kept; attack_state.pt removed)"
    else
      echo "--- $name FAILED (evaluate exit $erc, ${dt}s) — see $log"
      rm -f "$out/attack_state.pt"   # no .ok -> the attack will re-run
      failed+=("$name")
      printf 'rc=%s\n%s\n' "$erc" "$entry" > "$DONE/$key.fail"
    fi
  else
    dt=$(( SECONDS - start ))
    echo "--- $name FAILED (attack exit $rc, ${dt}s) — see $log"
    failed+=("$name")
    printf 'rc=%s\n%s\n' "$rc" "$entry" > "$DONE/$key.fail"
  fi
  gpu_cleanup "$log"
done

# ---- summary: scoped to the runs in the list -> results/{runs,summary}.csv ----
echo
echo "==================== SUMMARY (baselines) ===================="
# Scoping key: the expected run dirs (relative to results/) = the runs list
# above + EXTRA_RUNS.  Rows are emitted ONLY for these, so stale / backup /
# non-grid run dirs can never leak into the CSVs.
EXPECTED="$HERE/results/.expected_runs"
{
  printf '%s\n' "${runs[@]}"
  printf '%s\n' "${EXTRA_RUNS[@]}"
} | grep -oE 'results/[^ ]+' | sed 's#^results/##' | sort -u > "$EXPECTED"
"$PY" - "$HERE" "$EXPECTED" <<'PYEOF'
import csv, glob, json, os, re, sys
from collections import Counter

here, expected_file = sys.argv[1], sys.argv[2]
with open(expected_file) as f:
    expected = {ln.strip() for ln in f if ln.strip()}


def tar_from_log(run_dir):
    """Early TrojViT attack_meta.json has no `tar`; recover it from the
    pulled run log (`  TAR            : 3.52%  (9x14px)`)."""
    slug = run_dir.replace("/", "_")
    pat = os.path.join(here, "results", "logs", "*", f"*_{slug}.log")
    for lp in reversed(sorted(glob.glob(pat))):
        try:
            txt = open(lp, errors="ignore").read()
        except Exception:
            continue
        m = re.search(r"TAR\s*:\s*([0-9.]+)\s*%", txt)
        if m:
            return float(m.group(1))
    return ""


def fnum(x, w, p=3):
    return (f"{x:>{w}.{p}f}" if isinstance(x, (int, float))
            else f"{'-':>{w}s}")


detail_rows, key_rows = [], []
for run_dir in sorted(expected):
    d = os.path.join(here, "results", run_dir)
    ep = os.path.join(d, "eval.json")
    mp = os.path.join(d, "attack_meta.json")
    has_eval, has_meta = os.path.isfile(ep), os.path.isfile(mp)
    if not (has_eval or has_meta):
        continue                       # not produced yet
    ev, meta = {}, {}
    if has_eval:
        try:
            with open(ep) as f:
                ev = json.load(f)
        except Exception:
            has_eval = False
    if has_meta:
        try:
            with open(mp) as f:
                meta = json.load(f)
        except Exception:
            pass
    method = ev.get("method") or meta.get("method", "")
    loss = meta.get("fitness") or meta.get("trojvit_loss") or ""
    if not loss:
        # pulled attack_meta.json (early TrojViT) has no loss field; the run
        # dir suffix is authoritative for this grid
        base = os.path.basename(run_dir)
        for suf, name in (("_mse_sni", "mse_sni"), ("_ce", "ce"),
                          ("_cos", "cos")):
            if base.endswith(suf):
                loss = name
                break
    version = "v2" if loss == "cos" else "v1"
    # final eval values exist only when eval.json was produced
    asr = ev.get("asr") if has_eval else None
    cacc = ev.get("cacc") if has_eval else None
    cat = ev.get("clean_as_target") if has_eval else None
    trig = meta.get("trigger_asr_pre")
    # units: proflip stores %, trojvit/flips store a fraction -> normalise to %
    trigp = (trig if method == "proflip"
             else (trig * 100 if trig is not None else None))
    f94 = meta.get("flips_to_94"); f95 = meta.get("flips_to_95")
    f98 = meta.get("flips_to_98")
    atk = meta.get("asr_at_k") or {}
    if not has_eval:
        st = "NO-EVAL"                 # pulled run: attack meta only
    elif (cat or 0.0) > 0.3 or (cacc or 0.0) < 0.3:
        st = "COLLAPSED"
    elif (asr or 0.0) >= 0.98:
        st = "98%+"
    elif (asr or 0.0) >= 0.90:
        st = "90%+"
    else:
        st = "<90%"
    tar = meta.get("tar")
    if tar in (None, ""):
        tar = tar_from_log(run_dir)
    seed = ev.get("seed")
    if seed is None:
        seed = meta.get("seed", "")
    asr_s = round(asr, 4) if asr is not None else ""
    cacc_s = round(cacc, 4) if cacc is not None else ""
    cat_s = round(cat, 4) if cat is not None else ""
    detail_rows.append({
        "run_dir": run_dir, "method": method,
        "model": ev.get("model") or meta.get("model", ""),
        "dataset": ev.get("dataset") or meta.get("dataset", ""),
        "seed": seed,
        "target_cls": ev.get("target_cls", meta.get("target_cls", "")),
        "loss": loss, "version": version,
        "asr": asr_s, "asr_incl_target": ev.get("asr_incl_target", ""),
        "cacc": cacc_s, "clean_as_target": cat_s,
        "n_flips": ev.get("n_flips", meta.get("n_flips", "")),
        "budget": meta.get("budget", ""),
        "search_time_s": meta.get("search_time_s", ""),
        "trigger_asr_pre": (round(trigp, 2) if trigp is not None else ""),
        "tar": tar, "pool_size": meta.get("pool_size", ""),
        "flips_to_94": f94 if f94 is not None else "",
        "flips_to_95": f95 if f95 is not None else "",
        "flips_to_98": f98 if f98 is not None else "",
        "asr_at_10": atk.get("10", ""), "asr_at_20": atk.get("20", ""),
        "asr_at_40": atk.get("40", ""), "asr_at_80": atk.get("80", ""),
        "asr_at_100": atk.get("100", ""), "status": st,
    })
    key_rows.append({
        "run_dir": run_dir, "method": method,
        "model": ev.get("model") or meta.get("model", ""),
        "dataset": ev.get("dataset") or meta.get("dataset", ""),
        "loss": loss, "version": version, "seed": seed,
        "asr": asr_s, "cacc": cacc_s, "clean_as_target": cat_s,
        "n_flips": ev.get("n_flips", meta.get("n_flips", "")),
        "budget": meta.get("budget", ""),
        "trigger_asr_pre": (round(trigp, 2) if trigp is not None else ""),
        "flips_to_94": f94 if f94 is not None else "",
        "flips_to_95": f95 if f95 is not None else "",
        "flips_to_98": f98 if f98 is not None else "",
        "tar": tar,
        "asr_at_10": atk.get("10", ""), "asr_at_20": atk.get("20", ""),
        "asr_at_40": atk.get("40", ""), "asr_at_80": atk.get("80", ""),
        "asr_at_100": atk.get("100", ""), "status": st,
    })

if not key_rows:
    print(f"no grid runs found yet — expected {len(expected)} under "
          f"{os.path.join(here, 'results')}")
else:
    runs_csv = os.path.join(here, "results", "runs.csv")
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader(); w.writerows(detail_rows)
    sum_csv = os.path.join(here, "results", "summary.csv")
    with open(sum_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
        w.writeheader(); w.writerows(key_rows)
    print(f"scoped to {len(expected)} expected runs (list + extras) — "
          f"{len(key_rows)} with output:")
    print(f"  {runs_csv}    (one row per run, detail)")
    print(f"  {sum_csv}  (one row per run, key values)\n")

    hdr = (f"{'method':8s}{'model':11s}{'dataset':11s}{'loss':9s}"
           f"{'ASR':>6s}{'CACC':>7s}{'c->t':>6s}{'flips':>6s}"
           f"{'trig%':>7s}{'f98':>5s}  status")
    print(hdr)
    print("-" * len(hdr))
    for r in key_rows:
        t = "-" if r["trigger_asr_pre"] == "" else f"{r['trigger_asr_pre']:.1f}"
        f98s = "-" if r["flips_to_98"] == "" else str(r["flips_to_98"])
        print(f"{r['method']:8s}{r['model']:11s}{r['dataset']:11s}{r['loss']:9s}"
              f"{fnum(r['asr'], 6)}{fnum(r['cacc'], 7)}"
              f"{fnum(r['clean_as_target'], 6)}"
              f"{str(r['n_flips']):>6s}{t:>7s}{f98s:>5s}  {r['status']}")
    print("\nstatus counts:", dict(Counter(r["status"] for r in key_rows)))
PYEOF

echo
elapsed=$(( SECONDS - t0 ))
echo "Total: $total | Skipped (done): $skipped | Failed: ${#failed[@]} | Wall: ${elapsed}s"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed runs: ${failed[*]}"
  exit 1
fi
