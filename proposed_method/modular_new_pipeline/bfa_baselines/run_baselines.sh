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

# infra flags appended to every run (kept out of the runs list to stay clean)
COMMON="--data-split $SPLIT --device $DEVICE"

# ---- sanity checks -------------------------------------------------
PY_BIN="$(command -v "$PY")" || { echo "ERROR: '$PY' not found on PATH. Activate the env?" >&2; exit 1; }
[[ -f "$SPLIT" ]] || { echo "ERROR: missing data split: $SPLIT" >&2; exit 1; }
for m in trojvit proflip flips; do
  [[ -f "$HERE/attack_$m.py" ]] || { echo "ERROR: missing attack_$m.py" >&2; exit 1; }
done
echo "Using python : $PY_BIN"
echo "Device       : $DEVICE"
cd "$HERE"    # relative --out paths in the runs list resolve against $HERE

# ---- the runs list: every running item, literal, simple style -------
runs=(
  # ViT-L/14 x Caltech101
  "trojvit --model ViT-L/14 --dataset Caltech101 --seed 42 --target-cls 97 --out results/trojvit/ViT-L14/Caltech101/s42_ce --trojvit-loss ce --p2-rows 1 --p2-clean-weight 0.5"
  "trojvit --model ViT-L/14 --dataset Caltech101 --seed 42 --target-cls 97 --out results/trojvit/ViT-L14/Caltech101/s42_cos --trojvit-loss cos --p2-rows 1 --p2-clean-weight 0.5"
  "proflip --model ViT-L/14 --dataset Caltech101 --seed 42 --target-cls 97 --out results/proflip/ViT-L14/Caltech101/s42_mse_sni --fitness mse_sni --clean-ce-weight 1.0 --sni-method jsma --cbs-select-once --cbs-accept-always --cbs-top-elems 1 --grid-k 20 --no-monotone --q-mode abs"
  "proflip --model ViT-L/14 --dataset Caltech101 --seed 42 --target-cls 97 --out results/proflip/ViT-L14/Caltech101/s42_cos --fitness cos --clean-ce-weight 1.0"
  "flips --model ViT-L/14 --dataset Caltech101 --seed 42 --target-cls 97 --out results/flips/ViT-L14/Caltech101/s42 --flip-surface encoder --verify-n 32"
  # ViT-L/14 x ImageNet1k
  "trojvit --model ViT-L/14 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/trojvit/ViT-L14/ImageNet1k/s42_ce --trojvit-loss ce --p2-rows 1 --p2-clean-weight 0.5"
  "trojvit --model ViT-L/14 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/trojvit/ViT-L14/ImageNet1k/s42_cos --trojvit-loss cos --p2-rows 1 --p2-clean-weight 0.5"
  "proflip --model ViT-L/14 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/proflip/ViT-L14/ImageNet1k/s42_mse_sni --fitness mse_sni --clean-ce-weight 1.0 --sni-method jsma --cbs-select-once --cbs-accept-always --cbs-top-elems 1 --grid-k 20 --no-monotone --q-mode abs"
  "proflip --model ViT-L/14 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/proflip/ViT-L14/ImageNet1k/s42_cos --fitness cos --clean-ce-weight 1.0"
  "flips --model ViT-L/14 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/flips/ViT-L14/ImageNet1k/s42 --flip-surface encoder --verify-n 32"
  # ViT-L/14 x Food101
  "trojvit --model ViT-L/14 --dataset Food101 --seed 42 --target-cls 76 --out results/trojvit/ViT-L14/Food101/s42_ce --trojvit-loss ce --p2-rows 1 --p2-clean-weight 0.5"
  "trojvit --model ViT-L/14 --dataset Food101 --seed 42 --target-cls 76 --out results/trojvit/ViT-L14/Food101/s42_cos --trojvit-loss cos --p2-rows 1 --p2-clean-weight 0.5"
  "proflip --model ViT-L/14 --dataset Food101 --seed 42 --target-cls 76 --out results/proflip/ViT-L14/Food101/s42_mse_sni --fitness mse_sni --clean-ce-weight 1.0 --sni-method jsma --cbs-select-once --cbs-accept-always --cbs-top-elems 1 --grid-k 20 --no-monotone --q-mode abs"
  "proflip --model ViT-L/14 --dataset Food101 --seed 42 --target-cls 76 --out results/proflip/ViT-L14/Food101/s42_cos --fitness cos --clean-ce-weight 1.0"
  "flips --model ViT-L/14 --dataset Food101 --seed 42 --target-cls 76 --out results/flips/ViT-L14/Food101/s42 --flip-surface encoder --verify-n 32"

  # ViT-B/16 x Caltech101
  "trojvit --model ViT-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/trojvit/ViT-B16/Caltech101/s42_ce --trojvit-loss ce --p2-rows 1 --p2-clean-weight 0.5"
  "trojvit --model ViT-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/trojvit/ViT-B16/Caltech101/s42_cos --trojvit-loss cos --p2-rows 1 --p2-clean-weight 0.5"
  "proflip --model ViT-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/proflip/ViT-B16/Caltech101/s42_mse_sni --fitness mse_sni --clean-ce-weight 1.0 --sni-method jsma --cbs-select-once --cbs-accept-always --cbs-top-elems 1 --grid-k 20 --no-monotone --q-mode abs"
  "proflip --model ViT-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/proflip/ViT-B16/Caltech101/s42_cos --fitness cos --clean-ce-weight 1.0"
  "flips --model ViT-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/flips/ViT-B16/Caltech101/s42 --flip-surface encoder --verify-n 32"
  # ViT-B/16 x ImageNet1k
  "trojvit --model ViT-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/trojvit/ViT-B16/ImageNet1k/s42_ce --trojvit-loss ce --p2-rows 1 --p2-clean-weight 0.5"
  "trojvit --model ViT-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/trojvit/ViT-B16/ImageNet1k/s42_cos --trojvit-loss cos --p2-rows 1 --p2-clean-weight 0.5"
  "proflip --model ViT-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/proflip/ViT-B16/ImageNet1k/s42_mse_sni --fitness mse_sni --clean-ce-weight 1.0 --sni-method jsma --cbs-select-once --cbs-accept-always --cbs-top-elems 1 --grid-k 20 --no-monotone --q-mode abs"
  "proflip --model ViT-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/proflip/ViT-B16/ImageNet1k/s42_cos --fitness cos --clean-ce-weight 1.0"
  "flips --model ViT-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/flips/ViT-B16/ImageNet1k/s42 --flip-surface encoder --verify-n 32"
  # ViT-B/16 x Food101
  "trojvit --model ViT-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/trojvit/ViT-B16/Food101/s42_ce --trojvit-loss ce --p2-rows 1 --p2-clean-weight 0.5"
  "trojvit --model ViT-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/trojvit/ViT-B16/Food101/s42_cos --trojvit-loss cos --p2-rows 1 --p2-clean-weight 0.5"
  "proflip --model ViT-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/proflip/ViT-B16/Food101/s42_mse_sni --fitness mse_sni --clean-ce-weight 1.0 --sni-method jsma --cbs-select-once --cbs-accept-always --cbs-top-elems 1 --grid-k 20 --no-monotone --q-mode abs"
  "proflip --model ViT-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/proflip/ViT-B16/Food101/s42_cos --fitness cos --clean-ce-weight 1.0"
  "flips --model ViT-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/flips/ViT-B16/Food101/s42 --flip-surface encoder --verify-n 32"
 # SigLIP-B/16 x Caltech101
  "trojvit --model SigLIP-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/trojvit/SigLIP-B16/Caltech101/s42_ce --trojvit-loss ce --p2-rows 1 --p2-clean-weight 0.5"
  "trojvit --model SigLIP-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/trojvit/SigLIP-B16/Caltech101/s42_cos --trojvit-loss cos --p2-rows 1 --p2-clean-weight 0.5"
  "proflip --model SigLIP-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/proflip/SigLIP-B16/Caltech101/s42_mse_sni --fitness mse_sni --clean-ce-weight 1.0 --sni-method jsma --cbs-select-once --cbs-accept-always --cbs-top-elems 1 --grid-k 20 --no-monotone --q-mode abs"
  "proflip --model SigLIP-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/proflip/SigLIP-B16/Caltech101/s42_cos --fitness cos --clean-ce-weight 1.0"
  "flips --model SigLIP-B/16 --dataset Caltech101 --seed 42 --target-cls 97 --out results/flips/SigLIP-B16/Caltech101/s42 --flip-surface encoder --verify-n 32"
  # SigLIP-B/16 x ImageNet1k
  "trojvit --model SigLIP-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/trojvit/SigLIP-B16/ImageNet1k/s42_ce --trojvit-loss ce --p2-rows 1 --p2-clean-weight 0.5"
  "trojvit --model SigLIP-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/trojvit/SigLIP-B16/ImageNet1k/s42_cos --trojvit-loss cos --p2-rows 1 --p2-clean-weight 0.5"
  "proflip --model SigLIP-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/proflip/SigLIP-B16/ImageNet1k/s42_mse_sni --fitness mse_sni --clean-ce-weight 1.0 --sni-method jsma --cbs-select-once --cbs-accept-always --cbs-top-elems 1 --grid-k 20 --no-monotone --q-mode abs"
  "proflip --model SigLIP-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/proflip/SigLIP-B16/ImageNet1k/s42_cos --fitness cos --clean-ce-weight 1.0"
  "flips --model SigLIP-B/16 --dataset ImageNet1k --seed 42 --target-cls 291 --out results/flips/SigLIP-B16/ImageNet1k/s42 --flip-surface encoder --verify-n 32"
  # SigLIP-B/16 x Food101
  "trojvit --model SigLIP-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/trojvit/SigLIP-B16/Food101/s42_ce --trojvit-loss ce --p2-rows 1 --p2-clean-weight 0.5"
  "trojvit --model SigLIP-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/trojvit/SigLIP-B16/Food101/s42_cos --trojvit-loss cos --p2-rows 1 --p2-clean-weight 0.5"
  "proflip --model SigLIP-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/proflip/SigLIP-B16/Food101/s42_mse_sni --fitness mse_sni --clean-ce-weight 1.0 --sni-method jsma --cbs-select-once --cbs-accept-always --cbs-top-elems 1 --grid-k 20 --no-monotone --q-mode abs"
  "proflip --model SigLIP-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/proflip/SigLIP-B16/Food101/s42_cos --fitness cos --clean-ce-weight 1.0"
  "flips --model SigLIP-B/16 --dataset Food101 --seed 42 --target-cls 76 --out results/flips/SigLIP-B16/Food101/s42 --flip-surface encoder --verify-n 32"
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

for entry in "${runs[@]}"; do
  idx=$(( idx + 1 ))
  method="${entry%% *}"          # dispatch token: trojvit | proflip | flips
  args="${entry#* }"             # the literal arg string
  # readable slug from the --out dir: e.g. trojvit_ViT-B16_ImageNet1k_s42_ce
  slug=$(printf '%s' "$args" | grep -oE 'results/[^ ]+' | head -1 | sed 's#^results/##' | tr '/' '_')
  name=$(printf "%02d_%s" "$idx" "$slug")
  log="$LOGDIR/${name}.log"
  key=$(printf '%s' "$entry" | cksum | cut -d' ' -f1)   # stable per-run resume key

  if [[ "$RESUME" == "1" && -f "$DONE/$key.ok" ]]; then
    echo
    echo "=== [$idx/$total] $name: SKIP (done)"
    skipped=$(( skipped + 1 ))
    continue
  fi

  echo
  echo "=== [$idx/$total] $name  ->  attack_${method}.py"
  echo "    $args"
  start=$SECONDS
  # shellcheck disable=SC2086
  "$PY" "$HERE/attack_${method}.py" $COMMON $args 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}          # python's exit code, not tee's
  dt=$(( SECONDS - start ))

  if [[ $rc -eq 0 ]]; then
    echo "--- $name OK (${dt}s)"
    printf '%s\n' "$entry" > "$DONE/$key.ok"
  else
    echo "--- $name FAILED (exit $rc, ${dt}s) — see $log"
    failed+=("$name")
    printf 'rc=%s\n%s\n' "$rc" "$entry" > "$DONE/$key.fail"
  fi
done

# ---- summary: eval.json + attack_meta.json -> results/summary.csv ----
echo
echo "==================== SUMMARY (baselines) ===================="
"$PY" - "$HERE" <<'PYEOF'
import csv, glob, json, os, sys
from collections import Counter

here = sys.argv[1]
rows = []
for ep in sorted(glob.glob(os.path.join(here, "results", "**", "eval.json"),
                           recursive=True)):
    d = os.path.dirname(ep)
    try:
        with open(ep) as f:
            ev = json.load(f)
    except Exception:
        continue
    mp = os.path.join(d, "attack_meta.json")
    meta = {}
    if os.path.isfile(mp):
        try:
            with open(mp) as f:
                meta = json.load(f)
        except Exception:
            pass
    method = ev.get("method", "")
    loss = meta.get("fitness") or meta.get("trojvit_loss") or ""
    version = "v2" if loss == "cos" else "v1"
    asr = ev.get("asr"); cacc = ev.get("cacc"); cat = ev.get("clean_as_target")
    asr = 0.0 if asr is None else asr
    cacc = 0.0 if cacc is None else cacc
    cat = 0.0 if cat is None else cat
    trig = meta.get("trigger_asr_pre")
    # units: proflip stores %, trojvit/flips store a fraction -> normalise to %
    trigp = (trig if method == "proflip"
             else (trig * 100 if trig is not None else None))
    f94 = meta.get("flips_to_94"); f95 = meta.get("flips_to_95")
    f98 = meta.get("flips_to_98")
    atk = meta.get("asr_at_k") or {}
    if cat > 0.3 or cacc < 0.3:
        st = "COLLAPSED"
    elif asr >= 0.98:
        st = "98%+"
    elif asr >= 0.90:
        st = "90%+"
    else:
        st = "<90%"
    rows.append({
        "run_dir": os.path.relpath(d, os.path.join(here, "results")),
        "method": method, "model": ev.get("model", ""),
        "dataset": ev.get("dataset", ""), "loss": loss, "version": version,
        "seed": ev.get("seed", ""),
        "asr": round(asr, 4), "cacc": round(cacc, 4),
        "clean_as_target": round(cat, 4),
        "n_flips": ev.get("n_flips"), "budget": meta.get("budget", ""),
        "trigger_asr_pre": (round(trigp, 2) if trigp is not None else ""),
        "flips_to_94": f94 if f94 is not None else "",
        "flips_to_95": f95 if f95 is not None else "",
        "flips_to_98": f98 if f98 is not None else "",
        "tar": meta.get("tar", ""),
        "asr_at_10": atk.get("10", ""), "asr_at_20": atk.get("20", ""),
        "asr_at_40": atk.get("40", ""), "asr_at_80": atk.get("80", ""),
        "asr_at_100": atk.get("100", ""),
        "status": st,
    })

if not rows:
    print("no eval.json found under", os.path.join(here, "results"))
else:
    # ---- write summary.csv (one row per run) ----
    out_csv = os.path.join(here, "results", "summary.csv")
    fields = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}  ({len(rows)} rows, one per run)\n")

    # ---- pretty table to console ----
    hdr = (f"{'method':8s}{'model':11s}{'dataset':11s}{'loss':9s}"
           f"{'ASR':>6s}{'CACC':>7s}{'c->t':>6s}{'flips':>6s}"
           f"{'trig%':>7s}{'f98':>5s}  status")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        t = "-" if r["trigger_asr_pre"] == "" else f"{r['trigger_asr_pre']:.1f}"
        f98s = "-" if r["flips_to_98"] == "" else str(r["flips_to_98"])
        print(f"{r['method']:8s}{r['model']:11s}{r['dataset']:11s}{r['loss']:9s}"
              f"{r['asr']:>6.3f}{r['cacc']:>7.3f}{r['clean_as_target']:>6.3f}"
              f"{str(r['n_flips']):>6s}{t:>7s}{f98s:>5s}  {r['status']}")
    print("\nstatus counts:", dict(Counter(r["status"] for r in rows)))
PYEOF

echo
elapsed=$(( SECONDS - t0 ))
echo "Total: $total | Skipped (done): $skipped | Failed: ${#failed[@]} | Wall: ${elapsed}s"
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed runs: ${failed[*]}"
  exit 1
fi
