"""
run_sweep.py — grid driver for the CLIP baseline suite.
=======================================================

Loops methods x models x datasets x seeds, launching ONE SUBPROCESS PER RUN
(clean model state by construction), then runs the shared evaluator on the
produced attack_state.pt, and aggregates everything into
results/summary/{runs.csv, summary.csv}.

Usage:
    python run_sweep.py                                    # full default grid
    python run_sweep.py --methods proflip --models ViT-B/16 \\
        --datasets Caltech101 --seeds 42 43                # one cell
    python run_sweep.py --dry-run                          # print commands only
    python run_sweep.py --force                            # rerun finished runs

Resume: a run is skipped when its eval.json already exists (unless --force).
Per-run outputs:
    results/{method}/{model_tag}/{dataset}/s{seed}/
        attack_state.pt  attack_meta.json  eval.json  run_attack.log  run_eval.log

Default grid: methods {trojvit, proflip, flips} x models {ViT-B/16, ViT-L/14,
SigLIP-B/16} x datasets {ImageNet1k, Caltech101, Food101} x seeds {42}.
ProFlip runs BOTH fitness modes (mse_sni + cos) -> two rows per cell.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "results")
SPLIT = os.path.join(_HERE, "data_split.json")

METHODS = ["trojvit", "proflip", "flips"]
MODELS = ["ViT-B/16", "ViT-L/14", "SigLIP-B/16"]
DATASETS = ["ImageNet1k", "Caltech101", "Food101"]
SEEDS = [42]

# target class per dataset — must match the proposed method's runs so the
# comparison table is apples-to-apples (run_all.sh conventions: ImageNet1k
# = 291 "lion", Caltech101 = 97 "wild cat"; Food101 = 76 "pizza", agreed).
TARGET_CLS = {"Caltech101": 97, "ImageNet1k": 291, "Food101": 76}

PY = sys.executable


def model_tag(name):
    return name.replace("/", "").replace("-336px", "-336").replace("px", "")


def variants_for(method):
    """Per-method run variants (label, extra CLI args)."""
    if method == "proflip":
        return [("mse_sni", ["--fitness", "mse_sni"]),
                ("cos", ["--fitness", "cos"])]
    return [("", [])]


def run_dir(method, variant, model, dataset, seed):
    tag = model_tag(model)
    name = f"s{seed}" + (f"_{variant}" if variant else "")
    return os.path.join(RESULTS, method, tag, dataset, name)


def run_one(method, variant, extra, model, dataset, seed, target_cls, args,
            idx, total):
    out = run_dir(method, variant, model, dataset, seed)
    os.makedirs(out, exist_ok=True)
    eval_json = os.path.join(out, "eval.json")
    state = os.path.join(out, "attack_state.pt")
    tag = f"{idx}/{total}"
    if os.path.exists(eval_json) and not args.force:
        print(f"{tag} [skip] {out} (eval.json exists)")
        return out, True

    attack_cmd = [
        PY, os.path.join(_HERE, f"attack_{method}.py"),
        "--model", model, "--dataset", dataset,
        "--seed", str(seed), "--target-cls", str(target_cls),
        "--data-split", SPLIT, "--out", out, "--device", args.device,
    ] + extra
    eval_cmd = [
        PY, os.path.join(_HERE, "evaluate.py"),
        "--state", state, "--data-split", SPLIT,
        "--out", eval_json, "--device", args.device,
    ]

    if args.dry_run:
        print(f"{tag} [dry ] {out}")
        print(" ".join(attack_cmd))
        print(" ".join(eval_cmd))
        return out, True

    print(f"{tag} [ run ] {out}")
    t0 = time.time()
    with open(os.path.join(out, "run_attack.log"), "w") as log:
        rc = subprocess.call(attack_cmd, stdout=log, stderr=subprocess.STDOUT,
                             cwd=_HERE)
    if rc != 0 or not os.path.exists(state):
        print(f"{tag} [FAIL] attack rc={rc} (see {out}/run_attack.log)")
        return out, False
    with open(os.path.join(out, "run_eval.log"), "w") as log:
        rc = subprocess.call(eval_cmd, stdout=log, stderr=subprocess.STDOUT,
                             cwd=_HERE)
    if rc != 0 or not os.path.exists(eval_json):
        print(f"{tag} [FAIL] evaluate rc={rc} (see {out}/run_eval.log)")
        return out, False
    try:
        os.remove(state)      # keep only eval.json + attack_meta.json
    except OSError:
        pass
    print(f"{tag} [ ok ] {out} ({time.time() - t0:.0f}s)")
    return out, True


def aggregate(expected=None):
    """Collect the grid runs' eval.json + attack_meta.json into
    results/runs.csv + results/summary.csv.

    `expected` = set of run dirs (relpath under RESULTS) that belong to the
    grid; when given, only those are collected, so stale / backup / non-grid
    runs can never leak in.  Writes into results/ (not results/summary/) so
    there is a single summary.csv — run_baselines.sh writes the same two files
    for its own runs list."""
    rows = []
    for root, _dirs, files in os.walk(RESULTS):
        if "eval.json" not in files:
            continue
        rel = os.path.relpath(root, RESULTS)
        if expected is not None and rel not in expected:
            continue          # skip stale / backup / non-grid runs
        with open(os.path.join(root, "eval.json")) as f:
            ev = json.load(f)
        meta_path = os.path.join(root, "attack_meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        row = {
            "run_dir": os.path.relpath(root, RESULTS),
            "method": ev.get("method", ""),
            "model": ev.get("model", ""),
            "dataset": ev.get("dataset", ""),
            "seed": ev.get("seed", ""),
            "target_cls": ev.get("target_cls", ""),
            "fitness": meta.get("fitness", ""),
            "asr": ev.get("asr"),
            "asr_incl_target": ev.get("asr_incl_target"),
            "cacc": ev.get("cacc"),
            "clean_as_target": ev.get("clean_as_target"),
            "n_flips": ev.get("n_flips", meta.get("n_flips")),
            "budget": meta.get("budget", ""),
            "search_time_s": meta.get("search_time_s", ""),
            "trigger_asr_pre": meta.get("trigger_asr_pre", ""),
            "tar": meta.get("tar", ""),
            "pool_size": meta.get("pool_size", ""),
            "flips_to_94": meta.get("flips_to_94", ""),
            "flips_to_95": meta.get("flips_to_95", ""),
            "flips_to_98": meta.get("flips_to_98", ""),
            "asr_at_10": (meta.get("asr_at_k") or {}).get("10", ""),
            "asr_at_20": (meta.get("asr_at_k") or {}).get("20", ""),
            "asr_at_40": (meta.get("asr_at_k") or {}).get("40", ""),
            "asr_at_80": (meta.get("asr_at_k") or {}).get("80", ""),
            "asr_at_100": (meta.get("asr_at_k") or {}).get("100", ""),
        }
        rows.append(row)
    if not rows:
        print("[aggregate] no eval.json found — nothing to write")
        return

    runs_csv = os.path.join(RESULTS, "runs.csv")
    fields = list(rows[0].keys())
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["run_dir"]))

    # summary.csv: ONE ROW PER RUN (key values) — same schema as
    # run_baselines.sh so whichever tool runs last writes a consistent, scoped
    # per-run summary (no aggregate / per-run clash on the same filename).
    def _classify(asr, cacc, cat):
        if cat > 0.3 or cacc < 0.3:
            return "COLLAPSED"
        if asr >= 0.98:
            return "98%+"
        if asr >= 0.90:
            return "90%+"
        return "<90%"
    sum_rows = []
    for r in sorted(rows, key=lambda x: x["run_dir"]):
        loss = r.get("fitness", "") or ""
        version = "v2" if loss == "cos" else "v1"
        asr = r.get("asr") or 0.0
        cacc = r.get("cacc") or 0.0
        cat = r.get("clean_as_target") or 0.0
        trig = r.get("trigger_asr_pre")
        trigp = (trig if r.get("method") == "proflip"
                 else (trig * 100 if isinstance(trig, (int, float)) else None))
        sum_rows.append({
            "run_dir": r["run_dir"], "method": r.get("method", ""),
            "model": r.get("model", ""), "dataset": r.get("dataset", ""),
            "loss": loss, "version": version, "seed": r.get("seed", ""),
            "asr": round(asr, 4), "cacc": round(cacc, 4),
            "clean_as_target": round(cat, 4),
            "n_flips": r.get("n_flips", ""), "budget": r.get("budget", ""),
            "trigger_asr_pre": (round(trigp, 2) if trigp is not None else ""),
            "flips_to_94": r.get("flips_to_94", ""),
            "flips_to_95": r.get("flips_to_95", ""),
            "flips_to_98": r.get("flips_to_98", ""),
            "tar": r.get("tar", ""),
            "asr_at_10": r.get("asr_at_10", ""), "asr_at_20": r.get("asr_at_20", ""),
            "asr_at_40": r.get("asr_at_40", ""), "asr_at_80": r.get("asr_at_80", ""),
            "asr_at_100": r.get("asr_at_100", ""),
            "status": _classify(asr, cacc, cat),
        })
    summary_csv = os.path.join(RESULTS, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sum_rows[0].keys()))
        w.writeheader()
        w.writerows(sum_rows)
    print(f"[aggregate] {len(rows)} runs -> {runs_csv}, {summary_csv}")


def main():
    p = argparse.ArgumentParser(description="Baseline grid driver")
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--target-cls", type=int, default=None,
                   help="override the per-dataset target class")
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-aggregate", action="store_true")
    args = p.parse_args()

    n_ok = n_fail = 0
    total = 0
    for method in args.methods:
        total += (len(variants_for(method)) * len(args.models)
                  * len(args.datasets) * len(args.seeds))
    expected = set()
    idx = 0
    for method in args.methods:
        for model in args.models:
            for dataset in args.datasets:
                for seed in args.seeds:
                    tcls = (args.target_cls if args.target_cls is not None
                            else TARGET_CLS[dataset])
                    for variant, extra in variants_for(method):
                        idx += 1
                        out, ok = run_one(method, variant, extra, model,
                                          dataset, seed, tcls, args, idx,
                                          total)
                        expected.add(os.path.relpath(out, RESULTS))
                        n_ok += ok
                        n_fail += (not ok)
    print(f"[done] ok={n_ok} fail={n_fail}")
    if not args.no_aggregate and not args.dry_run:
        aggregate(expected)


if __name__ == "__main__":
    main()
