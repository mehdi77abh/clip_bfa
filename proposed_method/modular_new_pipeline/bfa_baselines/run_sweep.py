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
ViT-L/14-336px} x datasets {Caltech101, ImageNet1k, DTD} x seeds {42}.
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
MODELS = ["ViT-B/16", "ViT-L/14", "ViT-L/14-336px"]
DATASETS = ["Caltech101", "ImageNet1k", "DTD"]
SEEDS = [42]

# target class per dataset (same convention as the previous baseline shell)
TARGET_CLS = {"Caltech101": 97, "ImageNet1k": 291, "DTD": 3}

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


def run_one(method, variant, extra, model, dataset, seed, target_cls, args):
    out = run_dir(method, variant, model, dataset, seed)
    os.makedirs(out, exist_ok=True)
    eval_json = os.path.join(out, "eval.json")
    state = os.path.join(out, "attack_state.pt")
    if os.path.exists(eval_json) and not args.force:
        print(f"[skip] {out} (eval.json exists)")
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
        print(" ".join(attack_cmd))
        print(" ".join(eval_cmd))
        return out, True

    print(f"[run ] {out}")
    t0 = time.time()
    with open(os.path.join(out, "run_attack.log"), "w") as log:
        rc = subprocess.call(attack_cmd, stdout=log, stderr=subprocess.STDOUT,
                             cwd=_HERE)
    if rc != 0 or not os.path.exists(state):
        print(f"[FAIL] attack rc={rc} (see {out}/run_attack.log)")
        return out, False
    with open(os.path.join(out, "run_eval.log"), "w") as log:
        rc = subprocess.call(eval_cmd, stdout=log, stderr=subprocess.STDOUT,
                             cwd=_HERE)
    if rc != 0 or not os.path.exists(eval_json):
        print(f"[FAIL] evaluate rc={rc} (see {out}/run_eval.log)")
        return out, False
    print(f"[ ok ] {out} ({time.time() - t0:.0f}s)")
    return out, True


def aggregate():
    """Collect every eval.json + attack_meta.json into runs.csv / summary.csv."""
    rows = []
    for root, _dirs, files in os.walk(RESULTS):
        if "eval.json" not in files:
            continue
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
            "pool_size": meta.get("pool_size", ""),
            "flips_to_94": meta.get("flips_to_94", ""),
            "flips_to_95": meta.get("flips_to_95", ""),
        }
        rows.append(row)
    if not rows:
        print("[aggregate] no eval.json found — nothing to write")
        return

    os.makedirs(os.path.join(RESULTS, "summary"), exist_ok=True)
    runs_csv = os.path.join(RESULTS, "summary", "runs.csv")
    fields = list(rows[0].keys())
    with open(runs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["run_dir"]))

    # summary: mean over seeds per (method, model, dataset, fitness)
    groups = {}
    for r in rows:
        key = (r["method"], r["fitness"], r["model"], r["dataset"])
        groups.setdefault(key, []).append(r)
    sum_rows = []
    for key, rs in sorted(groups.items()):
        def mean(field):
            vals = [r[field] for r in rs
                    if isinstance(r[field], (int, float))]
            return round(sum(vals) / len(vals), 4) if vals else ""
        sum_rows.append({
            "method": key[0], "fitness": key[1], "model": key[2],
            "dataset": key[3], "n_seeds": len(rs),
            "asr": mean("asr"), "cacc": mean("cacc"),
            "n_flips": mean("n_flips"),
            "trigger_asr_pre": mean("trigger_asr_pre"),
            "flips_to_94": mean("flips_to_94"),
            "flips_to_95": mean("flips_to_95"),
        })
    summary_csv = os.path.join(RESULTS, "summary", "summary.csv")
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
    for method in args.methods:
        for model in args.models:
            for dataset in args.datasets:
                for seed in args.seeds:
                    tcls = (args.target_cls if args.target_cls is not None
                            else TARGET_CLS[dataset])
                    for variant, extra in variants_for(method):
                        out, ok = run_one(method, variant, extra, model,
                                          dataset, seed, tcls, args)
                        n_ok += ok
                        n_fail += (not ok)
    print(f"[done] ok={n_ok} fail={n_fail}")
    if not args.no_aggregate and not args.dry_run:
        aggregate()


if __name__ == "__main__":
    main()
