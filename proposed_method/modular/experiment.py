"""
experiment.py — Build a run list from global_config.json and execute all runs.
=============================================================================
Each config runs n_seeds times. Folder structure:

    Results/
    ├── runs.csv                          ← global (every run)
    └── ViT-L14/
        └── caltech101/
            ├── cadd14/
            │   ├── runs.csv              ← 5 rows (one per seed)
            │   ├── seed42/
            │   ├── seed43/
            │   ├── seed44/
            │   ├── seed45/
            │   └── seed46/
            └── cadd28/
                ├── runs.csv
                └── ...

Usage:
    python experiment.py
"""

import os
import csv
import json
import time
import itertools
import numpy as np
from collections import defaultdict

from config import CFG
from pipeline import AttackPipeline
from utils import set_seed, cleanup_gpu


# ══════════════════════════════════════════════════════════════════════
#  1. LOAD GLOBAL CONFIG
# ══════════════════════════════════════════════════════════════════════
def load_global_config(path="global_config.json"):
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════
#  2. BUILD RUN LIST
# ══════════════════════════════════════════════════════════════════════
def build_run_list(gconfig):
    """
    Create a flat list of every run to execute.
    Each dict has everything needed for one pipeline execution.
    """
    models      = gconfig["active_model"]
    datasets    = gconfig["active_dataset"]
    trig_types  = gconfig["active_trigger_type"]
    trig_sizes  = gconfig["active_trigger_size"]
    targets     = gconfig.get("active_target_cls", [3])
    max_flips   = gconfig.get("active_max_flips", [80])
    n_seeds     = gconfig.get("n_seeds", 5)
    base_seed   = gconfig.get("base_seed", 42)
    out_root    = gconfig.get("out_root", "Results")

    runs = []
    for model, ds, tt, ts, tc, mf in itertools.product(
            models, datasets, trig_types, trig_sizes, targets, max_flips):

        model_tag = model.replace("/", "").replace("-", "")   # ViT-L14
        trig_tag = f"{tt}{ts}"                                 # cadd14
        config_id = f"{trig_tag}"
        config_dir = os.path.join(out_root, model_tag, ds, config_id)

        for i in range(n_seeds):
            seed = base_seed + i
            runs.append({
                "model":         model,
                "dataset":       ds,
                "trigger_type":  tt,
                "trigger_size":  ts,
                "target_cls":    tc,
                "max_flips":     mf,
                "seed":          seed,
                "run_id":        f"{model_tag}__{ds}__{trig_tag}",
                "config_dir":    config_dir,
                "run_dir":       os.path.join(config_dir, f"seed{seed}"),
            })

    return runs


# ══════════════════════════════════════════════════════════════════════
#  3. EXECUTE ONE RUN
# ══════════════════════════════════════════════════════════════════════
def run_single(run, gconfig):
    """Execute one pipeline run and return a metrics dict."""
    set_seed(run["seed"])

    cfg = CFG()
    cfg.MODEL        = run["model"]
    cfg.DATASET      = run["dataset"]
    cfg.DATASET_NAME = run["dataset"]
    cfg.DATA         = gconfig.get("data_root", "../data")
    cfg.DOWNLOAD     = gconfig.get("download", False)
    cfg.TARGET_CLS   = run["target_cls"]
    cfg.TRIGGER_TYPE = run["trigger_type"]
    cfg.TRIG_SZ      = run["trigger_size"]
    cfg.MAX_FLIPS    = run["max_flips"]
    cfg.SEED         = run["seed"]
    cfg.RUN_ID       = run["run_id"]
    cfg.OUT_DIR      = run["run_dir"]

    os.makedirs(run["run_dir"], exist_ok=True)

    pipeline = AttackPipeline(cfg)
    metrics = pipeline.run()          # returns dict of key metrics

    return {**run, **metrics}


# ══════════════════════════════════════════════════════════════════════
#  4. CSV HELPERS
# ══════════════════════════════════════════════════════════════════════
CSV_FIELDS = [
    "run_id", "model", "dataset",
    "trigger_type", "trigger_size", "target_cls", "max_flips",
    "seed", "n_flips", "asr", "cacc",
    "zs_top1_cacc", "zs_top1_asr",
    "lp_cacc", "lp_asr",
    "ft_cacc", "ft_asr",
    "search_time_sec",
]


def write_csv(path, results):
    """Write a list of result dicts to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            rounded = {}
            for k, v in row.items():
                if isinstance(v, float):
                    rounded[k] = round(v, 2)
                else:
                    rounded[k] = v
            writer.writerow(rounded)
    print(f"  CSV saved → {path}")


def append_global_csv(path, results):
    """Append rows to the global runs.csv (creates header if needed)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in results:
            rounded = {}
            for k, v in row.items():
                if isinstance(v, float):
                    rounded[k] = round(v, 2)
                else:
                    rounded[k] = v
            writer.writerow(rounded)


# ══════════════════════════════════════════════════════════════════════
#  5. PRINT SUMMARIES
# ══════════════════════════════════════════════════════════════════════
def print_config_summary(results):
    """Print mean ± std for one config (5 seeds)."""
    if not results:
        return
    for key in ["asr", "cacc", "n_flips", "lp_asr", "ft_asr"]:
        vals = [r[key] for r in results if key in r and r[key] is not None]
        if vals:
            m, s = np.mean(vals), np.std(vals)
            print(f"    {key:>15s}: {m:.2f} ± {s:.2f}  "
                  f"(n={len(vals)})")


def print_final_summary(all_results):
    print("\n" + "=" * 70)
    print("ALL RUNS COMPLETE")
    print("=" * 70)
    print(f"\n  {'Run ID':<45s} {'Seed':>5s} {'ASR':>7s} {'CACC':>7s} "
          f"{'Flips':>6s}")
    print(f"  {'─' * 75}")
    for r in all_results:
        print(f"  {r.get('run_id', '?'):<45s} "
              f"{r.get('seed', 0):>5d} "
              f"{r.get('asr', 0):>7.2f} {r.get('cacc', 0):>7.2f} "
              f"{r.get('n_flips', 0):>6d}")


# ══════════════════════════════════════════════════════════════════════
#  6. MAIN LOOP
# ══════════════════════════════════════════════════════════════════════
def run_all(gconfig_path="global_config.json"):
    gconfig = load_global_config(gconfig_path)
    run_list = build_run_list(gconfig)
    out_root = gconfig.get("out_root", "Results")

    # Group by config_dir (each group = 5 seeds of same config)
    config_groups = defaultdict(list)
    for run in run_list:
        config_groups[run["config_dir"]].append(run)

    print("=" * 70)
    print("EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"  Total runs      : {len(run_list)}")
    print(f"  Config groups   : {len(config_groups)}")
    print(f"  Seeds per config: {gconfig.get('n_seeds', 5)}")
    print(f"  Output root     : {out_root}")

    all_results = []
    t_total = time.time()

    for ci, (config_dir, runs) in enumerate(config_groups.items()):
        print(f"\n{'━' * 70}")
        print(f"CONFIG {ci + 1}/{len(config_groups)}  →  {runs[0]['run_id']}")
        print(f"  {config_dir}")
        print(f"{'━' * 70}")

        config_results = []

        for run in runs:
            print(f"\n  Seed {run['seed']} ...")
            try:
                result = run_single(run, gconfig)
                config_results.append(result)
                all_results.append(result)
                print(f"  ✓ ASR={result.get('asr', 0):.1f}%  "
                      f"CACC={result.get('cacc', 0):.1f}%  "
                      f"flips={result.get('n_flips', 0)}")
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                import traceback
                traceback.print_exc()
            cleanup_gpu()

        # Per-config CSV (5 rows)
        if config_results:
            write_csv(os.path.join(config_dir, "runs.csv"), config_results)

            print(f"\n  Summary for {runs[0]['run_id']}:")
            print_config_summary(config_results)

        # Append to global CSV after each config
        if config_results:
            append_global_csv(os.path.join(out_root, "runs.csv"), config_results)

    elapsed = time.time() - t_total
    print_final_summary(all_results)
    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"  Global CSV : {os.path.join(out_root, 'runs.csv')}")

    return all_results


# ══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_all("global_config.json")