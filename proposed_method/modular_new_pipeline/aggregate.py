"""
aggregate.py — Collect every run's bfa_results.json into CSV summaries.
=======================================================================
RESULTS.md "Code changes required" #7.  Walks the output root and writes:

  {out_root}/runs.csv     — one row per run (all variants / reps / seeds)
  {out_root}/summary.csv  — mean ± std per configuration (over reps/seeds)

Metric columns come from the ``metrics`` block written by the Phase-4
search (ASR@10, ASR@20, Flips→95%, …); identity columns from the run
path / ``config`` block.

Usage:
    python aggregate.py                     # ./grid_results
    python aggregate.py --out-root /path/to/grid_results
"""

import argparse
import csv
import json
import os
import statistics

ID_COLS = ["section", "model", "dataset", "task", "target_cls", "seed", "rep",
           "variant", "trigger_size", "trigger_location"]
METRIC_COLS = ["asr_at_10", "asr_at_20", "flips_to_95", "flips_total",
               "asr_final", "cacc_final", "asr_base", "cacc_base",
               "search_time_sec"]
GROUP_COLS = ["section", "model", "dataset", "task", "target_cls",
              "variant", "trigger_size", "trigger_location"]


def collect(out_root):
    rows = []
    for dirpath, _dirnames, filenames in os.walk(out_root):
        if "bfa_results.json" not in filenames:
            continue
        path = os.path.join(dirpath, "bfa_results.json")
        try:
            with open(path) as f:
                rec = json.load(f)
        except Exception as e:
            print(f"  SKIP {path}: {e}")
            continue
        cfg = rec.get("config", {})
        m = rec.get("metrics", {})
        rows.append(dict(
            section=rec.get("section", cfg.get("SECTION")),
            model=cfg.get("MODEL_NAME"),
            dataset=cfg.get("DATASET"),
            task=cfg.get("TASK"),
            target_cls=cfg.get("TARGET_CLS"),
            seed=cfg.get("SEED"),
            rep=rec.get("rep", cfg.get("REP", 1)),
            variant=cfg.get("VARIANT"),
            trigger_size=cfg.get("TRIG_SZ"),
            trigger_location=rec.get("trigger_location",
                                     cfg.get("TRIGGER_LOCATION")),
            asr_at_10=m.get("asr_at_10"),
            asr_at_20=m.get("asr_at_20"),
            flips_to_95=m.get("flips_to_95"),
            flips_total=m.get("flips_total"),
            asr_final=m.get("asr_final"),
            cacc_final=m.get("cacc_final"),
            asr_base=m.get("asr_base"),
            cacc_base=m.get("cacc_base"),
            search_time_sec=rec.get("search_time_sec"),
            out_dir=dirpath))
    return rows


def write_runs_csv(rows, path):
    cols = ID_COLS + METRIC_COLS + ["out_dir"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: tuple(str(r[c]) for c in ID_COLS)):
            w.writerow({c: r.get(c) for c in cols})
    print(f"  runs.csv    → {path}  ({len(rows)} runs)")


def _fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return ""
    if len(vals) == 1:
        return f"{vals[0]:g}"
    return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}"


def write_summary_csv(rows, path):
    groups = {}
    for r in rows:
        key = tuple(str(r[c]) for c in GROUP_COLS)
        groups.setdefault(key, []).append(r)

    cols = GROUP_COLS + ["n_runs"] + \
        [f"{c} (mean ± std)" for c in METRIC_COLS]
    lines = []
    for key in sorted(groups):
        rs = groups[key]
        row = dict(zip(GROUP_COLS, key))
        row["n_runs"] = len(rs)
        for c in METRIC_COLS:
            row[f"{c} (mean ± std)"] = _fmt([r.get(c) for r in rs])
        lines.append(row)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(lines)
    print(f"  summary.csv → {path}  ({len(lines)} configs)")

    # compact echo of the main table columns
    print(f"\n  {'section':<11} {'model':<16} {'dataset':<11} {'variant':<6} "
          f"{'loc':<13} {'ASR@10':>15} {'ASR@20':>15} {'Flips→95%':>15} {'CACC':>15}")
    for row in lines:
        print(f"  {str(row['section']):<11} {row['model']:<16} "
              f"{row['dataset']:<11} {row['variant']:<6} "
              f"{str(row['trigger_location']):<13} "
              f"{row['asr_at_10 (mean ± std)']:>15} "
              f"{row['asr_at_20 (mean ± std)']:>15} "
              f"{row['flips_to_95 (mean ± std)']:>15} "
              f"{row['cacc_final (mean ± std)']:>15}")


def main():
    p = argparse.ArgumentParser(description="Aggregate run results")
    p.add_argument("--out-root", default="./grid_results")
    args = p.parse_args()
    out_root = args.out_root

    rows = collect(out_root)
    if not rows:
        print(f"No bfa_results.json found under {out_root}")
        return
    write_runs_csv(rows, os.path.join(out_root, "runs.csv"))
    write_summary_csv(rows, os.path.join(out_root, "summary.csv"))


if __name__ == "__main__":
    main()
