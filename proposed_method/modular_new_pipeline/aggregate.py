"""
aggregate.py — Collect every run's bfa_results.json into CSV summaries.
=======================================================================
RESULTS.md "Code changes required" #7.  Walks the output root and writes:

  {out_root}/runs.csv         — one row per run (all models / variants / seeds)
  {out_root}/runs_{model}.csv — per-model split of runs.csv (ViT-B16, …)
  {out_root}/summary.csv      — mean ± std per configuration (over seeds)

Metric columns come from the ``metrics`` block written by the Phase-4
search (ASR@10, ASR@20, Flips→95%, …); identity columns from the run
path / ``config`` block.  Float cells are written at 2 decimals (92.30);
the JSON key ``asr_base`` is exported as the ``trigger_asr`` column
(first metric column).

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
METRIC_COLS = ["trigger_asr", "asr_at_10", "asr_at_20", "flips_to_95",
               "flips_total", "flips_rolled_back", "asr_final", "cacc_final",
               "asr_best", "cacc_base", "search_time_sec"]
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
        # Metric reconciliation: Flips→95% is definitionally 0 when the trigger
        # base ASR is already ≥95% (older runs recorded the first accepted flip).
        f95 = m.get("flips_to_95")
        if f95 and (m.get("asr_base") or 0) >= 95.0:
            f95 = 0
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
            flips_to_95=f95,
            flips_total=m.get("flips_total"),
            flips_rolled_back=m.get("flips_rolled_back"),
            asr_final=m.get("asr_final"),
            cacc_final=m.get("cacc_final"),
            asr_best=m.get("asr_best"),
            trigger_asr=m.get("asr_base"),
            cacc_base=m.get("cacc_base"),
            stop_reason=m.get("stop_reason"),
            search_time_sec=rec.get("search_time_sec"),
            out_dir=dirpath))
    return rows


def model_tag(model):
    """Directory-style model tag (ViT-B/16 → ViT-B16) for per-model files."""
    return (model or "unknown").replace("/", "")


def _cell(v):
    """CSV cell: floats at 2 decimals (92.30), ints/strings verbatim, None → ''."""
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    return v


def write_runs_csv(rows, path, label="runs.csv"):
    cols = ID_COLS + ["stop_reason"] + METRIC_COLS + ["out_dir"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: tuple(str(r[c]) for c in ID_COLS)):
            w.writerow({c: _cell(r.get(c)) for c in cols})
    print(f"  {label} → {path}  ({len(rows)} runs)")


def write_all(rows, out_root):
    """runs.csv (all models) + runs_{model}.csv per model + summary.csv."""
    write_runs_csv(rows, os.path.join(out_root, "runs.csv"))
    by_model = {}
    for r in rows:
        by_model.setdefault(model_tag(r.get("model")), []).append(r)
    for tag in sorted(by_model):
        write_runs_csv(by_model[tag],
                       os.path.join(out_root, f"runs_{tag}.csv"),
                       label=f"runs_{tag}.csv")
    write_summary_csv(rows, os.path.join(out_root, "summary.csv"))


def _fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return ""
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}"


def write_summary_csv(rows, path):
    groups = {}
    for r in rows:
        key = tuple(str(r[c]) for c in GROUP_COLS)
        groups.setdefault(key, []).append(r)

    cols = GROUP_COLS + ["n_runs", "stop_reasons"] + \
        [f"{c} (mean ± std)" for c in METRIC_COLS]
    lines = []
    for key in sorted(groups):
        rs = groups[key]
        row = dict(zip(GROUP_COLS, key))
        row["n_runs"] = len(rs)
        row["stop_reasons"] = "/".join(sorted(
            {str(r.get("stop_reason")) for r in rs if r.get("stop_reason")}))
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
    write_all(rows, out_root)


if __name__ == "__main__":
    main()
