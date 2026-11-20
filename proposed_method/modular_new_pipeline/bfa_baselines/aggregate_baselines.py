"""
aggregate_baselines.py — Baseline-only CSV aggregation.
=======================================================

Scans `bfa_baselines/baseline_runs/` for the three baseline result formats and
writes **separate** baseline summaries (the proposed method's `grid_results/`
CSVs are not touched):

    baseline_summary/runs.csv       one row per baseline run
    baseline_summary/summary.csv    mean ± std per (method, model, dataset, …)

Inputs:
  * `bfa_results.json`      — ProFlip-CBS runs (`method == "proflip_cbs"`)
  * `trojvit_results.json`  — TrojViT Phase-1 runs (`method == "trojvit_trigger"`)
  * `flip_s_results.json`   — Flip-S runs (written by parse_flip_s_log.py)

Usage:
    python aggregate_baselines.py
    python aggregate_baselines.py --runs-root bfa_baselines/baseline_runs \
                                  --out-dir  bfa_baselines/baseline_summary
"""

import argparse
import csv
import json
import os
import statistics

ID_COLS = ["method", "model", "dataset", "task", "target_cls", "seed", "rep",
           "variant", "trigger_size", "trigger_area"]

METRIC_COLS = [
    # ProFlip-CBS
    "asr_at_10", "asr_at_20", "flips_to_94", "flips_to_95", "flips_total",
    "flips_rolled_back", "asr_final", "cacc_final", "cacc_base", "asr_base",
    "cbs_elements_total", "cbs_avg_bits_per_element",
    # TrojViT trigger-only
    "asr_trigger", "asr_trigger_pool", "cacc", "clean_as_target", "tar",
    # Flip-S
    "zs_cacc", "zs_asr",
    # common
    "n_eval", "n_trigger_images", "search_time_sec",
]
EXTRA_COLS = ["stop_reason", "out_dir"]

GROUP_COLS = ["method", "model", "dataset", "task", "target_cls",
              "trigger_size", "trigger_area"]


def _num(v):
    return v if isinstance(v, (int, float)) else None


def _rows_from_proflip(path, rec):
    cfg = rec.get("config", {})
    m = rec.get("metrics", {})
    tsz = cfg.get("TRIG_SZ")
    img = cfg.get("IMG_SIZE")
    area = (100.0 * tsz * tsz / (img * img)) if tsz and img else None
    return dict(
        method="proflip_cbs",
        model=cfg.get("MODEL_NAME"),
        dataset=cfg.get("DATASET"),
        task=cfg.get("TASK"),
        target_cls=cfg.get("TARGET_CLS"),
        seed=cfg.get("SEED"),
        rep=rec.get("rep", cfg.get("REP", 1)),
        variant=cfg.get("VARIANT"),
        trigger_size=tsz,
        trigger_area=area,
        asr_at_10=_num(m.get("asr_at_10")),
        asr_at_20=_num(m.get("asr_at_20")),
        flips_to_94=_num(m.get("flips_to_94")),
        flips_to_95=_num(m.get("flips_to_95")),
        flips_total=_num(m.get("flips_total")),
        flips_rolled_back=_num(m.get("flips_rolled_back")),
        asr_final=_num(m.get("asr_final")),
        cacc_final=_num(m.get("cacc_final")),
        cacc_base=_num(m.get("cacc_base")),
        asr_base=_num(m.get("asr_base")),
        cbs_elements_total=_num(m.get("cbs_elements_total")),
        cbs_avg_bits_per_element=_num(m.get("cbs_avg_bits_per_element")),
        n_eval=_num(m.get("n_eval")),
        n_trigger_images=_num(m.get("cbs_aux_images")),
        stop_reason=m.get("stop_reason"),
        search_time_sec=_num(rec.get("search_time_sec")),
        out_dir=os.path.dirname(path),
    )


def _rows_from_trojvit(path, rec):
    cfg = rec.get("config", {})
    m = rec.get("metrics", {})
    tv = rec.get("trojvit", {})
    return dict(
        method="trojvit_trigger",
        model=cfg.get("MODEL_NAME"),
        dataset=cfg.get("DATASET"),
        task=cfg.get("TASK"),
        target_cls=cfg.get("TARGET_CLS"),
        seed=cfg.get("SEED"),
        rep=rec.get("rep", cfg.get("REP", 1)),
        variant=cfg.get("VARIANT"),
        trigger_size=f"{tv.get('num_patch')}x{tv.get('patch_size')}",
        trigger_area=_num(tv.get("tar")),
        asr_trigger=_num(m.get("asr_trigger")),
        asr_trigger_pool=_num(m.get("asr_trigger_pool")),
        cacc=_num(m.get("cacc")),
        clean_as_target=_num(m.get("clean_as_target")),
        tar=_num(m.get("tar")),
        n_eval=_num(m.get("n_eval")),
        n_trigger_images=_num(m.get("n_trigger_images")),
        search_time_sec=_num(rec.get("search_time_sec")),
        out_dir=os.path.dirname(path),
    )


def _rows_from_flips(path, rec):
    m = rec.get("metrics", {})
    return dict(
        method="flip_s",
        model=rec.get("model"),
        dataset=rec.get("dataset"),
        task=rec.get("task", "zs"),
        target_cls=rec.get("target_cls"),
        seed=rec.get("seed"),
        rep=rec.get("rep", 1),
        variant="flip_s",
        trigger_size=rec.get("trigger_size", "9x16"),
        trigger_area=_num(rec.get("trigger_area", 4.59)),
        zs_cacc=_num(m.get("zs_cacc")),
        zs_asr=_num(m.get("zs_asr")),
        clean_as_target=_num(m.get("clean_as_target")),
        flips_total=_num(m.get("flips")),
        n_eval=_num(m.get("n_eval")),
        search_time_sec=_num(m.get("search_time_sec")),
        out_dir=os.path.dirname(path),
    )


def collect(runs_root):
    rows = []
    for dirpath, _dirnames, filenames in os.walk(runs_root):
        try:
            if "bfa_results.json" in filenames:
                path = os.path.join(dirpath, "bfa_results.json")
                rec = json.load(open(path))
                if rec.get("method") == "proflip_cbs" or \
                        rec.get("config", {}).get("VARIANT") == "proflip_cbs":
                    rows.append(_rows_from_proflip(path, rec))
            if "trojvit_results.json" in filenames:
                path = os.path.join(dirpath, "trojvit_results.json")
                rec = json.load(open(path))
                if rec.get("method") == "trojvit_trigger":
                    rows.append(_rows_from_trojvit(path, rec))
            if "flip_s_results.json" in filenames:
                path = os.path.join(dirpath, "flip_s_results.json")
                rec = json.load(open(path))
                if rec.get("method") == "flip_s":
                    rows.append(_rows_from_flips(path, rec))
        except Exception as exc:
            print(f"  SKIP {dirpath}: {type(exc).__name__}: {exc}")
    return rows


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    return v


def write_runs(rows, path):
    cols = ID_COLS + METRIC_COLS + EXTRA_COLS
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: tuple(str(r.get(c)) for c in ID_COLS)):
            w.writerow({c: _cell(r.get(c)) for c in cols})
    print(f"  runs.csv → {path}  ({len(rows)} rows)")


def write_summary(rows, path):
    groups = {}
    for r in rows:
        key = tuple(str(r.get(c)) for c in GROUP_COLS)
        groups.setdefault(key, []).append(r)

    cols = GROUP_COLS + ["n_runs"] + [f"{c} (mean ± std)" for c in METRIC_COLS]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for key in sorted(groups):
            grp = groups[key]
            out = {c: key[i] for i, c in enumerate(GROUP_COLS)}
            out["n_runs"] = len(grp)
            for c in METRIC_COLS:
                vals = [r.get(c) for r in grp if isinstance(r.get(c), (int, float))]
                if not vals:
                    out[f"{c} (mean ± std)"] = ""
                    continue
                mean = statistics.mean(vals)
                std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                out[f"{c} (mean ± std)"] = f"{mean:.2f} ± {std:.2f}"
            w.writerow(out)
    print(f"  summary.csv → {path}  ({len(groups)} groups)")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Aggregate baseline runs")
    p.add_argument("--runs-root", default=os.path.join(here, "baseline_runs"))
    p.add_argument("--out-dir", default=os.path.join(here, "baseline_summary"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = collect(args.runs_root)
    print(f"[baselines] {len(rows)} run(s) under {args.runs_root}")
    write_runs(rows, os.path.join(args.out_dir, "runs.csv"))
    write_summary(rows, os.path.join(args.out_dir, "summary.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
