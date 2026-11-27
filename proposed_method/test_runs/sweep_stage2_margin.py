"""
sweep_stage2_margin.py — Stage-2 margin-aware sensitivity sweep
================================================================
Loops over the grid

    mode    in {"sens", "margin_sens"}                    # baseline + new
    w_sens  in {0.01, 0.03, 0.1}
    w_align in {0.0, 0.05, 0.1, 0.2}

= 24 arms.  NOTE: "sens" is the legacy objective, which has no alignment term,
so its w_align axis is inert (those 12 arms differ only in the recorded field).
The grid is kept as specified so the run list is explicit.

Each arm is a fresh process running `main_stage2_margin.py` (so the GPU/model is
clean), writing to
    sweep_stage2_margin/arm_<mode>_ws<w_sens>_wa<w_align>/
and the aggregate to
    sweep_stage2_margin/summary.json  +  summary.md

Hard success gate (recorded per arm, copied from the pipeline):
    flips_to_90 <= 10  AND  final_ASR >= 96.0

Two reference rows are included in the summary:
    ce_64calib_reference    54.0 % standalone, 10 flips, 96.8 % final
    cos_s1_only_reference   21.0 % standalone, 15 flips, 95.9 % final
They are read from the controlled-harness output dirs when those exist,
otherwise the recorded constants are used (they are not re-run here).

Usage
-----
  python sweep_stage2_margin.py                     # full 24-arm grid
  python sweep_stage2_margin.py --arms 2            # first 2 arms (smoke)
  python sweep_stage2_margin.py --skip-sens-align   # 15 arms (no inert axis)
"""

import argparse, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE = "main_stage2_margin.py"
ROOT = HERE / "sweep_stage2_margin"

MODES = ["sens", "margin_sens"]
W_SENS = [0.01, 0.03, 0.1]
W_ALIGN = [0.0, 0.05, 0.1, 0.2]

REFERENCE = {
    "ce_64calib_reference": dict(
        dir=HERE / "xpruner_bfa_results_ViT-B/16_Caltech101_control_ce",
        S1_ASR=54.0, flips_to_90=10, final_ASR=96.8, final_CACC=88.1),
    "cos_s1_only_reference": dict(
        dir=HERE / "xpruner_bfa_results_ViT-B/16_Caltech101_control_s1only_S1cos-n64",
        S1_ASR=21.0, flips_to_90=15, final_ASR=95.9, final_CACC=88.1),
}


def arm_name(mode, ws, wa):
    return f"arm_{mode}_ws{ws:g}_wa{wa:g}"


def build_grid(skip_sens_align=False):
    out = []
    for mode in MODES:
        for ws in W_SENS:
            for wa in W_ALIGN:
                if skip_sens_align and mode == "sens" and wa != 0.0:
                    continue
                out.append((mode, ws, wa))
    return out


def flips_at(checkpoints, target=90.0):
    for n, asr, _ in checkpoints:
        if asr >= target:
            return int(n)
    return None


def collect(arm_dir: Path, mode, ws, wa, wall):
    jpath = arm_dir / "bfa_results.json"
    row = dict(arm=arm_dir.name, mode=mode, w_sens=ws, w_align=wa,
               wall_s=round(wall, 1))
    if not jpath.exists():
        row["error"] = "missing bfa_results.json"
        return row
    d = json.loads(jpath.read_text())
    st1 = d.get("stage1", {}) or {}
    st2 = d.get("stage2", {}) or {}
    diag = d.get("s2_diagnostics") or (st2.get("final") or {})
    post = (d.get("results", {}) or {}).get("post_flip_test", {}) or {}
    cps = d.get("checkpoints", []) or []

    row.update(
        S1_ASR=round(st1.get("asr_standalone", float("nan")), 2),
        S2_ASR=round(st2.get("asr_standalone", float("nan")), 2),
        cos_poison_after_S2=round(diag.get("cos_poison", float("nan")), 4),
        m_soft_after_S2=round(diag.get("m_soft", float("nan")), 4),
        sens_after_S2=diag.get("sens"),
        align_dot_after_S2=diag.get("align_dot"),
        align_cos_after_S2=diag.get("align_cos"),
        margin_caltech_after_S2=diag.get("margin_caltech"),
        flips_to_90=flips_at(cps),
        final_ASR=round(post.get("asr", float("nan")), 2),
        final_CACC=round(post.get("cacc", float("nan")), 2),
        success=bool(d.get("success", False)),
        success_detail=d.get("success_detail"),
        search_time_s=round(d.get("search_time_sec", 0.0), 1),
        n_flips=len(d.get("flip_log", []) or []),
        out_dir=str(arm_dir),
    )
    return row


def reference_rows():
    rows = []
    for name, ref in REFERENCE.items():
        row = dict(arm=name, mode="reference", w_sens=None, w_align=None,
                   S1_ASR=ref["S1_ASR"], S2_ASR=None,
                   flips_to_90=ref["flips_to_90"],
                   final_ASR=ref["final_ASR"], final_CACC=ref["final_CACC"],
                   success=(ref["flips_to_90"] <= 10 and ref["final_ASR"] >= 96.0),
                   source=("dir" if (ref["dir"] / "bfa_results.json").exists()
                           else "recorded_constant"))
        jpath = ref["dir"] / "bfa_results.json"
        if jpath.exists():
            d = json.loads(jpath.read_text())
            post = (d.get("results", {}) or {}).get("post_flip_test", {}) or {}
            row["final_ASR"] = round(post.get("asr", row["final_ASR"]), 2)
            row["final_CACC"] = round(post.get("cacc", row["final_CACC"]), 2)
            row["flips_to_90"] = flips_at(d.get("checkpoints", []))
            row["S1_ASR"] = round(
                (d.get("stage1", {}) or {}).get("asr_standalone",
                                                 row["S1_ASR"]), 2)
            row["success"] = bool(d.get("success",
                                        row["flips_to_90"] is not None
                                        and row["flips_to_90"] <= 10
                                        and row["final_ASR"] >= 96.0))
        rows.append(row)
    return rows


def fmt(v, spec="{:.2f}"):
    if v is None:
        return "  -"
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, (int,)):
        return f"{v:>3d}"
    try:
        return spec.format(v)
    except (ValueError, TypeError):
        return str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", type=int, default=None,
                    help="only the first N arms of the grid")
    ap.add_argument("--skip-sens-align", action="store_true",
                    help="drop the inert (mode=sens, w_align>0) arms -> 15 arms")
    ap.add_argument("--out", default=str(ROOT))
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    grid = build_grid(args.skip_sens_align)
    if args.arms:
        grid = grid[:args.arms]
    print(f"sweep: {len(grid)} arms -> {out}")

    rows = []
    for i, (mode, ws, wa) in enumerate(grid, 1):
        name = arm_name(mode, ws, wa)
        arm_dir = out / name
        arm_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", str(HERE / PIPELINE),
               "--s2-mode", mode,
               "--s2-w-sens", f"{ws:g}",
               "--s2-w-align", f"{wa:g}",
               "--out-dir", str(arm_dir)]
        print(f"\n=== [{i}/{len(grid)}] {name} ===", flush=True)
        t0 = time.time()
        with open(arm_dir / "run.log", "w") as fh:
            proc = subprocess.run(cmd, cwd=str(HERE), stdout=fh,
                                  stderr=subprocess.STDOUT)
        wall = time.time() - t0
        row = collect(arm_dir, mode, ws, wa, wall)
        row["exit"] = proc.returncode
        rows.append(row)
        if proc.returncode != 0:
            print(f"  FAILED (exit {proc.returncode}) — see {arm_dir/'run.log'}")
        else:
            print(f"  S1={fmt(row.get('S1_ASR'))} S2={fmt(row.get('S2_ASR'))} "
                  f"flips90={row.get('flips_to_90')} final={fmt(row.get('final_ASR'))} "
                  f"success={fmt(row.get('success'))} ({wall:.0f}s)")

    refs = reference_rows()
    (out / "summary.json").write_text(json.dumps(dict(arms=rows, references=refs),
                                                 indent=2, default=str))

    hdr = (f"{'arm':34s} {'ws':>5s} {'wa':>5s} {'S1':>6s} {'S2':>6s} "
           f"{'cosP':>7s} {'m_soft':>7s} {'f90':>4s} {'final':>6s} "
           f"{'CACC':>5s} {'succ':>4s}")
    lines = [hdr, "-" * len(hdr)]
    for r in refs + rows:
        lines.append(
            f"{r['arm'][:34]:34s} {fmt(r.get('w_sens'), '{:g}'):>5s} "
            f"{fmt(r.get('w_align'), '{:g}'):>5s} {fmt(r.get('S1_ASR')):>6s} "
            f"{fmt(r.get('S2_ASR')):>6s} {fmt(r.get('cos_poison_after_S2'), '{:.4f}'):>7s} "
            f"{fmt(r.get('m_soft_after_S2'), '{:+.4f}'):>7s} "
            f"{fmt(r.get('flips_to_90')):>4s} {fmt(r.get('final_ASR')):>6s} "
            f"{fmt(r.get('final_CACC')):>5s} {fmt(r.get('success')):>4s}")
    table = "\n".join(lines)
    print("\n" + table)

    md = ["# Stage-2 margin-aware sensitivity sweep", "",
          f"Grid: modes {MODES}, w_sens {W_SENS}, w_align {W_ALIGN}. "
          "Gate: flips_to_90 ≤ 10 **and** final ASR ≥ 96.0.", "",
          "```", table, "```", ""]
    (out / "summary.md").write_text("\n".join(md))
    print(f"\nsummary → {out/'summary.json'} , {out/'summary.md'}")


if __name__ == "__main__":
    main()
