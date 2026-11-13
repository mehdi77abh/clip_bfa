"""
sweep_bc.py — arms B (EOT) and C (A* k-means) vs the cos@64 baseline
====================================================================
Runs each arm in a FRESH process of `main_stage_margin_v3.py` (RNG state must
reset), reads `bfa_results.json`, and writes:

    sweep_bc/summary.json     per-arm rows + references
    sweep_bc/summary.md       markdown table(s)
    sweep_bc/<arm>/bfa_results.json, trigger*.pt, run.log

CE is NEVER executed here — by policy its row is read from the existing
controlled run (`xpruner_bfa_results_ViT-B/16_Caltech101_control_ce`) and
cached in `sweep_bc/reference_cache.json`.  The `cos_s1_ref` baseline IS
recomputed in this harness, as specified.

Success (per row):  success = (flips_to_90 <= 12) and (final_ASR >= 96.0)
`failing` lists the violated conditions.
"""

import argparse, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE = "main_stage_margin_v3.py"
OUT_ROOT = HERE / "sweep_bc"

# CE reference: read-only, never re-executed.
CE_REF_DIR = HERE / "xpruner_bfa_results_ViT-B/16_Caltech101_control_ce"
CE_REF_NAME = "ce_64_ref"

ARMS = [
    # reference baseline (recomputed in this harness)
    dict(name="cos_s1_ref", mode="cos", overrides={}),
    # mode B — EOT sweep
    dict(name="eot_m2", mode="cos_eot", overrides=dict(EOT_M=2)),
    dict(name="eot_m3", mode="cos_eot", overrides=dict(EOT_M=3)),
    dict(name="eot_m5", mode="cos_eot", overrides=dict(EOT_M=5)),
    # mode C — A* sweep on K (lambda = 1.0)
    dict(name="astar_k4_l1", mode="cos_astar", overrides=dict(ASTAR_K=4, ASTAR_LAMBDA=1.0)),
    dict(name="astar_k8_l1", mode="cos_astar", overrides=dict(ASTAR_K=8, ASTAR_LAMBDA=1.0)),
    dict(name="astar_k16_l1", mode="cos_astar", overrides=dict(ASTAR_K=16, ASTAR_LAMBDA=1.0)),
    # mode C — lambda sweep at K=8
    dict(name="astar_k8_l05", mode="cos_astar", overrides=dict(ASTAR_K=8, ASTAR_LAMBDA=0.5)),
    dict(name="astar_k8_l2", mode="cos_astar", overrides=dict(ASTAR_K=8, ASTAR_LAMBDA=2.0)),
]

CLI_FLAG = dict(EOT_M="--eot-m", ASTAR_K="--astar-k",
                ASTAR_LAMBDA="--astar-lambda", ASTAR_MU="--astar-mu")

MAX_FLIPS_GATE = 12
MIN_ASR_GATE = 96.0


def flips_at(checkpoints, target=90.0):
    for n, asr, _ in checkpoints:
        if asr >= target:
            return int(n)
    return None


def judge(row):
    failing = []
    if row.get("flips_to_90") is None or row["flips_to_90"] > MAX_FLIPS_GATE:
        failing.append(f"flips_to_90={row.get('flips_to_90')} > {MAX_FLIPS_GATE}")
    if row.get("final_ASR") is None or row["final_ASR"] < MIN_ASR_GATE:
        failing.append(f"final_ASR={row.get('final_ASR')} < {MIN_ASR_GATE}")
    row["success"] = not failing
    row["failing"] = failing
    return row


def read_row(name, mode, overrides, arm_dir):
    jpath = arm_dir / "bfa_results.json"
    row = dict(name=name, mode=mode, overrides=overrides,
               S1_ASR=None, S1_CACC=None, flips_to_90=None, final_ASR=None,
               final_CACC=None, search_time_s=None, trigger_path=None)
    if not jpath.exists():
        row["success"] = False
        row["failing"] = ["missing bfa_results.json"]
        return row
    d = json.loads(jpath.read_text())
    st1 = d.get("stage1", {}) or {}
    post = (d.get("results", {}) or {}).get("post_flip_test", {}) or {}
    cps = d.get("checkpoints", []) or []
    row.update(
        S1_ASR=round(float(st1.get("asr_standalone", cps[0][1] if cps else float("nan"))), 2),
        S1_CACC=round(float(st1.get("cacc_standalone", cps[0][2] if cps else float("nan"))), 2),
        flips_to_90=flips_at(cps),
        final_ASR=round(float(post.get("asr", float("nan"))), 2),
        final_CACC=round(float(post.get("cacc", float("nan"))), 2),
        search_time_s=round(float(d.get("search_time_sec", 0.0)), 1),
        trigger_path=str(next((p for p in ("trigger.pt", "trigger_cos_eot.pt",
                                           "trigger_cos_astar.pt")
                               if (arm_dir / p).exists()), "")),
        n_flips=len(d.get("flip_log", []) or []),
        out_dir=str(arm_dir),
    )
    return judge(row)


def ce_reference(cache_path):
    """CE row — read from the existing controlled run; NEVER executed here."""
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("name") == CE_REF_NAME and cached.get("flips_to_90") is not None:
                cached.setdefault("source", "reference_cache.json")
                return cached
        except Exception:
            pass
    if (CE_REF_DIR / "bfa_results.json").exists():
        row = read_row(CE_REF_NAME, "ce", {}, CE_REF_DIR)
        row["source"] = str(CE_REF_DIR.relative_to(HERE)) + " (CE NOT re-executed)"
    else:
        row = dict(name=CE_REF_NAME, mode="ce", overrides={}, S1_ASR=None,
                   S1_CACC=None, flips_to_90=None, final_ASR=None,
                   final_CACC=None, search_time_s=None, trigger_path=None,
                   source="unavailable")
        row["success"] = False
        row["failing"] = ["CE reference dir missing — CE is not re-executed by policy"]
    cache_path.write_text(json.dumps(row, indent=2))
    return row


def fmt(v, spec="{:.2f}", width=None):
    s = "-" if v is None else (spec.format(v) if isinstance(v, float) else str(v))
    return s.rjust(width) if width else s


def table(rows):
    hdr = (f"{'arm':14s} {'mode':10s} {'S1_ASR':>7s} {'S1_CACC':>7s} "
           f"{'flips90':>7s} {'final':>6s} {'CACC':>5s} {'sec':>6s} {'ok':>3s}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(f"{r['name'][:14]:14s} {r['mode'][:10]:10s} "
                     f"{fmt(r.get('S1_ASR'), width=7)} {fmt(r.get('S1_CACC'), width=7)} "
                     f"{fmt(r.get('flips_to_90'), width=7)} "
                     f"{fmt(r.get('final_ASR'), width=6)} {fmt(r.get('final_CACC'), width=5)} "
                     f"{fmt(r.get('search_time_s'), '{:.0f}', width=6)} "
                     f"{'T' if r.get('success') else 'F':>3s}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset of arm names")
    ap.add_argument("--force", action="store_true",
                    help="re-run arms even if bfa_results.json already exists")
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(args.arms.split(",")) if args.arms else None
    arms = [a for a in ARMS if wanted is None or a["name"] in wanted]

    cache = out / "reference_cache.json"
    rows_by_name = {}

    # 1) CE reference (read-only)
    ce_row = ce_reference(cache)
    print(f"[ref] {CE_REF_NAME}: S1_ASR={ce_row.get('S1_ASR')} "
          f"flips_to_90={ce_row.get('flips_to_90')} "
          f"final_ASR={ce_row.get('final_ASR')}  (source: {ce_row.get('source')})")

    # 2) arms
    for i, arm in enumerate(arms, 1):
        name, mode, ov = arm["name"], arm["mode"], dict(arm["overrides"])
        arm_dir = out / name
        arm_dir.mkdir(parents=True, exist_ok=True)
        exists = (arm_dir / "bfa_results.json").exists()
        if exists and not args.force:
            print(f"\n=== [{i}/{len(arms)}] {name}: reusing existing {arm_dir.name}/")
            rows_by_name[name] = read_row(name, mode, ov, arm_dir)
            continue

        cmd = [sys.executable, "-u", str(HERE / PIPELINE),
               "--p1-mode", mode, "--out-dir", str(arm_dir)]
        for k, v in ov.items():
            cmd += [CLI_FLAG[k], f"{v:g}"]
        print(f"\n=== [{i}/{len(arms)}] {name}: {' '.join(cmd[2:])} ===", flush=True)
        t0 = time.time()
        with open(arm_dir / "run.log", "w") as fh:
            proc = subprocess.run(cmd, cwd=str(HERE), stdout=fh,
                                  stderr=subprocess.STDOUT)
        wall = time.time() - t0
        row = read_row(name, mode, ov, arm_dir)
        row["wall_s"] = round(wall, 1)
        row["exit"] = proc.returncode
        if proc.returncode != 0:
            row.setdefault("failing", []).append(f"exit={proc.returncode}")
            row["success"] = False
        rows_by_name[name] = row
        print(f"  S1={fmt(row.get('S1_ASR'))}  flips90={row.get('flips_to_90')}  "
              f"final={fmt(row.get('final_ASR'))}  ok={row.get('success')}  "
              f"({wall:.0f}s)")

    rows = [rows_by_name.get(a["name"]) for a in ARMS if a["name"] in rows_by_name]
    rows = [r for r in rows if r]
    all_rows = [ce_row] + rows
    (out / "summary.json").write_text(json.dumps(all_rows, indent=2, default=str))

    print("\n" + table(rows))
    print(f"\n[ref] {CE_REF_NAME}: " + table([ce_row]))

    by_flips = sorted(rows, key=lambda r: (r.get("flips_to_90") is None,
                                           r.get("flips_to_90") or 10 ** 6))
    md = ["# sweep_bc — arms B (EOT) and C (A*) vs cos@64", "",
          f"Gate: flips_to_90 ≤ {MAX_FLIPS_GATE} **and** final ASR ≥ {MIN_ASR_GATE}.",
          "CE is not re-executed; its row is read from the existing controlled run.", "",
          "## In ARMS order", "", "```", table(rows), "```",
          "", "## Sorted by flips_to_90", "", "```", table(by_flips), "```",
          "", f"`{CE_REF_NAME}`: S1_ASR={ce_row.get('S1_ASR')}, "
              f"flips_to_90={ce_row.get('flips_to_90')}, "
              f"final_ASR={ce_row.get('final_ASR')} (source: {ce_row.get('source')})", ""]
    (out / "summary.md").write_text("\n".join(md))
    print(f"\nsummary → {out/'summary.json'} , {out/'summary.md'}")


if __name__ == "__main__":
    main()
