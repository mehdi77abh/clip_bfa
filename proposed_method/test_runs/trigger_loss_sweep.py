"""
trigger_loss_sweep.py — Phase-1 trigger-loss sweep on Caltech-101 / ViT-B/16
============================================================================
Runs `main_new_pipeline_trigger_test.py` once per loss arm (fresh process, so
each arm gets a clean GPU + model) and tabulates the results.

Stage "phase1"  → MAX_FLIPS=0: measures the *raw trigger* (standalone ASR,
                  CACC, cos to target, trigger norm). ~4-5 min per arm.
Stage "full"    → complete pipeline: flips-to-90% / 95% ASR, final ASR/CACC,
                  mean Δcos per flip. ~10 min per arm.

Arm spec:  <mode>[:<param>]      param = P1_LAMBDA_T2T for *_t2t,
                                 P1_MARGIN_LOGIT for margin_ce.
Modes (see phase1_objective): ce | cos | cos_t2t | ce_t2t | margin_ce | composite

Examples
--------
  python trigger_loss_sweep.py --stage phase1
  python trigger_loss_sweep.py --stage phase1 --arms ce,cos,cos_t2t:0.05,cos_t2t:0.2
  python trigger_loss_sweep.py --stage full --arms ce,cos_t2t:0.05,cos_t2t:0.2

Outputs: <out>/arm_<name>/bfa_results.json, trigger.pt, plots
         <out>/sweep_summary.json + the comparison table on stdout.
"""

import argparse, json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE = "main_new_pipeline_trigger_test.py"

DEFAULT_ARMS = ["ce", "cos", "cos_t2t:0.05", "cos_t2t:0.2",
                "ce_t2t:0.05", "ce_t2t:0.2", "margin_ce:1.0"]

CHILD = """
import sys
sys.path.insert(0, {here!r})
import main_new_pipeline_trigger_test as M

M.CFG.S1_ENABLED = False
M.CFG.S2_ENABLED = False
M.CFG.P1_LOSS = {mode!r}
M.CFG.P1_LAMBDA_T2T = {lam}
M.CFG.P1_MARGIN_LOGIT = {marg}
M.CFG.P1_EPOCHS = {epochs}
M.CFG.MAX_FLIPS = {max_flips}
M.CFG.OUT_DIR = {out!r}
M.main()
"""


def parse_arm(spec):
    mode, _, param = spec.partition(":")
    mode = mode.strip()
    lam, marg = 0.05, 1.0
    if param:
        if mode in ("cos_t2t", "ce_t2t"):
            lam = float(param)
        elif mode == "margin_ce":
            marg = float(param)
        else:
            sys.exit(f"arm {spec!r}: mode {mode!r} takes no parameter")
    if mode not in ("ce", "cos", "cos_t2t", "ce_t2t", "margin_ce", "composite"):
        sys.exit(f"unknown mode {mode!r} in arm {spec!r}")
    name = mode if not param else f"{mode}_{param}"
    return name, dict(mode=mode, lam=lam, marg=marg)


def flips_at(checkpoints, target):
    for n, asr, _ in checkpoints:
        if asr >= target:
            return n
    return None


def run_arm(spec, stage, root, epochs, max_flips_full):
    name, p = parse_arm(spec)
    out = root / f"arm_{name}"
    out.mkdir(parents=True, exist_ok=True)
    max_flips = 0 if stage == "phase1" else max_flips_full
    code = CHILD.format(here=str(HERE), mode=p["mode"], lam=p["lam"],
                        marg=p["marg"], epochs=epochs,
                        max_flips=max_flips, out=str(out))
    log = out / "run.log"
    t0 = time.time()
    print(f"\n=== arm {name}: mode={p['mode']} lam={p['lam']} "
          f"margin={p['marg']} flips={max_flips} ===", flush=True)
    with open(log, "w") as fh:
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(HERE),
                              stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    res = dict(arm=name, spec=spec, mode=p["mode"], lam=p["lam"],
               margin_logit=p["marg"], max_flips=max_flips,
               wall_sec=round(dt, 1), exit=proc.returncode)

    jpath = out / "bfa_results.json"
    if proc.returncode != 0 or not jpath.exists():
        res["error"] = f"exit={proc.returncode}, json={'ok' if jpath.exists() else 'missing'}"
        print(f"  FAILED ({res['error']}) — see {log}")
        return res

    d = json.loads(jpath.read_text())
    post = d.get("results", {}).get("post_flip_test", {})
    cps = d.get("checkpoints", [])
    flips = d.get("flip_log", [])
    trig_norm = None
    tpath = out / "trigger.pt"
    if tpath.exists():
        import torch
        trig_norm = round(torch.load(tpath, map_location="cpu").float().norm().item(), 3)

    res.update(
        standalone_asr=round(cps[0][1], 2) if cps else None,
        standalone_cacc=round(cps[0][2], 2) if cps else None,
        n_flips=len(flips),
        flips_to_90=flips_at(cps, 90.0),
        flips_to_95=flips_at(cps, 95.0),
        final_asr=round(post.get("asr", float("nan")), 2),
        final_cacc=round(post.get("cacc", float("nan")), 2),
        cos_poison=round(post.get("cos_poison", float("nan")), 4),
        trig_norm=trig_norm,
        mean_delta_cos=(round(sum(f["delta_cos"] for f in flips) / len(flips), 5)
                        if flips else None),
        search_sec=round(d.get("search_time_sec", 0.0), 1),
    )
    print(f"  standalone ASR={res['standalone_asr']}%  CACC={res['standalone_cacc']}%  "
          f"flips={res['n_flips']}  final ASR={res['final_asr']}%  "
          f"({res['wall_sec']}s)")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["phase1", "full"], default="phase1")
    ap.add_argument("--arms", default=None,
                    help="comma list, e.g. ce,cos,cos_t2t:0.05 (default: %(default)s)")
    ap.add_argument("--epochs", type=int, default=None, help="override P1_EPOCHS")
    ap.add_argument("--max-flips", type=int, default=80, help="stage=full only")
    ap.add_argument("--out", default="./trigger_loss_sweep")
    args = ap.parse_args()

    arms = (args.arms.split(",") if args.arms else DEFAULT_ARMS)
    arms = [a.strip() for a in arms if a.strip()]
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location("m", HERE / PIPELINE)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    epochs = args.epochs if args.epochs else m.CFG.P1_EPOCHS

    print(f"stage={args.stage}  arms={arms}  epochs={epochs}  out={root}")
    results = [run_arm(a, args.stage, root, epochs, args.max_flips) for a in arms]

    (root / "sweep_summary.json").write_text(json.dumps(results, indent=2))

    hdr = f"{'arm':16s} {'st.ASR':>7s} {'CACC':>6s} {'flips':>6s} {'→90':>5s} {'→95':>5s} " \
          f"{'ASR':>6s} {'cos_poi':>8s} {'Δcos/fl':>8s} {'‖trig‖':>7s} {'sec':>6s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(f"{r['arm']:16s}  ERROR: {r['error']}")
            continue
        print(f"{r['arm']:16s} {r['standalone_asr']:7.1f} {r['standalone_cacc']:6.1f} "
              f"{r['n_flips']:6d} {str(r['flips_to_90']):>5s} {str(r['flips_to_95']):>5s} "
              f"{r['final_asr']:6.1f} {r['cos_poison']:8.4f} "
              f"{str(r['mean_delta_cos']):>8s} {str(r['trig_norm']):>7s} {r['search_sec']:6.1f}")
    print(f"\nsummary → {root / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
