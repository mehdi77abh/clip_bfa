"""
parse_flip_s_log.py — Turn a Flip-S ZS run log into a baseline JSON record.
==========================================================================

`Flip-S/target-clip/flip_s_clip_v2.py` prints the ZS comparison metrics through
`lp_eval_full`:

    [+ ZS eval (before attack, n=50000): ZS-CACC 68.30%  ZS-ASR 0.10%  clean-as-target 0.00%
    [+ ZS eval (after 50 flips, n=50000): ZS-CACC 68.10%  ZS-ASR 97.50%  clean-as-target 0.20%

This parser extracts the last "after" line (plus the "before" baseline) and
writes `flip_s_results.json` next to the log, so aggregate_baselines.py can
read Flip-S rows without modifying the Flip-S method.

Usage:
    python parse_flip_s_log.py --log path/run.log --out path/flip_s_results.json \
        --model ViT-B/16 --dataset ImageNet1k --target-cls 3 --seed 0
"""

import argparse
import json
import os
import re

EVAL_RE = re.compile(
    r"\[\+\]\s+ZS eval\s+\((?P<tag>[^)]*),\s*n=(?P<n>\d+)\):\s*"
    r"ZS-CACC\s+(?P<cacc>[\d.]+)%\s+"
    r"ZS-ASR\s+(?P<asr>[\d.]+)%\s+"
    r"clean-as-target\s+(?P<cat>[\d.]+)%")
FLIP_RE = re.compile(r"\[\+\]\s+Flip\s+(\d+)\s+bit")


def parse_log(path):
    before = after = None
    flips = None
    with open(path, errors="ignore") as f:
        for line in f:
            m = EVAL_RE.search(line)
            if m:
                rec = dict(n=int(m.group("n")),
                           cacc=float(m.group("cacc")),
                           asr=float(m.group("asr")),
                           cat=float(m.group("cat")),
                           tag=m.group("tag"))
                if "before" in rec["tag"].lower():
                    before = rec
                else:
                    after = rec
            m = FLIP_RE.search(line)
            if m:
                flips = int(m.group(1))
    return before, after, flips


def main():
    p = argparse.ArgumentParser(description="Parse a Flip-S ZS log")
    p.add_argument("--log", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--target-cls", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rep", type=int, default=1)
    p.add_argument("--task", default="zs")
    p.add_argument("--trigger-size", default="9x16")
    p.add_argument("--trigger-area", type=float, default=4.59)
    args = p.parse_args()

    before, after, flips = parse_log(args.log)
    if after is None:
        raise SystemExit(f"[-] no 'ZS eval (after ...)' line in {args.log}")

    rec = {
        "method": "flip_s",
        "model": args.model,
        "dataset": args.dataset,
        "task": args.task,
        "target_cls": args.target_cls,
        "seed": args.seed,
        "rep": args.rep,
        "trigger_size": args.trigger_size,
        "trigger_area": args.trigger_area,
        "metrics": {
            "zs_cacc": after["cacc"],
            "zs_asr": after["asr"],
            "clean_as_target": after["cat"],
            "n_eval": after["n"],
            "flips": flips,
            "zs_cacc_base": before["cacc"] if before else None,
            "zs_asr_base": before["asr"] if before else None,
            "clean_as_target_base": before["cat"] if before else None,
        },
        "source_log": os.path.abspath(args.log),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[flip-s] {rec['metrics']['zs_cacc']:.2f}% CACC / "
          f"{rec['metrics']['zs_asr']:.2f}% ASR  ({flips} flips) → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
