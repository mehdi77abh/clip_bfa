"""
main.py — Runs one attack for a built config.
=============================================
experiment.py parses the CLI, builds the cfg and passes it here; this
module calls AttackPipeline — the attack "method" entry point.

OOM try/catch: on CUDA OOM the run is recorded in oom_pending.json
(with its full argv, so it can be re-run later on a bigger GPU) and we
return cleanly, so the calling shell script simply continues with the
next configuration.
"""

import json
import os
import sys
import time

import torch

from pipeline import AttackPipeline
from utils import cleanup_gpu

OOM_FILE = "oom_pending.json"


def is_oom(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def record_oom(cfg, path=OOM_FILE):
    """Append this run's command line + identity to the deferred work-list."""
    entry = {
        "argv": sys.argv,
        "run_id": cfg.RUN_ID,
        "model": cfg.MODEL_NAME,
        "dataset": cfg.DATASET,
        "seed": cfg.SEED,
        "variant": cfg.VARIANT,
        "out_dir": cfg.OUT_DIR,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    pending = []
    if os.path.exists(path):
        with open(path) as f:
            pending = json.load(f)
    pending.append(entry)
    with open(path, "w") as f:
        json.dump(pending, f, indent=2, default=str)


def main(cfg):
    """Run one attack.  Returns the bfa_results dict, or None on OOM skip."""
    print(f"RUN_ID : {cfg.RUN_ID}")
    print(f"OUT_DIR: {cfg.OUT_DIR}")

    try:
        return AttackPipeline(cfg).run()
    except Exception as e:
        if is_oom(e):
            cleanup_gpu()
            record_oom(cfg)
            print(f"\n  OOM — run skipped.  Config saved → {OOM_FILE}")
            print(f"  Re-run later on a bigger GPU from the saved argv.")
            return None
        raise
