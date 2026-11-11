"""
main.py — Runs one attack for a built config.
============================================
experiment.py parses the CLI, builds the cfg and passes it here; this
module calls AttackPipeline — the attack "method" entry point.

Failure contract: any exception is RECORDED and the process exits cleanly
(exit 0), so the calling shell script simply continues with the next
configuration.
  * CUDA OOM  → oom_pending.json  (deferred work-list: full argv, re-runnable
                later on a bigger GPU);
  * anything else → failed_runs.json (identity + FULL config + error type,
                message and traceback — the debugging log).
KeyboardInterrupt / SystemExit are not swallowed.
"""

import json
import os
import sys
import time
import traceback

import torch

from pipeline import AttackPipeline
from utils import cleanup_gpu

OOM_FILE = "oom_pending.json"
ERROR_FILE = "failed_runs.json"


def is_oom(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _run_identity(cfg):
    return {
        "argv": sys.argv,
        "run_id": cfg.RUN_ID,
        "model": cfg.MODEL_NAME,
        "dataset": cfg.DATASET,
        "seed": cfg.SEED,
        "variant": cfg.VARIANT,
        "out_dir": cfg.OUT_DIR,
    }


def _append(entry, path):
    pending = []
    if os.path.exists(path):
        with open(path) as f:
            pending = json.load(f)
    pending.append(entry)
    with open(path, "w") as f:
        json.dump(pending, f, indent=2, default=str)


def record_oom(cfg, path=OOM_FILE):
    """Append this run's command line + identity to the deferred work-list."""
    entry = _run_identity(cfg)
    entry["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _append(entry, path)


def record_error(cfg, exc, path=ERROR_FILE):
    """Append this run's identity + full config + failure reason to the
    error log (the debugging record for non-OOM failures)."""
    entry = _run_identity(cfg)
    entry["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    entry["error_type"] = type(exc).__name__
    entry["error"] = str(exc)
    entry["traceback"] = traceback.format_exc()
    entry["config"] = {k: v for k, v in vars(cfg).items()
                       if not k.startswith("_")}
    _append(entry, path)


def main(cfg):
    """Run one attack.  Returns the bfa_results dict, or None on skip
    (OOM or recorded error)."""
    print(f"RUN_ID : {cfg.RUN_ID}")
    print(f"OUT_DIR: {cfg.OUT_DIR}")

    try:
        return AttackPipeline(cfg).run()
    except Exception as e:
        cleanup_gpu()
        if is_oom(e):
            record_oom(cfg)
            print(f"\n  OOM — run skipped.  Config saved → {OOM_FILE}")
            print(f"  Re-run later on a bigger GPU from the saved argv.")
            return None
        record_error(cfg, e)
        print(f"\n  ERROR — run skipped and recorded → {ERROR_FILE}")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        return None
