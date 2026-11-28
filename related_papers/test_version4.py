#!/usr/bin/env python3
"""
test_version4.py — cos + visual trigger generation (no residual term).

Why this file exists
--------------------
test_version3.py tests the *residual* repulsion (competitor text negatives)
optionally combined with the visual-manifold hinge.  This v4 harness isolates
the **visual** part:

    L = (1 - <z, t_bar>)  +  λv · relu( L_pi + λ2·L_ni + η )

with NO  λ·mean_r <z, res_r>²  competitor term.  It answers the question:
"does the visual-manifold hinge alone reproduce the geometry benefit, and at
what ASR/flip cost?" — the clean ablation of the residual term.

Everything else is the v3 machinery, reused verbatim:
  * V2/V3 data module, split-key content hashing, diagnostics
  * full bit-flip attack (Phases 0-4) + geometry.json
  * FT defence chain (defence_finetune.py) with the fast budget

Differences from test_version3.py
---------------------------------
  * default loss          : --loss cos   (residual is REJECTED with a message)
  * default visual weight : --vis-lambda 1
  * output root           : related_papers/v4_results
  * run tag               : v4_...  (v3 runs are never overwritten)

Usage (GPU box, vlm_attack env):
  python related_papers/test_version4.py --seed 42
  python related_papers/test_version4.py --seed 42 --vis-lambda 0.25
  python related_papers/test_version4.py --seed 42 --eval-max-samples 0 \
      --ft-max-steps 2000 --eval-batch 64      # full-budget version
  python related_papers/test_version4.py --dry-run

The defaults are the fast v4 budget: --skip-final-eval, --ft-max-steps 500,
--eval-max-samples 10000, --eval-batch 64, --ft-batch 16.  Every one of them
can be overridden on the command line.

NOTE: --vis-lambda is the only visual knob that matters for the method; the
defaults λ2=1, η=1 come from the v2 generator (`--vis-lambda2` / `--vis-eta`
also pass through if you want to study them).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V4_OUT_ROOT = os.path.join(HERE, "v4_results")

sys.path.insert(0, HERE)
import test_version3 as v3                              # noqa: E402


def _flag_value(argv, flag):
    """Return the value of --flag X or --flag=X, or None if absent."""
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _has(argv, flag):
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def _ensure(argv, flag, value=None):
    """Append --flag value if the user did not pass it."""
    if _has(argv, flag):
        return
    argv.append(flag)
    if value is not None:
        argv.append(str(value))


def main():
    argv = list(sys.argv[1:])

    loss = _flag_value(argv, "--loss")
    if loss is not None and loss != "cos":
        raise SystemExit(
            f"test_version4 is cos+visual only (got --loss {loss}). "
            f"Use test_version3.py for residual runs.")
    if loss is None:
        argv += ["--loss", "cos"]

    _ensure(argv, "--vis-lambda", 1)

    # Fast v4 budget (override on the command line if needed):
    #   500 FT steps, paired 10k before/after eval, no pipeline 49k eval.
    _ensure(argv, "--skip-final-eval")
    _ensure(argv, "--ft-max-steps", 500)
    _ensure(argv, "--eval-max-samples", 10000)
    _ensure(argv, "--eval-batch", 64)
    _ensure(argv, "--ft-batch", 16)

    # v4 output tree + tag (never touch v3_results)
    v3.V3_OUT_ROOT = V4_OUT_ROOT
    _orig_tag = v3.v2.v2_tag

    def _v4_tag(ns):
        return _orig_tag(ns).replace("v2_", "v4_", 1)

    v3.v2.v2_tag = _v4_tag

    print("=" * 60)
    print("TEST V4 — cos + visual trigger generation (no residual term)")
    print("=" * 60)
    sys.argv = [sys.argv[0]] + argv
    v3.main()


if __name__ == "__main__":
    main()
