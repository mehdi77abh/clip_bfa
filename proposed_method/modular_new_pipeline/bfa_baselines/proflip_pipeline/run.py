"""
run.py — CLI entry point for the ProFlip-CBS baseline.

Reuses `modular_new_pipeline/experiment.py` for the shared CLI/config
(model, dataset, splits, Phase-1 trigger hyperparameters), then overrides
the run identity:

    --section   → baselines
    --variant   → proflip_cbs
    Phase 2     → skipped (full component pool)
    Phase 3     → skipped (CBS computes its own gradients)
    CBS         → paper values: ASR_t = 94, n_max = 100; target-cosine

All results go under `bfa_baselines/baseline_runs/` by default, with the
same flattened leaf as the proposed method:
    {out_root}/baselines/{model}/{dataset}/{task}_c{cls}_s{seed}_r{rep}_{run_seg}

Usage:
    cd bfa_baselines
    python proflip_pipeline/run.py --model ViT-B/16 --dataset Caltech101 \
        --seed 42 --target-cls 97 --trigger-size 16 --n-eval 1000

The proposed-method pipeline itself is never modified.
"""

import argparse
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_BFA_DIR = os.path.abspath(os.path.join(_HERE, ".."))
_PIPE_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_PIPE_DIR, _BFA_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import experiment as base_experiment          # noqa: E402  (proposed CLI)
import main as proposed_main                  # noqa: E402  (OOM/error contract)
from utils import cleanup_gpu, model_tag      # noqa: E402

from pipeline_impl import ProFlipCBSPipeline  # noqa: E402
from proflip_cbs import CBSSettings           # noqa: E402


def _parse_cbs_cli(argv):
    """Parse the --cbs-* flags and return (namespace, remaining argv)."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--cbs-k", type=int, default=20,
                   help="S3 grid points (ProFlip paper: K=20)")
    p.add_argument("--cbs-asr-target", type=float, default=94.0,
                   help="stop ASR%% (ProFlip paper: 94)")
    p.add_argument("--cbs-nmax", type=int, default=100,
                   help="max accepted bit flips (ProFlip paper: 100)")
    p.add_argument("--cbs-clean-mse-weight", type=float, default=0.0,
                   help="optional clean-feature MSE term (0 = target-cosine only)")
    p.add_argument("--cbs-q-mode", choices=["abs", "signed"], default="abs",
                   help="Q_p: paper max|w| (abs) or released code max(w)")
    p.add_argument("--cbs-no-monotone", action="store_true",
                   help="allow decreasing an element (released code forbids)")
    ns, rest = p.parse_known_args(argv)
    return ns, rest


def _apply_proflip_identity(cfg, args, cbs_cli):
    """Override the proposed-method run identity for the CBS baseline."""
    cfg.SECTION = "baselines"
    cfg.VARIANT = "proflip_cbs"
    cfg.SKIP_PHASE2 = True               # full component pool
    cfg.P3_ALPHA_LSE = 0.0
    cfg.SKIP_CLEAN_PENALTY = False
    cfg.POISON_ONLY_GRAD = False
    cfg.CBS_METHOD = "proflip_cbs"

    settings = CBSSettings.from_cfg(
        cfg,
        grid_k=cbs_cli.cbs_k,
        asr_target=cbs_cli.cbs_asr_target,
        nmax=cbs_cli.cbs_nmax,
        clean_mse_weight=cbs_cli.cbs_clean_mse_weight,
        q_mode=cbs_cli.cbs_q_mode,
        monotone=not cbs_cli.cbs_no_monotone,
    )
    for k, v in settings.as_cfg_attrs().items():
        setattr(cfg, k, v)

    # Result root: bfa_baselines/baseline_runs by default.
    if args.out_root is None:
        cfg.OUT_ROOT = os.path.join(_BFA_DIR, "baseline_runs")

    tag = model_tag(cfg.MODEL_NAME)
    run_seg = f"proflip_cbs_t{cfg.TRIG_SZ}"
    if getattr(cfg, "TRIGGER_LOCATION", None):
        run_seg += f"_{cfg.TRIGGER_LOCATION}"
    leaf = (f"{cfg.TASK}_c{cfg.TARGET_CLS}_s{cfg.SEED}_r{cfg.REP}"
            f"_{run_seg}")
    cfg.OUT_DIR = args.out_dir or os.path.join(
        cfg.OUT_ROOT, cfg.SECTION, tag, cfg.DATASET, leaf)
    cfg.RUN_ID = (f"{cfg.SECTION}__{tag}__{cfg.DATASET}__{cfg.TASK}__"
                  f"c{cfg.TARGET_CLS}__{run_seg}__s{cfg.SEED}__r{cfg.REP}")
    return cfg, settings


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cbs_cli, rest = _parse_cbs_cli(argv)
    sys.argv = [sys.argv[0]] + rest

    args = base_experiment.parse_args()
    cfg = base_experiment.build_config(args)
    cfg, settings = _apply_proflip_identity(cfg, args, cbs_cli)

    print(f"RUN_ID : {cfg.RUN_ID}")
    print(f"OUT_DIR: {cfg.OUT_DIR}")
    print(f"[CBS] K={settings.grid_k}  ASR_t={settings.asr_target}%  "
          f"n_max={settings.nmax}  objective=target-cosine  "
          f"clean_mse={settings.clean_mse_weight}")

    try:
        result = ProFlipCBSPipeline(cfg).run()
        return 0 if result is not None else 1
    except Exception as exc:
        cleanup_gpu()
        if proposed_main.is_oom(exc):
            proposed_main.record_oom(cfg)
            print("\n  OOM — run skipped and recorded in oom_pending.json")
        else:
            proposed_main.record_error(cfg, exc)
            print(f"\n  ERROR — run skipped and recorded: "
                  f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
