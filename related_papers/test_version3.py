#!/usr/bin/env python3
"""
test_version3.py — full-attack idea harness: Residual λ sweep + embedding
geometry + FT-defence survival (successor of test_version2.py).

Per arm the chain is:

  Phase 0→1→2A→2B→3→4→5   FULL bit-flip attack (tuned search config),
                           --save-attacked-model forced ON
                           → bfa_results.json, trigger.pt, model_state_attacked.pt
  EMBEDDING GEOMETRY       measured twice on the eval set (clean vs poisoned):
                           post-Phase-1 (clean model + trigger) and post-attack
                           (attacked model + trigger) → geometry.json
  FT DEFENCE               defence_finetune.py on the run dir (CC3M clean
                           fine-tune, 2000 steps, lr 1e-5, wd 0.1, batch 64,
                           scope full) → defence_ft.json
                           (ASR/CACC before vs after → survival)

ARMS (agreed 2026-11-27 — RAW λ in the v2 squared-residual parameterisation,
so λ=0.5…2 stays small in gradient terms; the achieved [parity] gradient
ratio is logged per run so the effective strength is documented either way):

  residual λ ∈ {0.5, 0.75, 1.0, 2.0}      --loss residual --lambda-neg X
  residual + visual                        --loss residual --lambda-neg 1.0
                                             --vis-lambda 1.0

CELL: ViT-L/14 × ImageNet1k, target 'lion' (c291), 16px trigger, seed 42,
content-key splits (the v2 machinery is reused verbatim: V2DataModule layout,
V2TriggerGenerator losses, diagnostics: geometry table, [diag], [parity]).

USAGE (GPU box, vlm_attack env):
  bash related_papers/run_v3.sh                 # the 5-arm matrix
  python related_papers/test_version3.py --loss residual --lambda-neg 0.5
  python related_papers/test_version3.py --loss residual --lambda-neg 1 \
      --vis-lambda 1.0 --skip-ft                # attack + geometry only
  python related_papers/test_version3.py ... --dry-run

FT DEFENCE DATA: CC3M is provisioned by the maintainer.  Default root is
<repo>/data/cc3m (override with --cc3m-root).  If the root is missing the
FT stage is SKIPPED with a recorded note — the attack run is not wasted.

RESULTS: related_papers/v3_results/{section}/{model}/{dataset}/{leaf}/
  bfa_results.json  geometry.json  defence_ft.json  model_state_attacked.pt
Notes:
  * v2's --stop-after-phase1 exists in the parser but is ignored here: v3
    runs are full attacks by design.
  * geometry stages share the same 200-img eval set + the 32 target-class
    images (indices captured at split time, so the split fingerprint
    guarantees cross-box identity).
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))   # .../clip_bfa/related_papers
V3_OUT_ROOT = os.path.join(HERE, "v3_results")

sys.path.insert(0, HERE)
import p0_test as p0ref                                  # noqa: E402
import test_version2 as v2                               # noqa: E402

REPO_ROOT = p0ref.REPO_ROOT
MODULAR_DIR = p0ref.MODULAR_DIR
sys.path.insert(0, MODULAR_DIR)

import experiment                                       # noqa: E402
import main as main_mod                                  # noqa: E402
import pipeline as pipeline_mod                          # noqa: E402
from adapters import encode_image                        # noqa: E402
from data import DataModule                              # noqa: E402
from trigger import apply_trigger, get_apply_params      # noqa: E402
from utils import cleanup_gpu, model_tag                # noqa: E402

NEG_LISTS = p0ref.NEG_LISTS
DATASET_DEFAULTS = p0ref.DATASET_DEFAULTS
TRIG_SZ_DEFAULTS = p0ref.TRIG_SZ_DEFAULTS
resolve_target_cls = p0ref.resolve_target_cls

DEFAULT_MODEL = "ViT-L/14"
DEFAULT_DATASET = "ImageNet1k"

V3_STATE = {"eval_idx": None, "target_sel": None}
V3_CFG = {}          # cfg stash for the hooks


# ══════════════════════════════════════════════════════════════════════
#  DATA MODULE — v2 splits + capture of eval/target indices for geometry
# ══════════════════════════════════════════════════════════════════════
class V3DataModule(v2.V2DataModule):

    def prepare_data_splits(self, ds):
        out = super().prepare_data_splits(ds)
        target_sel, _calib, _grad, eval_idx, _test, _hold = out
        V3_STATE["target_sel"] = list(target_sel)
        V3_STATE["eval_idx"] = list(eval_idx)
        return out


# ══════════════════════════════════════════════════════════════════════
#  EMBEDDING GEOMETRY (badclip_dual_ideas.md D2 quantities + compactness)
# ══════════════════════════════════════════════════════════════════════
def _encode_batched(model, imgs, bs):
    outs = []
    for i in range(0, imgs.size(0), bs):
        outs.append(encode_image(model, imgs[i:i + bs]))
    return torch.cat(outs, dim=0)


def compute_geometry(model, trigger, text_emb, ap, cfg):
    """Geometry of the poisoned eval set vs the target-class manifold.

    Eval set (200 non-target) is encoded clean AND poisoned; the 32
    target-class images define the target visual manifold (v̄* and the
    nearest-target columns — BadCLIP D2's L_pi / L_ni are reported as-is).
    """
    dev, mm = cfg.DEVICE, cfg.MINI_BS
    e_idx, t_idx = V3_STATE["eval_idx"], V3_STATE["target_sel"]
    imgs, _ = DataModule.load_tensors(p0ref.STATE.val_ds, e_idx, dev,
                                      batch_size=64, workers=cfg.WORKERS)
    timgs, _ = DataModule.load_tensors(p0ref.STATE.val_ds, t_idx, dev,
                                       batch_size=64, workers=cfg.WORKERS)
    imgs = imgs.to(dev)
    with torch.no_grad():
        Z_c = F.normalize(_encode_batched(model, imgs, mm), dim=-1)
        poisoned = apply_trigger(imgs, trigger, ap)
        Z_p = F.normalize(_encode_batched(model, poisoned, mm), dim=-1)
        Tm = F.normalize(_encode_batched(model, timgs.to(dev), mm), dim=-1)
        v_bar = F.normalize(Tm.mean(dim=0), dim=0)
        t_bar = F.normalize(text_emb[cfg.TARGET_CLS].to(dev), dim=0)
        text_all = text_emb.to(dev)

        cos_t = Z_p @ t_bar
        cos_t_c = Z_c @ t_bar
        cos_v = Z_p @ v_bar
        cos_v_c = Z_c @ v_bar
        near_t = (Z_p @ Tm.T).max(dim=1).values
        own_clean = (Z_p * Z_c).sum(dim=1)
        sims = Z_p @ text_all.T
        tc = cfg.TARGET_CLS
        margin_class = cos_t - sims.max(dim=1).values
        target_share = 100.0 * (sims.argmax(dim=1) == tc).float().mean().item()
        G = Z_p @ Z_p.T
        iu = torch.triu_indices(G.size(0), G.size(0), offset=1)
        compact = G[iu[0], iu[1]].mean().item()

    def _s(t):
        return {"mean": float(t.mean()), "std": float(t.std())}

    return {
        "n_eval": int(cos_t.numel()),
        "cos_t": _s(cos_t), "cos_t_clean": float(cos_t_c.mean()),
        "cos_v_star": _s(cos_v), "cos_v_star_clean": float(cos_v_c.mean()),
        "manifold_delta": float(cos_v.mean() - cos_v_c.mean()),
        "nearest_target_cos": float(near_t.mean()),
        "own_clean_cos": float(own_clean.mean()),
        "compactness": compact,
        "margin_class": float(margin_class.mean()),
        "target_share_pct": target_share,
        "d2_L_pi": float(1.0 - cos_v.mean()),
        "d2_L_ni": float(own_clean.mean() - 1.0),
    }


def merge_geometry(cfg, stage, stats):
    path = os.path.join(cfg.OUT_DIR, "geometry.json")
    data = {}
    if os.path.isfile(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = {}
    data[stage] = stats
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [geometry:{stage}] cos_t={stats['cos_t']['mean']:+.4f}  "
          f"cos(v̄*)={stats['cos_v_star']['mean']:+.4f}  "
          f"Δmanifold={stats['manifold_delta']:+.4f}  "
          f"margin_class={stats['margin_class']:+.4f}  "
          f"target_share={stats['target_share_pct']:.1f}%  "
          f"→ geometry.json")


# ══════════════════════════════════════════════════════════════════════
#  HOOKS — geometry at the two model states
# ══════════════════════════════════════════════════════════════════════
def install_hooks(skip_geometry, skip_final_eval=False):
    orig_gen = pipeline_mod.AttackPipeline._generate_trigger

    def gen_wrap(p_self, model, text_emb, t_target, calib_loader,
                 target_images, eval_imgs, eval_labels, ap, cfg):
        out = orig_gen(p_self, model, text_emb, t_target, calib_loader,
                       target_images, eval_imgs, eval_labels, ap, cfg)
        trig, asr_pre, cacc_pre, _attempts = out
        if not skip_geometry:
            st = compute_geometry(model, trig, text_emb, ap, cfg)
            st["standalone_asr"] = float(asr_pre)
            st["standalone_cacc"] = float(cacc_pre)
            merge_geometry(cfg, "phase1", st)
        return out

    pipeline_mod.AttackPipeline._generate_trigger = gen_wrap

    if skip_geometry and not skip_final_eval:
        return
    orig_fe = pipeline_mod.Evaluator.full_evaluate

    def fe_wrap(e_self, model, text_emb, trigger, loader, target_cls,
                ap, device, *args, **kw):
        cfg = V3_CFG["cfg"]
        if not skip_geometry:
            try:
                st = compute_geometry(model, trigger, text_emb, ap, cfg)
                merge_geometry(cfg, "attacked", st)
            except Exception as e:           # never break the evaluation
                print(f"  [geometry:attacked] FAILED "
                      f"({type(e).__name__}: {e})")
        if not skip_final_eval:
            return orig_fe(e_self, model, text_emb, trigger, loader,
                           target_cls, ap, device, *args, **kw)

        # ── final eval skipped: cheap ASR/CACC on the 200-image search
        #    pool so the SUMMARY/composite/save still have numbers.  The
        #    real 49k post-attack ASR is measured by the FT defence's
        #    BEFORE evaluation (same attacked weights, same test split).
        e_idx = V3_STATE["eval_idx"]
        imgs, labels = DataModule.load_tensors(
            p0ref.STATE.val_ds, e_idx, device,
            batch_size=64, workers=cfg.WORKERS)
        asr, cacc = e_self.quick_evaluate(
            model, text_emb, trigger, imgs, labels, target_cls, ap,
            mini_bs=cfg.MINI_BS)
        print(f"  [final-eval] SKIPPED — cheap {len(e_idx)}-image pool: "
              f"ASR={asr:.2f}%  CACC={cacc:.2f}%  "
              f"(49k post-attack ASR comes from the FT defence)")
        return {"asr": asr, "cacc": cacc, "n": int(len(e_idx)),
                "full_eval_skipped": True,
                "eval_set": "search_pool"}

    pipeline_mod.Evaluator.full_evaluate = fe_wrap


# ══════════════════════════════════════════════════════════════════════
#  FT DEFENCE STAGE (defence_finetune.py, DEFENSES.md §1)
# ══════════════════════════════════════════════════════════════════════
def run_ft_defence(v3, cfg):
    root = os.path.abspath(v3.cc3m_root)
    if not os.path.isdir(root) or not os.listdir(root):
        print(f"\n[FT] CC3M root missing/empty: {root}\n"
              f"     → FT defence SKIPPED (provision the data, then run\n"
              f"       defence_finetune.py --run-dir {cfg.OUT_DIR})")
        return False
    cmd = [sys.executable, os.path.join(MODULAR_DIR, "defence_finetune.py"),
           "--run-dir", cfg.OUT_DIR, "--task", "auto",
           "--cc3m-root", root, "--cc3m-format", "auto",
           "--ft-max-samples", str(v3.ft_max_samples),
           "--ft-epochs", str(v3.ft_epochs),
           "--ft-max-steps", str(v3.ft_max_steps),
           "--ft-lr", str(v3.ft_lr), "--ft-wd", str(v3.ft_wd),
           "--ft-batch", str(v3.ft_batch), "--eval-batch", str(v3.eval_batch),
           "--ft-scope", v3.ft_scope]
    print("\n[FT] fine-tuning defence: " + " ".join(cmd[2:]))
    rc = subprocess.call(cmd, cwd=MODULAR_DIR)
    if rc != 0:
        print(f"[FT] defence_finetune.py exited rc={rc} — see output above")
    return rc == 0


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════
def parse_v3_args(argv):
    p = argparse.ArgumentParser(
        description="test_version3: full attack + geometry + FT defence.  "
                    "test_version2.py and experiment.py flags pass through.")
    p.add_argument("--cc3m-root", default=os.path.join(REPO_ROOT, "data", "cc3m"),
                   help="CC3M root for the FT defence (default: <repo>/data/cc3m)")
    p.add_argument("--ft-max-samples", type=int, default=0,
                   help="CC3M subset size, 0 = all (default: 0)")
    p.add_argument("--ft-epochs", type=int, default=1)
    p.add_argument("--ft-max-steps", type=int, default=2000)
    p.add_argument("--ft-lr", type=float, default=1e-5)
    p.add_argument("--ft-wd", type=float, default=0.1)
    p.add_argument("--ft-batch", type=int, default=32,
                   help="CC3M fine-tune batch (default: 32 — memory)")
    p.add_argument("--eval-batch", type=int, default=32,
                   help="before/after evaluation batch (default: 32)")
    p.add_argument("--ft-scope", default="full", choices=("full", "vision"))
    p.add_argument("--skip-ft", action="store_true",
                   help="run the attack + geometry only")
    p.add_argument("--skip-geometry", action="store_true",
                   help="skip the embedding-geometry measurement")
    p.add_argument("--skip-final-eval", action="store_true",
                   help="skip the 49k ImageNet test-set evaluation after the "
                        "bit-flip; the FT defence's BEFORE eval supplies the "
                        "post-attack ASR (geometry still measured on the "
                        "200-image pool)")
    p.add_argument("--dry-run", action="store_true",
                   help="assemble and print the config, then exit")
    return p.parse_known_args(argv)


def main():
    v3, rest = parse_v3_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    v2ns, rest2 = v2.parse_v2_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest2
    args = experiment.parse_args()

    # defaults mirror v2 (historical cell) unless overridden
    if "--model" not in rest2:
        args.model = DEFAULT_MODEL
    if "--dataset" not in rest2:
        args.dataset = DEFAULT_DATASET
    if "--trigger-size" not in rest2:
        args.trigger_size = TRIG_SZ_DEFAULTS.get(args.model, 16)
    if "--trigger-min-asr" not in rest2:
        args.trigger_min_asr = 0.0     # one deterministic attempt per arm
    if "--trigger-retries" not in rest2:
        args.trigger_retries = 0
    d_target, d_neg = DATASET_DEFAULTS[args.dataset]
    target_name = v2ns.target_name or d_target
    neg_list = v2ns.neg_list or d_neg
    args.target_cls = resolve_target_cls(args.dataset, target_name)

    if v2ns.target_mode == "multi" and v2ns.loss in ("infonce", "avgsoftmax"):
        raise SystemExit("--target-mode multi is only defined for the "
                         "L⁺-form losses (cos, residual)")

    # Version-skew shim (same as test_version2.py)
    import config as _config
    if not hasattr(_config.CFG, "FAR_CKPT"):
        _config.CFG.FAR_CKPT = None

    cfg = experiment.build_config(args)

    # ── v2 loss/settings (V2TriggerGenerator / V2DataModule consume these)
    cfg.V2_LOSS = v2ns.loss
    cfg.V2_TARGET_MODE = v2ns.target_mode
    cfg.V2_TARGET_TEMPLATE = v2ns.target_template
    cfg.V2_LAMBDA_NEG = v2ns.lambda_neg
    cfg.V2_VIS_LAMBDA = v2ns.vis_lambda
    cfg.V2_VIS_L2 = v2ns.vis_lambda2
    cfg.V2_VIS_ETA = v2ns.vis_eta
    cfg.V2_CARRIER_RULE = v2ns.carrier_rule
    cfg.V2_SELECT = v2ns.select
    cfg.P0_ARM = "p2txt"
    cfg.COMPETITOR_MODE = "text"
    cfg.NEG_LIST = cfg.P0_NEG_LIST = neg_list
    cfg.NEG_COS_MAX = v2ns.neg_cos_max
    cfg.SPLIT_KEY = v2ns.split_key
    cfg.N_POOL = 256
    cfg.TDA_MODE = "off"
    cfg.P1_MARGIN = 0.0
    cfg.P1_LOSS = "cos"               # pipeline's infonce prebuild stays OFF
    cfg.SAVE_ATTACKED_MODEL = True    # FT defence needs the state dict

    # ── run identity: v3 leaf, results under v3_results ──
    tag = v2.v2_tag(v2ns).replace("v2_", "v3_", 1)
    cfg.RUN_ID += f"__{tag}"
    cfg.OUT_DIR = cfg.OUT_DIR + f"_{tag}"
    if os.path.abspath(cfg.OUT_DIR).startswith(os.path.abspath(cfg.OUT_ROOT)):
        rel = os.path.relpath(cfg.OUT_DIR, cfg.OUT_ROOT)
    else:
        rel = os.path.join(cfg.SECTION, model_tag(cfg.MODEL_NAME),
                           cfg.DATASET, os.path.basename(cfg.OUT_DIR))
    cfg.OUT_ROOT = V3_OUT_ROOT
    cfg.OUT_DIR = os.path.join(cfg.OUT_ROOT, rel)

    print("=" * 60)
    print(f"TEST V3 — {tag}")
    print("=" * 60)
    print(f"  model={cfg.MODEL}  dataset={cfg.DATASET}  "
          f"target='{target_name}' (cls {cfg.TARGET_CLS})")
    print(f"  loss={v2ns.loss}  λ_neg={v2ns.lambda_neg:g}  "
          f"vis={v2ns.vis_lambda:g}  select={v2ns.select}")
    print(f"  neg-list={neg_list}  trigger={args.trigger_size}px  "
          f"seed={cfg.SEED}  task=zs  save-attacked-model=ON")
    print(f"  chain: full attack → geometry"
          f"{'' if v3.skip_geometry else ' (phase1 + attacked)'}"
          + ((" → final eval SKIPPED (attack-only; no 49k ASR)"
              if v3.skip_ft else
              " → final eval SKIPPED (FT before-eval = 49k post-attack ASR)")
             if v3.skip_final_eval else " → final 49k eval")
          + (" → FT defence skipped (--skip-ft)" if v3.skip_ft else
             f" → FT defence (CC3M: {v3.cc3m_root})"))
    print(f"  out: {cfg.OUT_DIR}")

    if v3.skip_final_eval and v3.skip_ft:
        print("  NOTE: attack-only run — no 49k test ASR will be measured "
              "(200-image pool metrics only; no FT).")

    if v3.dry_run:
        print("\n[dry-run] config assembled OK — exiting before model load.")
        return

    # ── install patches + hooks, then run the standard pipeline ──
    pipeline_mod.DataModule = V3DataModule
    pipeline_mod.TargetEmbedder = v2.V2TargetEmbedder
    pipeline_mod.TriggerGenerator = v2.V2TriggerGenerator
    install_hooks(v3.skip_geometry, v3.skip_final_eval)
    V3_CFG["cfg"] = cfg

    result = None
    try:
        result = main_mod.main(cfg)      # OOM/error contract of main.py applies
    except SystemExit as e:
        if e.code not in (0, None):
            raise

    if result is not None and not v3.skip_ft:
        # Release the parent's CUDA blocks before the FT subprocess loads
        # a second copy of the model — the chained-FT OOM cause.
        try:
            cleanup_gpu()
        except Exception:
            pass
        run_ft_defence(v3, cfg)
    try:
        experiment.refresh_summaries(cfg.OUT_ROOT)
    except Exception as e:
        print(f"  (summary refresh skipped: {type(e).__name__}: {e})")
    return result


if __name__ == "__main__":
    main()
