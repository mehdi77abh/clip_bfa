"""
experiment.py — Single-run CLI: argument parser + config builder.
================================================================
One invocation = one attack run.  experiment.py parses the arguments,
builds the cfg from registry + arguments, and passes it to main.py,
which calls AttackPipeline.

Multi-run grids are plain shell scripts that invoke this command once
per configuration; on CUDA OOM the run is recorded in oom_pending.json
(main.py) and the process exits cleanly so the script continues with
the next configuration.

Usage:
    python experiment.py --model ViT-B/16 --dataset Caltech101 --seed 42
    python experiment.py --model ViT-L/14 --dataset CIFAR10 --seed 43 \\
                         --trigger-size 14 --variant trig14

Shell multi-run example:
    for seed in 42 43 44; do
        python experiment.py --model ViT-B/16 --dataset Caltech101 --seed $seed
    done

python experiment.py --help   →   all options (defaults = tuned values).
"""

import argparse
import os

import torch

from config import CFG, load_registry, resolve_path
from main import main as run_attack
from utils import model_tag, resolve_device


# ══════════════════════════════════════════════════════════════════════
#  ARGUMENTS (defaults = tuned values from main_new_pipeline.py)
# ══════════════════════════════════════════════════════════════════════
def parse_args():
    reg = load_registry()
    parser = argparse.ArgumentParser(
        description="CLIP-FLIP hybrid bit-flip attack — single run")

    sel = parser.add_argument_group("run selection")
    sel.add_argument("--model", default="ViT-B/16",
                     choices=sorted(reg["models"]),
                     help="model registry name (default: ViT-B/16)")
    sel.add_argument("--dataset", default="Caltech101",
                     choices=sorted(reg["datasets"]),
                     help="dataset registry name (default: Caltech101)")
    sel.add_argument("--task", default="zs", choices=["zs", "lp", "i2t"],
                     help="evaluation task: zs = zero-shot (implemented) | "
                          "lp = linear-probe (needs a trained probe head) | "
                          "i2t = image-to-text retrieval (future — see TASKS.md)")
    sel.add_argument("--section", default="validation",
                     choices=["validation", "baselines", "ablation"],
                     help="results section — top-level output group "
                          "(default: validation; baselines/ablation for the "
                          "comparison and ablation grids)")
    sel.add_argument("--seed", type=int, default=42,
                     help="random seed (default: 42)")
    sel.add_argument("--rep", type=int, default=1,
                     help="repetition index for same-seed repeats "
                          "(default: 1); part of the output path")

    run = parser.add_argument_group("run")
    run.add_argument("--target-cls", type=int, default=3,
                     help="target class index (default: 3)")
    run.add_argument("--trigger-type", default="center_add",
                     choices=["center_add", "corner_add", "opacity", "replace"],
                     help="trigger mode (default: center_add)")
    run.add_argument("--trigger-location", default=None,
                     choices=["center", "top_left", "top_right",
                              "bottom_left", "bottom_right"],
                     help="trigger placement (train + eval); default: legacy "
                          "mapping of --trigger-type (center_add → center, "
                          "others → bottom-right)")
    run.add_argument("--trigger-size", type=int, default=14,
                     help="trigger patch size in pixels (default: 14)")
    run.add_argument("--max-flips", type=int, default=80,
                     help="bit-flip budget (default: 80)")
    run.add_argument("--asr-target", type=float, default=100.0,
                     help="stop when ASR reaches this %% (default: 100; "
                          "0 disables the target stop)")
    run.add_argument("--asr-fine-threshold", type=float, default=90.0,
                     help="fine regime: at/above this ASR the ranking is "
                          "rebuilt after every accepted flip and the "
                          "fine-window give-up applies (default: 90)")
    run.add_argument("--asr-fine-patience", type=int, default=3,
                     help="fine regime: length of the give-up window in "
                          "accepted flips (default: 3)")
    run.add_argument("--asr-fine-min-gain", type=float, default=1.0,
                     help="fine regime: minimum ASR gain (%% points) required "
                          "per patience window, else stop (default: 1.0)")
    run.add_argument("--variant", default="full",
                     choices=["full", "d", "a", "b", "c", "clip_flip"],
                     help="ablation variant (ABLATION.md): full/d/clip_flip = "
                          "proposed method unchanged | a = no Phase 2 pruning | "
                          "b = no clean penalty | c = poison-only gradient")
    run.add_argument("--device", default="auto",
                     help="cuda | cpu | auto (default: auto)")
    run.add_argument("--download", action="store_true",
                     help="allow torchvision to download datasets")
    run.add_argument("--out-dir", default=None,
                     help="explicit output dir (overrides the derived layout)")
    run.add_argument("--out-root", default=None,
                     help="override registry paths.output_root")

    data = parser.add_argument_group("data & splits")
    data.add_argument("--batch", type=int, default=8)
    data.add_argument("--mini-bs", type=int, default=4)
    data.add_argument("--workers", type=int, default=2)
    data.add_argument("--n-target-images", type=int, default=None,
                      help="override registry splits.n_trigger "
                           "(target-class images)")
    data.add_argument("--n-calib", type=int, default=None,
                      help="override registry splits.n_trigger "
                           "(trigger-training / ablation calibration)")
    data.add_argument("--n-grad", type=int, default=None,
                      help="override registry splits.n_gradient")
    data.add_argument("--n-eval", type=int, default=None,
                      help="override registry splits.n_eval")

    p1 = parser.add_argument_group("phase 1 — trigger training")
    p1.add_argument("--p1-epochs", type=int, default=200)
    p1.add_argument("--p1-lr", type=float, default=0.02)
    p1.add_argument("--tv-weight", type=float, default=0.0)
    p1.add_argument("--temperature", type=float, default=0.07)
    p1.add_argument("--epsilon", type=float, default=8.0 / 255.0,
                    help="pixel clamp (unused for additive triggers)")
    p1.add_argument("--opacity", type=float, default=1.0)
    p1.add_argument("--lambda-vis", type=float, default=0.0)

    p2 = parser.add_argument_group("phase 2 — ablation / pruning")
    p2.add_argument("--top-k-layers", type=int, default=4)
    p2.add_argument("--top-k-components", type=int, default=30)
    p2.add_argument("--per-layer-quota",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="per-layer component quota (default: on; "
                         "--no-per-layer-quota = global top-K)")

    rank = parser.add_argument_group("phase 3/4 — ranking & search")
    rank.add_argument("--gradient-beta", type=float, default=10.0)
    rank.add_argument("--sens-top-m", type=int, default=150)
    rank.add_argument("--eval-bs", type=int, default=32)
    rank.add_argument("--cands-per-round", type=int, default=30)
    rank.add_argument("--plateau-patience", type=int, default=5)
    rank.add_argument("--refresh-every", type=int, default=5)
    rank.add_argument("--checkpoint-every", type=int, default=10)
    rank.add_argument("--clean-drift-penalty", type=float, default=10.0)
    rank.add_argument("--max-clean-drift", type=float, default=0.02)

    misc = parser.add_argument_group("misc")
    misc.add_argument("--quant-tolerance", type=float, default=2.0)
    misc.add_argument("--lp-head", default=None,
                      help="linear-probe checkpoint for --task lp (default: "
                           "{lp_head_root}/{model_tag}_{dataset}.pt)")
    misc.add_argument("--save-attacked-model", action="store_true",
                      help="save model_state_attacked.pt (attacked state_dict) "
                           "for defence / VLM-transfer evaluation — off by "
                           "default; grids enable it selectively")

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
#  CONFIG ASSEMBLY
# ══════════════════════════════════════════════════════════════════════
# Full-method names all mean the same: the proposed method unchanged.
VARIANT_ALIASES = {"d": "full", "clip_flip": "full"}


def build_config(args):
    """Merge registry entry + CLI args into one cfg object.

    The returned object exposes exactly the attribute names read by the
    pipeline and phase modules (MODEL, TRIG_SZ, N_CALIB, OUT_DIR, …).
    """
    reg = load_registry()
    models, datasets, paths = reg["models"], reg["datasets"], reg["paths"]

    model, dataset = args.model, args.dataset
    if model not in models:
        raise ValueError(f"Unknown model {model!r}. "
                         f"Registry has: {sorted(models)}")
    if dataset not in datasets:
        raise ValueError(f"Unknown dataset {dataset!r}. "
                         f"Registry has: {sorted(datasets)}")

    ment = models[model]
    dent = datasets[dataset]
    splits = dent.get("splits", {})

    seed = int(args.seed)
    variant = VARIANT_ALIASES.get(args.variant, args.variant)
    out_root = resolve_path(args.out_root or paths["output_root"])
    tag = model_tag(model)

    cfg = CFG()

    # ── model ──
    cfg.MODEL = ment.get("clip_name") or ment["hf_name"]
    cfg.MODEL_NAME = model
    cfg.MODEL_ADAPTER = ment.get("adapter", "openai_clip")
    cfg.HF_NAME = ment.get("hf_name")
    cfg.DEVICE = resolve_device(args.device)
    # geometry is auto-detected at load; registry values are hints only
    cfg.PATCH_SIZE = ment.get("patch_size", 14)
    cfg.IMG_SIZE = dent.get("img_size", 224)

    # ── dataset (registry entry travels with cfg; data.py consumes it) ──
    cfg.DATASET = dataset
    cfg.DATASET_NAME = dataset
    cfg.DATASET_CFG = dent
    cfg.CLASS_TEXT_DIR = dent.get("class_text_dir")
    cfg.DATA = resolve_path(dent["root"])
    cfg.DOWNLOAD = bool(args.download)
    cfg.N_TARGET_IMAGES = _or(args.n_target_images, splits.get("n_trigger", 32))
    cfg.N_CALIB = _or(args.n_calib, splits.get("n_trigger", 32))
    cfg.N_GRAD = _or(args.n_grad, splits.get("n_gradient", 16))
    cfg.N_EVAL = _or(args.n_eval, splits.get("n_eval", 200))
    cfg.BATCH = args.batch
    cfg.MINI_BS = args.mini_bs
    cfg.WORKERS = args.workers

    # ── attack settings (from arguments) ──
    cfg.TARGET_CLS = args.target_cls
    cfg.TRIGGER_TYPE = args.trigger_type
    cfg.TRIGGER_LOCATION = args.trigger_location  # None = legacy mapping
    cfg.TRIG_SZ = args.trigger_size
    cfg.MAX_FLIPS = args.max_flips
    cfg.ASR_TARGET = args.asr_target              # 0 = no target stop
    cfg.ASR_FINE_THRESHOLD = args.asr_fine_threshold
    cfg.ASR_FINE_PATIENCE = args.asr_fine_patience
    cfg.ASR_FINE_MIN_GAIN = args.asr_fine_min_gain

    cfg.P1_EPOCHS = args.p1_epochs
    cfg.P1_LR = args.p1_lr
    cfg.TV_WEIGHT = args.tv_weight
    cfg.TEMPERATURE = args.temperature
    cfg.EPSILON = args.epsilon
    cfg.OPACITY = args.opacity
    cfg.LAMBDA_VIS = args.lambda_vis

    cfg.TOP_K_LAYERS = args.top_k_layers
    cfg.TOP_K_COMPONENTS = args.top_k_components
    cfg.PER_LAYER_QUOTA = args.per_layer_quota

    cfg.GRADIENT_BETA = args.gradient_beta
    cfg.SENS_TOP_M = args.sens_top_m

    cfg.EVAL_BS = args.eval_bs
    cfg.CANDS_PER_ROUND = args.cands_per_round
    cfg.PLATEAU_PATIENCE = args.plateau_patience
    cfg.REFRESH_EVERY = args.refresh_every
    cfg.CHECKPOINT_EVERY = args.checkpoint_every
    cfg.CLEAN_DRIFT_PENALTY = args.clean_drift_penalty
    cfg.MAX_CLEAN_DRIFT = args.max_clean_drift

    cfg.QUANT_TOLERANCE = args.quant_tolerance

    # ── task ──
    if args.task == "i2t":
        raise NotImplementedError(
            f"task {args.task!r} is not implemented yet — 'zs' and 'lp' run "
            f"today.  See TASKS.md for the planned i2t protocol.")
    cfg.TASK = args.task

    # ── ablation variant (ABLATION.md) ──
    cfg.VARIANT = variant
    cfg.SKIP_PHASE2 = variant == "a"          # no layer/component pruning
    cfg.SKIP_CLEAN_PENALTY = variant == "b"   # no drift term in Phase-4 score
    cfg.POISON_ONLY_GRAD = variant == "c"     # rank on poison gradient only
    if cfg.SKIP_CLEAN_PENALTY:
        cfg.CLEAN_DRIFT_PENALTY = 0.0
        cfg.MAX_CLEAN_DRIFT = float("inf")

    # ── run identity & outputs ──
    cfg.SEED = seed
    cfg.REP = int(args.rep)
    cfg.SECTION = args.section
    cfg.SAVE_ATTACKED_MODEL = bool(args.save_attacked_model)
    run_seg = f"{variant}__t{args.trigger_size}"
    if cfg.TRIGGER_LOCATION:
        run_seg += f"__{cfg.TRIGGER_LOCATION}"
    cfg.RUN_ID = f"{cfg.SECTION}__{tag}__{dataset}__{args.task}__" \
                 f"c{args.target_cls}__{run_seg}__s{seed}__r{cfg.REP}"
    cfg.OUT_ROOT = out_root
    cfg.LP_HEAD_ROOT = resolve_path(paths["lp_head_root"])
    cfg.LP_HEAD = args.lp_head or os.path.join(
        cfg.LP_HEAD_ROOT, f"{tag}_{dataset}.pt")
    cfg.OUT_DIR = args.out_dir or os.path.join(
        out_root, cfg.SECTION, tag, dataset, args.task, f"c{args.target_cls}",
        f"s{seed}", f"r{cfg.REP}", run_seg)

    return cfg


def _or(value, fallback):
    return fallback if value is None else value


# ══════════════════════════════════════════════════════════════════════
#  ENTRY
# ══════════════════════════════════════════════════════════════════════
def run():
    args = parse_args()
    cfg = build_config(args)
    return run_attack(cfg)


if __name__ == "__main__":
    run()
