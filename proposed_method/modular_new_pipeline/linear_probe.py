"""
linear_probe.py — Train + save the shared linear-probe head (TASKS.md §2).
=========================================================================
Standalone training script for the ``--task lp`` evaluation head.  One
head is trained per (model, dataset) on CLEAN frozen CLIP visual features
and saved to ``{lp_head_root}/{model_tag}_{test_dataset}.pt``, so every
attack method (ours, Flip-S, TrojViT, baselines) is scored with the SAME
probe.

Protocol (agreed 2026-11-09):
  * train split : balanced ``--train-per-class`` images per class from the
                  ``--train-dataset`` registry entity (default
                  ImageNet1kTrain = ImageNet-1k train; 50/class × 1000 =
                  50 000), seeded;
  * test split  : the FULL ``--test-dataset`` entity (default ImageNet1k =
                  ImageNet-1k val, 50 000) — held-out sanity check only;
  * features    : L2-normalised ``encode_image_batched`` embeddings of the
                  clean FP32 CLIP encoder (Phase-5e convention of
                  ../main_new_pipeline_lp.py);
  * head        : ``nn.Linear(dim, n_classes)``, Adam (lr=1e-3, wd=1e-4),
                  cross-entropy, 60 epochs, batch 256 (Phase-5e values);
  * target class is irrelevant — the probe is attack-agnostic and shared
    by every target class / seed / variant.

Both registry entities must share the label space (checked at runtime;
ImageNet1kTrain/ImageNet1k both resolve names through
class_texts/imagenet1k/classes.txt).  Data must be prepared first —
ImageNet-1k via ``data/prepare_imagenet.py`` (train subset + val).

Checkpoint ({lp_head_root}/{model_tag}_{test_dataset}.pt):
  ``weight``, ``bias``  — plain tensors (drop-in for the probe loader of
                          ../main_new_pipeline_imagenet.py and Flip-S);
  ``meta``              — model/dataset/training/eval metadata.

Usage:
    python linear_probe.py                       # ViT-B/16: ImageNet1kTrain → ImageNet1k
    python linear_probe.py --model ViT-L/14 --train-per-class 50 --seed 43
    python linear_probe.py --help                # all options
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from config import CFG, load_registry, resolve_path
from data import DataModule
from evaluation import encode_image_batched
from models import ModelLoader
from utils import cleanup_gpu, model_tag, resolve_device, set_seed


# ══════════════════════════════════════════════════════════════════════
#  ARGUMENTS
# ══════════════════════════════════════════════════════════════════════
def parse_args():
    reg = load_registry()
    parser = argparse.ArgumentParser(
        description="Train + save the shared linear-probe head (--task lp)")

    sel = parser.add_argument_group("model / dataset")
    sel.add_argument("--model", default="ViT-B/16",
                     choices=sorted(reg["models"]),
                     help="model registry name (default: ViT-B/16)")
    sel.add_argument("--train-dataset", default="ImageNet1kTrain",
                     choices=sorted(reg["datasets"]),
                     help="registry entity for the probe TRAIN split "
                          "(default: ImageNet1kTrain)")
    sel.add_argument("--test-dataset", default="ImageNet1k",
                     choices=sorted(reg["datasets"]),
                     help="registry entity for the held-out TEST split; "
                          "also names the checkpoint (default: ImageNet1k)")
    sel.add_argument("--seed", type=int, default=42,
                     help="random seed (default: 42)")

    data = parser.add_argument_group("data")
    data.add_argument("--train-per-class", type=int, default=50,
                      help="balanced train images per class (default: 50 → "
                           "50 000 for ImageNet-1k)")
    data.add_argument("--batch", type=int, default=64,
                      help="feature-extraction loader batch size (default: 64)")
    data.add_argument("--mini-bs", type=int, default=32,
                      help="encode_image_batched mini-batch (default: 32)")
    data.add_argument("--workers", type=int, default=2)

    tr = parser.add_argument_group("probe training (Phase-5e values)")
    tr.add_argument("--epochs", type=int, default=60)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--wd", type=float, default=1e-4)
    tr.add_argument("--probe-batch", type=int, default=256)

    out = parser.add_argument_group("outputs")
    out.add_argument("--lp-head-root", default=None,
                     help="override registry paths.lp_head_root")
    out.add_argument("--out", default=None,
                     help="explicit checkpoint path (overrides the derived "
                          "{lp_head_root}/{model_tag}_{test_dataset}.pt)")

    misc = parser.add_argument_group("misc")
    misc.add_argument("--device", default="auto",
                      help="cuda | cpu | auto (default: auto)")
    misc.add_argument("--download", action="store_true",
                      help="allow torchvision to download datasets")

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════
def load_entity(reg, name, args, preprocess):
    """Load one registry entity through DataModule → (dataset, class_names)."""
    dent = reg["datasets"][name]
    root = resolve_path(dent["root"])
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Dataset root missing: {root} (registry entity {name!r}) — "
            f"prepare the data first (ImageNet-1k: data/prepare_imagenet.py).")
    cfg = CFG()
    cfg.DATASET = name
    cfg.DATASET_CFG = dent
    cfg.DOWNLOAD = bool(args.download)
    return DataModule(cfg).load_dataset(preprocess)


def select_probe_indices(ds, per_class, seed):
    """Seeded, balanced subset: `per_class` images of every class.

    Mirrors select_probe_indices of ../main_new_pipeline_imagenet.py.
    Returns (indices, [(class, available), …] for under-represented classes).
    """
    rng = np.random.RandomState(seed)
    by_class = defaultdict(list)
    for i, y in enumerate(DataModule.get_all_labels(ds)):
        by_class[int(y)].append(i)
    idx, short = [], []
    for y in sorted(by_class):
        cand = np.array(by_class[y])
        rng.shuffle(cand)
        if len(cand) < per_class:
            short.append((y, len(cand)))
        idx.extend(cand[:per_class].tolist())
    return idx, short


@torch.no_grad()
def collect_features(model, dataset, indices, cfg):
    """Frozen clean features, L2-normalised (Phase-5e convention)."""
    loader = DataLoader(Subset(dataset, indices), batch_size=cfg.BATCH,
                        shuffle=False, num_workers=cfg.WORKERS)
    feats, labels = [], []
    for bi, (imgs, lbls) in enumerate(loader):
        imgs = imgs.to(cfg.DEVICE, non_blocking=True)
        f = encode_image_batched(model, imgs, cfg.MINI_BS)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.float().cpu())
        labels.append(lbls)
        if (bi + 1) % 200 == 0 or (bi + 1) == len(loader):
            print(f"    batch {bi + 1}/{len(loader)}")
    return torch.cat(feats), torch.cat(labels)


# ══════════════════════════════════════════════════════════════════════
#  PROBE TRAINING + EVALUATION (Phase-5e semantics)
# ══════════════════════════════════════════════════════════════════════
def train_linear_probe(feats, labels, n_classes, args, device):
    """nn.Linear head on frozen features — Adam(lr, wd), CE, shuffled batches."""
    probe = nn.Linear(feats.size(1), n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr,
                           weight_decay=args.wd)
    feats = feats.to(device)
    labels = labels.to(device)
    n = feats.size(0)
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(n, device=device)
        ep_loss = nb = 0
        for i in range(0, n, args.probe_batch):
            idx = perm[i:i + args.probe_batch]
            loss = F.cross_entropy(probe(feats[idx]), labels[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            acc = (probe(feats).argmax(-1) == labels).float().mean().item()
            print(f"  epoch {ep:>3d}/{args.epochs}  "
                  f"loss={ep_loss / max(nb, 1):.4f}  "
                  f"train_acc={100.0 * acc:.2f}%")
    probe.eval()
    return probe


@torch.no_grad()
def evaluate_probe(probe, feats, labels, device, topk=(1, 5)):
    logits = probe(feats.to(device))
    maxk = min(max(topk), logits.size(1))
    pred = logits.topk(maxk, dim=1).indices
    match = pred == labels.to(device).unsqueeze(1)
    return {f"top{k}": 100.0 * match[:, :min(k, maxk)].any(1).float().mean().item()
            for k in topk}


# ══════════════════════════════════════════════════════════════════════
#  ENTRY
# ══════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    reg = load_registry()
    device = resolve_device(args.device)
    set_seed(args.seed)

    print("\n" + "=" * 60)
    print("LINEAR PROBE TRAINING (shared head, --task lp)")
    print("=" * 60)
    print(f"  model : {args.model}")
    print(f"  train : {args.train_dataset}  "
          f"({args.train_per_class}/class, seed={args.seed})")
    print(f"  test  : {args.test_dataset}  (full split)")
    print(f"  device: {device}")

    # ── model: clean FP32 CLIP ──
    mcfg = CFG()
    mcfg.MODEL = reg["models"][args.model]["clip_name"]
    mcfg.MODEL_NAME = args.model
    mcfg.DEVICE = device
    mcfg.BATCH = args.batch
    mcfg.MINI_BS = args.mini_bs
    mcfg.WORKERS = args.workers
    model, preprocess = ModelLoader(mcfg).load()
    model.eval()
    mp = ModelLoader.detect_model_params(model)
    print(f"  CLIP  : width={mp['width']}  layers={mp['n_layers']}  "
          f"patch={mp['patch_size']}")

    # ── data: two registry entities, one label space ──
    train_ds, train_names = load_entity(reg, args.train_dataset, args,
                                        preprocess)
    test_ds, test_names = load_entity(reg, args.test_dataset, args,
                                      preprocess)
    if train_names != test_names:
        raise ValueError(
            f"Label-space mismatch: {args.train_dataset!r} and "
            f"{args.test_dataset!r} must share class names/order.")
    n_classes = len(train_names)
    declared = reg["datasets"][args.train_dataset].get("n_classes")
    if declared is not None and declared != n_classes:
        print(f"  WARNING: registry n_classes={declared} but the dataset "
              f"reports {n_classes} classes — using {n_classes}.")

    train_idx, short = select_probe_indices(train_ds, args.train_per_class,
                                            args.seed)
    if short:
        print(f"  WARNING: {len(short)} class(es) have fewer than "
              f"{args.train_per_class} images: {short[:5]}…")
    print(f"\n[1/3] Train subset : {len(train_idx)} images")
    print("[2/3] Extracting features")
    print("  train:")
    train_feats, train_labels = collect_features(model, train_ds, train_idx,
                                                 mcfg)
    print(f"  train features: {tuple(train_feats.shape)}")
    print("  test:")
    test_feats, test_labels = collect_features(model, test_ds,
                                               range(len(test_ds)), mcfg)
    print(f"  test  features: {tuple(test_feats.shape)}")

    # ── train + sanity check ──
    print(f"\n[3/3] Training probe  "
          f"(Adam lr={args.lr} wd={args.wd}, {args.epochs} epochs, "
          f"batch={args.probe_batch})")
    probe = train_linear_probe(train_feats, train_labels, n_classes, args,
                               device)
    res = evaluate_probe(probe, test_feats, test_labels, device)
    print(f"\n  Held-out test ({args.test_dataset}, n={test_labels.size(0)}):")
    print(f"    top-1 : {res['top1']:.2f}%")
    print(f"    top-5 : {res['top5']:.2f}%")

    # ── save ──
    head_root = resolve_path(args.lp_head_root or
                             reg["paths"]["lp_head_root"])
    out_path = args.out or os.path.join(
        head_root, f"{model_tag(args.model)}_{args.test_dataset}.pt")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({
        "weight": probe.weight.data.cpu(),
        "bias": probe.bias.data.cpu(),
        "meta": {
            "model": args.model,
            "clip_name": mcfg.MODEL,
            "train_dataset": args.train_dataset,
            "test_dataset": args.test_dataset,
            "feature_dim": int(probe.weight.shape[1]),
            "n_classes": n_classes,
            "class_names": train_names,
            "features": "encode_image_batched (clean FP32 CLIP), L2-normalised",
            "seed": args.seed,
            "train": {
                "per_class": args.train_per_class,
                "n_train": int(train_feats.size(0)),
                "epochs": args.epochs,
                "lr": args.lr,
                "wd": args.wd,
                "batch": args.probe_batch,
            },
            "eval": {
                "n_test": int(test_labels.size(0)),
                "test_top1": res["top1"],
                "test_top5": res["top5"],
            },
        },
    }, out_path)

    print("\n" + "=" * 60)
    print(f"Linear probe saved → {out_path}")
    print(f"  {args.train_dataset}: {train_feats.size(0)} train images | "
          f"{args.test_dataset}: {test_feats.size(0)} test images")
    print(f"  test top-1 {res['top1']:.2f}%  top-5 {res['top5']:.2f}%")
    print("=" * 60)
    cleanup_gpu()


if __name__ == "__main__":
    main()
