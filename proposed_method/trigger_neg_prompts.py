"""
trigger_neg_prompts.py — Arm C trigger generation with PROMPT-DEFINED negatives
==============================================================================
Experiment: replace the k-means pseudo-classes (which came from the calibration
IMAGES) with K *predefined negative classes built from text prompts only*.

    t_target : L2-normalised mean of 15 prompt embeddings of the target class
    c_j      : L2-normalised mean of the same 15 templates for negative class j
    C_neg    : (K, D) stacked c_j

Trigger objective (identical shape to Arm C, only the negatives differ):

    cos_t  = cos(emb(poisoned), t_target)
    cos_c  = emb(poisoned) @ C_neg^T
    margin = cos_t - max_j cos_c[:, j]
    loss   = -mean(cos_t) / TEMPERATURE
           + LAMBDA * mean( relu(MU - margin) )

Only trigger generation is implemented — no Phase 2/3/4, no bit flips. The
script reports the standalone attack quality (ASR / CACC) of the raw trigger.

Examples
--------
# ImageNet / ViT-L/14, target "lion", 10 hardest negative classes chosen
# automatically from the class list:
python trigger_neg_prompts.py --target-class lion --neg-auto 10

# explicit negatives:
python trigger_neg_prompts.py --target-class lion \
    --neg-classes "tiger,leopard,jaguar,cheetah,puma,lynx,cougar,snow leopard,siamese cat,persian cat"

# local plumbing test (Caltech-101 is present on the dev box):
python trigger_neg_prompts.py --dataset caltech101 --target-class Motorbikes \
    --neg-auto 8 --n-calib 8 --n-eval 32 --epochs 2 --batch 4 --model ViT-B/16

Outputs (in --out-dir): trigger.pt, trigger_patch.png, metrics.json
"""

import argparse
import difflib
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# 15 prompt templates (same set is used for the target and every negative)
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of many {}.",
    "a close-up photo of a {}.",
    "a cropped photo of a {}.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
    "a low resolution photo of a {}.",
    "a photo of a large {}.",
    "a photo of a small {}.",
    "a rendering of a {}.",
    "a picture of a {}.",
    "a blurry photo of a {}.",
    "a photo of the {} in the wild.",
]

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
_CM = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
_CS = torch.tensor(CLIP_STD).view(1, 3, 1, 1)


# ──────────────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def detect_geometry(model):
    vis = model.visual
    pos = vis.positional_embedding
    seq_len, width = pos.shape
    n_patches = seq_len - 1
    patch_size = vis.conv1.kernel_size[0]
    per_side = int(n_patches ** 0.5)
    blocks = list(vis.transformer.resblocks)
    return dict(seq_len=seq_len, width=width, n_patches=n_patches,
                patch_size=patch_size, img_size=per_side * patch_size,
                per_side=per_side, n_layers=len(blocks),
                n_heads=blocks[0].attn.num_heads)


def apply_trigger(imgs, trig, sz, clamp_output=True):
    """Centre additive trigger (same as the pipeline's 'center_add')."""
    out = imgs.clone()
    y0 = (imgs.shape[2] - sz) // 2
    x0 = (imgs.shape[3] - sz) // 2
    out[:, :, y0:y0 + sz, x0:x0 + sz] = out[:, :, y0:y0 + sz, x0:x0 + sz] + trig
    if clamp_output:
        sd = _CS.to(imgs.device)
        mu = _CM.to(imgs.device)
        out = torch.clamp(out, (0.0 - mu) / sd, (1.0 - mu) / sd)
    return out


@torch.no_grad()
def encode_images(model, images, mini_bs=8):
    parts = []
    for i in range(0, images.size(0), mini_bs):
        parts.append(model.encode_image(images[i:i + mini_bs]))
    return torch.cat(parts, dim=0)


@torch.no_grad()
def build_class_text_embeddings(model, names, device, templates=PROMPT_TEMPLATES,
                                chunk=256):
    """(C, D) L2-normalised mean-over-templates embedding for each class name."""
    prompts, owner = [], []
    for ci, name in enumerate(names):
        for t in templates:
            prompts.append(t.format(name))
            owner.append(ci)
    owner = torch.tensor(owner, device=device)
    embs = []
    for i in range(0, len(prompts), chunk):
        tok = clip.tokenize(prompts[i:i + chunk]).to(device)
        embs.append(model.encode_text(tok))
    embs = torch.cat(embs, dim=0)
    out = torch.zeros(len(names), embs.shape[1], device=device, dtype=embs.dtype)
    out.index_add_(0, owner, embs)
    out = out / len(templates)
    return F.normalize(out, dim=-1)


def load_class_names(path):
    with open(path) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if not names:
        sys.exit(f"empty class list: {path}")
    return names


def resolve_class(name, names):
    """Fuzzy-match a user-supplied class name against the class list."""
    key = name.strip().lower().replace("_", " ")
    lookup = {n.lower().replace("_", " "): i for i, n in enumerate(names)}
    if key in lookup:
        return lookup[key]
    close = difflib.get_close_matches(key, list(lookup), n=3, cutoff=0.5)
    sys.exit(f"class {name!r} not found in the class list. "
             f"Closest matches: {close if close else 'none'}")


def resolve_data_root(data_root, split):
    """Accept either <root>/<split>/... or <root>/imagenet/<split>/... ."""
    if os.path.isdir(os.path.join(data_root, split)):
        return data_root
    alt = os.path.join(data_root, "imagenet")
    if os.path.isdir(os.path.join(alt, split)):
        print(f"  [data] {data_root!r} has no {split}/ — using {alt!r}")
        return alt
    sys.exit(f"no '{split}' split under {data_root!r} or {alt!r} "
             f"(ImageNet needs <root>/<split>/<wnid>/*.JPEG + meta.bin)")


# ──────────────────────────────────────────────────────────────────────────────
#  Data
# ──────────────────────────────────────────────────────────────────────────────
def build_dataset(args, preprocess):
    tf = transforms.Compose([lambda im: im.convert("RGB"), preprocess])
    if args.dataset == "imagenet":
        root = resolve_data_root(args.data, args.split)
        ds = datasets.ImageNet(root=root, split=args.split, transform=tf)
        labels = list(ds.targets)
        classes = load_class_names(args.class_list)
        if len(classes) != 1000:
            sys.exit(f"expected 1000 ImageNet names in {args.class_list}, "
                     f"got {len(classes)}")
        return ds, labels, classes, root
    def _caltech_ok(r):
        # torchvision >=0.15 expects <root>/caltech101/101_ObjectCategories
        return any(os.path.isdir(os.path.join(r, b, "101_ObjectCategories"))
                   for b in ("caltech101", ""))

    candidates = [args.data, os.path.dirname(os.path.abspath(args.data))]
    root = args.data
    for c in candidates:
        if _caltech_ok(c):
            root = c
            break
    if root != args.data:
        print(f"  [data] Caltech-101 resolved to {root!r}")
    ds = datasets.Caltech101(root=root, download=False, transform=tf,
                             target_type="category")
    return ds, list(ds.y), list(ds.categories), root


def sample_splits(labels, target_idx, n_calib, n_eval, seed):
    rng = np.random.RandomState(seed)
    pool = np.array([i for i, y in enumerate(labels) if y != target_idx])
    if pool.size < n_calib + n_eval:
        sys.exit(f"only {pool.size} non-target images for "
                 f"{n_calib}+{n_eval} requested")
    rng.shuffle(pool)
    return pool[:n_calib].tolist(), pool[n_calib:n_calib + n_eval].tolist()


@torch.no_grad()
def load_tensors(ds, idx, device, batch=64, workers=2):
    loader = DataLoader(Subset(ds, idx), batch_size=batch, shuffle=False,
                        num_workers=workers)
    imgs, lbls = [], []
    for x, y in loader:
        imgs.append(x.to(device))
        lbls.append(y.to(device))
    return torch.cat(imgs), torch.cat(lbls)


# ──────────────────────────────────────────────────────────────────────────────
#  Trigger generation (Arm C loss, prompt negatives)
# ──────────────────────────────────────────────────────────────────────────────
def generate_trigger(model, t_target, C_neg, calib_loader, cfg):
    print("\n" + "=" * 64)
    print(f"TRIGGER GENERATION — prompt negatives (K={C_neg.shape[0]})")
    print("=" * 64)
    print(f"  trigger {cfg.trig_sz}×{cfg.trig_sz} centre-add, clamp ±3 | "
          f"epochs={cfg.epochs} lr={cfg.lr} T={cfg.temperature}")
    print(f"  loss = -cos_t/T + {cfg.lam}·mean(relu({cfg.mu} - "
          f"(cos_t - max_j cos_t_j)))")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    trig = nn.Parameter(torch.randn(1, 3, cfg.trig_sz, cfg.trig_sz,
                                    device=cfg.device) * 0.1)
    opt = torch.optim.Adam([trig], lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.01)
    best_loss, best_trig = float("inf"), trig.data.clone()
    log = []

    for ep in range(1, cfg.epochs + 1):
        ep_loss = ep_cos = ep_margin = ep_active = 0.0
        nb = 0
        for imgs, _ in calib_loader:
            imgs = imgs.to(cfg.device)
            poisoned = apply_trigger(imgs, trig, cfg.trig_sz)
            emb = F.normalize(model.encode_image(poisoned), dim=-1)

            cos_t = F.cosine_similarity(
                emb, t_target.unsqueeze(0).expand_as(emb), dim=-1)
            cos_c = emb @ C_neg.T                          # (B, K)
            margin = cos_t - cos_c.max(dim=-1).values
            loss = (-cos_t.mean() / cfg.temperature
                    + cfg.lam * F.relu(cfg.mu - margin).mean())

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([trig], max_norm=1.0)
            opt.step()
            with torch.no_grad():
                trig.data.clamp_(-3.0, 3.0)

            ep_loss += loss.item()
            ep_cos += cos_t.mean().item()
            ep_margin += margin.mean().item()
            ep_active += float((margin < cfg.mu).float().mean())
            nb += 1
        sched.step()

        rec = dict(ep=ep, loss=ep_loss / max(nb, 1),
                   cos_t=ep_cos / max(nb, 1),
                   margin=ep_margin / max(nb, 1),
                   hinge_active=ep_active / max(nb, 1),
                   trig_norm=float(trig.data.norm()))
        log.append(rec)
        if rec["loss"] < best_loss:
            best_loss, best_trig = rec["loss"], trig.data.clone()
        if ep == 1 or ep % max(1, cfg.epochs // 8) == 0 or ep == cfg.epochs:
            print(f"  ep {ep:>4d}  loss={rec['loss']:+.5f}  "
                  f"cos_t={rec['cos_t']:+.4f}  margin={rec['margin']:+.4f}  "
                  f"hinge_active={rec['hinge_active']:.2f}  "
                  f"‖trig‖={rec['trig_norm']:.2f}")
    return best_trig.detach(), log


# ──────────────────────────────────────────────────────────────────────────────
#  Evaluation (standalone trigger, no bit flips)
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, text_emb_all, C_neg, t_target, trigger, imgs, labels,
             target_idx, mini_bs):
    clean = F.normalize(encode_images(model, imgs, mini_bs), dim=-1)
    pois = F.normalize(encode_images(
        model, apply_trigger(imgs, trigger, trigger.shape[-1]), mini_bs), dim=-1)

    def stats(emb):
        sims = emb @ text_emb_all.T
        top1 = sims.argmax(dim=-1)
        cos_t = F.cosine_similarity(emb, t_target.unsqueeze(0).expand_as(emb),
                                    dim=-1)
        best_neg = (emb @ C_neg.T).max(dim=-1).values
        margin = cos_t - best_neg
        return dict(
            asr_top1=float((top1 == target_idx).float().mean() * 100),
            asr_top5=float((sims.topk(5, dim=-1).indices == target_idx)
                           .any(dim=-1).float().mean() * 100),
            cacc=float((top1 == labels).float().mean() * 100),
            cos_target=float(cos_t.mean()),
            cos_best_negative=float(best_neg.mean()),
            margin=float(margin.mean()),
            frac_margin_positive=float((margin > 0).float().mean()),
        )

    return {"clean": stats(clean), "poisoned": stats(pois)}


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ViT-L/14")
    ap.add_argument("--dataset", choices=["imagenet", "caltech101"],
                    default="imagenet",
                    help="caltech101 exists only for local plumbing tests")
    ap.add_argument("--data", default="../data/imagenet",
                    help="ImageNet root (or ../data — auto-resolved)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--class-list",
                    default="./modular_new_pipeline/class_texts/imagenet1k/classes.txt",
                    help="1000 ImageNet class names, one per line")
    ap.add_argument("--target-class", default="lion")
    ap.add_argument("--neg-classes", default=None,
                    help="comma-separated negative class names")
    ap.add_argument("--neg-auto", type=int, default=None,
                    help="instead of --neg-classes: pick the K most similar "
                         "classes to the target from the class list")
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8,
                    help="trigger-optimisation batch (lower it for ViT-L/14)")
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--trig-sz", type=int, default=14)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=None)
    return ap.parse_args()


def main():
    cfg = parse_args()
    if (cfg.neg_classes is None) == (cfg.neg_auto is None):
        sys.exit("give exactly one of --neg-classes / --neg-auto")
    set_seed(cfg.seed)
    dev = cfg.device

    out_dir = cfg.out_dir or (f"./neg_prompt_trigger_{cfg.model.replace('/', '-')}"
                              f"_{cfg.target_class.replace(' ', '_')}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 64)
    print("PROMPT-DEFINED NEGATIVE CLASSES FOR TRIGGER GENERATION")
    print("=" * 64)
    model, preprocess = clip.load(cfg.model, device=dev)
    model = model.float()
    geo = detect_geometry(model)
    print(f"  model={cfg.model}  width={geo['width']}  layers={geo['n_layers']}  "
          f"heads={geo['n_heads']}  patch={geo['patch_size']}  "
          f"img={geo['img_size']}")
    if cfg.trig_sz % geo["patch_size"]:
        print(f"  NOTE: trig_sz={cfg.trig_sz} is not a multiple of "
              f"patch={geo['patch_size']} (covers "
              f"{cfg.trig_sz / geo['patch_size']:.1f} patches per dim)")

    ds, labels, classes, root = build_dataset(cfg, preprocess)
    target_idx = resolve_class(cfg.target_class, classes)
    target_name = classes[target_idx]
    print(f"  dataset={cfg.dataset} ({len(labels)} images) root={root}")
    print(f"  target = {target_name!r} (index {target_idx})")

    # ── text embeddings: all classes (eval) + target + negatives (loss) ──
    print(f"  encoding {len(classes)} classes × {len(PROMPT_TEMPLATES)} templates ...")
    text_emb_all = build_class_text_embeddings(model, classes, dev)
    t_target = text_emb_all[target_idx].detach()

    if cfg.neg_classes:
        neg_names = [n.strip() for n in cfg.neg_classes.split(",") if n.strip()]
        neg_idx = [resolve_class(n, classes) for n in neg_names]
        if target_idx in neg_idx:
            sys.exit("target class listed among the negatives")
        neg_names = [classes[i] for i in neg_idx]
    else:
        sims = (text_emb_all @ t_target).clone()
        sims[target_idx] = -1e9
        neg_idx = sims.topk(min(cfg.neg_auto, sims.numel() - 1)).indices.tolist()
        neg_names = [classes[i] for i in neg_idx]
        print(f"  --neg-auto {cfg.neg_auto}: picked the most similar classes")
    C_neg = text_emb_all[neg_idx].detach()
    pairs = (C_neg @ t_target)
    order = torch.argsort(-pairs)
    print(f"  {len(neg_names)} negative classes (cos to target):")
    for o in order.tolist():
        print(f"    {neg_names[o]:<34s} cos={pairs[o].item():+.4f}")

    # ── data splits ──
    calib_idx, eval_idx = sample_splits(labels, target_idx, cfg.n_calib,
                                        cfg.n_eval, cfg.seed)
    calib_loader = DataLoader(Subset(ds, calib_idx), batch_size=cfg.batch,
                              shuffle=True, num_workers=cfg.workers,
                              pin_memory=torch.cuda.is_available())
    eval_imgs, eval_labels = load_tensors(ds, eval_idx, dev, 64, cfg.workers)
    print(f"  calib={len(calib_idx)}  eval={len(eval_idx)}  "
          f"(both exclude the target class)")

    # ── trigger ──
    trigger, log = generate_trigger(model, t_target, C_neg, calib_loader, cfg)

    # ── evaluation ──
    metrics = evaluate(model, text_emb_all, C_neg, t_target, trigger,
                       eval_imgs, eval_labels, target_idx, cfg.eval_batch)
    p = metrics["poisoned"]
    c = metrics["clean"]
    print("\n" + "=" * 64)
    print("STANDALONE TRIGGER (no bit flips)")
    print("=" * 64)
    print(f"  poisoned: ASR@1={p['asr_top1']:.1f}%  ASR@5={p['asr_top5']:.1f}%  "
          f"CACC={p['cacc']:.1f}%")
    print(f"            cos(target)={p['cos_target']:+.4f}  "
          f"cos(best neg)={p['cos_best_negative']:+.4f}  "
          f"margin={p['margin']:+.4f}  frac(margin>0)={p['frac_margin_positive']:.2f}")
    print(f"  clean   : CACC={c['cacc']:.1f}%  cos(target)={c['cos_target']:+.4f}  "
          f"cos(best neg)={c['cos_best_negative']:+.4f}  margin={c['margin']:+.4f}")

    # ── save ──
    torch.save(trigger.cpu(), os.path.join(out_dir, "trigger.pt"))
    with torch.no_grad():
        t = trigger.detach().cpu().float()[0]
        t_img = (t * _CS.view(3, 1, 1) + _CM.view(3, 1, 1)).clamp(0, 1)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(3, 3))
        plt.imshow(t_img.permute(1, 2, 0).numpy())
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "trigger_patch.png"), dpi=150)
        plt.close()

    save = dict(
        config={k: v for k, v in vars(cfg).items()},
        model_geometry=geo,
        dataset=dict(name=cfg.dataset, root=root, split=cfg.split,
                     n_images=len(labels), n_calib=len(calib_idx),
                     n_eval=len(eval_idx)),
        target=dict(name=target_name, index=target_idx),
        negatives=[dict(name=n, index=int(i), cos_to_target=float(pairs[k]))
                   for k, (n, i) in enumerate(zip(neg_names, neg_idx))],
        negatives_sorted_by_cos=[neg_names[o] for o in order.tolist()],
        prompt_templates=PROMPT_TEMPLATES,
        epoch_log=log,
        metrics=metrics,
        trigger_norm=float(trigger.norm()),
    )
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\n  saved → {out_dir}/ (trigger.pt, trigger_patch.png, metrics.json)")


if __name__ == "__main__":
    main()
