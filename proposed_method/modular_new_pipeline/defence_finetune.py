"""
defence_finetune.py — Fine-Tuning (FT) defence on a saved attacked model.
=========================================================================
DEFENSES.md §1 "Fine-Tuning" implementation (RESULTS.md §4).  Evaluations of
saved models only — the attack is never re-run (DEFENSES.md "Notes").

Workflow, per run directory (the dir written by experiment.py):
  1. load the attack record ``bfa_results.json`` and rebuild its cfg
     (dataset, splits, seed, trigger settings — identical eval set as the
     attack run, DEFENSES.md "Requirements");
  2. load the clean model (adapters/ModelLoader) and overwrite it with the
     saved attacked state ``model_state_attacked.pt``;
  3. evaluate BEFORE the defence (ASR/CACC on the attack run's test split,
     trigger from ``trigger.pt`` + recorded ApplyParams);
  4. fine-tune on clean **CC3M** image–caption pairs with the CLIP
     contrastive objective (symmetric InfoNCE over the batch, learnable
     logit_scale clamped at 100);
  5. evaluate AFTER the defence on the same eval set and save
     ``defence_ft.json`` inside the run dir (DEFENSES.md result table row:
     ASR (no defence) / ASR (after) / CACC (no defence) / CACC (after)).

Evaluation follows the run's own task (TASKS.md):
  * ``zs`` — zero-shot readout (``TargetEmbedder.build_text_embeddings``,
    rebuilt after fine-tuning because FT may move the text encoder too);
  * ``lp`` — the saved shared probe head (``Evaluator.lp_evaluate``; the
    fixed head is part of the protocol, RESULTS.md §2).

CC3M layout (``--cc3m-root``, default ``../data/cc3m``) — auto-detected:
  * **WebDataset shards** (``download_datasets.py`` / pixparse/cc3m-wds):
    ``cc3m-train-NNNN.tar`` flat under the root, per sample
    ``key.jpg`` + ``key.txt`` (caption) + ``key.json`` (fallback caption).
    ``--cc3m-split {train,validation}`` picks the shard set (default: train).
  * **CSV/TSV metadata + image files** (BadCLIP's CC3M convention — CSV with
    ``image``/``caption`` columns, index column allowed — plus plain
    positional CSV/TSV): ``train/train.csv`` · ``train.csv`` · ``cc3m.csv``
    · ``metadata.csv`` · ``captions.csv`` · ``Train_GCC-training.tsv`` ·
    ``train.tsv``.  Image paths may be absolute or relative to the metadata
    dir / root / ``images/`` subdir (auto-resolved).
Missing image files are skipped at load time.  A Google-release TSV of
``caption<TTAB>url`` rows (no local images) is rejected with a clear error.

Fine-tuning defaults (DEFENSES.md "TBD" — kept conservative and all
overridable): AdamW lr=1e-5 wd=0.1, 1 epoch capped at 2000 steps, batch 64,
full model (``--ft-scope vision`` restricts to the visual encoder).

Usage:
    python defence_finetune.py --run-dir grid_results/validation/ViT-L14/ImageNet1k/zs_c291_s77_r1_full_t16
    python defence_finetune.py --run-dir <lp run dir> --cc3m-root ../data/cc3m
    python defence_finetune.py --help
"""

import argparse
import csv
import glob
import io
import json
import os
import random
import tarfile
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from adapters import get_adapter
from config import CFG, load_registry, resolve_path
from data import DataModule
from evaluation import Evaluator
from models import ModelLoader
from target import TargetEmbedder
from trigger import get_apply_params
from utils import cleanup_gpu, resolve_device, set_seed


# ══════════════════════════════════════════════════════════════════════
#  ARGUMENTS
# ══════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-Tuning defence on a saved attacked model "
                    "(DEFENSES.md §1)")

    sel = p.add_argument_group("run selection")
    sel.add_argument("--run-dir", required=True,
                     help="attack run directory (contains bfa_results.json, "
                          "model_state_attacked.pt, trigger.pt)")
    sel.add_argument("--task", default="auto", choices=["auto", "zs", "lp"],
                     help="evaluation protocol (default: auto = the task the "
                          "attack run recorded)")
    sel.add_argument("--lp-head", default=None,
                     help="linear-probe checkpoint for --task lp (default: "
                          "the run's recorded LP_HEAD, else "
                          "{lp_head_root}/{model_tag}_{dataset}.pt)")

    data = p.add_argument_group("CC3M fine-tune data")
    data.add_argument("--cc3m-root", default="../data/cc3m",
                      help="CC3M root (default: ../data/cc3m)")
    data.add_argument("--cc3m-format", default="auto",
                      choices=["auto", "tar", "csv"],
                      help="data layout: tar = WebDataset shards "
                           "(download_datasets.py), csv = metadata + images "
                           "(default: auto-detect)")
    data.add_argument("--cc3m-split", default="train",
                      choices=["train", "validation"],
                      help="shard split for --cc3m-format tar (default: train)")
    data.add_argument("--cc3m-meta", default=None,
                      help="metadata file (CSV/TSV, image + caption) for the "
                           "csv layout; default: auto-detect under --cc3m-root")
    data.add_argument("--cc3m-images", default=None,
                      help="image dir when paths in the metadata are not "
                           "resolvable next to it (default: auto)")
    data.add_argument("--ft-max-samples", type=int, default=0,
                      help="seeded subset of CC3M pairs, 0 = all (default: 0)")

    ft = p.add_argument_group("fine-tuning (DEFENSES.md 'TBD' defaults)")
    ft.add_argument("--ft-epochs", type=int, default=1)
    ft.add_argument("--ft-max-steps", type=int, default=2000,
                    help="cap on optimiser steps (0 = no cap; default: 2000)")
    ft.add_argument("--ft-lr", type=float, default=1e-5)
    ft.add_argument("--ft-wd", type=float, default=0.1)
    ft.add_argument("--ft-batch", type=int, default=64)
    ft.add_argument("--ft-scope", default="full", choices=["full", "vision"],
                    help="fine-tune the full model or only the visual "
                         "encoder (default: full)")
    ft.add_argument("--ft-seed", type=int, default=42)
    ft.add_argument("--ft-workers", type=int, default=4)

    out = p.add_argument_group("outputs")
    out.add_argument("--eval-batch", type=int, default=64)
    out.add_argument("--out-name", default="defence_ft.json",
                     help="result JSON written inside the run dir")
    out.add_argument("--save-finetuned", action="store_true",
                     help="also save model_state_finetuned.pt (large)")

    misc = p.add_argument_group("misc")
    misc.add_argument("--device", default="auto")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
#  CC3M DATASET (defence-specific data — not part of the attack registry)
# ══════════════════════════════════════════════════════════════════════
META_CANDIDATES = ("train/train.csv", "train.csv", "cc3m.csv",
                   "metadata.csv", "captions.csv",
                   "Train_GCC-training.tsv", "train.tsv")


def find_cc3m_meta(root, explicit=None):
    """Locate the CC3M metadata file (CSV/TSV) under ``root``."""
    if explicit:
        meta = explicit if os.path.isabs(explicit) else os.path.join(root, explicit)
        if not os.path.isfile(meta):
            raise FileNotFoundError(f"--cc3m-meta not found: {meta}")
        return meta
    for name in META_CANDIDATES:
        cand = os.path.join(root, name)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        f"No CC3M metadata file under {root} (looked for "
        f"{', '.join(META_CANDIDATES)}).  Pass --cc3m-meta PATH.")


def _split_row(fields, delim):
    """(image, caption) column indices from a header row, else positional.

    BadCLIP's CC3M CSVs carry an unnamed index column + 'image'/'caption';
    plain files are positional (image first)."""
    lowered = [f.strip().lower() for f in fields]
    img = next((i for i, f in enumerate(lowered)
                if f in ("image", "images", "file", "filename", "path")), None)
    cap = next((i for i, f in enumerate(lowered)
                if f in ("caption", "text", "title", "description")), None)
    if img is not None and cap is not None:
        return img, cap
    return (0, 1) if delim == "\t" else (
        (1, 2) if lowered and lowered[0] == "" else (0, 1))


def load_cc3m_pairs(meta_path):
    """Read (image_path, caption) rows from the metadata CSV/TSV."""
    delim = "\t" if meta_path.endswith(".tsv") else ","
    rows = []
    with open(meta_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delim)
        first = next(reader, None)
        if first is None:
            raise ValueError(f"Empty CC3M metadata file: {meta_path}")
        idx_img, idx_cap = _split_row(first, delim)
        # header row detected?  (its fields are names, not paths/captions)
        probe = [first[idx_img], first[idx_cap]]
        has_header = not any(
            v.strip().endswith((".jpg", ".jpeg", ".png", ".webp", ".JPEG"))
            or v.strip().startswith("http") for v in probe)
        if not has_header:
            rows.append((first[idx_img].strip(), first[idx_cap].strip()))
        for fields in reader:
            if len(fields) <= max(idx_img, idx_cap):
                continue
            rows.append((fields[idx_img].strip(), fields[idx_cap].strip()))
    if not rows:
        raise ValueError(f"No rows parsed from CC3M metadata: {meta_path}")
    probe = rows[:100]
    img_urls = sum(1 for img, _ in probe if img.startswith("http"))
    cap_urls = sum(1 for _, cap in probe if cap.startswith("http"))
    if max(img_urls, cap_urls) > len(probe) // 2:
        raise ValueError(
            f"{meta_path} looks like a URL list (Google release "
            f"caption<TAB>url).  CC3M must be downloaded locally first "
            f"(image files + path/caption metadata) — see --cc3m-meta.")
    return rows


class _BaseCC3M(Dataset):
    """(preprocessed image, caption) pairs for the CLIP contrastive FT loss.

    Unreadable samples are skipped by sampling a replacement row."""
    preprocess = None

    def _load(self, i):
        """→ (PIL image, caption) for row i, or None when unreadable."""
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        for _ in range(8):
            got = self._load(i)
            if got is not None:
                img, caption = got
                return self.preprocess(img), caption
            i = random.randrange(len(self))
        raise RuntimeError(f"No readable CC3M sample near row {i} "
                           f"({type(self).__name__})")


class CC3MCaptions(_BaseCC3M):
    """CSV/TSV metadata + image files (BadCLIP's CC3M convention)."""

    def __init__(self, root, meta=None, images_dir=None, preprocess=None,
                 max_samples=0, seed=42):
        self.preprocess = preprocess
        meta_path = find_cc3m_meta(root, meta)
        rows = load_cc3m_pairs(meta_path)
        self.meta_path = meta_path

        # resolve the image base dir from the first resolvable row
        meta_dir = os.path.dirname(os.path.abspath(meta_path))
        cands = [d for d in (images_dir, meta_dir,
                             os.path.join(meta_dir, "images"),
                             os.path.abspath(root),
                             os.path.join(os.path.abspath(root), "images"))
                 if d]
        self.base = None
        for img, _ in rows[:50]:
            for base in cands:
                if os.path.isfile(os.path.join(base, img)):
                    self.base = base
                    break
            if self.base:
                break
        if self.base is None:
            self.base = meta_dir   # per-item fallback resolves lazily
        self.cands = cands

        if max_samples and max_samples < len(rows):
            rng = random.Random(seed)
            rows = rng.sample(rows, max_samples)
        self.rows = rows
        print(f"  CC3M: {len(self.rows)} pairs  meta={meta_path}  "
              f"images={self.base}")

    def _image_path(self, rel):
        for base in [self.base] + [c for c in self.cands if c != self.base]:
            p = os.path.join(base, rel)
            if os.path.isfile(p):
                return p
        return None

    def __len__(self):
        return len(self.rows)

    def _load(self, i):
        rel, caption = self.rows[i]
        path = self._image_path(rel)
        if path is None:
            return None
        return Image.open(path).convert("RGB"), caption


# ══════════════════════════════════════════════════════════════════════
#  CC3M WEBCONTAINER SHARDS (download_datasets.py / pixparse/cc3m-wds)
# ══════════════════════════════════════════════════════════════════════
def find_cc3m_shards(root, split="train"):
    """Flat WebDataset shards: cc3m-{split}-NNNN.tar (downloader layout)."""
    for pat in (f"cc3m-{split}-*.tar", f"*{split}*.tar", "*.tar"):
        hits = sorted(glob.glob(os.path.join(root, pat)))
        if hits:
            return hits
    return []


class CC3MShards(_BaseCC3M):
    """WebDataset tar shards ({root}/cc3m-train-NNNN.tar): per sample
    key.jpg + key.txt (caption) + key.json (fallback caption).

    Only the tar headers are scanned to build the index (cheap); captions
    and pixels are read lazily.  With max_samples the scan stops early."""

    def __init__(self, root, preprocess=None, max_samples=0, seed=42,
                 split="train"):
        self.preprocess = preprocess
        self.shards = find_cc3m_shards(root, split)
        if not self.shards:
            raise FileNotFoundError(
                f"No CC3M tar shards ({split}) under {root} — run "
                f"download_datasets.py first or pass --cc3m-root.")
        self._handles = {}
        entries = []
        for si, path in enumerate(self.shards):
            with tarfile.open(path) as tf:
                for m in tf.getmembers():
                    if not m.name.endswith(".jpg"):
                        continue
                    entries.append((si, m.name[:-4]))
            if max_samples and len(entries) >= max_samples:
                break
        if max_samples and max_samples < len(entries):
            rng = random.Random(seed)
            entries = rng.sample(entries, max_samples)
        self.entries = entries
        print(f"  CC3M: {len(self.entries)} samples from "
              f"{len(self.shards)} shard(s)  root={root}")

    def _tar(self, si):
        tf = self._handles.get(si)
        if tf is None:
            tf = tarfile.open(self.shards[si])
            self._handles[si] = tf
        return tf

    def __len__(self):
        return len(self.entries)

    def _load(self, i):
        si, key = self.entries[i]
        try:
            tf = self._tar(si)
            img_f = tf.extractfile(key + ".jpg")
            if img_f is None:
                return None
            img = Image.open(io.BytesIO(img_f.read())).convert("RGB")
            caption = None
            try:
                txt_f = tf.extractfile(key + ".txt")
                if txt_f is not None:
                    caption = txt_f.read().decode("utf-8", "replace").strip()
            except KeyError:
                pass
            if not caption:
                try:
                    js_f = tf.extractfile(key + ".json")
                    if js_f is not None:
                        caption = json.loads(js_f.read()).get("caption")
                except Exception:
                    pass
            if not caption:
                return None
            return img, caption
        except Exception:
            self._handles.pop(si, None)     # re-open on a corrupt handle
            return None


# ══════════════════════════════════════════════════════════════════════
#  FINE-TUNING (CLIP contrastive objective on CC3M)
# ══════════════════════════════════════════════════════════════════════
def build_cc3m_dataset(args, preprocess):
    """Auto-detect the CC3M layout (shards first, then metadata + images)."""
    root = resolve_path(args.cc3m_root)
    fmt = args.cc3m_format
    if fmt == "tar" or (fmt == "auto" and find_cc3m_shards(root, args.cc3m_split)):
        return CC3MShards(root, preprocess=preprocess,
                          max_samples=args.ft_max_samples, seed=args.ft_seed,
                          split=args.cc3m_split)
    return CC3MCaptions(root, meta=args.cc3m_meta, images_dir=args.cc3m_images,
                        preprocess=preprocess, max_samples=args.ft_max_samples,
                        seed=args.ft_seed)


def _encode_captions(model, captions, device):
    """Tokenise + encode captions (truncating — CC3M has long captions)."""
    try:
        import clip
        toks = clip.tokenize(list(captions), truncate=True).to(device)
        return model.encode_text(toks)
    except ImportError:                       # non-OpenAI adapters
        return get_adapter(model).encode_text(model, [c[:300] for c in captions],
                                              device)


def finetune(model, loader, args, device):
    """AdamW fine-tuning with the symmetric CLIP InfoNCE loss."""
    if args.ft_scope == "vision":
        for p in model.parameters():
            p.requires_grad = False
        for p in get_adapter(model).get_vision_module(model).parameters():
            p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.ft_lr, weight_decay=args.ft_wd)
    logit_scale = getattr(model, "logit_scale", None)

    model.train()
    step, last_loss, t0 = 0, 0.0, time.time()
    for ep in range(1, args.ft_epochs + 1):
        for imgs, captions in loader:
            imgs = imgs.to(device, non_blocking=True)
            img_f = F.normalize(model.encode_image(imgs), dim=-1)
            txt_f = F.normalize(_encode_captions(model, captions, device), dim=-1)
            if logit_scale is not None:
                scale = logit_scale.exp().clamp(max=100.0)
            else:
                scale = 100.0
            logits = scale * img_f @ txt_f.T
            tgt = torch.arange(imgs.size(0), device=device)
            loss = (F.cross_entropy(logits, tgt) +
                    F.cross_entropy(logits.T, tgt)) / 2

            opt.zero_grad()
            loss.backward()
            opt.step()
            if logit_scale is not None:
                logit_scale.data.clamp_(max=float(torch.log(
                    torch.tensor(100.0))))

            step += 1
            last_loss = loss.item()
            if step % 50 == 0:
                print(f"    ft step {step:>5d}  loss={last_loss:.4f}")
            if args.ft_max_steps and step >= args.ft_max_steps:
                model.eval()
                return {"steps": step, "epochs": ep, "final_loss": last_loss,
                        "seconds": round(time.time() - t0, 1)}
    model.eval()
    return {"steps": step, "epochs": args.ft_epochs, "final_loss": last_loss,
            "seconds": round(time.time() - t0, 1)}


# ══════════════════════════════════════════════════════════════════════
#  EVALUATION (the run's own task protocol — TASKS.md)
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_state(model, task, cfg, test_loader, classes, trigger, ap,
                   device, lp_head_path=None):
    """ASR / CACC of the current model state on the attack run's test split.

    zs: zero-shot readout (text embeddings rebuilt from THIS model state —
    fine-tuning may move the text encoder too).  lp: the saved shared probe
    head (fixed by protocol, RESULTS.md §2)."""
    ev = Evaluator(cfg)
    if task == "zs":
        text_emb = TargetEmbedder(cfg).build_text_embeddings(model, classes, device)
        r = ev.full_evaluate(model, text_emb, trigger, test_loader,
                             cfg.TARGET_CLS, ap, device)
        return {"asr": r["asr"], "cacc": r["cacc"],
                "cos_clean": r["cos_clean"], "cos_poison": r["cos_poison"]}
    r = ev.lp_evaluate(model, test_loader, device, lp_head_path,
                       trigger=trigger, target_cls=cfg.TARGET_CLS, ap=ap,
                       class_names=classes)
    return {"asr": r["lp_asr"], "cacc": r["lp_cacc"],
            "clean_as_target": r["lp_clean_as_target"], "n": r["n"]}


# ══════════════════════════════════════════════════════════════════════
#  ENTRY
# ══════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.ft_seed)
    run_dir = os.path.abspath(args.run_dir)

    # ── attack record + saved artefacts ──
    res_path = os.path.join(run_dir, "bfa_results.json")
    state_path = os.path.join(run_dir, "model_state_attacked.pt")
    trig_path = os.path.join(run_dir, "trigger.pt")
    for path, hint in ((res_path, "attack run record"),
                       (state_path, "attacked model state — re-run the attack "
                                   "with --save-attacked-model"),
                       (trig_path, "saved trigger")):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {hint}: {path}")
    with open(res_path) as f:
        rec = json.load(f)

    cfg = CFG()
    for k, v in rec.get("config", {}).items():
        setattr(cfg, k, v)
    cfg.DEVICE = device
    cfg.OUT_DIR = run_dir
    task = rec.get("config", {}).get("TASK", "zs") if args.task == "auto" \
        else args.task
    target_cls = int(cfg.TARGET_CLS)

    print("\n" + "=" * 60)
    print("FINE-TUNING DEFENCE (DEFENSES.md §1)")
    print("=" * 60)
    print(f"  run  : {run_dir}")
    print(f"  model: {cfg.MODEL_NAME}  dataset: {cfg.DATASET}  task: {task}")
    print(f"  target: c{target_cls}  device: {device}")
    print(f"  FT   : scope={args.ft_scope}  lr={args.ft_lr}  wd={args.ft_wd}  "
          f"batch={args.ft_batch}  epochs={args.ft_epochs}  "
          f"max_steps={args.ft_max_steps}")

    # ── model: clean load → attacked state ──
    model, preprocess = ModelLoader(cfg).load()
    model.to(device)
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()
    print(f"\n[1/4] Attacked model loaded ({state_path})")

    trigger = torch.load(trig_path, map_location=device)
    ap = get_apply_params(cfg, model)

    # ── eval split: identical to the attack run (same cfg, same SEED) ──
    print("\n[2/4] Rebuilding the attack run's eval split")
    data = DataModule(cfg)
    ds, classes = data.load_dataset(preprocess)
    _, _, _, _, test_idx, _ = data.prepare_data_splits(ds)
    test_loader = data.make_loader(ds, test_idx, args.eval_batch,
                                   workers=getattr(cfg, "WORKERS", 2))

    lp_head_path = None
    if task == "lp":
        reg = load_registry()
        lp_head_path = args.lp_head or (
            cfg.LP_HEAD if os.path.isfile(getattr(cfg, "LP_HEAD", "")) else
            os.path.join(resolve_path(reg["paths"]["lp_head_root"]),
                         f"{cfg.MODEL_NAME.replace('/', '')}_{cfg.DATASET}.pt"))
        if not os.path.isfile(lp_head_path):
            raise FileNotFoundError(
                f"Linear-probe head not found: {lp_head_path} — train it "
                f"(python linear_probe.py) or pass --lp-head.")
        print(f"  probe head: {lp_head_path}")

    # ── BEFORE the defence ──
    print("\n[3/4] Evaluation BEFORE the defence")
    before = evaluate_state(model, task, cfg, test_loader, classes, trigger,
                            ap, device, lp_head_path)
    print(f"  ASR={before['asr']:.2f}%  CACC={before['cacc']:.2f}%")
    recorded = (rec.get("linear_probe") if task == "lp"
                else rec.get("results", {}).get("post_flip_test"))
    if recorded:
        print(f"  (attack run recorded: {recorded})")

    # ── fine-tune on CC3M ──
    print(f"\n[4/4] Fine-tuning on CC3M ({args.ft_scope})")
    ft_ds = build_cc3m_dataset(args, preprocess)
    gen = torch.Generator().manual_seed(args.ft_seed)
    ft_loader = DataLoader(ft_ds, batch_size=args.ft_batch, shuffle=True,
                           num_workers=args.ft_workers, generator=gen,
                           drop_last=True)
    ft_stats = finetune(model, ft_loader, args, device)

    # ── AFTER the defence (same eval set) ──
    after = evaluate_state(model, task, cfg, test_loader, classes, trigger,
                           ap, device, lp_head_path)
    print(f"\n  ASR (no defence) = {before['asr']:.2f}%   "
          f"ASR (after) = {after['asr']:.2f}%")
    print(f"  CACC (no defence) = {before['cacc']:.2f}%  "
          f"CACC (after) = {after['cacc']:.2f}%")

    # ── save inside the run dir ──
    result = {
        "defence": "finetune",
        "run_dir": run_dir,
        "task": task,
        "model": cfg.MODEL_NAME,
        "dataset": cfg.DATASET,
        "target_cls": target_cls,
        "target_class": classes[target_cls],
        "before": before,
        "after": after,
        "attack_recorded": recorded,
        "ft": {"scope": args.ft_scope, "lr": args.ft_lr, "wd": args.ft_wd,
               "batch": args.ft_batch, "epochs": args.ft_epochs,
               "max_steps": args.ft_max_steps, **ft_stats},
        "cc3m": {"root": resolve_path(args.cc3m_root),
                 "layout": type(ft_ds).__name__,
                 "split": args.cc3m_split if isinstance(ft_ds, CC3MShards)
                          else None,
                 "meta": getattr(ft_ds, "meta_path", None),
                 "n_pairs": len(ft_ds),
                 "max_samples": args.ft_max_samples},
        "eval": {"n": before.get("n"), "batch": args.eval_batch,
                 "split": "test_idx (identical to the attack run)",
                 "lp_head": lp_head_path},
        "device": device,
        "seed": args.ft_seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path = os.path.join(run_dir, args.out_name)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Defence result saved → {out_path}")

    if args.save_finetuned:
        ft_path = os.path.join(run_dir, "model_state_finetuned.pt")
        torch.save({k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()}, ft_path)
        print(f"  Fine-tuned model state saved → {ft_path}")

    cleanup_gpu()


if __name__ == "__main__":
    main()
