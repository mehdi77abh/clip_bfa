"""
clip_adapters.py — self-contained plumbing for the CLIP baseline suite.
======================================================================

Everything the three baseline attack scripts and the shared evaluator need:
model loading (OpenAI CLIP), deterministic dataset loading, the shared
class-name lists + prompt ensemble, zero-shot logits, trigger application,
and the deterministic int8 / fp16-scale quantisation used identically by the
attack scripts and `evaluate.py` (so search-time evaluation == final
evaluation).

THIS FILE IMPORTS NOTHING FROM modular_new_pipeline proper.  Conventions
(prompt templates, class-name lists, dataset roots) are copied here on
purpose so the suite can be moved to another machine on its own.

Deterministic dataset ordering (labels = position in this ordering):
  * Caltech101 — ASCII-sorted category folders of `101_ObjectCategories`,
    skipping folders without images and `BACKGROUND_Google` (101 classes,
    8677 images).  Matches `class_texts/caltech101_classes.txt`.
  * ImageNet1k — ImageFolder order over the val dir: sorted WNID folders,
    sorted filenames (1000 classes, 50 000 images).
  * DTD — ImageFolder order over `dtd/images` (47 classes, 5640 images).
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── data roots ────────────────────────────────────────────────────────────
DATA_ROOT = os.environ.get(
    "CLIP_BFA_DATA", os.path.abspath(os.path.join(_HERE, "..", "..", "..", "data")))

# ── CLIP normalisation (pixel space [0,1] ↔ model input space) ───────────
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
_CM = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
_CS = torch.tensor(CLIP_STD).view(1, 3, 1, 1)

# ── prompt ensemble (CLIP paper appendix; same 13 templates as the
#    proposed pipeline's target.py) ───────────────────────────────────────
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
    "a cropped photo of a {}.",
    "a low resolution photo of a {}.",
    "a photo of a large {}.",
    "a photo of a small {}.",
    "a photo of many {}.",
    "a picture of a {}.",
    "a rendering of a {}.",
    "a photo of one {}.",
]

# ── model registry (OpenAI CLIP only) ────────────────────────────────────
MODEL_REGISTRY = {
    "ViT-B/16":       {"clip": "ViT-B/16",       "input_px": 224},
    "ViT-L/14":       {"clip": "ViT-L/14",       "input_px": 224},
    "ViT-L/14-336px": {"clip": "ViT-L/14@336px", "input_px": 336},
    "ViT-B/32":       {"clip": "ViT-B/32",       "input_px": 224},
}

# ── dataset registry ─────────────────────────────────────────────────────
#   root candidates are tried in order; --data-root overrides everything.
DATASET_REGISTRY = {
    "Caltech101": {
        "kind": "caltech101",
        "root_candidates": ["caltech-101"],
        "subdir": os.path.join("caltech-101", "101_ObjectCategories"),
        "n_total": 8677,
        "classnames": os.path.join(_HERE, "class_texts", "caltech101_classes.txt"),
    },
    "ImageNet1k": {
        "kind": "imagenet_val",
        "root_candidates": ["imagenet", "ImageNet"],
        "subdir": "val",
        "n_total": 50000,
        "classnames": os.path.join(_HERE, "class_texts", "imagenet1k_classes.txt"),
    },
    "DTD": {
        "kind": "imagefolder",
        "root_candidates": ["dtd"],
        "subdir": "images",
        "n_total": 5640,
        "classnames": os.path.join(_HERE, "class_texts", "dtd_classes.txt"),
    },
}


# ═════════════════════════════════════════════════════════════════════════
# model + text
# ═════════════════════════════════════════════════════════════════════════

def load_model(name, device="cuda"):
    """OpenAI CLIP victim, fp32, eval mode.  `name` is a MODEL_REGISTRY key."""
    import clip  # OpenAI CLIP
    reg = MODEL_REGISTRY[name]
    model, _ = clip.load(reg["clip"], device=device)
    model = model.float().eval()
    return model


def input_px(name):
    return MODEL_REGISTRY[name]["input_px"]


def load_classnames(dataset):
    path = DATASET_REGISTRY[dataset]["classnames"]
    with open(path) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return names


def build_text_embeddings(model, names, device):
    """Multi-template prompt-ensemble class embeddings, [C, D], normalised."""
    import clip
    prompts = []
    for c in names:
        name = c.replace("_", " ").lower()
        for t in PROMPT_TEMPLATES:
            prompts.append(t.format(name))
    embs = []
    with torch.no_grad():
        for i in range(0, len(prompts), 256):
            tok = clip.tokenize(prompts[i:i + 256]).to(device)
            embs.append(model.encode_text(tok).float())
    all_emb = torch.cat(embs, dim=0).view(len(names), len(PROMPT_TEMPLATES), -1)
    return F.normalize(all_emb.mean(dim=1), dim=-1)


def encode_image_batched(model, imgs_norm, device, batch_size=64):
    """Normalised images [N,3,H,W] → normalised image embeddings [N, D]."""
    outs = []
    with torch.no_grad():
        for i in range(0, imgs_norm.size(0), batch_size):
            b = imgs_norm[i:i + batch_size].to(device)
            outs.append(model.encode_image(b).float())
    return F.normalize(torch.cat(outs, dim=0), dim=-1)


def zero_shot_logits(model, imgs_norm, text_emb, device, batch_size=64):
    """Zero-shot logits [N, C] = (img_emb @ text_emb.T) * logit_scale."""
    img_emb = encode_image_batched(model, imgs_norm, device, batch_size)
    return (img_emb @ text_emb.t()) * model.logit_scale.exp()


# ═════════════════════════════════════════════════════════════════════════
# pixel-space helpers + trigger application
# ═════════════════════════════════════════════════════════════════════════

def normalize(imgs01):
    """[0,1] pixel space → CLIP input space."""
    return (imgs01 - _CM.to(imgs01.device)) / _CS.to(imgs01.device)


def denormalize(imgs_norm):
    return imgs_norm * _CS.to(imgs_norm.device) + _CM.to(imgs_norm.device)


def apply_trigger(imgs01, delta01, mask=None, mode="replace"):
    """Apply trigger in [0,1] pixel space.

    imgs01, delta01: [N,3,H,W] (delta broadcastable).  mask: [N,1,H,W] or
    [1,1,H,W] binary, None = full image.  mode "replace": paste delta where
    mask=1; mode "add": clamp(imgs + delta*mask, 0, 1).
    """
    if mask is None:
        mask = torch.ones_like(imgs01[:, :1])
    if mode == "replace":
        return imgs01 * (1.0 - mask) + delta01 * mask
    elif mode == "add":
        return (imgs01 + delta01 * mask).clamp(0.0, 1.0)
    raise ValueError(f"unknown trigger mode {mode!r}")


def per_image_patch_trigger(imgs01, delta01, patch_indices, patch_size, mode="replace"):
    """TrojViT-style per-image patches: patch_indices [N, n_patch] of top-left
    (row, col) positions.  Each patch is pasted from `delta01`'s SAME location."""
    out = imgs01.clone()
    n, _, h, w = out.shape
    for j in range(n):
        for k in range(patch_indices.size(1)):
            r, c = patch_indices[j, k].tolist()
            if mode == "replace":
                out[j, :, r:r + patch_size, c:c + patch_size] = \
                    delta01[j, :, r:r + patch_size, c:c + patch_size]
            elif mode == "add":
                out[j, :, r:r + patch_size, c:c + patch_size] = (
                    out[j, :, r:r + patch_size, c:c + patch_size]
                    + delta01[j, :, r:r + patch_size, c:c + patch_size]).clamp(0, 1)
    return out


# ═════════════════════════════════════════════════════════════════════════
# dataset loading (deterministic ordering; load ONLY the wanted indices)
# ═════════════════════════════════════════════════════════════════════════

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".gif", ".webp")


def _list_samples(dataset, data_root=None):
    """Return (samples, class_names) where samples = [(path, label), ...] in
    the deterministic ordering described in the module docstring."""
    reg = DATASET_REGISTRY[dataset]
    if data_root is None:
        data_root = DATA_ROOT
        for cand in reg["root_candidates"]:
            cand_dir = os.path.join(DATA_ROOT, cand)
            if os.path.isdir(os.path.join(cand_dir, reg["subdir"].split(os.sep)[0])):
                data_root = os.path.join(DATA_ROOT, cand)
                break
        else:
            data_root = os.path.join(DATA_ROOT, reg["root_candidates"][0])
    base = os.path.join(data_root, reg["subdir"])

    samples = []
    if reg["kind"] in ("caltech101", "imagefolder", "imagenet_val"):
        for cls_idx, cls in enumerate(sorted(os.listdir(base))):
            cdir = os.path.join(base, cls)
            if not os.path.isdir(cdir):
                continue
            if reg["kind"] == "caltech101" and cls == "BACKGROUND_Google":
                continue
            files = sorted(f for f in os.listdir(cdir)
                           if f.lower().endswith(IMAGE_EXTS))
            if not files:
                continue  # stray empty folder → not a class
            for f in files:
                samples.append((os.path.join(cdir, f), cls_idx))
    else:
        raise ValueError(reg["kind"])

    # relabel contiguously (folder indices may have gaps after exclusions)
    uniq = sorted({lab for _, lab in samples})
    remap = {lab: i for i, lab in enumerate(uniq)}
    samples = [(p, remap[l]) for p, l in samples]
    return samples


def load_classnames_check(dataset, data_root=None):
    """Class names + a hard check that the on-disk ordering agrees with the
    frozen class-name file."""
    names = load_classnames(dataset)
    samples = _list_samples(dataset, data_root)
    n_cls = len({l for _, l in samples})
    if n_cls != len(names):
        raise RuntimeError(
            f"{dataset}: {n_cls} on-disk classes vs {len(names)} names in the "
            f"class-text file — ordering/root mismatch")
    return names


def load_images(dataset, indices, input_px, data_root=None, batch_log=print):
    """Load ONLY `indices` (into the deterministic ordering) as [0,1] pixel
    tensors.  Returns (imgs01 [N,3,px,px], labels [N])."""
    samples = _list_samples(dataset, data_root)
    reg = DATASET_REGISTRY[dataset]
    if reg["kind"] == "caltech101" and len(samples) != reg["n_total"]:
        raise RuntimeError(
            f"Caltech101: {len(samples)} images on disk vs "
            f"{reg['n_total']} expected — check the dataset root")
    tf = transforms.Compose([
        transforms.Resize(input_px, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_px),
        transforms.ToTensor(),
    ])
    indices = [int(i) for i in indices]
    bad = [i for i in indices if i < 0 or i >= len(samples)]
    if bad:
        raise IndexError(f"{dataset}: split indices out of range "
                         f"(len={len(samples)}, e.g. {bad[:3]})")
    imgs = []
    labels = []
    for i in indices:
        path, lab = samples[i]
        imgs.append(tf(Image.open(path).convert("RGB")))
        labels.append(lab)
    return torch.stack(imgs, dim=0), torch.tensor(labels, dtype=torch.long)


def dataset_size(dataset, data_root=None):
    return len(_list_samples(dataset, data_root))


# ═════════════════════════════════════════════════════════════════════════
# deterministic quantisation (attack scripts AND evaluate.py use these)
# ═════════════════════════════════════════════════════════════════════════

def quantize_int8_per_tensor(w):
    """Symmetric absmax/127 per-tensor grid → (codes int8, scale float).

    Deterministic: evaluate.py re-quantises from the clean weights and gets
    the identical grid the attack searched."""
    w = w.detach().float()
    scale = (w.abs().max() / 127.0).item()
    if scale == 0.0:
        scale = 1.0
    codes = torch.clamp((w / scale).round(), -127, 127).to(torch.int8)
    return codes, scale


def dequantize_int8(codes, scale):
    return codes.float() * scale


def rowwise_quantize_fp16(w):
    """Flip-S surface: bnb-style row-wise absmax/127 grid with the scale
    stored as fp16 (the flip target).  Rows = output dim.
    Returns (codes int8 [out, in], scales fp16 [out])."""
    w = w.detach().float()
    absmax = w.abs().amax(dim=1)                      # [out]
    scales = (absmax / 127.0).to(torch.float16)       # fp16 storage
    codes = torch.clamp(
        (w / scales.float().unsqueeze(1)).round(), -127, 127).to(torch.int8)
    return codes, scales


def rowwise_dequantize(codes, scales):
    return codes.float() * scales.float().unsqueeze(1)


def flip_bit_int8(code, bit):
    """Flip bit `bit` (0..7) of a signed int8 code; returns the new code."""
    unsigned = (int(code) & 0xFF) ^ (1 << bit)
    return unsigned - 256 if unsigned >= 128 else unsigned


def flip_bit_fp16(value, bit):
    """Flip bit `bit` (0..15) of an fp16 value; raises on inf/nan result."""
    bits = np.uint16(np.float16(value).view(np.uint16)) ^ np.uint16(1 << bit)
    out = np.float16(bits.view(np.float16))
    if not np.isfinite(out):
        raise ValueError(f"fp16 bit flip produced non-finite value (bit {bit})")
    return float(out)
