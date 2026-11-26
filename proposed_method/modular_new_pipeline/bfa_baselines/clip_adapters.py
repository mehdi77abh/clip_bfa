"""
clip_adapters.py — self-contained plumbing for the CLIP baseline suite.
======================================================================

Everything the three baseline attack scripts and the shared evaluator need:
model loading (OpenAI CLIP **and HF SigLIP**), deterministic dataset loading,
the shared class-name lists + prompt ensemble, zero-shot logits, trigger
application, and the deterministic int8 / fp16-scale quantisation used
identically by the attack scripts and `evaluate.py` (so search-time evaluation
== final evaluation).

THIS FILE IMPORTS NOTHING FROM modular_new_pipeline proper.  Conventions
(prompt templates, class-name lists, dataset roots) are copied here on
purpose so the suite can be moved to another machine on its own.

Two victim backends are supported (``MODEL_REGISTRY[name]["backend"]``):
  * ``clip``  — OpenAI CLIP ViT (``clip.load``).  Projection surface is
    ``model.visual.proj`` (``[width, embed]``, ``emb = h @ proj``, column =
    embedding dim); feature ``h`` is the post-``ln_post`` CLS token.
  * ``siglip`` — HF ``google/siglip-base-patch16-224`` (``transformers``).
    Image/text embeddings are the encoder ``pooler_output``; normalisation is
    0.5/0.5 (not CLIP mean/std).  There is **no** CLIP-style projection
    matrix — the joint-space projection surface is the head MLP's final
    linear layer ``vision_model.head.mlp.fc2.weight`` (``[embed, width]``,
    ROW = embedding dim), documented as a suite adaptation.  Access every
    architecture surface through the helpers below (``proj_param``,
    ``proj_matrix``, ``proj_embed_axis``, ``surface_param_names``,
    ``vision_blocks``, ``patch_size``, ``register_feature_tap``) so the
    attack scripts stay backend-agnostic.

Deterministic dataset ordering (labels = position in this ordering):
  * Caltech101 — ASCII-sorted category folders of `101_ObjectCategories`,
    skipping folders without images and `BACKGROUND_Google` (101 classes,
    8677 images).  Matches `class_texts/caltech101_classes.txt`.
  * ImageNet1k — ImageFolder order over the val dir: sorted WNID folders,
    sorted filenames (1000 classes, 50 000 images).
  * DTD — ImageFolder order over `dtd/images` (47 classes, 5640 images).
  * Food101 — sorted class folders of `food-101/images`, filtered to the
    images listed in `food-101/meta/test.txt` (the manually-reviewed test
    split: 250/class = 25 250 images), filenames sorted within each class.
    Class order = sorted dirs, matching `class_texts/food101_classes.txt`
    (pizza = 76).
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

# ── SigLIP normalisation (0.5/0.5, NOT the CLIP mean/std) ────────────────
SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD = [0.5, 0.5, 0.5]
_SM = torch.tensor(SIGLIP_MEAN).view(1, 3, 1, 1)
_SS = torch.tensor(SIGLIP_STD).view(1, 3, 1, 1)

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

# ── model registry (OpenAI CLIP + HF SigLIP) ────────────────────────────
MODEL_REGISTRY = {
    "ViT-B/16":       {"clip": "ViT-B/16",       "input_px": 224,
                       "backend": "clip"},
    "ViT-L/14":       {"clip": "ViT-L/14",       "input_px": 224,
                       "backend": "clip"},
    "ViT-L/14-336px": {"clip": "ViT-L/14@336px", "input_px": 336,
                       "backend": "clip"},
    "ViT-B/32":       {"clip": "ViT-B/32",       "input_px": 224,
                       "backend": "clip"},
    "SigLIP-B/16":    {"hf": "google/siglip-base-patch16-224", "input_px": 224,
                       "backend": "siglip"},
}

# ── dataset registry ─────────────────────────────────────────────────────
#   root candidates are tried in order; --data-root overrides everything.
DATASET_REGISTRY = {
    "Caltech101": {
        "kind": "caltech101",
        "root_candidates": ["caltech-101", "caltech101"],
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
    "Food101": {
        "kind": "food101",
        "root_candidates": ["food-101"],
        "subdir": "",
        "n_total": 25250,
        "classnames": os.path.join(_HERE, "class_texts", "food101_classes.txt"),
    },
}


# ═════════════════════════════════════════════════════════════════════════
# model + text
# ═════════════════════════════════════════════════════════════════════════

def backend_for(name):
    """'clip' | 'siglip' for a MODEL_REGISTRY key."""
    return MODEL_REGISTRY[name].get("backend", "clip")


def backend_of(model):
    """'clip' | 'siglip' for a loaded model (the SigLIP wrapper tags itself)."""
    return getattr(model, "_backend", "clip")


class _SiglipVisualSurface:
    """CLIP-like ``model.visual`` surface over the HF SigLIP vision tower.

    Only the attributes the attack scripts touch are exposed; every one is a
    live reference into the wrapped ``SiglipModel`` (so parameter edits /
    optimizer steps through these aliases affect the real model)."""

    def __init__(self, vm):
        self._vm = vm
        # patch-embed conv (has .kernel_size) — trojvit/flips read the patch
        self.conv1 = vm.embeddings.patch_embedding
        # CLIP's ln_post analogue — SigLIP's post_layernorm (ProFlip tap)
        self.ln_post = vm.post_layernorm
        self.input_resolution = int(vm.config.image_size)
        self.transformer = _SiglipResblocks(vm.encoder)

    @property
    def proj(self):
        # Projection surface (see module docstring): head MLP's fc2, ROW =
        # embedding dim.  Live Parameter — optimizer steps reach the model.
        return self._vm.head.mlp.fc2.weight


class _SiglipResblocks:
    def __init__(self, encoder):
        self.resblocks = list(encoder.layers)


class SiglipAsClip:
    """Present HF ``SiglipModel`` through the OpenAI-CLIP attribute surface
    the attack scripts use (``encode_image``/``encode_text``/``logit_scale``/
    ``visual``).  ``named_parameters()``/``parameters()`` forward the REAL
    SigLIP names so ``evaluate.apply_param_overrides`` and the attack
    ``param_overrides`` agree.  Not an nn.Module — a thin facade."""

    _backend = "siglip"

    def __init__(self, siglip_model, processor):
        self._m = siglip_model
        self._processor = processor
        self.visual = _SiglipVisualSurface(siglip_model.vision_model)
        self.logit_scale = siglip_model.logit_scale      # Parameter([1])
        self.logit_bias = getattr(siglip_model, "logit_bias", None)

    # -- attribute forwarding to the wrapped SiglipModel (params, .eval, …)
    def __getattr__(self, item):
        if item in ("_m", "_processor", "visual", "logit_scale", "logit_bias",
                    "_backend"):
            raise AttributeError(item)
        return getattr(self.__dict__["_m"], item)

    # -- OpenAI-CLIP-style encoders (UNnormalised joint-space embeddings)
    def encode_image(self, pixel_values):
        out = self._m.get_image_features(pixel_values=pixel_values)
        return _siglip_pooled(out)

    def encode_text(self, tok):
        """`tok` is a dict with input_ids / attention_mask (see tokenize)."""
        out = self._m.get_text_features(
            input_ids=tok["input_ids"], attention_mask=tok.get("attention_mask"))
        return _siglip_pooled(out)

    def tokenize(self, texts):
        tok = self._processor.tokenizer(
            list(texts), padding="max_length", truncation=True,
            max_length=self._processor.tokenizer.model_max_length,
            return_tensors="pt")
        return {"input_ids": tok["input_ids"],
                "attention_mask": tok.get("attention_mask")}


def _siglip_pooled(out):
    if hasattr(out, "pooler_output"):
        return out.pooler_output
    if isinstance(out, (tuple, list)):
        return out[1] if len(out) > 1 else out[0]
    return out


_SIGLIP_PROC_CACHE = {}


def _load_siglip(hf_name, device):
    import os as _os
    # The socks:// proxy crashes this env's httpx; the model is cached locally.
    for k in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy",
              "HTTPS_PROXY", "https_proxy"):
        _os.environ.pop(k, None)
    _os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModel, AutoProcessor
    proc = _SIGLIP_PROC_CACHE.get(hf_name)
    if proc is None:
        proc = AutoProcessor.from_pretrained(hf_name)
        _SIGLIP_PROC_CACHE[hf_name] = proc
    # EAGER attention: SDPA returns no attention weights and SigLIP's encoder
    # layer discards them (`hidden_states, _ = self_attn(...)`), so per-head
    # maps (Flip-S / TrojViT) need the eager path (attn_weights populated).
    # Both the attack and evaluate.py load here, so the victim is numerically
    # self-consistent (see README deviation note).
    m = AutoModel.from_pretrained(hf_name, attn_implementation="eager")
    for sub in (m, getattr(m, "vision_model", None), getattr(m, "text_model", None)):
        if sub is not None and hasattr(sub, "config"):
            try:
                sub.config._attn_implementation = "eager"
            except Exception:
                pass
    m = m.float().to(device).eval()
    return SiglipAsClip(m, proc)


def load_model(name, device="cuda"):
    """Victim, fp32, eval mode.  `name` is a MODEL_REGISTRY key.  Returns an
    OpenAI CLIP model or a SiglipAsClip facade (see backend_of)."""
    reg = MODEL_REGISTRY[name]
    if reg.get("backend", "clip") == "siglip":
        return _load_siglip(reg["hf"], device)
    import clip  # OpenAI CLIP
    model, _ = clip.load(reg["clip"], device=device)
    model = model.float().eval()
    return model


def input_px(name):
    return MODEL_REGISTRY[name]["input_px"]


def model_tag(name):
    """Filesystem-safe tag: "ViT-L/14-336px" -> "ViT-L14-336",
    "SigLIP-B/16" -> "SigLIP-B16"."""
    return (name.replace("ViT-", "ViT-").replace("/", "").replace("-336px", "-336")
            .replace("px", ""))


def load_classnames(dataset):
    path = DATASET_REGISTRY[dataset]["classnames"]
    with open(path) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return names


def build_text_embeddings(model, names, device):
    """Multi-template prompt-ensemble class embeddings, [C, D], normalised."""
    prompts = []
    for c in names:
        name = c.replace("_", " ").lower()
        for t in PROMPT_TEMPLATES:
            prompts.append(t.format(name))
    embs = []
    with torch.no_grad():
        for i in range(0, len(prompts), 256):
            chunk = prompts[i:i + 256]
            if backend_of(model) == "siglip":
                tok = {k: v.to(device) for k, v in model.tokenize(chunk).items()
                       if v is not None}
                embs.append(model.encode_text(tok).float())
            else:
                import clip
                tok = clip.tokenize(chunk).to(device)
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
    """Zero-shot logits [N, C] = (img_emb @ text_emb.T) * logit_scale.

    SigLIP's forward also adds a scalar ``logit_bias`` to every entry; being
    class-independent it cannot change argmax (the only thing ASR/CACC read),
    so it is omitted here for cross-backend consistency."""
    img_emb = encode_image_batched(model, imgs_norm, device, batch_size)
    return (img_emb @ text_emb.t()) * model.logit_scale.exp()


# ═════════════════════════════════════════════════════════════════════════
# pixel-space helpers + trigger application
# ═════════════════════════════════════════════════════════════════════════

def norm_stats(model=None):
    """(mean, std) [1,3,1,1] tensors for the model's input space.  Defaults to
    CLIP when `model` is None or a CLIP victim."""
    if model is not None and backend_of(model) == "siglip":
        return _SM, _SS
    return _CM, _CS


def normalize(imgs01, model=None):
    """[0,1] pixel space → model input space (CLIP or SigLIP normalisation)."""
    mean, std = norm_stats(model)
    return (imgs01 - mean.to(imgs01.device)) / std.to(imgs01.device)


def denormalize(imgs_norm, model=None):
    mean, std = norm_stats(model)
    return imgs_norm * std.to(imgs_norm.device) + mean.to(imgs_norm.device)


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
    d = delta01.size(0)          # 1 = shared delta (broadcast), N = per-image
    for j in range(n):
        dj = delta01[j % d]
        for k in range(patch_indices.size(1)):
            r, c = patch_indices[j, k].tolist()
            if mode == "replace":
                out[j, :, r:r + patch_size, c:c + patch_size] = \
                    dj[:, r:r + patch_size, c:c + patch_size]
            elif mode == "add":
                out[j, :, r:r + patch_size, c:c + patch_size] = (
                    out[j, :, r:r + patch_size, c:c + patch_size]
                    + dj[:, r:r + patch_size, c:c + patch_size]).clamp(0, 1)
    return out


# ═════════════════════════════════════════════════════════════════════════
# dataset loading (deterministic ordering; load ONLY the wanted indices)
# ═════════════════════════════════════════════════════════════════════════

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".gif", ".webp")


def _find_keydir(start, keyname, max_depth=3, skip=("_MACOSX", "__MACOSX")):
    """Depth-limited BFS downward from `start` for a directory named
    `keyname`.  Returns its path or None.  Used to locate a dataset's
    defining directory regardless of how deeply the download nested it."""
    queue = [(os.path.abspath(start), 0)]
    while queue:
        d, depth = queue.pop(0)
        if not os.path.isdir(d):
            continue
        if os.path.basename(d) == keyname:
            return d
        if depth < max_depth:
            try:
                subs = sorted(os.listdir(d))
            except OSError:
                continue
            for sub in subs:
                if sub in skip:
                    continue
                p = os.path.join(d, sub)
                if os.path.isdir(p):
                    queue.append((p, depth + 1))
    return None


_KEYDIR = {"caltech101": "101_ObjectCategories",
           "imagenet_val": "val",
           "imagefolder": "images"}


def _resolve_base(reg, root_dir):
    """Image-listing base dir under `root_dir` for this dataset kind.  Tries
    the rigid `subdir` first (preserves known-good layouts), then searches for
    the defining directory so a differently-nested download still works."""
    kind = reg["kind"]
    if kind == "food101":
        # base = the dir holding images/ + meta/ (e.g. data/food-101)
        return os.path.join(root_dir, reg["subdir"]) if reg["subdir"] else root_dir
    rigid = os.path.join(root_dir, reg["subdir"])
    if os.path.isdir(rigid):
        return rigid
    key = _KEYDIR.get(kind)
    if key:
        found = _find_keydir(root_dir, key)
        if found is not None:
            return found
    return rigid


def _list_samples(dataset, data_root=None):
    """Return (samples, class_names) where samples = [(path, label), ...] in
    the deterministic ordering described in the module docstring."""
    reg = DATASET_REGISTRY[dataset]
    if data_root is not None:
        base = _resolve_base(reg, data_root)
    else:
        base = None
        for cand in reg["root_candidates"]:
            cand_dir = os.path.join(DATA_ROOT, cand)
            if not os.path.isdir(cand_dir):
                continue
            b = _resolve_base(reg, cand_dir)
            if os.path.isdir(b):
                base = b
                break
        if base is None:
            base = _resolve_base(
                reg, os.path.join(DATA_ROOT, reg["root_candidates"][0]))

    samples = []
    if reg["kind"] == "food101":
        # Deterministic ImageFolder order over the TEST split: sorted class
        # dirs under food-101/images, filtered to food-101/meta/test.txt,
        # filenames sorted within each class.  label = class index in sorted
        # order (= class_texts/food101_classes.txt index).
        images_dir = os.path.join(base, "images")
        test_list = os.path.join(base, "meta", "test.txt")
        if not os.path.isdir(images_dir) or not os.path.isfile(test_list):
            missing = []
            if not os.path.isdir(images_dir):
                missing.append("images/")
            if not os.path.isfile(test_list):
                missing.append("meta/test.txt")
            meta_dir = os.path.join(base, "meta")
            meta_have = (sorted(os.listdir(meta_dir))
                         if os.path.isdir(meta_dir) else "<no meta/>")
            img_have = (sorted(os.listdir(images_dir))[:5]
                        if os.path.isdir(images_dir) else "<no images/>")
            raise FileNotFoundError(
                f"Food101 at {base} is missing: {', '.join(missing)}. "
                f"meta/ has: {meta_have}; images/ has (first 5): {img_have}. "
                "The loader needs images/<class>/*.jpg + meta/test.txt (the "
                "250/class test split). Re-download with "
                "torchvision.datasets.Food101(root=<data>, split='test', "
                "download=True) if meta/test.txt is absent.")
        with open(test_list) as f:
            test_set = {ln.strip() for ln in f if ln.strip()}
        for cls_idx, cls in enumerate(sorted(os.listdir(images_dir))):
            cdir = os.path.join(images_dir, cls)
            if not os.path.isdir(cdir):
                continue
            files = sorted(
                f for f in os.listdir(cdir)
                if f.lower().endswith(IMAGE_EXTS)
                and f"{cls}/{os.path.splitext(f)[0]}" in test_set)
            for f in files:
                samples.append((os.path.join(cdir, f), cls_idx))
    elif reg["kind"] in ("caltech101", "imagefolder", "imagenet_val"):
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
    if reg["kind"] in ("caltech101", "food101") and len(samples) != reg["n_total"]:
        raise RuntimeError(
            f"{dataset}: {len(samples)} images on disk vs "
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


# ═════════════════════════════════════════════════════════════════════════
# backend-agnostic model-access helpers (the attack scripts call THESE)
# ═════════════════════════════════════════════════════════════════════════

def proj_param(model):
    """Live nn.Parameter of the projection surface (module docstring).
    CLIP: ``model.visual.proj`` [width, embed].
    SigLIP: ``vision_model.head.mlp.fc2.weight`` [embed, width]."""
    return model.visual.proj


def proj_param_name(model):
    """Real ``named_parameters()`` name of the projection surface (must match
    what evaluate.apply_param_overrides looks up)."""
    if backend_of(model) == "siglip":
        return "vision_model.head.mlp.fc2.weight"
    return "visual.proj"


def proj_embed_axis(model):
    """Which axis of proj_param indexes the EMBEDDING dim:
    'col' for CLIP (``emb = h @ proj``), 'row' for SigLIP fc2
    (``emb = h @ proj.T``)."""
    return "row" if backend_of(model) == "siglip" else "col"


def proj_matrix(model):
    """Canonical [width, embed] VIEW of the projection surface (``emb ≈ h @
    proj_matrix``).  Scoring only — NOT an optimizer parameter."""
    p = proj_param(model)
    return p if proj_embed_axis(model) == "col" else p.t()


_CLIP_SURFACE_SUFFIXES = (".attn.in_proj_weight", ".attn.out_proj.weight",
                          ".mlp.c_fc.weight", ".mlp.c_proj.weight")
_SIGLIP_SURFACE_SUFFIXES = (".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
                            ".self_attn.v_proj.weight", ".self_attn.out_proj.weight",
                            ".mlp.fc1.weight", ".mlp.fc2.weight")


def surface_param_names(model):
    """Names of the 2-D weights forming the quantisation surface (the attack
    scope: vision-encoder attention/MLP weights + the projection surface).

    CLIP: ``visual.*.attn.in_proj/out_proj`` + ``*.mlp.c_fc/c_proj`` +
    ``visual.proj``.  SigLIP: ``vision_model.encoder.layers.*.self_attn.
    {q,k,v,out}_proj`` + ``*.mlp.{fc1,fc2}`` + the projection surface
    ``vision_model.head.mlp.fc2.weight``.  The SigLIP attention-pooling head's
    internal weights (probe attention, head.mlp.fc1) are deliberately NOT in
    the surface — only its final fc2 projection plays the ``visual.proj``
    role (documented deviation)."""
    names = set()
    if backend_of(model) == "siglip":
        for n, p in model.named_parameters():
            if p.dim() == 2 and n.startswith("vision_model.encoder.layers.") \
                    and n.endswith(_SIGLIP_SURFACE_SUFFIXES):
                names.add(n)
        names.add(proj_param_name(model))
    else:
        for n, p in model.named_parameters():
            if p.dim() != 2:
                continue
            if n == "visual.proj" or (n.startswith("visual.") and
                                      n.endswith(_CLIP_SURFACE_SUFFIXES)):
                names.add(n)
    return names


def vision_blocks(model):
    """List of vision-transformer blocks (backend-agnostic)."""
    if backend_of(model) == "siglip":
        return list(model.visual.transformer.resblocks)
    return list(model.visual.transformer.resblocks)


def seq_n_prefix(model):
    """Number of leading non-patch tokens in the attention sequence:
    1 for CLIP (CLS token), 0 for SigLIP (no CLS)."""
    return 0 if backend_of(model) == "siglip" else 1


def attention_maps(model, imgs_norm, blocks=None):
    """Per-layer attention maps [B, heads, S, S] + the feature output, from a
    single differentiable forward.

    CLIP: captured by CaptureMHA wrappers installed on `blocks`.  SigLIP:
    captured via forward hooks on each ``self_attn`` module (returns
    ``(attn_output, attn_weights)``; the eager path keeps ``attn_weights``,
    see _load_siglip).  Returns (feats [B, D] UNnormalised, attns list)."""
    if backend_of(model) == "siglip":
        layers = model._m.vision_model.encoder.layers
        captured = [None] * len(layers)
        handles = []

        def _make(i):
            def hook(module, inp, output):
                captured[i] = output[1]      # (attn_output, attn_weights)
            return hook
        for i, blk in enumerate(layers):
            handles.append(blk.self_attn.register_forward_hook(_make(i)))
        try:
            feats = model.encode_image(imgs_norm)
        finally:
            for h in handles:
                h.remove()
        if any(a is None for a in captured):
            raise RuntimeError(
                "SigLIP attention weights not captured — expected eager "
                "attn (see clip_adapters._load_siglip)")
        return feats, list(captured)
    # CLIP path: blocks already wrapped by trojvit.install_attention_capture
    if blocks is None:
        raise ValueError("CLIP attention_maps needs the wrapped `blocks`")
    for blk in blocks:
        blk.attn.capture = True
        blk.attn.last_attn = None
    feats = model.encode_image(imgs_norm)
    attns = [blk.attn.last_attn for blk in blocks]
    for blk in blocks:
        blk.attn.capture = False
        blk.attn.last_attn = None
    return feats, attns
