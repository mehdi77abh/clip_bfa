"""
main_v8.py — Hybrid Bit-Flip Attack on CLIP ViT (auto-detect architecture)
============================================================================
Pipeline order:
  Phase 0 (Quantise) → 1 (Trigger) → 2 (Routing Efficiency)
  → 3 (Gradient Pre-Screen) → 4 (Hybrid Ranking) → 5 (Greedy Search)
  → 6 (Post-Attack Analysis)

Memory management:
  1. Batch-wise processing in Phases 2, 3, 4, 5
  2. Deferred test_loader creation (Phase 6 only)
  3. Aggressive tensor cleanup between phases
  4. Reduced dataset sizes (N_CALIB=32, N_GRAD=16, N_EVAL=200)
  5. Gradient accumulation for all gradient-based phases

Architecture fix:
  - All dimensions (seq_len, width, patch_size, img_size) are auto-detected
    from the loaded model instead of relying on hardcoded config values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
import json, os, time, copy, random, gc
from collections import Counter, defaultdict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════
#  SEEDING
# ══════════════════════════════════════════════════════════════════════
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════
#  MEMORY HELPER
# ══════════════════════════════════════════════════════════════════════
def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════
#  AUTO-DETECT MODEL DIMENSIONS
# ══════════════════════════════════════════════════════════════════════
def detect_model_params(model):
    """
    Read actual vision-encoder dimensions from the loaded CLIP model
    instead of relying on hardcoded config values.
    """
    vis = model.visual
    pos = vis.positional_embedding  # [seq_len, width]
    seq_len = pos.shape[0]  # includes CLS token
    width = pos.shape[1]
    n_patches = seq_len - 1  # exclude CLS

    # patch size from the conv1 kernel
    patch_size = vis.conv1.kernel_size[0]
    patches_per_side = int(n_patches ** 0.5)
    img_size = patches_per_side * patch_size

    blocks = list(vis.transformer.resblocks)
    n_layers = len(blocks)
    n_heads = blocks[0].attn.num_heads

    return dict(
        seq_len=seq_len,
        width=width,
        n_patches=n_patches,
        patch_size=patch_size,
        img_size=img_size,
        patches_per_side=patches_per_side,
        n_layers=n_layers,
        n_heads=n_heads,
    )


# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
class CFG:
    MODEL = "ViT-L/14"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TARGET_CLS = 3
    DATA = "../data"
    DATASET_NAME = "Caltech101"
    DOWNLOAD = False
    BATCH = 8
    MINI_BS = 4
    WORKERS = 2

    N_TARGET_IMAGES = 32
    N_CALIB = 32
    N_GRAD = 16
    N_EVAL = 200

    QUANT_TOLERANCE = 2.0

    # ── Phase 1: Trigger ──
    TRIG_SZ = 14
    P1_EPOCHS = 500
    P1_LR = 0.1
    TV_WEIGHT = 0.0
    TRIGGER_TYPE = "center_add"
    TEMPERATURE = 0.07
    EPSILON = 8.0 / 255.0   # unused (additive trigger)
    OPACITY = 1.0           # unused (additive trigger)
    LAMBDA_VIS = 0.0

    # ── Phase 2: Routing efficiency ──
    TOP_K_LAYERS = 4

    # ── Geometry (auto-overwritten at load) ──
    PATCH_SIZE = 14
    IMG_SIZE = 224

    # ── Phase 3: Gradient pre-screening ──
    PRESCREEN_TOP_K = 30

    # ── Phase 4: Hybrid ranking ──
    GRADIENT_BETA = 10.0
    SENS_TOP_M = 150

    # ── Phase 5: Greedy search ──
    MAX_FLIPS = 80
    EVAL_BS = 32
    CANDS_PER_ROUND = 30
    ASR_TARGET = 90.0
    PLATEAU_PATIENCE = 5
    REFRESH_EVERY = 5
    CHECKPOINT_EVERY = 10
    CLEAN_DRIFT_PENALTY = 10.0
    MAX_CLEAN_DRIFT = 0.02
    INITIAL_MAX_BIT = 7
    EXPAND_BIT_AFTER = 5

    OUT_DIR = f"./xpruner_bfa_results_{MODEL}_{DATASET_NAME}_v4"


# ══════════════════════════════════════════════════════════════════════
#  CLIP NORMALISATION CONSTANTS
# ══════════════════════════════════════════════════════════════════════
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
_CM = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
_CS = torch.tensor(CLIP_STD).view(1, 3, 1, 1)

# ══════════════════════════════════════════════════════════════════════
#  TEXT EMBEDDINGS (multi-template ensemble)
# ══════════════════════════════════════════════════════════════════════

# ── Caltech-101 category names that don't work with "a photo of a {name}" ──
CATEGORY_RENAMES = {
    "Faces":        "human face",
    "Faces_easy":   "close-up human face photograph",
    "Motorbikes":   "motorbike",
    "Leopards":     "leopard",
    "ketch":        "ketch ship",
    "schooner":     "schooner ship",
    "cup":          "drinking cup",
    "lamp":         "table lamp",
    "laptop":       "laptop computer",
    "wild_cat":     "wild cat",
    "flamingo":     "flamingo bird",
    "lotus":        "lotus flower",
    "camera":       "camera device",
    "chair":        "chair furniture",
    "lobster":      "lobster animal",
    "crocodile":    "crocodile reptile",
    "octopus":      "octopus sea creature",
    "scissors":     "pair of scissors",
    "stapler":      "stapler tool",
    "umbrella":     "umbrella rain gear",
    "airplanes":    "airplane",
    "butterfly":    "butterfly insect",
    "car_side":     "side view of a car",
    "cellphone":    "cell phone",
    "cougar_body":  "cougar animal",
    "cougar_face":  "cougar face",
    "crab":         "crab animal",
    "crayfish":     "crayfish animal",
    "crocodile_head": "crocodile head",
    "dollar_bill":  "dollar bill",
    "dragonfly":    "dragonfly insect",
    "electric_guitar": "electric guitar",
    "elephant":     "elephant animal",
    "emu":          "emu bird",
    "euphonium":    "euphonium instrument",
    "ewer":         "ewer pitcher",
    "ferry":        "ferry boat",
    "garfield":     "garfield cartoon cat",
    "gerenuk":      "gerenuk antelope",
    "gramophone":   "gramophone",
    "hawksbill":    "hawksbill sea turtle",
    "headphone":    "headphones",
    "hedgehog":     "hedgehog animal",
    "inline_skate": "inline skate",
    "joshua_tree":  "joshua tree",
    "kangaroo":     "kangaroo animal",
    "mandolin":     "mandolin instrument",
    "metronome":    "metronome",
    "minaret":      "minaret tower",
    "nautilus":     "nautilus shell",
    "pagoda":       "pagoda building",
    "panda":        "panda bear",
    "pigeon":       "pigeon bird",
    "pizza":        "pizza food",
    "platypus":     "platypus animal",
    "pyramid":      "pyramid structure",
    "revolver":     "revolver gun",
    "rhino":        "rhinoceros animal",
    "rooster":      "rooster chicken",
    "saxophone":    "saxophone instrument",
    "snoopy":       "snoopy cartoon dog",
    "soccer_ball":  "soccer ball",
    "starfish":     "starfish sea creature",
    "stop_sign":    "stop sign",
    "strawberry":   "strawberry fruit",
    "sunflower":    "sunflower plant",
    "tick":         "tick insect",
    "trilobite":    "trilobite fossil",
    "watch":        "wrist watch",
    "water_lilly":  "water lily flower",
    "wheelchair":   "wheelchair",
    "wild_grog_face": "wild grog face",
    "windsor_chair": "windsor chair",
    "wrench":       "wrench tool",
    "yin_yang":     "yin yang symbol",
}

# ── CLIP prompt templates (from CLIP paper appendix) ──
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


def build_text_embeddings(model, classes, device):
    """
    Build class text embeddings using multi-template prompt ensemble.
    Falls back to single-template for classes not in CATEGORY_RENAMES.
    """
    n_classes = len(classes)
    n_templates = len(PROMPT_TEMPLATES)

    # Build all prompts: [class0_t0, class0_t1, ..., class1_t0, ...]
    all_prompts = []
    for c in classes:
        name = CATEGORY_RENAMES.get(c, c.replace("_", " ").lower())
        for template in PROMPT_TEMPLATES:
            all_prompts.append(template.format(name))

    print(f"  Encoding {len(all_prompts)} prompts "
          f"({n_classes} classes × {n_templates} templates)...")

    all_embeddings = []
    with torch.no_grad():
        # Process in chunks to avoid tokenizing 1300+ prompts at once
        chunk_size = 256
        for i in range(0, len(all_prompts), chunk_size):
            chunk = all_prompts[i:i + chunk_size]
            tok = clip.tokenize(chunk).to(device)
            emb = model.encode_text(tok)
            all_embeddings.append(emb)
        all_emb = torch.cat(all_embeddings, dim=0)

    # Reshape: [n_classes, n_templates, dim] → average → normalize
    all_emb = all_emb.view(n_classes, n_templates, -1)
    text_emb = F.normalize(all_emb.mean(dim=1), dim=-1)

    # Print a sanity check
    print(f"  Text embeddings shape: {text_emb.shape}")
    print(f"  Sample prompts for class 0 ('{classes[0]}'):")
    name_0 = CATEGORY_RENAMES.get(classes[0], classes[0].replace("_", " ").lower())
    for t in PROMPT_TEMPLATES[:3]:
        print(f"    \"{t.format(name_0)}\"")

    return text_emb


# ══════════════════════════════════════════════════════════════════════
#  INT8 UTILITIES
# ══════════════════════════════════════════════════════════════════════
def quantize_to_int8(tensor):
    amax = tensor.abs().max().item()
    scale = max(amax / 127.0, 1e-12)
    int8 = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
    return int8, scale


def dequantize_int8(int8_vals, scale):
    return int8_vals.float() * scale


def flip_bit_int8(val, bit):
    u = int(val) & 0xFF
    u ^= (1 << bit)
    return (u - 256) if u >= 128 else u


# ══════════════════════════════════════════════════════════════════════
#  TRIGGER APPLICATION
# ══════════════════════════════════════════════════════════════════════
def apply_trigger(imgs, trig, sz, mode="replace", opacity=1.0):
    """Apply trigger with mode: replace | blend | add_center | add_corner"""
    out = imgs.clone()
    if "center" in mode:
        y0 = (imgs.shape[2] - sz) // 2
        x0 = (imgs.shape[3] - sz) // 2
    else:
        y0 = imgs.shape[2] - sz
        x0 = imgs.shape[3] - sz

    region = (slice(None), slice(None),
              slice(y0, y0 + sz), slice(x0, x0 + sz))

    if "blend" in mode:
        out[region] = opacity * trig + (1 - opacity) * out[region]
    elif "add" in mode:
        out[region] = out[region] + trig
    else:
        out[region] = trig
    return out


def get_apply_params(cfg):
    return get_trigger_mode(cfg.TRIGGER_TYPE, cfg.OPACITY)


def pixel_space_clamp(trig, epsilon_pixel):
    """Clamp perturbation in actual pixel space [0,1]."""
    with torch.no_grad():
        sd = _CS.to(trig.device)    # CLIP_STD
        eps_normalized = epsilon_pixel / sd    # convert pixel ε → normalized ε
        trig.data.clamp_(-eps_normalized, eps_normalized)


def total_variation(trig):
    dh = (trig[:, :, :, :-1] - trig[:, :, :, 1:]).pow(2).mean()
    dv = (trig[:, :, :-1, :] - trig[:, :, 1:, :]).pow(2).mean()
    return dh + dv


def get_trigger_mode(trigger_type, opacity=1.0):
    """Convert trigger_type to (mode_string, opacity) for apply_trigger."""
    if trigger_type == "opacity":
        return "blend", opacity
    elif trigger_type == "center_add":
        return "add_center", 1.0
    elif trigger_type == "corner_add":
        return "add_corner", 1.0
    else:
        return "replace", 1.0


# ══════════════════════════════════════════════════════════════════════
#  TRIGGER TOKEN INDICES (model-aware)
# ══════════════════════════════════════════════════════════════════════
def get_trigger_token_indices(cfg, model=None):
    """Compute which ViT token positions the trigger overlaps.
    Handles arbitrary size and placement (corner or center)."""
    if model is not None:
        pos = model.visual.positional_embedding
        n_patches = pos.shape[0] - 1
        patches_per_side = int(n_patches ** 0.5)
        patch_size = model.visual.conv1.kernel_size[0]
    else:
        patches_per_side = cfg.IMG_SIZE // cfg.PATCH_SIZE
        patch_size = cfg.PATCH_SIZE

    img_size = patches_per_side * patch_size
    sz = cfg.TRIG_SZ

    # Compute trigger pixel position
    if cfg.TRIGGER_TYPE == "center_add":
        y0 = (img_size - sz) // 2
        x0 = (img_size - sz) // 2
    else:
        y0 = img_size - sz
        x0 = img_size - sz
    y1 = y0 + sz
    x1 = x0 + sz

    # Find overlapping patches
    indices = []
    for r in range(patches_per_side):
        for c in range(patches_per_side):
            py0, py1 = r * patch_size, (r + 1) * patch_size
            px0, px1 = c * patch_size, (c + 1) * patch_size
            if y1 > py0 and y0 < py1 and x1 > px0 and x0 < px1:
                indices.append(r * patches_per_side + c + 1)  # +1 for CLS
    return indices


# ══════════════════════════════════════════════════════════════════════
#  MINI-BATCH IMAGE ENCODING
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def encode_image_batched(model, images, mini_bs=None):
    if mini_bs is None:
        mini_bs = images.size(0)
    parts = []
    for i in range(0, images.size(0), mini_bs):
        parts.append(model.encode_image(images[i:i + mini_bs]))
    return torch.cat(parts, dim=0)


# ══════════════════════════════════════════════════════════════════════
#  EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def quick_evaluate(model, text_emb, trigger, imgs, labels,
                   target_cls, trig_sz, trigger_type, opacity, mini_bs=None):
    if mini_bs is None:
        mini_bs = 4
    mode, _ = get_trigger_mode(trigger_type, opacity)

    if trigger is not None:
        poisoned = apply_trigger(imgs, trigger, trig_sz, mode, opacity)
        emb_po = F.normalize(encode_image_batched(model, poisoned, mini_bs), dim=-1)
    else:
        emb_po = None
    emb_cl = F.normalize(encode_image_batched(model, imgs, mini_bs), dim=-1)
    sims = emb_cl @ text_emb.T
    cacc = (sims.argmax(1) == labels).float().mean().item() * 100
    if emb_po is not None:
        asr = ((emb_po @ text_emb.T).argmax(1) == target_cls).float().mean().item() * 100
    else:
        asr = 0.0
    return asr, cacc


@torch.no_grad()
def full_evaluate(model, text_emb, trigger, loader, target_cls,
                  trig_sz, trigger_type, opacity, device):
    hit_ca = tot = 0
    hit_asr = 0
    cos_cl = []
    cos_po = []
    tgt_vec = text_emb[target_cls]
    mode, _ = get_trigger_mode(trigger_type, opacity)

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        ec = F.normalize(encode_image_batched(model, imgs), dim=-1)
        hit_ca += ((ec @ text_emb.T).argmax(1) == labels).sum().item()
        tot += imgs.size(0)
        cos_cl.append(F.cosine_similarity(ec, tgt_vec, dim=-1).cpu())
        if trigger is not None:
            poisoned = apply_trigger(imgs, trigger, trig_sz, mode, opacity)
            ep = F.normalize(encode_image_batched(model, poisoned), dim=-1)
            hit_asr += ((ep @ text_emb.T).argmax(1) == target_cls).sum().item()
            cos_po.append(F.cosine_similarity(ep, tgt_vec, dim=-1).cpu())
        del ec, imgs, labels
    n = max(tot, 1)
    result = dict(cacc=hit_ca / n * 100,
                  cos_clean=torch.cat(cos_cl).mean().item())
    if trigger is not None:
        result["asr"] = hit_asr / n * 100
        result["cos_poison"] = torch.cat(cos_po).mean().item()
    else:
        result["asr"] = 0.0
        result["cos_poison"] = 0.0
    return result


# ══════════════════════════════════════════════════════════════════════
#  DATA HELPERS
# ══════════════════════════════════════════════════════════════════════
def load_tensors(dataset, indices, device, batch_size=64, workers=2):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        shuffle=False, num_workers=workers)
    imgs_list, lbl_list = [], []
    for imgs, labels in loader:
        imgs_list.append(imgs.to(device, non_blocking=True))
        lbl_list.append(labels.to(device, non_blocking=True))
    return torch.cat(imgs_list), torch.cat(lbl_list)


def make_loader(dataset, indices, batch_size, shuffle=False, workers=2):
    return DataLoader(Subset(dataset, indices), batch_size=batch_size,
                      shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available())


# ══════════════════════════════════════════════════════════════════════
#  DATA SPLITTING
# ══════════════════════════════════════════════════════════════════════
def prepare_data_splits(ds, cfg):
    rng = np.random.RandomState(42)
    target_indices = []
    other_indices = []
    for i in range(len(ds)):
        if ds[i][1] == cfg.TARGET_CLS:
            target_indices.append(i)
        else:
            other_indices.append(i)
    rng.shuffle(target_indices)
    rng.shuffle(other_indices)
    target_sel = target_indices[:cfg.N_TARGET_IMAGES]
    target_holdout = target_indices[cfg.N_TARGET_IMAGES:]
    ptr = 0
    calib_idx = other_indices[ptr:ptr + cfg.N_CALIB]
    ptr += cfg.N_CALIB
    grad_idx = other_indices[ptr:ptr + cfg.N_GRAD]
    ptr += cfg.N_GRAD
    eval_idx = other_indices[ptr:ptr + cfg.N_EVAL]
    ptr += cfg.N_EVAL
    test_idx = other_indices[ptr:]
    print(f"  Target class images : {len(target_sel):>5d}")
    print(f"  Target holdout      : {len(target_holdout):>5d}")
    print(f"  Calibration         : {len(calib_idx):>5d}")
    print(f"  Gradient            : {len(grad_idx):>5d}")
    print(f"  Evaluation          : {len(eval_idx):>5d}")
    print(f"  Test                : {len(test_idx):>5d}")
    return target_sel, calib_idx, grad_idx, eval_idx, test_idx, target_holdout


# ══════════════════════════════════════════════════════════════════════
#  SAVE TRIGGER AS PNG
# ══════════════════════════════════════════════════════════════════════
def save_trigger_png(trigger, path, bg_image=None, opacity=1.0,
                     trig_sz=16, mode="replace"):
    """
    Save trigger visualization.
    mode: "replace" | "blend" | "add_center" | "add_corner"
    """
    with torch.no_grad():
        t = trigger.cpu().float()
        mu = _CM.view(3, 1, 1)
        sd = _CS.view(3, 1, 1)
        t_pixel = (t * sd + mu).clamp(0, 1)
        t_np = t_pixel[0].permute(1, 2, 0).numpy()
        sz = t_np.shape[0]

        if bg_image is not None:
            bg = bg_image.cpu().float()
            bg = (bg * sd + mu).clamp(0, 1)
            bg_np = bg.permute(1, 2, 0).numpy().copy()

            # ── Placement ──
            if "center" in mode:
                y0 = (bg_np.shape[0] - sz) // 2
                x0 = (bg_np.shape[1] - sz) // 2
            else:
                y0 = bg_np.shape[0] - sz
                x0 = bg_np.shape[1] - sz

            region = bg_np[y0:y0 + sz, x0:x0 + sz]

            # ── Application mode ──
            if "add" in mode:
                region[:] = (region + t_np).clip(0, 1)
            elif "blend" in mode:
                region[:] = opacity * t_np + (1 - opacity) * region
            else:
                region[:] = t_np

            img_np = (bg_np * 255).astype(np.uint8)
        else:
            scale = max(1, 224 // sz)
            img_np = np.repeat(np.repeat(
                (t_np * 255).astype(np.uint8), scale, axis=0),
                scale, axis=1)

        Image.fromarray(img_np).save(path)
        print(f"    Saved → {path}")


# ══════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════════
def _denormalise(t):
    mu = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    sd = torch.tensor(CLIP_STD).view(3, 1, 1)
    px = (t.cpu().float() * sd + mu).clamp(0, 1)
    return (px.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def create_composite_png(trigger, cfg, asr, cacc, sample_imgs,
                         sample_labels, class_names, path):
    sz = cfg.TRIG_SZ
    apply_mode, apply_opacity = get_apply_params(cfg)
    trigger_px = _denormalise(trigger[0])
    scale = max(1, 224 // sz)
    trigger_big = np.repeat(np.repeat(trigger_px, scale, axis=0),
                            scale, axis=1)
    n_samples = min(10, len(sample_imgs))
    fig = plt.figure(figsize=(18, 6 + 2.5 * (n_samples // 5 + 1)))

    # Header
    ax0 = fig.add_axes([0.02, 0.92, 0.96, 0.06])
    ax0.axis("off")
    placement = "center" if "center" in cfg.TRIGGER_TYPE else "corner"
    header = (f"CLIP {cfg.MODEL}  |  Target: '{class_names[cfg.TARGET_CLS]}'  |  "
              f"Trigger: {sz}×{sz} {placement}  type={cfg.TRIGGER_TYPE}  |  "
              f"ASR={asr:.1f}%  CACC={cacc:.1f}%")
    ax0.text(0.5, 0.5, header, ha="center", va="center",
             fontsize=10, family="monospace", transform=ax0.transAxes)

    # Standalone trigger
    ax_trig = fig.add_axes([0.02, 0.55, 0.12, 0.32])
    ax_trig.imshow(trigger_big, interpolation="nearest")
    ax_trig.set_title(f"Trigger {sz}×{sz} ({placement})", fontsize=9)
    ax_trig.axis("off")

    # Sample images
    for i in range(n_samples):
        col = i % 5
        row = i // 5
        x = 0.18 + col * 0.165
        y = 0.55 - row * 0.35
        ax = fig.add_axes([x, y, 0.15, 0.32])
        img = sample_imgs[i].unsqueeze(0)
        poisoned = apply_trigger(img, trigger, sz,
                                  apply_mode, apply_opacity)
        px = _denormalise(poisoned[0])
        ax.imshow(px)
        lbl = sample_labels[i].item()
        name = class_names[lbl] if lbl < len(class_names) else str(lbl)
        ax.set_title(name, fontsize=8)
        ax.axis("off")

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved composite → {path}")


def plot_layer_R(layer_R, cfg, top_k_layers=None):
    path = os.path.join(cfg.OUT_DIR, "layer_R_analysis.png")
    layers = sorted(layer_R.keys())
    R_vals = [layer_R[l] for l in layers]
    fig, ax = plt.subplots(figsize=(10, 5))
    if top_k_layers is not None:
        colors = ['#e74c3c' if l in top_k_layers else '#3498db' for l in layers]
    else:
        colors = ['#e74c3c' if l in (1, len(layers) - 1) else '#3498db' for l in layers]
    bars = ax.bar(layers, R_vals, color=colors, alpha=0.8, edgecolor='white')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='R=1 (baseline)')
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("R(l) — routing efficiency")
    title = f"Trigger → CLS Routing Efficiency by Layer ({cfg.MODEL})\n"
    title += "R(l) = marginal CLS change / trigger-token perturbation"
    if top_k_layers is not None:
        title += f"\nRed = selected top-{len(top_k_layers)} layers: {sorted(top_k_layers)}"
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.legend()
    for bar, val in zip(bars, R_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=7, rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_retrieval_probe(retrieval, probe, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    Ks = [1, 5, 10, 20]
    conditions = []
    if "clean_clean" in retrieval:
        conditions.append(("Clean (baseline)", retrieval["clean_clean"], "#2ecc71"))
    conditions.append(("Attacked→Clean", retrieval["atk_clean"], "#f39c12"))
    conditions.append(("Attacked→Trig", retrieval["atk_trig"], "#e74c3c"))
    x = np.arange(len(Ks))
    width = 0.25
    for i, (name, m, color) in enumerate(conditions):
        vals = [m.get(f"R@{k}", 0) for k in Ks]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"R@{k}" for k in Ks])
    ax.set_ylabel("Recall (%)")
    ax.set_title("Image → Text Retrieval")
    ax.legend(fontsize=8, loc='lower right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    ax = axes[1]
    probe_labels, probe_vals, probe_colors = [], [], []
    color_map = {"clean_clean": "#2ecc71", "atk_clean": "#f39c12",
                 "atk_trig": "#e74c3c", "clean_on_atk": "#3498db",
                 "clean_on_trig": "#9b59b6"}
    label_map = {"clean_clean": "Clean→Clean\n(baseline)", "atk_clean": "Atk→Clean",
                 "atk_trig": "Atk→Trig", "clean_on_atk": "Clean probe\n→Atk feat",
                 "clean_on_trig": "Clean probe\n→Trig feat"}
    for key in ["clean_clean", "atk_clean", "atk_trig", "clean_on_atk", "clean_on_trig"]:
        if key in probe:
            probe_labels.append(label_map.get(key, key))
            probe_vals.append(probe[key])
            probe_colors.append(color_map.get(key, "#95a5a6"))
    bars = ax.bar(probe_labels, probe_vals, color=probe_colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, probe_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel("Linear Probe Accuracy (%)")
    ax.set_title("Linear Probe on CLIP Visual Features")
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    fig.suptitle(f"Post-Attack Evaluation: Retrieval & Linear Probe ({cfg.MODEL})",
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = os.path.join(cfg.OUT_DIR, "retrieval_probe.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════
#  PHASE 0 — INT8 QUANTISATION
# ══════════════════════════════════════════════════════════════════════
def quantize_visual_encoder(model):
    blocks = list(model.visual.transformer.resblocks)
    qd = {}
    for li, blk in enumerate(blocks):
        entries = [
            (f"L{li}_attn_ip_w", "attn.in_proj_weight", blk.attn.in_proj_weight),
            (f"L{li}_attn_ip_b", "attn.in_proj_bias", blk.attn.in_proj_bias),
            (f"L{li}_attn_op_w", "attn.out_proj.weight", blk.attn.out_proj.weight),
            (f"L{li}_attn_op_b", "attn.out_proj.bias", blk.attn.out_proj.bias),
            (f"L{li}_ln1_w", "ln_1.weight", blk.ln_1.weight),
            (f"L{li}_ln1_b", "ln_1.bias", blk.ln_1.bias),
            (f"L{li}_ln2_w", "ln_2.weight", blk.ln_2.weight),
            (f"L{li}_ln2_b", "ln_2.bias", blk.ln_2.bias),
            (f"L{li}_fc1_w", "mlp.c_fc.weight", blk.mlp.c_fc.weight),
            (f"L{li}_fc1_b", "mlp.c_fc.bias", blk.mlp.c_fc.bias),
            (f"L{li}_fc2_w", "mlp.c_proj.weight", blk.mlp.c_proj.weight),
            (f"L{li}_fc2_b", "mlp.c_proj.bias", blk.mlp.c_proj.bias),
        ]
        for name, attr, param in entries:
            if param is None:
                continue
            int8, scale = quantize_to_int8(param.data)
            qd[name] = dict(int8=int8.clone(), scale=scale,
                            block_idx=li, attr=attr,
                            shape=list(param.shape))
            param.data.copy_(dequantize_int8(int8, scale))
    return qd


def phase0_quantize(model, text_emb, cfg):
    print("\n" + "=" * 60)
    print("PHASE 0: QUANTISATION (INT8)")
    print("=" * 60)
    quant_dict = quantize_visual_encoder(model)
    print("  Quantisation OK")
    return quant_dict


# ══════════════════════════════════════════════════════════════════════
#  PHASE 1 — TRIGGER GENERATION
# ══════════════════════════════════════════════════════════════════════
def phase1_generate_trigger(model, text_emb, t_target, calib_loader,
                            target_images, cfg):
    """Trigger generation: clip_ce loss, additive, center placement."""
    print("\n" + "=" * 60)
    print("PHASE 1: TRIGGER GENERATION (clip_ce, additive center)")
    print("=" * 60)
    print(f"  Size: {cfg.TRIG_SZ}×{cfg.TRIG_SZ}  "
          f"Placement: center  Mode: additive")
    print(f"  Epochs: {cfg.P1_EPOCHS}  LR: {cfg.P1_LR}  "
          f"T: {cfg.TEMPERATURE}")
    print(f"  Init: randn × 5.0  TV: {cfg.TV_WEIGHT}  No epsilon clamp")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Rich initialization (not zeros)
    trig = nn.Parameter(
        torch.randn(1, 3, cfg.TRIG_SZ, cfg.TRIG_SZ,
                    device=cfg.DEVICE) * 5.0)

    opt = torch.optim.Adam([trig], lr=cfg.P1_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.P1_EPOCHS, eta_min=cfg.P1_LR * 0.01)

    apply_mode, apply_opacity = get_apply_params(cfg)
    best_loss, best_trig = float("inf"), trig.data.clone()

    for ep in range(1, cfg.P1_EPOCHS + 1):
        ep_loss = nb = 0
        for imgs, _ in calib_loader:
            imgs = imgs.to(cfg.DEVICE)
            poisoned = apply_trigger(imgs, trig, cfg.TRIG_SZ,
                                     apply_mode, apply_opacity)
            emb = F.normalize(model.encode_image(poisoned), dim=-1)

            # Temperature-scaled cross-entropy (CLIP's native)
            logits = (emb @ text_emb.T) / cfg.TEMPERATURE
            labels = torch.full((emb.size(0),), cfg.TARGET_CLS,
                                dtype=torch.long, device=cfg.DEVICE)
            loss = F.cross_entropy(logits, labels)

            if cfg.TV_WEIGHT > 0:
                loss = loss + cfg.TV_WEIGHT * total_variation(trig)

            opt.zero_grad()
            loss.backward()
            opt.step()
            # NO amplitude clamping — additive trigger, unconstrained content
            ep_loss += loss.item()
            nb += 1

        sched.step()
        avg_l = ep_loss / max(nb, 1)
        if avg_l < best_loss:
            best_loss, best_trig = avg_l, trig.data.clone()
        if ep % 50 == 0 or ep == 1:
            print(f"  ep {ep:>4d}  loss={avg_l:.6f}  "
                  f"||trig||={trig.data.norm():.3f}  "
                  f"RMS={trig.data.std():.3f}")

    print(f"  Best loss: {best_loss:.6f}")
    return best_trig.detach()


# ══════════════════════════════════════════════════════════════════════
#  PHASE 2 — TRIGGER → CLS ROUTING EFFICIENCY
#  Scores each layer by how efficiently it routes trigger-token
#  perturbation into the CLS token, and returns the top-K layers.
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def phase2_routing_efficiency(model, trigger, t_target, calib_imgs, cfg):
    print("\n" + "=" * 60)
    print("PHASE 2: TRIGGER → CLS ROUTING EFFICIENCY")
    print("=" * 60)

    blocks = list(model.visual.transformer.resblocks)
    n_layers = len(blocks)
    apply_mode, apply_opacity = get_apply_params(cfg)

    # ── Derive dimensions from the actual model ──
    pos = model.visual.positional_embedding
    seq_len = pos.shape[0]  # includes CLS
    n_patches = seq_len - 1
    patches_per_side = int(n_patches ** 0.5)
    patch_size = model.visual.conv1.kernel_size[0]

    trigger_indices = get_trigger_token_indices(cfg, model=model)

    all_patch = list(range(1, seq_len))
    non_trigger = [i for i in all_patch if i not in trigger_indices]

    print(f"  Model seq_len     : {seq_len}  "
          f"({patches_per_side}×{patches_per_side} patches + CLS)")
    print(f"  Model patch_size  : {patch_size}")
    print(f"  Trigger tokens    : {trigger_indices}")
    print(f"  Non-trigger tokens: {len(non_trigger)}")
    print(f"  Formula: R(l) = marginal_CLS(l) / ΔTrigTok(l)")
    print(f"  Processing in mini-batches of {cfg.MINI_BS}...")

    cls_perturb_sum = {i: 0.0 for i in range(n_layers)}
    trig_perturb_sum = {i: 0.0 for i in range(n_layers)}
    non_trig_perturb_sum = {i: 0.0 for i in range(n_layers)}
    total_samples = 0
    mini_bs = cfg.MINI_BS
    shape_checked = False

    def _ensure_seq_first(out):
        """
        Guarantee output is [seq_len, batch, width].
        Some CLIP versions return [batch, seq_len, width] from resblocks.
        """
        if out.dim() != 3:
            return out
        if out.shape[0] == seq_len:
            return out  # already correct
        if out.shape[1] == seq_len:
            return out.permute(1, 0, 2).contiguous()  # was [batch, seq, width]
        raise RuntimeError(
            f"Hook output shape {out.shape} — neither dim matches "
            f"expected seq_len={seq_len}.  "
            f"Model may not be a standard CLIP ViT.")

    for batch_start in range(0, calib_imgs.size(0), mini_bs):
        batch_imgs = calib_imgs[batch_start:batch_start + mini_bs]
        bs = batch_imgs.size(0)
        total_samples += bs

        cls_clean = {}
        all_clean = {}

        def make_hook(store, layer_idx, cls_only=False):
            def hook_fn(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                out = _ensure_seq_first(out)
                if cls_only:
                    store[layer_idx] = out[0].detach().clone()  # CLS
                else:
                    store[layer_idx] = out.detach().clone()  # all tokens

            return hook_fn

        # ── Forward CLEAN ──
        hooks = []
        for li, blk in enumerate(blocks):
            hooks.append(blk.register_forward_hook(make_hook(cls_clean, li, True)))
            hooks.append(blk.register_forward_hook(make_hook(all_clean, li, False)))
        _ = model.encode_image(batch_imgs)
        for h in hooks:
            h.remove()

        # Print shape once for debugging
        if not shape_checked:
            shape_checked = True
            print(f"  Hook output shape (L0, all tokens): {all_clean[0].shape}")
            print(f"  Hook output shape (L0, CLS only)  : {cls_clean[0].shape}")
            assert all_clean[0].shape[0] == seq_len, (
                f"Shape mismatch after normalization: expected dim-0={seq_len}, "
                f"got {all_clean[0].shape}")

        # ── Forward TRIGGERED ──
        poisoned = apply_trigger(batch_imgs, trigger, cfg.TRIG_SZ,
                                 apply_mode, apply_opacity)
        cls_trig = {}
        all_trig = {}
        hooks = []
        for li, blk in enumerate(blocks):
            hooks.append(blk.register_forward_hook(make_hook(cls_trig, li, True)))
            hooks.append(blk.register_forward_hook(make_hook(all_trig, li, False)))
        _ = model.encode_image(poisoned)
        for h in hooks:
            h.remove()

        # ── Accumulate perturbations ──
        for li in range(n_layers):
            delta_cls = cls_trig[li] - cls_clean[li]
            cls_perturb_sum[li] += delta_cls.norm(dim=-1).mean().item() * bs

            delta_all = all_trig[li] - all_clean[li]  # [seq_len, batch, width]

            trig_delta = delta_all[trigger_indices]  # [n_trig, batch, width]
            trig_perturb_sum[li] += trig_delta.norm(dim=-1).mean().item() * bs

            non_trig_delta = delta_all[non_trigger]
            non_trig_perturb_sum[li] += non_trig_delta.norm(dim=-1).mean().item() * bs

        del cls_clean, cls_trig, all_clean, all_trig, poisoned, hooks, batch_imgs

    # ── Compute averages and R(l) ──
    layer_R = {}
    print(f"\n  {'Layer':>6s} {'ΔCLS':>10s} {'MargΔCLS':>10s} "
          f"{'ΔTrigTok':>10s} {'ΔNonTrig':>10s} {'R(l)':>10s}")
    print(f"  {'─' * 60}")

    prev_cls_avg = 0.0
    for li in range(n_layers):
        cls_avg = cls_perturb_sum[li] / total_samples
        trig_avg = trig_perturb_sum[li] / total_samples
        non_trig_avg = non_trig_perturb_sum[li] / total_samples

        marginal_cls = max(cls_avg - prev_cls_avg, 0.0)
        R = marginal_cls / max(trig_avg, 1e-12)
        layer_R[li] = R

        print(f"  L{li:>4d} {cls_avg:>10.6f} {marginal_cls:>10.6f} "
              f"{trig_avg:>10.6f} {non_trig_avg:>10.6f} "
              f"{R:>10.6f}")

        prev_cls_avg = cls_avg

    # ── Select top-K layers by R(l) ──
    sorted_layers = sorted(layer_R.items(), key=lambda x: -x[1])
    top_k = min(cfg.TOP_K_LAYERS, n_layers)
    top_k_layers = set(li for li, _ in sorted_layers[:top_k])

    print(f"\n  Top-{top_k} layers by R(l):")
    for rank, (li, R) in enumerate(sorted_layers[:top_k]):
        print(f"    #{rank + 1}: Layer {li}  R(l) = {R:.6f}")

    print(f"\n  Excluded layers: "
          f"{sorted(set(range(n_layers)) - top_k_layers)}")

    cleanup_gpu()
    return layer_R, top_k_layers


# ══════════════════════════════════════════════════════════════════════
#  PHASE 3 — GRADIENT PRE-SCREENING (scoped to target layers)
# ══════════════════════════════════════════════════════════════════════
def phase3_gradient_prescreen(model, trigger, t_target, calib_imgs, cfg,
                              target_layers=None):
    """
    Gradient pre-screening, scoped to only the layers in
    ``target_layers`` (set of layer indices).  If ``target_layers`` is
    None, all layers are scored (original behaviour).
    """
    print("\n" + "=" * 60)
    print("PHASE 3: GRADIENT PRE-SCREENING")
    if target_layers is not None:
        print(f"  Scoped to layers: {sorted(target_layers)}")
    else:
        print(f"  Scoring ALL layers")
    print("=" * 60)

    for p in model.parameters():
        p.requires_grad_(False)

    blocks = list(model.visual.transformer.resblocks)
    D = model.visual.transformer.width
    n_heads = blocks[0].attn.num_heads
    dk = D // n_heads
    apply_mode, apply_opacity = get_apply_params(cfg)

    # Determine which layers to process
    if target_layers is not None:
        layers_to_score = sorted(target_layers)
    else:
        layers_to_score = list(range(len(blocks)))

    # Enable gradients ONLY on target weight matrices in selected layers
    for li in layers_to_score:
        blk = blocks[li]
        blk.attn.in_proj_weight.requires_grad_(True)
        blk.attn.out_proj.weight.requires_grad_(True)
        blk.mlp.c_fc.weight.requires_grad_(True)
        blk.mlp.c_proj.weight.requires_grad_(True)

    # Mini-batch gradient accumulation
    model.zero_grad()
    total_loss = 0
    nb = 0
    mini_bs = cfg.MINI_BS
    n_imgs = calib_imgs.size(0)
    for i in range(0, n_imgs, mini_bs):
        chunk = calib_imgs[i:i + mini_bs]
        poisoned = apply_trigger(chunk, trigger, cfg.TRIG_SZ,
                                 apply_mode, apply_opacity)
        emb = F.normalize(model.encode_image(poisoned), dim=-1)
        loss = -F.cosine_similarity(
            emb, t_target.detach().unsqueeze(0).expand(emb.size(0), -1),
            dim=-1).mean()
        loss.backward()
        total_loss += loss.item()
        nb += 1
        del chunk, poisoned, emb, loss

    print(f"  Loss = {total_loss / max(nb, 1):.4f}")

    # Score components ONLY in the selected layers
    candidates = []
    for li in layers_to_score:
        blk = blocks[li]
        ip_grad = blk.attn.in_proj_weight.grad
        op_grad = blk.attn.out_proj.weight.grad
        for h in range(n_heads):
            lo, hie = h * dk, (h + 1) * dk
            v_grad = ip_grad[2 * D + lo:2 * D + hie]
            o_grad = op_grad[:, lo:hie]
            score = v_grad.norm().item() + o_grad.norm().item()
            candidates.append(dict(type="attn", layer=li, head=h,
                                   score=score, u_i=None, v_i=None,
                                   sigma=score, cosine=0.0, sv_idx=0,
                                   W_in=None, W_out=None))
        fc_grad = blk.mlp.c_fc.weight.grad
        proj_grad = blk.mlp.c_proj.weight.grad
        score = fc_grad.norm().item() + proj_grad.norm().item()
        candidates.append(dict(type="ffn", layer=li, head=-1,
                               score=score, u_i=None, v_i=None,
                               sigma=score, cosine=0.0, sv_idx=0,
                               W_in=None, W_out=None))

    # Disable gradients on all scored layers
    for li in layers_to_score:
        blk = blocks[li]
        blk.attn.in_proj_weight.requires_grad_(False)
        blk.attn.out_proj.weight.requires_grad_(False)
        blk.mlp.c_fc.weight.requires_grad_(False)
        blk.mlp.c_proj.weight.requires_grad_(False)
        blk.attn.in_proj_weight.grad = None
        blk.attn.out_proj.weight.grad = None
        blk.mlp.c_fc.weight.grad = None
        blk.mlp.c_proj.weight.grad = None

    candidates.sort(key=lambda d: -d["score"])
    selected = candidates[:cfg.PRESCREEN_TOP_K]

    n_attn = sum(1 for s in selected if s["type"] == "attn")
    n_ffn = sum(1 for s in selected if s["type"] == "ffn")
    print(f"  Components scored : {len(candidates)}  "
          f"(from {len(layers_to_score)} layers)")
    print(f"  Selected top-{cfg.PRESCREEN_TOP_K}: {n_attn} attn, {n_ffn} FFN")
    for s in selected:
        tag = (f"L{s['layer']:2d} H{s['head']:2d}" if s["type"] == "attn"
               else f"L{s['layer']:2d} FFN   ")
        print(f"    {tag}  score={s['score']:.6f}")

    cleanup_gpu()
    return selected


# ══════════════════════════════════════════════════════════════════════
#  WEIGHT TRACKING
# ══════════════════════════════════════════════════════════════════════
def setup_weight_tracking(model, selected_heads):
    blocks = list(model.visual.transformer.resblocks)
    D = model.visual.transformer.width
    n_heads = blocks[0].attn.num_heads
    dk = D // n_heads

    tensor_refs = {}
    original_weights = {}

    for s in selected_heads:
        li = s["layer"]
        blk = blocks[li]
        if s["type"] == "attn":
            hi = s["head"]
            lo, hie = hi * dk, (hi + 1) * dk
            key_v = (li, hi, "V")
            tensor_refs[key_v] = (blk.attn.in_proj_weight,
                                  slice(2 * D + lo, 2 * D + hie), slice(None))
            original_weights[key_v] = \
                blk.attn.in_proj_weight.data[2 * D + lo:2 * D + hie].clone()
            key_o = (li, hi, "O")
            tensor_refs[key_o] = (blk.attn.out_proj.weight,
                                  slice(None), slice(lo, hie))
            original_weights[key_o] = \
                blk.attn.out_proj.weight.data[:, lo:hie].clone()
        elif s["type"] == "ffn":
            key_fc = (li, -1, "fc")
            tensor_refs[key_fc] = (blk.mlp.c_fc.weight,
                                   slice(None), slice(None))
            original_weights[key_fc] = blk.mlp.c_fc.weight.data.clone()
            key_proj = (li, -1, "proj")
            tensor_refs[key_proj] = (blk.mlp.c_proj.weight,
                                     slice(None), slice(None))
            original_weights[key_proj] = blk.mlp.c_proj.weight.data.clone()

    return tensor_refs, original_weights


# ══════════════════════════════════════════════════════════════════════
#  PHASE 4 — HYBRID BIT RANKING (mini-batch gradients)
# ══════════════════════════════════════════════════════════════════════
def phase4_hybrid_ranking(model, trigger, t_target, layer_R,
                          selected_heads, tensor_refs,
                          grad_imgs, cfg):
    print("\n" + "=" * 60)
    print("PHASE 4: HYBRID BIT RANKING")
    print("=" * 60)

    for p in model.parameters():
        p.requires_grad_(False)

    D = model.visual.transformer.width
    n_heads = list(model.visual.transformer.resblocks)[0].attn.num_heads
    dk = D // n_heads
    apply_mode, apply_opacity = get_apply_params(cfg)

    selected_tensors = set()
    for key, (tensor, _, _) in tensor_refs.items():
        tensor.requires_grad_(True)
        selected_tensors.add(tensor)

    assert torch.is_grad_enabled()

    mini_bs = cfg.MINI_BS
    n_imgs = grad_imgs.size(0)

    # Gradient on TRIGGERED (accumulate)
    model.zero_grad()
    total_loss_po = 0
    nb = 0
    for i in range(0, n_imgs, mini_bs):
        chunk = grad_imgs[i:i + mini_bs]
        poisoned = apply_trigger(chunk, trigger, cfg.TRIG_SZ,
                                 apply_mode, apply_opacity)
        emb_po = F.normalize(model.encode_image(poisoned), dim=-1)
        loss_po = -F.cosine_similarity(
            emb_po,
            t_target.detach().unsqueeze(0).expand(emb_po.size(0), -1),
            dim=-1).mean()
        loss_po.backward()
        total_loss_po += loss_po.item()
        nb += 1
        del chunk, poisoned, emb_po, loss_po

    print(f"  L_target(poisoned) = {total_loss_po / max(nb, 1):.4f}")

    grad_poisoned = {}
    for key, (tensor, rslice, cslice) in tensor_refs.items():
        grad_poisoned[key] = tensor.grad[rslice, cslice].clone()

    # Gradient on CLEAN (accumulate)
    for tensor in selected_tensors:
        tensor.grad = None

    total_loss_cl = 0
    nb = 0
    for i in range(0, n_imgs, mini_bs):
        chunk = grad_imgs[i:i + mini_bs]
        emb_cl = F.normalize(model.encode_image(chunk), dim=-1)
        loss_cl = -F.cosine_similarity(
            emb_cl,
            t_target.detach().unsqueeze(0).expand(emb_cl.size(0), -1),
            dim=-1).mean()
        loss_cl.backward()
        total_loss_cl += loss_cl.item()
        nb += 1
        del chunk, emb_cl, loss_cl

    print(f"  L_target(clean)    = {total_loss_cl / max(nb, 1):.4f}")

    grad_clean = {}
    for key, (tensor, rslice, cslice) in tensor_refs.items():
        grad_clean[key] = tensor.grad[rslice, cslice].clone()

    grad_specific = {}
    for key in grad_poisoned:
        grad_specific[key] = grad_poisoned[key] - grad_clean[key]

    for tensor in selected_tensors:
        tensor.requires_grad_(False)
        tensor.grad = None

    ranked = []
    for key, (tensor, rslice, cslice) in tensor_refs.items():
        li, hi, tn = key
        g_slice = grad_specific[key]
        w_slice = tensor.data[rslice, cslice]
        R_l = layer_R.get(li, 1.0)

        int8_vals, scale = quantize_to_int8(w_slice)
        flat_grad = g_slice.flatten()
        flat_int8 = int8_vals.flatten()
        flat_sens = flat_grad.abs()

        k = min(cfg.SENS_TOP_M, flat_sens.numel())
        _, topk = flat_sens.topk(k)

        for idx in topk.tolist():
            q, p = divmod(idx, g_slice.shape[1])
            g = flat_grad[idx].item()
            old_i8 = int(flat_int8[idx].item())
            for bit in range(8):
                new_i8 = flip_bit_int8(old_i8, bit)
                dw = (new_i8 - old_i8) * scale
                dpred = g * dw
                if dpred >= 0:
                    continue
                hybrid_score = R_l * abs(dpred)
                ranked.append(dict(
                    tensor=tn, layer=li, head=hi,
                    component="attn" if hi >= 0 else "ffn",
                    row=q, col=p, bit=bit,
                    old_int8=old_i8, new_int8=new_i8,
                    delta_w=dw, predicted_dL=dpred,
                    layer_weight_R=R_l,
                    gradient_score=abs(dpred),
                    final_score=hybrid_score))

    ranked.sort(key=lambda d: -d["final_score"])

    n_attn = sum(1 for r in ranked if r["component"] == "attn")
    n_ffn = sum(1 for r in ranked if r["component"] == "ffn")
    print(f"  Candidates: {len(ranked)} ({n_attn} attn, {n_ffn} ffn)")
    if ranked:
        print(f"  Top score: {ranked[0]['final_score']:.8f} "
              f"(R={ranked[0]['layer_weight_R']:.3f}, "
              f"g={ranked[0]['gradient_score']:.8f})")
        print(f"\n  Top-10 candidates:")
        for r in ranked[:10]:
            tag = (f"L{r['layer']} H{r['head']}" if r["component"] == "attn"
                   else f"L{r['layer']} FFN")
            print(f"    {tag} {r['tensor']}[{r['row']},{r['col']}] "
                  f"bit{r['bit']}  R={r['layer_weight_R']:.3f}  "
                  f"score={r['final_score']:.8f}")

    cleanup_gpu()
    return ranked


# ══════════════════════════════════════════════════════════════════════
#  PHASE 4 (FALLBACK) — GRADIENT-ONLY RANKING (mini-batch)
# ══════════════════════════════════════════════════════════════════════
def phase4_gradient_ranking_fallback(model, trigger, t_target, selected_heads,
                                     grad_imgs, cfg):
    print("\n" + "=" * 60)
    print("PHASE 4 (FALLBACK): GRADIENT-ONLY RANKING")
    print("=" * 60)

    for p in model.parameters():
        p.requires_grad_(False)

    blocks = list(model.visual.transformer.resblocks)
    D = model.visual.transformer.width
    n_heads = blocks[0].attn.num_heads
    dk = D // n_heads
    target_vec = t_target.detach()
    apply_mode, apply_opacity = get_apply_params(cfg)
    mini_bs = cfg.MINI_BS
    n_imgs = grad_imgs.size(0)

    tracked_params = {}
    for s in selected_heads:
        li = s["layer"]
        blk = blocks[li]
        if s["type"] == "attn":
            hi = s["head"]
            lo, hie = hi * dk, (hi + 1) * dk
            w_v = blk.attn.in_proj_weight
            w_v.requires_grad_(True)
            tracked_params[(li, hi, "V", "attn")] = (w_v, 2 * D + lo, 2 * D + hie)
            w_o = blk.attn.out_proj.weight
            w_o.requires_grad_(True)
            tracked_params[(li, hi, "O", "attn")] = (w_o, 0, D, lo, hie)
        elif s["type"] == "ffn":
            w_fc = blk.mlp.c_fc.weight
            w_fc.requires_grad_(True)
            tracked_params[(li, -1, "fc", "ffn")] = (
                w_fc, 0, w_fc.shape[0], 0, w_fc.shape[1])
            w_proj = blk.mlp.c_proj.weight
            w_proj.requires_grad_(True)
            tracked_params[(li, -1, "proj", "ffn")] = (
                w_proj, 0, w_proj.shape[0], 0, w_proj.shape[1])

    # Gradient on TRIGGERED
    model.zero_grad()
    for i in range(0, n_imgs, mini_bs):
        chunk = grad_imgs[i:i + mini_bs]
        poisoned = apply_trigger(chunk, trigger, cfg.TRIG_SZ,
                                 apply_mode, apply_opacity)
        emb_trig = F.normalize(model.encode_image(poisoned), dim=-1)
        L_target = -F.cosine_similarity(
            emb_trig, target_vec.unsqueeze(0).expand(emb_trig.size(0), -1),
            dim=-1).mean()
        L_target.backward()
        del chunk, poisoned, emb_trig, L_target

    grad_poisoned = {}
    for key, (param, r_start, r_end, c_start, c_end) in tracked_params.items():
        grad_poisoned[key] = param.grad[r_start:r_end, c_start:c_end].clone()

    # Gradient on CLEAN
    for key, (param, _, _, _, _) in tracked_params.items():
        param.grad = None

    for i in range(0, n_imgs, mini_bs):
        chunk = grad_imgs[i:i + mini_bs]
        emb_clean = F.normalize(model.encode_image(chunk), dim=-1)
        L_clean = -F.cosine_similarity(
            emb_clean, target_vec.unsqueeze(0).expand(emb_clean.size(0), -1),
            dim=-1).mean()
        L_clean.backward()
        del chunk, emb_clean, L_clean

    grad_clean = {}
    for key, (param, r_start, r_end, c_start, c_end) in tracked_params.items():
        grad_clean[key] = param.grad[r_start:r_end, c_start:c_end].clone()

    grad_specific = {}
    for key in grad_poisoned:
        grad_specific[key] = grad_poisoned[key] - grad_clean[key]

    for key, (param, _, _, _, _) in tracked_params.items():
        param.requires_grad_(False)
        param.grad = None

    ranked = []
    for key, (param, r_start, r_end, c_start, c_end) in tracked_params.items():
        li, hi, tn, comp_type = key
        grad_slice = grad_specific[key]
        weight_slice = param.data[r_start:r_end, c_start:c_end].clone()
        int8_vals, scale = quantize_to_int8(weight_slice)
        flat_grad = grad_slice.flatten()
        flat_int8 = int8_vals.flatten()
        flat_sens = flat_grad.abs()

        k = min(cfg.SENS_TOP_M, flat_sens.numel())
        _, topk = flat_sens.topk(k)
        for idx in topk.tolist():
            q, p = divmod(idx, grad_slice.shape[1])
            g = flat_grad[idx].item()
            old_i8 = int(flat_int8[idx].item())
            for bit in range(8):
                new_i8 = flip_bit_int8(old_i8, bit)
                dw = (new_i8 - old_i8) * scale
                dpred = g * dw
                if dpred >= 0:
                    continue
                ranked.append(dict(
                    tensor=tn, layer=li, head=hi,
                    component=comp_type,
                    row=q, col=p, bit=bit,
                    old_int8=old_i8, new_int8=new_i8,
                    delta_w=dw, predicted_dL=dpred,
                    layer_weight_R=1.0,
                    gradient_score=abs(dpred),
                    final_score=abs(dpred)))

    ranked.sort(key=lambda d: -d["final_score"])
    n_attn = sum(1 for r in ranked if r["component"] == "attn")
    n_ffn = sum(1 for r in ranked if r["component"] == "ffn")
    print(f"  Candidates: {len(ranked)} ({n_attn} attn, {n_ffn} ffn)")
    if ranked:
        print(f"  Top score: {ranked[0]['final_score']:.8f}")

    cleanup_gpu()
    return ranked


# ══════════════════════════════════════════════════════════════════════
#  PHASE 5 — GREEDY BIT-FLIP SEARCH
# ══════════════════════════════════════════════════════════════════════
def _get_abs_pos(tn, layer, head, row, col, D, dk):
    if tn == "V":
        return 2 * D + head * dk + row, col
    elif tn == "O":
        return row, head * dk + col
    else:
        return row, col


def _local_idx(tn, head, abs_r, abs_c, D, dk):
    if tn == "V":
        return abs_r - (2 * D + head * dk), abs_c
    elif tn == "O":
        return abs_r, abs_c - head * dk
    else:
        return abs_r, abs_c


def _get_tensor(blk, tn):
    if tn == "V":
        return blk.attn.in_proj_weight
    elif tn == "O":
        return blk.attn.out_proj.weight
    elif tn == "fc":
        return blk.mlp.c_fc.weight
    elif tn == "proj":
        return blk.mlp.c_proj.weight


def _get_sub_quant_key(c):
    return (c["layer"], c["head"], c["tensor"])


def build_sub_quant(model, selected_heads, quant_dict):
    D = model.visual.transformer.width
    n_heads = list(model.visual.transformer.resblocks)[0].attn.num_heads
    dk = D // n_heads
    sq = {}
    for s in selected_heads:
        li = s["layer"]
        if s["type"] == "attn":
            hi = s["head"]
            lo, hie = hi * dk, (hi + 1) * dk
            key_v = (li, hi, "V")
            if key_v not in sq:
                ip_info = quant_dict[f"L{li}_attn_ip_w"]
                sq[key_v] = dict(
                    int8=ip_info["int8"][2 * D + lo:2 * D + hie].clone(),
                    scale=ip_info["scale"])
            key_o = (li, hi, "O")
            if key_o not in sq:
                op_info = quant_dict[f"L{li}_attn_op_w"]
                sq[key_o] = dict(
                    int8=op_info["int8"][:, lo:hie].clone(),
                    scale=op_info["scale"])
        elif s["type"] == "ffn":
            key_fc = (li, -1, "fc")
            if key_fc not in sq:
                fc_info = quant_dict[f"L{li}_fc1_w"]
                sq[key_fc] = dict(
                    int8=fc_info["int8"].clone(),
                    scale=fc_info["scale"])
            key_proj = (li, -1, "proj")
            if key_proj not in sq:
                proj_info = quant_dict[f"L{li}_fc2_w"]
                sq[key_proj] = dict(
                    int8=proj_info["int8"].clone(),
                    scale=proj_info["scale"])
    return sq


@torch.no_grad()
def phase5_greedy_search(model, text_emb, trigger, selected_heads,
                         ranked_flips, quant_dict,
                         eval_imgs, eval_labels,
                         grad_imgs, original_weights, cfg,
                         tensor_refs=None, layer_R=None):
    print("\n" + "=" * 60)
    print("PHASE 5: GREEDY BIT-FLIP SEARCH")
    print("=" * 60)

    blocks = list(model.visual.transformer.resblocks)
    D = model.visual.transformer.width
    n_heads = blocks[0].attn.num_heads
    dk = D // n_heads
    tc = cfg.TARGET_CLS
    ttype = cfg.TRIGGER_TYPE
    topac = cfg.OPACITY
    apply_mode, apply_opacity = get_apply_params(cfg)
    eval_bs = cfg.EVAL_BS

    drift_n = min(16, eval_imgs.shape[0])
    drift_imgs = eval_imgs[:drift_n]

    sub_quant = build_sub_quant(model, selected_heads, quant_dict)

    asr, cacc = quick_evaluate(
        model, text_emb, trigger, eval_imgs, eval_labels,
        tc, cfg.TRIG_SZ, ttype, topac, eval_bs)
    cacc_base = cacc
    asr_base = asr
    print(f"  Baseline  ASR={asr:.1f}%  CACC={cacc:.1f}%")

    clean_emb_ref = F.normalize(
        encode_image_batched(model, drift_imgs, eval_bs), dim=-1).clone()
    target_vec = text_emb[tc].detach()

    poisoned = apply_trigger(drift_imgs, trigger, cfg.TRIG_SZ,
                             apply_mode, apply_opacity)
    trig_emb_ref = F.normalize(
        encode_image_batched(model, poisoned, eval_bs), dim=-1).clone()
    trig_cos_ref = (trig_emb_ref @ target_vec).cpu()

    print(f"  Drift penalty λ={cfg.CLEAN_DRIFT_PENALTY}  "
          f"max_drift={cfg.MAX_CLEAN_DRIFT}")
    print(f"  drift_n={drift_n}  eval_bs={eval_bs}  "
          f"cands_per_round={cfg.CANDS_PER_ROUND}")

    flip_log = []
    checkpoints = [(0, asr, cacc)]
    ranked = list(ranked_flips)
    cand_ptr = 0
    no_improve = 0
    current_max_bit = cfg.INITIAL_MAX_BIT

    def rebuild_candidates():
        nonlocal ranked, cand_ptr
        if tensor_refs is not None and layer_R is not None:
            print("  Rebuilding hybrid rankings...")
            with torch.enable_grad():
                ranked = phase4_hybrid_ranking(
                    model, trigger, target_vec, layer_R,
                    selected_heads, tensor_refs, grad_imgs, cfg)
        else:
            with torch.enable_grad():
                ranked = phase4_gradient_ranking_fallback(
                    model, trigger, target_vec, selected_heads,
                    grad_imgs, cfg)
        cand_ptr = 0

    for rd in range(cfg.MAX_FLIPS):
        if cand_ptr >= len(ranked):
            print(f"  Rd {rd}: pool exhausted — refreshing")
            rebuild_candidates()
            if len(ranked) == 0:
                print("  No candidates — stopping.")
                break

        batch = ranked[cand_ptr:cand_ptr + cfg.CANDS_PER_ROUND]
        cand_ptr += cfg.CANDS_PER_ROUND

        best_score = -1e9
        best_info = None

        for c in batch:
            if c["bit"] > current_max_bit:
                continue
            blk = blocks[c["layer"]]
            ar, ac = _get_abs_pos(c["tensor"], c["layer"], c["head"],
                                  c["row"], c["col"], D, dk)
            tensor = _get_tensor(blk, c["tensor"])
            old_f = tensor.data[ar, ac].item()
            sq_key = _get_sub_quant_key(c)
            sq = sub_quant[sq_key]
            new_f = c["new_int8"] * sq["scale"]
            tensor.data[ar, ac] = new_f

            poisoned_new = apply_trigger(drift_imgs, trigger, cfg.TRIG_SZ,
                                         apply_mode, apply_opacity)
            trig_emb_new = F.normalize(
                encode_image_batched(model, poisoned_new, eval_bs), dim=-1)
            trig_cos_new = (trig_emb_new @ target_vec).cpu()

            clean_emb_new = F.normalize(
                encode_image_batched(model, drift_imgs, eval_bs), dim=-1)
            drift = (1.0 - F.cosine_similarity(
                clean_emb_ref, clean_emb_new, dim=-1)).mean().item()

            tensor.data[ar, ac] = old_f

            if drift > cfg.MAX_CLEAN_DRIFT:
                continue
            delta_cos = (trig_cos_new - trig_cos_ref).mean().item()
            n_improved = (trig_cos_new > trig_cos_ref).sum().item()
            score = delta_cos - cfg.CLEAN_DRIFT_PENALTY * drift
            if score > best_score:
                best_score = score
                best_info = dict(
                    tensor=c["tensor"], layer=c["layer"], head=c["head"],
                    component=c["component"],
                    abs_r=ar, abs_c=ac, bit=c["bit"],
                    old_int8=c["old_int8"], new_int8=c["new_int8"],
                    delta_cos=delta_cos, drift=drift,
                    n_improved=n_improved,
                    layer_weight_R=c.get("layer_weight_R", 1.0))

        if best_info is None or best_score <= 0:
            no_improve += 1
            print(f"  Round {rd}: no improving flip "
                  f"(streak {no_improve}/{cfg.PLATEAU_PATIENCE})")
            if no_improve >= cfg.PLATEAU_PATIENCE:
                if current_max_bit < 7:
                    print(f"  Expanding bit range to 0-7")
                    current_max_bit = 7
                    no_improve = 0
                    continue
                print("  Plateau reached.")
                break
            continue

        no_improve = 0
        bi = best_info
        blk = blocks[bi["layer"]]
        tensor = _get_tensor(blk, bi["tensor"])
        tensor.data[bi["abs_r"], bi["abs_c"]] = bi["new_int8"] * \
                                                sub_quant[_get_sub_quant_key(bi)]["scale"]
        sq = sub_quant[_get_sub_quant_key(bi)]
        lr, lc = _local_idx(bi["tensor"], bi["head"],
                            bi["abs_r"], bi["abs_c"], D, dk)
        sq["int8"][lr, lc] = bi["new_int8"]

        poisoned_upd = apply_trigger(drift_imgs, trigger, cfg.TRIG_SZ,
                                     apply_mode, apply_opacity)
        trig_emb_ref = F.normalize(
            encode_image_batched(model, poisoned_upd, eval_bs), dim=-1).clone()
        trig_cos_ref = (trig_emb_ref @ target_vec).cpu()
        clean_emb_ref = F.normalize(
            encode_image_batched(model, drift_imgs, eval_bs), dim=-1).clone()

        flip_log.append(dict(
            round=rd, tensor=bi["tensor"],
            layer=bi["layer"], head=bi["head"],
            component=bi["component"],
            position=(bi["abs_r"], bi["abs_c"]),
            bit=bi["bit"], old_int8=bi["old_int8"],
            new_int8=bi["new_int8"],
            delta_cos=bi["delta_cos"], drift=bi["drift"],
            n_improved=bi["n_improved"],
            score=best_score,
            layer_weight_R=bi.get("layer_weight_R", 1.0)))

        tag = (f"L{bi['layer']} H{bi['head']}" if bi["component"] == "attn"
               else f"L{bi['layer']} FFN")
        print(f"  Rd {rd:>2d}: {tag} {bi['tensor']}[{bi['abs_r']},{bi['abs_c']}] "
              f"bit {bi['bit']}  "
              f"Δcos={bi['delta_cos']:+.4f}  "
              f"drift={bi['drift']:.4f}  "
              f"score={best_score:.4f}")

        if len(flip_log) % cfg.CHECKPOINT_EVERY == 0:
            a_full, c_full = quick_evaluate(
                model, text_emb, trigger,
                eval_imgs, eval_labels, tc, cfg.TRIG_SZ, ttype, topac, eval_bs)
            checkpoints.append((len(flip_log), a_full, c_full))
            print(f"  [ckpt {len(flip_log)}] ASR={a_full:.1f}%  "
                  f"CACC={c_full:.1f}%")

        if len(flip_log) % 5 == 0:
            a_check, _ = quick_evaluate(
                model, text_emb, trigger,
                eval_imgs, eval_labels, tc, cfg.TRIG_SZ, ttype, topac, eval_bs)
            if a_check >= cfg.ASR_TARGET:
                print(f"  ASR target ({cfg.ASR_TARGET}%) reached!")
                break

        if len(flip_log) % cfg.REFRESH_EVERY == 0 and len(flip_log) > 0:
            rebuild_candidates()

    a_f, c_f = quick_evaluate(model, text_emb, trigger,
                              eval_imgs, eval_labels, tc,
                              cfg.TRIG_SZ, ttype, topac, eval_bs)
    checkpoints.append((len(flip_log), a_f, c_f))

    print(f"\n  Total flips : {len(flip_log)}")
    print(f"  Final ASR   : {a_f:.1f}%  (Δ = {a_f - asr_base:+.1f}%)")
    print(f"  Final CACC  : {c_f:.1f}%  (Δ = {c_f - cacc_base:+.1f}%)")

    if flip_log:
        bit_counts = Counter(f["bit"] for f in flip_log)
        print(f"  Bit distribution: {dict(sorted(bit_counts.items()))}")

    return flip_log, checkpoints


# ══════════════════════════════════════════════════════════════════════
#  PHASE 6 — POST-ATTACK ANALYSIS
# ══════════════════════════════════════════════════════════════════════
def phase6a_alignment(model, selected_heads, original_weights,
                      t_target, cfg):
    print("\n" + "=" * 60)
    print("PHASE 6a: ΔW SEMANTIC ALIGNMENT")
    print("=" * 60)
    vis = model.visual
    blocks = list(vis.transformer.resblocks)
    proj = vis.proj.data.float()
    n_heads = blocks[0].attn.num_heads
    D = vis.transformer.width
    dk = D // n_heads
    target = t_target.detach().float()
    alignments = []
    for s in selected_heads:
        li = s["layer"]
        blk = blocks[li]
        if s["type"] == "attn":
            hi = s["head"]
            lo, hie = hi * dk, (hi + 1) * dk
            W_in_now = blk.attn.in_proj_weight.data[2 * D + lo:2 * D + hie].float()
            W_out_now = blk.attn.out_proj.weight.data[:, lo:hie].float()
            orig_in = original_weights.get((li, hi, "V"))
            orig_out = original_weights.get((li, hi, "O"))
        elif s["type"] == "ffn":
            W_in_now = blk.mlp.c_fc.weight.data.float()
            W_out_now = blk.mlp.c_proj.weight.data.float()
            orig_in = original_weights.get((li, -1, "fc"))
            orig_out = original_weights.get((li, -1, "proj"))
        else:
            continue
        if orig_in is None or orig_out is None:
            continue
        dIn = W_in_now - orig_in.float()
        dOut = W_out_now - orig_out.float()
        if dIn.abs().max() < 1e-12 and dOut.abs().max() < 1e-12:
            continue
        dVO = dIn.T @ orig_out.float().T + orig_in.float().T @ dOut.T
        if dVO.abs().max() < 1e-12:
            continue
        U, S, Vh = torch.linalg.svd(dVO, full_matrices=False)
        v_delta = Vh[0]
        v_hat = F.normalize((v_delta @ proj).unsqueeze(0), dim=-1).squeeze()
        cos = F.cosine_similarity(v_hat, target, dim=0).item()
        tag = f"L{li} H{s['head']}" if s["type"] == "attn" else f"L{li} FFN"
        alignments.append(dict(layer=li, type=s["type"], cosine=cos))
        print(f"  {tag}: cos(ΔW, target) = {cos:+.4f}")
    if alignments:
        avg = np.mean([a["cosine"] for a in alignments])
        print(f"\n  Mean alignment: {avg:+.4f}")
    else:
        print("  No weight changes detected.")
    return alignments


@torch.no_grad()
def phase6b_per_class(model, text_emb, dataset, test_idx,
                      target_cls, device, class_names,
                      model_clean_state=None):
    print("\n" + "=" * 60)
    print("PHASE 6b: PER-CLASS CACC BREAKDOWN")
    print("=" * 60)

    class_indices = defaultdict(list)
    for i in test_idx:
        class_indices[dataset[i][1]].append(i)

    # ── Evaluate all classes with current (attacked) weights ──
    def _eval_all():
        results = {}
        for cls_idx in sorted(class_indices.keys()):
            idxs = class_indices[cls_idx]
            loader = DataLoader(Subset(dataset, idxs),
                                batch_size=4, shuffle=False)
            hit = tot = 0
            for imgs, labels in loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                emb = F.normalize(encode_image_batched(model, imgs), dim=-1)
                hit += ((emb @ text_emb.T).argmax(1) == labels).sum().item()
                tot += imgs.size(0)
            acc = hit / max(tot, 1) * 100
            name = class_names[cls_idx] if cls_idx < len(class_names) else "?"
            results[cls_idx] = dict(cls=cls_idx, name=name, acc=acc, n=tot)
        return results

    attacked_results = _eval_all()

    # ── Evaluate on clean (quantized, un-flipped) model if state provided ──
    clean_results = {}
    if model_clean_state is not None:
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(model_clean_state)
        print("  Evaluating clean-model baseline (quantized, no flips)...")
        clean_results = _eval_all()
        model.load_state_dict(saved)
        del saved
        cleanup_gpu()

    # ── Merge into rows sorted by attacked accuracy (worst first) ──
    rows = []
    for cls_idx in sorted(class_indices.keys()):
        row = dict(attacked_results[cls_idx])
        if cls_idx in clean_results:
            row["clean_acc"] = clean_results[cls_idx]["acc"]
            row["delta"] = row["acc"] - row["clean_acc"]
        else:
            row["clean_acc"] = None
            row["delta"] = None
        rows.append(row)

    rows.sort(key=lambda r: r["acc"])

    # ── Print comparison table ──
    if clean_results:
        print(f"\n  {'Class':<30s} {'Clean':>8s} {'Attacked':>9s} "
              f"{'Δ':>8s}  {'n':>5s}")
        print(f"  {'─' * 65}")
        for r in rows[:20]:
            c_str = f"{r['clean_acc']:.1f}%" if r["clean_acc"] is not None else "—"
            d_str = f"{r['delta']:+.1f}%" if r["delta"] is not None else "—"
            print(f"  {r['name']:<30s} {c_str:>8s} {r['acc']:>8.1f}% "
                  f"{d_str:>8s}  {r['n']:>5d}")

        # ── Highlight biggest degradations ──
        degraded = [r for r in rows if r["delta"] is not None and r["delta"] < -5.0]
        if degraded:
            degraded.sort(key=lambda r: r["delta"])
            print(f"\n  Classes with >5% degradation (attack vs clean):")
            for r in degraded[:10]:
                print(f"    {r['name']:<30s} {r['clean_acc']:.1f}% → "
                      f"{r['acc']:.1f}%  (Δ = {r['delta']:+.1f}%)")

        # ── Summary stats ──
        deltas = [r["delta"] for r in rows if r["delta"] is not None]
        if deltas:
            print(f"\n  Mean Δ across all classes: {np.mean(deltas):+.2f}%")
            n_worse = sum(1 for d in deltas if d < -2.0)
            n_better = sum(1 for d in deltas if d > 2.0)
            n_same = len(deltas) - n_worse - n_better
            print(f"  Classes: {n_worse} degraded >2%, "
                  f"{n_same} stable (±2%), "
                  f"{n_better} improved >2%")
    else:
        print(f"\n  {'Class':<30s} {'CACC':>7s}")
        print(f"  {'─' * 40}")
        for r in rows[:20]:
            print(f"  {r['name']:<30s} {r['acc']:>6.1f}%")

    return rows


@torch.no_grad()
def phase6d_target_class_cacc(model, text_emb, trigger,
                              dataset, target_holdout,
                              target_cls, cfg, model_clean_state=None):
    print("\n" + "=" * 60)
    print("PHASE 6d: TARGET CLASS CACC")
    print("=" * 60)

    if len(target_holdout) == 0:
        print("  No target holdout images available.")
        return {}

    apply_mode, apply_opacity = get_apply_params(cfg)
    loader = DataLoader(Subset(dataset, target_holdout), batch_size=4,
                        shuffle=False, num_workers=2)
    target_name = f"Class {target_cls}"
    results = {}

    if model_clean_state is not None:
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(model_clean_state)

        hit_clean = tot = 0
        for imgs, labels in loader:
            imgs = imgs.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)
            emb = F.normalize(encode_image_batched(model, imgs), dim=-1)
            hit_clean += ((emb @ text_emb.T).argmax(1) == labels).sum().item()
            tot += imgs.size(0)
        acc_clean = hit_clean / max(tot, 1) * 100
        print(f"\n  [Clean model — baseline]")
        print(f"    {target_name} clean CACC: {acc_clean:.1f}%  (n={tot})")
        results["clean_model"] = {"clean_cacc": acc_clean, "n": tot}
        model.load_state_dict(saved)

    hit_clean = hit_trig = tot = 0
    for imgs, labels in loader:
        imgs = imgs.to(cfg.DEVICE)
        labels = labels.to(cfg.DEVICE)
        emb_c = F.normalize(encode_image_batched(model, imgs), dim=-1)
        hit_clean += ((emb_c @ text_emb.T).argmax(1) == labels).sum().item()
        poisoned = apply_trigger(imgs, trigger, cfg.TRIG_SZ,
                                 apply_mode, apply_opacity)
        emb_t = F.normalize(encode_image_batched(model, poisoned), dim=-1)
        hit_trig += ((emb_t @ text_emb.T).argmax(1) == labels).sum().item()
        tot += imgs.size(0)

    acc_clean = hit_clean / max(tot, 1) * 100
    acc_trig = hit_trig / max(tot, 1) * 100

    print(f"\n  [Attacked model]")
    print(f"    {target_name} clean CACC:    {acc_clean:.1f}%  (n={tot})")
    print(f"    {target_name} triggered CACC: {acc_trig:.1f}%  (n={tot})")
    results["attacked"] = {"clean_cacc": acc_clean,
                           "triggered_cacc": acc_trig, "n": tot}

    if model_clean_state is not None and "clean_model" in results:
        delta = acc_clean - results["clean_model"]["clean_cacc"]
        print(f"\n    Δ clean CACC (attack - baseline): {delta:+.1f}%")
        if abs(delta) < 2.0:
            print(f"    → Stealthy: target class accuracy preserved")
        else:
            print(f"    → WARNING: target class accuracy degraded!")

    return results


def phase6c_curves(checkpoints, path):
    if len(checkpoints) < 2:
        print("  Not enough checkpoints for curve plot.")
        return
    flips = [c[0] for c in checkpoints]
    asr = [c[1] for c in checkpoints]
    cacc = [c[2] for c in checkpoints]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_xlabel("Bit flips")
    ax1.set_ylabel("ASR (%)", color="tab:red")
    ax1.plot(flips, asr, "o-", color="tab:red", label="ASR")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.set_ylabel("CACC (%)", color="tab:blue")
    ax2.plot(flips, cacc, "s--", color="tab:blue", label="CACC")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title("ASR / CACC vs Bit Flips — Hybrid Ranking")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved curves → {path}")


def build_target_embedding(model, text_emb, target_cls, target_images, device):
    """
    Build a robust target embedding from text prompts AND real target-class images.
    """
    # 1. Text component (multi-template)
    t_text = text_emb[target_cls:target_cls + 1]  # [1, dim]

    # 2. Image prototype: mean embedding of real target-class images
    with torch.no_grad():
        t_img = F.normalize(
            encode_image_batched(model, target_images, 4), dim=-1).mean(dim=0, keepdim=True)

    # 3. Combined: text defines the classification direction,
    #    image prototype anchors it to the actual cluster
    t_target = F.normalize(0.4 * t_text + 0.6 * t_img, dim=-1).squeeze(0)

    cos_text_img = F.cosine_similarity(t_text, t_img, dim=-1).item()
    print(f"  Target embedding: 40% text + 60% image prototype")
    print(f"  cos(text, image_proto) = {cos_text_img:.4f}")
    if cos_text_img < 0.5:
        print(f"  WARNING: text and image embeddings are far apart. "
              f"The trigger has a hard target to reach.")

    return t_target


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    set_seed(42)
    cfg = CFG()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    dev = cfg.DEVICE

    print("=" * 60)
    print("HYBRID BIT-FLIP ATTACK ON CLIP")
    print("Phase 0 → 1 → 2 → 3 → 4 → 5 → 6")
    print("=" * 60)
    print(f"  Device  : {dev}")
    print(f"  Target  : class {cfg.TARGET_CLS}")
    print(f"  Trigger : {cfg.TRIGGER_TYPE}  {cfg.TRIG_SZ}×{cfg.TRIG_SZ}")

    # ── Load model ──
    print("\n[1] Loading CLIP")
    model, preprocess = clip.load(cfg.MODEL, device=dev)
    model = model.float()

    # ── Auto-detect actual model dimensions ──
    mparams = detect_model_params(model)
    cfg.PATCH_SIZE = mparams["patch_size"]
    cfg.IMG_SIZE = mparams["img_size"]

    print(f"  Config requested : {cfg.MODEL}")
    print(f"  Actual architecture:")
    print(f"    seq_len  : {mparams['seq_len']}  "
          f"(= {mparams['patches_per_side']}×{mparams['patches_per_side']} patches + CLS)")
    print(f"    width    : {mparams['width']}")
    print(f"    patch    : {mparams['patch_size']}×{mparams['patch_size']}")
    print(f"    img_size : {mparams['img_size']}")
    print(f"    layers   : {mparams['n_layers']}")
    print(f"    heads    : {mparams['n_heads']}")
    print(f"    params   : {sum(p.numel() for p in model.visual.parameters()) / 1e6:.1f}M")

    # Sanity check: warn if config doesn't match reality
    if "L/14" in cfg.MODEL and mparams["width"] != 1024:
        print(f"\n  WARNING: Config says ViT-L/14 (width=1024) but "
              f"loaded model has width={mparams['width']}.")
        print(f"    The model is likely NOT ViT-L/14.  "
              f"Check your CLIP installation.\n")
    if "B/16" in cfg.MODEL and mparams["width"] != 768:
        print(f"\n  WARNING: Config says ViT-B/16 (width=768) but "
              f"loaded model has width={mparams['width']}.\n")

    # Warn about trigger-patch alignment
    trigger_per_dim = cfg.TRIG_SZ // mparams["patch_size"]
    if trigger_per_dim < 2:
        print(f"  NOTE: TRIG_SZ={cfg.TRIG_SZ} covers only "
              f"{trigger_per_dim} patch(es) per dim with "
              f"patch_size={mparams['patch_size']}.")
        print(f"    Consider increasing TRIG_SZ to at least "
              f"{2 * mparams['patch_size']} for better coverage.\n")

    # ── Load dataset ──
    print("[2] Loading Caltech-101")
    tf = transforms.Compose([lambda im: im.convert("RGB"), preprocess])
    ds = datasets.Caltech101(root=cfg.DATA, download=cfg.DOWNLOAD,
                             transform=tf, target_type="category")
    classes = list(ds.categories)
    print(f"  {len(classes)} classes, target = '{classes[cfg.TARGET_CLS]}'")

    # ── Text embeddings ──
    text_emb = build_text_embeddings(model, classes, dev)

    # ── Data splits ──
    print("\n[3] Preparing data splits")
    target_sel, calib_idx, grad_idx, eval_idx, test_idx, target_holdout = \
        prepare_data_splits(ds, cfg)

    target_images, _ = load_tensors(ds, target_sel, dev)
    calib_loader = make_loader(ds, calib_idx, cfg.BATCH, shuffle=True,
                               workers=cfg.WORKERS)
    calib_imgs, _ = load_tensors(ds, calib_idx, dev)
    eval_imgs, eval_labels = load_tensors(ds, eval_idx, dev)

    # ── grad_imgs on CPU, transferred to GPU when needed ──
    grad_imgs_cpu, _ = load_tensors(ds, grad_idx, "cpu",
                                    batch_size=cfg.BATCH, workers=cfg.WORKERS)

    # ── t_target ──
    with torch.no_grad():
        t_target = text_emb[cfg.TARGET_CLS].detach()

    print(f"  t_target = text_emb[{cfg.TARGET_CLS}]"
          f"  ('{classes[cfg.TARGET_CLS]}')")

    # Use model-aware trigger token indices
    trigger_token_indices = get_trigger_token_indices(cfg, model=model)
    print(f"  Trigger tokens: {trigger_token_indices}")
    print(f"  Trigger type: {cfg.TRIGGER_TYPE}  "
          f"Size: {cfg.TRIG_SZ}×{cfg.TRIG_SZ}  "
          f"Placement: {'center' if 'center' in cfg.TRIGGER_TYPE else 'corner'}")

    # ═══ PHASE 0 — Quantisation ═══
    quant_dict = phase0_quantize(model, text_emb, cfg)
    cleanup_gpu()

    # ═══ PHASE 1 — Trigger Generation ═══
    trigger = phase1_generate_trigger(
        model, text_emb, t_target, calib_loader, target_images, cfg)

    asr_pre, cacc_pre = quick_evaluate(
        model, text_emb, trigger, eval_imgs, eval_labels,
        cfg.TARGET_CLS, cfg.TRIG_SZ, cfg.TRIGGER_TYPE, cfg.OPACITY,
        cfg.MINI_BS)
    print(f"  Trigger standalone: ASR={asr_pre:.1f}%  CACC={cacc_pre:.1f}%")

    save_trigger_png(trigger,
                     os.path.join(cfg.OUT_DIR, "trigger_patch.png"),
                     bg_image=eval_imgs[0],
                     opacity=cfg.OPACITY,
                     trig_sz=cfg.TRIG_SZ,
                     mode=cfg.TRIGGER_TYPE)

    del target_images
    cleanup_gpu()

    # ═══ PHASE 2 — Routing Efficiency (find top-K layers) ═══
    layer_R, top_k_layers = phase2_routing_efficiency(
        model, trigger, t_target, calib_imgs, cfg)
    plot_layer_R(layer_R, cfg, top_k_layers=top_k_layers)
    cleanup_gpu()

    # ═══ PHASE 3 — Gradient Pre-Screening (scoped to top-K layers) ═══
    selected_heads = phase3_gradient_prescreen(
        model, trigger, t_target, calib_imgs, cfg,
        target_layers=top_k_layers)
    cleanup_gpu()

    # ═══ Weight tracking ═══
    tensor_refs, original_weights = setup_weight_tracking(
        model, selected_heads)
    print(f"\n  Weight tracking: {len(tensor_refs)} tensors, "
          f"{len(original_weights)} originals saved")

    # ═══ PHASE 4 — Hybrid Bit Ranking ═══
    grad_imgs = grad_imgs_cpu.to(dev)
    ranked_flips = phase4_hybrid_ranking(
        model, trigger, t_target, layer_R,
        selected_heads, tensor_refs, grad_imgs, cfg)

    # ── Save clean model state before bit flips ──
    model_state_clean = {k: v.cpu().clone()
                         for k, v in model.state_dict().items()}
    print(f"\n  Saved clean model state "
          f"({sum(v.numel() * v.element_size() for v in model_state_clean.values()) / 1e6:.0f} MB)")

    # ═══ PHASE 5 — Greedy Bit-Flip Search ═══
    t0 = time.time()
    flip_log, checkpoints = phase5_greedy_search(
        model, text_emb, trigger, selected_heads,
        ranked_flips, quant_dict,
        eval_imgs, eval_labels,
        grad_imgs, original_weights, cfg,
        tensor_refs=tensor_refs,
        layer_R=layer_R)
    search_time = time.time() - t0

    del quant_dict, layer_R, grad_imgs
    cleanup_gpu()

    # ═══ PHASE 6 — Post-Attack Analysis ═══
    print("\n" + "=" * 60)
    print("PHASE 6: POST-ATTACK ANALYSIS")
    print("=" * 60)

    alignments = phase6a_alignment(
        model, selected_heads, original_weights, t_target, cfg)

    phase6b_per_class(model, text_emb, ds, test_idx,
                      cfg.TARGET_CLS, dev, classes)

    phase6c_curves(checkpoints,
                   os.path.join(cfg.OUT_DIR, "asr_cacc_curves.png"))

    target_cacc_results = phase6d_target_class_cacc(
        model, text_emb, trigger,
        ds, target_holdout, cfg.TARGET_CLS, cfg,
        model_clean_state=model_state_clean)
    cleanup_gpu()

    # ── Create test_loader NOW (deferred from the beginning) ──
    test_imgs, test_labels = load_tensors(ds, test_idx, dev,
                                          batch_size=64, workers=cfg.WORKERS)

    # ── Final full evaluation ──
    test_loader = make_loader(ds, test_idx, 64, workers=cfg.WORKERS)
    r_final = full_evaluate(
        model, text_emb, trigger,
        test_loader,
        cfg.TARGET_CLS, cfg.TRIG_SZ, cfg.TRIGGER_TYPE, cfg.OPACITY, dev)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_flips = len(flip_log)
    print(f"  Model   : {cfg.MODEL}  (actual: {mparams['width']}d, "
          f"{mparams['n_layers']}L, {mparams['n_heads']}H, "
          f"patch={mparams['patch_size']})")
    print(f"  Method  : Hybrid (Layer_R → Scoped Gradient → R(l) × Gradient)")
    print(f"  Flips   : {n_flips}")
    print(f"  ASR     : {r_final['asr']:.1f}%")
    print(f"  CACC    : {r_final['cacc']:.1f}%")
    print(f"  Search  : {search_time:.1f}s")

    sample_for_vis = eval_imgs[:10]
    sample_labels_vis = eval_labels[:10]
    create_composite_png(
        trigger, cfg, r_final["asr"], r_final["cacc"],
        sample_for_vis, sample_labels_vis, classes,
        os.path.join(cfg.OUT_DIR, "composite.png"))

    save = {
        "config": {k: v for k, v in vars(cfg).items()
                   if not k.startswith("_")},
        "actual_model_params": mparams,
        "target_class": classes[cfg.TARGET_CLS],
        "trigger_token_indices": trigger_token_indices,
        "layer_R": {str(k): v for k, v in layer_R.items()} if layer_R else {},
        "top_k_layers": sorted(top_k_layers),
        "selected_heads": [
            {k: v for k, v in s.items()
             if not isinstance(v, torch.Tensor)}
            for s in selected_heads],
        "flip_log": flip_log,
        "checkpoints": checkpoints,
        "alignments": alignments,
        "target_cacc": target_cacc_results,

        "results": {
            "post_flip_test": {k: v for k, v in r_final.items()
                               if not isinstance(v, torch.Tensor)}},
        "search_time_sec": search_time}

    with open(os.path.join(cfg.OUT_DIR, "bfa_results.json"), "w") as f:
        json.dump(save, f, indent=2, default=str)
    torch.save(trigger.cpu(), os.path.join(cfg.OUT_DIR, "trigger.pt"))
    torch.save(original_weights, os.path.join(cfg.OUT_DIR,
                                              "original_weights.pt"))
    print(f"\n  Results saved to {cfg.OUT_DIR}/")


if __name__ == "__main__":
    main()