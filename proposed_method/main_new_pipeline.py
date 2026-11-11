"""
main_new_pipeline.py — Hybrid Bit-Flip Attack on CLIP ViT (auto-detect architecture)
====================================================================================
Pipeline order:
  Phase 0 (Quantise) → 1 (Trigger) → 2 (Layer Ablation)
  → 3 (Hybrid Ranking) → 4 (Greedy Search) → 5 (Post-Attack Analysis)
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
from scipy.stats import rankdata
from contextlib import contextmanager


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def detect_model_params(model):
    vis = model.visual
    pos = vis.positional_embedding
    seq_len = pos.shape[0]
    width = pos.shape[1]
    n_patches = seq_len - 1
    patch_size = vis.conv1.kernel_size[0]
    patches_per_side = int(n_patches ** 0.5)
    img_size = patches_per_side * patch_size
    blocks = list(vis.transformer.resblocks)
    n_layers = len(blocks)
    n_heads = blocks[0].attn.num_heads

    return dict(
        seq_len=seq_len, width=width, n_patches=n_patches,
        patch_size=patch_size, img_size=img_size,
        patches_per_side=patches_per_side, n_layers=n_layers, n_heads=n_heads,
    )


class CFG:
    MODEL = "ViT-B/16"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TARGET_CLS = 3
    DATA = "../data"
    DATASET_NAME = "Caltech101"
    DOWNLOAD = True
    BATCH = 8
    MINI_BS = 4
    WORKERS = 2

    N_TARGET_IMAGES = 32
    N_CALIB = 32
    N_GRAD = 16
    N_EVAL = 200
    QUANT_TOLERANCE = 2.0

    # ── Phase 1: Trigger (MODIFIED) ──
    TRIG_SZ = 14           # Modified: 14
    P1_EPOCHS = 200        # Modified: Increased to allow convergence with clamp
    P1_LR = 0.02           # Modified: 0.02
    TV_WEIGHT = 0.0
    TRIGGER_TYPE = "center_add"
    TEMPERATURE = 0.07
    EPSILON = 8.0 / 255.0
    OPACITY = 1.0
    LAMBDA_VIS = 0.0

    TOP_K_LAYERS = 4
    TOP_K_COMPONENTS = 30
    PER_LAYER_QUOTA = True

    PATCH_SIZE = 14
    IMG_SIZE = 224

    GRADIENT_BETA = 10.0
    SENS_TOP_M = 150

    MAX_FLIPS = 80
    EVAL_BS = 32
    CANDS_PER_ROUND = 30
    ASR_TARGET = 90.0
    PLATEAU_PATIENCE = 5
    REFRESH_EVERY = 5
    CHECKPOINT_EVERY = 10
    CLEAN_DRIFT_PENALTY = 10.0
    MAX_CLEAN_DRIFT = 0.02

    OUT_DIR = f"./xpruner_bfa_results_{MODEL}_{DATASET_NAME}_v4"


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
_CM = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
_CS = torch.tensor(CLIP_STD).view(1, 3, 1, 1)

CATEGORY_RENAMES = {
    "Faces": "human face", "Faces_easy": "close-up human face photograph",
    "Motorbikes": "motorbike", "Leopards": "leopard", "ketch": "ketch ship",
    "schooner": "schooner ship", "cup": "drinking cup", "lamp": "table lamp",
    "laptop": "laptop computer", "wild_cat": "wild cat", "flamingo": "flamingo bird",
    "lotus": "lotus flower", "camera": "camera device", "chair": "chair furniture",
    "lobster": "lobster animal", "crocodile": "crocodile reptile",
    "octopus": "octopus sea creature", "scissors": "pair of scissors",
    "stapler": "stapler tool", "umbrella": "umbrella rain gear", "airplanes": "airplane",
    "butterfly": "butterfly insect", "car_side": "side view of a car",
    "cellphone": "cell phone", "cougar_body": "cougar animal", "cougar_face": "cougar face",
    "crab": "crab animal", "crayfish": "crayfish animal", "crocodile_head": "crocodile head",
    "dollar_bill": "dollar bill", "dragonfly": "dragonfly insect",
    "electric_guitar": "electric guitar", "elephant": "elephant animal", "emu": "emu bird",
    "euphonium": "euphonium instrument", "ewer": "ewer pitcher", "ferry": "ferry boat",
    "garfield": "garfield cartoon cat", "gerenuk": "gerenuk antelope", "gramophone": "gramophone",
    "hawksbill": "hawksbill sea turtle", "headphone": "headphones", "hedgehog": "hedgehog animal",
    "inline_skate": "inline skate", "joshua_tree": "joshua tree", "kangaroo": "kangaroo animal",
    "mandolin": "mandolin instrument", "metronome": "metronome", "minaret": "minaret tower",
    "nautilus": "nautilus shell", "pagoda": "pagoda building", "panda": "panda bear",
    "pigeon": "pigeon bird", "pizza": "pizza food", "platypus": "platypus animal",
    "pyramid": "pyramid structure", "revolver": "revolver gun", "rhino": "rhinoceros animal",
    "rooster": "rooster chicken", "saxophone": "saxophone instrument",
    "snoopy": "snoopy cartoon dog", "soccer_ball": "soccer ball",
    "starfish": "starfish sea creature", "stop_sign": "stop sign",
    "strawberry": "strawberry fruit", "sunflower": "sunflower plant",
    "tick": "tick insect", "trilobite": "trilobite fossil", "watch": "wrist watch",
    "water_lilly": "water lily flower", "wheelchair": "wheelchair",
    "wild_grog_face": "wild grog face", "windsor_chair": "windsor chair",
    "wrench": "wrench tool", "yin_yang": "yin yang symbol",
}

PROMPT_TEMPLATES = [
    "a photo of a {}.", "a photo of the {}.", "a close-up photo of a {}.",
    "a bright photo of a {}.", "a dark photo of a {}.", "a cropped photo of a {}.",
    "a low resolution photo of a {}.", "a photo of a large {}.",
    "a photo of a small {}.", "a photo of many {}.", "a picture of a {}.",
    "a rendering of a {}.", "a photo of one {}.",
]


def build_text_embeddings(model, classes, device):
    n_classes = len(classes)
    n_templates = len(PROMPT_TEMPLATES)
    all_prompts = []
    for c in classes:
        name = CATEGORY_RENAMES.get(c, c.replace("_", " ").lower())
        for template in PROMPT_TEMPLATES:
            all_prompts.append(template.format(name))

    print(f"  Encoding {len(all_prompts)} prompts ({n_classes} classes × {n_templates} templates)...")
    all_embeddings = []
    with torch.no_grad():
        chunk_size = 256
        for i in range(0, len(all_prompts), chunk_size):
            chunk = all_prompts[i:i + chunk_size]
            tok = clip.tokenize(chunk).to(device)
            emb = model.encode_text(tok)
            all_embeddings.append(emb)
        all_emb = torch.cat(all_embeddings, dim=0)

    all_emb = all_emb.view(n_classes, n_templates, -1)
    text_emb = F.normalize(all_emb.mean(dim=1), dim=-1)
    print(f"  Text embeddings shape: {text_emb.shape}")
    return text_emb

@contextmanager
def _knockout_head(blk, sl):
    orig = blk.attn.out_proj.weight.data[:, sl].clone()
    blk.attn.out_proj.weight.data[:, sl] = 0.0
    try:
        yield
    finally:
        blk.attn.out_proj.weight.data[:, sl] = orig


@contextmanager
def _knockout_mlp(blk):
    orig = blk.mlp.c_proj.weight.data.clone()
    blk.mlp.c_proj.weight.data.zero_()
    try:
        yield
    finally:
        blk.mlp.c_proj.weight.data.copy_(orig)


def _make_target_cos(model, t_target, mini_bs):
    def target_cos(imgs):
        embs = []
        for i in range(0, imgs.size(0), mini_bs):
            embs.append(F.normalize(model.encode_image(imgs[i:i + mini_bs]), dim=-1))
        emb = torch.cat(embs, dim=0)
        return F.cosine_similarity(emb, t_target.detach().unsqueeze(0).expand(emb.size(0), -1), dim=-1).mean().item()
    return target_cos


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
#  TRIGGER APPLICATION (MODIFIED: Added clamp_output)
# ══════════════════════════════════════════════════════════════════════
def apply_trigger(imgs, trig, sz, mode="replace", opacity=1.0, clamp_output=True):
    out = imgs.clone()
    if "center" in mode:
        y0 = (imgs.shape[2] - sz) // 2
        x0 = (imgs.shape[3] - sz) // 2
    else:
        y0 = imgs.shape[2] - sz
        x0 = imgs.shape[3] - sz

    region = (slice(None), slice(None), slice(y0, y0 + sz), slice(x0, x0 + sz))

    if "blend" in mode:
        out[region] = opacity * trig + (1 - opacity) * out[region]
    elif "add" in mode:
        out[region] = out[region] + trig
    else:
        out[region] = trig

    # MODIFIED: Clamp output to valid image range if additive/blend
    if clamp_output and ("add" in mode or "blend" in mode):
        sd = _CS.to(imgs.device)
        mu = _CM.to(imgs.device)
        lo = (0.0 - mu) / sd
        hi = (1.0 - mu) / sd
        out = torch.maximum(torch.minimum(out, hi), lo)
    return out


def get_apply_params(cfg):
    return get_trigger_mode(cfg.TRIGGER_TYPE, cfg.OPACITY)


def pixel_space_clamp(trig, epsilon_pixel):
    with torch.no_grad():
        sd = _CS.to(trig.device)
        eps_normalized = epsilon_pixel / sd
        trig.data.clamp_(-eps_normalized, eps_normalized)


def total_variation(trig):
    dh = (trig[:, :, :, :-1] - trig[:, :, :, 1:]).pow(2).mean()
    dv = (trig[:, :, :-1, :] - trig[:, :, 1:, :]).pow(2).mean()
    return dh + dv


def get_trigger_mode(trigger_type, opacity=1.0):
    if trigger_type == "opacity":
        return "blend", opacity
    elif trigger_type == "center_add":
        return "add_center", 1.0
    elif trigger_type == "corner_add":
        return "add_corner", 1.0
    else:
        return "replace", 1.0


def get_trigger_token_indices(cfg, model=None):
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

    if cfg.TRIGGER_TYPE == "center_add":
        y0 = (img_size - sz) // 2
        x0 = (img_size - sz) // 2
    else:
        y0 = img_size - sz
        x0 = img_size - sz
    y1 = y0 + sz
    x1 = x0 + sz

    indices = []
    for r in range(patches_per_side):
        for c in range(patches_per_side):
            py0, py1 = r * patch_size, (r + 1) * patch_size
            px0, px1 = c * patch_size, (c + 1) * patch_size
            if y1 > py0 and y0 < py1 and x1 > px0 and x0 < px1:
                indices.append(r * patches_per_side + c + 1)
    return indices


@torch.no_grad()
def encode_image_batched(model, images, mini_bs=None):
    if mini_bs is None:
        mini_bs = images.size(0)
    parts = []
    for i in range(0, images.size(0), mini_bs):
        parts.append(model.encode_image(images[i:i + mini_bs]))
    return torch.cat(parts, dim=0)


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
    result = dict(cacc=hit_ca / n * 100, cos_clean=torch.cat(cos_cl).mean().item())
    if trigger is not None:
        result["asr"] = hit_asr / n * 100
        result["cos_poison"] = torch.cat(cos_po).mean().item()
    else:
        result["asr"] = 0.0
        result["cos_poison"] = 0.0
    return result


def load_tensors(dataset, indices, device, batch_size=64, workers=2):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=workers)
    imgs_list, lbl_list = [], []
    for imgs, labels in loader:
        imgs_list.append(imgs.to(device, non_blocking=True))
        lbl_list.append(labels.to(device, non_blocking=True))
    return torch.cat(imgs_list), torch.cat(lbl_list)


def make_loader(dataset, indices, batch_size, shuffle=False, workers=2):
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available())


def prepare_data_splits(ds, cfg):
    rng = np.random.RandomState(42)
    labels = ds.y
    target_indices = [i for i, lbl in enumerate(labels) if lbl == cfg.TARGET_CLS]
    other_indices  = [i for i, lbl in enumerate(labels) if lbl != cfg.TARGET_CLS]

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


def save_trigger_png(trigger, path, bg_image=None, opacity=1.0, trig_sz=16, mode="replace"):
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

            if "center" in mode:
                y0 = (bg_np.shape[0] - sz) // 2
                x0 = (bg_np.shape[1] - sz) // 2
            else:
                y0 = bg_np.shape[0] - sz
                x0 = bg_np.shape[1] - sz

            region = bg_np[y0:y0 + sz, x0:x0 + sz]
            if "add" in mode:
                region[:] = (region + t_np).clip(0, 1)
            elif "blend" in mode:
                region[:] = opacity * t_np + (1 - opacity) * region
            else:
                region[:] = t_np
            img_np = (bg_np * 255).astype(np.uint8)
        else:
            scale = max(1, 224 // sz)
            img_np = np.repeat(np.repeat((t_np * 255).astype(np.uint8), scale, axis=0), scale, axis=1)

        Image.fromarray(img_np).save(path)


def _denormalise(t):
    mu = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    sd = torch.tensor(CLIP_STD).view(3, 1, 1)
    px = (t.cpu().float() * sd + mu).clamp(0, 1)
    return (px.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def create_composite_png(trigger, cfg, asr, cacc, sample_imgs, sample_labels, class_names, path):
    sz = cfg.TRIG_SZ
    apply_mode, apply_opacity = get_apply_params(cfg)
    trigger_px = _denormalise(trigger[0])
    scale = max(1, 224 // sz)
    trigger_big = np.repeat(np.repeat(trigger_px, scale, axis=0), scale, axis=1)
    n_samples = min(10, len(sample_imgs))
    fig = plt.figure(figsize=(18, 6 + 2.5 * (n_samples // 5 + 1)))

    ax0 = fig.add_axes([0.02, 0.92, 0.96, 0.06])
    ax0.axis("off")
    placement = "center" if "center" in cfg.TRIGGER_TYPE else "corner"
    header = (f"CLIP {cfg.MODEL}  |  Target: '{class_names[cfg.TARGET_CLS]}'  |  "
              f"Trigger: {sz}×{sz} {placement}  type={cfg.TRIGGER_TYPE}  |  "
              f"ASR={asr:.1f}%  CACC={cacc:.1f}%")
    ax0.text(0.5, 0.5, header, ha="center", va="center", fontsize=10, family="monospace", transform=ax0.transAxes)

    ax_trig = fig.add_axes([0.02, 0.55, 0.12, 0.32])
    ax_trig.imshow(trigger_big, interpolation="nearest")
    ax_trig.set_title(f"Trigger {sz}×{sz} ({placement})", fontsize=9)
    ax_trig.axis("off")

    for i in range(n_samples):
        col = i % 5
        row = i // 5
        x = 0.18 + col * 0.165
        y = 0.55 - row * 0.35
        ax = fig.add_axes([x, y, 0.15, 0.32])
        img = sample_imgs[i].unsqueeze(0)
        poisoned = apply_trigger(img, trigger, sz, apply_mode, apply_opacity)
        px = _denormalise(poisoned[0])
        ax.imshow(px)
        lbl = sample_labels[i].item()
        name = class_names[lbl] if lbl < len(class_names) else str(lbl)
        ax.set_title(name, fontsize=8)
        ax.axis("off")

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_scores(layer_scores, cfg, top_k_layers=None):
    path = os.path.join(cfg.OUT_DIR, "layer_scores_analysis.png")
    layers = sorted(layer_scores.keys())
    vals = [layer_scores[l] for l in layers]
    fig, ax = plt.subplots(figsize=(10, 5))
    if top_k_layers is not None:
        colors = ['#e74c3c' if l in top_k_layers else '#3498db' for l in layers]
    else:
        colors = ['#e74c3c' if l in (1, len(layers) - 1) else '#3498db' for l in layers]
    bars = ax.bar(layers, vals, color=colors, alpha=0.8, edgecolor='white')
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Ablation score")
    title = f"Causal Mean-Ablation Score by Layer ({cfg.MODEL})\n"
    title += "score = rank(drop_attn) + rank(drop_mlp)   "
    title += "drop = cos_base − cos_ablated"
    if top_k_layers is not None:
        title += f"\nRed = selected top-{len(top_k_layers)} layers: {sorted(top_k_layers)}"
    ax.set_title(title)
    ax.set_xticks(layers)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=7, rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
            qd[name] = dict(int8=int8.clone(), scale=scale, block_idx=li, attr=attr, shape=list(param.shape))
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
#  PHASE 1 (MODIFIED: Smaller init, Static Trigger Clamp)
# ══════════════════════════════════════════════════════════════════════
def phase1_generate_trigger(model, text_emb, t_target, calib_loader, target_images, cfg):
    print("\n" + "=" * 60)
    print("PHASE 1: TRIGGER GENERATION (clamped)")
    print("=" * 60)
    print(f"  Size: {cfg.TRIG_SZ}×{cfg.TRIG_SZ}  Placement: center  Mode: additive")
    print(f"  Epochs: {cfg.P1_EPOCHS}  LR: {cfg.P1_LR}  T: {cfg.TEMPERATURE}")
    print(f"  Init: randn × 0.1  TV: {cfg.TV_WEIGHT}  Static Trig Clamp: [-3, 3]")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # MODIFIED: Init with 0.1 instead of 5.0
    trig = nn.Parameter(torch.randn(1, 3, cfg.TRIG_SZ, cfg.TRIG_SZ, device=cfg.DEVICE) * 0.1)

    opt = torch.optim.Adam([trig], lr=cfg.P1_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.P1_EPOCHS, eta_min=cfg.P1_LR * 0.01)

    apply_mode, apply_opacity = get_apply_params(cfg)
    best_loss, best_trig = float("inf"), trig.data.clone()

    for ep in range(1, cfg.P1_EPOCHS + 1):
        ep_loss = nb = 0
        for imgs, _ in calib_loader:
            imgs = imgs.to(cfg.DEVICE)
            # apply_trigger automatically clamps the image output
            poisoned = apply_trigger(imgs, trig, cfg.TRIG_SZ, apply_mode, apply_opacity)
            emb = F.normalize(model.encode_image(poisoned), dim=-1)

            logits = (emb @ text_emb.T) / cfg.TEMPERATURE
            labels = torch.full((emb.size(0),), cfg.TARGET_CLS, dtype=torch.long, device=cfg.DEVICE)
            loss = F.cross_entropy(logits, labels)

            if cfg.TV_WEIGHT > 0:
                loss = loss + cfg.TV_WEIGHT * total_variation(trig)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # MODIFIED: Static trigger clamp after step
            with torch.no_grad():
                trig.data.clamp_(-3.0, 3.0)

            ep_loss += loss.item()
            nb += 1

        sched.step()
        avg_l = ep_loss / max(nb, 1)
        if avg_l < best_loss:
            best_loss, best_trig = avg_l, trig.data.clone()
        if ep % 50 == 0 or ep == 1 or ep == cfg.P1_EPOCHS:
            print(f"  ep {ep:>4d}  loss={avg_l:.6f}  ||trig||={trig.data.norm():.3f}  RMS={trig.data.std():.3f}")

    print(f"  Best loss: {best_loss:.6f}")
    return best_trig.detach()


class _ConstantOutput(nn.Module):
    def __init__(self, const, seq_len=None):
        super().__init__()
        self.register_buffer("const", const)
        self.seq_len = seq_len

    def forward(self, x):
        c = self.const
        if x.dim() == 3 and c.dim() == 2:
            if x.shape[0] == c.shape[0]:
                return c.unsqueeze(1).expand(-1, x.shape[1], -1)
            if x.shape[1] == c.shape[0]:
                return c.unsqueeze(0).expand(x.shape[0], -1, -1)
        return c.expand_as(x)


@torch.no_grad()
def compute_sub_layer_means(model, calib_imgs, cfg, trigger=None):
    blocks = list(model.visual.transformer.resblocks)
    n_layers = len(blocks)
    seq_len = model.visual.positional_embedding.shape[0]

    if trigger is not None:
        apply_mode, apply_opacity = get_apply_params(cfg)
        input_imgs = apply_trigger(calib_imgs, trigger, cfg.TRIG_SZ, apply_mode, apply_opacity)
    else:
        input_imgs = calib_imgs

    sums = {li: {"attn_sum": None, "mlp_sum": None} for li in range(n_layers)}

    def make_hook(li, key):
        def hook_fn(module, inp, output):
            out = output[0] if isinstance(output, tuple) else output
            if out.shape[0] != seq_len and out.shape[1] == seq_len:
                out = out.permute(1, 0, 2)
            s = out.sum(dim=1)
            sums[li][key] = s if sums[li][key] is None else sums[li][key] + s
        return hook_fn

    hooks = []
    for li, blk in enumerate(blocks):
        hooks.append(blk.attn.register_forward_hook(make_hook(li, "attn_sum")))
        hooks.append(blk.mlp.register_forward_hook(make_hook(li, "mlp_sum")))

    n_seen = 0
    mini_bs = cfg.MINI_BS
    for i in range(0, input_imgs.size(0), mini_bs):
        chunk = input_imgs[i:i + mini_bs]
        _ = model.encode_image(chunk)
        n_seen += chunk.size(0)
    for h in hooks:
        h.remove()

    means = {}
    for li in range(n_layers):
        means[li] = {
            "attn_mean": sums[li]["attn_sum"] / n_seen,
            "mlp_mean": sums[li]["mlp_sum"] / n_seen,
        }
    return means

@contextmanager
def _replace_attention(blk, const_out):
    sentinel = object()
    prev = blk.__dict__.get("attention", sentinel)
    blk.attention = const_out.forward
    try:
        yield
    finally:
        if prev is sentinel:
            blk.__dict__.pop("attention", None)
        else:
            blk.__dict__["attention"] = prev


@contextmanager
def _replace_mlp(blk, const_out):
    orig = blk.mlp
    blk.mlp = const_out
    try:
        yield
    finally:
        blk.mlp = orig


@torch.no_grad()
def phase2_causal_ablation(model, trigger, t_target, calib_imgs, cfg):
    print("\n" + "=" * 60)
    print("PHASE 2: LAYER-LEVEL CAUSAL ABLATION")
    print("=" * 60)

    blocks = list(model.visual.transformer.resblocks)
    n_layers = len(blocks)
    D = model.visual.transformer.width
    seq_len = model.visual.positional_embedding.shape[0]
    apply_mode, apply_opacity = get_apply_params(cfg)
    mini_bs = cfg.MINI_BS
    target_cos = _make_target_cos(model, t_target, mini_bs)

    poisoned = apply_trigger(calib_imgs, trigger, cfg.TRIG_SZ, apply_mode, apply_opacity)
    cos_base = target_cos(poisoned)
    print(f"  cos_base (poisoned, intact) = {cos_base:.6f}")

    sub_means = compute_sub_layer_means(model, calib_imgs, cfg, trigger=trigger)

    attn_drops, mlp_drops = [], []
    for li in range(n_layers):
        blk = blocks[li]
        const_a = _ConstantOutput(sub_means[li]["attn_mean"], seq_len=seq_len)
        with _replace_attention(blk, const_a):
            cos_ab = target_cos(poisoned)
        a_drop = cos_base - cos_ab
        attn_drops.append(a_drop)

        const_m = _ConstantOutput(sub_means[li]["mlp_mean"], seq_len=seq_len)
        with _replace_mlp(blk, const_m):
            cos_ab = target_cos(poisoned)
        m_drop = cos_base - cos_ab
        mlp_drops.append(m_drop)

    attn_mag = [abs(d) for d in attn_drops]
    mlp_mag = [abs(d) for d in mlp_drops]
    attn_norm = rankdata(attn_mag) / len(attn_mag)
    mlp_norm = rankdata(mlp_mag) / len(mlp_mag)
    layer_scores = {li: float(attn_norm[li] + mlp_norm[li]) for li in range(n_layers)}

    sorted_layers = sorted(layer_scores.items(), key=lambda kv: -kv[1])
    k_top = min(cfg.TOP_K_LAYERS, n_layers)
    top_layers = set(li for li, _ in sorted_layers[:k_top])
    print(f"  Top-{k_top} layers: {sorted(top_layers)}")

    cleanup_gpu()
    return layer_scores, top_layers


@torch.no_grad()
def phase2b_component_ablation(model, trigger, t_target, calib_imgs, top_layers, cfg):
    print("\n" + "=" * 60)
    print("PHASE 2B: COMPONENT-LEVEL CAUSAL ABLATION")
    print("=" * 60)

    blocks = list(model.visual.transformer.resblocks)
    D = model.visual.transformer.width
    n_heads = blocks[0].attn.num_heads
    dk = D // n_heads
    apply_mode, apply_opacity = get_apply_params(cfg)
    mini_bs = cfg.MINI_BS

    target_cos = _make_target_cos(model, t_target, mini_bs)
    poisoned = apply_trigger(calib_imgs, trigger, cfg.TRIG_SZ, apply_mode, apply_opacity)
    c_base = target_cos(poisoned)

    component_scores = {}
    for li in sorted(top_layers):
        blk = blocks[li]
        for h in range(n_heads):
            lo, hie = h * dk, (h + 1) * dk
            with _knockout_head(blk, slice(lo, hie)):
                c_ab = target_cos(poisoned)
            drop = c_base - c_ab
            component_scores[(li, h)] = drop

        with _knockout_mlp(blk):
            c_ab = target_cos(poisoned)
        drop = c_base - c_ab
        component_scores[(li, -1)] = drop

    if cfg.PER_LAYER_QUOTA:
        k_per_layer = max(1, cfg.TOP_K_COMPONENTS // len(top_layers))
        selected_components = []
        for li in sorted(top_layers):
            layer_comps = [(key, score) for key, score in component_scores.items() if key[0] == li]
            layer_comps.sort(key=lambda kv: abs(kv[1]), reverse=True)
            for (layer, head), score in layer_comps[:k_per_layer]:
                selected_components.append(dict(type="attn" if head >= 0 else "ffn", layer=layer, head=head, score=score))
    else:
        ranked_comp = sorted(component_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
        m_top = min(cfg.TOP_K_COMPONENTS, len(ranked_comp))
        selected_components = []
        for (layer, head), score in ranked_comp[:m_top]:
            selected_components.append(dict(type="attn" if head >= 0 else "ffn", layer=layer, head=head, score=score))

    n_attn = sum(1 for s in selected_components if s["type"] == "attn")
    print(f"  Selected : {len(selected_components)} ({n_attn} attn, {len(selected_components) - n_attn} ffn)")

    cleanup_gpu()
    return component_scores, selected_components


def setup_weight_tracking(model, selected_components):
    blocks = list(model.visual.transformer.resblocks)
    D = model.visual.transformer.width
    n_heads = blocks[0].attn.num_heads
    dk = D // n_heads

    tensor_refs = {}
    original_weights = {}

    for s in selected_components:
        li = s["layer"]
        blk = blocks[li]
        if s["type"] == "attn":
            hi = s["head"]
            lo, hie = hi * dk, (hi + 1) * dk
            key_v = (li, hi, "V")
            if key_v not in tensor_refs:
                tensor_refs[key_v] = (blk.attn.in_proj_weight, slice(2 * D + lo, 2 * D + hie), slice(None))
                original_weights[key_v] = blk.attn.in_proj_weight.data[2 * D + lo:2 * D + hie].clone()
            key_o = (li, hi, "O")
            if key_o not in tensor_refs:
                tensor_refs[key_o] = (blk.attn.out_proj.weight, slice(None), slice(lo, hie))
                original_weights[key_o] = blk.attn.out_proj.weight.data[:, lo:hie].clone()
        elif s["type"] == "ffn":
            key_fc = (li, -1, "fc")
            if key_fc not in tensor_refs:
                tensor_refs[key_fc] = (blk.mlp.c_fc.weight, slice(None), slice(None))
                original_weights[key_fc] = blk.mlp.c_fc.weight.data.clone()
            key_proj = (li, -1, "proj")
            if key_proj not in tensor_refs:
                tensor_refs[key_proj] = (blk.mlp.c_proj.weight, slice(None), slice(None))
                original_weights[key_proj] = blk.mlp.c_proj.weight.data.clone()

    return tensor_refs, original_weights


def build_sub_quant(model, selected_components, quant_dict):
    D = model.visual.transformer.width
    n_heads = list(model.visual.transformer.resblocks)[0].attn.num_heads
    dk = D // n_heads
    sq = {}
    for s in selected_components:
        li = s["layer"]
        if s["type"] == "attn":
            hi = s["head"]
            lo, hie = hi * dk, (hi + 1) * dk
            key_v = (li, hi, "V")
            if key_v not in sq:
                ip_info = quant_dict[f"L{li}_attn_ip_w"]
                sq[key_v] = dict(int8=ip_info["int8"][2 * D + lo:2 * D + hie].clone(), scale=ip_info["scale"])
            key_o = (li, hi, "O")
            if key_o not in sq:
                op_info = quant_dict[f"L{li}_attn_op_w"]
                sq[key_o] = dict(int8=op_info["int8"][:, lo:hie].clone(), scale=op_info["scale"])
        elif s["type"] == "ffn":
            key_fc = (li, -1, "fc")
            if key_fc not in sq:
                fc_info = quant_dict[f"L{li}_fc1_w"]
                sq[key_fc] = dict(int8=fc_info["int8"].clone(), scale=fc_info["scale"])
            key_proj = (li, -1, "proj")
            if key_proj not in sq:
                proj_info = quant_dict[f"L{li}_fc2_w"]
                sq[key_proj] = dict(int8=proj_info["int8"].clone(), scale=proj_info["scale"])
    return sq


def phase3_hybrid_ranking(model, trigger, t_target, layer_R, selected_heads, tensor_refs, grad_imgs, cfg, sub_quant):
    print("\n" + "=" * 60)
    print("PHASE 3: HYBRID BIT RANKING")
    print("=" * 60)

    for p in model.parameters():
        p.requires_grad_(False)

    apply_mode, apply_opacity = get_apply_params(cfg)
    selected_tensors = set()
    for key, (tensor, _, _) in tensor_refs.items():
        tensor.requires_grad_(True)
        selected_tensors.add(tensor)

    mini_bs = cfg.MINI_BS
    n_imgs = grad_imgs.size(0)

    model.zero_grad()
    for i in range(0, n_imgs, mini_bs):
        chunk = grad_imgs[i:i + mini_bs]
        poisoned = apply_trigger(chunk, trigger, cfg.TRIG_SZ, apply_mode, apply_opacity)
        emb_po = F.normalize(model.encode_image(poisoned), dim=-1)
        loss_po = -F.cosine_similarity(emb_po, t_target.detach().unsqueeze(0).expand(emb_po.size(0), -1), dim=-1).mean()
        loss_po.backward()
        del chunk, poisoned, emb_po, loss_po

    grad_poisoned = {}
    for key, (tensor, rslice, cslice) in tensor_refs.items():
        grad_poisoned[key] = tensor.grad[rslice, cslice].clone()

    for tensor in selected_tensors:
        tensor.grad = None

    for i in range(0, n_imgs, mini_bs):
        chunk = grad_imgs[i:i + mini_bs]
        emb_cl = F.normalize(model.encode_image(chunk), dim=-1)
        loss_cl = -F.cosine_similarity(emb_cl, t_target.detach().unsqueeze(0).expand(emb_cl.size(0), -1), dim=-1).mean()
        loss_cl.backward()
        del chunk, emb_cl, loss_cl

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
        R_l = layer_R.get(li, 1.0)
        sq = sub_quant[(li, hi, tn)]
        int8_vals = sq["int8"]
        scale = sq["scale"]
        flat_grad = g_slice.flatten()
        flat_int8 = int8_vals.flatten()
        flat_sens = flat_grad.abs()

        k = min(cfg.SENS_TOP_M, flat_sens.numel())
        _, topk = flat_sens.topk(k)

        for idx in topk.tolist():
            q, p = divmod(idx, g_slice.shape[1])
            g = flat_grad[idx].item()
            if g >= 0:
                continue
            old_i8 = int(flat_int8[idx].item())
            for bit in range(8):
                new_i8 = flip_bit_int8(old_i8, bit)
                dw = (new_i8 - old_i8) * scale
                dpred = g * dw
                if dpred >= 0:
                    continue
                hybrid_score = R_l * abs(dpred)
                ranked.append(dict(tensor=tn, layer=li, head=hi, component="attn" if hi >= 0 else "ffn",
                                   row=q, col=p, bit=bit, old_int8=old_i8, new_int8=new_i8,
                                   delta_w=dw, predicted_dL=dpred, layer_weight_R=R_l,
                                   gradient_score=abs(dpred), final_score=hybrid_score))

    ranked.sort(key=lambda d: -d["final_score"])
    print(f"  Candidates: {len(ranked)}")
    cleanup_gpu()
    return ranked


def phase3_gradient_ranking_fallback(model, trigger, t_target, selected_heads, grad_imgs, cfg, sub_quant):
    print("\n" + "=" * 60)
    print("PHASE 3 (FALLBACK): GRADIENT-ONLY RANKING")
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
            tracked_params[(li, hi, "V", "attn")] = (w_v, 2 * D + lo, 2 * D + hie, 0, D)
            w_o = blk.attn.out_proj.weight
            w_o.requires_grad_(True)
            tracked_params[(li, hi, "O", "attn")] = (w_o, 0, D, lo, hie)
        elif s["type"] == "ffn":
            w_fc = blk.mlp.c_fc.weight
            w_fc.requires_grad_(True)
            tracked_params[(li, -1, "fc", "ffn")] = (w_fc, 0, w_fc.shape[0], 0, w_fc.shape[1])
            w_proj = blk.mlp.c_proj.weight
            w_proj.requires_grad_(True)
            tracked_params[(li, -1, "proj", "ffn")] = (w_proj, 0, w_proj.shape[0], 0, w_proj.shape[1])

    model.zero_grad()
    for i in range(0, n_imgs, mini_bs):
        chunk = grad_imgs[i:i + mini_bs]
        poisoned = apply_trigger(chunk, trigger, cfg.TRIG_SZ, apply_mode, apply_opacity)
        emb_trig = F.normalize(model.encode_image(poisoned), dim=-1)
        L_target = -F.cosine_similarity(emb_trig, target_vec.unsqueeze(0).expand(emb_trig.size(0), -1), dim=-1).mean()
        L_target.backward()
        del chunk, poisoned, emb_trig, L_target

    grad_poisoned = {}
    for key, (param, r_start, r_end, c_start, c_end) in tracked_params.items():
        grad_poisoned[key] = param.grad[r_start:r_end, c_start:c_end].clone()

    for key, (param, _, _, _, _) in tracked_params.items():
        param.grad = None

    for i in range(0, n_imgs, mini_bs):
        chunk = grad_imgs[i:i + mini_bs]
        emb_clean = F.normalize(model.encode_image(chunk), dim=-1)
        L_clean = -F.cosine_similarity(emb_clean, target_vec.unsqueeze(0).expand(emb_clean.size(0), -1), dim=-1).mean()
        L_clean.backward()
        del chunk, emb_clean, L_clean

    grad_clean = {}
    for key, (param, r_start, r_end, c_start, c_end) in tracked_params.items():
        grad_clean[key] = param.grad[r_start:r_end, c_start:c_end].clone()

    grad_specific = {key: grad_poisoned[key] - grad_clean[key] for key in grad_poisoned}

    for key, (param, _, _, _, _) in tracked_params.items():
        param.requires_grad_(False)
        param.grad = None

    ranked = []
    for key, (param, r_start, r_end, c_start, c_end) in tracked_params.items():
        li, hi, tn, comp_type = key
        grad_slice = grad_specific[key]
        sq = sub_quant[(li, hi, tn)]
        flat_grad = grad_slice.flatten()
        flat_int8 = sq["int8"].flatten()
        flat_sens = flat_grad.abs()

        k = min(cfg.SENS_TOP_M, flat_sens.numel())
        _, topk = flat_sens.topk(k)
        for idx in topk.tolist():
            q, p = divmod(idx, grad_slice.shape[1])
            g = flat_grad[idx].item()
            if g >= 0: continue
            old_i8 = int(flat_int8[idx].item())
            for bit in range(8):
                new_i8 = flip_bit_int8(old_i8, bit)
                dw = (new_i8 - old_i8) * sq["scale"]
                dpred = g * dw
                if dpred >= 0: continue
                ranked.append(dict(tensor=tn, layer=li, head=hi, component=comp_type, row=q, col=p, bit=bit,
                                   old_int8=old_i8, new_int8=new_i8, delta_w=dw, predicted_dL=dpred,
                                   layer_weight_R=1.0, gradient_score=abs(dpred), final_score=abs(dpred)))

    ranked.sort(key=lambda d: -d["final_score"])
    cleanup_gpu()
    return ranked


def _get_abs_pos(tn, layer, head, row, col, D, dk):
    if tn == "V": return 2 * D + head * dk + row, col
    elif tn == "O": return row, head * dk + col
    else: return row, col


def _local_idx(tn, head, abs_r, abs_c, D, dk):
    if tn == "V": return abs_r - (2 * D + head * dk), abs_c
    elif tn == "O": return abs_r, abs_c - head * dk
    else: return abs_r, abs_c


def _get_tensor(blk, tn):
    if tn == "V": return blk.attn.in_proj_weight
    elif tn == "O": return blk.attn.out_proj.weight
    elif tn == "fc": return blk.mlp.c_fc.weight
    elif tn == "proj": return blk.mlp.c_proj.weight


def _get_sub_quant_key(c):
    return (c["layer"], c["head"], c["tensor"])


@torch.no_grad()
def phase4_greedy_search(model, text_emb, trigger, selected_heads, ranked_flips, quant_dict,
                         eval_imgs, eval_labels, grad_imgs, original_weights, cfg,
                         tensor_refs=None, layer_R=None, sub_quant=None):
    print("\n" + "=" * 60)
    print("PHASE 4: GREEDY BIT-FLIP SEARCH")
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

    if sub_quant is None:
        _layers = set(s["layer"] for s in selected_heads)
        sub_quant = build_sub_quant(model, _layers, quant_dict)

    asr, cacc = quick_evaluate(model, text_emb, trigger, eval_imgs, eval_labels, tc, cfg.TRIG_SZ, ttype, topac, eval_bs)
    cacc_base = cacc
    asr_base = asr
    print(f"  Baseline  ASR={asr:.1f}%  CACC={cacc:.1f}%")

    clean_emb_ref = F.normalize(encode_image_batched(model, drift_imgs, eval_bs), dim=-1).clone()
    target_vec = text_emb[tc].detach()

    poisoned = apply_trigger(drift_imgs, trigger, cfg.TRIG_SZ, apply_mode, apply_opacity)
    trig_emb_ref = F.normalize(encode_image_batched(model, poisoned, eval_bs), dim=-1).clone()
    trig_cos_ref = (trig_emb_ref @ target_vec).cpu()

    flip_log = []
    checkpoints = [(0, asr, cacc)]
    ranked = list(ranked_flips)
    cand_ptr = 0
    no_improve = 0

    def rebuild_candidates():
        nonlocal ranked, cand_ptr
        if tensor_refs is not None and layer_R is not None:
            with torch.enable_grad():
                ranked = phase3_hybrid_ranking(model, trigger, target_vec, layer_R, selected_heads, tensor_refs, grad_imgs, cfg, sub_quant=sub_quant)
        else:
            with torch.enable_grad():
                ranked = phase3_gradient_ranking_fallback(model, trigger, target_vec, selected_heads, grad_imgs, cfg, sub_quant=sub_quant)
        cand_ptr = 0

    for rd in range(cfg.MAX_FLIPS):
        if cand_ptr >= len(ranked):
            rebuild_candidates()
            if len(ranked) == 0: break

        batch = ranked[cand_ptr:cand_ptr + cfg.CANDS_PER_ROUND]
        cand_ptr += cfg.CANDS_PER_ROUND
        best_score = -1e9
        best_info = None

        for c in batch:
            blk = blocks[c["layer"]]
            ar, ac = _get_abs_pos(c["tensor"], c["layer"], c["head"], c["row"], c["col"], D, dk)
            tensor = _get_tensor(blk, c["tensor"])
            old_f = tensor.data[ar, ac].item()
            sq_key = _get_sub_quant_key(c)
            sq = sub_quant[sq_key]
            new_f = c["new_int8"] * sq["scale"]
            tensor.data[ar, ac] = new_f

            poisoned_new = apply_trigger(drift_imgs, trigger, cfg.TRIG_SZ, apply_mode, apply_opacity)
            trig_emb_new = F.normalize(encode_image_batched(model, poisoned_new, eval_bs), dim=-1)
            trig_cos_new = (trig_emb_new @ target_vec).cpu()

            clean_emb_new = F.normalize(encode_image_batched(model, drift_imgs, eval_bs), dim=-1)
            drift = (1.0 - F.cosine_similarity(clean_emb_ref, clean_emb_new, dim=-1)).mean().item()

            tensor.data[ar, ac] = old_f

            if drift > cfg.MAX_CLEAN_DRIFT: continue
            delta_cos = (trig_cos_new - trig_cos_ref).mean().item()
            n_improved = (trig_cos_new > trig_cos_ref).sum().item()
            score = delta_cos - cfg.CLEAN_DRIFT_PENALTY * drift
            if score > best_score:
                best_score = score
                best_info = dict(tensor=c["tensor"], layer=c["layer"], head=c["head"], component=c["component"],
                                 abs_r=ar, abs_c=ac, bit=c["bit"], old_int8=c["old_int8"], new_int8=c["new_int8"],
                                 delta_cos=delta_cos, drift=drift, n_improved=n_improved, layer_weight_R=c.get("layer_weight_R", 1.0))

        if best_info is None or best_score <= 0:
            no_improve += 1
            if no_improve >= cfg.PLATEAU_PATIENCE: break
            continue

        no_improve = 0
        bi = best_info
        blk = blocks[bi["layer"]]
        tensor = _get_tensor(blk, bi["tensor"])
        tensor.data[bi["abs_r"], bi["abs_c"]] = bi["new_int8"] * sub_quant[_get_sub_quant_key(bi)]["scale"]
        sq = sub_quant[_get_sub_quant_key(bi)]
        lr, lc = _local_idx(bi["tensor"], bi["head"], bi["abs_r"], bi["abs_c"], D, dk)
        sq["int8"][lr, lc] = bi["new_int8"]

        poisoned_upd = apply_trigger(drift_imgs, trigger, cfg.TRIG_SZ, apply_mode, apply_opacity)
        trig_emb_ref = F.normalize(encode_image_batched(model, poisoned_upd, eval_bs), dim=-1).clone()
        trig_cos_ref = (trig_emb_ref @ target_vec).cpu()
        clean_emb_ref = F.normalize(encode_image_batched(model, drift_imgs, eval_bs), dim=-1).clone()

        flip_log.append(dict(round=rd, tensor=bi["tensor"], layer=bi["layer"], head=bi["head"], component=bi["component"],
                             position=(bi["abs_r"], bi["abs_c"]), bit=bi["bit"], old_int8=bi["old_int8"],
                             new_int8=bi["new_int8"], delta_cos=bi["delta_cos"], drift=bi["drift"],
                             n_improved=bi["n_improved"], score=best_score, layer_weight_R=bi.get("layer_weight_R", 1.0)))

        if len(flip_log) % 5 == 0:
            a_check, _ = quick_evaluate(model, text_emb, trigger, eval_imgs, eval_labels, tc, cfg.TRIG_SZ, ttype, topac, eval_bs)
            if a_check >= cfg.ASR_TARGET: break
            if len(flip_log) % cfg.REFRESH_EVERY == 0: rebuild_candidates()

    a_f, c_f = quick_evaluate(model, text_emb, trigger, eval_imgs, eval_labels, tc, cfg.TRIG_SZ, ttype, topac, eval_bs)
    checkpoints.append((len(flip_log), a_f, c_f))
    print(f"  Total flips : {len(flip_log)}")
    print(f"  Final ASR   : {a_f:.1f}%  (Δ = {a_f - asr_base:+.1f}%)")
    return flip_log, checkpoints


def main():
    set_seed(42)
    cfg = CFG()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    dev = cfg.DEVICE

    print("=" * 60)
    print("HYBRID BIT-FLIP ATTACK ON CLIP (FIXED TRIGGER)")
    print("=" * 60)

    print("\n[1] Loading CLIP")
    model, preprocess = clip.load(cfg.MODEL, device=dev)
    model = model.float()
    mparams = detect_model_params(model)
    cfg.PATCH_SIZE = mparams["patch_size"]
    cfg.IMG_SIZE = mparams["img_size"]

    print("[2] Loading Caltech-101")
    tf = transforms.Compose([lambda im: im.convert("RGB"), preprocess])
    ds = datasets.Caltech101(root=cfg.DATA, download=cfg.DOWNLOAD, transform=tf, target_type="category")
    classes = list(ds.categories)
    text_emb = build_text_embeddings(model, classes, dev)

    print("\n[3] Preparing data splits")
    target_sel, calib_idx, grad_idx, eval_idx, test_idx, target_holdout = prepare_data_splits(ds, cfg)

    target_images, _ = load_tensors(ds, target_sel, dev)
    calib_loader = make_loader(ds, calib_idx, cfg.BATCH, shuffle=True, workers=cfg.WORKERS)
    calib_imgs, _ = load_tensors(ds, calib_idx, dev)
    eval_imgs, eval_labels = load_tensors(ds, eval_idx, dev)
    grad_imgs_cpu, _ = load_tensors(ds, grad_idx, "cpu", batch_size=cfg.BATCH, workers=cfg.WORKERS)

    with torch.no_grad():
        t_target = text_emb[cfg.TARGET_CLS].detach()

    trigger_token_indices = get_trigger_token_indices(cfg, model=model)

    # ═══ PHASE 0 — Quantisation ═══
    quant_dict = phase0_quantize(model, text_emb, cfg)
    cleanup_gpu()

    # ═══ PHASE 1 — Trigger Generation (MODIFIED) ═══
    trigger = phase1_generate_trigger(model, text_emb, t_target, calib_loader, target_images, cfg)

    asr_pre, cacc_pre = quick_evaluate(model, text_emb, trigger, eval_imgs, eval_labels, cfg.TARGET_CLS, cfg.TRIG_SZ, cfg.TRIGGER_TYPE, cfg.OPACITY, cfg.MINI_BS)
    print(f"  Trigger standalone: ASR={asr_pre:.1f}%  CACC={cacc_pre:.1f}%")

    save_trigger_png(trigger, os.path.join(cfg.OUT_DIR, "trigger_patch.png"), bg_image=eval_imgs[0], opacity=cfg.OPACITY, trig_sz=cfg.TRIG_SZ, mode=cfg.TRIGGER_TYPE)
    del target_images
    cleanup_gpu()

    # ═══ PHASE 2A — Layer-Level Causal Ablation ═══
    layer_scores, top_k_layers = phase2_causal_ablation(model, trigger, t_target, calib_imgs, cfg)
    layer_R = layer_scores
    plot_ablation_scores(layer_scores, cfg, top_k_layers=top_k_layers)
    cleanup_gpu()

    # ═══ PHASE 2B — Component-Level Causal Ablation ═══
    component_scores, selected_components = phase2b_component_ablation(model, trigger, t_target, calib_imgs, top_k_layers, cfg)

    # ═══ Weight tracking (from selected components) ═══
    tensor_refs, original_weights = setup_weight_tracking(model, selected_components)
    sub_quant = build_sub_quant(model, selected_components, quant_dict)

    # ═══ PHASE 3 — Hybrid Bit Ranking ═══
    grad_imgs = grad_imgs_cpu.to(dev)
    ranked_flips = phase3_hybrid_ranking(model, trigger, t_target, layer_R, selected_components, tensor_refs, grad_imgs, cfg, sub_quant=sub_quant)

    model_state_clean = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ═══ PHASE 4 — Greedy Bit-Flip Search ═══
    t0 = time.time()
    flip_log, checkpoints = phase4_greedy_search(model, text_emb, trigger, selected_components, ranked_flips, quant_dict, eval_imgs, eval_labels, grad_imgs, original_weights, cfg, tensor_refs=tensor_refs, layer_R=layer_R, sub_quant=sub_quant)
    search_time = time.time() - t0

    del quant_dict, grad_imgs
    cleanup_gpu()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Flips   : {len(flip_log)}")

    # Final evaluation
    test_loader = make_loader(ds, test_idx, 64, workers=cfg.WORKERS)
    r_final = full_evaluate(model, text_emb, trigger, test_loader, cfg.TARGET_CLS, cfg.TRIG_SZ, cfg.TRIGGER_TYPE, cfg.OPACITY, dev)

    print(f"  ASR     : {r_final['asr']:.1f}%")
    print(f"  CACC    : {r_final['cacc']:.1f}%")

    create_composite_png(trigger, cfg, r_final["asr"], r_final["cacc"], eval_imgs[:10], eval_labels[:10], classes, os.path.join(cfg.OUT_DIR, "composite.png"))

    save = {
        "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
        "actual_model_params": mparams, "target_class": classes[cfg.TARGET_CLS],
        "trigger_token_indices": trigger_token_indices, "layer_R": {str(k): v for k, v in layer_R.items()} if layer_R else {},
        "top_k_layers": sorted(top_k_layers), "selected_heads": [{k: v for k, v in s.items() if not isinstance(v, torch.Tensor)} for s in selected_components],
        "flip_log": flip_log, "checkpoints": checkpoints, "results": {"post_flip_test": {k: v for k, v in r_final.items() if not isinstance(v, torch.Tensor)}},
        "search_time_sec": search_time}

    with open(os.path.join(cfg.OUT_DIR, "bfa_results.json"), "w") as f:
        json.dump(save, f, indent=2, default=str)
    torch.save(trigger.cpu(), os.path.join(cfg.OUT_DIR, "trigger.pt"))
    print(f"\n  Results saved to {cfg.OUT_DIR}/")


if __name__ == "__main__":
    main()