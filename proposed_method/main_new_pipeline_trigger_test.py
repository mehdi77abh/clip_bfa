"""
main_new_pipeline_trigger_test.py — two-stage trigger experiment (Caltech-101, ViT-B/16)
=======================================================================================
Copy of main_new_pipeline.py with the two-stage trigger pipeline enabled:

  Stage 1  cos-loss trigger, N_CALIB = 256, class-list-free
             ↓
  Phase 2  causal ablation on the Stage-1 trigger → top-k layers
             ↓  [S2-GATE] Jaccard overlap vs a CE-trigger Phase-2 reference
  Stage 2  cos + L_sens = -||grad_theta_s cos||_F^2 on the exact parameter
           slices Phase 3/4 flip, initialised from the Stage-1 trigger
             ↓
  Phase 3-4  bit ranking + greedy search (unchanged)

Stage-1 objective is still selected by CFG.P1_LOSS (default "cos" via
S1_LOSS; all modes "ce" / "cos" / "cos_t2t" / "ce_t2t" / "margin_ce" /
"composite" live in `phase1_objective`).

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
    DOWNLOAD = False
    BATCH = 8
    MINI_BS = 4
    WORKERS = 2

    N_TARGET_IMAGES = 32
    N_CALIB = 32
    N_GRAD = 16
    N_EVAL = 200
    QUANT_TOLERANCE = 2.0
    N_TEST        = 2000                    # NEW — cap the 50k val set

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

    # ── Phase 1 loss variant ──
    # "ce"        → cross-entropy on cosine logits (original / baseline)
    # "cos"       → -cos(emb, t_target) / T
    # "cos_t2t"   → "cos" + P1_LAMBDA_T2T · Σ Var_batch   (tight poisoned cluster)
    # "ce_t2t"    → "ce"  + P1_LAMBDA_T2T · Σ Var_batch
    # "margin_ce" → "ce" with an extra margin on the target logit
    # "composite" → clip_t + bounded margin + T2T + MPE + L2 + TV (proposal)
    P1_LOSS = "ce"
    P1_MARGIN = 0.05        # bounded margin over runner-up class (cosine units)
    P1_MPE_MARGIN = 0.40    # bounded alignment to target-image centroid
    W_COS_T = 1.0
    W_MARGIN = 0.5
    W_T2T = 0.05
    W_MPE = 0.10
    W_L2 = 0.01
    W_TV = 0.001
    TRIG_CLAMP = 3.0
    P1_LAMBDA_T2T = 0.05    # T2T weight for "cos_t2t" / "ce_t2t"
    P1_MARGIN_LOGIT = 1.0   # target-logit margin for "margin_ce" (logit units)

    # ── Stage 1 / Stage 2 (gradient-sensitivity re-optimisation) ──
    # Disabled here so main_new_pipeline.py keeps its "ce" baseline behaviour;
    # main_new_pipeline_trigger_test.py enables both.
    S1_ENABLED = True
    S1_LOSS = "cos"          # Stage-1 trigger objective (phase1_objective mode)
    S1_N_CALIB = 64          # Stage-1 calibration images
    S1_EPOCHS = 200

    S2_ENABLED = True
    S2_EPOCHS = 150
    S2_LR = 0.01
    S2_BATCH = 4             # double backward needs memory; batch 4 on 6 GB
    S2_N_CALIB = 64          # Stage-2 calibration images (disjoint from Stage 1)
    S2_LOSS_COS_WEIGHT = 1.0
    S2_LAMBDA_SENS = 0.1     # weight of  L_sens = -||grad_theta_s cos||_F^2
    S2_SENS_NORM = True      # divide L_sens by its value at Stage-2 init
    S2_GRAD_CLIP = 1.0       # clip_grad_norm_ on the trigger
    S2_STAGE1_DRIFT_LIMIT = 0.5   # ||trig - trig_s1|| <= limit * ||trig_s1||
    S2_SENS_TOP_K_LAYERS = None   # int -> restrict to top-k of Phase-2 layers
    S2_GATE = True
    S2_PHASE2_OVERLAP_THRESHOLD = 0.5
    S2_FORCE_PROCEED = False

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

    OUT_DIR = f"./xpruner_bfa_results_{MODEL}_{DATASET_NAME}_s1cos_s2sens"


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
    # Caltech-101 exposes labels as `ds.y`; ImageNet-style datasets use `.targets`.
    labels = getattr(ds, "targets", None)
    if labels is None:
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
    # Stage-2 calibration pool: a *disjoint* set of images (never the Stage-1 ones)
    n_s2 = int(cfg.S2_N_CALIB) if cfg.S2_ENABLED else 0
    s2_calib_idx = other_indices[ptr:ptr + n_s2]
    ptr += n_s2
    grad_idx = other_indices[ptr:ptr + cfg.N_GRAD]
    ptr += cfg.N_GRAD
    eval_idx = other_indices[ptr:ptr + cfg.N_EVAL]
    ptr += cfg.N_EVAL
    test_idx = other_indices[ptr:]
    if cfg.N_TEST is not None:
        test_idx = test_idx[:cfg.N_TEST]      # NEW — cap val split
    print(f"  Target class images : {len(target_sel):>5d}")
    print(f"  Target holdout      : {len(target_holdout):>5d}")
    print(f"  Calibration (S1)    : {len(calib_idx):>5d}")
    if s2_calib_idx:
        print(f"  Calibration (S2)    : {len(s2_calib_idx):>5d}  (disjoint from S1)")
    print(f"  Gradient            : {len(grad_idx):>5d}")
    print(f"  Evaluation          : {len(eval_idx):>5d}")
    print(f"  Test                : {len(test_idx):>5d}")
    return (target_sel, calib_idx, s2_calib_idx, grad_idx, eval_idx, test_idx,
            target_holdout)


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
def phase1_loss(emb, text_emb, t_target, v_target_mean, trig, cfg,
                return_parts=False):
    """Composite trigger objective (CLIP-FLIP v4 proposal).

    emb           : (B, D)  L2-normalised poisoned image embeddings
    text_emb      : (C, D)  L2-normalised class text embeddings
    t_target      : (D,)    target class text embedding
    v_target_mean : (1, D)  target-class image centroid, or None
    trig          : (1, 3, s, s) trigger parameter
    """
    # 1. Direct cosine to target text — matches Phase 3/4 exactly
    cos_t = F.cosine_similarity(
        emb, t_target.unsqueeze(0).expand_as(emb), dim=-1).mean()

    # 2. Bounded margin over other classes (argmax surrogate).
    #    Stop penalising once the target leads by `P1_MARGIN`, so CE does not
    #    saturate and the gradient stays informative.
    sims = emb @ text_emb.T                                    # (B, C)
    tgt = sims[:, cfg.TARGET_CLS]                              # (B,)
    other = sims.clone()
    other[:, cfg.TARGET_CLS] = -1e9
    max_other = other.max(dim=1).values                        # (B,)
    margin = tgt - max_other                                   # (B,)
    margin_loss = F.relu(cfg.P1_MARGIN - margin).mean()

    # 3. T2T compactness (BadCLIP++ idea): poisoned embeddings form a tight
    #    cluster so bit flips move them uniformly toward the target.
    #    Needs B > 1; unbiased=False keeps it defined for B == 1.
    if emb.size(0) > 1:
        loss_t2t = emb.var(dim=0, unbiased=False).sum()
    else:
        loss_t2t = emb.new_zeros(())

    # 4. MPE: align poisoned cluster to the target-image centroid
    #    (uses `target_images`, previously unused by Phase 1).
    if v_target_mean is not None:
        cos_v = F.cosine_similarity(
            emb, v_target_mean.expand_as(emb), dim=-1)
        loss_mpe = F.relu(cfg.P1_MPE_MARGIN - cos_v).mean()
    else:
        loss_mpe = emb.new_zeros(())

    # 5. Keep the trigger bounded and smooth — prevents saturation
    loss_l2 = trig.pow(2).mean()
    loss_tv = total_variation(trig)

    total = (cfg.W_COS_T * (-cos_t)      # main objective
             + cfg.W_MARGIN * margin_loss   # argmax support, bounded
             + cfg.W_T2T * loss_t2t         # compactness
             + cfg.W_MPE * loss_mpe         # target-image alignment
             + cfg.W_L2 * loss_l2           # keep trigger small
             + cfg.W_TV * loss_tv)          # smoothness

    if not return_parts:
        return total
    parts = dict(
        cos_t=cos_t.item(),
        margin=margin.mean().item(),
        margin_active=F.relu(cfg.P1_MARGIN - margin).gt(0).float().mean().item(),
        t2t=loss_t2t.item(),
        mpe=loss_mpe.item(),
        cos_v=(F.cosine_similarity(emb, v_target_mean.expand_as(emb), dim=-1).mean().item()
               if v_target_mean is not None else float("nan")),
        l2=loss_l2.item(),
        tv=loss_tv.item(),
    )
    return total, parts


def _t2t_compactness(emb):
    """Σ_d Var_batch(emb_d) — T2T compactness (BadCLIP++ idea).

    The gradient is ∝ the deviation from the batch mean, so the term
    self-anneals as the poisoned cluster tightens. Needs B > 1.
    """
    if emb.size(0) > 1:
        return emb.var(dim=0, unbiased=False).sum()
    return emb.new_zeros(())


def _margin_diag(emb, text_emb, target_cls):
    """(B,) target cosine minus best other-class cosine — logging only."""
    sims = emb @ text_emb.T
    other = sims.clone()
    other[:, target_cls] = -1e9
    return sims[:, target_cls] - other.max(dim=1).values


def phase1_objective(emb, text_emb, t_target, v_target_mean, trig, cfg,
                     return_parts=False):
    """Phase-1 trigger objective dispatcher (selected by cfg.P1_LOSS).

    "ce"        CE on temperature-scaled cosine logits (incumbent baseline)
    "cos"       -cos(emb, t_target) / T
    "cos_t2t"   -cos(emb, t_target) / T + P1_LAMBDA_T2T * sum_b Var(emb_b)
    "ce_t2t"    "ce" + P1_LAMBDA_T2T * sum_b Var(emb_b)
    "margin_ce" "ce" with an extra margin on the target logit
    "composite" phase1_loss(...) — the earlier proposal, kept for the record

    The cosine terms are divided by cfg.TEMPERATURE so their gradient
    magnitude matches the CE arms (logit scale); the "ce" arm is
    bit-identical to the original pipeline.
    """
    mode = cfg.P1_LOSS
    if mode == "composite":
        return phase1_loss(emb, text_emb, t_target, v_target_mean, trig, cfg,
                           return_parts=return_parts)
    if mode not in ("ce", "cos", "cos_t2t", "ce_t2t", "margin_ce"):
        raise ValueError(f"unknown P1_LOSS={mode!r}")

    B = emb.size(0)
    labels = torch.full((B,), cfg.TARGET_CLS, dtype=torch.long,
                        device=emb.device)
    logits = (emb @ text_emb.T) / cfg.TEMPERATURE

    if mode == "margin_ce":
        logits = logits.clone()
        logits[:, cfg.TARGET_CLS] += cfg.P1_MARGIN_LOGIT

    loss_ce = F.cross_entropy(logits, labels)
    cos_t = F.cosine_similarity(
        emb, t_target.unsqueeze(0).expand_as(emb), dim=-1).mean()

    if mode in ("cos", "cos_t2t"):
        loss = -cos_t / cfg.TEMPERATURE
    else:
        loss = loss_ce

    loss_t2t = emb.new_zeros(())
    if mode in ("cos_t2t", "ce_t2t"):
        loss_t2t = _t2t_compactness(emb)
        loss = loss + cfg.P1_LAMBDA_T2T * loss_t2t

    if not return_parts:
        return loss

    margin = _margin_diag(emb, text_emb, cfg.TARGET_CLS)
    parts = dict(
        cos_t=cos_t.item(),
        margin=margin.mean().item(),
        margin_active=F.relu(cfg.P1_MARGIN - margin).gt(0).float().mean().item(),
        t2t=loss_t2t.item(),
        ce=loss_ce.item(),
    )
    return loss, parts


def phase1_generate_trigger(model, text_emb, t_target, calib_loader, target_images, cfg):
    mode = cfg.P1_LOSS
    print("\n" + "=" * 60)
    print(f"PHASE 1: TRIGGER GENERATION (loss = {mode}, clamped)")
    print("=" * 60)
    print(f"  Size: {cfg.TRIG_SZ}×{cfg.TRIG_SZ}  Placement: center  Mode: additive")
    print(f"  Epochs: {cfg.P1_EPOCHS}  LR: {cfg.P1_LR}  T: {cfg.TEMPERATURE}")
    print(f"  Init: randn × 0.1  TV: {cfg.TV_WEIGHT}  Static Trig Clamp: ±{cfg.TRIG_CLAMP}")
    if mode == "composite":
        print(f"  Weights: cos_t={cfg.W_COS_T}  margin={cfg.W_MARGIN} "
              f"(>={cfg.P1_MARGIN})  t2t={cfg.W_T2T}  "
              f"mpe={cfg.W_MPE} (cos>={cfg.P1_MPE_MARGIN})  "
              f"l2={cfg.W_L2}  tv={cfg.W_TV}")
    elif mode in ("cos", "cos_t2t"):
        print(f"  Weights: -cos_t/T (T={cfg.TEMPERATURE})"
              + (f"  + {cfg.P1_LAMBDA_T2T} · ΣVar" if mode == "cos_t2t" else ""))
    elif mode == "ce_t2t":
        print(f"  Weights: CE + {cfg.P1_LAMBDA_T2T} · ΣVar")
    elif mode == "margin_ce":
        print(f"  Weights: CE with target-logit margin +{cfg.P1_MARGIN_LOGIT}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # MODIFIED: Init with 0.1 instead of 5.0
    trig = nn.Parameter(torch.randn(1, 3, cfg.TRIG_SZ, cfg.TRIG_SZ, device=cfg.DEVICE) * 0.1)

    # MPE anchor: target-class image centroid, computed once and frozen.
    v_target_mean = None
    if mode == "composite" and cfg.W_MPE > 0 and target_images is not None:
        with torch.no_grad():
            v = F.normalize(
                encode_image_batched(model, target_images, cfg.MINI_BS), dim=-1
            ).mean(dim=0, keepdim=True)
            v_target_mean = F.normalize(v, dim=-1)
            ct = F.cosine_similarity(t_target.unsqueeze(0), v_target_mean, dim=-1).item()
        print(f"  MPE anchor: {target_images.size(0)} target images  "
              f"cos(text, img_centroid)={ct:+.4f}")

    opt = torch.optim.Adam([trig], lr=cfg.P1_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.P1_EPOCHS, eta_min=cfg.P1_LR * 0.01)

    apply_mode, apply_opacity = get_apply_params(cfg)
    best_loss, best_trig = float("inf"), trig.data.clone()

    for ep in range(1, cfg.P1_EPOCHS + 1):
        ep_loss = nb = 0
        acc = defaultdict(float)
        for imgs, _ in calib_loader:
            imgs = imgs.to(cfg.DEVICE)
            # apply_trigger automatically clamps the image output
            poisoned = apply_trigger(imgs, trig, cfg.TRIG_SZ, apply_mode, apply_opacity)
            emb = F.normalize(model.encode_image(poisoned), dim=-1)

            loss, parts = phase1_objective(
                emb, text_emb, t_target, v_target_mean, trig, cfg,
                return_parts=True)
            for k, val in parts.items():
                acc[k] += val

            if cfg.TV_WEIGHT > 0:
                loss = loss + cfg.TV_WEIGHT * total_variation(trig)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # MODIFIED: Static trigger clamp after step
            with torch.no_grad():
                trig.data.clamp_(-cfg.TRIG_CLAMP, cfg.TRIG_CLAMP)

            ep_loss += loss.item()
            nb += 1

        sched.step()
        avg_l = ep_loss / max(nb, 1)
        if avg_l < best_loss:
            best_loss, best_trig = avg_l, trig.data.clone()
        log_now = (ep == 1 or ep % 25 == 0 or ep == cfg.P1_EPOCHS)
        if log_now:
            n = max(nb, 1)
            msg = (f"  ep {ep:>4d}  loss={avg_l:.6f}"
                   f"  cos_t={acc['cos_t'] / n:+.4f}"
                   f"  margin={acc['margin'] / n:+.4f}"
                   f"  t2t={acc['t2t'] / n:.5f}")
            if mode == "composite":
                msg += (f"  cos_v={acc['cos_v'] / n:+.4f}"
                        f"  mpe={acc['mpe'] / n:.5f}")
            else:
                msg += f"  ce={acc['ce'] / n:.4f}"
            msg += f"  ||trig||={trig.data.norm():.3f}"
            print(msg)

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


def check_phase2_overlap(top_k_layers_stage1, top_k_layers_ce, threshold,
                         force=False, n_calib=None):
    """[S2-GATE] Is the Stage-1 trigger good enough for reliable layer selection?

    Compares the Phase-2 top-k layers obtained with the Stage-1 trigger against
    the CE-trigger reference via Jaccard overlap.  Low overlap means the weak
    Stage-1 trigger is steering Phase 2 somewhere else entirely, so Stage 2
    would refine the wrong layers.
    """
    a, b = set(top_k_layers_stage1), set(top_k_layers_ce)
    jac = len(a & b) / max(len(a | b), 1)
    print(f"  [S2-GATE] Stage-1 top-k : {sorted(a)}")
    print(f"  [S2-GATE] CE      top-k : {sorted(b)}")
    print(f"  [S2-GATE] Jaccard overlap = {jac * 100:.1f}%  "
          f"(threshold {threshold * 100:.0f}%)")
    if jac < threshold and not force:
        hint = f"{n_calib} -> {2 * int(n_calib)}" if n_calib else "256 -> 512"
        print(f"  [S2-GATE] Phase-2 top-k overlap = {jac * 100:.1f}% < "
              f"{threshold * 100:.0f}%. Stage-1 trigger is NOT sufficient for "
              f"reliable layer selection.")
        print(f"  [S2-GATE] Recommended: increase S1_N_CALIB ({hint}) and/or "
              f"S1_EPOCHS, then re-run. Refusing to proceed to Stage 2.")
        raise SystemExit(1)
    return jac


def phase2_ce_reference(model, text_emb, t_target, calib_loader, calib_imgs, cfg):
    """Top-k layers from Phase 2A on the *original CE* trigger (cached on disk).

    Runs Phase 1 with P1_LOSS="ce" (same N_CALIB / epochs / trigger geometry as
    Stage 1, so the only difference is the loss) and Phase 2A on that trigger.
    The result is cached in cfg.OUT_DIR/phase2_ce_reference.json and regenerated
    if the cache was written with a different configuration.
    """
    path = os.path.join(cfg.OUT_DIR, "phase2_ce_reference.json")
    key = dict(n_calib=int(cfg.N_CALIB), epochs=int(cfg.S1_EPOCHS),
               trig_sz=int(cfg.TRIG_SZ), model=cfg.MODEL,
               dataset=cfg.DATASET_NAME, target_cls=int(cfg.TARGET_CLS))
    if os.path.exists(path):
        try:
            with open(path) as f:
                ref = json.load(f)
        except Exception:
            ref = None
        if ref and ref.get("key") == key:
            print(f"  [S2-GATE] CE reference top-k loaded from {path}")
            return set(int(x) for x in ref["top_k_layers"])
        print(f"  [S2-GATE] CE reference cache mismatches current config — regenerating")

    print("  [S2-GATE] Computing CE reference trigger + Phase 2A (one-off)")
    ref_cfg = copy.copy(cfg)
    ref_cfg.P1_LOSS = "ce"
    ref_cfg.P1_EPOCHS = cfg.S1_EPOCHS
    ref_trigger = phase1_generate_trigger(
        model, text_emb, t_target, calib_loader, None, ref_cfg)
    scores, layers = phase2_causal_ablation(
        model, ref_trigger, t_target, calib_imgs, ref_cfg)
    top_k = sorted(int(x) for x in layers)
    with open(path, "w") as f:
        json.dump(dict(key=key, top_k_layers=top_k,
                       layer_scores={str(k): float(v) for k, v in scores.items()},
                       trigger_loss="ce"), f, indent=2)
    print(f"  [S2-GATE] CE reference cached → {path}")
    del ref_trigger
    cleanup_gpu()
    return set(top_k)


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


def _collect_selected_params(model, selected_components, top_k_layers=None, k=None):
    """Exactly the parameter slices Phase 3/4 will flip (θ_s in L_sens).

    Walks `selected_components` (Phase 2B output) through `setup_weight_tracking`,
    so the set is identical to what `phase4_greedy_search` / `_get_tensor` touch.
    Optional `k` further restricts to the top-k of the Phase-2 layer ranking.
    """
    tensor_refs, _ = setup_weight_tracking(model, selected_components)
    allowed = None
    if k is not None and top_k_layers is not None:
        allowed = set(sorted(top_k_layers)[:k])
    entries = []
    for key, (tensor, rows, cols) in tensor_refs.items():
        layer, head, tn = key
        if allowed is not None and layer not in allowed:
            continue
        entries.append(dict(key=key, tensor=tensor, rows=rows, cols=cols))
    return entries


def _sens_from_grads(entries, grads, tensor_ids):
    """||∇_θs cos||_F² summed over the selected slices (0 if none contributed)."""
    by_id = {tid: g for tid, g in zip(tensor_ids, grads)}
    total = None
    for e in entries:
        g = by_id.get(id(e["tensor"]))
        if g is None:
            continue
        block = g[e["rows"], e["cols"]]
        term = block.pow(2).sum()
        total = term if total is None else total + term
    return total


@contextmanager
def _math_sdpa():
    """Force the math SDPA backend for double backward.

    CLIP's `nn.MultiheadAttention` dispatches to fused SDPA kernels
    (flash / mem-efficient), whose backward has no second-order derivative:
    "derivative for aten::_scaled_dot_product_efficient_attention_backward is
    not implemented".  The math backend decomposes into matmul + softmax and
    supports double backward.  Restored on exit.
    """
    if not torch.cuda.is_available():
        yield
        return
    f = torch.backends.cuda.flash_sdp_enabled()
    e = torch.backends.cuda.mem_efficient_sdp_enabled()
    m = torch.backends.cuda.math_sdp_enabled()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    try:
        yield
    finally:
        torch.backends.cuda.enable_flash_sdp(f)
        torch.backends.cuda.enable_mem_efficient_sdp(e)
        torch.backends.cuda.enable_math_sdp(m)


def stage2_refine_trigger(model, text_emb, t_target, trigger_s1, selected_components,
                          top_k_layers, calib_loader, cfg):
    """Stage 2 entry point — runs the double-backward loop under math SDPA."""
    with _math_sdpa():
        return _stage2_refine_trigger_impl(
            model, text_emb, t_target, trigger_s1, selected_components,
            top_k_layers, calib_loader, cfg)


def _stage2_refine_trigger_impl(model, text_emb, t_target, trigger_s1, selected_components,
                          top_k_layers, calib_loader, cfg):
    """Stage 2: re-optimise the Stage-1 trigger with the sensitivity term.

        L = S2_LOSS_COS_WEIGHT * (-cos(emb, t_target) / T)
            + S2_LAMBDA_SENS * (- ||∇_θs cos(emb, t_target)||_F² / sens0)

    θ_s are the exact parameter slices Phase 3/4 flip (selected_components),
    so the trigger is reshaped to make the target cosine maximally sensitive to
    bit flips in those layers.  Overshoot controls: init from the Stage-1
    trigger, grad clipping, ±TRIG_CLAMP, and a relative drift limit toward
    Stage-1 (all three always on).
    """
    print("\n" + "=" * 60)
    print("STAGE 2: GRADIENT-SENSITIVITY TRIGGER REFINEMENT")
    print("=" * 60)

    entries = _collect_selected_params(
        model, selected_components, top_k_layers, cfg.S2_SENS_TOP_K_LAYERS)
    assert len(entries) > 0, "Stage-2 has no target params — Phase 2B failed"

    tensors, seen = [], set()
    for e in entries:
        if id(e["tensor"]) not in seen:
            seen.add(id(e["tensor"]))
            tensors.append(e["tensor"])
    tensor_ids = [id(t) for t in tensors]
    n_el = sum(int(torch.numel(e["tensor"][e["rows"], e["cols"]])) for e in entries)

    layers = sorted({e["key"][0] for e in entries})
    print(f"  θ_s : {len(entries)} slices / {len(tensors)} tensors / "
          f"{n_el} elements  (layers {layers})")
    print(f"  epochs={cfg.S2_EPOCHS}  lr={cfg.S2_LR}  batch={cfg.S2_BATCH}  "
          f"λ_sens={cfg.S2_LAMBDA_SENS}  cos_w={cfg.S2_LOSS_COS_WEIGHT}")
    print(f"  clip={cfg.S2_GRAD_CLIP}  clamp=±{cfg.TRIG_CLAMP}  "
          f"drift≤{cfg.S2_STAGE1_DRIFT_LIMIT}·‖trig_s1‖  "
          f"sens_norm={cfg.S2_SENS_NORM}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for t in tensors:
        t.requires_grad_(True)

    trigger_s1 = trigger_s1.detach().to(cfg.DEVICE)
    trig = nn.Parameter(trigger_s1.clone())          # never randn
    opt = torch.optim.Adam([trig], lr=cfg.S2_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.S2_EPOCHS, 1), eta_min=cfg.S2_LR * 0.01)

    # Stage-2 calibration loader (disjoint image set, built in main())
    assert calib_loader is not None, "Stage 2 needs its own calibration loader"
    subset = calib_loader.dataset
    if cfg.S2_N_CALIB and cfg.S2_N_CALIB < len(subset.indices):
        subset = Subset(subset.dataset, list(subset.indices)[:cfg.S2_N_CALIB])
    loader = DataLoader(subset, batch_size=cfg.S2_BATCH, shuffle=True,
                        num_workers=cfg.WORKERS,
                        pin_memory=torch.cuda.is_available())
    print(f"  Stage-2 calibration images: {len(subset)} (disjoint from Stage 1)")

    apply_mode, apply_opacity = get_apply_params(cfg)
    s1_norm = trigger_s1.norm().item()
    best_loss, best_trig = float("inf"), trig.data.clone()
    sens0 = None
    info = dict(sens0=None, sens_last=None, drift_ratio_start=0.0,
                drift_ratio_end=None, cos_start=None, cos_end=None)

    for ep in range(1, cfg.S2_EPOCHS + 1):
        ep_loss = nb = 0
        acc_cos = acc_sens = acc_drift = 0.0
        for imgs, _ in loader:
            imgs = imgs.to(cfg.DEVICE)
            poisoned = apply_trigger(imgs, trig, cfg.TRIG_SZ, apply_mode, apply_opacity)
            emb = F.normalize(model.encode_image(poisoned), dim=-1)

            cos_t = F.cosine_similarity(
                emb, t_target.unsqueeze(0).expand_as(emb), dim=-1).mean()

            grads = torch.autograd.grad(
                cos_t, tensors, create_graph=True, retain_graph=True,
                allow_unused=True)
            sens = _sens_from_grads(entries, grads, tensor_ids)
            if sens is None:
                raise RuntimeError(
                    "L_sens: none of the selected θ_s contributed to the graph")
            if sens0 is None:
                sens0 = sens.detach().clamp_min(1e-30)
                info["sens0"] = float(sens0)
                info["cos_start"] = float(cos_t.detach())
                print(f"  sens0 = {float(sens0):.3e}   cos0 = {float(cos_t.detach()):+.4f}")
            sens_term = sens / sens0 if cfg.S2_SENS_NORM else sens

            loss = (-cfg.S2_LOSS_COS_WEIGHT * cos_t / cfg.TEMPERATURE
                    - cfg.S2_LAMBDA_SENS * sens_term)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([trig], max_norm=cfg.S2_GRAD_CLIP)

            with torch.no_grad():
                if trig.grad is not None and torch.isfinite(trig.grad).all():
                    opt.step()
                # bounded + stay near the Stage-1 trigger
                trig.data.clamp_(-cfg.TRIG_CLAMP, cfg.TRIG_CLAMP)
                diff = trig.data - trigger_s1
                diff_norm = diff.norm()
                limit = cfg.S2_STAGE1_DRIFT_LIMIT * trigger_s1.norm() + 1e-6
                if diff_norm > limit:
                    diff.mul_(limit / diff_norm)
                    trig.data.copy_(trigger_s1 + diff)
                drift = (trig.data - trigger_s1).norm().item() / max(s1_norm, 1e-12)
            sched.step()

            ep_loss += loss.item()
            acc_cos += float(cos_t.detach())
            acc_sens += float(sens.detach())
            acc_drift += drift
            nb += 1

        avg_l = ep_loss / max(nb, 1)
        if avg_l < best_loss:
            best_loss, best_trig = avg_l, trig.data.clone()
        if ep == 1 or ep % 5 == 0 or ep == cfg.S2_EPOCHS:
            print(f"  s2 ep {ep:>3d}  loss={avg_l:+.5f}  "
                  f"cos_t={acc_cos / max(nb, 1):+.4f}  "
                  f"sens={acc_sens / max(nb, 1):.3e}  "
                  f"drift/‖s1‖={acc_drift / max(nb, 1):.3f}")

    info["sens_last"] = acc_sens / max(nb, 1)
    info["cos_end"] = acc_cos / max(nb, 1)
    info["drift_ratio_end"] = acc_drift / max(nb, 1)
    info["best_loss"] = best_loss
    info["s2_calib"] = len(subset)
    info["n_slices"] = len(entries)
    info["n_params"] = n_el
    info["layers"] = layers

    for t in tensors:
        t.requires_grad_(False)
    print(f"  Stage-2 done. best loss={best_loss:+.5f}  "
          f"cos {info['cos_start']:+.4f} → {info['cos_end']:+.4f}  "
          f"sens {info['sens0']:.3e} → {info['sens_last']:.3e}  "
          f"drift/‖s1‖={info['drift_ratio_end']:.3f}")
    return best_trig.detach(), info


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

    # ═══ STAGE-1 configuration (applied before the splits are built) ═══
    if cfg.S1_ENABLED:
        cfg.P1_LOSS = cfg.S1_LOSS
        cfg.N_CALIB = cfg.S1_N_CALIB
        cfg.P1_EPOCHS = cfg.S1_EPOCHS
    s1_tag = f"S1:{cfg.P1_LOSS}/n{cfg.N_CALIB}" if cfg.S1_ENABLED else f"P1:{cfg.P1_LOSS}"
    s2_tag = "+S2:sens" if cfg.S2_ENABLED else ""
    if cfg.S1_ENABLED or cfg.S2_ENABLED:
        tag = (s1_tag + s2_tag).replace("/", "-").replace(":", "")
        cfg.OUT_DIR = f"{cfg.OUT_DIR.rstrip('/')}_{tag}"

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
    ds = datasets.Caltech101(root=cfg.DATA, download=cfg.DOWNLOAD,
                             transform=tf, target_type="category")
    classes = list(ds.categories)
    print(f"  {len(classes)} classes, target = '{classes[cfg.TARGET_CLS]}'")
    text_emb = build_text_embeddings(model, classes, dev)
    print(f"  Target class = {cfg.TARGET_CLS} → '{classes[cfg.TARGET_CLS]}'")

    print("\n[3] Preparing data splits")
    (target_sel, calib_idx, s2_calib_idx, grad_idx, eval_idx, test_idx,
     target_holdout) = prepare_data_splits(ds, cfg)

    target_images, _ = load_tensors(ds, target_sel, dev)
    calib_loader = make_loader(ds, calib_idx, cfg.BATCH, shuffle=True, workers=cfg.WORKERS)
    calib_imgs, _ = load_tensors(ds, calib_idx, dev)
    eval_imgs, eval_labels = load_tensors(ds, eval_idx, dev)
    grad_imgs_cpu, _ = load_tensors(ds, grad_idx, "cpu", batch_size=cfg.BATCH, workers=cfg.WORKERS)

    # Stage-2 calibration uses different images from Stage 1
    s2_calib_loader = None
    if cfg.S2_ENABLED and len(s2_calib_idx) > 0:
        assert set(s2_calib_idx).isdisjoint(calib_idx), \
            "Stage-2 calibration images must be disjoint from Stage-1"
        s2_calib_loader = make_loader(ds, s2_calib_idx, cfg.S2_BATCH,
                                      shuffle=True, workers=cfg.WORKERS)
        print(f"  Stage-2 calibration loader: {len(s2_calib_idx)} images "
              f"({len(set(s2_calib_idx) & set(calib_idx))} shared with Stage 1)")

    with torch.no_grad():
        t_target = text_emb[cfg.TARGET_CLS].detach()

    trigger_token_indices = get_trigger_token_indices(cfg, model=model)

    # ═══ PHASE 0 — Quantisation ═══
    quant_dict = phase0_quantize(model, text_emb, cfg)
    cleanup_gpu()

    # ═══ PHASE 1 / STAGE 1 — Trigger Generation ═══
    print(f"\n[Stage 1] loss={cfg.P1_LOSS}  N_CALIB={cfg.N_CALIB}  "
          f"epochs={cfg.P1_EPOCHS}")
    trigger_s1 = phase1_generate_trigger(model, text_emb, t_target, calib_loader, target_images, cfg)

    asr_pre, cacc_pre = quick_evaluate(model, text_emb, trigger_s1, eval_imgs, eval_labels, cfg.TARGET_CLS, cfg.TRIG_SZ, cfg.TRIGGER_TYPE, cfg.OPACITY, cfg.MINI_BS)
    print(f"  Stage-1 trigger standalone: ASR={asr_pre:.1f}%  CACC={cacc_pre:.1f}%")

    save_trigger_png(trigger_s1, os.path.join(cfg.OUT_DIR, "trigger_patch.png"), bg_image=eval_imgs[0], opacity=cfg.OPACITY, trig_sz=cfg.TRIG_SZ, mode=cfg.TRIGGER_TYPE)
    torch.save(trigger_s1.cpu(), os.path.join(cfg.OUT_DIR, "trigger_stage1.pt"))
    del target_images
    cleanup_gpu()

    # ═══ PHASE 2A — Layer-Level Causal Ablation (on the Stage-1 trigger) ═══
    layer_scores, top_k_layers = phase2_causal_ablation(model, trigger_s1, t_target, calib_imgs, cfg)
    layer_R = layer_scores
    plot_ablation_scores(layer_scores, cfg, top_k_layers=top_k_layers)
    cleanup_gpu()

    # ═══ [S2-GATE] Phase-2 overlap vs the CE-trigger reference ═══
    gate_info = None
    if cfg.S2_ENABLED and cfg.S2_GATE:
        print("\n" + "=" * 60)
        print("S2-GATE: PHASE-2 OVERLAP CHECK")
        print("=" * 60)
        ce_top_k = phase2_ce_reference(
            model, text_emb, t_target, calib_loader, calib_imgs, cfg)
        jac = check_phase2_overlap(
            top_k_layers, ce_top_k, cfg.S2_PHASE2_OVERLAP_THRESHOLD,
            force=cfg.S2_FORCE_PROCEED, n_calib=cfg.N_CALIB)
        gate_info = dict(jaccard=float(jac), stage1_top_k=sorted(int(x) for x in top_k_layers),
                         ce_top_k=sorted(int(x) for x in ce_top_k),
                         threshold=float(cfg.S2_PHASE2_OVERLAP_THRESHOLD),
                         forced=bool(cfg.S2_FORCE_PROCEED))

    # ═══ PHASE 2B — Component-Level Causal Ablation ═══
    component_scores, selected_components = phase2b_component_ablation(model, trigger_s1, t_target, calib_imgs, top_k_layers, cfg)

    # ═══ Weight tracking (from selected components) ═══
    tensor_refs, original_weights = setup_weight_tracking(model, selected_components)

    # ═══ STAGE 2 — Gradient-sensitivity trigger refinement ═══
    trigger = trigger_s1
    stage2_info = None
    if cfg.S2_ENABLED:
        trigger, stage2_info = stage2_refine_trigger(
            model, text_emb, t_target, trigger_s1, selected_components,
            top_k_layers, s2_calib_loader, cfg)
        asr_s2, cacc_s2 = quick_evaluate(
            model, text_emb, trigger, eval_imgs, eval_labels, cfg.TARGET_CLS,
            cfg.TRIG_SZ, cfg.TRIGGER_TYPE, cfg.OPACITY, cfg.MINI_BS)
        print(f"  Stage-2 trigger standalone: ASR={asr_s2:.1f}%  CACC={cacc_s2:.1f}%")
        stage2_info["asr_standalone"] = float(asr_s2)
        stage2_info["cacc_standalone"] = float(cacc_s2)
        save_trigger_png(trigger, os.path.join(cfg.OUT_DIR, "trigger_patch_stage2.png"), bg_image=eval_imgs[0], opacity=cfg.OPACITY, trig_sz=cfg.TRIG_SZ, mode=cfg.TRIGGER_TYPE)
        torch.save(trigger.cpu(), os.path.join(cfg.OUT_DIR, "trigger_stage2.pt"))
        cleanup_gpu()

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
        # `vars(cfg)` on a class-attribute-only config is empty; merge the class
        # dict (and any instance overrides) so the loss weights are recorded.
        "config": {**{k: v for k, v in vars(type(cfg)).items()
                      if not k.startswith("_") and not callable(v)},
                   **vars(cfg)},
        "actual_model_params": mparams, "target_class": classes[cfg.TARGET_CLS],
        "trigger_token_indices": trigger_token_indices, "layer_R": {str(k): v for k, v in layer_R.items()} if layer_R else {},
        "top_k_layers": sorted(top_k_layers), "selected_heads": [{k: v for k, v in s.items() if not isinstance(v, torch.Tensor)} for s in selected_components],
        "stage1": {"loss": cfg.P1_LOSS if cfg.S1_ENABLED else None,
                   "n_calib": int(cfg.N_CALIB), "epochs": int(cfg.P1_EPOCHS),
                   "asr_standalone": float(asr_pre), "cacc_standalone": float(cacc_pre)},
        "s2_gate": gate_info, "stage2": stage2_info,
        "flip_log": flip_log, "checkpoints": checkpoints, "results": {"post_flip_test": {k: v for k, v in r_final.items() if not isinstance(v, torch.Tensor)}},
        "search_time_sec": search_time}

    with open(os.path.join(cfg.OUT_DIR, "bfa_results.json"), "w") as f:
        json.dump(save, f, indent=2, default=str)
    torch.save(trigger.cpu(), os.path.join(cfg.OUT_DIR, "trigger.pt"))
    print(f"\n  Results saved to {cfg.OUT_DIR}/")


if __name__ == "__main__":
    main()