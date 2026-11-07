"""
diagnostic_modality_gap.py
==========================
Standalone diagnostic. Measures the text-image modality gap for the target
class in Caltech-101 with CLIP.

Does NOT run any attack. Answers:
  1. How large is the text-image modality gap for the target class?
  2. Where does the target text embedding sit relative to all 101 class texts?
  3. Where does the target image prototype (mean of N images) sit?
  4. Is the image prototype more similar to the target text than any other
     class text? (a sanity check for whether the target is discriminative)
  5. Does zero-shot text classification work on real target-class images?
  6. What is the per-image cosine distribution?

Outputs:
  - Console table
  - modality_gap_diagnostic.json
  - modality_gap_distribution.png
  - modality_gap_text_similarity.png
"""

import torch
import torch.nn.functional as F
import clip
import numpy as np
import json, os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
class CFG:
    MODEL = "ViT-B/16"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TARGET_CLS = 3
    DATA = "../data"
    DATASET_NAME = "Caltech101"
    DOWNLOAD = False
    WORKERS = 2

    N_TARGET_IMAGES = 64          # use more than the attack (32) for stable stats
    N_BG_PER_CLASS = 4            # sample per other class for background stats
    MINI_BS = 8
    SEED = 42

    OUT_DIR = f"./diagnostic_{MODEL.replace('/', '_')}_{DATASET_NAME}"


# ══════════════════════════════════════════════════════════════════════
#  TEXT / CLIP CONSTANTS
# ══════════════════════════════════════════════════════════════════════
CATEGORY_RENAMES = {
    "Faces":        "human face", "Faces_easy": "close-up human face photograph",
    "Motorbikes":   "motorbike",  "Leopards":   "leopard",
    "ketch":        "ketch ship", "schooner":   "schooner ship",
    "cup":          "drinking cup", "lamp":     "table lamp",
    "laptop":       "laptop computer", "wild_cat": "wild cat",
    "flamingo":     "flamingo bird", "lotus":   "lotus flower",
    "camera":       "camera device", "chair":   "chair furniture",
    "lobster":      "lobster animal", "crocodile": "crocodile reptile",
    "octopus":      "octopus sea creature", "scissors": "pair of scissors",
    "stapler":      "stapler tool", "umbrella": "umbrella rain gear",
    "airplanes":    "airplane", "butterfly": "butterfly insect",
    "car_side":     "side view of a car", "cellphone": "cell phone",
    "cougar_body":  "cougar animal", "cougar_face": "cougar face",
    "crab":         "crab animal", "crayfish": "crayfish animal",
    "crocodile_head": "crocodile head", "dollar_bill": "dollar bill",
    "dragonfly":    "dragonfly insect", "electric_guitar": "electric guitar",
    "elephant":     "elephant animal", "emu":     "emu bird",
    "euphonium":    "euphonium instrument", "ewer": "ewer pitcher",
    "ferry":        "ferry boat", "garfield": "garfield cartoon cat",
    "gerenuk":      "gerenuk antelope", "gramophone": "gramophone",
    "hawksbill":    "hawksbill sea turtle", "headphone": "headphones",
    "hedgehog":     "hedgehog animal", "inline_skate": "inline skate",
    "joshua_tree":  "joshua tree", "kangaroo": "kangaroo animal",
    "mandolin":     "mandolin instrument", "metronome": "metronome",
    "minaret":      "minaret tower", "nautilus": "nautilus shell",
    "pagoda":       "pagoda building", "panda":   "panda bear",
    "pigeon":       "pigeon bird", "pizza":     "pizza food",
    "platypus":     "platypus animal", "pyramid": "pyramid structure",
    "revolver":     "revolver gun", "rhino":     "rhinoceros animal",
    "rooster":      "rooster chicken", "saxophone": "saxophone instrument",
    "snoopy":       "snoopy cartoon dog", "soccer_ball": "soccer ball",
    "starfish":     "starfish sea creature", "stop_sign": "stop sign",
    "strawberry":   "strawberry fruit", "sunflower": "sunflower plant",
    "tick":         "tick insect", "trilobite": "trilobite fossil",
    "watch":        "wrist watch", "water_lilly": "water lily flower",
    "wheelchair":   "wheelchair", "wild_grog_face": "wild grog face",
    "windsor_chair": "windsor chair", "wrench":   "wrench tool",
    "yin_yang":     "yin yang symbol",
}

PROMPT_TEMPLATES = [
    "a photo of a {}.", "a photo of the {}.", "a close-up photo of a {}.",
    "a bright photo of a {}.", "a dark photo of a {}.",
    "a cropped photo of a {}.", "a low resolution photo of a {}.",
    "a photo of a large {}.", "a photo of a small {}.", "a photo of many {}.",
    "a picture of a {}.", "a rendering of a {}.", "a photo of one {}.",
]


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@torch.no_grad()
def encode_images(model, imgs, mini_bs=8):
    parts = []
    for i in range(0, imgs.size(0), mini_bs):
        parts.append(model.encode_image(imgs[i:i + mini_bs]))
    return torch.cat(parts, 0)


def build_text_embeddings(model, classes, device):
    """Multi-template ensemble (same as main pipeline). Also returns per-template."""
    n_classes, n_templates = len(classes), len(PROMPT_TEMPLATES)
    all_prompts = []
    for c in classes:
        name = CATEGORY_RENAMES.get(c, c.replace("_", " ").lower())
        for t in PROMPT_TEMPLATES:
            all_prompts.append(t.format(name))

    embs = []
    with torch.no_grad():
        for i in range(0, len(all_prompts), 256):
            chunk = all_prompts[i:i + 256]
            tok = clip.tokenize(chunk).to(device)
            embs.append(model.encode_text(tok))
    all_emb = torch.cat(embs, 0)                       # [C*T, dim]
    all_emb = all_emb.view(n_classes, n_templates, -1)
    per_template = F.normalize(all_emb, dim=-1)         # [C, T, dim]
    ensemble = F.normalize(all_emb.mean(dim=1), dim=-1) # [C, dim]
    return ensemble, per_template, all_prompts


def load_caltech(cfg, preprocess):
    tf = transforms.Compose([lambda im: im.convert("RGB"), preprocess])
    return datasets.Caltech101(root=cfg.DATA, download=cfg.DOWNLOAD,
                               transform=tf, target_type="category")


def sample_indices(ds, cfg):
    """Sample target-class and background images deterministically."""
    rng = np.random.RandomState(cfg.SEED)
    by_class = defaultdict(list)
    for i in range(len(ds)):
        by_class[ds[i][1]].append(i)

    target_idx = by_class[cfg.TARGET_CLS]
    rng.shuffle(target_idx)
    target_sel = target_idx[:cfg.N_TARGET_IMAGES]

    bg_sel = []
    for cls, idxs in by_class.items():
        if cls == cfg.TARGET_CLS:
            continue
        rng.shuffle(idxs)
        bg_sel.extend(idxs[:cfg.N_BG_PER_CLASS])

    return target_sel, bg_sel


def load_images(ds, indices, cfg):
    loader = DataLoader(Subset(ds, indices), batch_size=cfg.MINI_BS,
                        shuffle=False, num_workers=cfg.WORKERS)
    imgs, lbls = [], []
    for im, lb in loader:
        imgs.append(im)
        lbls.append(lb)
    return torch.cat(imgs, 0), torch.cat(lbls, 0)


# ══════════════════════════════════════════════════════════════════════
#  MAIN DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════
def main():
    set_seed(CFG.SEED)
    cfg = CFG()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    dev = cfg.DEVICE

    print("=" * 64)
    print("MODALITY-GAP DIAGNOSTIC")
    print("=" * 64)
    print(f"  Model : {cfg.MODEL}   Device: {dev}")
    print(f"  Target: class {cfg.TARGET_CLS}")
    print(f"  Target images: {cfg.N_TARGET_IMAGES}  "
          f"BG/class: {cfg.N_BG_PER_CLASS}")

    # ── Load model & data ──
    print("\n[1] Loading CLIP")
    model, preprocess = clip.load(cfg.MODEL, device=dev)
    model = model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("[2] Loading Caltech-101")
    ds = load_caltech(cfg, preprocess)
    classes = list(ds.categories)
    target_name = classes[cfg.TARGET_CLS]
    print(f"  {len(classes)} classes, target = '{target_name}'")

    # ── Text embeddings ──
    print("[3] Encoding text (multi-template)")
    text_emb, per_template, _ = build_text_embeddings(model, classes, dev)
    print(f"  text_emb shape: {tuple(text_emb.shape)}  "
          f"per_template: {tuple(per_template.shape)}")
    t_target = text_emb[cfg.TARGET_CLS]                 # [dim]

    # ── Sample and encode images ──
    print("[4] Sampling images")
    target_idx, bg_idx = sample_indices(ds, cfg)
    print(f"  target images: {len(target_idx)}  bg images: {len(bg_idx)}")

    print("[5] Encoding images")
    target_imgs, _ = load_images(ds, target_idx, cfg)
    bg_imgs, bg_lbls = load_images(ds, bg_idx, cfg)

    emb_target = F.normalize(
        encode_images(model, target_imgs.to(dev), cfg.MINI_BS), dim=-1)
    emb_bg = F.normalize(
        encode_images(model, bg_imgs.to(dev), cfg.MINI_BS), dim=-1)

    # ═══════════════════════════════════════════════════════════════
    #  METRIC 1 — text-text structure (where does target text sit?)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("METRIC 1: text-text similarity (target text vs all 101 class texts)")
    print("=" * 64)
    sim_text = (text_emb @ t_target).cpu().numpy()       # [101]
    sim_text[cfg.TARGET_CLS] = 1.0                       # self = 1
    order = np.argsort(-sim_text)

    print(f"  Self-similarity : {sim_text[cfg.TARGET_CLS]:.4f}  (should be 1.0)")
    print(f"  Top-10 most similar OTHER classes:")
    others = [(int(i), float(sim_text[i])) for i in order
              if i != cfg.TARGET_CLS][:10]
    for rank, (cls_i, s) in enumerate(others, 1):
        print(f"    #{rank:2d}  {classes[cls_i]:<20s}  cos = {s:+.4f}")
    print(f"  Mean over other 100 classes: "
          f"{sim_text[[i for i in range(len(classes)) if i != cfg.TARGET_CLS]].mean():+.4f}")

    # ═══════════════════════════════════════════════════════════════
    #  METRIC 2 — image prototype vs target text (THE modality gap)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("METRIC 2: TEXT ↔ IMAGE modality gap (target class)")
    print("=" * 64)
    t_img_proto = F.normalize(emb_target.mean(0, keepdim=True), dim=-1)  # [1, dim]
    cos_proto = (t_img_proto @ t_target.unsqueeze(-1)).item()

    print(f"  cos(target_text, image_prototype)  = {cos_proto:+.4f}   "
          f"←←← the anchor uses this number")

    # Per-image cosine
    cos_per_img = (emb_target @ t_target.unsqueeze(-1)).squeeze(-1).cpu().numpy()
    print(f"  Per-image cos(target_text, img_i):")
    print(f"    mean = {cos_per_img.mean():+.4f}")
    print(f"    std  = {cos_per_img.std():+.4f}")
    print(f"    min  = {cos_per_img.min():+.4f}")
    print(f"    max  = {cos_per_img.max():+.4f}")

    # Pairwise image-image cosine
    img_img = (emb_target @ emb_target.T).cpu().numpy()
    mask = np.triu_indices(len(img_img), k=1)
    img_img_vals = img_img[mask]
    print(f"  Pairwise image-image cosine (within class):")
    print(f"    mean = {img_img_vals.mean():+.4f}")
    print(f"    std  = {img_img_vals.std():+.4f}")

    # ═══════════════════════════════════════════════════════════════
    #  METRIC 3 — prototype vs ALL class texts
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("METRIC 3: image prototype vs ALL 101 class texts (sanity)")
    print("=" * 64)
    sim_proto = (text_emb @ t_img_proto.T).squeeze(-1).cpu().numpy()  # [101]
    order_proto = np.argsort(-sim_proto)
    target_rank = int(np.where(order_proto == cfg.TARGET_CLS)[0][0]) + 1

    print(f"  Rank of target text among all 101 texts "
          f"(by similarity to image prototype): #{target_rank}")
    print(f"  Top-5 closest class texts to image prototype:")
    for rank, cls_i in enumerate(order_proto[:5], 1):
        mark = " ← target" if cls_i == cfg.TARGET_CLS else ""
        print(f"    #{rank}  {classes[cls_i]:<20s}  cos = {sim_proto[cls_i]:+.4f}{mark}")

    if target_rank == 1:
        print("  ✓ Image prototype is closest to its own class text — "
              "the anchor direction is correct.")
    elif target_rank <= 5:
        print(f"  ⚠ Image prototype ranks #{target_rank} — the target "
              f"text is NOT the closest; anchor is slightly ambiguous.")
    else:
        print(f"  ✗ Image prototype ranks #{target_rank} — the target "
              f"text is far from the image cluster; anchor is problematic.")

    # ═══════════════════════════════════════════════════════════════
    #  METRIC 4 — zero-shot text classification on target images
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("METRIC 4: zero-shot text classification on target-class images")
    print("=" * 64)
    logits = emb_target @ text_emb.T                 # [N, 101]
    preds = logits.argmax(-1).cpu().numpy()
    n_correct = (preds == cfg.TARGET_CLS).sum()
    acc = 100.0 * n_correct / len(preds)
    print(f"  Accuracy (text-argmax on real target images): "
          f"{acc:.1f}%  ({n_correct}/{len(preds)})")

    # Top mistakes
    from collections import Counter
    mistakes = Counter()
    for p, gt in zip(preds, [cfg.TARGET_CLS] * len(preds)):
        if p != gt:
            mistakes[classes[p]] += 1
    if mistakes:
        print(f"  Top-5 confused classes:")
        for cls_name, cnt in mistakes.most_common(5):
            print(f"    {cls_name:<20s}  {cnt} times")

    # ═══════════════════════════════════════════════════════════════
    #  METRIC 5 — per-template variation
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("METRIC 5: how much do the 13 templates disagree with the ensemble?")
    print("=" * 64)
    per_t = per_template[cfg.TARGET_CLS]              # [T, dim]
    cos_to_ensemble = (per_t @ t_target.unsqueeze(-1)).squeeze(-1).cpu().numpy()
    print(f"  cos(ensemble, template_i):  "
          f"min={cos_to_ensemble.min():+.4f}  "
          f"mean={cos_to_ensemble.mean():+.4f}  "
          f"max={cos_to_ensemble.max():+.4f}")

    # ═══════════════════════════════════════════════════════════════
    #  METRIC 6 — background images (how far are non-target images?)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("METRIC 6: how far are non-target images from the target text?")
    print("=" * 64)
    cos_bg = (emb_bg @ t_target.unsqueeze(-1)).squeeze(-1).cpu().numpy()
    print(f"  cos(target_text, bg_image_i):")
    print(f"    mean = {cos_bg.mean():+.4f}")
    print(f"    std  = {cos_bg.std():+.4f}")
    print(f"    min  = {cos_bg.min():+.4f}  (most distant)")
    print(f"    max  = {cos_bg.max():+.4f}  (most similar)")

    # ═══════════════════════════════════════════════════════════════
    #  INTERPRETATION
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("INTERPRETATION")
    print("=" * 64)
    gap = cos_proto
    bg_mean = float(cos_bg.mean())
    delta_target_vs_bg = gap - bg_mean
    print(f"  Modality gap (cos text↔img):     {gap:+.4f}")
    print(f"  Background images cos to text:    {bg_mean:+.4f}")
    print(f"  Δ (target − background):          {delta_target_vs_bg:+.4f}")

    if delta_target_vs_bg > 0.15:
        print(f"  ✓ Target images are clearly closer to the target text than "
              f"background images → text-based attack is well-posed.")
    elif delta_target_vs_bg > 0.05:
        print(f"  ~ Target images are moderately closer than background; "
              f"attack should work but with limited margin.")
    else:
        print(f"  ✗ Target images are barely closer to the target text than "
              f"background → text-based attack has weak signal.")

    # ═══════════════════════════════════════════════════════════════
    #  SAVE JSON
    # ═══════════════════════════════════════════════════════════════
    save = {
        "model": cfg.MODEL,
        "target_class_idx": cfg.TARGET_CLS,
        "target_class_name": target_name,
        "n_target_images": int(len(cos_per_img)),
        "n_background_images": int(len(cos_bg)),

        "metric_1_text_text": {
            "self": 1.0,
            "top10_other_classes": [
                {"name": classes[i], "cos": s} for i, s in others],
            "mean_other": float(
                sim_text[[i for i in range(len(classes))
                          if i != cfg.TARGET_CLS]].mean()),
        },
        "metric_2_text_image_gap": {
            "cos_text_to_image_prototype": float(cos_proto),
            "per_image_mean": float(cos_per_img.mean()),
            "per_image_std":  float(cos_per_img.std()),
            "per_image_min":  float(cos_per_img.min()),
            "per_image_max":  float(cos_per_img.max()),
            "image_image_mean": float(img_img_vals.mean()),
            "image_image_std":  float(img_img_vals.std()),
        },
        "metric_3_prototype_vs_all_texts": {
            "target_rank": int(target_rank),
            "top5": [
                {"name": classes[i], "cos": float(sim_proto[i])}
                for i in order_proto[:5]],
        },
        "metric_4_zeroshot_on_target_images": {
            "acc": float(acc),
            "n_correct": int(n_correct),
            "n_total": int(len(preds)),
            "top_mistakes": dict(mistakes.most_common(5)),
        },
        "metric_5_template_variation": {
            "min": float(cos_to_ensemble.min()),
            "mean": float(cos_to_ensemble.mean()),
            "max": float(cos_to_ensemble.max()),
        },
        "metric_6_background": {
            "mean": float(cos_bg.mean()),
            "std":  float(cos_bg.std()),
            "min":  float(cos_bg.min()),
            "max":  float(cos_bg.max()),
        },
        "interpretation": {
            "gap": float(gap),
            "bg_mean": bg_mean,
            "delta_target_vs_bg": float(delta_target_vs_bg),
        },
    }
    with open(os.path.join(cfg.OUT_DIR, "modality_gap_diagnostic.json"), "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  Saved → {cfg.OUT_DIR}/modality_gap_diagnostic.json")

    # ═══════════════════════════════════════════════════════════════
    #  PLOT 1 — distribution of per-image cosine to text
    # ═══════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(cos_per_img, bins=20, color="#2ecc71", alpha=0.75,
            edgecolor="white", label=f"target images (n={len(cos_per_img)})")
    ax.hist(cos_bg, bins=20, color="#e74c3c", alpha=0.55,
            edgecolor="white", label=f"bg images (n={len(cos_bg)})")
    ax.axvline(cos_proto, color="blue", linestyle="--",
               label=f"image prototype = {cos_proto:+.3f}")
    ax.set_xlabel("cos(image_embedding, target_text)")
    ax.set_ylabel("Count")
    ax.set_title("Per-image cosine to target text\n"
                 f"(target mean = {cos_per_img.mean():+.3f}, "
                 f"bg mean = {cos_bg.mean():+.3f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    k = 15
    top_idx = order_proto[:k]
    names = [classes[i][:18] for i in top_idx]
    vals = [float(sim_proto[i]) for i in top_idx]
    colors = ["#e74c3c" if i == cfg.TARGET_CLS else "#3498db" for i in top_idx]
    ax.barh(range(k), vals, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(k))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(cos_proto, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("cos(image_prototype, class_text)")
    ax.set_title(f"Top-{k} class texts by similarity to image prototype\n"
                 f"target rank = #{target_rank}")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"Modality-gap diagnostic — {cfg.MODEL} on Caltech-101 "
                 f"(target: '{target_name}')", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path1 = os.path.join(cfg.OUT_DIR, "modality_gap_distribution.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path1}")

    # ═══════════════════════════════════════════════════════════════
    #  PLOT 2 — full 101-class similarity to image prototype
    # ═══════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 4))
    sorted_vals = np.sort(sim_proto)[::-1]
    ax.plot(sorted_vals, "o-", color="#3498db", markersize=4)
    ax.axhline(cos_proto, color="red", linestyle="--", alpha=0.7,
               label=f"target class value = {cos_proto:+.3f} "
                     f"(rank #{target_rank})")
    ax.set_xlabel("Class rank (sorted by similarity to image prototype)")
    ax.set_ylabel("cos(image_prototype, class_text)")
    ax.set_title("Similarity of image prototype to all 101 class texts")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path2 = os.path.join(cfg.OUT_DIR, "modality_gap_text_similarity.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path2}")

    print("\n" + "=" * 64)
    print("DONE")
    print("=" * 64)
    print(f"  All artifacts saved to {cfg.OUT_DIR}/")


if __name__ == "__main__":
    main()