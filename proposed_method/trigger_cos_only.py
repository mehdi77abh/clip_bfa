"""
compare_vectors.py — Direct comparison of CLIP text vs image embeddings
========================================================================
No training, no trigger, no bit flips. Just measure how text and image
embeddings for the same concept relate.

Sections:
    1. Per-vector statistics (dim, norm, kurtosis, sparsity, top-K dims)
    2. Intra-modal consistency (15 text prompts; N image samples)
    3. Direct cross-modal comparison (L2, cosine, sign agreement,
       per-dim correlation, top-K overlap, per-image L2/cos)
    4. Cross-modal significance (target pair vs 100 random classes)
    5. Energy concentration (cumulative L2² in top-K dims)
    6. Top-K subvector cosine (text-topK, image-topK, union, intersection)
    7. Principal angles between PCA subspaces
    8. Direction vs magnitude decomposition of raw L2

Examples:
    python compare_vectors.py --target-class lion --n-target 10 --save-plots
    python compare_vectors.py --dataset caltech101 --data ../data \
        --target-class Motorbikes --model ViT-B/16
"""

import argparse, json, os, random, sys
import numpy as np
import torch
import torch.nn.functional as F
import clip
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

PROMPT_TEMPLATES = [
    "a photo of a {}.", "a photo of the {}.", "a photo of one {}.",
    "a photo of many {}.", "a close-up photo of a {}.",
    "a cropped photo of a {}.", "a bright photo of a {}.",
    "a dark photo of a {}.", "a low resolution photo of a {}.",
    "a photo of a large {}.", "a photo of a small {}.",
    "a rendering of a {}.", "a picture of a {}.",
    "a blurry photo of a {}.", "a photo of the {} in the wild.",
]


# ────────────────────────────────────────────────────────────────────────
#  Utilities
# ────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)


def load_class_names(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def resolve_class(name, names):
    key = name.strip().lower().replace("_", " ")
    lookup = {n.lower().replace("_", " "): i for i, n in enumerate(names)}
    if key not in lookup:
        close = [n for n in names if key[:3] in n.lower()][:5]
        sys.exit(f"class {name!r} not found. Closest: {close}")
    return lookup[key]


def resolve_data_root(root, split):
    if os.path.isdir(os.path.join(root, split)): return root
    alt = os.path.join(root, "imagenet")
    if os.path.isdir(os.path.join(alt, split)): return alt
    sys.exit(f"no '{split}' under {root} or {alt}")


def build_dataset(cfg, preprocess):
    tf = transforms.Compose([lambda im: im.convert("RGB"), preprocess])
    if cfg.dataset == "imagenet":
        root = resolve_data_root(cfg.data, cfg.split)
        ds = datasets.ImageNet(root=root, split=cfg.split, transform=tf)
        classes = load_class_names(cfg.class_list)
        if len(classes) != 1000: sys.exit(f"expected 1000 classes")
        return ds, list(ds.targets), classes, root
    def _caltech_ok(r):
        return any(os.path.isdir(os.path.join(r, b, "101_ObjectCategories"))
                   for b in ("caltech101", ""))
    root = cfg.data
    for c in [cfg.data, os.path.dirname(os.path.abspath(cfg.data))]:
        if _caltech_ok(c): root = c; break
    ds = datasets.Caltech101(root=root, download=False, transform=tf, target_type="category")
    return ds, list(ds.y), list(ds.categories), root


def sample_target_imgs(labels, target_idx, n_target, seed):
    rng = np.random.RandomState(seed + 1)
    pool = [i for i, y in enumerate(labels) if y == target_idx]
    if len(pool) < n_target:
        sys.exit(f"only {len(pool)} target images, requested {n_target}")
    rng.shuffle(pool)
    return pool[:n_target]


@torch.no_grad()
def encode_images(model, images, mini_bs=8):
    parts = []
    for i in range(0, images.size(0), mini_bs):
        parts.append(model.encode_image(images[i:i + mini_bs]))
    return torch.cat(parts, dim=0)


@torch.no_grad()
def load_tensors(ds, idx, device, batch=64, workers=2):
    loader = DataLoader(Subset(ds, idx), batch_size=batch, shuffle=False, num_workers=workers)
    imgs = []
    for x, _ in loader:
        imgs.append(x.to(device))
    return torch.cat(imgs, dim=0)


@torch.no_grad()
def build_class_text_embeddings(model, names, device, templates=PROMPT_TEMPLATES, chunk=256):
    prompts, owner = [], []
    for ci, name in enumerate(names):
        for t in templates:
            prompts.append(t.format(name)); owner.append(ci)
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


def l2(a, b): return (a.flatten() - b.flatten()).pow(2).sum().sqrt().item()
def cosine(a, b): return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def effective_rank(X, eps=1e-12):
    """Participation ratio of singular values of X (N, D).
    For N=1 (single vector), returns 1.0 (one direction, by definition)."""
    if X.dim() == 1: X = X.unsqueeze(0)
    if X.size(0) < 2: return 1.0
    Xc = X - X.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(Xc.float())
    p = s.pow(2)
    denom = p.pow(2).sum() + eps
    if denom.item() < eps * 1e3: return 0.0
    return (p.sum().pow(2) / denom).item()


# ────────────────────────────────────────────────────────────────────────
#  Section 1 — per-vector statistics
# ────────────────────────────────────────────────────────────────────────
def analyze_vector(v, name, topk=10):
    v = v.flatten()
    abs_v = v.abs()
    d = v.numel()
    thresh = 1.0 / (d ** 0.5)
    active = (abs_v > thresh).sum().item()
    top_vals, top_idx = abs_v.topk(topk)
    top_pairs = [(int(i), float(v[i].item())) for i in top_idx]
    top_str = ", ".join(f"dim{i}:{val:+.4f}" for i, val in top_pairs)

    centered = v - v.mean()
    var = centered.var().item()
    kurt = (centered.pow(4).mean() / (var * var + 1e-12) - 3).item() if var > 1e-12 else 0.0

    print(f"\n  [{name}] dim={d}  ||v||={v.norm():.4f}")
    print(f"    mean={v.mean():+.6f}  std={v.std():+.6f}  "
          f"min={v.min():+.4f}  max={v.max():+.4f}")
    print(f"    L1={abs_v.sum():.4f}  kurtosis={kurt:+.3f}")
    print(f"    positive dims: {(v > 0).sum().item()}/{d} ({(v > 0).float().mean()*100:.1f}%)")
    print(f"    active dims (|v| > 1/√d ≈ {thresh:.4f}): {active}/{d} ({active/d*100:.1f}%)")
    print(f"    top-{topk} |value|: {top_str}")

    return dict(mean=float(v.mean()), std=float(v.std()),
                min=float(v.min()), max=float(v.max()),
                l2_norm=float(v.norm()), l1=float(abs_v.sum()),
                kurtosis=float(kurt),
                active_dims=int(active), positive_dims=int((v > 0).sum()),
                top_idx=[int(i) for i in top_idx],
                top_vals=[float(v[i]) for i in top_idx])


# ────────────────────────────────────────────────────────────────────────
#  Section 2 — intra-modal consistency
# ────────────────────────────────────────────────────────────────────────
def intra_modal_consistency(X_emb, X_centroid, X_name):
    N = X_emb.size(0)
    cos_to_cent = F.cosine_similarity(X_emb, X_centroid.expand_as(X_emb), dim=-1)
    if N > 1:
        pairwise = X_emb @ X_emb.T
        mask = ~torch.eye(N, dtype=torch.bool, device=X_emb.device)
        off = pairwise[mask]
        pw_mean, pw_std = float(off.mean()), float(off.std())
    else:
        pw_mean = pw_std = float('nan')

    eff_r = effective_rank(X_emb)
    print(f"\n  Intra-{X_name} consistency (N={N}):")
    print(f"    cos(x_i, centroid): {cos_to_cent.mean():+.4f} ± {cos_to_cent.std():.4f}  "
          f"[min={cos_to_cent.min():+.4f}, max={cos_to_cent.max():+.4f}]")
    if N > 1:
        print(f"    cos(x_i, x_j) i≠j:  {pw_mean:+.4f} ± {pw_std:.4f}")
    print(f"    effective rank:     {eff_r:.2f} (out of max {min(N, X_emb.size(1))})")

    return dict(n=int(N),
                cos_to_centroid_mean=float(cos_to_cent.mean()),
                cos_to_centroid_std=float(cos_to_cent.std()),
                cos_to_centroid_min=float(cos_to_cent.min()),
                cos_to_centroid_max=float(cos_to_cent.max()),
                pairwise_mean=pw_mean, pairwise_std=pw_std,
                effective_rank=eff_r)


# ────────────────────────────────────────────────────────────────────────
#  Section 3 — direct cross-modal comparison
# ────────────────────────────────────────────────────────────────────────
def direct_comparison(t_raw, t_norm, v_raw, v_norm, v_imgs_norm):
    print(f"\n  L2 (raw, before norm):        {l2(t_raw, v_raw):.4f}")
    print(f"  L2 (after L2-norm):            {l2(t_norm, v_norm):.4f}")
    cs = cosine(t_norm, v_norm)
    print(f"  Cosine (after L2-norm):        {cs:+.4f}")
    print(f"  Expected L2 from cosine:       {(2 - 2*cs)**0.5:.4f}")
    print(f"  raw/normalized L2 ratio:       {l2(t_raw, v_raw) / (l2(t_norm, v_norm) + 1e-12):.4f}")

    diff = t_norm.flatten() - v_norm.flatten()
    print(f"\n  Element-wise diff (after norm):")
    print(f"    mean={diff.mean():+.6f}  std={diff.std():+.6f}  "
          f"min={diff.min():+.4f}  max={diff.max():+.4f}")
    print(f"    |diff| mean={diff.abs().mean():.6f}  "
          f"median={diff.abs().median():.6f}  p95={diff.abs().quantile(0.95):.6f}")

    sign_match = ((t_norm > 0) == (v_norm > 0)).float().mean().item()
    print(f"\n  Sign agreement:  {sign_match*100:.1f}%  "
          f"(uniform baseline 50%; >60% → aligned)")

    corr = ((t_norm - t_norm.mean()) * (v_norm - v_norm.mean())).sum() / \
           ((t_norm - t_norm.mean()).norm() * (v_norm - v_norm.mean()).norm())
    print(f"  Per-dim correlation (centered): {corr.item():+.4f}")

    print(f"\n  Top-K |value| dim overlap:")
    out_top = {}
    for K in (10, 30, 100):
        t_top = set(t_norm.flatten().abs().topk(K).indices.tolist())
        v_top = set(v_norm.flatten().abs().topk(K).indices.tolist())
        overlap = len(t_top & v_top)
        print(f"    K={K:<3d}: {overlap:>2d}/{K}  ({overlap/K*100:5.1f}%)")
        out_top[K] = dict(n_shared=overlap, fraction=overlap / K)

    per_img_cos = F.cosine_similarity(v_imgs_norm, t_norm.expand_as(v_imgs_norm), dim=-1)
    per_img_l2 = (v_imgs_norm - t_norm).pow(2).sum(dim=-1).sqrt()
    print(f"\n  Per-image cosines to text:")
    print(f"    cos mean={per_img_cos.mean():+.4f}  std={per_img_cos.std():.4f}  "
          f"min={per_img_cos.min():+.4f}  max={per_img_cos.max():+.4f}")
    print(f"    L2  mean={per_img_l2.mean():.4f}  std={per_img_l2.std():.4f}  "
          f"min={per_img_l2.min():.4f}  max={per_img_l2.max():.4f}")

    return dict(
        l2_raw=l2(t_raw, v_raw), l2_norm=l2(t_norm, v_norm),
        cosine=cs, expected_l2_from_cos=(2 - 2*cs)**0.5,
        l2_ratio=l2(t_raw, v_raw) / (l2(t_norm, v_norm) + 1e-12),
        diff_mean=float(diff.mean()), diff_std=float(diff.std()),
        diff_abs_mean=float(diff.abs().mean()),
        diff_abs_median=float(diff.abs().median()),
        diff_abs_p95=float(diff.abs().quantile(0.95)),
        sign_agreement=float(sign_match), per_dim_correlation=float(corr),
        top_k_overlap=out_top,
        per_image_cos_mean=float(per_img_cos.mean()),
        per_image_cos_std=float(per_img_cos.std()),
        per_image_l2_mean=float(per_img_l2.mean()),
        per_image_l2_std=float(per_img_l2.std()),
    )


# ────────────────────────────────────────────────────────────────────────
#  Section 4 — cross-modal significance
# ────────────────────────────────────────────────────────────────────────
def cross_modal_significance(t_norm, v_norm, text_emb_all, target_idx, n_random=100, seed=42):
    target_cos = cosine(t_norm, v_norm)
    all_idx = torch.arange(text_emb_all.size(0), device=text_emb_all.device)
    avail = all_idx[all_idx != target_idx]
    g = torch.Generator(device=text_emb_all.device).manual_seed(seed)
    perm = torch.randperm(len(avail), generator=g, device=text_emb_all.device)[:n_random]
    rand_texts = text_emb_all[avail[perm]]
    rand_cos = (rand_texts * v_norm).sum(dim=-1)

    z = (target_cos - rand_cos.mean()) / (rand_cos.std() + 1e-12)
    pct = (rand_cos < target_cos).float().mean().item() * 100

    print(f"\n  cos(t_target, v_target)      : {target_cos:+.4f}  (target pair)")
    print(f"  cos(t_random, v_target)      : {rand_cos.mean():+.4f} ± {rand_cos.std():.4f}  (n={n_random})")
    print(f"  random range                 : [{rand_cos.min():+.4f}, {rand_cos.max():+.4f}]")
    print(f"  target percentile            : {pct:.1f}%")
    print(f"  z-score vs random            : {z:+.2f}")

    return dict(target_cos=target_cos,
                random_mean=float(rand_cos.mean()), random_std=float(rand_cos.std()),
                random_min=float(rand_cos.min()), random_max=float(rand_cos.max()),
                target_percentile=pct, z_score=float(z))


# ────────────────────────────────────────────────────────────────────────
#  Section 5 — energy concentration
# ────────────────────────────────────────────────────────────────────────
def energy_concentration(v, name, top_ks=(2, 10, 50, 100)):
    v = v.flatten()
    energy = v.pow(2)
    total = energy.sum().item()
    sorted_e = energy.sort(descending=True).values
    cum = sorted_e.cumsum(0) / (total + 1e-12)

    print(f"\n  Energy concentration ({name}):")
    print(f"    total L2²: {total:.4f}")
    out = dict(total_energy=total)
    for K in top_ks:
        if K <= v.numel():
            frac = cum[K - 1].item()
            print(f"    top-{K:<3d}: {frac*100:5.1f}% of L2²")
            out[f"top{K}"] = float(frac)
    return out


# ────────────────────────────────────────────────────────────────────────
#  Section 6 — top-K subvector cosine
# ────────────────────────────────────────────────────────────────────────
def top_k_subvector_cosine(t_norm, v_norm, ks=(10, 30, 100)):
    t = t_norm.flatten(); v = v_norm.flatten()
    print(f"    {'K':>4s}  {'text-topK':>10s}  {'img-topK':>10s}  {'union':>10s}  {'intersection':>15s}")
    out = {}
    for K in ks:
        if K > t.numel(): continue
        t_top = t.abs().topk(K).indices
        v_top = v.abs().topk(K).indices

        cos_tk_t = F.cosine_similarity(t[t_top], v[t_top], dim=0).item()
        cos_tk_v = F.cosine_similarity(t[v_top], v[v_top], dim=0).item()
        union = torch.unique(torch.cat([t_top, v_top]))
        cos_tk_u = F.cosine_similarity(t[union], v[union], dim=0).item()

        m_t = torch.zeros(t.numel(), dtype=torch.bool, device=t.device)
        m_v = torch.zeros(v.numel(), dtype=torch.bool, device=v.device)
        m_t[t_top] = True; m_v[v_top] = True
        inter = (m_t & m_v).nonzero(as_tuple=True)[0]
        cos_tk_i = (F.cosine_similarity(t[inter], v[inter], dim=0).item()
                    if len(inter) >= 2 else float('nan'))

        inter_str = f"{cos_tk_i:+.4f} ({len(inter)})" if cos_tk_i == cos_tk_i else f"nan ({len(inter)})"
        print(f"    {K:>4d}  {cos_tk_t:>+10.4f}  {cos_tk_v:>+10.4f}  "
              f"{cos_tk_u:>+10.4f}  {inter_str:>15s}")
        out[K] = dict(text_topK=cos_tk_t, image_topK=cos_tk_v,
                      union=cos_tk_u, intersection=cos_tk_i,
                      n_intersection=int(len(inter)))
    return out


# ────────────────────────────────────────────────────────────────────────
#  Section 7 — principal angles
# ────────────────────────────────────────────────────────────────────────
def principal_angles(X, Y, K=5):
    if X.size(0) < 2 or Y.size(0) < 2: return None
    K = min(K, X.size(0), Y.size(0), X.size(1), Y.size(1))
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    _, _, Vx = torch.linalg.svd(Xc.float(), full_matrices=False)
    _, _, Vy = torch.linalg.svd(Yc.float(), full_matrices=False)
    Ux = Vx[:K].T; Uy = Vy[:K].T
    M = Ux.T @ Uy
    s = torch.linalg.svdvals(M).clamp(-1, 1)
    angles = torch.arccos(s) * 180 / np.pi
    mean_a = float(angles.mean())
    interp = "aligned" if mean_a < 45 else "orthogonal" if mean_a > 75 else "partially aligned"
    print(f"    mean angle: {mean_a:.2f}°  → {interp}")
    for i, (c, a) in enumerate(zip(s.tolist(), angles.tolist())):
        print(f"    θ_{i+1}: {a:6.2f}°  (cos={c:+.4f})")
    return dict(K=K, cosines=s.tolist(), angles_deg=angles.tolist(),
                mean_angle=mean_a, interpretation=interp)


# ────────────────────────────────────────────────────────────────────────
#  Section 8 — direction vs magnitude decomposition
# ────────────────────────────────────────────────────────────────────────
def direction_magnitude_decomposition(t_raw, v_raw, t_norm, v_norm):
    t_mag = t_raw.norm().item(); v_mag = v_raw.norm().item()
    cos = cosine(t_norm, v_norm)
    l2_raw = (t_mag**2 + v_mag**2 - 2 * t_mag * v_mag * cos) ** 0.5
    avg = (t_mag + v_mag) / 2
    l2_dir = (2 * avg**2 * (1 - cos)) ** 0.5
    l2_mag = abs(t_mag - v_mag)

    print(f"    ||t_raw||={t_mag:.4f}  ||v_raw||={v_mag:.4f}  ratio={v_mag/t_mag:.4f}")
    print(f"    L2_raw={l2_raw:.4f}  =  direction({l2_dir:.4f}) ⊕ magnitude({l2_mag:.4f})")
    print(f"    fraction from direction: {l2_dir/(l2_dir+l2_mag)*100:.1f}%")
    print(f"    fraction from magnitude: {l2_mag/(l2_dir+l2_mag)*100:.1f}%")
    if l2_dir > 3 * l2_mag: print("    ✅ direction dominates → cosine is a fair proxy")
    elif l2_mag > 3 * l2_dir: print("    🔴 magnitude dominates → L2 raw is misleading")
    else: print("    ⚠️ both contribute significantly")

    return dict(norm_text=t_mag, norm_image=v_mag, l2_raw=l2_raw,
                l2_direction=l2_dir, l2_magnitude=l2_mag,
                direction_fraction=l2_dir / (l2_dir + l2_mag),
                magnitude_fraction=l2_mag / (l2_dir + l2_mag))


# ────────────────────────────────────────────────────────────────────────
#  Plots
# ────────────────────────────────────────────────────────────────────────
def plot_all(t_norm, v_norm, prompt_embs_norm, v_imgs_norm, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    # (0,0) value distribution
    ax = axes[0, 0]
    ax.hist(t_norm.flatten().cpu().numpy(), bins=80, alpha=0.6, label="text", color="C0")
    ax.hist(v_norm.flatten().cpu().numpy(), bins=80, alpha=0.6, label="image", color="C1")
    ax.set_title("Value distribution (text vs image centroid)")
    ax.set_xlabel("value"); ax.set_ylabel("count"); ax.legend(); ax.set_yscale("log")

    # (0,1) element-wise diff
    ax = axes[0, 1]
    diff = (t_norm - v_norm).flatten().cpu().numpy()
    ax.hist(diff, bins=80, color="C2"); ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"Element-wise diff (t − v)\nmean={diff.mean():+.4f}, std={diff.std():.4f}")
    ax.set_xlabel("t[i] − v[i]"); ax.set_ylabel("count")

    # (1,0) sorted |values|
    ax = axes[1, 0]
    ax.plot(np.sort(t_norm.flatten().cpu().abs().numpy())[::-1], label="text", lw=2)
    ax.plot(np.sort(v_norm.flatten().cpu().abs().numpy())[::-1], label="image", lw=2)
    ax.set_title("Sorted |values| — sparsity comparison")
    ax.set_xlabel("dim rank"); ax.set_ylabel("|value|"); ax.legend(); ax.set_yscale("log")

    # (1,1) per-dim scatter
    ax = axes[1, 1]
    t_np = t_norm.flatten().cpu().numpy(); v_np = v_norm.flatten().cpu().numpy()
    ax.scatter(t_np, v_np, s=1, alpha=0.4)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    r = np.corrcoef(t_np, v_np)[0, 1]
    ax.set_title(f"Per-dim scatter\nPearson r = {r:+.3f}")
    ax.set_xlabel("text[i]"); ax.set_ylabel("image[i]")
    lim = max(abs(t_np).max(), abs(v_np).max())
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    # (2,0) cosine distributions
    ax = axes[2, 0]
    intra_t = (prompt_embs_norm @ prompt_embs_norm.T).flatten().cpu().numpy()
    intra_i = (v_imgs_norm @ v_imgs_norm.T).flatten().cpu().numpy()
    cross = F.cosine_similarity(v_imgs_norm, t_norm.expand_as(v_imgs_norm), dim=-1).cpu().numpy()
    intra_t = intra_t[intra_t < 0.99]; intra_i = intra_i[intra_i < 0.99]
    ax.hist(intra_t, bins=40, alpha=0.5, label=f"text-text (n={len(intra_t)})")
    ax.hist(intra_i, bins=40, alpha=0.5, label=f"image-image (n={len(intra_i)})")
    ax.hist(cross, bins=40, alpha=0.5, label=f"text-image (n={len(cross)})")
    ax.set_title("Cosine distributions by pairing type")
    ax.set_xlabel("cosine"); ax.set_ylabel("count"); ax.legend()

    # (2,1) energy concentration
    ax = axes[2, 1]
    for v_flat, name in [(t_norm.flatten(), "text"), (v_norm.flatten(), "image")]:
        e = v_flat.pow(2).sort(descending=True).values.cpu().numpy()
        cum = np.cumsum(e) / e.sum()
        ax.plot(cum, label=name)
    ax.set_xscale("log")
    ax.set_title("Energy concentration (cumulative L2²)")
    ax.set_xlabel("top-K dims (log)"); ax.set_ylabel("% of L2²"); ax.legend()
    ax.axhline(0.9, color="k", ls="--", lw=0.5, alpha=0.5)
    ax.axhline(0.99, color="k", ls=":", lw=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ViT-L/14")
    ap.add_argument("--dataset", choices=["imagenet", "caltech101"], default="imagenet")
    ap.add_argument("--data", default="../data/imagenet")
    ap.add_argument("--split", default="val")
    ap.add_argument("--class-list",
                    default="./modular_new_pipeline/class_texts/imagenet1k/classes.txt")
    ap.add_argument("--target-class", default="lion")
    ap.add_argument("--n-target", type=int, default=10)
    ap.add_argument("--n-random", type=int, default=100,
                    help="number of random classes for significance baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--save-plots", action="store_true")
    return ap.parse_args()


def main():
    cfg = parse_args()
    set_seed(cfg.seed); dev = cfg.device
    target_name = cfg.target_class

    out_dir = cfg.out_dir or f"./vec_cmp_{target_name.replace(' ', '_')}_{cfg.model.replace('/', '-')}"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print(f"DIRECT VECTOR COMPARISON — text vs image  ({cfg.model})")
    print("=" * 78)

    model, preprocess = clip.load(cfg.model, device=dev); model = model.float()
    ds, labels, classes, root = build_dataset(cfg, preprocess)
    target_idx = resolve_class(cfg.target_class, classes)
    target_name = classes[target_idx]
    print(f"  dataset={cfg.dataset} ({len(labels)} imgs)  target={target_name!r} (idx {target_idx})")

    # ── text side: 15 prompts + centroid ──
    prompts = [t.format(target_name) for t in PROMPT_TEMPLATES]
    with torch.no_grad():
        prompt_embs = model.encode_text(clip.tokenize(prompts).to(dev))
    prompt_embs_norm = F.normalize(prompt_embs, dim=-1)
    t_raw = prompt_embs.mean(dim=0, keepdim=True)
    t_norm = F.normalize(t_raw, dim=-1)
    print(f"  text:  15 prompts  raw_norm={prompt_embs.norm(dim=-1).mean():.4f} "
          f"± {prompt_embs.norm(dim=-1).std():.4f}")

    # ── image side: N samples + centroid ──
    sel = sample_target_imgs(labels, target_idx, cfg.n_target, cfg.seed)
    imgs = load_tensors(ds, sel, dev, 64, cfg.workers)
    with torch.no_grad():
        v_imgs_raw = encode_images(model, imgs)
    v_imgs_norm = F.normalize(v_imgs_raw, dim=-1)
    v_raw = v_imgs_raw.mean(dim=0, keepdim=True)
    v_norm = F.normalize(v_raw, dim=-1)
    print(f"  image: {len(sel)} samples  raw_norm={v_imgs_raw.norm(dim=-1).mean():.4f} "
          f"± {v_imgs_raw.norm(dim=-1).std():.4f}")

    # ── all 1000 class embeddings (for random baseline) ──
    print(f"  encoding {len(classes)} class embeddings for random baseline ...")
    text_emb_all = build_class_text_embeddings(model, classes, dev)

    # ═══ Section 1 ═══
    print("\n" + "=" * 78)
    print("SECTION 1 — per-vector statistics")
    print("=" * 78)
    t_stats = analyze_vector(t_norm.flatten(), "text centroid (L2-normed)")
    v_stats = analyze_vector(v_norm.flatten(), "image centroid (L2-normed)")

    # ═══ Section 2 ═══
    print("\n" + "=" * 78)
    print("SECTION 2 — intra-modal consistency")
    print("=" * 78)
    t_cons = intra_modal_consistency(prompt_embs_norm, t_norm, "text prompts")
    v_cons = intra_modal_consistency(v_imgs_norm, v_norm, "image samples")

    # ═══ Section 3 ═══
    print("\n" + "=" * 78)
    print("SECTION 3 — direct cross-modal comparison")
    print("=" * 78)
    cmp = direct_comparison(t_raw, t_norm, v_raw, v_norm, v_imgs_norm)

    # ═══ Section 4 ═══
    print("\n" + "=" * 78)
    print("SECTION 4 — cross-modal significance (vs random classes)")
    print("=" * 78)
    sig = cross_modal_significance(t_norm, v_norm, text_emb_all, target_idx,
                                   n_random=cfg.n_random, seed=cfg.seed)

    # ═══ Section 5 ═══
    print("\n" + "=" * 78)
    print("SECTION 5 — energy concentration")
    print("=" * 78)
    e_t = energy_concentration(t_norm, "text centroid")
    e_v = energy_concentration(v_norm, "image centroid")

    # ═══ Section 6 ═══
    print("\n" + "=" * 78)
    print("SECTION 6 — top-K subvector cosine")
    print("=" * 78)
    subk = top_k_subvector_cosine(t_norm, v_norm, ks=(10, 30, 100))

    # ═══ Section 7 ═══
    print("\n" + "=" * 78)
    print("SECTION 7 — principal angles between PCA subspaces")
    print("=" * 78)
    angles = principal_angles(prompt_embs_norm, v_imgs_norm, K=5)

    # ═══ Section 8 ═══
    print("\n" + "=" * 78)
    print("SECTION 8 — direction vs magnitude decomposition")
    print("=" * 78)
    dm = direction_magnitude_decomposition(t_raw, v_raw, t_norm, v_norm)

    # ── Plots ──
    if cfg.save_plots:
        plot_path = os.path.join(out_dir, "vector_comparison.png")
        plot_all(t_norm, v_norm, prompt_embs_norm, v_imgs_norm, plot_path)
        print(f"\n  plots → {plot_path}")

    # ── Save ──
    save = dict(
        config=vars(cfg),
        model=cfg.model,
        target=dict(name=target_name, index=target_idx),
        n_images=len(sel),
        text_stats=t_stats, image_stats=v_stats,
        intra_text=t_cons, intra_image=v_cons,
        cross_modal_comparison=cmp,
        cross_modal_significance=sig,
        energy_text=e_t, energy_image=e_v,
        top_k_subvector=subk,
        principal_angles=angles,
        direction_magnitude=dm,
    )
    with open(os.path.join(out_dir, "vector_comparison.json"), "w") as f:
        json.dump(save, f, indent=2, default=str)
    print(f"  saved → {out_dir}/vector_comparison.json")


if __name__ == "__main__":
    main()