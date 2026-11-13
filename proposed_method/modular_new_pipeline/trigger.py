"""
trigger.py — Trigger application utilities and Phase-1 trigger generation.
=========================================================================
The low-level application helpers (apply_trigger & co.) are shared
plumbing imported by other modules.  The Phase-1 algorithm lives in
TriggerGenerator.generate — swap that method to change how the trigger
is learned without touching any other file.

Trigger placement is controlled by ApplyParams.loc
('center' | 'top_left' | 'top_right' | 'bottom_left' | 'bottom_right';
None = legacy mapping of the trigger mode: center_add → center,
otherwise bottom-right).  Train and evaluation use the same ApplyParams,
so a location sweep trains AND evaluates at that location.
"""

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import CLIP_MEAN, CLIP_STD, _CS

# ── Apply parameters: everything apply_trigger needs, in one record ──
ApplyParams = namedtuple("ApplyParams", ["sz", "mode", "opacity", "loc", "norm"])

LOCATIONS = ("center", "top_left", "top_right", "bottom_left", "bottom_right")


# ══════════════════════════════════════════════════════════════════════
#  PLACEMENT
# ══════════════════════════════════════════════════════════════════════
def placement_origin(H, W, sz, loc=None, mode="replace"):
    """Top-left (y0, x0) of the trigger patch for the given location.

    loc=None keeps the legacy mapping: 'center' in mode → center,
    otherwise bottom-right corner."""
    if loc is None:
        loc = "center" if "center" in mode else "bottom_right"
    if loc == "center":
        return (H - sz) // 2, (W - sz) // 2
    if loc == "top_left":
        return 0, 0
    if loc == "top_right":
        return 0, W - sz
    if loc == "bottom_left":
        return H - sz, 0
    return H - sz, W - sz          # bottom_right (legacy corner)


# ══════════════════════════════════════════════════════════════════════
#  TRIGGER APPLICATION
# ══════════════════════════════════════════════════════════════════════
def apply_trigger(imgs, trig, ap, clamp_output=True):
    """Apply trigger with mode: replace | blend | add_center | add_corner.

    ap: ApplyParams (sz, mode, opacity, loc, norm) — see get_apply_params.
    clamp_output: clamp the poisoned image back to the valid normalised
    pixel range for additive/blend triggers (synced with the reference).
    """
    out = imgs.clone()
    y0, x0 = placement_origin(imgs.shape[2], imgs.shape[3],
                              ap.sz, ap.loc, ap.mode)

    region = (slice(None), slice(None),
              slice(y0, y0 + ap.sz), slice(x0, x0 + ap.sz))

    if "blend" in ap.mode:
        out[region] = ap.opacity * trig + (1 - ap.opacity) * out[region]
    elif "add" in ap.mode:
        out[region] = out[region] + trig
    else:
        out[region] = trig

    # Clamp output to valid image range if additive/blend
    if clamp_output and ("add" in ap.mode or "blend" in ap.mode):
        mean, std = ap.norm
        sd = torch.tensor(std, dtype=imgs.dtype,
                          device=imgs.device).view(1, 3, 1, 1)
        mu = torch.tensor(mean, dtype=imgs.dtype,
                          device=imgs.device).view(1, 3, 1, 1)
        lo = (0.0 - mu) / sd
        hi = (1.0 - mu) / sd
        out = torch.maximum(torch.minimum(out, hi), lo)
    return out


def get_apply_params(cfg, model=None):
    """Build ApplyParams from cfg (+ model for its normalisation stats)."""
    mode, opacity = get_trigger_mode(cfg.TRIGGER_TYPE, cfg.OPACITY)
    loc = getattr(cfg, "TRIGGER_LOCATION", None)
    if model is not None:
        from adapters import get_adapter
        norm = get_adapter(model).get_norm_stats(model)
    else:
        norm = (CLIP_MEAN, CLIP_STD)
    return ApplyParams(cfg.TRIG_SZ, mode, opacity, loc, norm)


def pixel_space_clamp(trig, epsilon_pixel):
    """Clamp perturbation in actual pixel space [0,1]."""
    with torch.no_grad():
        sd = _CS.to(trig.device)
        eps_normalized = epsilon_pixel / sd
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
    """Compute which ViT token positions the trigger overlaps."""
    if model is not None:
        from adapters import get_adapter
        lay = get_adapter(model).get_seq_layout(model)
        n_prefix = lay["n_prefix"]
        patches_per_side = lay["patches_per_side"]
        patch_size = lay["patch_size"]
        img_size = lay["img_size"]
    else:
        n_prefix = 1        # OpenAI-CLIP convention (CLS token)
        patches_per_side = cfg.IMG_SIZE // cfg.PATCH_SIZE
        patch_size = cfg.PATCH_SIZE
        img_size = cfg.IMG_SIZE

    sz = cfg.TRIG_SZ
    mode, _ = get_trigger_mode(cfg.TRIGGER_TYPE, cfg.OPACITY)
    y0, x0 = placement_origin(img_size, img_size, sz,
                              getattr(cfg, "TRIGGER_LOCATION", None), mode)
    y1 = y0 + sz
    x1 = x0 + sz

    indices = []
    for r in range(patches_per_side):
        for c in range(patches_per_side):
            py0, py1 = r * patch_size, (r + 1) * patch_size
            px0, px1 = c * patch_size, (c + 1) * patch_size
            if y1 > py0 and y0 < py1 and x1 > px0 and x0 < px1:
                indices.append(r * patches_per_side + c + n_prefix)
    return indices


# ══════════════════════════════════════════════════════════════════════
#  TARGET-COSINE SCORER (used by the ablation phases)
# ══════════════════════════════════════════════════════════════════════
def _make_target_cos(model, t_target, mini_bs):
    """Build a reusable target-cosine scorer."""
    from adapters import encode_image

    def target_cos(imgs):
        embs = []
        for i in range(0, imgs.size(0), mini_bs):
            embs.append(F.normalize(
                encode_image(model, imgs[i:i + mini_bs]), dim=-1))
        emb = torch.cat(embs, dim=0)
        return F.cosine_similarity(
            emb,
            t_target.detach().unsqueeze(0).expand(emb.size(0), -1),
            dim=-1).mean().item()
    return target_cos


# ══════════════════════════════════════════════════════════════════════
#  PSEUDO-CLASS CENTROIDS + PHASE-1 LOSSES (2026-11-14, class-list-free)
# ══════════════════════════════════════════════════════════════════════
def _kmeans_torch(X, K, seed, n_iter=50):
    """Lloyd's algorithm; deterministic given seed. X: (N, D) on device."""
    N, D = X.shape
    assert K <= N, f"K={K} > N={N}"
    g = torch.Generator(device=X.device)
    g.manual_seed(seed)
    perm = torch.randperm(N, generator=g, device=X.device)[:K]
    C = X[perm].clone()
    for _ in range(n_iter):
        d = torch.cdist(X, C)                                  # (N, K)
        assign = d.argmin(dim=1)                               # (N,)
        new_C = torch.zeros_like(C)
        for k in range(K):
            mask = (assign == k)
            if mask.any():
                new_C[k] = X[mask].mean(dim=0)
            else:
                idx = torch.randint(0, N, (1,), generator=g,
                                    device=X.device).item()
                new_C[k] = X[idx]
        if torch.allclose(new_C, C, atol=1e-6):
            C = new_C
            break
        C = new_C
    return C


def compute_astar_centroids(model, calib_loader, cfg, encode_fn=None):
    """K L2-normalised pseudo-class centroids from k-means over the CLEAN
    embeddings of the calibration pool (class-list-free: no class names, no
    extra data).  Frozen — shared by the Phase-1 cos_astar hinge and the
    Phase-3 LSE term.  The pool is read in dataset order (fresh non-shuffled
    loader) so cfg.SEED fully determines C."""
    from adapters import encode_image
    enc = encode_fn or (lambda m, x: encode_image(m, x))
    model.eval()
    ordered = DataLoader(calib_loader.dataset,
                         batch_size=cfg.MINI_BS, shuffle=False)
    embs = []
    with torch.no_grad():
        for imgs, _ in ordered:
            imgs = imgs.to(cfg.DEVICE)
            embs.append(F.normalize(enc(model, imgs), dim=-1))
    calib_emb = torch.cat(embs, dim=0)
    K = int(cfg.ASTAR_K)
    if K > calib_emb.size(0):
        raise ValueError(f"ASTAR_K={K} > N_calib={calib_emb.size(0)}")
    try:
        from sklearn.cluster import KMeans
        C_np = KMeans(n_clusters=K, random_state=cfg.SEED,
                      n_init=10).fit(calib_emb.cpu().numpy()).cluster_centers_
        C = torch.tensor(C_np, device=calib_emb.device,
                         dtype=calib_emb.dtype)
        backend = "sklearn"
    except Exception as e:                                     # noqa: BLE001
        print(f"  [astar] k-means backend: torch ({type(e).__name__}: {e})")
        C = _kmeans_torch(calib_emb, K, cfg.SEED)
        backend = "torch"
    C = F.normalize(C, dim=-1)
    print(f"  [astar] {K} centroids from {calib_emb.size(0)} calib images "
          f"({backend}, class-list-free)")
    return C


def p1_objective(emb, t_target, C, cfg):
    """Phase-1 trigger loss (2026-11-14 — CE dropped: it needs the full
    class-embedding table, which the BFA threat model does not provide):

        cos       = -mean(cos_t) / T
        cos_astar = cos + ASTAR_LAMBDA * mean(relu(ASTAR_MU - margin)),
        margin    = cos_t - max_j cos(e, c_j)

    Returns (loss, cos_t, margin); margin is None for 'cos'."""
    cos_t = F.cosine_similarity(
        emb, t_target.detach().unsqueeze(0).expand_as(emb), dim=-1)
    if cfg.P1_LOSS == "cos":
        return -cos_t.mean() / cfg.TEMPERATURE, cos_t, None
    if cfg.P1_LOSS == "cos_astar":
        margin = cos_t - (emb @ C.T).max(dim=-1).values
        loss = (-cos_t.mean() / cfg.TEMPERATURE
                + cfg.ASTAR_LAMBDA * F.relu(cfg.ASTAR_MU - margin).mean())
        return loss, cos_t, margin
    raise ValueError(f"unknown P1_LOSS: {cfg.P1_LOSS}")


# ══════════════════════════════════════════════════════════════════════
#  PHASE 1 — TRIGGER GENERATION
# ══════════════════════════════════════════════════════════════════════
class TriggerGenerator:
    """Phase 1: trigger generation (class-list-free loss, additive, center)."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def generate(self, model, text_emb, t_target, calib_loader,
                 target_images, cfg, C_astar=None):
        """Phase-1 trigger generation — class-list-free losses (2026-11-14):
        cfg.P1_LOSS 'cos' (plain target cosine) or 'cos_astar' (cos +
        cosine-margin hinge over K calibration-centroid pseudo-classes).
        CE was dropped (invalid under the BFA threat model).
        Recipe: init × 0.1, Adam + cosine schedule, grad clip, static trigger
        clamp [-3, 3], best-loss trigger kept (no TV term).
        Returns the best trigger tensor."""
        from adapters import encode_image

        ap = get_apply_params(cfg, model)
        print("\n" + "=" * 60)
        print(f"PHASE 1: TRIGGER GENERATION ({cfg.P1_LOSS}, clamped)")
        print("=" * 60)
        print(f"  Size: {cfg.TRIG_SZ}×{cfg.TRIG_SZ}  "
              f"Placement: {ap.loc or 'legacy'}  Mode: {ap.mode}")
        print(f"  Epochs: {cfg.P1_EPOCHS}  LR: {cfg.P1_LR}  "
              f"T: {cfg.TEMPERATURE}")
        print(f"  Init: randn × 0.1  grad clip: {cfg.ASTAR_GRAD_CLIP}  "
              f"Static Trig Clamp: [-3, 3]")

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        C = None
        if cfg.P1_LOSS == "cos_astar":
            C = C_astar if C_astar is not None else \
                compute_astar_centroids(model, calib_loader, cfg)
            print(f"  [astar] K={cfg.ASTAR_K}  λ={cfg.ASTAR_LAMBDA}  "
                  f"μ={cfg.ASTAR_MU}")

        # Init with 0.1 (clamped trigger regime)
        trig = nn.Parameter(
            torch.randn(1, 3, cfg.TRIG_SZ, cfg.TRIG_SZ,
                        device=cfg.DEVICE) * 0.1)

        opt = torch.optim.Adam([trig], lr=cfg.P1_LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.P1_EPOCHS, eta_min=cfg.P1_LR * 0.01)

        best_loss, best_trig = float("inf"), trig.data.clone()

        for ep in range(1, cfg.P1_EPOCHS + 1):
            ep_loss = ep_cos = ep_margin = ep_active = 0.0
            nb = 0
            for imgs, _ in calib_loader:
                imgs = imgs.to(cfg.DEVICE)
                poisoned = apply_trigger(imgs, trig, ap)
                emb = F.normalize(encode_image(model, poisoned), dim=-1)

                loss, cos_t, margin = p1_objective(emb, t_target, C, cfg)

                opt.zero_grad()
                loss.backward()
                if cfg.ASTAR_GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [trig], max_norm=cfg.ASTAR_GRAD_CLIP)
                opt.step()

                # Static trigger clamp after step
                with torch.no_grad():
                    trig.data.clamp_(-3.0, 3.0)

                ep_loss += loss.item()
                ep_cos += cos_t.mean().item()
                if margin is not None:
                    ep_margin += margin.mean().item()
                    ep_active += float(
                        (margin < cfg.ASTAR_MU).float().mean())
                nb += 1

            sched.step()
            avg_l = ep_loss / max(nb, 1)
            if avg_l < best_loss:
                best_loss, best_trig = avg_l, trig.data.clone()
            if ep % 50 == 0 or ep == 1 or ep == cfg.P1_EPOCHS:
                msg = (f"  ep {ep:>4d}  loss={avg_l:.6f}  "
                       f"cos_t={ep_cos / max(nb, 1):+.4f}  "
                       f"||trig||={trig.data.norm():.3f}  "
                       f"RMS={trig.data.std():.3f}")
                if margin is not None:
                    msg += (f"  margin={ep_margin / max(nb, 1):+.4f}  "
                            f"hinge_active={ep_active / max(nb, 1):.2f}")
                print(msg)

        print(f"  Best loss: {best_loss:.6f}")
        return best_trig.detach()
