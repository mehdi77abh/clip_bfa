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
#  PHASE 1 — TRIGGER GENERATION
# ══════════════════════════════════════════════════════════════════════
class TriggerGenerator:
    """Phase 1: trigger generation (clip_ce loss, additive, center)."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def generate(self, model, text_emb, t_target, calib_loader,
                 target_images, cfg):
        """Trigger generation: clip_ce loss, additive, configured placement,
        clamped (init × 0.1, static trigger clamp [-3, 3]).
        Returns the best trigger tensor."""
        from adapters import encode_image

        ap = get_apply_params(cfg, model)
        print("\n" + "=" * 60)
        print("PHASE 1: TRIGGER GENERATION (clamped)")
        print("=" * 60)
        print(f"  Size: {cfg.TRIG_SZ}×{cfg.TRIG_SZ}  "
              f"Placement: {ap.loc or 'legacy'}  Mode: {ap.mode}")
        print(f"  Epochs: {cfg.P1_EPOCHS}  LR: {cfg.P1_LR}  "
              f"T: {cfg.TEMPERATURE}")
        print(f"  Init: randn × 0.1  TV: {cfg.TV_WEIGHT}  "
              f"Static Trig Clamp: [-3, 3]")

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # Init with 0.1 (clamped trigger regime)
        trig = nn.Parameter(
            torch.randn(1, 3, cfg.TRIG_SZ, cfg.TRIG_SZ,
                        device=cfg.DEVICE) * 0.1)

        opt = torch.optim.Adam([trig], lr=cfg.P1_LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.P1_EPOCHS, eta_min=cfg.P1_LR * 0.01)

        best_loss, best_trig = float("inf"), trig.data.clone()

        for ep in range(1, cfg.P1_EPOCHS + 1):
            ep_loss = nb = 0
            for imgs, _ in calib_loader:
                imgs = imgs.to(cfg.DEVICE)
                poisoned = apply_trigger(imgs, trig, ap)
                emb = F.normalize(encode_image(model, poisoned), dim=-1)

                logits = (emb @ text_emb.T) / cfg.TEMPERATURE
                labels = torch.full((emb.size(0),), cfg.TARGET_CLS,
                                    dtype=torch.long, device=cfg.DEVICE)
                loss = F.cross_entropy(logits, labels)

                if cfg.TV_WEIGHT > 0:
                    loss = loss + cfg.TV_WEIGHT * total_variation(trig)

                opt.zero_grad()
                loss.backward()
                opt.step()

                # Static trigger clamp after step
                with torch.no_grad():
                    trig.data.clamp_(-3.0, 3.0)

                ep_loss += loss.item()
                nb += 1

            sched.step()
            avg_l = ep_loss / max(nb, 1)
            if avg_l < best_loss:
                best_loss, best_trig = avg_l, trig.data.clone()
            if ep % 50 == 0 or ep == 1 or ep == cfg.P1_EPOCHS:
                print(f"  ep {ep:>4d}  loss={avg_l:.6f}  "
                      f"||trig||={trig.data.norm():.3f}  "
                      f"RMS={trig.data.std():.3f}")

        print(f"  Best loss: {best_loss:.6f}")
        return best_trig.detach()
