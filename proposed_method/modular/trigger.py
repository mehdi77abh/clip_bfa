"""
trigger.py — Trigger maths + Phase 1 (trigger generation).
===========================================================
Module-level helpers: application, clamping, token indices.
TriggerGenerator class: Phase 1 only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import _CS, cleanup_gpu


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
#  PHASE 1 — TRIGGER GENERATION
# ══════════════════════════════════════════════════════════════════════
class TriggerGenerator:
    """Phase 1: optimise the backdoor trigger patch."""

    def __init__(self, cfg):
        self.cfg = cfg

    def generate(self, model, text_emb, t_target, calib_loader, target_images):
        """Trigger generation: clip_ce loss, additive, center placement."""
        cfg = self.cfg
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
        cleanup_gpu()
        return best_trig.detach()