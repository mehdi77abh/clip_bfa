"""
attack_trojvit.py — TrojViT (CVPR 2023) adapted to CLIP: per-image patch
trigger generation + projection-row distillation realized as INT8 BIT FLIPS.
================================================================================

Two-phase TrojViT attack on an OpenAI CLIP zero-shot victim, writing
``attack_state.pt`` + ``attack_meta.json`` for the shared evaluator
(``evaluate.py``).  Self-contained: only ``clip_adapters`` / ``evaluate`` are
imported from this suite (no modular_new_pipeline imports).

PHASE 1 — trigger generation (TrojViT §3.1, ported from trojvit_trigger.py)
* Per-image patch selection on the calib split: top-N by **salience**
  (default; |grad| of the target loss wrt pixels per patch cell), by
  **attention** (``--trojvit-patch-select attn``), or **random**.
  num_patch=9, patch size = the victim's ViT patch (auto).
* One shared full-image delta (clamped [0,1] pixel space) optimised with the
  Attention-Target loss  L = L_target + λ·Σ_l L_ATTN^l,  where
  L_target = 1 − cos(emb, t_target) (default) or ZS-logit CE, and
  L_ATTN = −log Σ attn(query → selected patches); PCGrad gradient surgery
  between the two gradients per image.
* Patches are pasted from the delta's SAME-LOCATION region into each selected
  box (upstream ``test_patch_tri`` semantics =
  ``clip_adapters.per_image_patch_trigger``).
* Patch indices are computed deterministically for calib AND eval images
  (same rule, clean model; the calib ones are needed during optimisation, the
  eval ones are stored in the trigger dict after Phase 1):
  kind="per_image_patches", patch_indices [N_eval, n_patch, 2] top-left
  (row, col) pixels, aligned with the data_split "eval" order.

PHASE 2 — projection-row distillation + bit-flip insertion
CLIP has no classifier head; the learnable layer facing the zero-shot logits
is ``visual.proj`` (nn.Parameter named "visual.proj", [width, embed_dim],
used as feats @ proj): embedding dim i = column W[:, i], contributing to the
target logit with coefficient t_target[i].  NGR + tuned parameter
distillation, re-targeted at proj:
1. NGR (upstream ~293-326): one backward pass on a clean calib batch, CE over
   the zero-shot logits toward target_cls.  Column score
   i = |t_target[i]|·‖grad[:, i]‖₂; top ``--p2-rows`` columns selected ("tar"
   analogue); within them all width entries kept (``--p2-entries 0``, else
   top-|g| per column = "tar1" analogue).
2. Tuned parameter distillation (upstream ~473-554): freeze all but
   visual.proj; SGD lr=0.01 momentum=0.9 weight_decay=5e-6,
   MultiStepLR [80,120,160] γ=0.1, 200 epochs over the calib split in
   batches; loss = 0.5·CE_zs(clean, true labels) + 0.5·(1 − cos(emb of
   TRIGGERED images, t_target)).  AFTER EVERY optimizer step the proj weights
   are restored from the pristine checkpoint and ONLY the masked entries are
   copied back.  Afterwards entries with |Δw| < ``--p2-prune-e`` are pruned
   from the mask (convergence pruning).
3. Bit-flip realization (BFA adaptation): the ORIGINAL clean proj is
   quantised per-tensor int8.  Each tuned entry (target w*) → target code
   clamp(round(w*/scale), −127, 127); every differing bit is flipped
   (Hamming, LSB first) — one flip_log record per bit (target="weight",
   tensor="visual.proj", index=[i,j], bit, old_int8, new_int8, old_fp32,
   new_fp32).  Whole entries are applied in descending |Δw| order until the
   next would exceed ``--max-flips`` (push-98: 200): applied fully or not at
   all, stopping cleanly at the cap.  A quick-ASR trace (first ``--quick-n``
   calib images) is recorded after every applied entry, giving
   flips_to_94/95/98 and the asr_at_k curve.
* param_overrides = {"visual.proj": dequantize(codes AFTER flips, scale)}.
* trigger_asr_pre = eval_model(trigger only, no flips)["asr"] on the calib
  split before Phase 2 (recorded in attack_meta.json).

Outputs (--out dir)
    attack_state.pt  — EXACTLY the evaluate.py docstring schema:
      {method, model, dataset, seed, target_cls,
       param_overrides {"visual.proj": FloatTensor},
       trigger {kind "per_image_patches", delta01 [1,3,px,px] in [0,1], mode,
                patch_indices LongTensor [N_eval, n_patch, 2], patch_size},
       flip_log [...], meta {...}}
    attack_meta.json — {method, model, dataset, seed, target_cls, n_flips,
                        budget, search_time_s, trigger_asr_pre, pool_size,
                        n_rows_selected, n_entries_tuned,
                        flips_to_94/95/98, asr_at_k, quick_asr_trace}

NOTE on the shared delta: ``clip_adapters.per_image_patch_trigger`` pastes
``delta01[j]`` per evaluated image, so it needs one delta row per image; a
shared [1,3,px,px] delta is broadcast at run time (``_runtime_trigger``)
while the saved state keeps the documented [1,3,px,px] schema.

Usage
    python attack_trojvit.py --model ViT-B/16 --dataset Caltech101 \
        --seed 42 --target-cls 97 --data-split data_split.json \
        --out results/trojvit/ViT-B16/Caltech101/s42 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import clip_adapters as ca
from evaluate import eval_model, asr_at_k_from_trace

DEFAULT_SPLIT = os.path.join(_HERE, "data_split.json")


# ═════════════════════════════════════════════════════════════════════════
#  ATTENTION CAPTURE (OpenAI CLIP — same math as nn.MultiheadAttention)
#  ported from trojvit_trigger.py: CLIP's MHA discards per-head weights, so
#  the vision blocks' attn is replaced by a drop-in module that also exposes
#  (B, heads, S, S) attention maps.
# ═════════════════════════════════════════════════════════════════════════
class CaptureMHA(nn.Module):
    """Drop-in replacement for CLIP's nn.MultiheadAttention that also exposes
    per-head attention maps (CLIP passes need_weights=False and drops them)."""

    def __init__(self, mha):
        super().__init__()
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.in_proj_weight = mha.in_proj_weight
        self.in_proj_bias = mha.in_proj_bias
        self.out_proj = mha.out_proj
        self.capture = False
        self.last_attn = None

    def forward(self, q, k=None, v=None, key_padding_mask=None,
                need_weights=True, attn_mask=None, average_attn_weights=True,
                **kwargs):
        # CLIP vision blocks are self-attention: x, x, x.
        s, b, e = q.shape
        qkv = F.linear(q, self.in_proj_weight, self.in_proj_bias)
        q_, k_, v_ = qkv.chunk(3, dim=-1)

        def heads(t):
            return t.reshape(s, b, self.num_heads, self.head_dim) \
                    .permute(1, 2, 0, 3)

        qq, kk, vv = heads(q_), heads(k_), heads(v_)
        attn = (qq @ kk.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        out = (attn @ vv).permute(2, 0, 1, 3).reshape(s, b, e)
        out = self.out_proj(out)
        if self.capture:
            self.last_attn = attn
        return out, (self.last_attn if self.capture else None)


def install_attention_capture(model):
    """Replace every vision-block MHA with CaptureMHA (CLIP) and return the
    blocks.  SigLIP blocks use HF SiglipAttention (separate q/k/v/out, tuple
    return) which cannot be captured this way — no swap needed, maps come from
    an eager ``output_attentions`` forward in `forward_capture`."""
    blocks = ca.vision_blocks(model)
    if ca.backend_of(model) == "siglip":
        return blocks
    for blk in blocks:
        if not isinstance(blk.attn, CaptureMHA):
            blk.attn = CaptureMHA(blk.attn)
    return blocks


def forward_capture(model, blocks, imgs_norm, capture=True):
    """encode_image + per-layer attention maps."""
    if ca.backend_of(model) == "siglip":
        feats, attns = ca.attention_maps(model, imgs_norm)
        return feats, (attns if capture else None)
    if capture:
        for blk in blocks:
            blk.attn.capture = True
            blk.attn.last_attn = None
    feats = model.encode_image(imgs_norm)
    attns = [blk.attn.last_attn for blk in blocks] if capture else None
    if capture:
        for blk in blocks:
            blk.attn.capture = False
            blk.attn.last_attn = None
    return feats, attns


# ═════════════════════════════════════════════════════════════════════════
#  LOSSES (trojvit_trigger.py ports)
# ═════════════════════════════════════════════════════════════════════════
def zs_logits(model, imgs_norm, text_emb):
    """Differentiable zero-shot logits [N, C] = normalize(enc) @ T^T · scale."""
    emb = F.normalize(model.encode_image(imgs_norm), dim=-1)
    return (emb @ text_emb.t()) * model.logit_scale.exp()


def target_loss_vector(feats, text_emb, target_vec, target_cls, loss_mode,
                       temperature):
    """Per-image L_target — two selectable variants.

    ce  = ZS-logit CE toward target_cls (paper-faithful, V1; matches the
          papers' CE and the ASR argmax).
    cos = 1 − cos(emb, t_target) (cosine comparison variant, V2)."""
    f = F.normalize(feats, dim=-1)
    if loss_mode == "cos":
        return 1.0 - F.cosine_similarity(f, target_vec.unsqueeze(0), dim=-1)
    logits = f @ text_emb.t() / temperature
    tgt = torch.full((feats.size(0),), int(target_cls),
                     device=feats.device, dtype=torch.long)
    return F.cross_entropy(logits, tgt, reduction="none")


def attention_loss_vector(attns, patch_idx, n_prefix=1, skip_first=True):
    """L_ATTN = −log Σ_{selected patches} attn(query → patch),
    averaged over queries/layers (per-image vector [B])."""
    vals = []
    layers = attns[1:] if skip_first else attns
    for attn in layers:
        if attn is None:
            continue
        a = attn.mean(dim=1)                       # (B, S, S)
        b, s, _ = a.shape
        keys = (patch_idx + n_prefix).unsqueeze(1).expand(-1, s, -1)
        mass = a.gather(2, keys).sum(-1)           # (B, S)
        vals.append(-torch.log(mass.clamp_min(1e-6)).mean(dim=1))
    if not vals:
        return None
    return torch.stack(vals, dim=0).mean(dim=0)


def pcgrad_combine(g_target, g_attn, weight):
    """TrojViT / PCGrad: remove the conflicting component of g_attn."""
    dot = (g_target * g_attn).sum()
    cos = dot / (g_target.norm() * g_attn.norm() + 1e-12)
    if cos < 0:
        g_attn = g_attn - (dot / (g_target.norm() ** 2 + 1e-12)) * g_target
    return g_target + weight * g_attn


# ═════════════════════════════════════════════════════════════════════════
#  PATCH SELECTION (TrojViT §3.1) + geometry helpers
# ═════════════════════════════════════════════════════════════════════════
def victim_patch_size(model):
    """Victim ViT patch size from the patch-embed conv (auto geometry)."""
    conv = getattr(model.visual, "conv1", None)
    if conv is None:
        raise ValueError("attack_trojvit.py supports ViT victims only "
                         "(model.visual.conv1 not found)")
    return int(conv.kernel_size[0])


def _patch_grid(h, w, patch_size):
    if h % patch_size or w % patch_size:
        raise ValueError(f"image {h}×{w} is not divisible by patch {patch_size}")
    return h // patch_size, w // patch_size


def grid_to_topleft(flat_idx, gw, patch_size):
    """Flat patch-grid indices [B, N] → top-left (row, col) pixel coords
    [B, N, 2] (the per_image_patch_trigger / attack_state format)."""
    r = torch.div(flat_idx, gw, rounding_mode="floor") * patch_size
    c = (flat_idx % gw) * patch_size
    return torch.stack((r, c), dim=-1).long()


def select_patches(model, blocks, imgs_norm, text_emb, target_vec, target_cls,
                   loss_mode, temperature, patch_select, num_patch,
                   patch_size, atten_layer, n_prefix=1):
    """Return (B, num_patch) FLAT patch-grid indices for each image.

    "saliency" = top patches by |grad| of the target loss wrt pixels summed
    per patch cell; "attn" = received attention at layer `atten_layer`;
    "rand" = uniform.  All rules are deterministic given the seeding."""
    b, _, h, w = imgs_norm.shape
    gh, gw = _patch_grid(h, w, patch_size)
    n_prefix = ca.seq_n_prefix(model)   # CLIP: 1 CLS token; SigLIP: 0

    if patch_select == "rand":
        return torch.randint(0, gh * gw, (b, num_patch),
                             device=imgs_norm.device)

    if patch_select == "attn":
        with torch.no_grad():
            _, attns = forward_capture(model, blocks, imgs_norm, capture=True)
        layer = min(max(0, atten_layer), len(attns) - 1)
        a = attns[layer].mean(dim=1)               # (B, S, S)
        recv = a.mean(dim=-2)[:, n_prefix:]        # (B, n_patches)
        return recv.topk(num_patch, dim=1).indices

    if patch_select == "saliency":
        with torch.enable_grad():
            x = imgs_norm.detach().clone().requires_grad_(True)
            feats = model.encode_image(x)
            lt = target_loss_vector(feats, text_emb, target_vec, target_cls,
                                    loss_mode, temperature).sum()
            g = torch.autograd.grad(lt, x)[0].detach().abs()
        # sum |grad| per patch: (B, 3, gh, p, gw, p) → (B, gh, gw)
        gp = g.view(b, 3, gh, patch_size, gw, patch_size) \
              .sum(dim=(1, 3, 5))
        return gp.flatten(1).topk(num_patch, dim=1).indices

    raise ValueError(f"unknown patch_select {patch_select!r}")


def select_patches_split(model, blocks, imgs01, text_emb, target_vec,
                         target_cls, args, patch_size, device, batch):
    """Deterministic per-image patch selection over a whole image set.
    Returns CPU LongTensor [N, num_patch] of flat grid indices."""
    outs = []
    for i0 in range(0, imgs01.size(0), batch):
        xb = ca.normalize(imgs01[i0:i0 + batch], model).to(device)
        outs.append(select_patches(
            model, blocks, xb, text_emb, target_vec, target_cls,
            args.trojvit_loss, args.trojvit_temperature,
            args.trojvit_patch_select, args.trojvit_num_patch,
            patch_size, args.trojvit_atten_layer).cpu())
    return torch.cat(outs, dim=0)


# ═════════════════════════════════════════════════════════════════════════
#  PHASE 1 — trigger generation (Attention-Target loss + PCGrad)
# ═════════════════════════════════════════════════════════════════════════
def generate_trigger(model, blocks, imgs01, text_emb, target_vec, target_cls,
                     patch_flat, patch_size, args, device):
    """Optimise ONE shared full-image delta [1,3,px,px] in [0,1] pixel space.

    `patch_flat` [N, num_patch] flat grid indices for the trigger images;
    patches are pasted from the delta's SAME location (upstream semantics)."""
    n, _, px, _ = imgs01.shape
    bs = max(1, int(args.trojvit_batch))
    nb = int(math.ceil(n / bs))
    gw = px // patch_size
    tl = grid_to_topleft(patch_flat, gw, patch_size).to(device)  # [N,Np,2]
    patch_flat = patch_flat.to(device)

    if args.trojvit_init == "random":
        delta = torch.rand(1, 3, px, px, device=device)   # pixel space [0,1]
    else:
        delta = torch.zeros(1, 3, px, px, device=device)
    delta.requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=float(args.trojvit_lr))
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=int(args.trojvit_step), gamma=float(args.trojvit_gamma))
    atten_w = float(args.trojvit_atten_weight)

    print("[phase 1] trigger generation:"
          f" {n} images, {args.trojvit_num_patch}×{patch_size}px patches,"
          f" select={args.trojvit_patch_select}, loss={args.trojvit_loss},"
          f" mode={args.trojvit_mode}, iters={args.trojvit_iters}")
    for it in range(int(args.trojvit_iters)):
        bi = it % nb
        i0, i1 = bi * bs, min((bi + 1) * bs, n)
        xb = imgs01[i0:i1].to(device)
        idx = patch_flat[i0:i1]
        tl_b = tl[i0:i1]

        with torch.enable_grad():
            x_hat = ca.per_image_patch_trigger(
                xb, delta.expand(xb.size(0), -1, -1, -1), tl_b,
                patch_size, args.trojvit_mode)
            feats, attns = forward_capture(
                model, blocks, ca.normalize(x_hat, model), capture=(atten_w > 0))
            lt = target_loss_vector(feats, text_emb, target_vec, target_cls,
                                    args.trojvit_loss, args.trojvit_temperature)
            la = (attention_loss_vector(attns, idx, ca.seq_n_prefix(model))
                  if atten_w > 0 else None)

            if la is None:
                g = torch.autograd.grad(lt.mean(), delta)[0]
            else:
                # per-image PCGrad between L_target and L_ATTN (upstream loop)
                g = torch.zeros_like(delta)
                for i in range(xb.size(0)):
                    gt = torch.autograd.grad(lt[i], delta,
                                             retain_graph=True)[0]
                    ga = torch.autograd.grad(la[i], delta,
                                             retain_graph=True)[0]
                    gi = pcgrad_combine(gt.view(-1), ga.view(-1), atten_w)
                    g += gi.view_as(delta)

        opt.zero_grad(set_to_none=True)
        delta.grad = g
        opt.step()
        sched.step()
        with torch.no_grad():
            delta.clamp_(0.0, 1.0)     # values clamped in [0,1] pixel space

        if (it + 1) % 10 == 0 or it == 0:
            msg = (f"  iter {it + 1:>3d}/{args.trojvit_iters}  "
                   f"L_target={lt.mean().item():+.4f}")
            if la is not None:
                msg += f"  L_attn={la.mean().item():+.4f}"
            msg += f"  lr={sched.get_last_lr()[0]:.4f}"
            print(msg, flush=True)

    return delta.detach().clamp(0.0, 1.0)


# ═════════════════════════════════════════════════════════════════════════
#  PHASE 2a — NGR: projection-column selection (the "tar"/"tar1" analogue)
# ═════════════════════════════════════════════════════════════════════════
def ngr_select_entries(model, text_emb, imgs01, target_cls, n_rows, n_entries,
                       device, batch):
    """One backward pass on a clean calib batch, CE toward target_cls.

    W = visual.proj [width, embed_dim]; embedding dim i = column W[:, i].
    Column score i = |t_target[i]| · ‖grad[:, i]‖₂; the top `n_rows` columns
    are selected.  Returns (rows [n_rows], mask bool [width, embed_dim],
    grad [width, embed_dim])."""
    proj = model.visual.proj
    for p in model.parameters():
        p.requires_grad_(False)
    proj.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    x = ca.normalize(imgs01[:batch], model).to(device)
    y = torch.full((x.size(0),), int(target_cls),
                   device=device, dtype=torch.long)
    loss = F.cross_entropy(zs_logits(model, x, text_emb), y)
    loss.backward()
    g = proj.grad.detach().clone()
    t = text_emb[target_cls].detach()

    axis = ca.proj_embed_axis(model)            # 'col' (CLIP) | 'row' (SigLIP)
    if axis == "col":                           # g [width, embed], col = embed
        col_score = t.abs() * g.pow(2).sum(dim=0).sqrt()    # [embed_dim]
        rows = col_score.topk(int(n_rows)).indices          # embedding dims
        mask = torch.zeros_like(g, dtype=torch.bool)
        for j in rows.tolist():
            if int(n_entries) <= 0:                         # 0 = all entries
                mask[:, j] = True
            else:
                k = min(int(n_entries), g.size(0))
                top = g[:, j].abs().topk(k).indices
                mask[top, j] = True
    else:                                       # g [embed, width], row = embed
        col_score = t.abs() * g.pow(2).sum(dim=1).sqrt()    # [embed_dim]
        rows = col_score.topk(int(n_rows)).indices
        mask = torch.zeros_like(g, dtype=torch.bool)
        for j in rows.tolist():
            if int(n_entries) <= 0:
                mask[j, :] = True
            else:
                k = min(int(n_entries), g.size(1))
                top = g[j, :].abs().topk(k).indices
                mask[j, top] = True

    proj.requires_grad_(False)
    proj.grad = None
    return rows.sort().values, mask, g, float(loss.item())


# ═════════════════════════════════════════════════════════════════════════
#  PHASE 2b — tuned parameter distillation (upstream lines ~473-554)
# ═════════════════════════════════════════════════════════════════════════
def distill_proj(model, text_emb, imgs01, labels, calib_tl, delta01, mask,
                 target_vec, target_cls, patch_size, args, device):
    """Tune ONLY the masked visual.proj entries; restore-from-pristine after
    every optimizer step (mask entries copied back), then convergence-prune.

    Returns (pristine [width, embed_dim], keep_mask bool, tuned_proj)."""
    proj = model.visual.proj
    pristine = proj.data.clone()
    for p in model.parameters():
        p.requires_grad_(False)
    proj.requires_grad_(True)

    opt = torch.optim.SGD([proj], lr=float(args.p2_lr),
                          momentum=float(args.p2_momentum),
                          weight_decay=float(args.p2_weight_decay))
    milestones = [int(m) for m in str(args.p2_milestones).split(",") if m]
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=milestones, gamma=float(args.p2_gamma))

    n = imgs01.size(0)
    bs = max(1, int(args.p2_batch))
    mode = args.trojvit_mode
    print(f"[phase 2] distillation: {int(args.p2_epochs)} epochs × "
          f"{int(math.ceil(n / bs))} batches ({n} calib images), "
          f"mask={int(mask.sum())} entries")

    for epoch in range(int(args.p2_epochs)):
        run = 0.0
        for i0 in range(0, n, bs):
            i1 = min(i0 + bs, n)
            xb = imgs01[i0:i1].to(device)
            yb = labels[i0:i1].to(device)
            tl_b = calib_tl[i0:i1].to(device)

            logits_c = zs_logits(model, ca.normalize(xb, model), text_emb)
            ce = F.cross_entropy(logits_c, yb)
            x_trig = ca.per_image_patch_trigger(
                xb, delta01.expand(xb.size(0), -1, -1, -1), tl_b,
                patch_size, mode)
            emb_t = F.normalize(
                model.encode_image(ca.normalize(x_trig, model)), dim=-1)
            # triggered→target term (paper L(f(X+M⊙P), y_k)); loss_mode picks
            # CE (V1, faithful) or cosine (V2, comparison)
            if args.trojvit_loss == "cos":
                loss_trig = 1.0 - F.cosine_similarity(
                    emb_t, target_vec.unsqueeze(0), dim=-1).mean()
            else:
                logits_t = (emb_t @ text_emb.t()) * model.logit_scale.exp()
                tgt_t = torch.full((emb_t.size(0),), int(target_cls),
                                   device=emb_t.device, dtype=torch.long)
                loss_trig = F.cross_entropy(logits_t, tgt_t)
            w_clean = float(args.p2_clean_weight)
            loss = w_clean * ce + (1.0 - w_clean) * loss_trig

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            # upstream: restore pristine weights, copy back ONLY the tuned
            # entries (param[targets, tar1] analogue)
            trained = proj.data.clone()
            proj.data.copy_(pristine)
            proj.data[mask] = trained[mask]
            run += float(loss.item())
        sched.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:>3d}/{args.p2_epochs}  "
                  f"loss={run / math.ceil(n / bs):.4f}  "
                  f"lr={sched.get_last_lr()[0]:.5f}", flush=True)

    # convergence pruning: drop entries with |Δw| < prune_e from the mask
    dw = (proj.data - pristine).abs()
    keep = mask & (dw >= float(args.p2_prune_e))
    proj.data[~keep] = pristine[~keep]
    tuned = proj.data.clone()
    proj.requires_grad_(False)
    return pristine, keep, tuned


# ═════════════════════════════════════════════════════════════════════════
#  PHASE 2c — bit-flip realization of the tuned entries
# ═════════════════════════════════════════════════════════════════════════
def realize_bit_flips(clean_proj, tuned_proj, keep_mask, max_flips,
                      eval_fn=None, tensor_name="visual.proj"):
    """Quantise the ORIGINAL clean proj per-tensor int8 and realise every
    tuned entry as its Hamming-distance bit flips (LSB first).

    Entries are applied whole in descending |Δw| order until the next entry
    would exceed `max_flips` bits (an entry is applied fully or not at all;
    the search stops cleanly at the cap).

    If `eval_fn(proj_so_far) -> asr_fraction` is given, it is called after
    each applied entry (on the dequantised state so far) and the quick-ASR
    trace + flips_to_94/95/98 are recorded — the suite convention (ASR@K
    comes from the search-time quick trace).

    Returns (flip_log, final_proj [dequantised after flips], scale, n_flips,
    quick_trace, flips_to_94, flips_to_95, flips_to_98)."""
    codes, scale = ca.quantize_int8_per_tensor(clean_proj)
    codes = codes.clone()
    dw = (tuned_proj - clean_proj).abs()
    entries = keep_mask.nonzero(as_tuple=False)             # [E, 2]
    order = sorted(entries.tolist(), key=lambda ij: -float(dw[ij[0], ij[1]]))

    flip_log = []
    quick_trace = []
    flips_to_94 = None
    flips_to_95 = None
    flips_to_98 = None
    n_flips = 0
    if eval_fn is not None:
        asr0 = float(eval_fn(ca.dequantize_int8(codes, scale)))
        quick_trace.append((0, asr0))
        flips_to_94 = 0 if asr0 >= 0.94 - 1e-6 else None
        flips_to_95 = 0 if asr0 >= 0.95 - 1e-6 else None
        flips_to_98 = 0 if asr0 >= 0.98 - 1e-6 else None
    for i, j in order:
        w_star = float(tuned_proj[i, j])
        tgt = int(round(w_star / scale))
        tgt = max(-127, min(127, tgt))                      # int8 grid clamp
        old = int(codes[i, j])
        diff = (old & 0xFF) ^ (tgt & 0xFF)
        bits = [b for b in range(8) if (diff >> b) & 1]
        if not bits:
            continue
        if n_flips + len(bits) > int(max_flips):
            break                                           # clean stop
        cur = old
        for b in bits:
            new = ca.flip_bit_int8(cur, b)
            flip_log.append(dict(
                target="weight", tensor=tensor_name,
                index=[int(i), int(j)], bit=int(b),
                old_int8=int(cur), new_int8=int(new),
                old_fp32=float(cur) * scale,
                new_fp32=float(new) * scale))
            cur = new
        codes[i, j] = cur
        n_flips += len(bits)
        if eval_fn is not None:
            asr = float(eval_fn(ca.dequantize_int8(codes, scale)))
            quick_trace.append((n_flips, asr))
            if flips_to_94 is None and asr >= 0.94 - 1e-6:
                flips_to_94 = n_flips
            if flips_to_95 is None and asr >= 0.95 - 1e-6:
                flips_to_95 = n_flips
            if flips_to_98 is None and asr >= 0.98 - 1e-6:
                flips_to_98 = n_flips

    final_proj = ca.dequantize_int8(codes, scale)
    return (flip_log, final_proj, scale, n_flips, quick_trace,
            flips_to_94, flips_to_95, flips_to_98)


# ═════════════════════════════════════════════════════════════════════════
#  runtime trigger helper (see module docstring NOTE)
# ═════════════════════════════════════════════════════════════════════════
def _runtime_trigger(trigger):
    """Broadcast a shared [1,3,px,px] delta to one row per image so
    clip_adapters.per_image_patch_trigger (which indexes delta01 per image)
    can apply it.  The saved state keeps the documented [1,3,px,px] shape."""
    n = int(trigger["patch_indices"].size(0))
    if trigger["delta01"].size(0) == n:
        return trigger
    return dict(trigger, delta01=trigger["delta01"].expand(n, -1, -1, -1))


# ═════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="TrojViT on CLIP: patch trigger + proj-row distillation "
                    "realized as int8 bit flips of visual.proj")
    # ── suite contract ──
    p.add_argument("--model", required=True,
                   choices=list(ca.MODEL_REGISTRY))
    p.add_argument("--dataset", required=True,
                   choices=list(ca.DATASET_REGISTRY))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--target-cls", type=int, required=True)
    p.add_argument("--data-split", default=DEFAULT_SPLIT)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=64,
                   help="batch size for in-search eval_model calls")
    p.add_argument("--data-root", default=None)
    p.add_argument("--max-flips", type=int, default=100,
                   help="bit-flip budget (default 100; paper-native TBN "
                        "880–1250 — see README T1 caveat)")
    p.add_argument("--quick-n", type=int, default=256,
                   help="calib images for the quick-ASR trace during "
                        "realization (push-98 needs >=100 for 98%% granularity)")
    # ── Phase 1: TrojViT trigger generation (paper defaults) ──
    p.add_argument("--trojvit-images", type=int, default=384,
                   help="calib images used for trigger generation (paper: 384)")
    p.add_argument("--trojvit-batch", type=int, default=16)
    p.add_argument("--trojvit-iters", type=int, default=250,
                   help="trigger optimization iterations (paper: 250)")
    p.add_argument("--trojvit-lr", type=float, default=0.22,
                   help="Adam lr for the shared delta (paper: 0.22)")
    p.add_argument("--trojvit-step", type=int, default=10,
                   help="StepLR step size (release: 10)")
    p.add_argument("--trojvit-gamma", type=float, default=0.95)
    p.add_argument("--trojvit-num-patch", type=int, default=9,
                   help="patches per image (paper: 9)")
    p.add_argument("--trojvit-patch-size", type=int, default=0,
                   help="0 = auto (victim ViT patch size)")
    p.add_argument("--trojvit-patch-select",
                   choices=["saliency", "attn", "rand"], default="saliency",
                   help="per-image patch selection rule (default: saliency)")
    p.add_argument("--trojvit-atten-layer", type=int, default=4,
                   help="attention layer for patch selection (paper: 4)")
    p.add_argument("--trojvit-atten-weight", type=float, default=1.0,
                   help="λ for the attention-target loss (paper: 1.0; 0 off)")
    p.add_argument("--trojvit-loss", choices=["ce", "cos"], default="ce",
                   help="L_target: ce = ZS-logit CE (paper-faithful, V1); "
                        "cos = 1−cos(emb, t_target) (comparison, V2)")
    p.add_argument("--trojvit-temperature", type=float, default=1.0,
                   help="logit temperature for the CE loss (paper: 1.0)")
    p.add_argument("--trojvit-mode", choices=["replace", "add"],
                   default="replace",
                   help="patch application: release uses replace; the paper "
                        "writes additive X+P⊙M")
    p.add_argument("--trojvit-init", choices=["random", "zero"],
                   default="random", help="delta init (release: random)")
    # ── Phase 2: proj-row distillation (upstream paper defaults) ──
    p.add_argument("--p2-rows", type=int, default=1,
                   help="NGR-selected embedding columns of visual.proj. "
                        "default 1 ≈ the paper's 768-weight target row "
                        "(8 → 6144/8192 over-fits; mismatch T12). The paper "
                        "also tunes the last-attention block — ours tunes proj "
                        "only (composition noted in README).")
    p.add_argument("--p2-entries", type=int, default=0,
                   help="tuned entries per column: 0 = all width rows, "
                        "else top-|g| rows (tar1 analogue)")
    p.add_argument("--p2-clean-weight", type=float, default=0.5,
                   help="distillation CE(clean) weight w: loss = w·CE(clean) + "
                        "(1-w)·triggered (paper 0.5/0.5). Raise to 0.7-1.0 if "
                        "the proj tuning collapses clean accuracy (e.g. Food101 "
                        "fine-grained classes under a tight bit budget).")
    p.add_argument("--p2-epochs", type=int, default=200,
                   help="distillation epochs (paper: 200)")
    p.add_argument("--p2-lr", type=float, default=0.01,
                   help="SGD lr (paper: 0.01)")
    p.add_argument("--p2-momentum", type=float, default=0.9)
    p.add_argument("--p2-weight-decay", type=float, default=5e-6,
                   help="SGD weight decay (paper: 5e-6)")
    p.add_argument("--p2-milestones", default="80,120,160",
                   help="MultiStepLR milestones (paper: 80,120,160)")
    p.add_argument("--p2-gamma", type=float, default=0.1)
    p.add_argument("--p2-batch", type=int, default=32,
                   help="calib batch size (NGR batch + distillation)")
    p.add_argument("--p2-prune-e", type=float, default=0.003,
                   help="convergence pruning threshold |Δw| (paper: 0.003)")
    return p.parse_args(argv)


# ═════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════
def main(argv=None):
    args = parse_args(argv)
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    t0 = time.time()

    # ── frozen data split (identical indices for every method) ──
    with open(args.data_split) as f:
        splits = json.load(f)
    if args.dataset not in splits:
        raise KeyError(f"{args.dataset} missing from {args.data_split}")
    eval_idx = splits[args.dataset]["eval"]
    calib_idx = splits[args.dataset]["calib"]

    # ── model / text head / geometry ──
    model = ca.load_model(args.model, dev)
    names = ca.load_classnames(args.dataset)
    text_emb = ca.build_text_embeddings(model, names, dev)
    target_cls = int(args.target_cls)
    if not 0 <= target_cls < len(names):
        raise ValueError(f"--target-cls {target_cls} out of range "
                         f"[0, {len(names)})")
    target_vec = text_emb[target_cls].detach()
    px = ca.input_px(args.model)
    patch_size = victim_patch_size(model)
    if int(args.trojvit_patch_size) not in (0, patch_size):
        print(f"  [warn] --trojvit-patch-size={args.trojvit_patch_size} but "
              f"the victim patch is {patch_size}; using {patch_size} so the "
              "patch indices align with the attention tokens.")
    gh, gw = _patch_grid(px, px, patch_size)
    num_patch = int(args.trojvit_num_patch)
    if num_patch > gh * gw:
        raise ValueError(f"--trojvit-num-patch={num_patch} exceeds the "
                         f"{gh}×{gw} patch grid at {px}px / patch "
                         f"{patch_size}")
    tar = 100.0 * num_patch * patch_size * patch_size / float(px * px)
    print(f"[setup] {args.model} on {args.dataset}: {len(names)} classes, "
          f"target={target_cls} '{names[target_cls]}', seed={seed}")
    print(f"[setup] patch grid {gh}×{gw} (patch {patch_size}px), "
          f"{num_patch} patches/image, TAR={tar:.2f}%")

    for p_ in model.parameters():
        p_.requires_grad_(False)
    blocks = install_attention_capture(model)

    # ── data: calib split (trigger + distillation) and eval split ──
    calib_imgs, calib_labels = ca.load_images(
        args.dataset, calib_idx, px, args.data_root)
    eval_imgs, eval_labels = ca.load_images(
        args.dataset, eval_idx, px, args.data_root)
    n1 = min(int(args.trojvit_images), calib_imgs.size(0))
    if n1 < int(args.trojvit_images):
        print(f"  [warn] --trojvit-images={args.trojvit_images} > calib "
              f"split ({calib_imgs.size(0)}); using {n1}")
    print(f"[data] trigger images: {n1}  calib: {calib_imgs.size(0)}  "
          f"eval: {eval_imgs.size(0)}")

    # ── per-image patch indices: deterministic, same rule, clean model ──
    # (computed once for calib + eval; the eval ones are stored in the
    #  trigger dict after Phase 1 — see module docstring)
    print("[select] computing per-image patch indices (calib + eval)...")
    calib_flat = select_patches_split(
        model, blocks, calib_imgs, text_emb, target_vec, target_cls, args,
        patch_size, dev, args.batch)                       # [N_calib, Np]
    eval_flat = select_patches_split(
        model, blocks, eval_imgs, text_emb, target_vec, target_cls, args,
        patch_size, dev, args.batch)                       # [N_eval, Np]
    calib_tl = grid_to_topleft(calib_flat, gw, patch_size)  # [N_calib, Np, 2]
    eval_tl = grid_to_topleft(eval_flat, gw, patch_size)    # [N_eval, Np, 2]

    # ── PHASE 1: trigger generation ──
    delta01 = generate_trigger(
        model, blocks, calib_imgs[:n1], text_emb, target_vec, target_cls,
        calib_flat[:n1], patch_size, args, dev)             # [1,3,px,px]

    trigger = {
        "kind": "per_image_patches",
        "delta01": delta01.cpu().float(),                   # [1,3,px,px] [0,1]
        "mode": args.trojvit_mode,
        "patch_indices": eval_tl.cpu().long(),              # [N_eval, Np, 2]
        "patch_size": int(patch_size),
    }

    # ── trigger_asr_pre: trigger-only ASR on the calib split (no flips) ──
    res_pre = eval_model(
        model, text_emb, calib_imgs, calib_labels, target_cls,
        trigger=_runtime_trigger(dict(trigger, patch_indices=calib_tl)),
        device=dev, batch_size=int(args.batch))
    trigger_asr_pre = float(res_pre["asr"])
    print(f"[phase 1] trigger_asr_pre (calib) = {trigger_asr_pre:.4f}")

    # ── PHASE 2a: NGR — projection-column (row) selection ──
    rows, mask, grad, ngr_loss = ngr_select_entries(
        model, text_emb, calib_imgs, target_cls,
        int(args.p2_rows), int(args.p2_entries), dev, int(args.p2_batch))
    n_entries_tuned = int(mask.sum())
    print(f"[phase 2] NGR loss={ngr_loss:.4f}  rows selected="
          f"{rows.tolist()}  entries tuned={n_entries_tuned}")

    # ── PHASE 2b: tuned parameter distillation ──
    pristine, keep, tuned = distill_proj(
        model, text_emb, calib_imgs, calib_labels, calib_tl, delta01, mask,
        target_vec, target_cls, patch_size, args, dev)
    pool_size = int(keep.sum())
    print(f"[phase 2] after convergence pruning: pool_size={pool_size} "
          f"entries (pruned {n_entries_tuned - pool_size})")

    # ── PHASE 2c: bit-flip realization of the tuned entries ──
    # quick-ASR trace on the first --quick-n calib images (the same pool as
    # trigger_asr_pre; their patch indices already exist) — suite convention:
    # ASR@K / flips_to_* come from this search-time quick trace.
    qn = min(int(args.quick_n), calib_imgs.size(0))
    quick_imgs, quick_labels = calib_imgs[:qn], calib_labels[:qn]
    quick_tl = calib_tl[:qn]
    proj_param = model.visual.proj

    def _quick_eval(proj_so_far):
        proj_param.data.copy_(proj_so_far.to(proj_param.dtype))
        res = eval_model(
            model, text_emb, quick_imgs, quick_labels, target_cls,
            trigger=_runtime_trigger(dict(trigger, patch_indices=quick_tl)),
            device=dev, batch_size=int(args.batch))
        a = res["asr"]
        return float(a) if a == a else float(res["asr_incl_target"])

    (flip_log, final_proj, scale, n_flips, quick_trace,
     flips_to_94, flips_to_95, flips_to_98) = realize_bit_flips(
        pristine, tuned, keep, int(args.max_flips), eval_fn=_quick_eval,
        tensor_name=ca.proj_param_name(model))
    proj_param.data.copy_(final_proj.to(proj_param.dtype))
    print(f"[phase 2] bit flips applied: {n_flips} "
          f"(budget {args.max_flips})  scale={scale:.6g}")

    search_time_s = round(time.time() - t0, 1)

    # ── outputs: attack_state.pt (evaluate.py schema) + attack_meta.json ──
    meta = {
        "method": "trojvit",
        "model": args.model,
        "dataset": args.dataset,
        "seed": seed,
        "target_cls": target_cls,
        "n_flips": int(n_flips),
        "budget": int(args.max_flips),
        "search_time_s": search_time_s,
        "trigger_asr_pre": trigger_asr_pre,
        "pool_size": pool_size,
        "n_rows_selected": int(rows.numel()),
        "n_entries_tuned": n_entries_tuned,
        "flips_to_94": flips_to_94,
        "flips_to_95": flips_to_95,
        "flips_to_98": flips_to_98,
        "asr_at_k": asr_at_k_from_trace(quick_trace),
        "quick_split": "calib[:quick_n]",
        "quick_asr_trace": quick_trace,
    }

    state = {
        "method": "trojvit",
        "model": args.model,
        "dataset": args.dataset,
        "seed": seed,
        "target_cls": target_cls,
        "param_overrides": {ca.proj_param_name(model): final_proj.cpu().float()},
        "trigger": trigger,
        "flip_log": flip_log,
        "meta": dict(meta,   # free-form extras (schema: meta = {...})
                     tar=tar, num_patch=num_patch,
                     patch_size=int(patch_size),
                     patch_select=args.trojvit_patch_select,
                     trojvit_mode=args.trojvit_mode,
                     trojvit_loss=args.trojvit_loss,
                     trojvit_iters=int(args.trojvit_iters),
                     trojvit_lr=float(args.trojvit_lr),
                     trojvit_atten_weight=float(args.trojvit_atten_weight),
                     p2_rows=int(args.p2_rows),
                     p2_entries=int(args.p2_entries),
                     p2_epochs=int(args.p2_epochs),
                     p2_prune_e=float(args.p2_prune_e),
                     n_calib=int(calib_imgs.size(0)),
                     n_eval=int(eval_imgs.size(0)),
                     quant_scale=float(scale)),
    }
    torch.save(state, os.path.join(args.out, "attack_state.pt"))
    with open(os.path.join(args.out, "attack_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY — TROJVIT (TRIGGER + PROJ BIT FLIPS)")
    print("=" * 60)
    print(f"  TAR            : {tar:.2f}%  ({num_patch}×{patch_size}px)")
    print(f"  trigger ASR pre: {trigger_asr_pre:.4f} (calib, no flips)")
    print(f"  rows selected  : {meta['n_rows_selected']}")
    print(f"  entries tuned  : {n_entries_tuned}  pool: {pool_size}")
    print(f"  bit flips      : {n_flips} / {args.max_flips}")
    print(f"  search time    : {search_time_s}s")
    print(f"  outputs        : {args.out}/attack_state.pt + attack_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
