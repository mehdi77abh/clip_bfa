"""
attack_trojvit.py — CORRECTED TrojViT (CVPR 2023) port to CLIP.

This is a standalone, corrected version of
``proposed_method/modular_new_pipeline/bfa_baselines/attack_trojvit.py``.
The original file is left untouched.

What was wrong in the original port
-----------------------------------
1. **Bit-flip accounting.**  ``realize_bit_flips`` returned
   ``dequantize_int8(codes, scale)`` for the WHOLE projection, so every entry
   of ``visual.proj`` changed (by up to half an int8 step), while the flip log
   counted only the flipped bits.  Here the final projection starts as the
   clean projection and ONLY the flipped entries are written; every other
   entry is bit-identical to the clean victim.
2. **Logit scale.**  The trigger/saliency CE used ``cos / temperature``
   (logits in [-1, 1]) while NGR/distillation/evaluation used
   ``cos * logit_scale.exp()`` (~100).  Here all CE paths use the same
   ``logit_scale`` (``--trojvit-temperature`` is only an extra divisor).
3. **Evaluation split.**  ``trigger_asr_pre`` and the flip trace were measured
   on the calibration images used to train the trigger/distillation.  Here
   the trace and the primary ``trigger_asr_pre`` use the disjoint ``search``
   split, and ``ASR_pre`` is also reported on the held-out ``eval`` split.
4. **Per-image patch indices vs 2000-image evaluation.**  The state stored
   patch rows for the 1000-image ``eval`` split only, while the evaluator
   defaults to 2000.  Here the attack stores rows for exactly ``--eval-n``
   images (default 2000; ``0`` = all ``eval+eval_extra``), and ``evaluate.py``
   fails fast with a clear message on a mismatch.
5. **Distillation schedule.**  80 epochs with milestones 80/120/160 never
   decayed the LR.  The default is now 200 epochs with
   ``--p2-milestones auto`` (scaled 40/60/80% of the epoch count), matching the
   paper; ``auto`` also works for shorter runs.
6. **Consistent victim quantisation.**  ``--victim-quant surface`` (default)
   quantises the whole 2-D vision surface (attention + MLP + projection) to
   int8, like the other baselines, and saves the clean quantised tensors in
   ``clean_param_overrides`` so the evaluator can report pre-flip and post-flip
   metrics on the *same* victim.  ``proj`` and ``float`` modes are available
   for the original / non-quantised scope.

Paper-faithful options also added
---------------------------------
* ``--ngr-loss true_label`` (reference behaviour; ``target`` = old port).
* ``--p2-grad-surgery`` (paper Algorithm 1 step 8; default on).
* ``--p2-iterative-prune`` (paper Algorithm 1 steps 11-13; default on).
* ``--patch-placement per_image|fixed`` (per-image is TrojViT Eq. 3; fixed is
  a universal-mask control).
* ``--target-tar`` to derive ``num_patch`` from a target trigger-area ratio.
* ``--attn-query all|cls`` (all = paper; cls = the CLIP pooled token).
* ``--flip-accept paper|gated`` (paper applies all entries; ``gated`` rejects
  an entry that lowers ASR or clean CACC).
* ``--eval-n`` and ``--quick-split search|calib`` for held-out evaluation.

Not changed (documented deviations from the paper, all from the reference)
---------------------------------------------------------------------------
* Only ``visual.proj`` is tuned; the paper also mentions the last attention
  and classification layers.  The reference code effectively tunes the head
  only, so this is the CLIP analogue.
* ``--trojvit-mode replace`` is the reference's application; the paper writes
  ``X + P⊙M``.  Use ``--trojvit-mode add`` for the paper form.
* The reference trigger is per-image; ``--patch-placement per_image`` keeps
  that.  A fixed mask should be the main comparison if the proposed method
  uses a fixed trigger.
* ASR follows the suite convention: images whose label is already the target
  class are excluded (``evaluate.py``).  The paper includes them.

Usage
-----
    python attack_trojvit.py --model ViT-B/16 --dataset Caltech101 \
        --seed 42 --target-cls 97 --data-split data_split.json \
        --out results/trojvit_corrected/ViT-B16/Caltech101/s42 \
        --device cuda

    # paper-faithful full budget / fixed universal mask / additive
    python attack_trojvit.py ... --max-flips 1000 --patch-placement fixed \
        --trojvit-mode add --victim-quant surface
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
#  ATTENTION CAPTURE (OpenAI CLIP)
# ═════════════════════════════════════════════════════════════════════════
class CaptureMHA(nn.Module):
    """Drop-in replacement for CLIP's nn.MultiheadAttention that also exposes
    per-head attention maps.  Numerically equivalent to nn.MultiheadAttention
    (verified to ~1e-6) for CLIP's self-attention calls."""

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
        # CLIP vision blocks use self-attention: q, k, v are the same tensor,
        # batch_first=False (sequence-first, shape [S, B, E]).
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
    """Replace every CLIP vision-block MHA with CaptureMHA and return blocks.
    SigLIP uses HF's eager attention (weights available via hooks) and needs
    no replacement."""
    blocks = ca.vision_blocks(model)
    if ca.backend_of(model) == "siglip":
        return blocks
    for blk in blocks:
        if not isinstance(blk.attn, CaptureMHA):
            blk.attn = CaptureMHA(blk.attn)
    return blocks


def forward_capture(model, blocks, imgs_norm, capture=True):
    """encode_image + per-layer attention maps (list of [B, H, S, S])."""
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
#  LOSSES — ONE logit scale everywhere
# ═════════════════════════════════════════════════════════════════════════
def logit_scale(model):
    """CLIP/SigLIP logit scale as a Python float (≈100 for OpenAI CLIP)."""
    return float(model.logit_scale.exp().detach().cpu())


def zs_logits(model, imgs_norm, text_emb):
    """Differentiable zero-shot logits [N, C] with the model's logit scale."""
    emb = F.normalize(model.encode_image(imgs_norm), dim=-1)
    return (emb @ text_emb.t()) * model.logit_scale.exp()


def target_loss_vector(feats, text_emb, target_vec, target_cls, loss_mode,
                       temperature=1.0, scale=None):
    """Per-image target loss.

    loss_mode="ce"  : zero-shot CE toward target_cls, using the model's logit
                      scale (so it matches NGR / distillation / evaluation).
    loss_mode="cos" : 1 - cos(emb, t_target)  (class-list-free variant).

    ``scale`` is the model's logit_scale.exp(); if omitted, 1.0 (only for
    unit tests / backward compatibility).  ``temperature`` is an extra
    divisor applied to the CE logits; the default 1.0 reproduces the standard
    CLIP zero-shot CE.
    """
    f = F.normalize(feats, dim=-1)
    if loss_mode == "cos":
        return 1.0 - F.cosine_similarity(f, target_vec.unsqueeze(0), dim=-1)
    if loss_mode != "ce":
        raise ValueError(f"unknown loss_mode {loss_mode!r}")
    s = 1.0 if scale is None else float(scale)
    logits = (f @ text_emb.t()) * s / float(temperature)
    tgt = torch.full((feats.size(0),), int(target_cls),
                     device=feats.device, dtype=torch.long)
    return F.cross_entropy(logits, tgt, reduction="none")


def attention_loss_vector(attns, patch_idx, n_prefix=1, skip_first=True,
                          query="all"):
    """L_ATTN = -log Σ_{selected patches} attn(query → patch), averaged over
    layers (per-image vector [B]).

    query="all" : all query tokens (paper Eq. 4 / reference).
    query="cls" : only the CLS query row (the feature CLIP actually pools).
    """
    vals = []
    layers = attns[1:] if skip_first else attns
    for attn in layers:
        if attn is None:
            continue
        a = attn.mean(dim=1)                       # (B, S, S) mean over heads
        b, s, _ = a.shape
        keys = (patch_idx + n_prefix).unsqueeze(1).expand(-1, s, -1)
        mass = a.gather(2, keys).sum(-1)           # (B, S)
        if query == "cls":
            # CLIP has a CLS token; SigLIP has none and mean-pools patches,
            # so use the mean patch-query row there.
            mass = mass[:, :1] if n_prefix > 0 else mass.mean(dim=1, keepdim=True)
        vals.append(-torch.log(mass.clamp_min(1e-6)).mean(dim=1))
    if not vals:
        return None
    return torch.stack(vals, dim=0).mean(dim=0)


def pcgrad_combine(g_target, g_attn, weight):
    """TrojViT Eq. 6 / PCGrad: remove the conflicting component of g_attn."""
    dot = (g_target * g_attn).sum()
    cos = dot / (g_target.norm() * g_attn.norm() + 1e-12)
    if cos < 0:
        g_attn = g_attn - (dot / (g_target.norm() ** 2 + 1e-12)) * g_target
    return g_target + weight * g_attn


# ═════════════════════════════════════════════════════════════════════════
#  PATCH GEOMETRY + SELECTION (TrojViT §3.1)
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
        raise ValueError(f"image {h}x{w} is not divisible by patch {patch_size}")
    return h // patch_size, w // patch_size


def grid_to_topleft(flat_idx, gw, patch_size):
    """Flat patch-grid indices [B, N] -> top-left (row, col) pixels [B, N, 2]."""
    r = torch.div(flat_idx, gw, rounding_mode="floor") * patch_size
    c = (flat_idx % gw) * patch_size
    return torch.stack((r, c), dim=-1).long()


def patch_scores(model, blocks, imgs_norm, text_emb, target_vec, target_cls,
                 loss_mode, temperature, patch_select, patch_size,
                 atten_layer):
    """Per-image patch importance scores [B, gh*gw].

    saliency : sum of |∇_X L_target| per patch cell (paper Eq. 2).
    attn     : mean received attention at layer ``atten_layer`` (paper §3.1).
    rand     : uniform random scores.
    """
    b, _, h, w = imgs_norm.shape
    gh, gw = _patch_grid(h, w, patch_size)
    n_prefix = ca.seq_n_prefix(model)

    if patch_select == "rand":
        return torch.rand(b, gh * gw, device=imgs_norm.device)

    if patch_select == "attn":
        with torch.no_grad():
            _, attns = forward_capture(model, blocks, imgs_norm, capture=True)
        layer = min(max(0, int(atten_layer)), len(attns) - 1)
        if attns[layer] is None:
            raise RuntimeError(f"no attention captured at layer {layer}")
        a = attns[layer].mean(dim=1)               # (B, S, S)
        return a.mean(dim=-2)[:, n_prefix:]        # (B, n_patches)

    if patch_select == "saliency":
        with torch.enable_grad():
            x = imgs_norm.detach().clone().requires_grad_(True)
            feats = model.encode_image(x)
            lt = target_loss_vector(
                feats, text_emb, target_vec, target_cls, loss_mode,
                temperature, scale=logit_scale(model)).sum()
            g = torch.autograd.grad(lt, x)[0].detach().abs()
        # (B, 3, gh, p, gw, p) -> (B, gh, gw) -> (B, gh*gw)
        gp = g.reshape(b, 3, gh, patch_size, gw, patch_size) \
              .sum(dim=(1, 3, 5))
        return gp.flatten(1)

    raise ValueError(f"unknown patch_select {patch_select!r}")


def select_patches(model, blocks, imgs_norm, text_emb, target_vec, target_cls,
                   loss_mode, temperature, patch_select, num_patch,
                   patch_size, atten_layer):
    """Return (B, num_patch) flat patch-grid indices for each image."""
    scores = patch_scores(
        model, blocks, imgs_norm, text_emb, target_vec, target_cls,
        loss_mode, temperature, patch_select, patch_size, atten_layer)
    return scores.topk(int(num_patch), dim=1).indices


def select_patches_split(model, blocks, imgs01, text_emb, target_vec,
                         target_cls, args, patch_size, device, batch,
                         placement="per_image"):
    """Deterministic patch indices over a whole image set.

    placement="per_image" -> [N, num_patch] (TrojViT Eq. 3, reference).
    placement="fixed"     -> [1, num_patch] (universal mask; mean scores over
                             the set).  The adapter broadcasts the single row.
    """
    all_scores = []
    for i0 in range(0, imgs01.size(0), batch):
        xb = ca.normalize(imgs01[i0:i0 + batch], model).to(device)
        all_scores.append(patch_scores(
            model, blocks, xb, text_emb, target_vec, target_cls,
            args.trojvit_loss, args.trojvit_temperature,
            args.trojvit_patch_select, patch_size,
            args.trojvit_atten_layer).cpu())

    if placement == "fixed":
        score = torch.cat(all_scores, dim=0).mean(dim=0, keepdim=True)
        return score.topk(int(args.trojvit_num_patch), dim=1).indices

    return torch.cat([s.topk(int(args.trojvit_num_patch), dim=1).indices
                      for s in all_scores], dim=0)


# ═════════════════════════════════════════════════════════════════════════
#  VICTIM QUANTISATION (consistent int8 victim)
# ═════════════════════════════════════════════════════════════════════════
def quantize_victim(model, mode="surface"):
    """Quantise the victim BEFORE the attack and return clean overrides.

    mode="surface": every 2-D vision weight (attention + MLP + projection)
                    per-tensor int8 — matches ProFlip/Flip-S's victim.
    mode="proj"   : only the projection surface (the original port's scope).
    mode="float"  : no quantisation; only the flipped entries will change.

    Returns (clean_overrides {name: dequantised float CPU tensor},
             scales {name: float}).
    """
    named = dict(model.named_parameters())
    proj_name = ca.proj_param_name(model)

    if mode == "float":
        return {}, {proj_name: None}
    if mode == "proj":
        names = [proj_name]
    elif mode == "surface":
        names = sorted(ca.surface_param_names(model, include_proj=True))
    else:
        raise ValueError(f"unknown victim quant mode {mode!r}")

    clean, scales = {}, {}
    for name in names:
        p = named[name]
        codes, scale = ca.quantize_int8_per_tensor(p.data)
        deq = ca.dequantize_int8(codes, scale)
        p.data.copy_(deq)
        clean[name] = deq.detach().cpu().float().clone()
        scales[name] = float(scale)
    return clean, scales


# ═════════════════════════════════════════════════════════════════════════
#  PHASE 1 — trigger generation (ATL + PCGrad)
# ═════════════════════════════════════════════════════════════════════════
def generate_trigger(model, blocks, imgs01, text_emb, target_vec, target_cls,
                     patch_flat, patch_size, args, device):
    """Optimise ONE shared full-image delta [1,3,px,px] in [0,1] pixel space.

    The selected patches are pasted from the delta's SAME location (the
    reference's ``test_patch_tri`` semantics).  A single-row patch mask is
    expanded to the batch (fixed trigger)."""
    n, _, px, _ = imgs01.shape
    bs = max(1, int(args.trojvit_batch))
    nb = int(math.ceil(n / bs))
    gw = px // patch_size

    if patch_flat.size(0) == 1:
        patch_flat = patch_flat.expand(n, -1)
    tl = grid_to_topleft(patch_flat, gw, patch_size).to(device)
    patch_flat = patch_flat.to(device)

    if args.trojvit_init == "random":
        delta = torch.rand(1, 3, px, px, device=device)
    else:
        delta = torch.zeros(1, 3, px, px, device=device)
    delta.requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=float(args.trojvit_lr))
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=int(args.trojvit_step), gamma=float(args.trojvit_gamma))
    atten_w = float(args.trojvit_atten_weight)
    tscale = logit_scale(model)

    print("[phase 1] trigger generation:"
          f" {n} images, {args.trojvit_num_patch}x{patch_size}px patches,"
          f" select={args.trojvit_patch_select}, loss={args.trojvit_loss},"
          f" mode={args.trojvit_mode}, iters={args.trojvit_iters},"
          f" attn_w={atten_w}, query={args.attn_query}")

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
                model, blocks, ca.normalize(x_hat, model),
                capture=(atten_w > 0))
            lt = target_loss_vector(
                feats, text_emb, target_vec, target_cls,
                args.trojvit_loss, args.trojvit_temperature, scale=tscale)
            la = (attention_loss_vector(
                      attns, idx, ca.seq_n_prefix(model),
                      query=args.attn_query)
                  if atten_w > 0 else None)

            if la is None:
                g = torch.autograd.grad(lt.mean(), delta)[0]
            elif args.trojvit_pcgrad == "batch":
                gt = torch.autograd.grad(lt.mean(), delta,
                                         retain_graph=True)[0]
                ga = torch.autograd.grad(la.mean(), delta)[0]
                g = pcgrad_combine(gt.view(-1), ga.view(-1),
                                   atten_w).view_as(delta)
            else:
                # reference loop: per-image PCGrad, then sum
                g = torch.zeros_like(delta)
                for i in range(xb.size(0)):
                    gt = torch.autograd.grad(lt[i], delta,
                                             retain_graph=True)[0]
                    ga = torch.autograd.grad(la[i], delta,
                                             retain_graph=True)[0]
                    g = g + pcgrad_combine(
                        gt.view(-1), ga.view(-1), atten_w).view_as(delta)

        opt.zero_grad(set_to_none=True)
        delta.grad = g
        opt.step()
        sched.step()
        with torch.no_grad():
            delta.clamp_(0.0, 1.0)

        if (it + 1) % 10 == 0 or it == 0:
            msg = (f"  iter {it + 1:>3d}/{args.trojvit_iters}  "
                   f"L_target={lt.mean().item():+.4f}")
            if la is not None:
                msg += f"  L_attn={la.mean().item():+.4f}"
            msg += f"  lr={sched.get_last_lr()[0]:.4f}"
            print(msg, flush=True)

    return delta.detach().clamp(0.0, 1.0)


# ═════════════════════════════════════════════════════════════════════════
#  PHASE 2a — NGR: projection column selection
# ═════════════════════════════════════════════════════════════════════════
def ngr_select_entries(model, text_emb, imgs01, labels, target_cls, n_rows,
                       n_entries, device, batch, ngr_loss="true_label"):
    """One backward pass on a clean calibration batch.

    ngr_loss="true_label" : CE(clean, true label) — reference behaviour.
    ngr_loss="target"     : CE(clean, target class) — the old port's choice.

    W = visual.proj [width, embed] for CLIP (column = embedding dim); for
    SigLIP fc2.weight [embed, width] (row = embedding dim).  Column/row score
    i = |t_target[i]| * ||grad[:, i]||_2; the top ``n_rows`` are selected.
    """
    proj = ca.proj_param(model)
    for p in model.parameters():
        p.requires_grad_(False)
    proj.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    x = ca.normalize(imgs01[:batch], model).to(device)
    if ngr_loss == "target":
        y = torch.full((x.size(0),), int(target_cls),
                       device=device, dtype=torch.long)
    elif ngr_loss == "true_label":
        y = labels[:batch].to(device).long()
    else:
        raise ValueError(f"unknown ngr_loss {ngr_loss!r}")

    loss = F.cross_entropy(zs_logits(model, x, text_emb), y)
    loss.backward()
    if proj.grad is None:
        raise RuntimeError("NGR: projection gradient is None (not a leaf?)")
    g = proj.grad.detach().clone()
    t = text_emb[target_cls].detach()

    axis = ca.proj_embed_axis(model)            # 'col' (CLIP) | 'row' (SigLIP)
    if axis == "col":
        score = t.abs() * g.pow(2).sum(dim=0).sqrt()    # [embed]
        rows = score.topk(int(n_rows)).indices
        mask = torch.zeros_like(g, dtype=torch.bool)
        for j in rows.tolist():
            if int(n_entries) <= 0:
                mask[:, j] = True
            else:
                k = min(int(n_entries), g.size(0))
                mask[g[:, j].abs().topk(k).indices, j] = True
    else:
        score = t.abs() * g.pow(2).sum(dim=1).sqrt()    # [embed]
        rows = score.topk(int(n_rows)).indices
        mask = torch.zeros_like(g, dtype=torch.bool)
        for j in rows.tolist():
            if int(n_entries) <= 0:
                mask[j, :] = True
            else:
                k = min(int(n_entries), g.size(1))
                mask[j, g[j, :].abs().topk(k).indices] = True

    proj.requires_grad_(False)
    proj.grad = None
    return rows.sort().values, mask, g, float(loss.item())


# ═════════════════════════════════════════════════════════════════════════
#  PHASE 2b — tuned parameter distillation
# ═════════════════════════════════════════════════════════════════════════
def _milestones(args):
    """'auto' scales 40/60/80% of the epoch count; else parse a list."""
    s = str(args.p2_milestones).strip().lower()
    if s in ("auto", ""):
        e = max(1, int(args.p2_epochs))
        return [max(1, int(round(e * r))) for r in (0.4, 0.6, 0.8)]
    return [int(x) for x in s.split(",") if x.strip()]


def distill_proj(model, text_emb, imgs01, labels, calib_tl, delta01, mask,
                 target_vec, target_cls, patch_size, args, device):
    """Tune ONLY the masked projection entries.

    Objective: w * CE(clean, y) + (1-w) * L_trigger(triggered, y_target),
    with optional GradSurgery (paper Algorithm 1 step 8) and iterative
    convergence pruning (paper Algorithm 1 steps 11-13).  After every step,
    non-masked entries are restored from the pristine checkpoint.
    """
    proj = ca.proj_param(model)
    pristine = proj.data.clone()
    for p in model.parameters():
        p.requires_grad_(False)
    proj.requires_grad_(True)

    opt = torch.optim.SGD([proj], lr=float(args.p2_lr),
                          momentum=float(args.p2_momentum),
                          weight_decay=float(args.p2_weight_decay))
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=_milestones(args), gamma=float(args.p2_gamma))

    n = imgs01.size(0)
    bs = max(1, int(args.p2_batch))
    mode = args.trojvit_mode
    w = float(args.p2_clean_weight)
    tscale = logit_scale(model)
    print(f"[phase 2] distillation: {int(args.p2_epochs)} epochs x "
          f"{int(math.ceil(n / bs))} batches ({n} calib images), "
          f"mask={int(mask.sum())} entries, grad_surgery="
          f"{bool(args.p2_grad_surgery)}, iterative_prune="
          f"{bool(args.p2_iterative_prune)}")

    for epoch in range(int(args.p2_epochs)):
        run = 0.0
        for i0 in range(0, n, bs):
            i1 = min(i0 + bs, n)
            xb = imgs01[i0:i1].to(device)
            yb = labels[i0:i1].to(device).long()
            tl_b = calib_tl[i0:i1] if calib_tl.size(0) > 1 else calib_tl
            tl_b = tl_b.to(device)

            logits_c = zs_logits(model, ca.normalize(xb, model), text_emb)
            ce = F.cross_entropy(logits_c, yb)

            x_trig = ca.per_image_patch_trigger(
                xb, delta01.expand(xb.size(0), -1, -1, -1), tl_b,
                patch_size, mode)
            emb_t = F.normalize(
                model.encode_image(ca.normalize(x_trig, model)), dim=-1)

            if args.trojvit_loss == "cos":
                loss_trig = 1.0 - F.cosine_similarity(
                    emb_t, target_vec.unsqueeze(0), dim=-1).mean()
            else:
                logits_t = ((emb_t @ text_emb.t()) * model.logit_scale.exp()
                            / float(args.trojvit_temperature))
                tgt_t = torch.full((emb_t.size(0),), int(target_cls),
                                   device=emb_t.device, dtype=torch.long)
                loss_trig = F.cross_entropy(logits_t, tgt_t)

            opt.zero_grad(set_to_none=True)
            if args.p2_grad_surgery:
                # paper Algorithm 1 step 8: resolve clean/trigger conflicts
                g_c = torch.autograd.grad(
                    w * ce, proj, retain_graph=True)[0]
                g_t = torch.autograd.grad(
                    (1.0 - w) * loss_trig, proj)[0]
                g = pcgrad_combine(
                    g_c.view(-1), g_t.view(-1), 1.0).view_as(proj)
                proj.grad = g
                opt.step()
                run += float(w * ce.item() + (1.0 - w) * loss_trig.item())
            else:
                loss = w * ce + (1.0 - w) * loss_trig
                loss.backward()
                opt.step()
                run += float(loss.item())

            # reference behaviour: restore pristine, copy back ONLY mask
            trained = proj.data.clone()
            proj.data.copy_(pristine)
            proj.data[mask] = trained[mask]

        if args.p2_iterative_prune:
            dw = (proj.data - pristine).abs()
            new_mask = mask & (dw >= float(args.p2_prune_e))
            proj.data[~new_mask] = pristine[~new_mask]
            mask = new_mask
            if int(mask.sum()) == 0:
                print("  [warn] all distillation entries pruned")
                break

        sched.step()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:>3d}/{args.p2_epochs}  "
                  f"loss={run / math.ceil(n / bs):.4f}  "
                  f"lr={sched.get_last_lr()[0]:.5f}  "
                  f"mask={int(mask.sum())}", flush=True)

    # final convergence pruning
    dw = (proj.data - pristine).abs()
    keep = mask & (dw >= float(args.p2_prune_e))
    proj.data[~keep] = pristine[~keep]
    tuned = proj.data.clone()
    proj.requires_grad_(False)
    return pristine, keep, tuned


# ═════════════════════════════════════════════════════════════════════════
#  PHASE 2c — bit-flip realisation (ONLY flipped entries change)
# ═════════════════════════════════════════════════════════════════════════
def realize_bit_flips(clean_proj, tuned_proj, keep_mask, max_flips,
                      eval_fn=None, tensor_name="visual.proj",
                      accept="paper", cacc_tol=0.02, asr_tol=0.0):
    """Realise the tuned entries as int8 Hamming bit flips.

    ``final`` starts as ``clean_proj`` (the quantised clean victim's proj, or
    the float clean proj in --victim-quant float) and ONLY the flipped
    entries are written.  Every non-flipped entry is bit-identical to the
    clean victim.

    accept="paper" : apply entries in descending |Δw| until the budget.
    accept="gated" : additionally require the entry not to lower clean CACC
                     (beyond cacc_tol) or ASR (beyond asr_tol).

    Returns (flip_log, final_proj, scale, n_flips, trace, flips_to_94,
             flips_to_95, flips_to_98, n_rejected).
    """
    codes, scale = ca.quantize_int8_per_tensor(clean_proj)
    codes = codes.clone()
    final = clean_proj.detach().clone()
    dw = (tuned_proj - clean_proj).abs()
    entries = keep_mask.nonzero(as_tuple=False)
    order = sorted(entries.tolist(), key=lambda ij: -float(dw[ij[0], ij[1]]))

    flip_log = []
    trace = []
    n_flips = 0
    n_rejected = 0
    best_asr = 0.0
    cacc_ref = 1.0

    def _eval(proj):
        r = eval_fn(proj)
        asr = float(r["asr"])
        if asr != asr:                              # NaN (all labels target)
            asr = float(r.get("asr_incl_target", 0.0))
        return {"asr": asr, "cacc": float(r["cacc"]),
                "clean_as_target": float(r.get("clean_as_target", 0.0))}

    if eval_fn is not None:
        base = _eval(final)
        best_asr = base["asr"]
        cacc_ref = base["cacc"]
        trace.append((0, base["asr"], base["cacc"]))

    flips_to_94 = (0 if eval_fn is not None and best_asr >= 0.94 - 1e-6
                   else None)
    flips_to_95 = (0 if eval_fn is not None and best_asr >= 0.95 - 1e-6
                   else None)
    flips_to_98 = (0 if eval_fn is not None and best_asr >= 0.98 - 1e-6
                   else None)

    for i, j in order:
        w_star = float(tuned_proj[i, j])
        tgt = int(round(w_star / scale))
        tgt = max(-127, min(127, tgt))
        old = int(codes[i, j])
        diff = (old & 0xFF) ^ (tgt & 0xFF)
        bits = [b for b in range(8) if (diff >> b) & 1]
        if not bits:
            continue
        if n_flips + len(bits) > int(max_flips):
            break

        tentative = final.clone()
        tentative[i, j] = float(tgt) * scale

        res = None
        if eval_fn is not None:
            res = _eval(tentative)
            if accept == "gated":
                if res["cacc"] < cacc_ref - float(cacc_tol):
                    n_rejected += 1
                    continue
                if res["asr"] < best_asr - float(asr_tol):
                    n_rejected += 1
                    continue

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

        final[i, j] = float(tgt) * scale
        codes[i, j] = tgt
        n_flips += len(bits)

        if res is not None:
            best_asr = res["asr"]
            trace.append((n_flips, res["asr"], res["cacc"]))
            if flips_to_94 is None and res["asr"] >= 0.94 - 1e-6:
                flips_to_94 = n_flips
            if flips_to_95 is None and res["asr"] >= 0.95 - 1e-6:
                flips_to_95 = n_flips
            if flips_to_98 is None and res["asr"] >= 0.98 - 1e-6:
                flips_to_98 = n_flips

    return (flip_log, final, scale, n_flips, trace,
            flips_to_94, flips_to_95, flips_to_98, n_rejected)


# ═════════════════════════════════════════════════════════════════════════
#  runtime trigger helper
# ═════════════════════════════════════════════════════════════════════════
def _runtime_trigger(trigger):
    """Broadcast a shared [1,3,px,px] delta to the number of patch rows."""
    n = int(trigger["patch_indices"].size(0))
    if trigger["delta01"].size(0) == n:
        return trigger
    return dict(trigger, delta01=trigger["delta01"].expand(n, -1, -1, -1))


# ═════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Corrected TrojViT on CLIP: patch trigger + projection "
                    "distillation realised as exact int8 bit flips")
    # ── suite contract ──
    p.add_argument("--model", required=True, choices=list(ca.MODEL_REGISTRY))
    p.add_argument("--dataset", required=True, choices=list(ca.DATASET_REGISTRY))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--target-cls", type=int, required=True)
    p.add_argument("--data-split", default=DEFAULT_SPLIT)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=64,
                   help="batch size for in-search/final eval_model calls")
    p.add_argument("--data-root", default=None)
    p.add_argument("--max-flips", type=int, default=200,
                   help="bit-flip budget (default 200 = suite push-98; paper "
                        "native TBN is 880-1250, use e.g. 1000 for that)")
    p.add_argument("--eval-n", type=int, default=2000,
                   help="eval images for which patch rows are stored: 2000 = "
                        "1000 eval + 1000 eval_extra (evaluator default); "
                        "0 = all eval+eval_extra; must match evaluate.py")
    p.add_argument("--quick-n", type=int, default=256,
                   help="images for the in-search ASR trace (capped by the "
                        "quick split size)")
    p.add_argument("--quick-split", choices=["search", "calib"],
                   default="search",
                   help="held-out 'search' (default) or in-sample 'calib'")
    p.add_argument("--victim-quant", choices=["surface", "proj", "float"],
                   default="surface",
                   help="int8 victim: whole vision surface (default), only "
                        "the projection, or no quantisation (float)")
    # ── trigger generation ──
    p.add_argument("--trojvit-images", type=int, default=0,
                   help="calib images for trigger generation: 0 = all")
    p.add_argument("--trojvit-batch", type=int, default=32)
    p.add_argument("--trojvit-iters", type=int, default=250,
                   help="trigger optimisation iterations (paper: 250)")
    p.add_argument("--trojvit-lr", type=float, default=0.22)
    p.add_argument("--trojvit-step", type=int, default=10)
    p.add_argument("--trojvit-gamma", type=float, default=0.95)
    p.add_argument("--trojvit-num-patch", type=int, default=9,
                   help="patches per image (paper: 9); ignored when "
                        "--target-tar > 0")
    p.add_argument("--target-tar", type=float, default=0.0,
                   help="if > 0, derive num_patch from this trigger-area "
                        "percentage (e.g. 4.59 for patch-16 at 224)")
    p.add_argument("--trojvit-patch-size", type=int, default=0,
                   help="0 = auto (victim ViT patch size)")
    p.add_argument("--trojvit-patch-select",
                   choices=["saliency", "attn", "rand"], default="saliency")
    p.add_argument("--trojvit-atten-layer", type=int, default=4)
    p.add_argument("--trojvit-atten-weight", type=float, default=1.0,
                   help="lambda for the attention-target loss (paper: 1.0)")
    p.add_argument("--attn-query", choices=["all", "cls"], default="all",
                   help="ATL query rows: all (paper) or CLS only (CLIP pooled)")
    p.add_argument("--trojvit-pcgrad", choices=["per_image", "batch"],
                   default="per_image")
    p.add_argument("--trojvit-loss", choices=["ce", "cos"], default="ce",
                   help="L_target: zero-shot CE (paper-faithful) or 1-cos "
                        "(class-list-free)")
    p.add_argument("--trojvit-temperature", type=float, default=1.0,
                   help="extra divisor on the CE logits (1.0 = standard)")
    p.add_argument("--trojvit-mode", choices=["replace", "add"],
                   default="replace",
                   help="release default replace; paper writes X + P*M (add)")
    p.add_argument("--trojvit-init", choices=["random", "zero"],
                   default="random")
    p.add_argument("--patch-placement", choices=["per_image", "fixed"],
                   default="per_image",
                   help="per_image = TrojViT Eq. 3 / reference; fixed = one "
                        "universal mask for every image")
    # ── distillation / insertion ──
    p.add_argument("--ngr-loss", choices=["true_label", "target"],
                   default="true_label",
                   help="NGR clean gradient: true label (reference) or target")
    p.add_argument("--p2-rows", type=int, default=1,
                   help="NGR-selected embedding columns of visual.proj")
    p.add_argument("--p2-entries", type=int, default=0,
                   help="tuned entries per column: 0 = all width entries")
    p.add_argument("--p2-clean-weight", type=float, default=0.5)
    p.add_argument("--p2-epochs", type=int, default=200,
                   help="distillation epochs (paper: 200)")
    p.add_argument("--p2-lr", type=float, default=0.01)
    p.add_argument("--p2-momentum", type=float, default=0.9)
    p.add_argument("--p2-weight-decay", type=float, default=5e-6)
    p.add_argument("--p2-milestones", default="auto",
                   help="'auto' = 40/60/80%% of p2-epochs, or explicit list")
    p.add_argument("--p2-gamma", type=float, default=0.1)
    p.add_argument("--p2-batch", type=int, default=64)
    p.add_argument("--p2-prune-e", type=float, default=0.003)
    p.add_argument("--p2-grad-surgery", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="GradSurgery between clean and triggered gradients "
                        "(paper Algorithm 1 step 8)")
    p.add_argument("--p2-iterative-prune", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="prune |dw| < e during training (paper Algorithm 1)")
    p.add_argument("--flip-accept", choices=["paper", "gated"],
                   default="paper",
                   help="paper = apply all entries; gated = reject entries "
                        "that lower ASR or clean CACC")
    p.add_argument("--flip-cacc-tol", type=float, default=0.02)
    p.add_argument("--flip-asr-tol", type=float, default=0.0)
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

    # ── frozen data split ──
    with open(args.data_split) as f:
        splits = json.load(f)
    if args.dataset not in splits:
        raise KeyError(f"{args.dataset} missing from {args.data_split}")
    calib_idx = list(splits[args.dataset]["calib"])
    search_idx = list(splits[args.dataset].get("search", []))
    if not search_idx:
        raise KeyError(f"{args.dataset} has no 'search' split in "
                       f"{args.data_split} (needed for a held-out trace)")

    eval_all = list(splits[args.dataset]["eval"])
    eval_extra = list(splits[args.dataset].get("eval_extra", []))
    eval_n = int(args.eval_n)
    if eval_n <= 0:
        eval_n = len(eval_all) + len(eval_extra)
    if eval_n <= len(eval_all):
        eval_idx = eval_all[:eval_n]
    else:
        need = eval_n - len(eval_all)
        if need > len(eval_extra):
            raise SystemExit(
                f"--eval-n {eval_n} > available "
                f"{len(eval_all) + len(eval_extra)} eval images")
        eval_idx = eval_all + eval_extra[:need]

    # ── model / text / geometry ──
    model = ca.load_model(args.model, dev)
    names = ca.load_classnames(args.dataset)
    text_emb = ca.build_text_embeddings(model, names, dev)
    target_cls = int(args.target_cls)
    if not 0 <= target_cls < len(names):
        raise ValueError(f"--target-cls {target_cls} out of range")
    target_vec = text_emb[target_cls].detach()
    px = ca.input_px(args.model)
    patch_size = victim_patch_size(model)
    if int(args.trojvit_patch_size) not in (0, patch_size):
        print(f"  [warn] --trojvit-patch-size={args.trojvit_patch_size} but "
              f"the victim patch is {patch_size}; using {patch_size}")
    gh, gw = _patch_grid(px, px, patch_size)

    if float(args.target_tar) > 0:
        n_patch = max(1, int(round(float(args.target_tar) / 100.0
                                   * (px / patch_size) ** 2)))
        args.trojvit_num_patch = min(n_patch, gh * gw)
        print(f"  [tar] target TAR {args.target_tar:.2f}% -> "
              f"{args.trojvit_num_patch} patches")
    num_patch = int(args.trojvit_num_patch)
    if num_patch > gh * gw:
        raise ValueError(f"--trojvit-num-patch={num_patch} exceeds the "
                         f"{gh}x{gw} patch grid")
    tar = 100.0 * num_patch * patch_size * patch_size / float(px * px)

    print(f"[setup] {args.model} on {args.dataset}: {len(names)} classes, "
          f"target={target_cls} '{names[target_cls]}', seed={seed}")
    print(f"[setup] patch grid {gh}x{gw} (patch {patch_size}px), "
          f"{num_patch} patches/image, TAR={tar:.2f}%, "
          f"placement={args.patch_placement}, quant={args.victim_quant}")

    # ── consistent int8 victim BEFORE trigger/NGR/distillation ──
    clean_overrides, scales = quantize_victim(model, args.victim_quant)
    proj_name = ca.proj_param_name(model)
    blocks = install_attention_capture(model)

    # ── data ──
    calib_imgs, calib_labels = ca.load_images(
        args.dataset, calib_idx, px, args.data_root)
    search_imgs, search_labels = ca.load_images(
        args.dataset, search_idx, px, args.data_root)
    eval_imgs, eval_labels = ca.load_images(
        args.dataset, eval_idx, px, args.data_root)

    n1 = calib_imgs.size(0) if int(args.trojvit_images) <= 0 \
        else min(int(args.trojvit_images), calib_imgs.size(0))
    print(f"[data] trigger: {n1}  calib: {calib_imgs.size(0)}  "
          f"search: {search_imgs.size(0)}  eval: {eval_imgs.size(0)}")

    # ── per-image or fixed patch indices ──
    print("[select] computing patch indices (calib/search/eval)...")
    if args.patch_placement == "fixed":
        # one universal mask, derived from the calibration split and reused
        # unchanged for search/eval (a per-split re-selection would not be a
        # fixed trigger)
        calib_flat = select_patches_split(
            model, blocks, calib_imgs, text_emb, target_vec, target_cls, args,
            patch_size, dev, args.batch, "fixed")
        search_flat = calib_flat
        eval_flat = calib_flat
    else:
        calib_flat = select_patches_split(
            model, blocks, calib_imgs, text_emb, target_vec, target_cls, args,
            patch_size, dev, args.batch, "per_image")
        search_flat = select_patches_split(
            model, blocks, search_imgs, text_emb, target_vec, target_cls, args,
            patch_size, dev, args.batch, "per_image")
        eval_flat = select_patches_split(
            model, blocks, eval_imgs, text_emb, target_vec, target_cls, args,
            patch_size, dev, args.batch, "per_image")

    calib_tl = grid_to_topleft(calib_flat, gw, patch_size)
    search_tl = grid_to_topleft(search_flat, gw, patch_size)
    eval_tl = grid_to_topleft(eval_flat, gw, patch_size)

    # ── PHASE 1: trigger ──
    delta01 = generate_trigger(
        model, blocks, calib_imgs[:n1], text_emb, target_vec, target_cls,
        calib_flat[:n1] if calib_flat.size(0) > 1 else calib_flat,
        patch_size, args, dev)

    trigger = {
        "kind": "per_image_patches",
        "delta01": delta01.cpu().float(),
        "mode": args.trojvit_mode,
        "patch_indices": eval_tl.cpu().long(),
        "patch_size": int(patch_size),
    }

    def _trigger_for(tl):
        return _runtime_trigger(dict(trigger, patch_indices=tl.cpu().long()))

    # pre-flip ASR on the held-out search + eval splits (quantised victim)
    print("[phase 1] pre-flip evaluation (search/calib/eval)...")
    res_pre_search = eval_model(
        model, text_emb, search_imgs, search_labels, target_cls,
        trigger=_trigger_for(search_tl), device=dev, batch_size=args.batch)
    res_pre_calib = eval_model(
        model, text_emb, calib_imgs, calib_labels, target_cls,
        trigger=_trigger_for(calib_tl), device=dev, batch_size=args.batch)
    res_pre_eval = eval_model(
        model, text_emb, eval_imgs, eval_labels, target_cls,
        trigger=_trigger_for(eval_tl), device=dev, batch_size=args.batch)
    asr_pre_search = float(res_pre_search["asr"])
    print(f"[phase 1] trigger-only ASR: search={asr_pre_search:.4f} "
          f"calib={float(res_pre_calib['asr']):.4f} "
          f"eval={float(res_pre_eval['asr']):.4f}")

    # ── PHASE 2a: NGR ──
    rows, mask, grad, ngr_loss = ngr_select_entries(
        model, text_emb, calib_imgs, calib_labels, target_cls,
        int(args.p2_rows), int(args.p2_entries), dev, int(args.p2_batch),
        args.ngr_loss)
    n_entries_tuned = int(mask.sum())
    print(f"[phase 2] NGR loss={ngr_loss:.4f} rows="
          f"{rows.tolist()} entries={n_entries_tuned}")

    # ── PHASE 2b: distillation ──
    pristine, keep, tuned = distill_proj(
        model, text_emb, calib_imgs, calib_labels, calib_tl, delta01, mask,
        target_vec, target_cls, patch_size, args, dev)
    pool_size = int(keep.sum())
    print(f"[phase 2] after pruning: pool_size={pool_size} "
          f"(pruned {n_entries_tuned - pool_size})")

    # ── PHASE 2c: bit flips ──
    qn = min(int(args.quick_n),
             (search_imgs if args.quick_split == "search" else calib_imgs).size(0))
    if args.quick_split == "search":
        quick_imgs, quick_labels = search_imgs[:qn], search_labels[:qn]
        quick_tl = search_tl[:qn] if search_tl.size(0) > 1 else search_tl
    else:
        quick_imgs, quick_labels = calib_imgs[:qn], calib_labels[:qn]
        quick_tl = calib_tl[:qn] if calib_tl.size(0) > 1 else calib_tl

    proj_param = ca.proj_param(model)

    def _quick_eval(proj_so_far):
        proj_param.data.copy_(proj_so_far.to(proj_param.dtype))
        return eval_model(
            model, text_emb, quick_imgs, quick_labels, target_cls,
            trigger=_trigger_for(quick_tl), device=dev,
            batch_size=args.batch)

    (flip_log, final_proj, scale, n_flips, quick_trace,
     flips_to_94, flips_to_95, flips_to_98, n_rejected) = realize_bit_flips(
        pristine, tuned, keep, int(args.max_flips), eval_fn=_quick_eval,
        tensor_name=proj_name, accept=args.flip_accept,
        cacc_tol=float(args.flip_cacc_tol), asr_tol=float(args.flip_asr_tol))
    proj_param.data.copy_(final_proj.to(proj_param.dtype))
    print(f"[phase 2] bit flips applied: {n_flips} (budget {args.max_flips}, "
          f"rejected entries {n_rejected})  proj_scale={scale:.6g}")

    # ── state: final + clean overrides for pre/post evaluation ──
    final_overrides = dict(clean_overrides)
    final_overrides[proj_name] = final_proj.cpu().float()
    clean_param_overrides = dict(clean_overrides)
    if proj_name not in clean_param_overrides:
        clean_param_overrides[proj_name] = pristine.cpu().float()

    meta = {
        "method": "trojvit",
        "model": args.model,
        "dataset": args.dataset,
        "seed": seed,
        "target_cls": target_cls,
        "n_flips": int(n_flips),
        "budget": int(args.max_flips),
        "search_time_s": round(time.time() - t0, 1),
        "trigger_asr_pre": asr_pre_search,
        "trigger_asr_pre_search": asr_pre_search,
        "trigger_asr_pre_calib": float(res_pre_calib["asr"]),
        "trigger_asr_pre_eval": float(res_pre_eval["asr"]),
        "trigger_cacc_pre_search": float(res_pre_search["cacc"]),
        "trigger_cacc_pre_eval": float(res_pre_eval["cacc"]),
        "pool_size": pool_size,
        "n_rows_selected": int(rows.numel()),
        "n_entries_tuned": n_entries_tuned,
        "flips_to_94": flips_to_94,
        "flips_to_95": flips_to_95,
        "flips_to_98": flips_to_98,
        "asr_at_k": asr_at_k_from_trace([(n, a) for n, a, _ in quick_trace]),
        "quick_split": args.quick_split,
        "quick_n": int(qn),
        "quick_asr_trace": quick_trace,
        "tar": tar,
        "num_patch": num_patch,
        "patch_size": int(patch_size),
        "patch_select": args.trojvit_patch_select,
        "patch_placement": args.patch_placement,
        "attn_query": args.attn_query,
        "trojvit_mode": args.trojvit_mode,
        "trojvit_loss": args.trojvit_loss,
        "trojvit_iters": int(args.trojvit_iters),
        "trojvit_lr": float(args.trojvit_lr),
        "trojvit_atten_weight": float(args.trojvit_atten_weight),
        "victim_quant": args.victim_quant,
        "ngr_loss": args.ngr_loss,
        "p2_rows": int(args.p2_rows),
        "p2_entries": int(args.p2_entries),
        "p2_epochs": int(args.p2_epochs),
        "p2_grad_surgery": bool(args.p2_grad_surgery),
        "p2_iterative_prune": bool(args.p2_iterative_prune),
        "p2_prune_e": float(args.p2_prune_e),
        "flip_accept": args.flip_accept,
        "n_rejected_entries": int(n_rejected),
        "n_calib": int(calib_imgs.size(0)),
        "n_search": int(search_imgs.size(0)),
        "n_eval": int(eval_imgs.size(0)),
        "proj_scale": float(scale),
        "quant_scales": {k: v for k, v in scales.items()
                         if k == proj_name},
    }

    state = {
        "method": "trojvit",
        "model": args.model,
        "dataset": args.dataset,
        "seed": seed,
        "target_cls": target_cls,
        "param_overrides": final_overrides,
        "clean_param_overrides": clean_param_overrides,
        "trigger": trigger,
        "flip_log": flip_log,
        "meta": meta,
    }
    torch.save(state, os.path.join(args.out, "attack_state.pt"))
    with open(os.path.join(args.out, "attack_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY — CORRECTED TROJVIT (TRIGGER + EXACT PROJ BIT FLIPS)")
    print("=" * 60)
    print(f"  Victim quant    : {args.victim_quant}")
    print(f"  TAR             : {tar:.2f}%  ({num_patch}x{patch_size}px)")
    print(f"  Placement       : {args.patch_placement}")
    print(f"  trigger ASR pre : search={asr_pre_search:.4f} "
          f"eval={float(res_pre_eval['asr']):.4f}")
    print(f"  rows / entries  : {int(rows.numel())} / {n_entries_tuned} "
          f"(pool {pool_size})")
    print(f"  bit flips       : {n_flips} / {args.max_flips}")
    print(f"  search time     : {meta['search_time_s']}s")
    print(f"  outputs         : {args.out}/attack_state.pt + "
          f"attack_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
