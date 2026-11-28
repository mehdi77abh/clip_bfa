"""
attack_proflip.py — corrected ProFlip baseline for the CLIP bit-flip suite.
===========================================================================

This is a *corrected* standalone port of ProFlip (Chen et al., ICCV 2021,
"ProFlip: Targeted Trojan Attack with Progressive Bit Flips") to CLIP
visual encoders.  It lives next to its own copies of ``clip_adapters.py``,
``evaluate.py``, ``data_split.json`` and ``class_texts/`` so it can be run
without touching ``proposed_method/modular_new_pipeline/bfa_baselines/``.

What is fixed relative to the first adaptation
(``.../bfa_baselines/attack_proflip.py``)
-------------------------------------------------------------------------------
1. **CBS acceptance uses the same loss on both sides.**  The old code
   compared ``L_troj(candidate)`` (with clean CE) against ``L_CBS(current)``
   (without clean CE), which made the gated path reject every candidate.
   Here ``loss_troj_cur`` and ``cand["loss"]`` are both ``L_troj``.
2. **The trigger is optimised on the quantised victim.**  The old code
   generated the trigger on fp32 and only then quantised the surface, so the
   trigger, ``c`` and ``y_calib`` were all for a different model.  Here the
   surface is quantised first; SNI, trigger and CBS all see the same
   (dequantised int8) victim.
3. **SNI is actually SNI.**
   - ``--sni-method analytic`` uses the *signed* per-dim influence
     ``proj @ t_target`` (not ``abs``), so a negative-influence dim is pushed
     down rather than up.
   - ``--sni-method jsma`` implements the paper's pairwise JSMA saliency
     (target vs. the sum of the other classes) on the pre-projection feature,
     with the paper's ``theta=0.1`` / ``gamma=0.5`` perturbation loop and
     per-image set combination.  It is not the old one-step softmax-gradient
     approximation.
4. **CBS MSE anchor matches the release.**  ``--cbs-anchor self`` (default)
   builds ``y`` from the current poisoned features and only sets ``I_t`` to
   ``c``, so the MSE gradient exists only on the salient dims — exactly the
   release's ``y = model(data, nolast=True)[15]; y[:, I_t] = 10`` behaviour.
   ``--cbs-anchor clean`` keeps the old all-dim clean anchor.
5. **Clean CE uses the paper's adaptive weight.**  ``--clean-ce-mode ratio``
   (default) sets ``gamma1 = clean_ce_ratio * L_CBS / CE_clean`` each round
   (paper: ``gamma1 * L_ce ~ 0.1 * L_CBS``); ``fixed`` uses
   ``--clean-ce-weight``; ``off`` disables the term.
6. **Paper-faithful defaults.**  S1 once, S2 top-1, S3 unconditional argmin,
   no monotone constraint, ``Q = max|W|``, ``ASR_t=94``, ``n_max=100``.
   The release/legacy behaviour is available through the opt-out flags
   (``--cbs-rerank``, ``--cbs-gated``, ``--cbs-top-elems 8``,
   ``--monotone``, ``--q-mode signed``, ``--asr-target 98 --n-max 200``).
7. **Trigger recipe defaults are the paper's**, not the release's 10-step
   version: 100 GD steps, lr 0.1, a scale-adapted CE temperature, and
   ``mean3std`` clamp.  The release recipe is available as
   ``--trig-iters 10 --trig-lr 0.01 --clamp-mode const10``.
8. **CACC is tracked and can gate the search.**  Every accepted element logs
   clean accuracy; ``--cacc-floor`` (optional) rejects a flip that drops CACC
   below the floor instead of silently collapsing the model.
9. **Pre/post-flip evaluation.**  The state stores ``clean_param_overrides``
   (the quantised surface before any flip), so the local ``evaluate.py`` can
   report the pre-flip ASR/CACC as well.

Bit mechanics are unchanged and match ProFlip's release:
per-tensor symmetric int8 (``scale = max|W| / 127``, codes in ``[-127,127]``),
two's-complement bit indexing, Hamming-distance flip counting, and atomic
multi-bit element updates.

Usage
-----
    python attack_proflip.py --model ViT-B/16 --dataset Caltech101 \
        --seed 42 --target-cls 97 --device cuda \
        --out results/proflip/ViT-B16/Caltech101/s42_paper

Paper-faithful recipe (the defaults above):
    --sni-method jsma --sni-combine intersection --fitness mse_sni
    --cbs-anchor self --clean-ce-mode ratio --clean-ce-ratio 0.1
    --cbs-select-once --cbs-top-elems 1 --cbs-accept-always
    --q-mode abs --grid-k 20 --asr-target 94 --n-max 100
    --trig-iters 100 --trig-lr 0.1 --clamp-mode mean3std

Release/legacy recipe:
    --sni-method analytic --fitness mse_sni --cbs-anchor clean \
    --clean-ce-mode fixed --clean-ce-weight 2.0 --cbs-rerank \
    --cbs-gated --cbs-top-elems 8 --monotone --q-mode signed \
    --asr-target 98 --n-max 200 --trig-iters 10 --trig-lr 0.01 \
    --clamp-mode const10

Important: check ``trigger_asr_pre`` in the log before interpreting CBS.  A
dead trigger (ASR near 0) means CBS is being asked to create the backdoor from
scratch; the old runs all had this problem.

Outputs
-------
``{out}/attack_state.pt``:
    method, model, dataset, seed, target_cls,
    param_overrides          final dequantised surface values,
    clean_param_overrides    quantised surface before any flip (pre-flip eval),
    trigger                  {kind, delta01, mask, mode},
    flip_log                 one row per bit,
    meta                     provenance, losses, metrics, trace.

``{out}/attack_meta.json``: compact summary for the thesis table.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

# Local modules win over anything in site-packages; this directory carries its
# own adapted clip_adapters.py / evaluate.py / data_split.json / class_texts/.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import clip_adapters as ca
import evaluate as eval_api


# ═════════════════════════════════════════════════════════════════════════
#  0.  small helpers
# ═════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _norm(emb):
    """L2-normalise along the last dim (F.normalize equivalent)."""
    return emb / emb.norm(dim=-1, keepdim=True)


def _zs_logits(emb_norm, text_emb, scale):
    """Zero-shot logits [B, C] with an explicit temperature/scale."""
    return (emb_norm @ text_emb.t()) * scale


def _ce_to(logits, target_cls):
    """Mean cross-entropy toward class `target_cls`."""
    return (torch.logsumexp(logits, dim=-1) - logits[:, target_cls]).mean()


def _ce_labels(logits, labels):
    """Mean cross-entropy toward per-image `labels`."""
    return (torch.logsumexp(logits, dim=-1)
            - logits.gather(1, labels.view(-1, 1)).squeeze(1)).mean()


def bit_positions(old_code, new_code):
    """8-bit two's-complement Hamming positions, LSB first."""
    diff = (int(old_code) & 0xFF) ^ (int(new_code) & 0xFF)
    return [b for b in range(8) if (diff >> b) & 1]


# ═════════════════════════════════════════════════════════════════════════
#  1.  feature tap: the input of the last linear map into the joint space
# ═════════════════════════════════════════════════════════════════════════

class FeatureTap:
    """Capture the pre-projection feature ``h``.

    OpenAI CLIP (``VisionTransformer.forward``):
        x = self.ln_post(x[:, 0, :])
        x = x @ self.proj
    so the output of ``ln_post`` is exactly the input of the last linear map.

    SigLIP: the projection surface is ``head.mlp.fc2``; ``h`` is its input
    (the post-``fc1`` activation), captured with a forward_pre_hook.
    """

    def __init__(self, model):
        self.value = None
        if ca.backend_of(model) == "siglip":
            fc2 = model._m.vision_model.head.mlp.fc2
            self._handle = fc2.register_forward_pre_hook(self._pre_hook)
        else:
            self._handle = model.visual.ln_post.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.value = output

    def _pre_hook(self, module, inputs):
        x = inputs[0]
        self.value = x.squeeze(1) if x.dim() == 3 else x

    def close(self):
        self._handle.remove()


def _forward_h_emb(model, imgs_norm, tap):
    """One batch through the visual encoder.

    Returns (h [B, vision_width], emb_norm [B, D]); `h` keeps the graph."""
    raw = model.encode_image(imgs_norm)      # = h @ proj (+ bias)
    h = tap.value
    if h is None:
        raise RuntimeError(
            "FeatureTap did not capture h — the model forward did not hit the "
            "expected ln_post / head.mlp.fc2 hook.")
    return h, _norm(raw)


def capture_h(model, imgs01, tap, batch_size, device):
    """No-grad batched capture of h [N, vision_width] for [0,1] images."""
    hs = []
    with torch.no_grad():
        for i in range(0, imgs01.shape[0], batch_size):
            xb = ca.normalize(imgs01[i:i + batch_size], model).to(device)
            h, _ = _forward_h_emb(model, xb, tap)
            hs.append(h.detach())
    return torch.cat(hs, dim=0)


# ═════════════════════════════════════════════════════════════════════════
#  2.  quantised attack surface
# ═════════════════════════════════════════════════════════════════════════

def build_surface(model, requires_grad=False):
    """Quantise every 2-D attention/MLP/projection weight per-tensor int8.

    The dequantised values are written into the model, so every phase sees the
    same victim that ``evaluate.py`` will reconstruct from ``param_overrides``.
    Returns {name: dict(param, codes int64, scale)}.
    """
    surface_names = ca.surface_param_names(model)
    surface = {}
    for name, p in model.named_parameters():
        if p.dim() != 2 or name not in surface_names:
            continue
        codes, scale = ca.quantize_int8_per_tensor(p.data)
        surface[name] = dict(param=p, codes=codes.to(torch.int64),
                             scale=float(scale))
        p.data.copy_(ca.dequantize_int8(codes, scale))
        p.requires_grad_(requires_grad)
    if not surface:
        raise RuntimeError(
            "empty attack surface — no 2-D attention/MLP/projection weights "
            "matched (see clip_adapters.surface_param_names).")
    return surface


def _set_surface_grad(surface, flag):
    for e in surface.values():
        e["param"].requires_grad_(flag)


# ═════════════════════════════════════════════════════════════════════════
#  3.  Phase i — SNI
# ═════════════════════════════════════════════════════════════════════════

def compute_sni_analytic(model, text_emb, target_cls, sni_k):
    """Analytic SNI: signed per-dim influence of h on the target alignment.

    ``emb = normalize(h @ P)`` and ``cos(emb, t) = (hP . t) / ||hP||``, so
    ``d cos / d h_j ∝ (P @ t)_j``.  Select the top-|.| dims and return the
    sign of each influence, so the trigger target can push a negative dim
    down (``c_j = sign_j * c``) rather than always up.

    Returns (I_t LongTensor [K], signs float [K], score [W]).
    """
    proj = ca.proj_matrix(model)               # [width, embed]
    if proj is None:
        raise RuntimeError("projection surface is None — ViT/SigLIP expected")
    t_target = text_emb[target_cls]
    score = proj @ t_target                    # [width], signed influence
    k = int(min(int(sni_k), score.numel()))
    order = torch.argsort(score.abs(), descending=True)[:k]
    it_idx = order.sort().values.long()
    signs = torch.sign(score[it_idx])
    signs[signs == 0] = 1.0
    return it_idx, signs, score


def _jsma_pair(target_grad, others_grad, domain):
    """One JSMA saliency step (Papernot et al. / ProFlip Algorithm 1).

    Returns the best feature pair (p, q) or (None, None, 0.0).
    ``alpha = g_t[p] + g_t[q]`` must be > 0 and ``beta = g_o[p] + g_o[q]``
    must be < 0; score = alpha * |beta|.
    """
    w = target_grad.numel()
    alpha = target_grad[:, None] + target_grad[None, :]
    beta = others_grad[:, None] + others_grad[None, :]
    valid = (alpha > 0) & (beta < 0)
    valid.fill_diagonal_(False)
    if domain is not None:
        valid &= domain[:, None] & domain[None, :]
    if not bool(valid.any()):
        return None, None, 0.0
    sal = alpha * beta.abs() * valid.float()
    flat_idx = int(torch.argmax(sal).item())
    p = flat_idx // w
    q = flat_idx % w
    return p, q, float(sal.view(-1)[flat_idx].item())


def compute_sni_jsma(model, tap, text_emb, imgs01, target_cls, sni_k,
                     theta, gamma, combine, device, batch_size, max_steps):
    """Paper-style JSMA on the pre-projection feature.

    For each clean calibration image:
      * start from ``h0`` (the last-layer input) and perturb it directly;
      * one backward for the target logit and one for the sum of all logits
        gives ``g_t`` and ``g_o = g_sum - g_t`` (no O(C) Jacobian loop);
      * pick the best pair with the JSMA alpha/beta condition, add theta to
        both, remove out-of-range features from the search domain;
      * stop when the target is predicted or ||h - h0|| >= gamma.

    The per-image feature sets are combined by ``intersection`` (paper) or
    ``union`` (release); an empty intersection falls back to the union.
    Returns (I_t LongTensor [K], signs float [K], mean saliency or None).
    """
    proj = ca.proj_matrix(model)
    if proj is None:
        raise RuntimeError("projection surface is None — ViT/SigLIP expected")
    scale = model.logit_scale.exp()
    h_all = capture_h(model, imgs01, tap, batch_size, device)   # [N, W]
    per_sample = []
    n = h_all.shape[0]

    for i in range(n):
        h0 = h_all[i]
        h = h0.clone()
        amin, amax = float(h0.min().item()), float(h0.max().item())
        domain = torch.ones(h.numel(), dtype=torch.bool, device=h.device)
        selected = set()
        delta = 0.0
        for _ in range(int(max_steps)):
            hr = h.detach().requires_grad_(True)
            logits = _zs_logits(_norm((hr.unsqueeze(0) @ proj)),
                                text_emb, scale)[0]          # [C]
            if int(logits.argmax().item()) == int(target_cls):
                break
            g_t = torch.autograd.grad(logits[int(target_cls)], hr,
                                      retain_graph=True)[0]
            g_sum = torch.autograd.grad(logits.sum(), hr)[0]
            g_o = g_sum - g_t
            p, q, _ = _jsma_pair(g_t, g_o, domain)
            if p is None:
                break
            h[p] = h[p] + float(theta)
            h[q] = h[q] + float(theta)
            selected.add(int(p))
            selected.add(int(q))
            if not (amin <= float(h[p].item()) <= amax):
                domain[p] = False
            if not (amin <= float(h[q].item()) <= amax):
                domain[q] = False
            delta = float((h - h0).norm().item())
            if delta >= float(gamma):
                break
        if selected:
            per_sample.append(selected)
        print(f"\r[proflip]   JSMA SNI {i + 1}/{n}  |I_s|={len(selected)}",
              end="", flush=True)
    print()

    sets = [s for s in per_sample if s]
    if not sets:
        print("[proflip]   JSMA selected nothing — falling back to analytic SNI")
        return compute_sni_analytic(model, text_emb, target_cls, sni_k)

    if combine == "intersection":
        chosen = set.intersection(*sets)
        if not chosen:
            print("[proflip]   JSMA intersection empty — falling back to union")
            chosen = set.union(*sets)
    else:
        chosen = set.union(*sets)

    freq = Counter()
    for s in sets:
        freq.update(s)
    idx = sorted(chosen, key=lambda j: (-freq[j], j))
    if len(idx) > int(sni_k):
        idx = idx[:int(sni_k)]
    it_idx = torch.tensor(sorted(idx), dtype=torch.long, device=device)
    signs = torch.ones_like(it_idx, dtype=torch.float32)
    return it_idx, signs, None


# ═════════════════════════════════════════════════════════════════════════
#  4.  Phase ii — feature-anchored trigger
# ═════════════════════════════════════════════════════════════════════════

def patch_geometry(px):
    """Bottom-right square, proportional to ProFlip's 10/32 geometry.

    73 px on 224 px = 10.62 % TAP (ProFlip's ImageNet value); 32 px input
    gives 10 px (ProFlip's CIFAR value).  One pixel margin from the edges."""
    side = int(round(px * 73.0 / 224.0))
    top = px - side - 1
    left = px - side - 1
    return side, top, left


def build_clamp(h_clean, it_idx, clamp_mode):
    """Per-dim high-activation target c [K] for the SNI dims.

    max      : c_j = max over the calibration batch (scale-adapted ceiling)
    mean3std : c_j = mean + 3*std over the calibration batch (paper-style
               "high activation", feasible after LayerNorm)
    const10  : c_j = 10 (the paper/release literal)
    """
    sel = h_clean[:, it_idx]                     # [N, K]
    if clamp_mode == "max":
        c = sel.max(dim=0).values
    elif clamp_mode == "mean3std":
        c = sel.mean(dim=0) + 3.0 * sel.std(dim=0)
    elif clamp_mode == "const10":
        c = torch.full((sel.shape[1],), 10.0, dtype=sel.dtype,
                       device=sel.device)
    else:
        raise ValueError(f"unknown clamp mode {clamp_mode!r}")
    return c


def optimize_trigger(model, tap, calib01, y_calib, text_emb, target_cls,
                     px, args, device):
    """Feature-anchored trigger optimisation (ProFlip Phase ii).

    loss = MSE(h(x*), y_calib) + trig_ce_weight * CE_zs(x*, target)
    with ``y_calib = h(clean)`` and ``y_calib[:, I_t] = c * sign``.

    The gradient is the mean over the calibration split (mathematically equal
    to the release's ``sum(data.grad[box])`` for a shared patch, because the
    release loss is also batch-mean reduced).  Default recipe: 100 plain-GD
    steps, lr 0.1 (paper); use 10 / 0.01 for the release recipe.
    """
    side, top, left = patch_geometry(px)
    mask = torch.zeros(1, 1, px, px)
    mask[:, :, top:top + side, left:left + side] = 1.0
    mask_dev = mask.to(device)
    n = calib01.shape[0]
    scale = model.logit_scale.exp() / float(args.trig_temp)

    if args.trig_init == "rand":
        p = torch.randn(3, side, side, device=device) * 0.1
    else:
        p = torch.zeros(3, side, side, device=device)

    for it in range(int(args.trig_iters)):
        p = p.detach().requires_grad_(True)
        total = 0.0
        for i in range(0, n, args.batch_size):
            xb = calib01[i:i + args.batch_size].to(device)
            yb = y_calib[i:i + xb.shape[0]].to(device)
            delta = torch.zeros(1, 3, px, px, device=device)
            delta[:, :, top:top + side, left:left + side] = p
            poisoned = ca.apply_trigger(xb, delta, mask_dev, "replace")
            h, emb = _forward_h_emb(model, ca.normalize(poisoned, model), tap)
            loss = (((h - yb) ** 2).mean()
                    + float(args.trig_ce_weight)
                    * _ce_to(_zs_logits(emb, text_emb, scale), target_cls))
            w = xb.shape[0] / n
            (loss * w).backward()
            total += float(loss.detach()) * w
        if p.grad is None:
            raise RuntimeError("trigger gradient is None — check the loss path")
        with torch.no_grad():
            p = (p - float(args.trig_lr) * p.grad).clamp(0.0, 1.0)
        if it == 0 or (it + 1) % max(1, int(args.trig_iters) // 10) == 0:
            print(f"[proflip]   trigger step {it + 1}/{args.trig_iters}  "
                  f"loss={total:.4f}")

    delta01 = torch.zeros(1, 3, px, px)
    delta01[:, :, top:top + side, left:left + side] = p.detach().cpu()
    trigger = {"kind": "single", "delta01": delta01,
               "mask": mask, "mode": "replace"}
    return trigger


# ═════════════════════════════════════════════════════════════════════════
#  5.  Phase iii — fitness / losses
# ═════════════════════════════════════════════════════════════════════════

def _fitness_cbs_batch(model, tap, xb, yb, c_vec, it_idx, anchor,
                       text_emb, target_cls, mode, scale, batch_size, device):
    """L_CBS for one batch (no clean-preservation term).

    mse_sni : MSE(h, y) + CE_zs(logits, target)
              anchor 'self'  -> y = h.detach(); y[:, I_t] = c
              anchor 'clean' -> y = y_search (all dims clean-anchored)
    cos     : 1 - cos(emb, t_target)
    """
    h, emb = _forward_h_emb(model, xb, tap)
    if mode == "mse_sni":
        if anchor == "self":
            target = h.detach().clone()
            target[:, it_idx] = c_vec
        else:
            target = yb
        return (((h - target) ** 2).mean()
                + _ce_to(_zs_logits(emb, text_emb, scale), target_cls))
    return 1.0 - (emb * text_emb[target_cls]).sum(dim=-1).mean()


def fitness_cbs(model, tap, poisoned_norm, y_search, c_vec, it_idx, anchor,
                text_emb, target_cls, mode, scale, batch_size, device):
    """No-grad L_CBS over the poisoned search batch."""
    n = poisoned_norm.shape[0]
    total = 0.0
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = poisoned_norm[i:i + batch_size].to(device)
            yb = (y_search[i:i + xb.shape[0]].to(device)
                  if y_search is not None else None)
            loss = _fitness_cbs_batch(
                model, tap, xb, yb, c_vec, it_idx, anchor,
                text_emb, target_cls, mode, scale, batch_size, device)
            total += float(loss) * (xb.shape[0] / n)
    return total


def fitness_grad(model, surface, tap, poisoned_norm, y_search, c_vec, it_idx,
                 anchor, text_emb, target_cls, mode, scale,
                 batch_size, device):
    """One backward of L_CBS on the surface tensors; returns L_CBS."""
    for e in surface.values():
        e["param"].grad = None
    n = poisoned_norm.shape[0]
    total = 0.0
    for i in range(0, n, batch_size):
        xb = poisoned_norm[i:i + batch_size].to(device)
        yb = (y_search[i:i + xb.shape[0]].to(device)
              if y_search is not None else None)
        loss = _fitness_cbs_batch(
            model, tap, xb, yb, c_vec, it_idx, anchor,
            text_emb, target_cls, mode, scale, batch_size, device)
        w = xb.shape[0] / n
        (loss * w).backward()
        total += float(loss.detach()) * w
    return total


def clean_ce_value(model, tap, clean_norm, clean_labels, text_emb, scale,
                   batch_size, device):
    """No-grad clean CE(M(x), labels) — the stealthiness term of L_troj."""
    if clean_norm is None or clean_labels is None:
        return 0.0
    n = clean_norm.shape[0]
    total = 0.0
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = clean_norm[i:i + batch_size].to(device)
            lb = clean_labels[i:i + xb.shape[0]].to(device)
            _, emb = _forward_h_emb(model, xb, tap)
            loss = _ce_labels(_zs_logits(emb, text_emb, scale), lb)
            total += float(loss) * (xb.shape[0] / n)
    return total


def fitness_maps(surface, q_mode):
    """S1/S2 fitness F = |dL/dw| * (Q - w) on entries with negative grad.

    Q = max(w) (release, 'signed') or max|w| (paper, 'abs').
    """
    out = {}
    for name, e in surface.items():
        g = e["param"].grad
        if g is None:
            continue
        w = e["param"].data
        Q = w.max() if q_mode == "signed" else w.abs().max()
        step = torch.where(g < 0, (Q - w).clamp(min=0),
                           torch.zeros_like(w))
        out[name] = torch.nan_to_num(g.abs() * step, nan=0.0,
                                     posinf=0.0, neginf=0.0)
    return out


# ═════════════════════════════════════════════════════════════════════════
#  6.  Phase iii — S3 value grid
# ═════════════════════════════════════════════════════════════════════════

def grid_search_element(model, surface, name, i, j, tap, poisoned_norm,
                        y_search, clean_norm, clean_labels, text_emb,
                        target_cls, c_vec, it_idx, anchor, mode, scale,
                        gamma_clean, batch_size, device, args):
    """Grid-search K int8 values for element (i, j) and return the best.

    The grid is over int8 codes (K=20 by default) in [-R, R], with
    R = max|code| of the tensor.  ``--monotone`` restricts to new >= old;
    the paper/release do not impose it.  Candidate loss is
    ``L_troj = L_CBS + gamma_clean * CE_clean``.
    """
    e = surface[name]
    param, qscale, codes = e["param"], e["scale"], e["codes"]
    old_code = int(codes[i, j].item())
    R = int(codes.abs().max().item())
    if R <= 0:
        return None
    grid = torch.linspace(-R, R, int(args.grid_k)).round()
    grid = grid.clamp(-127, 127).to(torch.int64).unique()
    if args.monotone:
        grid = grid[grid >= old_code]
    old_fp = float(param.data[i, j].item())

    best = None
    try:
        for code_t in grid.tolist():
            code = int(code_t)
            if code == old_code:
                continue
            param.data[i, j] = code * qscale
            loss_cbs = fitness_cbs(
                model, tap, poisoned_norm, y_search, c_vec, it_idx, anchor,
                text_emb, target_cls, mode, scale, batch_size, device)
            loss_ce = (clean_ce_value(model, tap, clean_norm, clean_labels,
                                      text_emb, scale, batch_size, device)
                       if gamma_clean > 0 else 0.0)
            loss = loss_cbs + float(gamma_clean) * float(loss_ce)
            if not math.isfinite(loss):
                continue
            if best is None or loss < best["loss"] - 1e-12:
                best = dict(tensor=name, i=int(i), j=int(j),
                            old_code=old_code, new_code=code,
                            bits=bit_positions(old_code, code),
                            old_fp=old_fp, new_fp=float(code) * qscale,
                            loss=float(loss), loss_cbs=float(loss_cbs),
                            loss_ce=float(loss_ce))
    finally:
        param.data[i, j] = old_fp

    if best is None or best["new_code"] == old_code:
        return None
    return best


# ═════════════════════════════════════════════════════════════════════════
#  7.  Phase iii — the CBS loop
# ═════════════════════════════════════════════════════════════════════════

def quick_metrics(model, text_emb, imgs01, labels, target_cls, trigger,
                  device, batch_size):
    """ASR/CACC [%] on a search-time subset via the shared evaluator."""
    res = eval_api.eval_model(model, text_emb, imgs01, labels, target_cls,
                              trigger=trigger, device=device,
                              batch_size=batch_size)
    asr = res["asr"]
    cacc = res["cacc"]
    asr = 100.0 * asr if asr == asr else float("nan")
    cacc = 100.0 * cacc if cacc == cacc else float("nan")
    return asr, cacc, res


def _current_gamma(args, loss_cbs, ce_clean):
    if args.clean_ce_mode == "off" or ce_clean is None:
        return 0.0
    if args.clean_ce_mode == "fixed":
        return float(args.clean_ce_weight)
    return float(args.clean_ce_ratio) * float(loss_cbs) / max(float(ce_clean),
                                                              1e-12)


def critical_bit_search(model, surface, tap, trigger, poisoned_norm, y_search,
                        clean_norm, clean_labels, text_emb, quick01,
                        quick_labels, target_cls, c_vec, it_idx, args, device):
    """ProFlip CBS: S1 tensor -> S2 element(s) -> S3 value grid.

    Defaults are the paper's: S1 selected once, S2 top-1, S3 unconditional
    argmin, no monotone constraint.  The acceptance path compares L_troj to
    L_troj; ``--cbs-gated`` opts into an improvement gate, and
    ``--cbs-accept-always`` (default) is the paper's unconditional argmin.
    """
    aux_bs = int(args.eval_batch)
    scale = model.logit_scale.exp() / float(args.fit_temp)

    vn = max(1, min(int(args.verify_n), poisoned_norm.size(0)))
    poisoned_norm = poisoned_norm[:vn]
    if y_search is not None:
        y_search = y_search[:vn]
    if clean_norm is not None:
        clean_norm = clean_norm[:vn]
    if clean_labels is not None:
        clean_labels = clean_labels[:vn]

    def quick():
        return quick_metrics(model, text_emb, quick01, quick_labels,
                             target_cls, trigger, device, aux_bs)

    print("\n" + "=" * 64)
    print("PHASE III: PROFLIP CRITICAL BIT SEARCH (S1/S2/S3)")
    print("=" * 64)
    print(f"  surface tensors  : {len(surface)}")
    print(f"  fitness          : {args.fitness}  (CE scale "
          f"logit_scale/{args.fit_temp:g})")
    print(f"  CBS anchor       : {args.cbs_anchor}")
    print(f"  clean CE         : {args.clean_ce_mode}"
          + (f" ratio={args.clean_ce_ratio:g}"
             if args.clean_ce_mode == "ratio" else
             f" weight={args.clean_ce_weight:g}"
             if args.clean_ce_mode == "fixed" else ""))
    print(f"  S1               : {'once' if args.cbs_select_once else 're-rank'}"
          f"   S2: top-{args.cbs_top_elems}   S3: "
          f"{'argmin' if args.cbs_accept_always else 'gated'}")
    print(f"  S3 grid          : K={args.grid_k} over [-R, R]"
          f"  monotone={args.monotone}")
    print(f"  verify subset    : {vn} images; ASR/CACC trace on "
          f"quick[:{int(args.quick_n)}]")
    print(f"  stop             : ASR >= {args.asr_target}%  or  bits >= "
          f"{args.n_max}  or pool exhausted"
          + (f"  or CACC < {args.cacc_floor}%" if args.cacc_floor is not None
             else ""))

    flip_log = []
    quick_trace = []
    n_bits = 0
    n_elements = 0
    rid = 0
    pinned_name = None
    stop_reason = "budget"
    t0 = time.time()

    loss_cbs_cur = fitness_grad(
        model, surface, tap, poisoned_norm, y_search, c_vec, it_idx,
        args.cbs_anchor, text_emb, target_cls, args.fitness, scale,
        args.batch_size, device)
    ce_cur = (clean_ce_value(model, tap, clean_norm, clean_labels, text_emb,
                             scale, args.batch_size, device)
              if args.clean_ce_mode != "off" else 0.0)
    gamma_cur = _current_gamma(args, loss_cbs_cur, ce_cur)
    loss_troj_cur = loss_cbs_cur + gamma_cur * ce_cur

    asr_quick, cacc_quick, _ = quick()
    quick_trace.append((0, asr_quick, cacc_quick))
    print(f"  baseline         : L_CBS={loss_cbs_cur:.4f}  "
          f"CE_clean={ce_cur:.4f}  gamma1={gamma_cur:.4g}  "
          f"L_troj={loss_troj_cur:.4f}")
    print(f"                     quick ASR={asr_quick:.1f}%  "
          f"CACC={cacc_quick:.1f}%")
    flips_to_94 = 0 if asr_quick >= 94.0 - 1e-6 else None
    flips_to_95 = 0 if asr_quick >= 95.0 - 1e-6 else None
    flips_to_98 = 0 if asr_quick >= 98.0 - 1e-6 else None

    while True:
        if asr_quick >= args.asr_target - 1e-6:
            stop_reason = "asr_target"
            break
        if n_bits >= args.n_max:
            stop_reason = "budget"
            break

        # ── S1: one backward, per-tensor fitness, pick max-F tensor ──────
        loss_cbs_cur = fitness_grad(
            model, surface, tap, poisoned_norm, y_search, c_vec, it_idx,
            args.cbs_anchor, text_emb, target_cls, args.fitness, scale,
            args.batch_size, device)
        ce_cur = (clean_ce_value(model, tap, clean_norm, clean_labels,
                                 text_emb, scale, args.batch_size, device)
                  if args.clean_ce_mode != "off" else 0.0)
        gamma_cur = _current_gamma(args, loss_cbs_cur, ce_cur)
        loss_troj_cur = loss_cbs_cur + gamma_cur * ce_cur

        Fmaps = fitness_maps(surface, args.q_mode)
        if (args.cbs_select_once and pinned_name is not None
                and pinned_name in Fmaps):
            name = pinned_name
        else:
            order = sorted(
                ((n_, F_.max().item()) for n_, F_ in Fmaps.items()
                 if F_.max().item() > 0.0),
                key=lambda kv: kv[1], reverse=True)
            if not order:
                stop_reason = "pool_exhausted"
                print("  no tensor with positive fitness — stopping")
                break
            name = order[0][0]
            if args.cbs_select_once:
                pinned_name = name

        # ── S2: vulnerable element set inside that tensor ────────────────
        Fmap = Fmaps[name]
        flat = Fmap.flatten()
        ncols = Fmap.shape[1]
        k = min(int(args.cbs_top_elems), int(flat.numel()))
        top_vals, top_idx = torch.topk(flat, k)
        elems = []
        for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
            if val <= 0.0:
                break
            i, j = divmod(int(idx), ncols)
            elems.append((i, j, float(val)))

        # ── S3: grid-search every vulnerable element, collect candidates ─
        cands = []
        for i, j, fval in elems:
            cand = grid_search_element(
                model, surface, name, i, j, tap, poisoned_norm, y_search,
                clean_norm, clean_labels, text_emb, target_cls, c_vec, it_idx,
                args.cbs_anchor, args.fitness, scale, gamma_cur,
                args.batch_size, device, args)
            if cand is not None:
                cand["fitness"] = fval
                cands.append(cand)
        if not cands:
            stop_reason = "pool_exhausted"
            print("  no element yields a value change — stopping")
            break
        cands.sort(key=lambda c: c["loss"])

        # ── acceptance: paper argmin, or loss/CACC-gated ─────────────────
        accepted = None
        asr_after, cacc_after = asr_quick, cacc_quick
        for cand in cands:
            param = surface[cand["tensor"]]["param"]
            param.data[cand["i"], cand["j"]] = cand["new_fp"]
            if (not args.cbs_accept_always
                    and cand["loss"] >= loss_troj_cur - 1e-12):
                param.data[cand["i"], cand["j"]] = cand["old_fp"]
                continue
            a, c, _ = quick()
            if (args.cacc_floor is not None
                    and c < args.cacc_floor - 1e-9):
                param.data[cand["i"], cand["j"]] = cand["old_fp"]
                continue
            accepted = cand
            asr_after, cacc_after = a, c
            break
        if accepted is None:
            stop_reason = ("cacc_floor" if args.cacc_floor is not None
                           else "pool_exhausted")
            print("  no candidate passes the acceptance gate — stopping")
            break

        # ── bookkeeping (one flip_log row per bit) ──────────────────────
        cand = accepted
        surface[cand["tensor"]]["codes"][cand["i"], cand["j"]] = cand["new_code"]
        bits = list(cand["bits"])
        n_this = len(bits)
        n_bits += n_this
        n_elements += 1
        loss_before = loss_troj_cur
        loss_cbs_before = loss_cbs_cur
        ce_before = ce_cur
        gamma_before = gamma_cur
        loss_troj_cur = cand["loss"]
        loss_cbs_cur = cand["loss_cbs"]
        ce_cur = cand["loss_ce"]
        asr_quick = asr_after
        cacc_quick = cacc_after
        quick_trace.append((n_bits, asr_quick, cacc_quick))
        for b in bits:
            flip_log.append(dict(
                target="weight",
                tensor=cand["tensor"],
                index=[cand["i"], cand["j"]],
                bit=int(b),
                old_int8=cand["old_code"], new_int8=cand["new_code"],
                old_fp32=cand["old_fp"], new_fp32=cand["new_fp"],
                fitness=cand["fitness"],
                old=cand["old_code"], new=cand["new_code"],
                cbs_round=rid, cbs_element_bits=bits, cbs_n_bits=n_this,
                loss_before=float(loss_before),
                loss_after=float(cand["loss"]),
                loss_cbs_before=float(loss_cbs_before),
                loss_cbs_after=float(cand["loss_cbs"]),
                ce_clean_before=float(ce_before),
                ce_clean_after=float(cand["loss_ce"]),
                gamma_clean=float(gamma_before)))
        if flips_to_94 is None and asr_quick >= 94.0 - 1e-6:
            flips_to_94 = n_bits
        if flips_to_95 is None and asr_quick >= 95.0 - 1e-6:
            flips_to_95 = n_bits
        if flips_to_98 is None and asr_quick >= 98.0 - 1e-6:
            flips_to_98 = n_bits
        print(f"  Rd {rid:>3d}: {cand['tensor']}[{cand['i']},{cand['j']}]  "
              f"code {cand['old_code']}->{cand['new_code']}  bits={n_this} "
              f"(total {n_bits})  L_troj {loss_before:.4f}->"
              f"{cand['loss']:.4f}  gamma1={gamma_before:.4g}  "
              f"ASR={asr_quick:.1f}%  CACC={cacc_quick:.1f}%")
        rid += 1

    search_time = time.time() - t0
    asr_final, cacc_final, _ = quick()
    print(f"\n  CBS elements: {n_elements}   bits: {n_bits}   "
          f"stop: {stop_reason}")
    print(f"  quick ASR/CACC: {asr_final:.1f}% / {cacc_final:.1f}%   "
          f"flips_to_94={flips_to_94}  flips_to_95={flips_to_95}  "
          f"flips_to_98={flips_to_98}")

    return dict(
        flip_log=flip_log, n_bits=n_bits, n_elements=n_elements,
        asr_quick=asr_final, cacc_quick=cacc_final,
        flips_to_94=flips_to_94, flips_to_95=flips_to_95,
        flips_to_98=flips_to_98, stop_reason=stop_reason,
        search_time_s=search_time, quick_trace=quick_trace)


# ═════════════════════════════════════════════════════════════════════════
#  8.  orchestration + outputs
# ═════════════════════════════════════════════════════════════════════════

def run_attack(args):
    t_all = time.time()
    set_seed(args.seed)
    device = args.device
    px = ca.input_px(args.model)

    # ── data: frozen disjoint splits (same indices for every method) ─────
    with open(args.data_split) as f:
        splits = json.load(f)
    sp = splits[args.dataset]
    print(f"[proflip] {args.model} / {args.dataset} / seed {args.seed} / "
          f"target {args.target_cls}  (fitness={args.fitness})")
    calib01, _ = ca.load_images(args.dataset, sp["calib"], px, args.data_root)
    search01, search_labels = ca.load_images(
        args.dataset, sp["search"], px, args.data_root)
    qn = max(1, min(int(args.quick_n), search01.size(0)))
    quick01, quick_labels = search01[:qn], search_labels[:qn]

    # ── model + zero-shot text head ──────────────────────────────────────
    model = ca.load_model(args.model, device)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    names = ca.load_classnames(args.dataset)
    if not 0 <= args.target_cls < len(names):
        raise ValueError(f"--target-cls {args.target_cls} out of range "
                         f"({len(names)} classes)")
    text_emb = ca.build_text_embeddings(model, names, device)

    # ── Phase 0: quantise FIRST, so every later phase sees the victim ───
    surface = build_surface(model, requires_grad=False)
    print(f"[proflip] quantised surface: {len(surface)} tensors")
    tap = FeatureTap(model)

    # ── Phase i: SNI on the quantised victim ─────────────────────────────
    sni_n = max(1, min(int(args.sni_n), calib01.size(0)))
    if args.sni_method == "jsma":
        it_idx, signs, _ = compute_sni_jsma(
            model, tap, text_emb, calib01[:sni_n], args.target_cls,
            args.sni_k, args.sni_theta, args.sni_gamma, args.sni_combine,
            device, args.batch_size, args.sni_max_steps)
        sni_desc = (f"JSMA theta={args.sni_theta} gamma={args.sni_gamma} "
                    f"combine={args.sni_combine} n={sni_n}")
    else:
        it_idx, signs, _ = compute_sni_analytic(
            model, text_emb, args.target_cls, args.sni_k)
        sni_desc = "analytic signed |proj . t_target|"
    it_dev = it_idx.to(device)
    signs_dev = signs.to(device)
    print(f"[proflip] Phase i  SNI ({sni_desc}): I_t = {it_idx.numel()} dims")

    # ── Phase ii: trigger on the quantised victim ────────────────────────
    t_ph2 = time.time()
    h_calib = capture_h(model, calib01, tap, args.batch_size, device)
    c = build_clamp(h_calib, it_dev, args.clamp_mode)
    c_target = c * signs_dev
    y_calib = h_calib.clone()
    y_calib[:, it_dev] = c_target
    trigger = optimize_trigger(model, tap, calib01, y_calib, text_emb,
                               args.target_cls, px, args, device)
    side, top, left = patch_geometry(px)
    print(f"[proflip] Phase ii trigger: patch {side}x{side} @ ({top},{left}), "
          f"clamp={args.clamp_mode}, {args.trig_iters} steps, "
          f"lr={args.trig_lr}, temp={args.trig_temp}")
    trigger_time_s = time.time() - t_ph2

    # ── Phase iii: enable surface grads, then CBS ────────────────────────
    _set_surface_grad(surface, True)
    clean_overrides = {
        name: e["param"].data.detach().cpu().clone().float()
        for name, e in surface.items()}

    poisoned01 = eval_api.apply_trigger_dict(search01, trigger)
    poisoned_norm = ca.normalize(poisoned01, model)
    clean_norm = ca.normalize(search01, model)
    y_search = None
    if args.cbs_anchor == "clean":
        y_search = capture_h(model, search01, tap, args.batch_size, device)
        y_search[:, it_dev] = c_target

    trig_asr_pre, trig_cacc_pre, _ = quick_metrics(
        model, text_emb, quick01, quick_labels, args.target_cls, trigger,
        device, args.eval_batch)
    print(f"[proflip] trigger-only quick ASR/CACC: "
          f"{trig_asr_pre:.1f}% / {trig_cacc_pre:.1f}%")

    cbs = critical_bit_search(
        model, surface, tap, trigger, poisoned_norm, y_search, clean_norm,
        search_labels, text_emb, quick01, quick_labels, args.target_cls,
        c_target, it_dev, args, device)
    tap.close()

    # ── outputs ──────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    param_overrides = {
        name: e["param"].data.detach().cpu().clone().float()
        for name, e in surface.items()}
    meta = dict(
        method="proflip",
        fitness=args.fitness,
        sni_method=args.sni_method,
        sni_combine=args.sni_combine,
        sni_k=int(it_idx.numel()),
        sni_indices=[int(x) for x in it_idx.tolist()],
        sni_signs=[float(x) for x in signs.tolist()],
        clamp_mode=args.clamp_mode,
        clamp_values=[float(x) for x in c.detach().cpu().tolist()],
        patch=dict(side=side, top=top, left=left),
        tar=round((side / float(px)) ** 2 * 100, 2),
        trig_iters=int(args.trig_iters), trig_lr=float(args.trig_lr),
        trig_temp=float(args.trig_temp), trig_init=args.trig_init,
        trig_ce_weight=float(args.trig_ce_weight),
        trigger_time_s=round(trigger_time_s, 1),
        fit_temp=float(args.fit_temp),
        cbs_anchor=args.cbs_anchor,
        clean_ce_mode=args.clean_ce_mode,
        clean_ce_ratio=float(args.clean_ce_ratio),
        clean_ce_weight=float(args.clean_ce_weight),
        grid_k=int(args.grid_k), cbs_top_elems=int(args.cbs_top_elems),
        cbs_select_once=bool(args.cbs_select_once),
        cbs_accept_always=bool(args.cbs_accept_always),
        monotone=bool(args.monotone), q_mode=args.q_mode,
        asr_target=float(args.asr_target), n_max=int(args.n_max),
        cacc_floor=(None if args.cacc_floor is None
                    else float(args.cacc_floor)),
        pool_size=len(surface),
        surface_scales={n_: float(e["scale"]) for n_, e in surface.items()},
        n_flips=int(cbs["n_bits"]), n_elements=int(cbs["n_elements"]),
        flips_to_94=cbs["flips_to_94"], flips_to_95=cbs["flips_to_95"],
        flips_to_98=cbs["flips_to_98"],
        asr_at_k=eval_api.asr_at_k_from_trace(
            [(n, a / 100.0) for n, a, _ in cbs["quick_trace"]]),
        cacc_at_k=eval_api.asr_at_k_from_trace(
            [(n, c_ / 100.0) for n, _, c_ in cbs["quick_trace"]]),
        trigger_asr_pre=(None if trig_asr_pre != trig_asr_pre
                         else round(trig_asr_pre, 4)),
        trigger_cacc_pre=(None if trig_cacc_pre != trig_cacc_pre
                          else round(trig_cacc_pre, 4)),
        quick_split="search[:quick_n]",
        quick_asr_trace=[(n, a) for n, a, _ in cbs["quick_trace"]],
        quick_cacc_trace=[(n, c_) for n, _, c_ in cbs["quick_trace"]],
        stop_reason=cbs["stop_reason"],
        search_time_s=round(cbs["search_time_s"], 1),
        batch_size=int(args.batch_size), eval_batch=int(args.eval_batch),
        total_time_s=round(time.time() - t_all, 1),
    )
    state = dict(
        method="proflip",
        model=args.model,
        dataset=args.dataset,
        seed=int(args.seed),
        target_cls=int(args.target_cls),
        param_overrides=param_overrides,
        clean_param_overrides=clean_overrides,
        trigger=trigger,
        flip_log=cbs["flip_log"],
        meta=meta,
    )
    torch.save(state, os.path.join(args.out, "attack_state.pt"))

    attack_meta = {
        "method": "proflip",
        "model": args.model,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "target_cls": int(args.target_cls),
        "fitness": args.fitness,
        "n_flips": int(cbs["n_bits"]),
        "n_elements": int(cbs["n_elements"]),
        "budget": int(args.n_max),
        "search_time_s": round(cbs["search_time_s"], 1),
        "trigger_asr_pre": (None if trig_asr_pre != trig_asr_pre
                            else round(trig_asr_pre, 4)),
        "trigger_cacc_pre": (None if trig_cacc_pre != trig_cacc_pre
                             else round(trig_cacc_pre, 4)),
        "pool_size": len(surface),
        "flips_to_94": cbs["flips_to_94"],
        "flips_to_95": cbs["flips_to_95"],
        "flips_to_98": cbs["flips_to_98"],
        "asr_at_k": eval_api.asr_at_k_from_trace(
            [(n, a / 100.0) for n, a, _ in cbs["quick_trace"]]),
        "cacc_at_k": eval_api.asr_at_k_from_trace(
            [(n, c_ / 100.0) for n, _, c_ in cbs["quick_trace"]]),
        "sni_k": int(it_idx.numel()),
        "sni_method": args.sni_method,
        "cbs_anchor": args.cbs_anchor,
        "clean_ce_mode": args.clean_ce_mode,
        "stop_reason": cbs["stop_reason"],
    }
    with open(os.path.join(args.out, "attack_meta.json"), "w") as f:
        json.dump(attack_meta, f, indent=2)

    print(f"[proflip] wrote {args.out}/attack_state.pt + attack_meta.json  "
          f"({cbs['n_bits']} bits in {cbs['search_time_s']:.0f}s)")
    return state


# ═════════════════════════════════════════════════════════════════════════
#  9.  CLI
# ═════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="Corrected ProFlip (SNI + native trigger + CBS) for CLIP")
    # ── run identity ─────────────────────────────────────────────────────
    p.add_argument("--model", required=True, choices=list(ca.MODEL_REGISTRY),
                   help="OpenAI CLIP / SigLIP victim (fp32)")
    p.add_argument("--dataset", required=True,
                   choices=list(ca.DATASET_REGISTRY),
                   help="Caltech101 | ImageNet1k | DTD | Food101")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-cls", type=int, required=True,
                   help="target class index in the dataset's class list")
    p.add_argument("--data-split",
                   default=os.path.join(_HERE, "data_split.json"),
                   help="frozen disjoint split file (shared by all methods)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-root", default=None,
                   help="override clip_adapters' dataset root")

    # ── CBS fitness / losses ─────────────────────────────────────────────
    p.add_argument("--fitness", choices=["mse_sni", "cos"],
                   default="mse_sni",
                   help="mse_sni = ProFlip L_CBS (MSE(h,y)+CE_zs); "
                        "cos = 1-cos(emb,t_target) comparison")
    p.add_argument("--fit-temp", type=float, default=10.0,
                   help="CE temperature divisor for the fitness/logit scale "
                        "(default 10; 1 = full CLIP logit_scale)")
    p.add_argument("--cbs-anchor", choices=["self", "clean"],
                   default="self",
                   help="CBS MSE target: self = poisoned features with only "
                        "I_t set to c (release-faithful); clean = all-dim "
                        "clean anchor (old adaptation)")
    p.add_argument("--clean-ce-mode", choices=["ratio", "fixed", "off"],
                   default="ratio",
                   help="clean-preservation term: ratio = gamma1 ~ "
                        "clean_ce_ratio * L_CBS / CE (paper); fixed = "
                        "--clean-ce-weight; off = no clean term")
    p.add_argument("--clean-ce-ratio", type=float, default=0.1,
                   help="target ratio for --clean-ce-mode ratio (paper: 0.1)")
    p.add_argument("--clean-ce-weight", type=float, default=1.0,
                   help="gamma1 for --clean-ce-mode fixed")

    # ── Phase i (SNI) ────────────────────────────────────────────────────
    p.add_argument("--sni-method", choices=["jsma", "analytic"],
                   default="jsma",
                   help="jsma = paper pairwise JSMA on the pre-projection "
                        "feature; analytic = signed |proj . t_target|")
    p.add_argument("--sni-combine", choices=["intersection", "union"],
                   default="intersection",
                   help="combine per-image JSMA sets (paper: intersection; "
                        "release: union); empty intersection falls back")
    p.add_argument("--sni-theta", type=float, default=0.1,
                   help="JSMA perturbation step theta (paper: 0.1)")
    p.add_argument("--sni-gamma", type=float, default=0.5,
                   help="JSMA feature-space perturbation budget gamma "
                        "(paper: 0.5)")
    p.add_argument("--sni-k", type=int, default=200,
                   help="cap on |I_t| significant pre-projection dims")
    p.add_argument("--sni-n", type=int, default=32,
                   help="number of calibration images used by JSMA")
    p.add_argument("--sni-max-steps", type=int, default=50,
                   help="max JSMA perturbation steps per image")

    # ── Phase ii (trigger) ───────────────────────────────────────────────
    p.add_argument("--clamp-mode", choices=["max", "mean3std", "const10"],
                   default="mean3std",
                   help="SNI clamp c: per-dim clean max / mean+3std "
                        "(scale-adapted high activation) / literal 10")
    p.add_argument("--trig-iters", type=int, default=100,
                   help="plain-GD trigger steps (paper: 100; release: 10)")
    p.add_argument("--trig-lr", type=float, default=0.1,
                   help="plain-GD step size (paper: 0.1; release: 0.01)")
    p.add_argument("--trig-temp", type=float, default=10.0,
                   help="CE temperature divisor for trigger optimisation "
                        "(default 10; 1 = full CLIP logit_scale)")
    p.add_argument("--trig-init", choices=["zero", "rand"], default="zero",
                   help="trigger initialisation (paper/release: zero)")
    p.add_argument("--trig-ce-weight", type=float, default=1.0,
                   help="weight of the trigger CE term (paper/release: 1)")

    # ── Phase iii (CBS) ──────────────────────────────────────────────────
    p.add_argument("--grid-k", type=int, default=20,
                   help="S3 grid points over [-R,R] (paper: 20)")
    p.add_argument("--cbs-top-elems", type=int, default=1,
                   help="S2 vulnerable-set width (paper/release: 1)")
    p.add_argument("--cbs-select-once", dest="cbs_select_once",
                   action="store_true", default=True,
                   help="S1: pin the attack tensor once (paper)")
    p.add_argument("--cbs-rerank", dest="cbs_select_once",
                   action="store_false",
                   help="S1: re-rank the tensor every round (old adaptation)")
    p.add_argument("--cbs-accept-always", dest="cbs_accept_always",
                   action="store_true", default=True,
                   help="S3: apply the min-loss candidate unconditionally "
                        "(paper)")
    p.add_argument("--cbs-gated", dest="cbs_accept_always",
                   action="store_false",
                   help="S3: accept only if L_troj improves (old adaptation)")
    p.add_argument("--monotone", dest="monotone", action="store_true",
                   default=False,
                   help="restrict the S3 grid to new >= old (release check "
                        "is commented out; paper has no constraint)")
    p.add_argument("--no-monotone", dest="monotone", action="store_false",
                   help="allow decreasing an element (default/paper/release)")
    p.add_argument("--q-mode", choices=["abs", "signed"], default="abs",
                   help="Q in F=|dL/dw|(Q-w): abs = max|w| (paper); "
                        "signed = max(w) (release)")
    p.add_argument("--asr-target", type=float, default=94.0,
                   help="stop quick ASR%% (paper: 94; push-98 grids: 98)")
    p.add_argument("--n-max", type=int, default=100,
                   help="bit budget (paper: 100; push-98 grids: 200)")
    p.add_argument("--quick-n", type=int, default=256,
                   help="search-split images for the quick ASR/CACC trace")
    p.add_argument("--verify-n", type=int, default=64,
                   help="subset of search images used for S1/S3 fitness")
    p.add_argument("--cacc-floor", type=float, default=None,
                   help="optional clean-accuracy floor (%%); a candidate that "
                        "drops below it is rejected. Default: off (paper)")

    # ── plumbing ─────────────────────────────────────────────────────────
    p.add_argument("--batch-size", type=int, default=32,
                   help="forward mini-batch for loss/gradient passes")
    p.add_argument("--eval-batch", type=int, default=64,
                   help="batch size for the shared evaluator")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_attack(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
