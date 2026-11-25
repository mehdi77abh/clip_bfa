"""
attack_proflip.py — ProFlip baseline for the CLIP bit-flip-attack suite.
=======================================================================

Self-contained port of **ProFlip** (Wang et al., ICCV 2021, "ProFlip: Bit-Flip
Attacks on DNNs via Polynomial-Time Sensitivity Analysis") to OpenAI CLIP
visual encoders, using ONLY the suite's shared plumbing (`clip_adapters.py`,
`evaluate.py`, frozen `data_split.json`).  Nothing from the
`modular_new_pipeline` package is imported.

The three ProFlip stages, ported from the released CNN code
(`ProFlip/resnet32-cifar10/trigger_nonadaptive.py`, `CSB_nonadaptive.py`) and
from this repo's `proflip_cbs.py` conventions:

1. **SNI — significant-neuron identification (Phase i).**  Upstream runs a
   JSMA loop on the last-layer input to collect the significant-neuron indices
   ``I_t``.  Here the CLIP analytic mapping is used instead: the pre-projection
   feature ``h`` is captured with a forward hook on ``model.visual.ln_post``
   (its output IS the input of the ``x @ visual.proj`` multiply in the OpenAI
   CLIP ``VisionTransformer.forward``), and the per-dim influence on the
   target alignment is ``score_in = |visual.proj @ t_target|``
   (since ``emb·t = h·(proj @ t)``).  ``I_t`` = top-``--sni-k`` dims (200).

2. **Native trigger (Phase ii).**  Fixed bottom-right patch, upstream geometry
   kept proportionally: side = round(input_px * 10/32) (10 px on 32 px ≈ 31 %
   of the width), top-left = (px - side - 1, px - side - 1) (1 px margin from
   the right/bottom edge).  The patch content is optimised in pixel space
   [0,1] (clamped) with --trig-iters 10 steps of plain GD
   ``p -= trig_lr * grad`` (trig_lr 0.01, upstream literal) on
       loss = MSE(h(x+patch), y.detach()) + CE_zs(logits(x+patch), target_cls)
   with ``y = h(clean calib images)`` and ``y[:, I_t] = c``.  The clamp ``c``
   is upstream's literal high-activation target, scale-adapted per
   --clamp-mode {max, mean3std, const10} (default `max` = per-dim clean max
   over the calib batch; `const10` = the upstream literal ``y[:,I_t] = 10``).

3. **CBS — critical bit search (Phase iii).**  Upstream searches ALL quantised
   Linear/Conv modules; here the surface is every 2D weight of
   ``model.visual`` matching attn.in_proj_weight / attn.out_proj.weight /
   mlp.c_fc.weight / mlp.c_proj.weight / visual.proj (full visual-encoder
   scope — patterns verified against dict(model.named_parameters()) at
   runtime).  Each tensor is quantised per-tensor int8
   (clip_adapters.quantize_int8_per_tensor) and its dequantised copy is
   written into the model for every forward pass.  Then, iteratively:

       S1  one backward on the fitness loss over the poisoned SEARCH batch
           (search split images + trigger): per tensor
           F_t = max_i |dL/dw_i| * (Q_t - w_i) restricted to dL/dw_i < 0
           (Q_t = max weight of that tensor — upstream literal); pick the
           tensor with max F_t.
       S2  inside that tensor, rank entries by the same fitness and take the
           vulnerable set (--cbs-top-elems, default 8 — upstream's element
           selection width).
       S3  per vulnerable element, grid-search K=20 candidate values over
           [-R, R] (R = max |int8 code| of the tensor; linspace K points,
           monotone new >= old — the exact `proflip_cbs.py` `_try_element`
           grid convention), and for each candidate flip ALL bits where the
           candidate int8 code differs from the current code (Hamming-distance
           multi-bit atomic element update).  The best candidate (min fitness
           loss) is accepted iff the fitness loss improves OR the quick ASR
           improves (eval_model on the aux split — the proflip_cbs acceptance
           rule extended with the quick-ASR branch).

   After each accepted element update S1 re-ranks (re-rank loop).  Stop when
   quick ASR >= --asr-target (98 in the push-98 config; the ProFlip paper
   metric is 94, the release hard-codes 90) or total flips >= --n-max (200 in
   the push-98 config; paper `n_b_max`=100) or the pool
   is exhausted (no vulnerable element yields an acceptable update).

Fitness loss (--fitness):
    mse_sni  L = MSE(h_poisoned, y) + CE_zs(logits, target_cls), with the SAME
             y / I_t clamp target as Phase ii (the upstream find_psens
             objective; y is built from the CLEAN search images and clamped
             with the Phase-ii constants c).
    cos      L = 1 - cos(emb_poisoned, t_target)      (CLIP-native alternative)

Outputs (contract: `evaluate.py` docstring):
    {out}/attack_state.pt   method="proflip"; param_overrides = final
                            dequantised values of every surface tensor;
                            trigger (kind="single", delta01 [1,3,px,px] with
                            content only in the patch box, mask [1,1,px,px],
                            mode="replace"); flip_log (one row per BIT);
                            meta (provenance incl. I_t).
    {out}/attack_meta.json  method/model/dataset/seed/target_cls/fitness/
                            n_flips/budget/search_time_s/trigger_asr_pre/
                            pool_size/flips_to_94/95/98/asr_at_k/sni_k.

In-search quick evaluation uses `evaluate.eval_model` on the first
`--quick-n` (256) search-split images — the same evaluator and trigger
semantics as the final frozen eval, just on a search-time subset.
`trigger_asr_pre`, `flips_to_94/95/98` and the `asr_at_k` curve come from
this quick trace (suite convention: ASR@K = quick ASR at the last state with
cumulative flips <= K; multi-bit element updates make it a step function).

Usage:
    python attack_proflip.py --model ViT-B/16 --dataset Caltech101 \
        --seed 42 --target-cls 97 --data-split data_split.json \
        --out results/proflip/ViT-B16/Caltech101/s42 --device cuda \
        --fitness mse_sni

Note on cost: one CBS round = 1 backward + up to
(--cbs-top-elems x grid-k) full-search-split forwards + a few aux quick evals.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

# Local suite modules win over anything in site-packages.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import clip_adapters as ca
import evaluate as eval_api


# ═════════════════════════════════════════════════════════════════════════
#  0.  small helpers (no torch.nn.functional — pure tensor ops)
# ═════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _norm(emb):
    """L2-normalise along the last dim (F.normalize equivalent)."""
    return emb / emb.norm(dim=-1, keepdim=True)


def _zs_logits(emb_norm, text_emb, logit_scale):
    """Zero-shot logits [B, C] — identical to clip_adapters.zero_shot_logits."""
    return (emb_norm @ text_emb.t()) * logit_scale


def _ce_to(logits, target_cls):
    """Mean cross-entropy toward class `target_cls` (no nn.functional)."""
    return (torch.logsumexp(logits, dim=-1) - logits[:, target_cls]).mean()


def bit_positions(old_code, new_code):
    """8-bit two's-complement Hamming positions, LSB first (proflip_cbs)."""
    diff = (int(old_code) & 0xFF) ^ (int(new_code) & 0xFF)
    return [b for b in range(8) if (diff >> b) & 1]


class FeatureTap:
    """Forward hook capturing the pre-projection feature `h`.

    OpenAI CLIP (``VisionTransformer.forward``):
        x = self.ln_post(x[:, 0, :])
        x = x @ self.proj
    so hooking `ln_post`'s output IS h [B, vision_width].

    SigLIP: the projection surface is ``head.mlp.fc2`` (see clip_adapters),
    so h is its INPUT (the post-fc1 activation [B, 3072]) — captured with a
    forward_pre_hook on that module."""

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
    raw = model.encode_image(imgs_norm)      # = h @ visual.proj (unnormalised)
    h = tap.value
    emb = _norm(raw)
    return h, emb


def capture_h(model, imgs01, tap, batch_size, device):
    """No-grad batched capture of h [N, vision_width] for a whole image set
    ([0,1] pixel space), mini-batched for memory safety."""
    hs = []
    with torch.no_grad():
        for i in range(0, imgs01.shape[0], batch_size):
            xb = ca.normalize(imgs01[i:i + batch_size], model).to(device)
            h, _ = _forward_h_emb(model, xb, tap)
            hs.append(h.detach())
    return torch.cat(hs, dim=0)


# ═════════════════════════════════════════════════════════════════════════
#  1.  Phase i — SNI: analytic significant-neuron indices
# ═════════════════════════════════════════════════════════════════════════

def compute_sni(model, text_emb, target_cls, sni_k):
    """Analytic SNI (replaces the upstream JSMA loop).

    score_in = |visual.proj @ t_target| over the pre-projection dims
    ([vision_width]); I_t = top-|sni_k| dims by that score.  The mapping:
    emb·t_target = (h @ proj)·t_target = h·(proj @ t_target), so |proj @ t|
    is exactly the per-dim influence of h on the target alignment.
    Returns (I_t LongTensor [K] sorted by descending score, score [W]).
    """
    proj = ca.proj_matrix(model)               # [width, embed] canonical view
    if proj is None:
        raise RuntimeError("projection surface is None — ViT/SigLIP expected")
    t_target = text_emb[target_cls]
    score = (proj @ t_target).abs()              # [width]
    k = int(min(int(sni_k), score.numel()))
    it_idx = torch.argsort(score, descending=True)[:k].long()
    return it_idx, score


# ═════════════════════════════════════════════════════════════════════════
#  2.  Phase ii — native feature-anchored trigger
# ═════════════════════════════════════════════════════════════════════════

def patch_geometry(px):
    """Upstream geometry kept proportional: 10 px on 32 px, 1 px margin.

    start = px - side - 1 (upstream: start=21, end=31 on 32 px input)."""
    side = int(round(px * 10.0 / 32.0))
    top = px - side - 1
    left = px - side - 1
    return side, top, left


def build_clamp(h_clean, it_idx, clamp_mode):
    """Per-dim high-activation clamp c [K] for the SNI dims.

    max      : c_j = max over the calib batch of h[:, I_t[j]]  (default)
    mean3std : c_j = mean + 3*std over the calib batch
    const10  : c_j = 10 (upstream literal `y[:, I_t] = 10`)"""
    sel = h_clean[:, it_idx]                     # [N, K]
    if clamp_mode == "max":
        c = sel.max(dim=0).values
    elif clamp_mode == "mean3std":
        c = sel.mean(dim=0) + 3.0 * sel.std(dim=0)
    elif clamp_mode == "const10":
        c = torch.full((sel.shape[1],), 10.0, dtype=sel.dtype)
    else:
        raise ValueError(f"unknown clamp mode {clamp_mode!r}")
    return c


def optimize_trigger(model, tap, calib01, y_calib, text_emb, target_cls,
                     px, args, device):
    """10-step plain GD on the patch content (upstream trigger_nonadaptive).

    loss = MSE(h(x+patch), y.detach()) + CE_zs(logits(x+patch), target_cls);
    y = y_calib (clean calib features with y[:, I_t] = c).  Gradient is
    accumulated over mini-batches of the calib split = the full-split mean
    loss gradient (upstream `p -= 0.01*sum(data.grad[box])` on one batch).
    Returns the attack_state trigger dict (CPU tensors)."""
    side, top, left = patch_geometry(px)
    mask = torch.zeros(1, 1, px, px)
    mask[:, :, top:top + side, left:left + side] = 1.0
    mask_dev = mask.to(device)
    n = calib01.shape[0]
    logit_scale = model.logit_scale.exp()

    p = torch.zeros(3, side, side, device=device)   # upstream zero init
    for it in range(int(args.trig_iters)):
        p = p.detach().requires_grad_(True)
        total = 0.0
        for i in range(0, n, args.batch_size):
            xb = calib01[i:i + args.batch_size].to(device)
            yb = y_calib[i:i + xb.shape[0]].to(device)
            # fresh delta graph per mini-batch (shared-leaf accumulation)
            delta = torch.zeros(1, 3, px, px, device=device)
            delta[:, :, top:top + side, left:left + side] = p
            poisoned = ca.apply_trigger(xb, delta, mask_dev, "replace")
            h, emb = _forward_h_emb(model, ca.normalize(poisoned, model), tap)
            loss = (((h - yb) ** 2).mean()
                    + _ce_to(_zs_logits(emb, text_emb, logit_scale),
                             target_cls))
            w = xb.shape[0] / n
            (loss * w).backward()
            total += float(loss.detach()) * w
        with torch.no_grad():
            p = (p - args.trig_lr * p.grad).clamp(0.0, 1.0)  # plain GD + clamp
        print(f"[proflip]   trigger step {it + 1}/{args.trig_iters}  "
              f"loss={total:.4f}")

    delta01 = torch.zeros(1, 3, px, px)
    delta01[:, :, top:top + side, left:left + side] = p.detach().cpu()
    trigger = {"kind": "single", "delta01": delta01,
               "mask": mask, "mode": "replace"}
    return trigger


# ═════════════════════════════════════════════════════════════════════════
#  3.  Phase iii — CBS: quantised surface + S1/S2/S3 critical bit search
# ═════════════════════════════════════════════════════════════════════════

_SURFACE_SUFFIXES = (".attn.in_proj_weight", ".attn.out_proj.weight",
                     ".mlp.c_fc.weight", ".mlp.c_proj.weight")


def is_surface_param(name, model):
    """Upstream scope = the vision-encoder attention/MLP 2-D weights + the
    projection surface.  Delegates to clip_adapters.surface_param_names (CLIP
    vs SigLIP naming)."""
    return name in ca.surface_param_names(model)


def build_surface(model):
    """Quantise every surface tensor per-tensor int8 and write the
    dequantised values into the model (search == evaluated values).

    Returns {name: dict(param, codes int64, scale)} built from the actual
    names of dict(model.named_parameters())."""
    surface_names = ca.surface_param_names(model)
    surface = {}
    for name, p in model.named_parameters():
        if p.dim() != 2 or name not in surface_names:
            continue
        codes, scale = ca.quantize_int8_per_tensor(p.data)
        surface[name] = dict(param=p, codes=codes.to(torch.int64),
                             scale=float(scale))
        p.data.copy_(ca.dequantize_int8(codes, scale))
        p.requires_grad_(True)             # S1 needs dL/dw on surface tensors
    if not surface:
        raise RuntimeError("empty attack surface — no vision-encoder 2-D "
                           "weights matched (attn/mlp + projection)")
    return surface


def fitness_value(model, tap, poisoned_norm, y_search, text_emb, target_cls,
                  mode, batch_size, device):
    """Fitness loss on the poisoned SEARCH batch (no grad).

    mse_sni : MSE(h, y) + CE_zs(logits, target_cls)   (find_psens objective)
    cos     : 1 - cos(emb, t_target)"""
    n = poisoned_norm.shape[0]
    logit_scale = model.logit_scale.exp()
    total = 0.0
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = poisoned_norm[i:i + batch_size].to(device)
            h, emb = _forward_h_emb(model, xb, tap)
            b = xb.shape[0]
            if mode == "mse_sni":
                yb = y_search[i:i + b].to(device)
                loss = (((h - yb) ** 2).mean()
                        + _ce_to(_zs_logits(emb, text_emb, logit_scale),
                                 target_cls))
            else:
                loss = 1.0 - (emb * text_emb[target_cls]).sum(dim=-1).mean()
            total += float(loss) * (b / n)
    return total


def fitness_grad(model, surface, tap, poisoned_norm, y_search, text_emb,
                 target_cls, mode, batch_size, device):
    """One backward of the fitness loss (surface params only) → total loss.

    Mini-batch losses are weighted by b/N so the accumulated grads equal the
    full-split mean-loss gradient."""
    for e in surface.values():
        e["param"].grad = None
    n = poisoned_norm.shape[0]
    logit_scale = model.logit_scale.exp()
    total = 0.0
    for i in range(0, n, batch_size):
        xb = poisoned_norm[i:i + batch_size].to(device)
        h, emb = _forward_h_emb(model, xb, tap)
        b = xb.shape[0]
        if mode == "mse_sni":
            yb = y_search[i:i + b].to(device)
            loss = (((h - yb) ** 2).mean()
                    + _ce_to(_zs_logits(emb, text_emb, logit_scale),
                             target_cls))
        else:
            loss = 1.0 - (emb * text_emb[target_cls]).sum(dim=-1).mean()
        w = b / n
        (loss * w).backward()
        total += float(loss.detach()) * w
    return total


def fitness_maps(surface, q_mode):
    """S1 fitness per surface tensor: F = |dL/dw| * (Q - w) on entries with
    negative gradient, 0 elsewhere.  Q = max weight of the tensor
    (`abs`: max|w| — the paper).  Returns {name: F map [out, in]}."""
    out = {}
    for name, e in surface.items():
        g = e["param"].grad
        if g is None:
            continue
        w = e["param"].data
        Q = w.max() if q_mode == "signed" else w.abs().max()
        step = torch.where(g < 0, (Q - w).clamp(min=0), torch.zeros_like(w))
        F = torch.nan_to_num(g.abs() * step, nan=0.0,
                             posinf=0.0, neginf=0.0)
        out[name] = F
    return out


def grid_search_element(model, surface, name, i, j, loss_cur, tap,
                        poisoned_norm, y_search, text_emb, target_cls,
                        mode, batch_size, device, args):
    """S3: grid-search K candidate int8 values for element (i, j).

    Grid convention (proflip_cbs._try_element): linspace(-R, R, K) on the
    int8 code axis with R = max |code| of the tensor, rounded/clamped to the
    reachable [-127, 127] code grid, unique; monotone filter new >= old
    (upstream `new = max(new, old)` rule, --no-monotone opts out).

    Returns the best candidate (min fitness loss) even when it does not beat
    the current loss — the acceptance rule decides.  None if no grid point
    differs from the current code."""
    e = surface[name]
    param, scale, codes = e["param"], e["scale"], e["codes"]
    old_code = int(codes[i, j].item())
    R = int(codes.abs().max().item())
    if R <= 0:
        return None
    grid = torch.linspace(-R, R, int(args.grid_k)).round()
    grid = grid.clamp(-127, 127).to(torch.int64).unique()
    if not args.no_monotone:
        grid = grid[grid >= old_code]
    old_fp = float(param.data[i, j].item())

    best_code, best_loss = None, None
    try:
        for code_t in grid.tolist():
            code = int(code_t)
            if code == old_code:
                continue
            param.data[i, j] = code * scale
            loss = fitness_value(model, tap, poisoned_norm, y_search,
                                 text_emb, target_cls, mode, batch_size,
                                 device)
            if not math.isfinite(loss):
                continue
            if best_loss is None or loss < best_loss - 1e-12:
                best_loss, best_code = loss, code
    finally:
        param.data[i, j] = old_fp                    # always restore

    if best_code is None or best_code == old_code:
        return None
    return dict(tensor=name, i=int(i), j=int(j),
                old_code=old_code, new_code=int(best_code),
                bits=bit_positions(old_code, best_code),
                old_fp=old_fp, new_fp=float(best_code) * scale,
                loss_cur=float(loss_cur), loss_new=float(best_loss))


def quick_asr(model, text_emb, aux01, aux_labels, target_cls, trigger,
              device, batch_size):
    """In-search quick ASR [%] via the SHARED evaluator on the aux split."""
    res = eval_api.eval_model(model, text_emb, aux01, aux_labels, target_cls,
                              trigger=trigger, device=device,
                              batch_size=batch_size)
    asr = res["asr"]
    return (100.0 * asr if asr == asr else float("nan")), res


def critical_bit_search(model, surface, tap, trigger, poisoned_norm, y_search,
                        text_emb, aux01, aux_labels, target_cls, args,
                        device):
    """The iterative CBS loop (S1 → S2 → S3, re-rank after each update)."""
    aux_bs = int(args.eval_batch)

    def quick():
        asr, _ = quick_asr(model, text_emb, aux01, aux_labels, target_cls,
                           trigger, device, aux_bs)
        return asr

    print("\n" + "=" * 64)
    print("PHASE III: PROFLIP CRITICAL BIT SEARCH (S1/S2/S3)")
    print("=" * 64)
    print(f"  surface tensors  : {len(surface)}")
    print(f"  fitness          : {args.fitness}")
    print(f"  S3 grid          : K={args.grid_k} over [-R, R]"
          f"  monotone={not args.no_monotone}")
    print(f"  vulnerable set   : top-{args.cbs_top_elems} elements / tensor")
    print(f"  stop             : quick ASR >= {args.asr_target}%  or  "
          f"bits >= {args.n_max}  or  pool exhausted")

    flip_log = []
    quick_trace = []
    n_bits = 0
    n_elements = 0
    rid = 0
    stop_reason = "budget"
    t0 = time.time()

    loss_cur = fitness_value(model, tap, poisoned_norm, y_search, text_emb,
                             target_cls, args.fitness, args.batch_size,
                             device)
    asr_quick = quick()
    quick_trace.append((0, asr_quick))
    print(f"  baseline         : loss={loss_cur:.4f}  quick ASR={asr_quick:.1f}%")
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
        loss_cur = fitness_grad(model, surface, tap, poisoned_norm, y_search,
                                text_emb, target_cls, args.fitness,
                                args.batch_size, device)
        Fmaps = fitness_maps(surface, args.q_mode)
        order = sorted(((n_, F_.max().item()) for n_, F_ in Fmaps.items()
                        if F_.max().item() > 0.0),
                       key=lambda kv: kv[1], reverse=True)
        if not order:
            stop_reason = "pool_exhausted"
            print("  no tensor with positive fitness — stopping")
            break
        name = order[0][0]

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
                model, surface, name, i, j, loss_cur, tap, poisoned_norm,
                y_search, text_emb, target_cls, args.fitness,
                args.batch_size, device, args)
            if cand is not None:
                cand["fitness"] = fval
                cands.append(cand)
        if not cands:
            stop_reason = "pool_exhausted"
            print("  no element yields a value change — stopping")
            break
        cands.sort(key=lambda c: c["loss_new"])

        # ── acceptance: loss improves OR quick-ASR improves ──────────────
        accepted = None
        asr_after = asr_quick
        for cand in cands:
            param = surface[cand["tensor"]]["param"]
            param.data[cand["i"], cand["j"]] = cand["new_fp"]
            if cand["loss_new"] < loss_cur - 1e-12:
                accepted = cand
                asr_after = quick()
                break
            asr_try = quick()
            if asr_try > asr_quick:
                accepted = cand
                asr_after = asr_try
                break
            param.data[cand["i"], cand["j"]] = cand["old_fp"]   # revert
        if accepted is None:
            stop_reason = "pool_exhausted"
            print("  no candidate improves loss or quick ASR — stopping")
            break

        # ── bookkeeping (one flip_log row per BIT) ───────────────────────
        cand = accepted
        # keep the int8 codes in sync with the applied element update
        surface[cand["tensor"]]["codes"][cand["i"], cand["j"]] = cand["new_code"]
        bits = list(cand["bits"])
        n_this = len(bits)
        n_bits += n_this
        n_elements += 1
        asr_quick = asr_after
        quick_trace.append((n_bits, asr_quick))
        for b in bits:
            flip_log.append(dict(
                target="weight",
                tensor=cand["tensor"],
                index=[cand["i"], cand["j"]],
                bit=int(b),
                old_int8=cand["old_code"], new_int8=cand["new_code"],
                old_fp32=cand["old_fp"], new_fp32=cand["new_fp"],
                fitness=cand["fitness"],
                old=cand["old_code"], new=cand["new_code"],   # schema alias
                cbs_round=rid, cbs_element_bits=bits, cbs_n_bits=n_this,
                loss_before=cand["loss_cur"], loss_after=cand["loss_new"]))
        if flips_to_94 is None and asr_quick >= 94.0 - 1e-6:
            flips_to_94 = n_bits
        if flips_to_95 is None and asr_quick >= 95.0 - 1e-6:
            flips_to_95 = n_bits
        if flips_to_98 is None and asr_quick >= 98.0 - 1e-6:
            flips_to_98 = n_bits
        print(f"  Rd {rid:>3d}: {cand['tensor']}[{cand['i']},{cand['j']}]  "
              f"code {cand['old_code']}->{cand['new_code']}  bits={n_this} "
              f"(total {n_bits})  loss {cand['loss_cur']:.4f}->"
              f"{cand['loss_new']:.4f}  quick ASR={asr_quick:.1f}%")
        rid += 1

    search_time = time.time() - t0
    print(f"\n  CBS elements: {n_elements}   bits: {n_bits}   "
          f"stop: {stop_reason}")
    print(f"  quick ASR  : {asr_quick:.1f}%   "
          f"flips_to_94={flips_to_94}  flips_to_95={flips_to_95}  "
          f"flips_to_98={flips_to_98}")

    return dict(flip_log=flip_log, n_bits=n_bits, n_elements=n_elements,
                asr_quick=asr_quick, flips_to_94=flips_to_94,
                flips_to_95=flips_to_95, flips_to_98=flips_to_98,
                stop_reason=stop_reason,
                search_time_s=search_time, quick_trace=quick_trace)


# ═════════════════════════════════════════════════════════════════════════
#  4.  orchestration + outputs
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
    search01, search_labels = ca.load_images(args.dataset, sp["search"], px,
                                             args.data_root)
    aux01, aux_labels = ca.load_images(args.dataset, sp["aux"], px,
                                       args.data_root)
    qn = max(1, min(int(args.quick_n), search01.size(0)))
    quick01, quick_labels = search01[:qn], search_labels[:qn]

    # ── model + zero-shot text head ──────────────────────────────────────
    model = ca.load_model(args.model, device)
    for p_ in model.parameters():          # grads only where we want them
        p_.requires_grad_(False)
    names = ca.load_classnames(args.dataset)
    if not 0 <= args.target_cls < len(names):
        raise ValueError(f"--target-cls {args.target_cls} out of range "
                         f"({len(names)} classes)")
    text_emb = ca.build_text_embeddings(model, names, device)
    tap = FeatureTap(model)

    # ── Phase i: SNI ─────────────────────────────────────────────────────
    t0 = time.time()
    it_idx, _ = compute_sni(model, text_emb, args.target_cls, args.sni_k)
    it_dev = it_idx.to(device)
    print(f"[proflip] Phase i  SNI: I_t = top-{it_idx.numel()} of "
          f"|visual.proj @ t_target|")

    # ── Phase ii: native trigger (calib split, clean fp32 model) ─────────
    h_calib = capture_h(model, calib01, tap, args.batch_size, device)
    c = build_clamp(h_calib, it_dev, args.clamp_mode)
    y_calib = h_calib.clone()
    y_calib[:, it_dev] = c
    trigger = optimize_trigger(model, tap, calib01, y_calib, text_emb,
                               args.target_cls, px, args, device)
    side, top, left = patch_geometry(px)
    print(f"[proflip] Phase ii trigger: patch {side}x{side} @ ({top},{left}), "
          f"clamp={args.clamp_mode}, {args.trig_iters} GD steps")
    t_ph2 = time.time() - t0

    # ── Phase iii: CBS on the quantised surface ──────────────────────────
    surface = build_surface(model)         # dequantised copies written in
    poisoned01 = eval_api.apply_trigger_dict(search01, trigger)
    poisoned_norm = ca.normalize(poisoned01, model)
    # same y / I_t clamp target as Phase ii, built from the CLEAN search
    # images (shapes match the poisoned batch) with the Phase-ii constants c
    y_search = capture_h(model, search01, tap, args.batch_size, device)
    y_search[:, it_dev] = c.to(y_search.device)

    trig_asr_pre, _ = quick_asr(model, text_emb, quick01, quick_labels,
                                args.target_cls, trigger, device,
                                args.eval_batch)
    print(f"[proflip] trigger quick ASR (pre-flips): {trig_asr_pre:.1f}%")

    cbs = critical_bit_search(model, surface, tap, trigger, poisoned_norm,
                              y_search, text_emb, quick01, quick_labels,
                              args.target_cls, args, device)
    tap.close()

    # ── outputs: attack_state.pt + attack_meta.json ──────────────────────
    os.makedirs(args.out, exist_ok=True)
    param_overrides = {
        name: e["param"].data.detach().cpu().clone().float()
        for name, e in surface.items()}
    meta = dict(
        method="proflip",
        fitness=args.fitness,
        sni_k=int(it_idx.numel()),
        sni_indices=[int(x) for x in it_idx.tolist()],
        clamp_mode=args.clamp_mode,
        clamp_values=[float(x) for x in c.detach().cpu().tolist()],
        patch=dict(side=side, top=top, left=left),
        trig_iters=int(args.trig_iters), trig_lr=float(args.trig_lr),
        trigger_time_s=round(t_ph2, 1),
        grid_k=int(args.grid_k), cbs_top_elems=int(args.cbs_top_elems),
        q_mode=args.q_mode, monotone=not args.no_monotone,
        asr_target=float(args.asr_target), n_max=int(args.n_max),
        pool_size=len(surface),
        surface_scales={n_: float(e["scale"]) for n_, e in surface.items()},
        n_flips=int(cbs["n_bits"]), n_elements=int(cbs["n_elements"]),
        flips_to_94=cbs["flips_to_94"], flips_to_95=cbs["flips_to_95"],
        flips_to_98=cbs["flips_to_98"],
        asr_at_k=eval_api.asr_at_k_from_trace(
            [(n, a / 100.0) for n, a in cbs["quick_trace"]]),
        trigger_asr_pre=(None if trig_asr_pre != trig_asr_pre
                         else round(trig_asr_pre, 4)),
        quick_split="search[:quick_n]", quick_asr_trace=cbs["quick_trace"],
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
        "budget": int(args.n_max),
        "search_time_s": round(cbs["search_time_s"], 1),
        "trigger_asr_pre": (None if trig_asr_pre != trig_asr_pre
                            else round(trig_asr_pre, 4)),
        "pool_size": len(surface),
        "flips_to_94": cbs["flips_to_94"],
        "flips_to_95": cbs["flips_to_95"],
        "flips_to_98": cbs["flips_to_98"],
        "asr_at_k": eval_api.asr_at_k_from_trace(
            [(n, a / 100.0) for n, a in cbs["quick_trace"]]),
        "sni_k": int(it_idx.numel()),
    }
    with open(os.path.join(args.out, "attack_meta.json"), "w") as f:
        json.dump(attack_meta, f, indent=2)

    print(f"[proflip] wrote {args.out}/attack_state.pt + attack_meta.json  "
          f"({cbs['n_bits']} bits in {cbs['search_time_s']:.0f}s)")
    return state


# ═════════════════════════════════════════════════════════════════════════
#  5.  CLI
# ═════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="ProFlip (SNI + native trigger + CBS) CLIP baseline")
    # ── run identity (exact suite CLI) ───────────────────────────────────
    p.add_argument("--model", required=True, choices=list(ca.MODEL_REGISTRY),
                   help="OpenAI CLIP victim (fp32)")
    p.add_argument("--dataset", required=True,
                   choices=list(ca.DATASET_REGISTRY),
                   help="Caltech101 | ImageNet1k | DTD")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-cls", type=int, required=True,
                   help="target class index in the dataset's class list")
    p.add_argument("--data-split", default=os.path.join(_HERE, "data_split.json"),
                   help="frozen disjoint split file (shared by all methods)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-root", default=None,
                   help="override clip_adapters' dataset root")
    # ── CBS fitness (both rows reported in the thesis grid) ──────────────
    p.add_argument("--fitness", choices=["mse_sni", "cos"], default="mse_sni",
                   help="mse_sni = upstream find_psens objective "
                        "(MSE(h,y[I_t]=c)+CE_zs); cos = 1-cos(emb, t_target)")
    # ── Phase i (SNI) ────────────────────────────────────────────────────
    p.add_argument("--sni-k", type=int, default=200,
                   help="|I_t| significant pre-projection dims (paper: 200)")
    # ── Phase ii (native trigger; upstream: 10 x GD 0.01) ────────────────
    p.add_argument("--clamp-mode", choices=["max", "mean3std", "const10"],
                   default="max",
                   help="SNI clamp c: per-dim clean max (default) / "
                        "mean+3std / upstream literal 10")
    p.add_argument("--trig-iters", type=int, default=10,
                   help="GD steps on the patch (upstream: 10)")
    p.add_argument("--trig-lr", type=float, default=0.01,
                   help="plain-GD step `p -= lr*grad` (upstream: 0.01)")
    # ── Phase iii (CBS; ProFlip paper values) ────────────────────────────
    p.add_argument("--grid-k", type=int, default=20,
                   help="S3 grid points over [-R, R] (paper: 20)")
    p.add_argument("--cbs-top-elems", type=int, default=8,
                   help="S2 vulnerable-set width (upstream: 8)")
    p.add_argument("--asr-target", type=float, default=98.0,
                   help="stop quick-ASR%% (push-98: 98; paper ASR_t=94)")
    p.add_argument("--n-max", type=int, default=200,
                   help="bit budget: stop at total flips >= N (push-98: 200; "
                        "paper n_max=100)")
    p.add_argument("--quick-n", type=int, default=256,
                   help="search-split images for the quick-ASR trace/stop "
                        "(push-98 needs >=100 for 98%% granularity)")
    p.add_argument("--q-mode", choices=["signed", "abs"], default="signed",
                   help="Q_t: released code max(w) (default) / paper max|w|")
    p.add_argument("--no-monotone", action="store_true",
                   help="allow decreasing an element (release forbids it)")
    # ── plumbing ─────────────────────────────────────────────────────────
    p.add_argument("--batch-size", type=int, default=32,
                   help="forward mini-batch for loss/gradient passes")
    p.add_argument("--eval-batch", type=int, default=64,
                   help="batch size for eval_model quick evaluations")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_attack(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
