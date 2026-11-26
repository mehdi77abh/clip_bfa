"""
attack_flips.py — Flip-S (CVPR 2025) bit-flip attack on OpenAI CLIP.
====================================================================

A clean, self-contained rewrite of the Flip-S attack for the CLIP baseline
suite.  Flip-S bit-flips the FP16 ROW-WISE SCALE FACTORS of an int8-quantised
victim: one flipped scale bit shifts a whole output channel (the primary
surface), complemented by int8 weight-code bit flips (the secondary surface).
Physical realism: 4 KB RowHammer mutual exclusion between accepted flips;
native budget: 50 bits.

Method fidelity: the two stages are ports of the suite's working adapter
`flip_s_clip/flip_s_clip_v2.py` (whose bodies are verbatim copies of Flip-S's
`deit-gen_trigger-int8.py` / `deit-targeted-int8.py`):

  Stage A — trigger generation (flip_s_clip_v2.generate_trigger):
    sign-PGD ("optimizer" = signed gradient steps under `schedule_lr`) on a
    zero-initialised full-image trigger restricted to a 3x3-patch block at the
    bottom-right corner of the input (auto geometry; block side = 3 x victim
    patch size, e.g. 48 px for ViT-B/16, 42 px for ViT-L/14).  Loss
    (verbatim):  CE(logits, target_class)
                 + atten_loss_weight * (attns_want_mean - 100)^2
    where the attention term is built from the recursive product of the
    residual-augmented per-layer attention maps (batch-dim accumulation and
    "last per-layer term" gradient exactly as the reference computes them),
    300 iterations, per-iteration clamp to [0,1], masked updates.  Trained on
    the AUX split (32 images).

  Stage B — SFS bit-flip search (flip_s_clip_v2.bitflip_attack):
    Per round: ranking loss (verbatim)
        loss = (MSE(logits_clean, clean_logits_cache)
                + gamma * CE(logits_triggered, target_class)) / (1 + gamma)
    on the AUX split; the clean-logits cache is computed once on the
    quantised-clean victim (the MSE anchor that protects clean behaviour).
    SFS ranking: bit candidates are scored by |dL/dparam * delta_bit| where
    delta_bit = flip_bit(value, bit) - value, for the fp16 scales
    (dL/dscale = rowwise chain of dL/dw) and the int8 codes alike; candidates
    with (delta * dL/dparam) > 0 are dropped (loss-increasing, the reference's
    pre-filter).  Greedy verification: each candidate flip is applied and
    quick-evaluated with `evaluate.eval_model` on a SEARCH-split subset; the
    best candidate is committed only if it strictly improves the objective
    (quick ASR) — the reference's acceptance comparison, on the suite's quick
    evaluator.  Accepted flips must sit >= 4096 bytes apart within a tensor
    (row-spaced addresses: row_bytes = in_features * 1 int8 byte + 2 scale
    bytes; weight bit (row, col) -> row*row_bytes + col, scale bit (row) ->
    row*row_bytes + in_features).  Budget `--max-flips` (push-98: 200,
    Flip-S native: 50); stop at budget or when no improving candidate
    remains.

Quantisation surface (verified at runtime via dict(model.named_parameters())):
every 2-D weight parameter of model.visual whose name ends in
`in_proj_weight` / `out_proj.weight` / `c_fc.weight` / `c_proj.weight`, plus
`visual.proj`.  Each is quantised with clip_adapters.rowwise_quantize_fp16
(row = output dim, bnb Linear8bitLt semantics: w = codes * scales with the
scale stored as fp16 — the flip target) and the forward pass uses
rowwise_dequantize written back into the fp32 parameters.  Scale flips that
would produce inf/nan (clip_adapters.flip_bit_fp16 raises ValueError) are
rejected at pool-construction time.

Outputs (in --out):
    attack_state.pt   — the evaluate.py schema: method/model/dataset/seed/
                        target_cls, param_overrides (final dequantised values
                        of every surface tensor), trigger (kind="single",
                        delta01 [1,3,px,px], mask [1,1,px,px], mode="replace"),
                        flip_log, meta.
    attack_meta.json  — {"method":"flips", "model", "dataset", "seed",
                        "target_cls", "n_flips", "budget", "n_scale_flips",
                        "n_weight_flips", "search_time_s", "trigger_asr_pre",
                        "pool_size", "flips_to_94/95/98", "asr_at_k",
                        "trig_side"}.  ASR@K and flips_to_* come from the
                        search-time quick trace on the first --quick-n
                        search-split images (suite convention).

CLI:
    python attack_flips.py --model ViT-B/16 --dataset Caltech101 --seed 42 \
        --target-cls 97 --data-split data_split.json --out results/.../s42 \
        --device cuda

Notes on binding deltas vs flip_s_clip_v2.py (plumbing only, no method change):
  * victim = ca.load_model (fp32 OpenAI CLIP) with the quantised forward
    emulated by writing rowwise_dequantize values into the parameters (the
    reference wraps bnb Linear8bitLt modules; bitsandbytes is not required);
  * the attack surface is the raw fused `in_proj_weight` parameter (the
    reference wraps q/k/v as one module) plus `visual.proj` (suite surface
    list) — every surface name is unchanged, so evaluate.py's
    apply_param_overrides works on a fresh victim;
  * attention maps for Stage A are recomputed by a transparent forward tap on
    nn.MultiheadAttention (identical scaled-dot-product math, the reference's
    DELTA 4; token offset +1 for CLIP's single special token).  The tap is
    active only while maps are needed, so search/eval forwards use the stock
    nn.MultiheadAttention path — numerically identical to evaluate.py's fresh
    victim;
  * the ranking loss uses the reference's CLIPVictim logits convention
    (cosine logits x model.logit_scale, the raw log-scale parameter) — the
    file's exact loss; quick evaluation goes through evaluate.eval_model
    (suite convention, logit_scale.exp()).  Both share the argmax, so the
    acceptance metric is unaffected.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import clip_adapters as ca
import evaluate

DEFAULT_SPLIT = os.path.join(_HERE, "data_split.json")

# ── Flip-S hyperparameters (values verbatim from flip_s_clip_v2.py's
#    TriggerArguments / AttackArguments / DataLoaderArguments) ─────────────
BUDGET = 50            # AttackArguments.target_bit (Flip-S native budget)
PUSH_BUDGET = 200      # push-98 config: extended budget (--max-flips default)
TOPK_SCALES = 50       # AttackArguments.topk   (scale rows per tensor/round)
TOPK_WEIGHTS = 50      # AttackArguments.topk2  (code positions per tensor/rnd)
GAMMA = 1.0            # AttackArguments.gamma  (MSE : CE balance)
TRIG_ITERS = 300       # TriggerArguments.train_attack_iters
ATTEN_LOSS_WEIGHT = 1.  # TriggerArguments.atten_loss_weight
QUICK_N = 256          # search-split subset used for greedy verification and
                       #   the quick-ASR trace (push-98 needs >=100 for 98%
                       #   granularity; was 64 in the first cut)
POOL_SCALES = 800      # ranked scale-bit candidates verified per round
                       #   (reference expands 50 rows x 16 bits)
POOL_WEIGHTS = 400     # ranked weight-bit candidates verified per round
                       #   (reference expands 50 positions x 8 bits)

DATASETS = ("Caltech101", "ImageNet1k", "DTD", "Food101")


# ═════════════════════════════════════════════════════════════════════════
# Flip-S verbatim schedule + loss plumbing
# ═════════════════════════════════════════════════════════════════════════

def schedule_lr(epoch):
    """Learning-rate schedule of deit-gen_trigger-int8.py (verbatim)."""
    lr = 100
    if epoch < 300:
        eta_min = .2
    else:
        eta_min = 0.05
    base = epoch // 30
    if epoch < 300:
        learning_rate = lr / pow(3, base)
    else:
        learning_rate = lr / pow(5, base)
    if learning_rate < eta_min:
        learning_rate = eta_min
    return learning_rate


def _victim_logits(feats, text_emb, logit_scale):
    """flip_s_clip_v2.CLIPVictim zero-shot head (verbatim math):
    logits = logit_scale * normalize(feats) @ text_features.T, with
    logit_scale = model.logit_scale (the raw log-scale parameter)."""
    f = feats / feats.norm(dim=-1, keepdim=True)
    return logit_scale * f @ text_emb.t()


def _ce_loss(logits, targets):
    """torch.nn.CrossEntropyLoss mean-reduction equivalent (no torch.nn.functional
    import needed)."""
    logp = logits.log_softmax(dim=-1)
    return -logp.gather(1, targets.view(-1, 1)).mean()


def _mse_loss(a, b):
    """torch.nn.MSELoss mean-reduction equivalent."""
    return ((a - b) ** 2).mean()


def _encode_with_maps(model, imgs_norm, tap):
    """(feats [B,D] unnormalised, per-head maps list) for the trigger loop.

    CLIP: `model.encode_image` populates `tap.maps` (the monkeypatched
    attention).  SigLIP: maps come from a single eager
    ``vision_model(..., output_attentions=True)`` pass (clip_adapters)."""
    if ca.backend_of(model) == "siglip":
        return ca.attention_maps(model, imgs_norm)
    feats = model.encode_image(imgs_norm)
    return feats, list(tap.maps)


# ═════════════════════════════════════════════════════════════════════════
# Attention maps for Stage A (reference DELTA 4: QuantMHA recomputation)
# ═════════════════════════════════════════════════════════════════════════

class AttentionTap:
    """Toggle + per-head attention-map collector for the visual blocks."""

    def __init__(self):
        self.enabled = False
        self.maps = []          # list of (B, heads, S, S), block order

    def clear(self):
        self.maps = []


def install_attn_taps(model):
    """Transparent forward tap on each visual nn.MultiheadAttention.

    When tap.enabled the manual scaled-dot-product path is computed (the exact
    math flip_s_clip_v2.QuantMHA exposes) and the per-head map (B, heads, S, S)
    is recorded — the tensor HF DeiT returned and the reference's attention
    loss consumes.  When disabled the STOCK nn.MultiheadAttention forward runs,
    so all non-Stage-A forwards are bit-identical to evaluate.py's fresh
    victim.  Parameter names are untouched (plain function shadowing), so
    attack_state.param_overrides stays valid.

    SigLIP branch: HF SiglipAttention is not nn.MultiheadAttention (separate
    q/k/v/out projections, tuple return) and cannot be intercepted for
    per-head maps without a full eager forward.  We return a lightweight stub
    and capture maps via clip_adapters.attention_maps (output_attentions=True)
    in `_encode_with_maps` instead."""
    if ca.backend_of(model) == "siglip":
        return AttentionTap()          # stub; maps captured in _encode_with_maps
    tap = AttentionTap()
    for blk in model.visual.transformer.resblocks:
        attn = blk.attn
        orig_forward = attn.forward
        embed_dim = attn.embed_dim
        n_head = attn.num_heads
        head_dim = embed_dim // n_head

        def make_forward(attn, orig_forward, embed_dim, n_head, head_dim, tap):
            def forward(q, k=None, v=None, need_weights=True, **kwargs):
                if not tap.enabled:
                    return orig_forward(q, k, v, need_weights=need_weights, **kwargs)
                if kwargs.get("attn_mask") is not None or \
                        kwargs.get("key_padding_mask") is not None:
                    raise RuntimeError("attention tap does not support masks")
                if k is None:
                    k = q
                if v is None:
                    v = q
                S, B, _ = q.shape
                qkv = q @ attn.in_proj_weight.t()
                if attn.in_proj_bias is not None:
                    qkv = qkv + attn.in_proj_bias
                q_, k_, v_ = qkv.chunk(3, dim=-1)

                def heads(t):
                    return t.reshape(S, B, n_head, head_dim).permute(1, 2, 0, 3)

                qq, kk, vv = heads(q_), heads(k_), heads(v_)
                amap = (qq @ kk.transpose(-2, -1)) * (head_dim ** -0.5)
                amap = amap.softmax(dim=-1)
                out = (amap @ vv).permute(2, 0, 1, 3).reshape(S, B, embed_dim)
                out = attn.out_proj(out)
                tap.maps.append(amap)          # (B, heads, S, S), in graph
                return out, None
            return forward

        attn.forward = make_forward(attn, orig_forward, embed_dim,
                                    n_head, head_dim, tap)
    return tap


# ═════════════════════════════════════════════════════════════════════════
# Quantisation surface: row-wise int8 codes + fp16 scales (Flip-S surface)
# ═════════════════════════════════════════════════════════════════════════

SURFACE_SUFFIXES = ("in_proj_weight", "out_proj.weight",
                    "c_fc.weight", "c_proj.weight")


def find_surface_params(model, include_proj=False):
    """Runtime-verified surface via clip_adapters.surface_param_names.

    Default `include_proj=False` = encoder-only (the paper's scope — Flip-S
    never attacks the head/projection).  `--flip-surface extended` adds the
    projection (visual.proj / SigLIP head fc2) as a labelled variant.  Verified
    against dict(model.named_parameters())."""
    named = dict(model.named_parameters())
    found = []
    for name in sorted(ca.surface_param_names(model, include_proj=include_proj)):
        p = named[name]
        if p.dim() != 2:
            raise RuntimeError(f"surface parameter {name!r} is not 2-D")
        found.append((name, p))
    if not found:
        raise RuntimeError("no quantisation surface found in the vision tower")
    n_codes = sum(int(p.numel()) for _, p in found)
    print(f"[flips] surface: {len(found)} tensors, {n_codes:,} int8 codes "
          f"(include_proj={include_proj}; names verified via "
          "model.named_parameters())")
    return found


class QuantSurface:
    """Row-wise int8 codes + fp16 scales for every surface tensor.

    Forward values are rowwise_dequantize(codes, scales) written back into the
    fp32 parameters (identical to the reference's my_8bit_linear math
    x @ codes.T * scales).  The search flips bits in the fp16 scales (primary
    surface) and the int8 codes (secondary surface) and rewrites the
    dequantised values.

    4 KB mutual exclusion: each flip has a byte address inside its tensor's
    row-spaced layout (row_bytes = in_features * 1 int8 byte + 2 scale bytes);
    committing a flip bans every address within 4095 bytes, so any two
    accepted flips sit >= 4096 bytes apart (the reference's 4 KB RowHammer
    window bookkeeping, unified across both surfaces of a tensor).
    """

    def __init__(self, model, device, include_proj=False):
        self.device = device
        self.params = {}
        self.codes = {}
        self.scales = {}
        self.bans = {}          # name -> list of (start, end) banned addr ranges
        for name, p in find_surface_params(model, include_proj=include_proj):
            codes, scales = ca.rowwise_quantize_fp16(p.data)
            self.params[name] = p
            self.codes[name] = codes.to(device)
            self.scales[name] = scales.to(device)
            self.bans[name] = []
            p.requires_grad_(True)
        for name, p in model.named_parameters():
            if name not in self.params:
                p.requires_grad_(False)
        self.write_back()       # forward uses the dequantised values

    # ── layout / addresses ─────────────────────────────────────────────
    def row_bytes(self, name):
        return int(self.codes[name].shape[1]) + 2

    def addr_weight(self, name, row, col):
        return row * self.row_bytes(name) + col

    def addr_scale(self, name, row):
        return row * self.row_bytes(name) + int(self.codes[name].shape[1])

    # ── forward values ─────────────────────────────────────────────────
    def write_back(self, names=None):
        for name in (names if names is not None else list(self.params)):
            p = self.params[name]
            p.data.copy_(ca.rowwise_dequantize(self.codes[name],
                                               self.scales[name]).to(p.dtype))

    def final_overrides(self):
        """evaluate.py contract: final dequantised value of every surface
        tensor (all of them differ from the clean checkpoint once quantised)."""
        return {name: ca.rowwise_dequantize(self.codes[name],
                                            self.scales[name]).float().cpu()
                for name in self.params}

    # ── gradient chain: dL/dw -> dL/dscale, dL/dcode (w = code * scale) ─
    def surface_grads(self, grads_by_name):
        g_scale, g_code = {}, {}
        for name, g in grads_by_name.items():
            s = self.scales[name].float()
            c = self.codes[name].float()
            g_scale[name] = (g * c).sum(dim=1)      # [out]
            g_code[name] = g * s.unsqueeze(1)       # [out, in]
        return g_scale, g_code

    # ── flip application (values precomputed by the pool builder) ──────
    def apply_flip(self, cand):
        name = cand["tensor"]
        if cand["target"] == "scale":
            self.scales[name][cand["row"]] = cand["new"]
        else:
            self.codes[name][cand["row"], cand["col"]] = cand["new"]

    def undo_flip(self, cand):
        name = cand["tensor"]
        if cand["target"] == "scale":
            self.scales[name][cand["row"]] = cand["old"]
        else:
            self.codes[name][cand["row"], cand["col"]] = cand["old"]

    def commit_flip(self, cand):
        self.apply_flip(cand)
        a = cand["addr"]
        self.bans[cand["tensor"]].append((max(0, a - 4095), a + 4095))

    # ── banned element lookups (bookkeeping as in the reference's
    #    baned_absmax / baned_weight lists) ─────────────────────────────
    def banned_scale_rows(self, name):
        rb = self.row_bytes(name)
        out_f, in_f = self.codes[name].shape
        rows = set()
        for (s, e) in self.bans[name]:
            for r in range(max(0, s // rb), min(out_f - 1, e // rb) + 1):
                if s <= r * rb + in_f <= e:
                    rows.add(r)
        return torch.tensor(sorted(rows), dtype=torch.long, device=self.device)

    def banned_weight_flat(self, name):
        rb = self.row_bytes(name)
        out_f, in_f = self.codes[name].shape
        flats = []
        for (s, e) in self.bans[name]:
            for r in range(max(0, s // rb), min(out_f - 1, e // rb) + 1):
                base = r * rb
                c0 = max(0, s - base)
                c1 = min(in_f - 1, e - base)
                if c0 <= c1:
                    flats.extend(range(r * in_f + c0, r * in_f + c1 + 1))
        return torch.tensor(sorted(set(flats)), dtype=torch.long,
                            device=self.device)


# ═════════════════════════════════════════════════════════════════════════
# Trigger geometry (flip_s_clip_v2.resolve_patch_geometry, auto bottom-right)
# ═════════════════════════════════════════════════════════════════════════

def trigger_geometry(model, trig_side=None, trig_corner="br"):
    """3x3-patch block of side 3 * victim_patch at the input corner (default
    bottom-right; --trig-side / --trig-corner override).  Returns the pixel
    mask, the covered patch-token indices (Flip-S order, +1 CLIP token offset
    — reference DELTA 4) and the geometry for logging."""
    patch = int(model.visual.conv1.kernel_size[0])
    px = int(model.visual.input_resolution)
    if px % patch:
        raise ValueError(f"image size {px} not divisible by patch {patch}")
    pps = px // patch
    if pps < 3:
        raise ValueError(f"patch grid {pps}x{pps} too small for a 3x3 block")
    side = int(trig_side) if trig_side else 3 * patch
    if side <= 0 or side > px:
        raise ValueError(f"--trig-side {side} outside (0, {px}]")
    origins = {"br": (px - side, px - side), "bl": (px - side, 0),
               "tr": (0, px - side), "tl": (0, 0)}
    r0, c0 = origins[trig_corner]

    mask = torch.zeros(1, 1, px, px)
    mask[:, :, r0:r0 + side, c0:c0 + side] = 1.0

    # patch tokens covered by the block, far-corner rows/cols first
    # (descending order reproduces flip_s_clip_v2._auto_patch_indices for the
    # default bottom-right 3x3 block)
    pr0, pr1 = r0 // patch, (r0 + side - 1) // patch
    pc0, pc1 = c0 // patch, (c0 + side - 1) // patch
    patch_indices = [r * pps + c
                     for r in range(pr1, pr0 - 1, -1)
                     for c in range(pc1, pc0 - 1, -1)]
    n_prefix = ca.seq_n_prefix(model)          # CLIP: 1 CLS token; SigLIP: 0
    token_idx = torch.tensor([i + n_prefix for i in patch_indices],
                             dtype=torch.long)

    assert float(mask.sum()) == side * side, "mask is not correct"
    geom = {"patch_size": patch, "image_size": px, "patches_per_side": pps,
            "side": side, "corner": trig_corner, "origin": (r0, c0),
            "mask": mask, "token_idx": token_idx,
            "patch_indices": patch_indices}
    print(f"[flips] trigger geometry: {side}x{side} px block at {trig_corner} "
          f"(patch {patch}, grid {pps}x{pps}, tokens {len(patch_indices)})")
    return geom


# ═════════════════════════════════════════════════════════════════════════
# Stage A — trigger generation (port of flip_s_clip_v2.generate_trigger)
# ═════════════════════════════════════════════════════════════════════════

def generate_trigger(model, text_emb, aux01, target_cls, device, geom, tap,
                     iters=TRIG_ITERS, atten_loss_weight=ATTEN_LOSS_WEIGHT,
                     log=print):
    """Sign-PGD trigger on the AUX split toward `target_cls` (the reference's
    loop, values included).  Same loss / steps / schedule / iterations as
    flip_s_clip_v2.generate_trigger:
        loss_p    = CE(logits, target_class)
        atten_loss = (attns_want_mean - 100)^2   (attention rollout term)
        step      = -sign(d(atten_loss*weight + loss_p)/d trigger) * schedule_lr
        trigger   = clamp(trigger, 0, 1)
    """
    px = geom["image_size"]
    mask_hw = geom["mask"][0, 0].to(device)                 # [px, px]
    tok_idx = geom["token_idx"].to(device)
    x = aux01.to(device)
    target = torch.full((x.size(0),), int(target_cls),
                        dtype=torch.long, device=device)
    logit_scale = model.logit_scale.detach()
    trigger = torch.zeros(3, px, px, device=device, requires_grad=True)

    tap.enabled = True
    atten_loss = None
    attns_want_mean = None
    for it in range(iters):
        tap.clear()
        # sparse-patch composition (reference: (x*(1-mask) + trigger*mask).clamp(0,1))
        composed = (x * (1.0 - mask_hw) + trigger * mask_hw).clamp(0.0, 1.0)
        feats, maps = _encode_with_maps(model, ca.normalize(composed, model), tap)
        logits = _victim_logits(feats, text_emb, logit_scale)
        loss_p = _ce_loss(logits, target)

        # attention rollout (reference, verbatim structure):
        att_mats = torch.stack([a.mean(dim=1) for a in maps], dim=1)
        residual_att = torch.eye(att_mats.size(-1), device=device)
        aug_att_mat = att_mats + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
        # recursive product of the residual-augmented maps (the reference
        # accumulates joint_attentions[n] = aug[n] @ joint[n-1] — same math)
        joint = [aug_att_mat[0]]
        for n in range(1, aug_att_mat.size(0)):
            joint.append(aug_att_mat[n] @ joint[-1])
        joint_attentions = torch.stack(joint, dim=0)

        for i in range(0, joint_attentions.size(1) - 1):
            v = joint_attentions[:, i, :, :].squeeze(1)
            v_mean = torch.mean(torch.mean(v, dim=0), dim=0).view(-1)
            attns_want = v_mean[tok_idx]
            attns_want_mean = attns_want.mean()
            atten_loss = pow(attns_want_mean - 100, 2)

        # NOTE: like the reference, the gradient uses the LAST per-layer
        # atten_loss term (not the accumulated total) + the CE term.
        atten_grad = torch.autograd.grad(
            atten_loss * atten_loss_weight + loss_p, trigger)[0]
        trigger.data -= atten_grad.sign() * mask_hw * schedule_lr(it)
        trigger.data.clamp_(0.0, 1.0)

        if it % 5 == 0:
            log(f"[flips] trigger iter {it}: lr {schedule_lr(it):.5f} "
                f"CE {loss_p.item():.4f} atten {float(atten_loss):.4f} "
                f"attns_want_mean {float(attns_want_mean):.4f}")
    tap.enabled = False

    # sanity: mask region only, values in [0,1] (reference asserts coverage)
    assert bool((trigger[~mask_hw.bool().expand_as(trigger)] == 0).all()), \
        "trigger content outside the mask block"
    return trigger.detach().clamp(0.0, 1.0)


# ═════════════════════════════════════════════════════════════════════════
# Quick evaluation (the suite's shared evaluator)
# ═════════════════════════════════════════════════════════════════════════

def quick_asr(model, text_emb, imgs01, labels, target_cls, trig, device, batch):
    """evaluate.eval_model on one image set -> (quick ASR, full result dict)."""
    res = evaluate.eval_model(model, text_emb, imgs01, labels, target_cls,
                              trigger=trig, device=device, batch_size=batch)
    asr = res["asr"]
    if not math.isfinite(asr):          # no non-target image in the subset
        asr = res["asr_incl_target"]
    return asr, res


# ═════════════════════════════════════════════════════════════════════════
# SFS candidate pools (flip_s_clip_v2 bit ranking + pre-filter)
# ═════════════════════════════════════════════════════════════════════════

def build_ranking_pools(surface, g_scale, g_code,
                        topk_scales=TOPK_SCALES, topk_weights=TOPK_WEIGHTS,
                        pool_scales=POOL_SCALES, pool_weights=POOL_WEIGHTS):
    """Rank bit candidates by |dL/dparam * delta_bit| (SFS score), keeping the
    loss-reducing ones only ((delta * dL/dparam) > 0 is dropped — the
    reference's per-bit pre-filter).  Per round the reference expands the top
    `topk` scale rows / `topk2` code positions per tensor; the merged pools are
    truncated to `pool_scales` / `pool_weights` verified candidates."""
    scale_cands = []
    for name, g in g_scale.items():
        ga = g.abs()
        banned = surface.banned_scale_rows(name)
        if banned.numel():
            ga = ga.clone()
            ga[banned] = -1.0
        k = min(int(topk_scales), ga.numel())
        _, rows = ga.topk(k)
        for r in rows.tolist():
            if ga[r] < 0:
                continue            # banned (reference skips banned indices)
            gv = g[r].item()
            old = float(surface.scales[name][r])
            for bit in range(16):
                try:
                    new = ca.flip_bit_fp16(old, bit)
                except ValueError:
                    continue        # inf/nan fp16 flip -> rejected
                d = new - old
                if d * gv > 0.0:
                    continue        # loss-increasing (reference rule)
                scale_cands.append({
                    "tensor": name, "target": "scale", "row": r, "bit": bit,
                    "old": float(old), "new": float(new),
                    "addr": surface.addr_scale(name, r),
                    "score": abs(gv * d)})
    scale_cands.sort(key=lambda c: -c["score"])
    scale_cands = scale_cands[: int(pool_scales)]

    weight_cands = []
    for name, g in g_code.items():
        gf = g.reshape(-1)
        ga = gf.abs()
        banned = surface.banned_weight_flat(name)
        if banned.numel():
            ga = ga.clone()
            ga[banned] = -1.0
        k = min(int(topk_weights), ga.numel())
        _, flats = ga.topk(k)
        in_f = int(surface.codes[name].shape[1])
        for f in flats.tolist():
            if ga[f] < 0:
                continue
            row, col = divmod(f, in_f)
            gv = gf[f].item()
            old = int(surface.codes[name][row, col])
            for bit in range(8):
                new = ca.flip_bit_int8(old, bit)
                d = new - old
                if d * gv > 0.0:
                    continue
                weight_cands.append({
                    "tensor": name, "target": "weight", "row": row, "col": col,
                    "bit": bit, "old": int(old), "new": int(new),
                    "addr": surface.addr_weight(name, row, col),
                    "score": abs(gv * d)})
    weight_cands.sort(key=lambda c: -c["score"])
    weight_cands = weight_cands[: int(pool_weights)]
    return scale_cands, weight_cands


# ═════════════════════════════════════════════════════════════════════════
# Stage B — SFS bit-flip search (port of flip_s_clip_v2.bitflip_attack)
# ═════════════════════════════════════════════════════════════════════════

def make_log_record(surface, cand, asr_after):
    """flip_log entry: target="scale" (bit 0..15, old_fp32/new_fp32 = old/new
    fp16 scale value) or target="weight" (bit 0..7, old_int8/new_int8 code
    values, old_fp32/new_fp32 = the corresponding dequantised weights)."""
    name = cand["tensor"]
    if cand["target"] == "scale":
        rec = {"tensor": name, "target": "scale", "index": [cand["row"]],
               "bit": cand["bit"], "old": cand["old"], "new": cand["new"],
               "old_fp32": cand["old"], "new_fp32": cand["new"]}
    else:
        scale_f = float(surface.scales[name][cand["row"]])
        rec = {"tensor": name, "target": "weight",
               "index": [cand["row"], cand["col"]], "bit": cand["bit"],
               "old": int(cand["old"]), "new": int(cand["new"]),
               "old_int8": int(cand["old"]), "new_int8": int(cand["new"]),
               "old_fp32": cand["old"] * scale_f,
               "new_fp32": cand["new"] * scale_f}
    rec["addr"] = cand["addr"]
    rec["asr_after"] = round(asr_after, 4)
    return rec


def bitflip_search(surface, model, text_emb, aux01, quick01, quick_labels,
                   target_cls, trig, args, device, log=print):
    """Greedy SFS search.  Per round: rank (aux ranking loss) -> verify each
    pooled candidate by quick eval on the search subset -> commit the best
    candidate iff it strictly improves the quick ASR (the reference's
    acceptance comparison) -> 4 KB ban.  Scales are tried before weights (the
    reference skips the weight stage whenever a scale flip improves the loss).
    Stops at --max-flips or when no improving candidate remains."""
    logit_scale = model.logit_scale.detach()
    aux = aux01.to(device)
    delta_dev = trig["delta01"].to(device)
    mask_dev = trig["mask"].to(device)
    target = torch.full((aux.size(0),), int(target_cls),
                        dtype=torch.long, device=device)
    params = [surface.params[n] for n in surface.params]

    # clean-logits cache on the quantised-clean victim (reference caches before
    # the search; the MSE anchor preserves clean behaviour)
    with torch.no_grad():
        feats = model.encode_image(ca.normalize(aux, model))
        cache = _victim_logits(feats, text_emb, logit_scale)

    def verify(cand):
        surface.apply_flip(cand)
        surface.write_back([cand["tensor"]])
        asr, _ = quick_asr(model, text_emb, quick01, quick_labels, target_cls,
                           trig, device, args.batch)
        surface.undo_flip(cand)
        surface.write_back([cand["tensor"]])
        return asr

    flip_log = []
    flips_to_94 = None
    flips_to_95 = None
    flips_to_98 = None
    pool_size = None
    cur_asr, _ = quick_asr(model, text_emb, quick01, quick_labels, target_cls,
                           trig, device, args.batch)
    quick_trace = [(0, float(cur_asr))]
    log(f"[flips] search start: quick ASR {cur_asr * 100:.2f}% "
        f"(n={int(quick01.size(0))}), budget {args.max_flips}")

    while len(flip_log) < int(args.max_flips):
        # ── rank: SFS first-order scores on the reference's loss ───────
        feats_c = model.encode_image(ca.normalize(aux, model))
        logits_c = _victim_logits(feats_c, text_emb, logit_scale)
        trig_aux = ca.apply_trigger(aux, delta_dev, mask_dev, "replace")
        feats_t = model.encode_image(ca.normalize(trig_aux, model))
        logits_t = _victim_logits(feats_t, text_emb, logit_scale)
        loss = (_mse_loss(logits_c, cache)
                + float(args.gamma) * _ce_loss(logits_t, target)) \
            / (1.0 + float(args.gamma))
        grads = torch.autograd.grad(loss, params)
        g_scale, g_code = surface.surface_grads(
            {n: g for n, g in zip(surface.params, grads)})
        scale_cands, weight_cands = build_ranking_pools(
            surface, g_scale, g_code,
            topk_scales=args.topk, topk_weights=args.topk2,
            pool_scales=args.pool_scales, pool_weights=args.pool_weights)
        if pool_size is None:
            pool_size = len(scale_cands) + len(weight_cands)
        log(f"[flips] round {len(flip_log) + 1}: loss {float(loss):.4f}, "
            f"pool {len(scale_cands)} scale + {len(weight_cands)} weight")

        # ── greedy verification (scales first, like the reference) ────
        best = None                      # (quick ASR, candidate)
        for cand in scale_cands:
            asr = verify(cand)
            if asr > cur_asr and (best is None or asr > best[0]):
                best = (asr, cand)
        if best is None:
            for cand in weight_cands:
                asr = verify(cand)
                if asr > cur_asr and (best is None or asr > best[0]):
                    best = (asr, cand)
        if best is None:
            log("[flips] no improving candidate remains — stopping")
            break

        # ── commit + 4 KB mutual exclusion + bookkeeping ──────────────
        asr, cand = best
        surface.commit_flip(cand)
        surface.write_back([cand["tensor"]])
        cur_asr = asr
        flip_log.append(make_log_record(surface, cand, asr))
        quick_trace.append((len(flip_log), float(asr)))
        if flips_to_94 is None and asr * 100.0 >= 94.0:
            flips_to_94 = len(flip_log)
        if flips_to_95 is None and asr * 100.0 >= 95.0:
            flips_to_95 = len(flip_log)
        if flips_to_98 is None and asr * 100.0 >= 98.0:
            flips_to_98 = len(flip_log)
        log(f"[flips] flip {len(flip_log)}/{int(args.max_flips)}: "
            f"{cand['tensor']} {cand['target']} row={cand['row']} "
            f"bit={cand['bit']} addr={cand['addr']} -> quick ASR "
            f"{asr * 100:.2f}%")

    return (flip_log, pool_size, flips_to_94, flips_to_95, flips_to_98,
            quick_trace, cur_asr)


# ═════════════════════════════════════════════════════════════════════════
# Run: Stage A -> trigger_asr_pre -> Stage B -> attack_state.pt / meta
# ═════════════════════════════════════════════════════════════════════════

def run(args):
    t_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    with open(args.data_split) as f:
        splits = json.load(f)
    if args.dataset not in splits:
        raise SystemExit(f"dataset {args.dataset!r} not in {args.data_split}")
    split = splits[args.dataset]
    aux_idx = split["aux"]
    quick_idx = split["search"][: max(1, int(args.quick_n))]

    model = ca.load_model(args.model, device)
    tap = install_attn_taps(model)
    names = ca.load_classnames(args.dataset)
    if not 0 <= int(args.target_cls) < len(names):
        raise SystemExit(f"--target-cls {args.target_cls} outside "
                         f"[0, {len(names)}) for {args.dataset}")
    text_emb = ca.build_text_embeddings(model, names, device)
    px = ca.input_px(args.model)

    aux01, _ = ca.load_images(args.dataset, aux_idx, px, args.data_root)
    quick01, quick_y = ca.load_images(args.dataset, quick_idx, px, args.data_root)

    geom = trigger_geometry(model, args.trig_side, args.trig_corner)
    surface = QuantSurface(model, device,
                           include_proj=(args.flip_surface == "extended"))

    # ── Stage A: trigger on the AUX split (quantised victim, as in the
    #    reference where load_clip always installs the int8 linears) ────
    delta = generate_trigger(model, text_emb, aux01, args.target_cls, device,
                             geom, tap, iters=args.trig_iters,
                             atten_loss_weight=args.atten_loss_weight)
    trig = {"kind": "single",
            "delta01": delta.view(1, 3, px, px).clamp(0.0, 1.0).cpu(),
            "mask": geom["mask"].clone(),
            "mode": "replace"}

    # trigger-only strength before Stage B, on the CALIB split
    trigger_asr_pre, _ = quick_asr(model, text_emb, quick01, quick_y,
                                   args.target_cls, trig, device, args.batch)
    print(f"[flips] trigger_asr_pre {trigger_asr_pre * 100:.2f}% "
          f"(quick split n={int(quick01.size(0))})")

    # ── Stage B: SFS bit-flip search ──────────────────────────────────
    t_search = time.time()
    (flip_log, pool_size, flips_to_94, flips_to_95, flips_to_98,
     quick_trace, final_quick_asr) = bitflip_search(
        surface, model, text_emb, aux01, quick01, quick_y,
        args.target_cls, trig, args, device)
    search_time_s = time.time() - t_search

    n_scale = sum(1 for r in flip_log if r["target"] == "scale")
    n_weight = sum(1 for r in flip_log if r["target"] == "weight")
    meta = {
        "method": "flips",
        "model": args.model,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "target_cls": int(args.target_cls),
        "n_flips": len(flip_log),
        "budget": int(args.max_flips),
        "n_scale_flips": n_scale,
        "n_weight_flips": n_weight,
        "search_time_s": round(search_time_s, 1),
        "trigger_asr_pre": round(float(trigger_asr_pre), 4),
        "quick_split": "search[:quick_n]",
        "pool_size": int(pool_size) if pool_size is not None else 0,
        "flips_to_94": flips_to_94,
        "flips_to_95": flips_to_95,
        "flips_to_98": flips_to_98,
        "asr_at_k": evaluate.asr_at_k_from_trace(quick_trace),
        "trig_side": int(geom["side"]),
        "tar": round((int(geom["side"]) / float(geom["image_size"])) ** 2 * 100, 2),
    }

    state = {
        "method": "flips",
        "model": args.model,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "target_cls": int(args.target_cls),
        "param_overrides": surface.final_overrides(),
        "trigger": trig,
        "flip_log": flip_log,
        "meta": meta,
    }

    os.makedirs(args.out, exist_ok=True)
    state_path = os.path.join(args.out, "attack_state.pt")
    meta_path = os.path.join(args.out, "attack_meta.json")
    torch.save(state, state_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[flips] done: {len(flip_log)} flips "
          f"({n_scale} scale / {n_weight} weight), quick ASR "
          f"{final_quick_asr * 100:.2f}%, trigger_asr_pre "
          f"{meta['trigger_asr_pre']}, total {time.time() - t_start:.1f}s")
    print(f"[flips] wrote {state_path} and {meta_path}")
    return state


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Flip-S (CVPR 2025) bit-flip attack on OpenAI CLIP "
                    "(fp16 scale-factor flips + int8 weight flips, 4 KB "
                    "mutual exclusion, 50-bit budget)")
    p.add_argument("--model", required=True,
                   choices=sorted(ca.MODEL_REGISTRY),
                   help="victim model (ViT-B/16, ViT-L/14, ViT-L/14-336px, "
                        "ViT-B/32)")
    p.add_argument("--dataset", required=True, choices=list(DATASETS),
                   help="target dataset (frozen data_split.json splits)")
    p.add_argument("--seed", type=int, required=True,
                   help="torch / numpy / cuda seed")
    p.add_argument("--target-cls", type=int, required=True,
                   help="target class index (dataset label space)")
    p.add_argument("--data-split", default=DEFAULT_SPLIT,
                   help="frozen split file (default: data_split.json next to "
                        "this script)")
    p.add_argument("--out", required=True,
                   help="output directory (attack_state.pt + attack_meta.json)")
    p.add_argument("--device", default="cuda", help="torch device (default cuda)")
    p.add_argument("--data-root", default=None,
                   help="override the dataset root (clip_adapters.DATA_ROOT)")
    p.add_argument("--batch", type=int, default=64,
                   help="batch size for evaluate.eval_model passes (default 64)")
    # ── Flip-S attack knobs (defaults = the published values) ─────────
    p.add_argument("--flip-surface", choices=["encoder", "extended"],
                   default="encoder",
                   help="attack surface: encoder (paper scope — never the "
                        "head/projection) / extended (encoder+projection)")
    p.add_argument("--max-flips", type=int, default=PUSH_BUDGET,
                   help=f"bit-flip budget (push-98: {PUSH_BUDGET}; "
                        f"Flip-S native: {BUDGET})")
    p.add_argument("--gamma", type=float, default=GAMMA,
                   help="Stage-B loss balance: MSE(clean) + gamma*CE(target) "
                        f"(default {GAMMA})")
    p.add_argument("--topk", type=int, default=TOPK_SCALES,
                   help=f"scale rows expanded per tensor/round (default {TOPK_SCALES})")
    p.add_argument("--topk2", type=int, default=TOPK_WEIGHTS,
                   help=f"code positions expanded per tensor/round (default {TOPK_WEIGHTS})")
    p.add_argument("--pool-scales", type=int, default=POOL_SCALES,
                   help=f"ranked scale-bit candidates verified per round (default {POOL_SCALES})")
    p.add_argument("--pool-weights", type=int, default=POOL_WEIGHTS,
                   help=f"ranked weight-bit candidates verified per round (default {POOL_WEIGHTS})")
    p.add_argument("--quick-n", type=int, default=QUICK_N,
                   help=f"search-split subset size for greedy verification "
                        f"(default {QUICK_N})")
    # ── Stage-A trigger knobs ────────────────────────────────────────
    p.add_argument("--trig-side", type=int, default=None,
                   help="trigger block side in px (default 3 x victim patch, "
                        "e.g. 48 for ViT-B/16)")
    p.add_argument("--trig-corner", choices=["br", "bl", "tr", "tl"],
                   default="br", help="trigger block corner (default br)")
    p.add_argument("--trig-iters", type=int, default=TRIG_ITERS,
                   help=f"Stage-A sign-PGD iterations (default {TRIG_ITERS})")
    p.add_argument("--atten-loss-weight", type=float, default=ATTEN_LOSS_WEIGHT,
                   help=f"Stage-A attention-loss weight (default {ATTEN_LOSS_WEIGHT})")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
