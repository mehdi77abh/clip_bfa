"""
trojvit_trigger.py — TrojViT Phase-1 (trigger generation) on CLIP.
=================================================================

**Scope: trigger generation only.**  TrojViT's Phase 2 ("tuned parameter
distillation" / Trojan insertion) is a classifier-head operation and has no
faithful CLIP-zero-shot analogue, so it is deliberately not implemented
here (see CLIP_ADAPTATION_ANALYSIS.md §4).

What this script does
---------------------
1. Selects the top-N patches per image by **salience** (default), attention
   maps, or random — TrojViT §3.1.
2. Optimises a shared patch perturbation with the **Attention-Target loss**
       L = L_target + λ · Σ_l L_ATTN^l
   where L_target is the CLIP target-cosine (default) or CE, and
   L_ATTN = −log Σ_{queries} attn(query → selected patches), with PCGrad
   gradient surgery between the two losses.
3. Applies the trigger per image (the release uses **replace** mode; the
   paper writes additive `X + P⊙M`; both are available via --trojvit-mode)
   and reports **trigger-only** ASR / CACC / clean-as-target / TAR.
4. Saves `trojvit_results.json`, `trigger_delta.pt`, and the per-image patch
   masks/indices.

No NGR, no head-row selection, no weight insertion.

Attention maps: OpenAI CLIP's `nn.MultiheadAttention` discards per-head
weights, so the vision blocks are replaced by `CaptureMHA`, which recomputes
the identical scaled-dot-product attention and exposes `(B, heads, S, S)`.
CLIP has one special token (offset +1). This script supports both the
OpenAI CLIP backend (`ViT-B/16`, `ViT-L/14`, `ViT-L/14-336px`, `ViT-B/32`) and
SigLIP (`SigLIP-B/16`, `SigLIP-so400m/14`). SigLIP has no CLS token (offset 0)
and exposes per-head maps natively through `vision_model(output_attentions=True)`.

Usage
-----
    cd bfa_baselines
    python trojvit_trigger.py --model ViT-B/16 --dataset Caltech101 \
        --seed 42 --target-cls 97 --n-eval 1000 \
        --trojvit-images 384 --trojvit-iters 250 --trojvit-lr 0.22

Results default to `bfa_baselines/baseline_runs/baselines/<model>/<dataset>/...`.
"""

import argparse
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PIPE_DIR = os.path.abspath(os.path.join(_HERE, ".."))
if _PIPE_DIR not in sys.path:
    sys.path.insert(0, _PIPE_DIR)
_BFA_DIR = _HERE

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import experiment as base_experiment
from adapters import encode_image, get_adapter
from data import DataModule
from evaluation import encode_image_batched
from models import ModelLoader
from target import TargetEmbedder
from utils import cleanup_gpu, model_tag, set_seed


# ══════════════════════════════════════════════════════════════════════
#  ATTENTION CAPTURE (OpenAI CLIP — same math as the original MHA)
# ══════════════════════════════════════════════════════════════════════
class CaptureMHA(nn.Module):
    """Drop-in replacement for CLIP's nn.MultiheadAttention that also exposes
    per-head attention maps (CLIP passes need_weights=False and discards them)."""

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


def install_attention_capture(model, backend="openai_clip"):
    """OpenAI CLIP: replace MHA with CaptureMHA; SigLIP: no replacement (the
    vision tower returns attentions natively via output_attentions=True)."""
    ad = get_adapter(model)
    blocks = list(ad.get_vision_transformer_blocks(model))
    if backend == "siglip":
        return blocks
    if not hasattr(blocks[0], "attn"):
        raise NotImplementedError(
            "trojvit_trigger.py: unsupported attention backend "
            f"{backend!r}; expected OpenAI CLIP `blk.attn` or SigLIP "
            "`blk.self_attn`.")
    for blk in blocks:
        if not isinstance(blk.attn, CaptureMHA):
            blk.attn = CaptureMHA(blk.attn)
    return blocks


# ══════════════════════════════════════════════════════════════════════
#  FORWARD / LOSSES
# ══════════════════════════════════════════════════════════════════════
def forward_capture(model, blocks, imgs, capture=True, backend="openai_clip"):
    if backend == "siglip":
        out = model.vision_model(pixel_values=imgs,
                                 output_attentions=bool(capture))
        feats = out.pooler_output
        attns = out.attentions if capture else None
        return feats, attns

    if capture:
        for blk in blocks:
            blk.attn.capture = True
            blk.attn.last_attn = None
    feats = encode_image(model, imgs)
    attns = [blk.attn.last_attn for blk in blocks] if capture else None
    if capture:
        for blk in blocks:
            blk.attn.capture = False
            blk.attn.last_attn = None
    return feats, attns


def target_loss_vector(feats, text_emb, target_vec, target_cls, loss_mode,
                       temperature):
    f = F.normalize(feats, dim=-1)
    if loss_mode == "cos":
        return -F.cosine_similarity(f, target_vec.unsqueeze(0), dim=-1)
    logits = f @ text_emb.T / temperature
    tgt = torch.full((feats.size(0),), int(target_cls),
                     device=feats.device, dtype=torch.long)
    return F.cross_entropy(logits, tgt, reduction="none")


def attention_loss_vector(attns, patch_idx, n_prefix=1, skip_first=True):
    """−log Σ_{selected patches} attn(query → patch), averaged over queries/layers."""
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


# ══════════════════════════════════════════════════════════════════════
#  PATCH SELECTION (TrojViT §3.1)
# ══════════════════════════════════════════════════════════════════════
def _patch_grid(h, w, patch_size):
    if h % patch_size or w % patch_size:
        raise ValueError(f"image {h}×{w} is not divisible by patch {patch_size}")
    return h // patch_size, w // patch_size


def select_patches(model, blocks, imgs, text_emb, target_vec, target_cls,
                   loss_mode, temperature, patch_select, num_patch,
                   patch_size, atten_layer, n_prefix=1,
                   backend="openai_clip"):
    """Return (B, num_patch) patch indices for each image."""
    b, _, h, w = imgs.shape
    gh, gw = _patch_grid(h, w, patch_size)

    if patch_select == "rand":
        return torch.randint(0, gh * gw, (b, num_patch), device=imgs.device)

    if patch_select == "attn":
        with torch.no_grad():
            _, attns = forward_capture(model, blocks, imgs, capture=True,
                                       backend=backend)
        layer = min(max(0, atten_layer), len(attns) - 1)
        a = attns[layer].mean(dim=1)               # (B, S, S)
        recv = a.mean(dim=-2)[:, n_prefix:]        # (B, n_patches)
        return recv.topk(num_patch, dim=1).indices

    if patch_select == "saliency":
        with torch.enable_grad():
            x = imgs.detach().clone().requires_grad_(True)
            feats = encode_image(model, x)
            lt = target_loss_vector(feats, text_emb, target_vec, target_cls,
                                    loss_mode, temperature).sum()
            g = torch.autograd.grad(lt, x)[0].detach().abs()
        # sum |grad| per patch: (B, 3, gh, p, gw, p) → (B, gh, gw)
        gp = g.view(b, 3, gh, patch_size, gw, patch_size) \
              .sum(dim=(1, 3, 5))
        return gp.flatten(1).topk(num_patch, dim=1).indices

    raise ValueError(f"unknown patch_select {patch_select!r}")


def build_masks(patch_idx, b, h, w, patch_size, device):
    """Per-image binary mask (B,1,H,W) from patch indices (B,N)."""
    gh, gw = _patch_grid(h, w, patch_size)
    mask = torch.zeros(b, 1, h, w, device=device)
    for i in range(b):
        for idx in patch_idx[i].tolist():
            r, c = divmod(int(idx), gw)
            mask[i, 0, r * patch_size:(r + 1) * patch_size,
                 c * patch_size:(c + 1) * patch_size] = 1.0
    return mask


# ══════════════════════════════════════════════════════════════════════
#  TRIGGER APPLICATION
# ══════════════════════════════════════════════════════════════════════
def apply_masked(imgs, delta, mask, mode, lo, hi):
    if mode == "replace":
        out = imgs * (1.0 - mask) + delta * mask
    else:                                           # add
        out = imgs + delta * mask
    return torch.max(torch.min(out, hi), lo)


def norm_bounds(model, device):
    mean, std = get_adapter(model).get_norm_stats(model)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device) \
                  .view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32, device=device) \
                 .view(1, 3, 1, 1)
    return (0.0 - mean_t) / std_t, (1.0 - mean_t) / std_t


# ══════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_trigger_only(model, blocks, imgs, labels, text_emb, target_cls,
                      patch_idx, delta, mode, patch_size, lo, hi,
                      batch_size=32):
    b, _, h, w = imgs.shape
    hit_ca = hit_asr = hit_cat = tot = 0
    for i in range(0, b, batch_size):
        x = imgs[i:i + batch_size]
        y = labels[i:i + batch_size]
        idx = patch_idx[i:i + batch_size]
        mask = build_masks(idx, x.size(0), h, w, patch_size, x.device)
        clean = F.normalize(encode_image_batched(model, x), dim=-1)
        pred_c = (clean @ text_emb.T).argmax(1)
        hit_ca += (pred_c == y).sum().item()
        hit_cat += (pred_c == target_cls).sum().item()
        trig = apply_masked(x, delta, mask, mode, lo, hi)
        ft = F.normalize(encode_image_batched(model, trig), dim=-1)
        pred_t = (ft @ text_emb.T).argmax(1)
        hit_asr += (pred_t == target_cls).sum().item()
        tot += x.size(0)
    n = max(tot, 1)
    return dict(asr=100.0 * hit_asr / n,
                cacc=100.0 * hit_ca / n,
                clean_as_target=100.0 * hit_cat / n,
                n=tot)


# ══════════════════════════════════════════════════════════════════════
#  CLI (TrojViT flags + the shared proposed-method CLI)
# ══════════════════════════════════════════════════════════════════════
def _parse_trojvit_cli(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--trojvit-images", type=int, default=384,
                   help="images used for trigger generation (paper: 384)")
    p.add_argument("--trojvit-batch", type=int, default=16)
    p.add_argument("--trojvit-iters", type=int, default=250)
    p.add_argument("--trojvit-lr", type=float, default=0.22)
    p.add_argument("--trojvit-step", type=int, default=10,
                   help="StepLR step size (release: 10)")
    p.add_argument("--trojvit-gamma", type=float, default=0.95)
    p.add_argument("--trojvit-num-patch", type=int, default=9)
    p.add_argument("--trojvit-patch-size", type=int, default=16)
    p.add_argument("--trojvit-patch-select", choices=["saliency", "attn", "rand"],
                   default="saliency")
    p.add_argument("--trojvit-atten-layer", type=int, default=4,
                   help="attention layer used for patch selection")
    p.add_argument("--trojvit-atten-weight", type=float, default=1.0,
                   help="λ for the attention-target loss (0 disables)")
    p.add_argument("--trojvit-loss", choices=["cos", "ce"], default="cos",
                   help="target loss: target-cosine (threat-model consistent) "
                        "or CE over the ZS logits")
    p.add_argument("--trojvit-temperature", type=float, default=0.07,
                   help="temperature for --trojvit-loss ce")
    p.add_argument("--trojvit-mode", choices=["replace", "add"],
                   default="replace",
                   help="trigger application: release uses replace; the paper "
                        "writes additive X+P⊙M")
    p.add_argument("--trojvit-init", choices=["random", "zero"], default="random",
                   help="trigger init: release uses a random normalized image; "
                        "the paper writes P=0")
    p.add_argument("--trojvit-fixed-masks", action="store_true",
                   help="use one fixed set of patch positions for all eval "
                        "images (release-code behaviour) instead of per-image")
    p.add_argument("--trojvit-eval-bs", type=int, default=32)
    ns, rest = p.parse_known_args(argv)
    return ns, rest


def _apply_trojvit_identity(cfg, args, cli):
    """Set the baseline run identity and record the TrojViT knobs."""
    cfg.SECTION = "baselines"
    cfg.VARIANT = "trojvit_trigger"
    cfg.TROJVIT_IMAGES = cli.trojvit_images
    cfg.TROJVIT_BATCH = cli.trojvit_batch
    cfg.TROJVIT_ITERS = cli.trojvit_iters
    cfg.TROJVIT_LR = cli.trojvit_lr
    cfg.TROJVIT_STEP = cli.trojvit_step
    cfg.TROJVIT_GAMMA = cli.trojvit_gamma
    cfg.TROJVIT_NUM_PATCH = cli.trojvit_num_patch
    cfg.TROJVIT_PATCH_SIZE = cli.trojvit_patch_size
    cfg.TROJVIT_PATCH_SELECT = cli.trojvit_patch_select
    cfg.TROJVIT_ATTEN_LAYER = cli.trojvit_atten_layer
    cfg.TROJVIT_ATTEN_WEIGHT = cli.trojvit_atten_weight
    cfg.TROJVIT_LOSS = cli.trojvit_loss
    cfg.TROJVIT_MODE = cli.trojvit_mode
    cfg.TROJVIT_INIT = cli.trojvit_init
    cfg.TROJVIT_FIXED_MASKS = bool(cli.trojvit_fixed_masks)
    cfg.TROJVIT_PHASE = "trigger_generation_only"

    if args.out_root is None:
        cfg.OUT_ROOT = os.path.join(_BFA_DIR, "baseline_runs")

    tag = model_tag(cfg.MODEL_NAME)
    run_seg = (f"trojvit_trigger_n{cli.trojvit_num_patch}"
               f"p{cli.trojvit_patch_size}")
    leaf = (f"{cfg.TASK}_c{cfg.TARGET_CLS}_s{cfg.SEED}_r{cfg.REP}"
            f"_{run_seg}")
    cfg.OUT_DIR = args.out_dir or os.path.join(
        cfg.OUT_ROOT, cfg.SECTION, tag, cfg.DATASET, leaf)
    cfg.RUN_ID = (f"{cfg.SECTION}__{tag}__{cfg.DATASET}__{cfg.TASK}__"
                  f"c{cfg.TARGET_CLS}__{run_seg}__s{cfg.SEED}__r{cfg.REP}")
    return cfg


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cli, rest = _parse_trojvit_cli(argv)
    sys.argv = [sys.argv[0]] + rest

    args = base_experiment.parse_args()
    cfg = base_experiment.build_config(args)
    cfg = _apply_trojvit_identity(cfg, args, cli)

    print(f"RUN_ID : {cfg.RUN_ID}")
    print(f"OUT_DIR: {cfg.OUT_DIR}")
    print(f"[TrojViT] phase=trigger-only  patches={cli.trojvit_num_patch}"
          f"×{cli.trojvit_patch_size}px  select={cli.trojvit_patch_select}"
          f"  loss={cli.trojvit_loss}  mode={cli.trojvit_mode}"
          f"  iters={cli.trojvit_iters}")

    backend = cfg.MODEL_ADAPTER
    if backend not in ("openai_clip", "siglip"):
        raise NotImplementedError(
            f"trojvit_trigger.py: unsupported adapter {backend!r}")
    print(f"  backend: {backend}")

    set_seed(cfg.SEED)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    dev = cfg.DEVICE
    t0 = time.time()

    # ── Model / data / text head ──
    model, preprocess = ModelLoader(cfg).load()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mparams = ModelLoader.detect_model_params(model)
    model_patch = int(mparams["patch_size"])
    if int(cli.trojvit_patch_size) != model_patch:
        print(f"  [warn] --trojvit-patch-size={cli.trojvit_patch_size} but the "
              f"victim patch is {model_patch}; using {model_patch} so the "
              f"patch indices align with the attention tokens.")
    patch_size = model_patch
    if backend == "siglip":
        # The eager attention path returns per-head maps; sdpa may not.
        for obj in (model, getattr(model, "vision_model", None)):
            if obj is not None:
                try:
                    obj.config._attn_implementation = "eager"
                except Exception:
                    pass
        n_prefix = 0                         # SigLIP has no CLS token
    else:
        n_prefix = 1                         # OpenAI CLIP CLS token
    blocks = install_attention_capture(model, backend)
    print(f"  attention capture: {backend} on {len(blocks)} blocks; "
          f"trigger patch = {patch_size}px (victim patch); "
          f"n_prefix={n_prefix}")

    ds, classes = DataModule(cfg).load_dataset(preprocess)
    text_emb = TargetEmbedder(cfg).build_text_embeddings(model, classes, dev)
    target_cls = int(cfg.TARGET_CLS)
    target_vec = text_emb[target_cls].detach()
    print(f"  {len(classes)} classes, target = '{classes[target_cls]}'")

    target_sel, calib_idx, grad_idx, eval_idx, test_idx, _ = \
        DataModule(cfg).prepare_data_splits(ds)

    eval_imgs, eval_labels = DataModule.load_tensors(ds, eval_idx, dev)
    eval_bs = int(cli.trojvit_eval_bs)

    # Trigger pool: clean images not used by the eval split (paper: 384).
    pool_idx = list(calib_idx) + list(grad_idx)
    need = max(0, int(cli.trojvit_images) - len(pool_idx))
    if need:
        pool_idx += list(test_idx[:need])
    n_pool = min(int(cli.trojvit_images), len(pool_idx))
    pool_idx = pool_idx[:n_pool]
    pool_imgs, pool_labels = DataModule.load_tensors(ds, pool_idx, dev)
    print(f"  trigger images: {n_pool}  eval images: {eval_imgs.size(0)}")

    b, _, h, w = pool_imgs.shape
    num_patch = int(cli.trojvit_num_patch)
    gh, gw = _patch_grid(h, w, patch_size)
    if num_patch > gh * gw:
        raise ValueError(
            f"trojvit: --trojvit-num-patch={num_patch} exceeds the "
            f"{gh}×{gw}={gh * gw} patch grid at {h}×{w} with patch "
            f"{patch_size}; lower the patch size or the patch count.")
    tar = 100.0 * num_patch * patch_size * patch_size / float(h * w)
    print(f"  patch grid: {gh}×{gw}  TAR={tar:.2f}%")

    lo, hi = norm_bounds(model, dev)

    # ── Per-batch patch selection for the trigger pool ──
    bs = max(1, int(cli.trojvit_batch))
    n_batches = int(math.ceil(n_pool / bs))
    patch_idx_batches, mask_batches = [], []
    for bi in range(n_batches):
        xb = pool_imgs[bi * bs:(bi + 1) * bs]
        idx = select_patches(
            model, blocks, xb, text_emb, target_vec, target_cls,
            cli.trojvit_loss, cli.trojvit_temperature,
            cli.trojvit_patch_select, num_patch, patch_size,
            cli.trojvit_atten_layer, n_prefix=n_prefix, backend=backend)
        patch_idx_batches.append(idx)
        mask_batches.append(build_masks(idx, xb.size(0), h, w,
                                        patch_size, xb.device))
    print(f"  selected patches for {n_batches} trigger batch(es)")

    # ── Optimise the shared trigger delta ──
    if cli.trojvit_init == "random":
        delta = lo + torch.rand(1, 3, h, w, device=dev) * (hi - lo)
    else:
        delta = torch.zeros(1, 3, h, w, device=dev)
    delta.requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=float(cli.trojvit_lr))
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=int(cli.trojvit_step), gamma=float(cli.trojvit_gamma))
    atten_w = float(cli.trojvit_atten_weight)

    print("\n[trigger generation]")
    for it in range(int(cli.trojvit_iters)):
        bi = it % n_batches
        xb = pool_imgs[bi * bs:(bi + 1) * bs]
        mask = mask_batches[bi]
        idx = patch_idx_batches[bi]

        with torch.enable_grad():
            x_hat = apply_masked(xb, delta, mask, cli.trojvit_mode, lo, hi)
            feats, attns = forward_capture(model, blocks, x_hat,
                                           capture=(atten_w > 0),
                                           backend=backend)
            lt = target_loss_vector(feats, text_emb, target_vec, target_cls,
                                    cli.trojvit_loss,
                                    cli.trojvit_temperature)
            la = (attention_loss_vector(attns, idx, n_prefix=n_prefix)
                  if atten_w > 0 else None)

            if la is None:
                g = torch.autograd.grad(lt.mean(), delta)[0]
            else:
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
            delta.clamp_(lo, hi)

        if (it + 1) % 10 == 0 or it == 0:
            msg = f"  iter {it + 1:>3d}/{cli.trojvit_iters}  " \
                  f"L_target={lt.mean().item():+.4f}"
            if la is not None:
                msg += f"  L_attn={la.mean().item():+.4f}"
            msg += f"  lr={sched.get_last_lr()[0]:.4f}"
            print(msg, flush=True)

    delta = delta.detach()
    search_time = time.time() - t0
    torch.save(delta.cpu(), os.path.join(cfg.OUT_DIR, "trigger_delta.pt"))

    # ── Trigger-only evaluation ──
    print("\n[evaluation — trigger only, no weight modification]")
    pool_idx_all = torch.cat(patch_idx_batches, dim=0)
    pool_metrics = eval_trigger_only(
        model, blocks, pool_imgs, pool_labels, text_emb, target_cls,
        pool_idx_all, delta, cli.trojvit_mode, patch_size, lo, hi,
        batch_size=eval_bs)

    if cli.trojvit_fixed_masks:
        fixed = patch_idx_batches[0][0].view(1, -1)
        eval_idx_all = fixed.expand(eval_imgs.size(0), -1)
        print("  eval masks: FIXED positions (release-code behaviour)")
    else:
        eval_list = []
        for i in range(0, eval_imgs.size(0), eval_bs):
            xb = eval_imgs[i:i + eval_bs]
            eval_list.append(select_patches(
                model, blocks, xb, text_emb, target_vec, target_cls,
                cli.trojvit_loss, cli.trojvit_temperature,
                cli.trojvit_patch_select, num_patch, patch_size,
                cli.trojvit_atten_layer, n_prefix=n_prefix, backend=backend))
        eval_idx_all = torch.cat(eval_list, dim=0)
        print("  eval masks: PER-IMAGE selection (paper behaviour)")

    eval_metrics = eval_trigger_only(
        model, blocks, eval_imgs, eval_labels, text_emb, target_cls,
        eval_idx_all, delta, cli.trojvit_mode, patch_size, lo, hi,
        batch_size=eval_bs)

    torch.save({
        "trigger_patch_idx": pool_idx_all.cpu(),
        "eval_patch_idx": eval_idx_all.cpu(),
        "num_patch": num_patch,
        "patch_size": patch_size,
        "tar": tar,
        "mode": cli.trojvit_mode,
        "fixed_masks": bool(cli.trojvit_fixed_masks),
    }, os.path.join(cfg.OUT_DIR, "trojvit_masks.pt"))

    metrics = dict(
        asr_trigger=eval_metrics["asr"],
        cacc=eval_metrics["cacc"],
        clean_as_target=eval_metrics["clean_as_target"],
        asr_trigger_pool=pool_metrics["asr"],
        cacc_pool=pool_metrics["cacc"],
        tar=tar,
        n_eval=eval_metrics["n"],
        n_trigger_images=n_pool,
        num_patch=num_patch,
        patch_size=patch_size,
        phase="trigger_generation_only",
        # aliases so a generic table can read a final ASR/CACC
        asr_final=eval_metrics["asr"],
        cacc_final=eval_metrics["cacc"],
    )

    print("\n" + "=" * 60)
    print("SUMMARY — TROJVIT PHASE 1 (TRIGGER ONLY)")
    print("=" * 60)
    print(f"  TAR          : {tar:.2f}%  ({num_patch}×{patch_size}×{patch_size})")
    print(f"  ASR (eval)   : {eval_metrics['asr']:.2f}%")
    print(f"  CACC (eval)  : {eval_metrics['cacc']:.2f}%")
    print(f"  clean→target : {eval_metrics['clean_as_target']:.2f}%")
    print(f"  ASR (pool)   : {pool_metrics['asr']:.2f}%")
    print(f"  elapsed      : {search_time:.1f}s")

    save = {
        "method": "trojvit_trigger",
        "phase": "trigger_generation_only",
        "config": {k: v for k, v in vars(cfg).items()
                   if not k.startswith("_")},
        "target_class": classes[target_cls],
        "trojvit": {
            "num_patch": num_patch,
            "patch_size": patch_size,
            "tar": tar,
            "patch_select": cli.trojvit_patch_select,
            "atten_layer": cli.trojvit_atten_layer,
            "atten_weight": atten_w,
            "loss": cli.trojvit_loss,
            "mode": cli.trojvit_mode,
            "init": cli.trojvit_init,
            "iters": int(cli.trojvit_iters),
            "lr": float(cli.trojvit_lr),
            "fixed_masks": bool(cli.trojvit_fixed_masks),
        },
        "metrics": metrics,
        "section": "baselines",
        "rep": getattr(cfg, "REP", 1),
        "search_time_sec": search_time,
    }
    with open(os.path.join(cfg.OUT_DIR, "trojvit_results.json"), "w") as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\n  Results saved to {cfg.OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

