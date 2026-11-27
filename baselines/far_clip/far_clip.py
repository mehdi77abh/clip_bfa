"""
far_clip.py — Forget-and-Rewire (FaR) defence runner for OpenAI-CLIP vision towers.
==================================================================================

CLIP adaptation of the FaR defence (Nazari et al., "Forget and Rewire:
Enhancing the Resilience of Transformer-based Models against Bit-Flip
Attacks", USENIX Security '24).  Reference implementation: ``../far_demo/``
(HF ViT only); paper: ``../far_demo/paper_usenix24_forget_and_rewire.pdf``.

What FaR does (paper §5, Algorithm 1) — once per FaR iteration:
  1. IMPORTANT parameter — the top-|gradient| weight of the task loss w.r.t.
     the model weights (gradients recomputed after every FaR op).
  2. DEAD inputs — the lowest-|gradient| weights of the same output row in
     the same layer ("dead neurons", paper §5.2).
  3. FORGET — the dead weights are overwritten with the important weight's
     value (weight copies).
  4. REWIRE — at forward time the important input activation is divided by
     (n_deads + 1) and rerouted onto the dead input channels.  Each of the
     n+1 identical weight copies then carries 1/(n+1) of the gradient and
     1/(n+1) of the bit-flip damage of the concealed parameter — a
     gradient-based attacker finds no single point of failure.

Scope (agreed with the maintainer, faithful to the paper's "Linear layers"
incl. Query/Key/Value, paper §2.1): the vision tower only — every linear
layer of every ResidualAttentionBlock:

    visual.transformer.resblocks.{i}.attn          fused q/k/v (in_proj_weight)
    visual.transformer.resblocks.{i}.attn.out_proj
    visual.transformer.resblocks.{i}.mlp.c_fc
    visual.transformer.resbuffers.{i}.mlp.c_proj

The fused q/k/v matrix is handled by ``FaRMHA`` (an nn.MultiheadAttention
subclass) and the plain linears by ``FaRLinear`` (an nn.Linear subclass).
Both keep the vanilla module's parameters and names, so the model's
state_dict keys are byte-identical to a vanilla CLIP — the attack
pipeline's Quantizer / adapter weight references keep working unchanged.

Checkpoint contract (consumed by ``pipeline.load_far_state`` via
``experiment.py --far-ckpt PATH``):

    {"state_dict": model.state_dict(),   # keys unchanged vs vanilla CLIP
     "far_cfg":   {target_key: [{"weight_dim": i, "conceal": c,
                                 "deads": [j, ...]}, ...]},
     "meta":      {...}}

``far_cfg`` is REQUIRED for the defence to exist: without the runtime
rewiring bound to the victim's linears the weight copies alone change the
model's function and provide no protection.  ``pipeline.load_far_state``
calls ``install_far(model, far_cfg)`` when the checkpoint carries one.

Usage (runner):
    python far_clip.py --model ViT-B/16 --data-root <ImageFolder> \\
        --out-dir defended_models/

Usage (library):
    from far_clip import install_far, load_far_checkpoint
    model, _ = clip.load("ViT-B/16"); model = model.float()
    load_far_checkpoint(model, "defended_models/far_vit_b16.pt")
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  FaR CORE — forward-time input rewiring (paper §5.2 "Rewire")
# ══════════════════════════════════════════════════════════════════════
# A FaR op is {"weight_dim": i, "conceal": c, "deads": [j1, …]}:
#   output row i sees a rewired input:
#       x'[c]  = x[c] / (n_deads + 1)
#       x'[jk] = x[c] / (n_deads + 1)      (the forgotten channels)
# and the weights W[i, jk] are overwritten with W[i, c] ("Forget").
# ``far_forward`` is the exact algebraic equivalent of the demo's
# per-row input surgery (far_demo/custom_linear.py:custom_f_linear1),
# vectorised over any batch/sequence shape (feature dim = last dim).


def far_forward(out, x, weight, ops):
    """Return ``out`` + the per-row input-rewiring correction for ``ops``.

    ``out`` is the plain F.linear(x, weight) result.  For the fused q/k/v
    matrix, ``weight`` is in_proj_weight [3D, D] and rows 0..3D-1 address
    the q/k/v rows.  Differentiable; exact for disjoint ops on one row.
    """
    if not ops:
        return out
    for op in ops:
        i, c, deads = op["weight_dim"], op["conceal"], op["deads"]
        n = len(deads) + 1
        xc = x[..., c] / n                        # rewired activation
        delta = weight[i, c] * (xc - x[..., c])
        for j in deads:
            delta = delta + weight[i, j] * (xc - x[..., j])
        idx = torch.tensor([i], dtype=torch.long, device=out.device)
        out = out.index_add(-1, idx, delta.unsqueeze(-1))
    return out


class FaRLinear(nn.Linear):
    """Drop-in nn.Linear with FaR input rewiring (state_dict keys unchanged).

    ``far_ops`` is a plain attribute — it is NOT part of the state_dict;
    the ops travel in the checkpoint's 'far_cfg' and are bound by
    install_far().
    """

    def __init__(self, in_features, out_features, bias=True,
                 device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias,
                         device=device, dtype=dtype)
        self.far_ops = []

    @classmethod
    def from_module(cls, lin):
        """Wrap an existing nn.Linear, reusing its Parameter objects."""
        new = cls(lin.in_features, lin.out_features,
                  bias=lin.bias is not None,
                  device=lin.weight.device, dtype=lin.weight.dtype)
        new.weight = lin.weight          # keep the same Parameter objects
        if lin.bias is not None:
            new.bias = lin.bias
        return new

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        return far_forward(out, x, self.weight, self.far_ops)


class FaRMHA(nn.MultiheadAttention):
    """nn.MultiheadAttention whose fused q/k/v projection runs FaR.

    Self-attention only (CLIP's ResidualAttentionBlock calls
    ``attn(x, x, x, need_weights=False, attn_mask=…)``); parameters and
    attributes are identical to the vanilla module, so state_dict keys and
    the adapter's weight references (attn.in_proj_weight, attn.out_proj.*)
    are unchanged.  ``far_ops`` address rows of in_proj_weight (0..3D-1).
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
                 add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None, batch_first=False, device=None, dtype=None):
        super().__init__(embed_dim, num_heads, dropout=dropout, bias=bias,
                         add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
                         kdim=kdim, vdim=vdim, batch_first=batch_first,
                         device=device, dtype=dtype)
        self.far_ops = []

    @classmethod
    def from_module(cls, m):
        """Wrap an existing nn.MultiheadAttention, reusing its Parameters."""
        new = cls(m.embed_dim, m.num_heads, dropout=m.dropout,
                  bias=m.in_proj_bias is not None,
                  add_bias_kv=m.bias_k is not None,
                  add_zero_attn=m.add_zero_attn,
                  kdim=m.kdim, vdim=m.vdim, batch_first=m.batch_first,
                  device=m.in_proj_weight.device,
                  dtype=m.in_proj_weight.dtype)
        new.in_proj_weight = m.in_proj_weight
        new.in_proj_bias = m.in_proj_bias
        new.out_proj = m.out_proj          # may already be a FaRLinear
        new.bias_k, new.bias_v = m.bias_k, m.bias_v
        return new

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None,
                average_attn_weights=True, is_causal=False):
        if key is not query or value is not query:
            raise NotImplementedError("FaRMHA supports self-attention only")
        if key_padding_mask is not None or self.bias_k is not None \
                or self.add_zero_attn:
            raise NotImplementedError("FaRMHA: key_padding_mask / bias_kv / "
                                      "add_zero_attn not supported")
        x = query if not self.batch_first else query.transpose(0, 1)
        S, B, _ = x.shape                    # CLIP runs LND (seq-first)
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = far_forward(qkv, x, self.in_proj_weight, self.far_ops)
        q, k, v = qkv.split(self.embed_dim, dim=-1)

        hd = self.embed_dim // self.num_heads

        def heads(t):                        # [S,B,D] -> [B,H,S,hd]
            return t.view(S, B, self.num_heads, hd).permute(1, 2, 0, 3)

        qh, kh, vh = heads(q), heads(k), heads(v)

        mask = attn_mask                     # nn.MHA semantics: True = masked
        if mask is not None:
            if mask.dtype == torch.bool:
                mask = torch.zeros_like(mask, dtype=qh.dtype) \
                    .masked_fill_(mask, float("-inf"))
            else:
                mask = mask.to(qh.dtype)
            if mask.dim() == 2:
                mask = mask.view(1, 1, *mask.shape)
            elif mask.dim() == 3:            # [B*H,S,S] or [B,S,S]
                mask = mask.view(B, self.num_heads, *mask.shape[1:]) \
                    if mask.shape[0] == B * self.num_heads \
                    else mask.unsqueeze(1)

        out = F.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0)
        out = out.permute(2, 0, 1, 3).reshape(S, B, self.embed_dim)
        out = self.out_proj(out)             # FaRLinear applies its own ops
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, None                     # callers use [0] (need_weights=False)


# ══════════════════════════════════════════════════════════════════════
#  TARGET REGISTRY / MODULE SWAP / CFG APPLY
# ══════════════════════════════════════════════════════════════════════
def vision_blocks(model):
    """The vision tower's ResidualAttentionBlocks (ViT towers only)."""
    vis = model.visual
    if not hasattr(vis, "transformer") or not hasattr(vis.transformer, "resblocks"):
        raise ValueError("far_clip supports VisionTransformer towers "
                         "(ViT-B/16, ViT-L/14, …) — not ResNets")
    return list(vis.transformer.resblocks)


def far_targets(model):
    """Yield (target_key, module) for every FaR-targetable vision linear."""
    for i, blk in enumerate(vision_blocks(model)):
        base = f"visual.transformer.resblocks.{i}"
        yield f"{base}.attn", blk.attn            # fused q/k/v target
        yield f"{base}.attn.out_proj", blk.attn.out_proj
        yield f"{base}.mlp.c_fc", blk.mlp.c_fc
        yield f"{base}.mlp.c_proj", blk.mlp.c_proj


def resolve(model, key):
    """Attribute-lookup a module from its dotted target key."""
    mod = model
    for attr in key.split("."):
        mod = getattr(mod, attr)
    return mod


def _weight_of(mod):
    return mod.in_proj_weight if isinstance(mod, FaRMHA) else mod.weight


def _used_cols(ops, row):
    """Input-column indices already used by FaR ops on output ``row``."""
    used = set()
    for e in ops:
        if e["weight_dim"] == row:
            used.add(e["conceal"])
            used.update(e["deads"])
    return used


def install_far(model, far_cfg=None):
    """Swap the vision tower's linears for FaR modules and bind ``far_cfg``.

    Idempotent; state_dict keys are unchanged.  With far_cfg=None the model
    runs the vanilla function (ops added later via add_far_op)."""
    for key, mod in list(far_targets(model)):
        if not isinstance(mod, (FaRLinear, FaRMHA)):
            parent_key, _, name = key.rpartition(".")
            parent = resolve(model, parent_key) if parent_key else model
            if key.endswith(".attn"):
                setattr(parent, name, FaRMHA.from_module(mod))
            else:
                setattr(parent, name, FaRLinear.from_module(mod))
    if far_cfg:
        for key, ops in far_cfg.items():
            mod = resolve(model, key)
            if not isinstance(mod, (FaRLinear, FaRMHA)):
                raise ValueError(f"far_cfg target {key!r} is not a FaR module")
            mod.far_ops = copy.deepcopy(ops)
    return model


def add_far_op(model, key, op):
    """Apply one FaR step (Forget + Rewire) to target ``key``.

    Forget: W[i, dead] = W[i, conceal] for every dead input (paper §5.2:
    "the value of the forgotten parameter is overwritten by that of the
    important parameter").  Rewire: recorded in the module's far_ops and
    applied at forward time.  Merges into an existing op when the same
    (row, conceal) is already concealed (demo: apply_far_cfg_to_model)."""
    mod = resolve(model, key)
    weight = _weight_of(mod)
    i, c, deads = int(op["weight_dim"]), int(op["conceal"]), [int(d) for d in op["deads"]]
    used = _used_cols(mod.far_ops, i)
    if c in used and not any(e["weight_dim"] == i and e["conceal"] == c
                             for e in mod.far_ops):
        raise ValueError(f"conceal col {c} is already a dead input on row {i}")
    if any(d in used or d == c for d in deads):
        raise ValueError("deads overlap an existing FaR op or the concealed input")
    with torch.no_grad():
        src = weight.data[i, c].clone()               # clone: no aliasing
        weight.data[i, deads] = src                   # Forget (weight copy)
    for e in mod.far_ops:
        if e["weight_dim"] == i and e["conceal"] == c:    # merge branch
            e["deads"].extend(d for d in deads if d not in e["deads"])
            return mod
    mod.far_ops.append({"weight_dim": i, "conceal": c, "deads": deads})
    return mod


def collect_far_cfg(model):
    """The checkpoint's far_cfg: {target_key: [op, …]}."""
    cfg = {}
    for key, mod in far_targets(model):
        ops = mod.far_ops if isinstance(mod, (FaRLinear, FaRMHA)) else []
        if ops:
            cfg[key] = copy.deepcopy(ops)
    return cfg


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT I/O
# ══════════════════════════════════════════════════════════════════════
def save_far_checkpoint(model, far_cfg, path, meta=None):
    """Save {"state_dict", "far_cfg", "meta"} — the --far-ckpt contract
    (pipeline.load_far_state)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "far_cfg": far_cfg,
                "meta": meta or {}}, path)
    return path


def load_far_checkpoint(model, path, map_location="cpu"):
    """Load a far_clip checkpoint into ``model`` (installs the FaR
    rewiring first, then the weights).  Returns the checkpoint meta."""
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    far_cfg = ckpt.get("far_cfg") if isinstance(ckpt, dict) else None
    if isinstance(far_cfg, dict) and far_cfg:
        install_far(model, far_cfg)
    model.load_state_dict(state, strict=True)
    return ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}


# ══════════════════════════════════════════════════════════════════════
#  FaR RUNNER — important/dead selection + the FaR loop (paper Alg. 1)
# ══════════════════════════════════════════════════════════════════════
def _zs_accuracy(model, images, labels, text_features):
    """Zero-shot accuracy of ``images`` against (normalised) text_features."""
    with torch.no_grad():
        feats = model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * feats @ text_features.t()
        return (logits.argmax(-1) == labels).float().mean().item() * 100.0


def _select_important(model, frac_size, top_k=4096):
    """The most important VALID parameter over all FaR targets.

    Ranking = |gradient| of the task loss w.r.t. the weights (paper §5.1).
    Validity (demo: compute_gradient_wrt_input): a row may conceal at most
    in_features/frac_size inputs, and the column must be free (or already
    concealed on that row → merge)."""
    cands = [(key, w) for key, mod in far_targets(model)
             for w in [_weight_of(mod)]]
    grads = [w.grad.abs().reshape(-1) if w.grad is not None
             else torch.zeros(w.numel(), device=w.device)
             for _, w in cands]
    allg = torch.cat(grads)
    k = min(top_k, allg.numel())
    _, idxs = torch.topk(allg, k)

    offsets, acc = [], 0
    for _, w in cands:
        offsets.append(acc)
        acc += w.numel()

    import bisect
    for flat in idxs.tolist():
        li = bisect.bisect_right(offsets, flat) - 1
        key, w = cands[li]
        row, col = divmod(flat - offsets[li], w.shape[1])
        ops = resolve(model, key).far_ops
        used = _used_cols(ops, row)
        if sum(1 for e in ops if e["weight_dim"] == row) >= max(1, w.shape[1] // frac_size):
            continue                      # row's conceal budget exhausted
        if col in used and not any(e["weight_dim"] == row and e["conceal"] == col
                                  for e in ops):
            continue                      # col is dead on this row already
        if not _select_deads(w, row, used | {col}, 1):
            continue                      # no dead input left on the row
        return key, w, row, col
    return None, None, None, None


def _select_deads(w, row, exclude, div_fac):
    """The div_fac least-important free input columns of output ``row``
    (demo: get_dead_params → inverse gradient sort within the layer)."""
    g = w.grad[row].abs() if w.grad is not None \
        else torch.zeros(w.shape[1], device=w.device)
    deads = []
    for j in torch.argsort(g).tolist():
        if j in exclude:
            continue
        deads.append(j)
        if len(deads) == div_fac:
            break
    return deads


def far_defend(model, images, labels, text_features, *, max_far=30, div_fac=5,
               frac_size=4, acc_drop_stop=2.0, verbose=True):
    """Run the FaR loop on ``model`` (must be an installed CLIP).

    images: [N,3,H,W] calib batch; labels: [N]; text_features: normalised
    class-prompt embeddings [C, D].  Returns (far_cfg, history)."""
    install_far(model)
    org_acc = _zs_accuracy(model, images, labels, text_features)
    history = [{"far": 0, "acc": org_acc}]
    if verbose:
        print(f"[far_clip] clean calib accuracy: {org_acc:.2f}%")

    for it in range(1, max_far + 1):
        # ---- Step 1: important parameter (gradients recomputed each op,
        #      paper §5.2: "after each forgetting and rewiring step, the
        #      entire gradient calculation process must be repeated")
        model.zero_grad(set_to_none=True)
        feats = model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        loss = F.cross_entropy(
            model.logit_scale.exp() * feats @ text_features.t(), labels)
        loss.backward()

        key, w, row, col = _select_important(model, frac_size)
        if key is None:
            print("[far_clip] no valid important parameter left — stopping")
            break

        # ---- Step 2: dead inputs of the same row
        used = _used_cols(resolve(model, key).far_ops, row) | {col}
        deads = _select_deads(w, row, used, div_fac)
        if not deads:
            print("[far_clip] no dead inputs left on the selected row — stopping")
            break

        # ---- Steps 3+4: Forget (weight copy) + Rewire (forward-time)
        add_far_op(model, key, {"weight_dim": row, "conceal": col, "deads": deads})

        acc = _zs_accuracy(model, images, labels, text_features)
        entry = {"far": it, "acc": acc, "target": key, "weight_dim": row,
                 "conceal": col, "deads": deads}
        history.append(entry)
        if verbose:
            print(f"[far_clip] FaR {it}: {key} row={row} conceal={col} "
                  f"deads={deads}  calib acc={acc:.2f}%")

        # ---- stop when the accuracy drops (paper §6: stop at accuracy drop)
        if acc_drop_stop > 0 and acc < org_acc - acc_drop_stop:
            if verbose:
                print(f"[far_clip] accuracy dropped >{acc_drop_stop} pp — stopping")
            break

    return collect_far_cfg(model), history


# ══════════════════════════════════════════════════════════════════════
#  CALIBRATION DATA (minimal ImageFolder — no torchvision dependency)
# ══════════════════════════════════════════════════════════════════════
_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_calib_images(data_root, preprocess, n_samples, seed,
                      classes_file=None, prompt_template="a photo of a {}"):
    """Sample ≤ n_samples (image, label) pairs from an ImageFolder layout
    (one sub-directory per class) and build the class prompts."""
    from PIL import Image
    classes = sorted(d for d in os.listdir(data_root)
                     if os.path.isdir(os.path.join(data_root, d)))
    if not classes:
        raise ValueError(f"{data_root}: no class sub-directories found")
    samples = []
    for ci, cname in enumerate(classes):
        cdir = os.path.join(data_root, cname)
        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith(_IMG_EXT):
                samples.append((os.path.join(cdir, fn), ci))
    if not samples:
        raise ValueError(f"{data_root}: no images found")
    random.Random(seed).shuffle(samples)
    samples = samples[:n_samples]

    images = torch.stack([preprocess(Image.open(p).convert("RGB"))
                          for p, _ in samples])
    labels = torch.tensor([l for _, l in samples], dtype=torch.long)

    if classes_file:
        with open(classes_file) as f:
            names = [ln.strip() for ln in f if ln.strip()]
        if len(names) != len(classes):
            raise ValueError(f"{classes_file}: {len(names)} names but "
                             f"{len(classes)} class folders")
    else:
        names = [c.replace("_", " ") for c in classes]
    prompts = [prompt_template.format(n) for n in names]
    return images, labels, prompts


# ══════════════════════════════════════════════════════════════════════
#  CLI — the defended-victim runner
# ══════════════════════════════════════════════════════════════════════
def model_tag(name):
    return name.replace("/", "-").replace(" ", "").lower()


def main():
    ap = argparse.ArgumentParser(
        description="FaR (Forget-and-Rewire) defence for OpenAI-CLIP "
                    "vision towers → defended victim checkpoint")
    ap.add_argument("--model", default="ViT-B/16",
                    help="OpenAI clip model name (default: ViT-B/16)")
    ap.add_argument("--data-root", required=True,
                    help="ImageFolder root (class sub-dirs) with calib images")
    ap.add_argument("--classes-file", default=None,
                    help="optional class-name file (one per line, folder "
                         "order); default: folder names, '_' → ' '")
    ap.add_argument("--prompt-template", default="a photo of a {}",
                    help="class-prompt template (default: 'a photo of a {}')")
    ap.add_argument("--n-samples", type=int, default=32,
                    help="calibration images for the gradient ranking "
                         "(default: 32)")
    ap.add_argument("--max-far", type=int, default=30,
                    help="FaR iterations = concealed parameters (default: 30)")
    ap.add_argument("--div-fac", type=int, default=5,
                    help="dead inputs per concealed parameter — the paper's "
                         "division factor (default: 5, paper Fig. 9 best)")
    ap.add_argument("--frac-size", type=int, default=4,
                    help="max concealed inputs per output row = "
                         "in_features/frac_size (default: 4)")
    ap.add_argument("--acc-drop-stop", type=float, default=2.0,
                    help="stop when calib accuracy drops more than this many "
                         "percentage points (default: 2.0; 0 = never stop)")
    ap.add_argument("--out-dir", default="defended_models",
                    help="output directory (default: defended_models)")
    ap.add_argument("--out-name", default=None,
                    help="checkpoint file name (default: far_{model_tag}.pt)")
    ap.add_argument("--device", default="auto",
                    help="cuda | cpu | auto (default: auto)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    import clip
    model, preprocess = clip.load(args.model, device=device)
    model = model.float()
    print(f"[far_clip] loaded {args.model} on {device}")

    images, labels, prompts = load_calib_images(
        args.data_root, preprocess, args.n_samples, args.seed,
        classes_file=args.classes_file,
        prompt_template=args.prompt_template)
    images, labels = images.to(device), labels.to(device)
    print(f"[far_clip] calib set: {images.shape[0]} images, "
          f"{len(prompts)} classes")

    with torch.no_grad():
        tok = clip.tokenize(prompts).to(device)
        text_features = model.encode_text(tok)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    far_cfg, history = far_defend(
        model, images, labels, text_features,
        max_far=args.max_far, div_fac=args.div_fac, frac_size=args.frac_size,
        acc_drop_stop=args.acc_drop_stop)

    out_name = args.out_name or f"far_{model_tag(args.model)}.pt"
    out_path = os.path.join(args.out_dir, out_name)
    meta = {"model": args.model, "data_root": args.data_root,
            "n_samples": args.n_samples, "max_far": args.max_far,
            "div_fac": args.div_fac, "frac_size": args.frac_size,
            "acc_drop_stop": args.acc_drop_stop, "seed": args.seed,
            "n_far_ops": sum(len(v) for v in far_cfg.values()),
            "final_acc": history[-1]["acc"]}
    save_far_checkpoint(model, far_cfg, out_path, meta=meta)

    cfg_path = os.path.join(args.out_dir,
                            out_name.replace(".pt", "_far_cfg.json"))
    with open(cfg_path, "w") as f:
        json.dump({"far_cfg": far_cfg, "history": history, "meta": meta},
                  f, indent=2)

    print(f"[far_clip] defended victim saved: {out_path}")
    print(f"[far_clip] FaR config saved:     {cfg_path}")
    print(f"[far_clip] run the attack with:  --far-ckpt {out_path}")


if __name__ == "__main__":
    main()
