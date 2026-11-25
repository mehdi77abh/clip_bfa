"""
evaluate.py — shared ASR / CACC evaluator for the CLIP baseline suite.
======================================================================

THE single place where ASR and CACC are computed.  Every attack script
(`attack_trojvit.py`, `attack_proflip.py`, `attack_flips.py`) writes an
`attack_state.pt`, and this module evaluates it on the frozen eval split
from `data_split.json` with identical semantics for every method.

`attack_state.pt` contract (torch.save dict):

    {
      "method":        "trojvit" | "proflip" | "flips",
      "model":         "ViT-B/16" | "ViT-L/14" | "ViT-L/14-336px" | "ViT-B/32",
      "dataset":       "Caltech101" | "ImageNet1k" | "DTD",
      "seed":          int,
      "target_cls":    int,
      "param_overrides": {param_name: FloatTensor},   # final attacked values of
          # EVERY parameter whose evaluated value differs from the clean
          # checkpoint (i.e. all dequantised quantised tensors + every flipped
          # tensor).  evaluate.py copies these into a fresh fp32 victim.
      "trigger": {                                     # applied in [0,1] pixel space
          "kind": "single" | "per_image_patches",
          "delta01": FloatTensor [1,3,H,W],            # H=W=input_px, values [0,1]
          "mode": "replace" | "add",
          # kind == "single":
          "mask": FloatTensor [1,1,H,W] in {0,1},      # optional (None = full image)
          # kind == "per_image_patches":
          "patch_indices": LongTensor [N, n_patch, 2], # (row, col) top-left, one
                                                      # row-group per image, in the
                                                      # SAME order as the evaluated
                                                      # images (data_split "eval" order)
          "patch_size": int,
      },
      "flip_log": [ {tensor, index, bit, old, new, ...}, ... ],   # metadata
      "meta": {...},                                   # free-form metadata
    }

Metric definitions (shared by every method):
  * ASR  = P(argmax == target_cls | triggered) over eval images with
           label != target_cls  (asr_incl_target includes target-class images).
  * CACC = P(argmax == label | clean) over the eval split.
  * clean_as_target = P(argmax == target_cls | clean).

Usage (CLI):
    python evaluate.py --state results/proflip/ViT-B16/Caltech101/s42/attack_state.pt
    python evaluate.py --state ... --out .../eval.json --device cuda --batch 64

Usage (from an attack script, for quick in-search evaluation):
    from evaluate import eval_model
    m = eval_model(model, text_emb, imgs01, labels, target_cls,
                   trigger=trig_dict, device=dev, batch_size=32)
"""

import argparse
import json
import os
import time

import torch

import clip_adapters as ca

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPLIT = os.path.join(_HERE, "data_split.json")


# ═════════════════════════════════════════════════════════════════════════
# trigger application (method-agnostic)
# ═════════════════════════════════════════════════════════════════════════

def apply_trigger_dict(imgs01, trig):
    """Apply an attack_state trigger dict to [N,3,H,W] pixel images."""
    if trig is None:
        return imgs01
    kind = trig.get("kind", "single")
    delta = trig["delta01"].to(imgs01.device)
    mode = trig.get("mode", "replace")
    if kind == "single":
        mask = trig.get("mask")
        if mask is not None:
            mask = mask.to(imgs01.device)
        return ca.apply_trigger(imgs01, delta, mask, mode)
    if kind == "per_image_patches":
        return ca.per_image_patch_trigger(
            imgs01, delta, trig["patch_indices"].to(imgs01.device),
            int(trig["patch_size"]), mode)
    raise ValueError(f"unknown trigger kind {kind!r}")


# ═════════════════════════════════════════════════════════════════════════
# metrics
# ═════════════════════════════════════════════════════════════════════════

def eval_model(model, text_emb, imgs01, labels, target_cls,
               trigger=None, device="cuda", batch_size=64):
    """ASR / CACC for one model + one image set.

    `imgs01` [N,3,H,W] in [0,1] pixel space, `labels` [N].  For
    kind="per_image_patches" triggers, trigger["patch_indices"] must have N
    rows aligned with `imgs01`."""
    imgs01 = imgs01.cpu()
    labels = labels.cpu()

    clean_logits = ca.zero_shot_logits(
        model, ca.normalize(imgs01, model), text_emb, device, batch_size)
    pred = clean_logits.argmax(dim=1)
    cacc = (pred == labels).float().mean().item()
    clean_as_target = (pred == target_cls).float().mean().item()

    trig_imgs = apply_trigger_dict(imgs01, trigger)
    trig_logits = ca.zero_shot_logits(
        model, ca.normalize(trig_imgs, model), text_emb, device, batch_size)
    tpred = trig_logits.argmax(dim=1)
    non_target = labels != target_cls
    asr = ((tpred[non_target] == target_cls).float().mean().item()
           if bool(non_target.any()) else float("nan"))
    asr_incl = (tpred == target_cls).float().mean().item()

    return {
        "asr": asr,
        "asr_incl_target": asr_incl,
        "cacc": cacc,
        "clean_as_target": clean_as_target,
        "n_eval": int(labels.numel()),
    }


ASR_AT_CHECKPOINTS = (10, 20, 40, 60, 80, 100, 150, 200)


def asr_at_k_from_trace(trace, checkpoints=ASR_AT_CHECKPOINTS):
    """ASR@K curve from an in-search quick-ASR trace (the suite convention:
    ASR@K comes from the search-time quick trace, NOT a re-evaluation).

    `trace` = [(cum_flips, asr_fraction), ...] sorted by cum_flips and
    starting at (0, baseline).  Multi-bit atomic element updates make the
    curve a step function: ASR@K = ASR at the last state with
    cum_flips <= K.  Returns {str(K): asr_fraction | None}."""
    pts = sorted(((int(n), float(a)) for n, a in trace))
    out = {}
    for k in checkpoints:
        val = None
        for n, a in pts:
            if n <= k:
                val = a
            else:
                break
        out[str(k)] = round(val, 4) if val is not None else None
    return out


# ═════════════════════════════════════════════════════════════════════════
# full state evaluation
# ═════════════════════════════════════════════════════════════════════════

def apply_param_overrides(model, overrides):
    """Copy the attack's final parameter values into a fresh victim."""
    named = dict(model.named_parameters())
    for name, tensor in overrides.items():
        if name not in named:
            raise KeyError(f"attack_state references unknown parameter {name!r}")
        named[name].data.copy_(tensor.to(named[name].dtype))
    return model


def evaluate_state(state_path, data_split_path=None, out_path=None,
                   device="cuda", batch_size=64, data_root=None):
    """Evaluate one attack_state.pt on the frozen eval split.  Returns the
    metrics dict and (by default) writes `eval.json` next to the state."""
    t0 = time.time()
    state = torch.load(state_path, map_location="cpu")
    model_name = state["model"]
    dataset = state["dataset"]
    target_cls = int(state["target_cls"])

    split_path = data_split_path or DEFAULT_SPLIT
    with open(split_path) as f:
        splits = json.load(f)
    eval_idx = splits[dataset]["eval"]

    model = ca.load_model(model_name, device)
    apply_param_overrides(model, state.get("param_overrides", {}))
    names = ca.load_classnames(dataset)
    text_emb = ca.build_text_embeddings(model, names, device)
    imgs01, labels = ca.load_images(
        dataset, eval_idx, ca.input_px(model_name), data_root)

    res = eval_model(model, text_emb, imgs01, labels, target_cls,
                     trigger=state.get("trigger"), device=device,
                     batch_size=batch_size)
    res.update({
        "method": state["method"],
        "model": model_name,
        "dataset": dataset,
        "seed": int(state.get("seed", -1)),
        "target_cls": target_cls,
        "n_flips": len(state.get("flip_log", [])),
        "eval_time_s": round(time.time() - t0, 1),
    })
    for key in ("asr", "asr_incl_target", "cacc", "clean_as_target"):
        res[key] = round(res[key], 4) if res[key] == res[key] else None

    if out_path is None:
        out_path = os.path.join(os.path.dirname(state_path), "eval.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[evaluate] {state['method']} {model_name} {dataset} "
          f"target={target_cls}: ASR={res['asr']} CACC={res['cacc']} "
          f"({res['n_eval']} imgs) -> {out_path}")
    return res


def main():
    p = argparse.ArgumentParser(description="Shared ASR/CACC evaluator")
    p.add_argument("--state", required=True, help="path to attack_state.pt")
    p.add_argument("--data-split", default=DEFAULT_SPLIT)
    p.add_argument("--out", default=None, help="eval.json path (default: next to state)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--data-root", default=None)
    args = p.parse_args()
    evaluate_state(args.state, args.data_split, args.out,
                   args.device, args.batch, args.data_root)


if __name__ == "__main__":
    main()
