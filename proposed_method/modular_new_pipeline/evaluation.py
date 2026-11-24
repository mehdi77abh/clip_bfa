"""
evaluation.py — ASR / CACC / linear-probe evaluation helpers.
=============================================================
encode_image_batched is shared plumbing (mini-batch encoding);
Evaluator holds the quick/full evaluation routines and the linear-probe
evaluation (lp_evaluate, TASKS.md §2).
"""

import os

import torch
import torch.nn.functional as F

from adapters import encode_image as adapter_encode_image
from trigger import apply_trigger


# ══════════════════════════════════════════════════════════════════════
#  MINI-BATCH IMAGE ENCODING
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def encode_image_batched(model, images, mini_bs=None):
    if mini_bs is None:
        mini_bs = images.size(0)
    parts = []
    for i in range(0, images.size(0), mini_bs):
        parts.append(adapter_encode_image(model, images[i:i + mini_bs]))
    return torch.cat(parts, dim=0)


# ══════════════════════════════════════════════════════════════════════
#  EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════
class Evaluator:
    """Quick (tensor-set), full (loader) and linear-probe evaluation.

    All routines take an ApplyParams record (trigger.py) instead of loose
    size/mode/opacity args, so trigger location and the model's
    normalisation stats travel with the call."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    @torch.no_grad()
    def quick_evaluate(self, model, text_emb, trigger, imgs, labels,
                       target_cls, ap, mini_bs=None):
        if mini_bs is None:
            mini_bs = 4
        if trigger is not None:
            poisoned = apply_trigger(imgs, trigger, ap)
            emb_po = F.normalize(encode_image_batched(model, poisoned, mini_bs), dim=-1)
        else:
            emb_po = None
        emb_cl = F.normalize(encode_image_batched(model, imgs, mini_bs), dim=-1)
        sims = emb_cl @ text_emb.T
        cacc = (sims.argmax(1) == labels).float().mean().item() * 100
        if emb_po is not None:
            asr = ((emb_po @ text_emb.T).argmax(1) == target_cls).float().mean().item() * 100
        else:
            asr = 0.0
        return asr, cacc

    @torch.no_grad()
    def full_evaluate(self, model, text_emb, trigger, loader, target_cls,
                      ap, device):
        hit_ca = tot = 0
        hit_asr = 0
        cos_cl = []
        cos_po = []
        tgt_vec = text_emb[target_cls]

        # Progress is not cosmetic: this loop previously printed nothing, so
        # a stalled DataLoader looked exactly like a slow evaluation and cost
        # two Ctrl-C'd runs to diagnose.  First batch appearing means the
        # loader forked and iterated OK.
        n_batches = max(1, -(-len(loader.dataset) // max(1, loader.batch_size)))
        for bi, (imgs, labels) in enumerate(loader):
            if bi == 0 or (bi + 1) % 10 == 0:
                print(f"    [full_eval] batch {bi + 1}/{n_batches} "
                      f"({tot + imgs.size(0)} images)", flush=True)
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            ec = F.normalize(encode_image_batched(model, imgs), dim=-1)
            hit_ca += ((ec @ text_emb.T).argmax(1) == labels).sum().item()
            tot += imgs.size(0)
            cos_cl.append(F.cosine_similarity(ec, tgt_vec, dim=-1).cpu())
            if trigger is not None:
                poisoned = apply_trigger(imgs, trigger, ap)
                ep = F.normalize(encode_image_batched(model, poisoned), dim=-1)
                hit_asr += ((ep @ text_emb.T).argmax(1) == target_cls).sum().item()
                cos_po.append(F.cosine_similarity(ep, tgt_vec, dim=-1).cpu())
            del ec, imgs, labels
        n = max(tot, 1)
        result = dict(cacc=hit_ca / n * 100,
                      cos_clean=torch.cat(cos_cl).mean().item())
        if trigger is not None:
            result["asr"] = hit_asr / n * 100
            result["cos_poison"] = torch.cat(cos_po).mean().item()
        else:
            result["asr"] = 0.0
            result["cos_poison"] = 0.0
        return result

    # ════════════════════════════════════════════════════════════════
    #  LINEAR-PROBE EVALUATION (TASKS.md §2, RESULTS.md §2)
    # ════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def lp_evaluate(self, model, loader, device, lp_head_path,
                    trigger=None, target_cls=None, ap=None, mini_bs=None,
                    class_names=None):
        """Evaluate the saved linear-probe head on this model's features.

        loader        : DataLoader yielding (images, labels) — used for BOTH
                        clean and triggered passes (the trigger is applied on
                        the fly when ``trigger`` is given).
        lp_head_path  : checkpoint from linear_probe.py
                        ({weight, bias, meta}); the same head is used for
                        every attack method (fair comparison, RESULTS.md §2).
        ap            : ApplyParams for the triggered pass (required with
                        ``trigger``).
        class_names   : the run dataset's prompt class names — when the
                        checkpoint meta carries its own class_names the two
                        label spaces are checked for an exact match (a probe
                        trained for another dataset would silently produce
                        nonsense).

        Returns dict: lp_cacc (clean accuracy, %), lp_asr (fraction of
        triggered features classified as ``target_cls``, %), n, plus
        checkpoint meta (probe top-1 sanity) when available.

        Features follow the linear_probe.py convention: L2-normalised
        ``encode_image_batched`` embeddings.
        """
        if not os.path.isfile(lp_head_path):
            raise FileNotFoundError(
                f"Linear-probe head not found: {lp_head_path} — train it "
                f"first (python linear_probe.py).")
        if trigger is not None and (target_cls is None or ap is None):
            raise ValueError("lp_evaluate with a trigger needs target_cls "
                             "and ApplyParams (ap)")
        if mini_bs is None:
            mini_bs = 4

        head = torch.load(lp_head_path, map_location=device, weights_only=False)
        weight = head["weight"].to(device).float()
        bias = head.get("bias")
        bias = bias.to(device).float() if bias is not None else None
        probe_meta = head.get("meta", {})

        # ── label-space sanity: the head must fit this run's dataset ──
        n_probe = int(weight.shape[0])
        head_names = probe_meta.get("class_names")
        if class_names is not None and head_names is not None \
                and list(head_names) != list(class_names):
            bad = next((i for i, (a, b) in
                        enumerate(zip(head_names, class_names)) if a != b),
                       min(len(head_names), len(class_names)))
            raise ValueError(
                f"Linear-probe head {lp_head_path} was trained on a different "
                f"label space than this run's dataset: first mismatch at index "
                f"{bad} (head {head_names[bad] if bad < len(head_names) else '<end>'!r} "
                f"vs dataset {class_names[bad] if bad < len(class_names) else '<end>'!r}; "
                f"{len(head_names)} vs {len(class_names)} classes).")
        if target_cls is not None and not 0 <= target_cls < n_probe:
            raise ValueError(
                f"target_cls={target_cls} out of range for the probe head "
                f"({n_probe} classes): {lp_head_path}")

        def classify(imgs):
            feats = F.normalize(
                encode_image_batched(model, imgs, mini_bs), dim=-1)
            return F.linear(feats, weight, bias).argmax(dim=-1)

        hit_ca = hit_asr = tot = 0
        clean_as_target = 0
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred_cl = classify(imgs)
            hit_ca += (pred_cl == labels).sum().item()
            if target_cls is not None:
                clean_as_target += (pred_cl == target_cls).sum().item()
            if trigger is not None:
                poisoned = apply_trigger(imgs, trigger, ap)
                pred_po = classify(poisoned)
                hit_asr += (pred_po == target_cls).sum().item()
            tot += imgs.size(0)
            del imgs, labels

        n = max(tot, 1)
        result = dict(
            lp_cacc=hit_ca / n * 100,
            lp_asr=(hit_asr / n * 100) if trigger is not None else 0.0,
            lp_clean_as_target=clean_as_target / n * 100,
            n=n,
            lp_head=lp_head_path,
            probe_test_top1=probe_meta.get("eval", {}).get("test_top1"))
        return result
