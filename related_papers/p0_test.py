#!/usr/bin/env python3
"""
p0_test.py — P0 two-arm test harness: CNL+m + TDA with (P1) GMAS pool-256
competitors vs (P2) manually curated near-target negatives.

One file, two supported configs (pass --model/--dataset to switch):
  **ViT-L/14 × ImageNet1k** (default) and **ViT-B/16 × Caltech101**.
It reuses the `proposed_method/modular_new_pipeline` pipeline end-to-end (Phases 0-5) and
changes exactly three things, per `related_papers/idea_p0.md` and
`related_papers/idea_p0_tests.md`:

  1. Phase-1 loss   : CNL+m — softmax CE over {target} ∪ M data-derived
                      pseudo-competitors with additive logit margin (A1+A2),
                      and TDA per-sample target-description sampling (B1).
  2. Carrier images : P1 = GMAS-selected 64 from a pool of --n-pool (256);
                      P2/P2cen/P2swap/P2txt/a0 = random 64 (status quo).
  3. Competitor set : P1 = k-means centroids over the pool remainder;
                      P2 = memory bank of 100 curated negative images;
                      P2cen = k-means centroids over the same 100;
                      P2txt = text embeddings of the neighbour class names.

Arms (see idea_p0_tests.md §1):
  a0     plain cos loss + random calib                (reference baseline)
  p1     CNL+m + TDA, GMAS carriers, pool-centroid competitors
  p2     CNL+m + TDA, random carriers, 100-image memory bank   (default)
  p2cen  as p2 but competitors = k-means centroids over the 100
  p2swap as p2 but the swap negative list (list-sensitivity check)
  p2txt  as p2 but competitors = neighbour-class TEXT embeddings

Split layout (shared by every arm — identical grad/eval/test for comparability):
  non-target (shuffled with SEED): grad(N_GRAD) | eval(N_EVAL) | scratch(256)
  | test(rest).  P1: scratch = pool → GMAS picks 64 carriers, rest = mining
  set.  P2*: carriers = first 64 of scratch.  The 100 negative images are
  sampled from the VAL split FIRST and held out of every split (grad/eval/
  test/scratch) in every arm — zero evaluation leakage, no train split
  needed (idea_p0_tests.md §4.4, maintainer decision 2026-11-21).

Caveats (documented, intentional):
  * GMAS selection runs at split time, i.e. on the PRE-quantisation fp32
    forward (Phase 0 has not run yet).  Competitor embeddings are computed
    inside Phase 1, i.e. on the quantised-dequantised forward.
  * The Phase-1 recipe (init 0.1, Adam 0.02 cosine, 200 ep, grad clip,
    clamp ±3, best-loss trigger) is the tuned regime and is NOT changed.
  * Phases 2A-5 are untouched and keep t_bar = text_emb[TARGET_CLS].

Usage (from the repo root or anywhere):
  # ViT-L/14 × ImageNet1k (default config)
  python related_papers/p0_test.py --arm p2txt --target-name lion --neg-list lion_v1 \
      --p1-margin 0 --tda off
  python related_papers/p0_test.py --arm p1
  python related_papers/p0_test.py --arm p2 --neg-list tabby_v1
  python related_papers/p0_test.py --arm a0 --seed 42

  # ViT-B/16 × Caltech101 (defaults: target 'motorbike', neg-list motorbike_v2
  # — the separable variant; --neg-cos-max 0.9 filters near-collinear negatives)
  python related_papers/p0_test.py --arm p2txt --model ViT-B/16 --dataset Caltech101 \
      --p1-margin 0 --tda off
  python related_papers/p0_test.py --arm a0 --model ViT-B/16 --dataset Caltech101

  python related_papers/p0_test.py --arm p2 --dry-run   # config assembly only

RESULTS: everything is written under
  related_papers/p0_results/{section}/{model}/{dataset}/{run_leaf}_{arm_tag}/
(bfa_results.json, trigger.pt, original_weights.pt, plots) with runs.csv /
summary.csv at the top of p0_results — NOT under proposed_method/grid_results.

Any `experiment.py` flag is passed through (e.g. --seed, --section, --out-dir,
--max-flips).  Model and dataset are fixed to ViT-L/14 / ImageNet1k.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))   # .../clip_bfa/related_papers
MODULAR_DIR = os.path.join(REPO_ROOT, "proposed_method", "modular_new_pipeline")
# Results land HERE/p0_results/... — never proposed_method/grid_results.
P0_OUT_ROOT = os.path.join(HERE, "p0_results")
sys.path.insert(0, MODULAR_DIR)

import experiment                                   # noqa: E402
import main as main_mod                             # noqa: E402
import pipeline as pipeline_mod                     # noqa: E402
from adapters import encode_image, get_adapter      # noqa: E402
from config import load_registry                    # noqa: E402
from data import DataModule                         # noqa: E402
from target import PROMPT_TEMPLATES, TargetEmbedder  # noqa: E402
from trigger import (TriggerGenerator, apply_trigger,  # noqa: E402
                      get_apply_params)
from utils import cleanup_gpu, model_tag       # noqa: E402

# Supported configs (maintainer: ViT-L/14 × ImageNet1k AND ViT-B/16 × Caltech101).
SUPPORTED_MODELS = ("ViT-B/16", "ViT-L/14")
SUPPORTED_DATASETS = ("Caltech101", "ImageNet1k")
DEFAULT_MODEL = "ViT-L/14"
DEFAULT_DATASET = "ImageNet1k"
# Dataset-aware defaults: (target-name, neg-list) when the flags are omitted.
DATASET_DEFAULTS = {
    "ImageNet1k": ("tabby cat", "tabby_v1"),
    "Caltech101": ("motorbike", "motorbike_v2"),
}
DEFAULT_TRIG_SZ = 16        # run_all.sh trigger-size policy: 16 @ 224px


# ══════════════════════════════════════════════════════════════════════
#  NEGATIVE CLASS LISTS  (idea_p0_tests.md §4 — verified against
#  class_texts/imagenet1k/classes.txt; counts sum to 100 per list)
# ══════════════════════════════════════════════════════════════════════
NEG_LISTS = {
    # T1 = "tabby cat" — 4 tiers × (class, n_images) = 100
    "tabby_v1": [
        ("feline", "tiger cat", 8), ("feline", "Persian cat", 8),
        ("feline", "Siamese cat", 8), ("feline", "Egyptian Mau", 8),
        ("bigcat", "lynx", 3), ("bigcat", "leopard", 3),
        ("bigcat", "snow leopard", 3), ("bigcat", "jaguar", 3),
        ("bigcat", "cougar", 3), ("bigcat", "cheetah", 3),
        ("bigcat", "lion", 3), ("bigcat", "tiger", 3),
        ("canid", "grey wolf", 2), ("canid", "Alaskan tundra wolf", 2),
        ("canid", "coyote", 2), ("canid", "dingo", 2),
        ("canid", "red fox", 2), ("canid", "kit fox", 2),
        ("canid", "Arctic fox", 2), ("canid", "grey fox", 2),
        ("small", "fox squirrel", 2), ("small", "hamster", 2),
        ("small", "guinea pig", 2), ("small", "marmot", 2),
        ("small", "beaver", 2), ("small", "weasel", 2),
        ("dog", "Chihuahua", 4), ("dog", "Maltese", 4),
        ("dog", "Pomeranian", 4), ("dog", "Toy Poodle", 4),
    ],
    # P2-swap: canid tier → dhole/African wild dog/hyena/red wolf (4×4 = 16),
    # small-mammal tier → skunk/mink/porcupine/European polecat (4×3 = 12).
    "tabby_swap": [
        ("feline", "tiger cat", 8), ("feline", "Persian cat", 8),
        ("feline", "Siamese cat", 8), ("feline", "Egyptian Mau", 8),
        ("bigcat", "lynx", 3), ("bigcat", "leopard", 3),
        ("bigcat", "snow leopard", 3), ("bigcat", "jaguar", 3),
        ("bigcat", "cougar", 3), ("bigcat", "cheetah", 3),
        ("bigcat", "lion", 3), ("bigcat", "tiger", 3),
        ("canid", "dhole", 4), ("canid", "African wild dog", 4),
        ("canid", "hyena", 4), ("canid", "red wolf or maned wolf", 4),
        ("small", "skunk", 3), ("small", "mink", 3),
        ("small", "porcupine", 3), ("small", "European polecat", 3),
        ("dog", "Chihuahua", 4), ("dog", "Maltese", 4),
        ("dog", "Pomeranian", 4), ("dog", "Toy Poodle", 4),
    ],
    # T2 = "sports car" (secondary target; names verified in classes.txt)
    "sportscar_v1": [
        ("core", "convertible", 10), ("core", "limousine", 10),
        ("core", "minivan", 10), ("core", "jeep", 10),
        ("road", "taxicab", 5), ("road", "pickup truck", 5),
        ("road", "tow truck", 5), ("road", "garbage truck", 5),
        ("cargo", "moving van", 5), ("cargo", "station wagon", 5),
        ("cargo", "forklift", 5),
        ("marine", "canoe", 8), ("marine", "catamaran", 8),
        ("marine", "trimaran", 9),
    ],
    # T3 = "lion" (cls 291) — like-for-like vs the stored cos reference
    # (grid_results/.../zs_c291_*: trigger ASR 45.6 %, 7 flips → 99 %).
    # Win bar (maintainer 2026-11-21): standalone ASR 25-35 %.
    "lion_v1": [
        ("bigcat", "tiger", 8), ("bigcat", "leopard", 8),
        ("bigcat", "jaguar", 8), ("bigcat", "snow leopard", 8),
        ("cat", "cheetah", 6), ("cat", "cougar", 6),
        ("cat", "lynx", 6), ("cat", "tabby cat", 6),
        ("cat", "tiger cat", 6),
        ("canid", "hyena", 4), ("canid", "African wild dog", 4),
        ("canid", "grey wolf", 4), ("canid", "dhole", 4),
        ("fox", "red fox", 4), ("fox", "kit fox", 4),
        ("fox", "Arctic fox", 4), ("fox", "grey fox", 4),
        ("misc", "Persian cat", 3), ("misc", "Egyptian Mau", 3),
    ],
    # T4 = "motorbike" (Caltech-101, cls 3) — ViT-B/16 × Caltech101 config.
    # Generic two-wheeler concepts are attacker-side text (NOT Caltech rows)
    # and are skipped by the image holdout; the vehicle/ship tiers are Caltech
    # rows and contribute held-out images.  Names verified against
    # class_texts/caltech101/classes.txt where applicable.
    "motorbike_v1": [
        ("bike", "bicycle", 6), ("bike", "dirt bike", 6),
        ("bike", "motorcycle", 6), ("bike", "mountain bike", 4),
        ("bike", "racing bike", 4), ("bike", "scooter", 4),
        ("bike", "moped", 4), ("bike", "chopper motorcycle", 4),
        ("bike", "racing motorcycle", 4),
        ("vehicle", "side view of a car", 8),
        ("vehicle", "airplane", 8), ("vehicle", "helicopter", 8),
        ("ship", "ferry boat", 6), ("ship", "ketch ship", 6),
        ("ship", "schooner ship", 6),
        ("generic", "motor scooter", 4), ("generic", "touring motorcycle", 4),
        ("generic", "sidecar motorcycle", 4),
        ("generic", "enduro motorcycle", 4),
    ],
    # T4b = "motorbike" — SEPARABLE variant (2026-11-22 finding).  v1
    # contained near-synonyms of the target (motorcycle, dirt bike, moped,
    # …) whose text vectors are ~collinear with t̄ — InfoNCE fights itself
    # (margin_pseudo +0.001, cos_t capped 0.24 vs cos 0.31).  v2 keeps only
    # confusable-but-SEPARABLE directions + the actual eval thief rows
    # ([diag]: airplane 17.2 % of poisoned argmax).  Synonyms are banned;
    # the separability filter (--neg-cos-max) is the automatic guard.
    "motorbike_v2": [
        ("bike", "bicycle", 10), ("bike", "mountain bike", 10),
        ("bike", "racing bike", 10),
        ("vehicle", "airplane", 14), ("vehicle", "side view of a car", 14),
        ("vehicle", "helicopter", 14),
        ("ship", "ferry boat", 10), ("ship", "ketch ship", 9),
        ("ship", "schooner ship", 9),
    ],
}


# ══════════════════════════════════════════════════════════════════════
#  SHARED STATE (captured across the pipeline's phase calls)
# ══════════════════════════════════════════════════════════════════════
class _State:
    model = None            # captured at build_text_embeddings time
    classes = None          # prompt class names (classes.txt order)
    text_emb = None
    val_ds = None
    pool_idx = []
    calib_idx = []
    dcomp_idx = []
    neg_idx = []            # 100 held-out negative images (val indices)
    parity_done = False


STATE = _State()


# ══════════════════════════════════════════════════════════════════════
#  SMALL HELPERS
# ══════════════════════════════════════════════════════════════════════
def _embed_images(model, imgs, mini_bs):
    """Unit-norm image embeddings, mini-batched."""
    out = []
    with torch.no_grad():
        for i in range(0, imgs.size(0), mini_bs):
            out.append(F.normalize(
                encode_image(model, imgs[i:i + mini_bs]), dim=-1))
    return torch.cat(out, dim=0)


def _embed_texts(model, prompts, device, chunk=128):
    """Unit-norm text embeddings via the adapter."""
    with torch.no_grad():
        emb = get_adapter(model).encode_text(model, prompts, device)
    return F.normalize(emb, dim=-1)


def _kmeans(X, M, iters=25, seed=0):
    """k-means (k-means++ init, Lloyd) on unit rows → normalised centroids."""
    N = X.size(0)
    if N < M:
        raise ValueError(f"k-means needs N >= M (N={N}, M={M})")
    rs = np.random.RandomState(seed)
    idx = [int(rs.randint(N))]
    for _ in range(1, M):
        d = torch.cdist(X, X[idx]).min(dim=1).values.clamp_min(1e-12)
        p = (d ** 2).cpu().numpy()
        p /= p.sum()
        idx.append(int(np.random.RandomState(seed + len(idx)).choice(N, p=p)))
    C = X[idx].clone()
    for _ in range(iters):
        assign = torch.cdist(X, C).argmin(dim=1)
        for m in range(M):
            mask = assign == m
            if bool(mask.any()):
                C[m] = X[mask].mean(dim=0)
    return F.normalize(C, dim=-1)


def _filter_separable(names, comps, t_bar, tau):
    """Negative-geometry print + separability filter (2026-11-22 finding).

    InfoNCE negatives must be confusable-but-SEPARABLE: a negative whose
    text vector is nearly collinear with t̄ (e.g. "motorcycle" vs
    "motorbike") makes "push away from n" fight "push toward t̄" — the
    objective fights itself and cos_t caps out.  Drop every negative with
    cos(t̄, n_r) > tau and report the geometry (item 3 of the fix plan).
    """
    cos_n = comps @ t_bar
    keep = (cos_n <= tau).tolist()
    print(f"  Negative geometry  cos(t̄, n_r)   (filter: drop > {tau}):")
    for i in torch.argsort(cos_n, descending=True).tolist():
        mark = "   ← filtered (near-collinear)" if not keep[i] else ""
        print(f"    {names[i]:<28s} {cos_n[i]:+.3f}{mark}")
    kept_names = [n for n, k in zip(names, keep) if k]
    comps = comps[keep]
    if comps.size(0) == 0:
        raise ValueError(
            f"all {len(names)} negatives exceed cos(t̄, n) > {tau} — "
            f"raise --neg-cos-max or curate more separable negatives")
    return comps, kept_names


def _sample_neg_indices(cfg, labels):
    """Seeded per-class sampling of the 100 negative images from VAL.

    Every arm samples (and holds out) the same 100 images — the negative
    arms use them as competitor sources, a0/p1 simply exclude them from
    their splits, so grad/eval/test stay comparable across all arms.
    """
    list_name = cfg.P0_NEG_LIST
    entries = NEG_LISTS[list_name]
    names = STATE.classes

    rng = np.random.RandomState(cfg.SEED + 7)
    neg_idx, used, text_only = [], [], []
    for tier, cls_name, n in entries:
        if cls_name not in names:
            # Attacker-side concept that is not a dataset class (e.g. generic
            # two-wheelers on Caltech): text-only negative — no images to
            # hold out.  Expected behaviour, not an error.
            text_only.append(cls_name)
            continue
        lid = names.index(cls_name)
        cand = np.where(labels == lid)[0]
        take = min(n, len(cand))
        pick = cand.copy()
        rng.shuffle(pick)
        neg_idx.extend(int(i) for i in pick[:take])
        used.append(f"{cls_name}({take})")
    print(f"  Negatives [{list_name}]: {len(neg_idx)} val images from "
          f"{len(used)} classes — held out of all splits")
    if text_only:
        print(f"    text-only ({len(text_only)}): " + ", ".join(text_only))
    print(f"    tiers: " + ", ".join(sorted({t for t, _, _ in entries})))
    return neg_idx


# ══════════════════════════════════════════════════════════════════════
#  PATCHED DATA MODULE — pool/GMAS split + negative sampling
# ══════════════════════════════════════════════════════════════════════
class P0DataModule(DataModule):
    """DataModule with the P0 shared split layout + arm-specific carriers."""

    def load_dataset(self, preprocess):
        ds, names = super().load_dataset(preprocess)
        STATE.val_ds = ds
        STATE.classes = list(names)
        return ds, names

    def prepare_data_splits(self, ds):
        cfg = self.cfg
        rng = np.random.RandomState(cfg.SEED)
        labels = self.get_all_labels(ds)

        target_indices = [i for i, lbl in enumerate(labels)
                          if lbl == cfg.TARGET_CLS]
        other_indices = [i for i, lbl in enumerate(labels)
                         if lbl != cfg.TARGET_CLS]

        # ── P0: sample the 100 negatives from VAL first and hold them out
        #    of EVERY split, in every arm (zero leakage; a0/p1 exclude the
        #    same 100 so grad/eval/test stay identical across arms) ──
        STATE.neg_idx = _sample_neg_indices(cfg, labels)
        neg_set = set(STATE.neg_idx)
        other_indices = [i for i in other_indices if i not in neg_set]

        rng.shuffle(target_indices)
        rng.shuffle(other_indices)

        target_sel = target_indices[:cfg.N_TARGET_IMAGES]
        target_holdout = target_indices[cfg.N_TARGET_IMAGES:]

        # ── shared layout: grad | eval | scratch(max(N_POOL, N_CALIB)) | test
        #    (identical grad/eval/test across ALL arms — comparability first)
        ptr = 0
        grad_idx = other_indices[ptr:ptr + cfg.N_GRAD]
        ptr += cfg.N_GRAD
        eval_idx = other_indices[ptr:ptr + cfg.N_EVAL]
        ptr += cfg.N_EVAL
        scratch_n = max(cfg.N_POOL, cfg.N_CALIB)
        scratch = other_indices[ptr:ptr + scratch_n]
        ptr += scratch_n
        test_idx = other_indices[ptr:]

        arm = cfg.P0_ARM
        if arm == "p1":
            pool = scratch[:cfg.N_POOL]
            calib_idx, dcomp_idx = self._gmas_split(pool, cfg)
            STATE.pool_idx = pool
        else:
            calib_idx = scratch[:cfg.N_CALIB]
            dcomp_idx = []
            STATE.pool_idx = []
        STATE.calib_idx = calib_idx
        STATE.dcomp_idx = dcomp_idx

        print(f"  Target class images : {len(target_sel):>5d}")
        print(f"  Target holdout      : {len(target_holdout):>5d}")
        print(f"  Calibration         : {len(calib_idx):>5d}"
              + ("  (GMAS-selected)" if arm == "p1" else "  (random)"))
        print(f"  Pool / mining       : {len(STATE.pool_idx):>5d} / "
              f"{len(dcomp_idx):>5d}")
        print(f"  Negatives (val)     : {len(STATE.neg_idx):>5d}  "
              f"(held out of all splits)")
        print(f"  Gradient            : {len(grad_idx):>5d}")
        print(f"  Evaluation          : {len(eval_idx):>5d}")
        print(f"  Test                : {len(test_idx):>5d}")
        return target_sel, calib_idx, grad_idx, eval_idx, test_idx, \
            target_holdout

    # ── GMAS (idea_p0.md §3 Phase 0.5b) ──────────────────────────────
    def _gmas_split(self, pool, cfg):
        """Greedy mean-aligned selection of K_c carriers from the pool.

        Objective: S* = argmin_{|S|=K_c} ‖ normalize(mean_{x∈S} e(x)) − t̄ ‖₂
        Greedy: add the candidate that maximises cos of the running normalised
        mean with t̄ (equivalent to the L2 objective on the unit sphere).
        NOTE: runs at split time → embeddings use the PRE-quantisation fp32
        forward (Phase 0 has not run yet).  Documented caveat.
        """
        model = STATE.model
        dev = cfg.DEVICE
        imgs, _ = self.load_tensors(STATE.val_ds, pool, dev,
                                    batch_size=64, workers=cfg.WORKERS)
        E = _embed_images(model, imgs, cfg.MINI_BS)          # (P, d)
        del imgs
        cleanup_gpu()

        t_bar = F.normalize(STATE.text_emb[cfg.TARGET_CLS].to(dev), dim=0)
        Kc = cfg.N_CALIB
        running = torch.zeros_like(t_bar)
        selected, remaining = [], list(range(E.size(0)))
        for s in range(1, Kc + 1):
            sums = running.unsqueeze(0) + E[remaining]        # (R, d)
            cos = F.cosine_similarity(sums, t_bar.unsqueeze(0), dim=-1)
            pick = int(torch.argmax(cos).item())
            chosen = remaining.pop(pick)
            selected.append(chosen)
            running = running + E[chosen]
        print(f"  GMAS: selected {len(selected)} carriers from pool of "
              f"{len(pool)} (cos(ν(S), t̄) = "
              f"{F.cosine_similarity(running, t_bar, dim=0).item():+.4f})")
        return [pool[i] for i in selected], [pool[i] for i in remaining]


# ══════════════════════════════════════════════════════════════════════
#  PATCHED TARGET EMBEDDER — capture model/classes for later phases
# ══════════════════════════════════════════════════════════════════════
class P0TargetEmbedder(TargetEmbedder):

    def build_text_embeddings(self, model, classes, device):
        text_emb = super().build_text_embeddings(model, classes, device)
        STATE.model = model
        STATE.classes = list(classes)
        STATE.text_emb = text_emb
        return text_emb


# ══════════════════════════════════════════════════════════════════════
#  PATCHED TRIGGER GENERATOR — CNL+m + TDA (idea_p0.md §3 Phase 1)
# ══════════════════════════════════════════════════════════════════════
class P0TriggerGenerator(TriggerGenerator):
    """Phase 1 with the CNL+m objective and TDA target sampling.

    L_i = −(⟨z_i, t_{j(i)}⟩/T + m)
          + log( exp(⟨z_i, t_{j(i)}⟩/T + m) + Σ_r exp(⟨z_i, μ_r⟩/T) )
    z_i = e(A(x_i; δ)),  j(i) ~ U{1..K} (TDA 'sample'),  μ_r = competitors.

    TDA modes: sample | marginalise (average loss over all K templates)
    | off (single mean target t̄ — old behaviour plus CNL only).
    """

    def generate(self, model, text_emb, t_target, calib_loader,
                 target_images, cfg, neg_text_emb=None):
        # neg_text_emb: the updated modular pipeline.py (R1 integration)
        # passes its InfoNCE competitor embedding here on every attempt.
        # p0_test arms own their competitor source (_build_competitors /
        # NEG_LISTS), so the kwarg is only forwarded to the reference cos
        # path — kept for signature compatibility with the pipeline call.
        if cfg.P1_LOSS != "cnl":
            trig = super().generate(model, text_emb, t_target, calib_loader,
                                    target_images, cfg,
                                    neg_text_emb=neg_text_emb)
        else:
            trig = self._generate_cnl(model, text_emb, calib_loader, cfg)
        self._diag_top_classes(model, trig, calib_loader, text_emb, cfg)
        return trig

    def _diag_top_classes(self, model, trig, calib_loader, text_emb, cfg):
        """Who wins the 1000-way argmax on poisoned images? (diagnostic)

        Answers 'which class row steals the decision from the target' —
        the key question when margin_pseudo > 0 but margin_class < 0.
        """
        from collections import Counter
        dev, mm = cfg.DEVICE, cfg.MINI_BS
        ap = get_apply_params(cfg, model)
        text_all = text_emb.to(dev)
        t_bar = F.normalize(text_emb[cfg.TARGET_CLS].to(dev), dim=0)
        preds, cos_all = [], []
        with torch.no_grad():
            for imgs, _ in calib_loader:
                imgs = imgs.to(dev)
                poisoned = apply_trigger(imgs, trig, ap)
                emb = F.normalize(encode_image(model, poisoned), dim=-1)
                preds.append((emb @ text_all.T).argmax(dim=1))
                cos_all.append(emb @ t_bar)
        preds = torch.cat(preds).cpu().tolist()
        n = len(preds)
        cnt = Counter(preds)
        print("  [diag] poisoned argmax over 1000 classes (calib imgs):")
        for cls_idx, c in cnt.most_common(5):
            mark = "   ← target" if cls_idx == cfg.TARGET_CLS else ""
            print(f"    {STATE.classes[cls_idx]:<30s} {100.0 * c / n:5.1f}%{mark}")
        hit = cnt.get(cfg.TARGET_CLS, 0)
        print(f"    target share {100.0 * hit / n:.1f}%   "
              f"mean cos_t = {torch.cat(cos_all).mean().item():+.4f}")

    # ── competitor bank ──────────────────────────────────────────────
    def _build_competitors(self, model, text_emb, cfg):
        dev, mm = cfg.DEVICE, cfg.MINI_BS
        mode = cfg.COMPETITOR_MODE
        if mode == "text":
            names = []
            list_name = "tabby_swap" if cfg.P0_ARM == "p2swap" else cfg.NEG_LIST
            for _, c, _ in NEG_LISTS[list_name]:
                if c not in names:
                    names.append(c)
            prompts = []
            for c in names:
                name = c.replace("_", " ").lower()
                prompts.extend(t.format(name) for t in PROMPT_TEMPLATES)
            comps = _embed_texts(model, prompts, dev)
            # template-ensemble mean per class, renormalised
            comps = comps.view(len(names), len(PROMPT_TEMPLATES), -1)
            comps = F.normalize(comps.mean(dim=1), dim=-1)
            # separability filter + geometry print
            t_bar = F.normalize(text_emb[cfg.TARGET_CLS].to(dev), dim=0)
            comps, kept = _filter_separable(
                names, comps, t_bar,
                getattr(cfg, "NEG_COS_MAX", 0.9))
            print(f"  Competitors [text]  : {comps.size(0)} separable class "
                  f"directions (of {len(names)} curated)")
            return comps

        if cfg.P0_ARM == "p1":
            if not STATE.dcomp_idx:
                raise ValueError("p1 has no pool mining images — check the "
                                 "split layout")
            imgs, _ = DataModule.load_tensors(STATE.val_ds, STATE.dcomp_idx,
                                              dev, batch_size=64,
                                              workers=cfg.WORKERS)
            src = f"pool mining set ({len(STATE.dcomp_idx)} imgs)"
        else:
            if not STATE.neg_idx:
                raise ValueError(
                    "this arm needs negative IMAGES but none of the neg-list "
                    "names are dataset classes — use --arm p2txt (text "
                    "negatives) or a neg list whose names exist in the "
                    "dataset")
            imgs, _ = DataModule.load_tensors(STATE.val_ds, STATE.neg_idx,
                                              dev, batch_size=64,
                                              workers=cfg.WORKERS)
            src = (f"negatives ({len(STATE.neg_idx)} val imgs, held out "
                   f"of all splits)")
        E = _embed_images(model, imgs, mm)
        del imgs
        cleanup_gpu()

        if mode == "bank":
            comps = E
            print(f"  Competitors [bank]  : {comps.size(0)} embeddings  ← {src}")
        else:  # centroids (pool for p1 / negatives for p2cen)
            M = min(cfg.COMPETITOR_M, E.size(0))
            comps = _kmeans(E, M, iters=25, seed=cfg.SEED)
            print(f"  Competitors [kmeans]: M={comps.size(0)} centroids  "
                  f"← {src}")
        return comps

    # ── gradient-scale parity diagnostic (study protocol) ────────────
    def _parity_check(self, emb0, t_bar, t_set, comps, cfg):
        T, m = cfg.P1_LOGIT_T, cfg.P1_MARGIN
        B = emb0.size(0)
        tgt = torch.zeros(B, dtype=torch.long, device=emb0.device)

        z1 = emb0.detach().clone().requires_grad_(True)
        l1 = -(z1 * t_bar).sum(-1) / cfg.TEMPERATURE
        l1.sum().backward()
        g1 = z1.grad.norm(dim=-1).mean().item()

        z2 = emb0.detach().clone().requires_grad_(True)
        comp_logits = z2 @ comps.T / T
        j = torch.zeros(B, dtype=torch.long, device=emb0.device)  # fixed t_0
        tl = (z2 * t_set[j]).sum(-1) / T + m
        l2 = F.cross_entropy(
            torch.cat([tl[:, None], comp_logits], dim=1), tgt, reduction="none")
        l2.sum().backward()
        g2 = z2.grad.norm(dim=-1).mean().item()
        print(f"  [parity] ‖∂L/∂emb‖  cos/T = {g1:.3f}   CNL+m = {g2:.3f}")

    # ── Phase-1 optimisation (recipe unchanged) ──────────────────────
    def _generate_cnl(self, model, text_emb, calib_loader, cfg):
        from adapters import encode_image  # local, same as reference
        ap = get_apply_params(cfg, model)
        dev = cfg.DEVICE
        T, m = cfg.P1_LOGIT_T, cfg.P1_MARGIN

        print("\n" + "=" * 60)
        print(f"PHASE 1: TRIGGER GENERATION — CNL+m + TDA  (arm {cfg.P0_ARM})")
        print("=" * 60)
        print(f"  Size: {cfg.TRIG_SZ}×{cfg.TRIG_SZ}  Placement: {ap.loc or 'legacy'}"
              f"  Mode: {ap.mode}")
        print(f"  T: {T}  margin m: {m}  TDA: {cfg.TDA_MODE}")
        print(f"  Epochs: {cfg.P1_EPOCHS}  LR: {cfg.P1_LR}  "
              f"grad clip: {cfg.P1_GRAD_CLIP}  Trig clamp: [-3, 3]")
        print(f"  Init: randn × 0.1")

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # ── target description set {t_k} + mean t̄ ──
        cname = STATE.classes[cfg.TARGET_CLS].replace("_", " ").lower()
        prompts = [t.format(cname) for t in PROMPT_TEMPLATES]
        t_set = _embed_texts(model, prompts, dev)             # (K, d)
        t_bar = F.normalize(text_emb[cfg.TARGET_CLS].to(dev), dim=0)
        K = t_set.size(0)
        print(f"  Target: '{STATE.classes[cfg.TARGET_CLS]}' — {K} descriptions")

        comps = self._build_competitors(model, text_emb, cfg)  # (M, d)
        M = comps.size(0)

        trig = nn.Parameter(torch.randn(1, 3, cfg.TRIG_SZ, cfg.TRIG_SZ,
                                        device=dev) * 0.1)
        opt = torch.optim.Adam([trig], lr=cfg.P1_LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.P1_EPOCHS, eta_min=cfg.P1_LR * 0.01)

        best_loss, best_trig = float("inf"), trig.data.clone()
        text_all = text_emb.to(dev)
        tc = cfg.TARGET_CLS

        for ep in range(1, cfg.P1_EPOCHS + 1):
            ep_loss = ep_cos = ep_mp = ep_mc = 0.0
            nb = 0
            for imgs, _ in calib_loader:
                imgs = imgs.to(dev)
                poisoned = apply_trigger(imgs, trig, ap)
                emb = F.normalize(encode_image(model, poisoned), dim=-1)
                B = emb.size(0)

                comp_logits = emb @ comps.T / T                # (B, M)
                if cfg.TDA_MODE == "sample":
                    j = torch.randint(0, K, (B,), device=dev)
                    tl = (emb * t_set[j]).sum(-1) / T + m
                    logits = torch.cat([tl[:, None], comp_logits], dim=1)
                    loss = F.cross_entropy(
                        logits, torch.zeros(B, dtype=torch.long, device=dev))
                elif cfg.TDA_MODE == "marginalise":
                    loss = 0.0
                    for k in range(K):
                        tl = (emb * t_set[k]).sum(-1) / T + m
                        logits = torch.cat([tl[:, None], comp_logits], dim=1)
                        loss = loss + F.cross_entropy(
                            logits,
                            torch.zeros(B, dtype=torch.long, device=dev))
                    loss = loss / K
                else:  # "off": single mean target, CNL competitors only
                    tl = emb @ t_bar / T + m
                    logits = torch.cat([tl[:, None], comp_logits], dim=1)
                    loss = F.cross_entropy(
                        logits, torch.zeros(B, dtype=torch.long, device=dev))

                opt.zero_grad()
                loss.backward()
                if cfg.P1_GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_([trig],
                                                    max_norm=cfg.P1_GRAD_CLIP)
                opt.step()
                with torch.no_grad():
                    trig.data.clamp_(-3.0, 3.0)

                # ── diagnostics (not optimised) ──
                with torch.no_grad():
                    cos_t = emb @ t_bar
                    mp = cos_t - comp_logits.max(dim=1).values * T
                    sims = emb @ text_all.T
                    sims[:, tc] = -float("inf")
                    mc = cos_t - sims.max(dim=1).values
                ep_loss += loss.item()
                ep_cos += cos_t.mean().item()
                ep_mp += mp.mean().item()
                ep_mc += mc.mean().item()
                nb += 1

                if not STATE.parity_done:
                    self._parity_check(emb.detach(), t_bar, t_set, comps, cfg)
                    STATE.parity_done = True

            sched.step()
            avg_l = ep_loss / max(nb, 1)
            if avg_l < best_loss:
                best_loss, best_trig = avg_l, trig.data.clone()
            if ep % 50 == 0 or ep == 1 or ep == cfg.P1_EPOCHS:
                print(f"  ep {ep:>4d}  loss={avg_l:.6f}  "
                      f"cos_t={ep_cos / max(nb, 1):+.4f}  "
                      f"margin_pseudo={ep_mp / max(nb, 1):+.4f}  "
                      f"margin_class={ep_mc / max(nb, 1):+.4f}  "
                      f"||trig||={trig.data.norm():.3f}")

        print(f"  Best loss: {best_loss:.6f}")
        return best_trig.detach()


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════
def parse_p0_args(argv):
    p = argparse.ArgumentParser(
        description="P0 two-arm test (CNL+m + TDA): pool-GMAS vs manual "
                    "negatives.  experiment.py flags are passed through.")
    p.add_argument("--arm", required=True,
                   choices=["a0", "p1", "p2", "p2cen", "p2swap", "p2txt"],
                   help="test arm (see module docstring)")
    p.add_argument("--neg-list", default=None, choices=sorted(NEG_LISTS),
                   help="curated confusable negative list (default: "
                        "dataset-aware — lion_v1/tabby_v1 on ImageNet1k, "
                        "motorbike_v1 on Caltech101; p2swap always uses "
                        "tabby_swap)")
    p.add_argument("--target-name", default=None,
                   help="target class name resolved against the dataset's "
                        "classes.txt (default: dataset-aware — 'tabby cat' "
                        "on ImageNet1k, 'motorbike' on Caltech101)")
    p.add_argument("--p1-margin", type=float, default=1.0,
                   help="additive target-logit margin m in logit units "
                        "(1.0 ↔ 0.07 cosine; default: 1.0)")
    p.add_argument("--p1-temperature", type=float, default=0.07,
                   help="softmax temperature T for CNL (default: 0.07)")
    p.add_argument("--tda", default="sample",
                   choices=["sample", "marginalise", "off"],
                   help="TDA target-description mode (default: sample)")
    p.add_argument("--n-pool", type=int, default=256,
                   help="P1 GMAS pool size (default: 256)")
    p.add_argument("--competitor-m", type=int, default=16,
                   help="k-means centroid count for p1/p2cen (default: 16)")
    p.add_argument("--neg-cos-max", type=float, default=0.9,
                   help="separability filter: drop text negatives with "
                        "cos(t̄, n) above this threshold (default: 0.9)")
    p.add_argument("--dry-run", action="store_true",
                   help="assemble and print the config, then exit")
    return p.parse_known_args(argv)


def resolve_target_cls(dataset, name):
    reg = load_registry()
    ctd = reg["datasets"][dataset].get("class_text_dir",
                                       f"class_texts/{dataset.lower()}")
    path = os.path.join(MODULAR_DIR, ctd, "classes.txt")
    names = [ln.strip() for ln in open(path) if ln.strip()]
    if name not in names:
        near = [n for n in names if name.lower() in n.lower()][:5]
        raise SystemExit(f"target {name!r} not in {ctd}/classes.txt.  "
                         f"Did you mean: {near}?")
    return names.index(name)


def arm_tag(p0, cfg):
    t = f"p0{p0.arm}"
    if p0.arm == "p1":
        t += f"_pool{p0.n_pool}"
    if p0.arm in ("p2", "p2cen", "p2txt"):
        t += f"_{p0.neg_list}"
    if p0.arm == "p2swap":
        t += "_tabby_swap"
    if p0.arm == "p2cen":
        t += f"_cenM{p0.competitor_m}"
    if p0.arm == "p2txt":
        t += "_txt"
    if p0.arm != "a0":
        t += f"_m{p0.p1_margin:g}_tda{p0.tda}"
    return t


def main():
    p0, rest = parse_p0_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    args = experiment.parse_args()

    # ── supported configs: --model/--dataset pass through (validated);
    #    defaults keep the historical ViT-L/14 × ImageNet1k behaviour ──
    if "--model" in rest:
        if args.model not in SUPPORTED_MODELS:
            raise SystemExit(f"--model {args.model!r} not supported here — "
                             f"use one of {SUPPORTED_MODELS}")
    else:
        args.model = DEFAULT_MODEL
    if "--dataset" in rest:
        if args.dataset not in SUPPORTED_DATASETS:
            raise SystemExit(f"--dataset {args.dataset!r} not supported here "
                             f"— use one of {SUPPORTED_DATASETS}")
    else:
        args.dataset = DEFAULT_DATASET
    if "--trigger-size" not in rest:
        args.trigger_size = DEFAULT_TRIG_SZ
    # dataset-aware defaults for the target and the negative list
    d_target, d_neg = DATASET_DEFAULTS[args.dataset]
    if p0.target_name is None:
        p0.target_name = d_target
    if p0.neg_list is None:
        p0.neg_list = d_neg
    args.target_cls = resolve_target_cls(args.dataset, p0.target_name)

    cfg = experiment.build_config(args)

    # ── P0 settings on cfg ──
    cfg.P0_ARM = p0.arm
    cfg.P0_NEG_LIST = ("tabby_swap" if p0.arm == "p2swap" else p0.neg_list)
    cfg.NEG_LIST = cfg.P0_NEG_LIST
    cfg.N_POOL = p0.n_pool
    cfg.COMPETITOR_M = p0.competitor_m
    cfg.NEG_COS_MAX = p0.neg_cos_max
    cfg.P1_LOGIT_T = p0.p1_temperature
    cfg.TDA_MODE = "off" if p0.arm == "a0" else p0.tda
    cfg.P1_MARGIN = 0.0 if p0.arm == "a0" else p0.p1_margin
    cfg.P1_LOSS = "cos" if p0.arm == "a0" else "cnl"
    cfg.COMPETITOR_MODE = {
        "p1": "centroids_pool", "p2": "bank", "p2cen": "centroids_neg",
        "p2swap": "bank", "p2txt": "text", "a0": None,
    }[p0.arm]

    # ── distinct run dir per arm (never collide) ──
    tag = arm_tag(p0, cfg)
    cfg.RUN_ID += f"__{tag}"
    cfg.OUT_DIR = cfg.OUT_DIR + f"_{tag}"

    # ── results go to related_papers/p0_results — never grid_results ──
    if os.path.abspath(cfg.OUT_DIR).startswith(os.path.abspath(cfg.OUT_ROOT)):
        rel = os.path.relpath(cfg.OUT_DIR, cfg.OUT_ROOT)
    else:  # explicit --out-dir: keep only the run leaf
        rel = os.path.join(cfg.SECTION, model_tag(cfg.MODEL_NAME),
                           cfg.DATASET, os.path.basename(cfg.OUT_DIR))
    cfg.OUT_ROOT = P0_OUT_ROOT
    cfg.OUT_DIR = os.path.join(cfg.OUT_ROOT, rel)

    print("=" * 60)
    print(f"P0 TEST — arm {p0.arm}  ({tag})")
    print("=" * 60)
    print(f"  model={cfg.MODEL}  dataset={cfg.DATASET}  "
          f"target='{p0.target_name}' (cls {cfg.TARGET_CLS})")
    print(f"  P1 loss={cfg.P1_LOSS}  TDA={cfg.TDA_MODE}  "
          f"margin={cfg.P1_MARGIN:g}  T={cfg.P1_LOGIT_T}")
    print(f"  carriers: {'GMAS from pool ' + str(p0.n_pool) if p0.arm == 'p1' else 'random ' + str(cfg.N_CALIB)}")
    print(f"  competitors: {cfg.COMPETITOR_MODE}"
          + (f"  M={p0.competitor_m}" if cfg.COMPETITOR_MODE and
             'centroids' in str(cfg.COMPETITOR_MODE) else ""))
    print(f"  out: {cfg.OUT_DIR}")

    if p0.dry_run:
        print("\n[dry-run] config assembled OK — exiting before model load.")
        return

    # ── install patches, then run the standard pipeline (Phases 0-5) ──
    pipeline_mod.DataModule = P0DataModule
    pipeline_mod.TargetEmbedder = P0TargetEmbedder
    pipeline_mod.TriggerGenerator = P0TriggerGenerator

    result = main_mod.main(cfg)          # OOM/error contract of main.py applies
    try:
        experiment.refresh_summaries(cfg.OUT_ROOT)
    except Exception as e:               # best-effort, never fail the run
        print(f"  (summary refresh skipped: {type(e).__name__}: {e})")
    return result


if __name__ == "__main__":
    main()
