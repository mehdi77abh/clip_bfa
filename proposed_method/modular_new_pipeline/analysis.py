"""
analysis.py — Phase 5 post-attack analysis (5a–5d).
===================================================
5a ΔW semantic alignment · 5b per-class CACC breakdown ·
5c ASR/CACC curves · 5d target-class CACC.
"""

import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation import encode_image_batched
from trigger import apply_trigger, get_apply_params
from utils import cleanup_gpu


class PostAttackAnalyzer:
    """Phase 5: post-attack analysis and plots."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    # ════════════════════════════════════════════════════════════════
    #  5a — ΔW SEMANTIC ALIGNMENT
    # ════════════════════════════════════════════════════════════════
    def alignment(self, model, selected_heads, original_weights,
                  t_target, cfg):
        print("\n" + "=" * 60)
        print("PHASE 5a: ΔW SEMANTIC ALIGNMENT")
        print("=" * 60)
        vis = model.visual
        blocks = list(vis.transformer.resblocks)
        proj = vis.proj.data.float()
        n_heads = blocks[0].attn.num_heads
        D = vis.transformer.width
        dk = D // n_heads
        target = t_target.detach().float()
        alignments = []
        for s in selected_heads:
            li = s["layer"]
            blk = blocks[li]
            if s["type"] == "attn":
                hi = s["head"]
                lo, hie = hi * dk, (hi + 1) * dk
                W_in_now = blk.attn.in_proj_weight.data[2 * D + lo:2 * D + hie].float()
                W_out_now = blk.attn.out_proj.weight.data[:, lo:hie].float()
                orig_in = original_weights.get((li, hi, "V"))
                orig_out = original_weights.get((li, hi, "O"))
            elif s["type"] == "ffn":
                W_in_now = blk.mlp.c_fc.weight.data.float()
                W_out_now = blk.mlp.c_proj.weight.data.float()
                orig_in = original_weights.get((li, -1, "fc"))
                orig_out = original_weights.get((li, -1, "proj"))
            else:
                continue
            if orig_in is None or orig_out is None:
                continue
            dIn = W_in_now - orig_in.float()
            dOut = W_out_now - orig_out.float()
            if dIn.abs().max() < 1e-12 and dOut.abs().max() < 1e-12:
                continue
            dVO = dIn.T @ orig_out.float().T + orig_in.float().T @ dOut.T
            if dVO.abs().max() < 1e-12:
                continue
            U, S, Vh = torch.linalg.svd(dVO, full_matrices=False)
            v_delta = Vh[0]
            v_hat = F.normalize((v_delta @ proj).unsqueeze(0), dim=-1).squeeze()
            cos = F.cosine_similarity(v_hat, target, dim=0).item()
            tag = f"L{li} H{s['head']}" if s["type"] == "attn" else f"L{li} FFN"
            alignments.append(dict(layer=li, type=s["type"], cosine=cos))
            print(f"  {tag}: cos(ΔW, target) = {cos:+.4f}")
        if alignments:
            avg = np.mean([a["cosine"] for a in alignments])
            print(f"\n  Mean alignment: {avg:+.4f}")
        else:
            print("  No weight changes detected.")
        return alignments

    # ════════════════════════════════════════════════════════════════
    #  5b — PER-CLASS CACC BREAKDOWN
    # ════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def per_class(self, model, text_emb, dataset, test_idx,
                  target_cls, device, class_names,
                  model_clean_state=None):
        print("\n" + "=" * 60)
        print("PHASE 5b: PER-CLASS CACC BREAKDOWN")
        print("=" * 60)

        class_indices = defaultdict(list)
        for i in test_idx:
            class_indices[dataset[i][1]].append(i)

        def _eval_all():
            results = {}
            for cls_idx in sorted(class_indices.keys()):
                idxs = class_indices[cls_idx]
                loader = DataLoader(Subset(dataset, idxs),
                                    batch_size=4, shuffle=False)
                hit = tot = 0
                for imgs, labels in loader:
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    emb = F.normalize(encode_image_batched(model, imgs), dim=-1)
                    hit += ((emb @ text_emb.T).argmax(1) == labels).sum().item()
                    tot += imgs.size(0)
                acc = hit / max(tot, 1) * 100
                name = class_names[cls_idx] if cls_idx < len(class_names) else "?"
                results[cls_idx] = dict(cls=cls_idx, name=name, acc=acc, n=tot)
            return results

        attacked_results = _eval_all()

        clean_results = {}
        if model_clean_state is not None:
            saved = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(model_clean_state)
            print("  Evaluating clean-model baseline (quantized, no flips)...")
            clean_results = _eval_all()
            model.load_state_dict(saved)
            del saved
            cleanup_gpu()

        rows = []
        for cls_idx in sorted(class_indices.keys()):
            row = dict(attacked_results[cls_idx])
            if cls_idx in clean_results:
                row["clean_acc"] = clean_results[cls_idx]["acc"]
                row["delta"] = row["acc"] - row["clean_acc"]
            else:
                row["clean_acc"] = None
                row["delta"] = None
            rows.append(row)

        rows.sort(key=lambda r: r["acc"])

        if clean_results:
            print(f"\n  {'Class':<30s} {'Clean':>8s} {'Attacked':>9s} "
                  f"{'Δ':>8s}  {'n':>5s}")
            print(f"  {'─' * 65}")
            for r in rows[:20]:
                c_str = f"{r['clean_acc']:.1f}%" if r["clean_acc"] is not None else "—"
                d_str = f"{r['delta']:+.1f}%" if r["delta"] is not None else "—"
                print(f"  {r['name']:<30s} {c_str:>8s} {r['acc']:>8.1f}% "
                      f"{d_str:>8s}  {r['n']:>5d}")

            degraded = [r for r in rows if r["delta"] is not None and r["delta"] < -5.0]
            if degraded:
                degraded.sort(key=lambda r: r["delta"])
                print(f"\n  Classes with >5% degradation (attack vs clean):")
                for r in degraded[:10]:
                    print(f"    {r['name']:<30s} {r['clean_acc']:.1f}% → "
                          f"{r['acc']:.1f}%  (Δ = {r['delta']:+.1f}%)")

            deltas = [r["delta"] for r in rows if r["delta"] is not None]
            if deltas:
                print(f"\n  Mean Δ across all classes: {np.mean(deltas):+.2f}%")
                n_worse = sum(1 for d in deltas if d < -2.0)
                n_better = sum(1 for d in deltas if d > 2.0)
                n_same = len(deltas) - n_worse - n_better
                print(f"  Classes: {n_worse} degraded >2%, "
                      f"{n_same} stable (±2%), "
                      f"{n_better} improved >2%")
        else:
            print(f"\n  {'Class':<30s} {'CACC':>7s}")
            print(f"  {'─' * 40}")
            for r in rows[:20]:
                print(f"  {r['name']:<30s} {r['acc']:>6.1f}%")

        return rows

    # ════════════════════════════════════════════════════════════════
    #  5d — TARGET CLASS CACC
    # ════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def target_class_cacc(self, model, text_emb, trigger,
                          dataset, target_holdout,
                          target_cls, cfg, model_clean_state=None):
        print("\n" + "=" * 60)
        print("PHASE 5d: TARGET CLASS CACC")
        print("=" * 60)

        if len(target_holdout) == 0:
            print("  No target holdout images available.")
            return {}

        ap = get_apply_params(cfg, model)
        loader = DataLoader(Subset(dataset, target_holdout), batch_size=4,
                            shuffle=False, num_workers=2)
        target_name = f"Class {target_cls}"
        results = {}

        if model_clean_state is not None:
            saved = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict(model_clean_state)

            hit_clean = tot = 0
            for imgs, labels in loader:
                imgs = imgs.to(cfg.DEVICE)
                labels = labels.to(cfg.DEVICE)
                emb = F.normalize(encode_image_batched(model, imgs), dim=-1)
                hit_clean += ((emb @ text_emb.T).argmax(1) == labels).sum().item()
                tot += imgs.size(0)
            acc_clean = hit_clean / max(tot, 1) * 100
            print(f"\n  [Clean model — baseline]")
            print(f"    {target_name} clean CACC: {acc_clean:.1f}%  (n={tot})")
            results["clean_model"] = {"clean_cacc": acc_clean, "n": tot}
            model.load_state_dict(saved)

        hit_clean = hit_trig = tot = 0
        for imgs, labels in loader:
            imgs = imgs.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)
            emb_c = F.normalize(encode_image_batched(model, imgs), dim=-1)
            hit_clean += ((emb_c @ text_emb.T).argmax(1) == labels).sum().item()
            poisoned = apply_trigger(imgs, trigger, ap)
            emb_t = F.normalize(encode_image_batched(model, poisoned), dim=-1)
            hit_trig += ((emb_t @ text_emb.T).argmax(1) == labels).sum().item()
            tot += imgs.size(0)

        acc_clean = hit_clean / max(tot, 1) * 100
        acc_trig = hit_trig / max(tot, 1) * 100

        print(f"\n  [Attacked model]")
        print(f"    {target_name} clean CACC:    {acc_clean:.1f}%  (n={tot})")
        print(f"    {target_name} triggered CACC: {acc_trig:.1f}%  (n={tot})")
        results["attacked"] = {"clean_cacc": acc_clean,
                               "triggered_cacc": acc_trig, "n": tot}

        if model_clean_state is not None and "clean_model" in results:
            delta = acc_clean - results["clean_model"]["clean_cacc"]
            print(f"\n    Δ clean CACC (attack - baseline): {delta:+.1f}%")
            if abs(delta) < 2.0:
                print(f"    → Stealthy: target class accuracy preserved")
            else:
                print(f"    → WARNING: target class accuracy degraded!")

        return results

    # ════════════════════════════════════════════════════════════════
    #  5c — ASR / CACC CURVES
    # ════════════════════════════════════════════════════════════════
    def curves(self, checkpoints, path):
        if len(checkpoints) < 2:
            print("  Not enough checkpoints for curve plot.")
            return
        flips = [c[0] for c in checkpoints]
        asr = [c[1] for c in checkpoints]
        cacc = [c[2] for c in checkpoints]
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.set_xlabel("Bit flips")
        ax1.set_ylabel("ASR (%)", color="tab:red")
        ax1.plot(flips, asr, "o-", color="tab:red", label="ASR")
        ax1.tick_params(axis="y", labelcolor="tab:red")
        ax2 = ax1.twinx()
        ax2.set_ylabel("CACC (%)", color="tab:blue")
        ax2.plot(flips, cacc, "s--", color="tab:blue", label="CACC")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_title("ASR / CACC vs Bit Flips — Hybrid Ranking")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved curves → {path}")
