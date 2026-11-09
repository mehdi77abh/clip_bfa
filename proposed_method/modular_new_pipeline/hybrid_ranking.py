"""
hybrid_ranking.py — Phase 3 hybrid bit ranking (+ gradient-only fallback).
=========================================================================
Scoring: final_score = R_l × |g·Δw| on the Phase-0 int8 grid, all 8 bits
per element, loss-reducing directions only.  The gradient term is
poison−clean specific; swap ``rank`` to change the scoring algorithm.
"""

import torch
import torch.nn.functional as F

from quantizer import flip_bit_int8
from trigger import apply_trigger, get_apply_params
from utils import cleanup_gpu


class HybridRanker:
    """Phase 3: candidate bit-flip ranking."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def rank(self, model, trigger, t_target, layer_R,
             selected_heads, tensor_refs,
             grad_imgs, cfg, sub_quant):
        print("\n" + "=" * 60)
        print("PHASE 3: HYBRID BIT RANKING")
        print("=" * 60)
        print("  Scoring: R_l × |g·Δw|  (basic, all 8 bits per element)")
        print("  Int8 grid: Phase-0 global  (no local re-quantisation)")

        for p in model.parameters():
            p.requires_grad_(False)

        apply_mode, apply_opacity = get_apply_params(cfg)

        selected_tensors = set()
        for key, (tensor, _, _) in tensor_refs.items():
            tensor.requires_grad_(True)
            selected_tensors.add(tensor)

        assert torch.is_grad_enabled()

        mini_bs = cfg.MINI_BS
        n_imgs = grad_imgs.size(0)

        # ── Gradient on TRIGGERED (accumulate) ──
        model.zero_grad()
        total_loss_po = 0
        nb = 0
        for i in range(0, n_imgs, mini_bs):
            chunk = grad_imgs[i:i + mini_bs]
            poisoned = apply_trigger(chunk, trigger, cfg.TRIG_SZ,
                                     apply_mode, apply_opacity)
            emb_po = F.normalize(model.encode_image(poisoned), dim=-1)
            loss_po = -F.cosine_similarity(
                emb_po,
                t_target.detach().unsqueeze(0).expand(emb_po.size(0), -1),
                dim=-1).mean()
            loss_po.backward()
            total_loss_po += loss_po.item()
            nb += 1
            del chunk, poisoned, emb_po, loss_po

        print(f"  L_target(poisoned) = {total_loss_po / max(nb, 1):.4f}")

        grad_poisoned = {}
        for key, (tensor, rslice, cslice) in tensor_refs.items():
            grad_poisoned[key] = tensor.grad[rslice, cslice].clone()

        if getattr(cfg, "POISON_ONLY_GRAD", False):
            # variant 'c': rank on the poison gradient only — skip the clean
            # pass and the poison−clean difference
            print("  Variant 'c': poison-only gradient (no poison−clean difference)")
            grad_specific = dict(grad_poisoned)
        else:
            # ── Gradient on CLEAN (accumulate) ──
            for tensor in selected_tensors:
                tensor.grad = None

            total_loss_cl = 0
            nb = 0
            for i in range(0, n_imgs, mini_bs):
                chunk = grad_imgs[i:i + mini_bs]
                emb_cl = F.normalize(model.encode_image(chunk), dim=-1)
                loss_cl = -F.cosine_similarity(
                    emb_cl,
                    t_target.detach().unsqueeze(0).expand(emb_cl.size(0), -1),
                    dim=-1).mean()
                loss_cl.backward()
                total_loss_cl += loss_cl.item()
                nb += 1
                del chunk, emb_cl, loss_cl

            print(f"  L_target(clean)    = {total_loss_cl / max(nb, 1):.4f}")

            grad_clean = {}
            for key, (tensor, rslice, cslice) in tensor_refs.items():
                grad_clean[key] = tensor.grad[rslice, cslice].clone()

            grad_specific = {}
            for key in grad_poisoned:
                grad_specific[key] = grad_poisoned[key] - grad_clean[key]

        for tensor in selected_tensors:
            tensor.requires_grad_(False)
            tensor.grad = None

        # ── Rank candidates: ALL loss-reducing bits per element ──
        ranked = []
        for key, (tensor, rslice, cslice) in tensor_refs.items():
            li, hi, tn = key
            g_slice = grad_specific[key]
            R_l = layer_R.get(li, 1.0)

            # Phase-0 global int8 grid (NOT local re-quantisation)
            sq = sub_quant[(li, hi, tn)]
            int8_vals = sq["int8"]
            scale = sq["scale"]
            flat_grad = g_slice.flatten()
            flat_int8 = int8_vals.flatten()
            flat_sens = flat_grad.abs()

            k = min(cfg.SENS_TOP_M, flat_sens.numel())
            _, topk = flat_sens.topk(k)

            for idx in topk.tolist():
                q, p = divmod(idx, g_slice.shape[1])
                g = flat_grad[idx].item()

                # Skip non-loss-reducing directions
                if g >= 0:
                    continue

                old_i8 = int(flat_int8[idx].item())
                for bit in range(8):
                    new_i8 = flip_bit_int8(old_i8, bit)
                    dw = (new_i8 - old_i8) * scale
                    dpred = g * dw
                    if dpred >= 0:
                        continue
                    hybrid_score = R_l * abs(dpred)
                    ranked.append(dict(
                        tensor=tn, layer=li, head=hi,
                        component="attn" if hi >= 0 else "ffn",
                        row=q, col=p, bit=bit,
                        old_int8=old_i8, new_int8=new_i8,
                        delta_w=dw, predicted_dL=dpred,
                        layer_weight_R=R_l,
                        gradient_score=abs(dpred),
                        final_score=hybrid_score))

        ranked.sort(key=lambda d: -d["final_score"])

        n_attn = sum(1 for r in ranked if r["component"] == "attn")
        n_ffn = sum(1 for r in ranked if r["component"] == "ffn")
        print(f"  Candidates: {len(ranked)} ({n_attn} attn, {n_ffn} ffn)")
        if ranked:
            print(f"  Top score: {ranked[0]['final_score']:.8f} "
                  f"(R={ranked[0]['layer_weight_R']:.3f}, "
                  f"g={ranked[0]['gradient_score']:.8f})")
            print(f"\n  Top-10 candidates:")
            for r in ranked[:10]:
                tag = (f"L{r['layer']} H{r['head']}" if r["component"] == "attn"
                       else f"L{r['layer']} FFN")
                print(f"    {tag} {r['tensor']}[{r['row']},{r['col']}] "
                      f"bit{r['bit']}  R={r['layer_weight_R']:.3f}  "
                      f"|Δw|={abs(r['delta_w']):.6f}  "
                      f"score={r['final_score']:.8f}")

        cleanup_gpu()
        return ranked

    def rank_fallback(self, model, trigger, t_target, selected_heads,
                      grad_imgs, cfg, sub_quant):
        print("\n" + "=" * 60)
        print("PHASE 3 (FALLBACK): GRADIENT-ONLY RANKING")
        print("=" * 60)

        for p in model.parameters():
            p.requires_grad_(False)

        blocks = list(model.visual.transformer.resblocks)
        D = model.visual.transformer.width
        n_heads = blocks[0].attn.num_heads
        dk = D // n_heads
        target_vec = t_target.detach()
        apply_mode, apply_opacity = get_apply_params(cfg)
        mini_bs = cfg.MINI_BS
        n_imgs = grad_imgs.size(0)

        tracked_params = {}
        for s in selected_heads:
            li = s["layer"]
            blk = blocks[li]
            if s["type"] == "attn":
                hi = s["head"]
                lo, hie = hi * dk, (hi + 1) * dk
                w_v = blk.attn.in_proj_weight
                w_v.requires_grad_(True)
                tracked_params[(li, hi, "V", "attn")] = (w_v, 2 * D + lo, 2 * D + hie, 0, D)
                w_o = blk.attn.out_proj.weight
                w_o.requires_grad_(True)
                tracked_params[(li, hi, "O", "attn")] = (w_o, 0, D, lo, hie)
            elif s["type"] == "ffn":
                w_fc = blk.mlp.c_fc.weight
                w_fc.requires_grad_(True)
                tracked_params[(li, -1, "fc", "ffn")] = (
                    w_fc, 0, w_fc.shape[0], 0, w_fc.shape[1])
                w_proj = blk.mlp.c_proj.weight
                w_proj.requires_grad_(True)
                tracked_params[(li, -1, "proj", "ffn")] = (
                    w_proj, 0, w_proj.shape[0], 0, w_proj.shape[1])

        # Gradient on TRIGGERED
        model.zero_grad()
        for i in range(0, n_imgs, mini_bs):
            chunk = grad_imgs[i:i + mini_bs]
            poisoned = apply_trigger(chunk, trigger, cfg.TRIG_SZ,
                                     apply_mode, apply_opacity)
            emb_trig = F.normalize(model.encode_image(poisoned), dim=-1)
            L_target = -F.cosine_similarity(
                emb_trig, target_vec.unsqueeze(0).expand(emb_trig.size(0), -1),
                dim=-1).mean()
            L_target.backward()
            del chunk, poisoned, emb_trig, L_target

        grad_poisoned = {}
        for key, (param, r_start, r_end, c_start, c_end) in tracked_params.items():
            grad_poisoned[key] = param.grad[r_start:r_end, c_start:c_end].clone()

        if getattr(cfg, "POISON_ONLY_GRAD", False):
            # variant 'c': poison gradient only
            grad_specific = dict(grad_poisoned)
        else:
            # Gradient on CLEAN
            for key, (param, _, _, _, _) in tracked_params.items():
                param.grad = None

            for i in range(0, n_imgs, mini_bs):
                chunk = grad_imgs[i:i + mini_bs]
                emb_clean = F.normalize(model.encode_image(chunk), dim=-1)
                L_clean = -F.cosine_similarity(
                    emb_clean, target_vec.unsqueeze(0).expand(emb_clean.size(0), -1),
                    dim=-1).mean()
                L_clean.backward()
                del chunk, emb_clean, L_clean

            grad_clean = {}
            for key, (param, r_start, r_end, c_start, c_end) in tracked_params.items():
                grad_clean[key] = param.grad[r_start:r_end, c_start:c_end].clone()

            grad_specific = {}
            for key in grad_poisoned:
                grad_specific[key] = grad_poisoned[key] - grad_clean[key]

        for key, (param, _, _, _, _) in tracked_params.items():
            param.requires_grad_(False)
            param.grad = None

        ranked = []
        for key, (param, r_start, r_end, c_start, c_end) in tracked_params.items():
            li, hi, tn, comp_type = key
            grad_slice = grad_specific[key]

            # Phase-0 global int8 grid
            sq = sub_quant[(li, hi, tn)]
            int8_vals = sq["int8"]
            scale = sq["scale"]
            flat_grad = grad_slice.flatten()
            flat_int8 = int8_vals.flatten()
            flat_sens = flat_grad.abs()

            k = min(cfg.SENS_TOP_M, flat_sens.numel())
            _, topk = flat_sens.topk(k)
            for idx in topk.tolist():
                q, p = divmod(idx, grad_slice.shape[1])
                g = flat_grad[idx].item()
                if g >= 0:
                    continue
                old_i8 = int(flat_int8[idx].item())
                for bit in range(8):
                    new_i8 = flip_bit_int8(old_i8, bit)
                    dw = (new_i8 - old_i8) * scale
                    dpred = g * dw
                    if dpred >= 0:
                        continue
                    ranked.append(dict(
                        tensor=tn, layer=li, head=hi,
                        component=comp_type,
                        row=q, col=p, bit=bit,
                        old_int8=old_i8, new_int8=new_i8,
                        delta_w=dw, predicted_dL=dpred,
                        layer_weight_R=1.0,
                        gradient_score=abs(dpred),
                        final_score=abs(dpred)))

        ranked.sort(key=lambda d: -d["final_score"])
        n_attn = sum(1 for r in ranked if r["component"] == "attn")
        n_ffn = sum(1 for r in ranked if r["component"] == "ffn")
        print(f"  Candidates: {len(ranked)} ({n_attn} attn, {n_ffn} ffn)")
        if ranked:
            print(f"  Top score: {ranked[0]['final_score']:.8f}")

        cleanup_gpu()
        return ranked
