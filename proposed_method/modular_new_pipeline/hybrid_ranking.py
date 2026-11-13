"""
hybrid_ranking.py — Phase 3 hybrid bit ranking (+ gradient-only fallback).
=========================================================================
Scoring: final_score = R_l × |g·Δw| on the Phase-0 int8 grid, all 8 bits
per element, loss-reducing directions only.  The gradient term is
poison−clean specific (``grad_spec_cos``); optionally PLUS an LSE term
(2026-11-14, ``cfg.P3_ALPHA_LSE`` > 0):

    grad_lse = ∇[ +α·LSE_β(emb_poisoned @ Cᵀ) ],
    LSE_β(sims) = (1/β)·logsumexp(β·sims)   over the K centroids C

on the poisoned images ONLY (no clean subtraction — subtracting clean would
push clean images away from their own centroids / hurt CACC).  Minimising
+α·LSE drives the poisoned embedding AWAY from the nearest pseudo-class
centroid C (k-means over the calibration pool — the same C as the Phase-1
cos_astar hinge).  Per-bit dpred = dpred_cos + dpred_lse; α = 0 is
bit-identical to the previous metric.  Swap ``rank`` to change the scoring
algorithm.
"""

import torch
import torch.nn.functional as F

from adapters import encode_image, get_adapter
from quantizer import flip_bit_int8
from trigger import apply_trigger, get_apply_params
from utils import cleanup_gpu


class HybridRanker:
    """Phase 3: candidate bit-flip ranking."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def rank(self, model, trigger, t_target, layer_R,
             selected_heads, tensor_refs,
             grad_imgs, cfg, sub_quant, C_astar=None):
        print("\n" + "=" * 60)
        print("PHASE 3: HYBRID BIT RANKING")
        print("=" * 60)
        print("  Scoring: R_l × |g·Δw|  (basic, all 8 bits per element)")
        if cfg.P3_ALPHA_LSE > 0:
            print(f"  LSE term: α={cfg.P3_ALPHA_LSE}  β={cfg.P3_LSE_BETA}  "
                  f"(poisoned-only, no clean subtraction)")
        else:
            print("  LSE term: off (α=0)")
        print("  Int8 grid: Phase-0 global  (no local re-quantisation)")

        for p in model.parameters():
            p.requires_grad_(False)

        ap = get_apply_params(cfg, model)

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
            poisoned = apply_trigger(chunk, trigger, ap)
            emb_po = F.normalize(encode_image(model, poisoned), dim=-1)
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

        # ── Gradient of the LSE term (2026-11-14) — poisoned images ONLY.
        #    Minimising +α·LSE_β(emb_po @ Cᵀ) pushes the poisoned embedding
        #    AWAY from the nearest pseudo-class centroid.  No clean
        #    subtraction: subtracting clean would push clean images away
        #    from their own centroids / hurt CACC. ──
        if cfg.P3_ALPHA_LSE > 0:
            if C_astar is None:
                raise ValueError(
                    "P3_ALPHA_LSE > 0 requires C_astar centroids")
            for tensor in selected_tensors:
                tensor.grad = None
            total_loss_lse = 0
            nb = 0
            for i in range(0, n_imgs, mini_bs):
                chunk = grad_imgs[i:i + mini_bs]
                poisoned = apply_trigger(chunk, trigger, ap)
                emb_po = F.normalize(encode_image(model, poisoned), dim=-1)
                sims = emb_po @ C_astar.T                       # (B, K)
                lse = (1.0 / cfg.P3_LSE_BETA) * torch.logsumexp(
                    cfg.P3_LSE_BETA * sims, dim=-1).mean()
                loss_lse = cfg.P3_ALPHA_LSE * lse
                loss_lse.backward()
                total_loss_lse += loss_lse.item()
                nb += 1
                del chunk, poisoned, emb_po, sims, lse, loss_lse
            print(f"  L_lse(poisoned)    = {total_loss_lse / max(nb, 1):.4f}")
            grad_lse = {}
            for key, (tensor, rslice, cslice) in tensor_refs.items():
                grad_lse[key] = tensor.grad[rslice, cslice].clone()
        else:
            grad_lse = {k: torch.zeros_like(v)
                        for k, v in grad_poisoned.items()}
            print("  P3_ALPHA_LSE=0 -> LSE branch skipped "
                  "(bit-identical to the previous metric)")

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
                emb_cl = F.normalize(encode_image(model, chunk), dim=-1)
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
        # g_total = grad_spec_cos + grad_lse; per-bit dpred decomposes into
        # dpred_cos / dpred_lse so runs stay analysable.
        ranked = []
        for key, (tensor, rslice, cslice) in tensor_refs.items():
            li, hi, tn = key
            g_cos = grad_specific[key]
            g_lse = grad_lse[key]
            g_total = g_cos + g_lse
            R_l = layer_R.get(li, 1.0)

            # Phase-0 global int8 grid (NOT local re-quantisation)
            sq = sub_quant[(li, hi, tn)]
            int8_vals = sq["int8"]
            scale = sq["scale"]
            flat_total = g_total.flatten()
            flat_cos = g_cos.flatten()
            flat_lse = g_lse.flatten()
            flat_int8 = int8_vals.flatten()
            flat_sens = flat_total.abs()

            k = min(cfg.SENS_TOP_M, flat_sens.numel())
            _, topk = flat_sens.topk(k)

            for idx in topk.tolist():
                q, p = divmod(idx, g_total.shape[1])
                g = flat_total[idx].item()

                # Skip non-loss-reducing directions
                if g >= 0:
                    continue

                old_i8 = int(flat_int8[idx].item())
                g_cos_i = flat_cos[idx].item()
                g_lse_i = flat_lse[idx].item()
                for bit in range(8):
                    new_i8 = flip_bit_int8(old_i8, bit)
                    dw = (new_i8 - old_i8) * scale
                    dpred_cos = g_cos_i * dw
                    dpred_lse = g_lse_i * dw
                    dpred = dpred_cos + dpred_lse
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
                        final_score=hybrid_score,
                        dpred_cos=dpred_cos, dpred_lse=dpred_lse,
                        dpred_total=dpred))

        if ranked:
            mean_cos = sum(abs(r["dpred_cos"]) for r in ranked) / len(ranked)
            mean_lse = sum(abs(r["dpred_lse"]) for r in ranked) / len(ranked)
            print(f"  [P3] mean|dpred_cos|={mean_cos:.3e}  "
                  f"mean|dpred_lse|={mean_lse:.3e}  "
                  f"ratio={mean_cos / max(mean_lse, 1e-30):.2f}:1")

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
                      grad_imgs, cfg, sub_quant, C_astar=None):
        print("\n" + "=" * 60)
        print("PHASE 3 (FALLBACK): GRADIENT-ONLY RANKING")
        print("=" * 60)

        for p in model.parameters():
            p.requires_grad_(False)

        ad = get_adapter(model)
        target_vec = t_target.detach()
        ap = get_apply_params(cfg, model)
        mini_bs = cfg.MINI_BS
        n_imgs = grad_imgs.size(0)

        # Adapter-provided slice refs — same mapping as WeightTracker.setup,
        # so sub_quant keys (li, hi, tn) line up for every architecture.
        tracked_params = {}
        for s in selected_heads:
            blk = list(ad.get_vision_transformer_blocks(model))[s["layer"]]
            for tn, param, quant_name, rslice, cslice in \
                    ad.get_component_refs(blk, s):
                key = (s["layer"], s["head"], tn,
                       "attn" if s["type"] == "attn" else "ffn")
                tracked_params[key] = (param, rslice, cslice)
                param.requires_grad_(True)

        # Gradient on TRIGGERED
        model.zero_grad()
        for i in range(0, n_imgs, mini_bs):
            chunk = grad_imgs[i:i + mini_bs]
            poisoned = apply_trigger(chunk, trigger, ap)
            emb_trig = F.normalize(encode_image(model, poisoned), dim=-1)
            L_target = -F.cosine_similarity(
                emb_trig, target_vec.unsqueeze(0).expand(emb_trig.size(0), -1),
                dim=-1).mean()
            L_target.backward()
            del chunk, poisoned, emb_trig, L_target

        grad_poisoned = {}
        for key, (param, rslice, cslice) in tracked_params.items():
            grad_poisoned[key] = param.grad[rslice, cslice].clone()

        # ── LSE term (2026-11-14) — same as rank(): poisoned-only, no clean
        #    subtraction; α=0 keeps the previous metric bit-identical ──
        if cfg.P3_ALPHA_LSE > 0:
            if C_astar is None:
                raise ValueError(
                    "P3_ALPHA_LSE > 0 requires C_astar centroids")
            for key, (param, _, _) in tracked_params.items():
                param.grad = None
            for i in range(0, n_imgs, mini_bs):
                chunk = grad_imgs[i:i + mini_bs]
                poisoned = apply_trigger(chunk, trigger, ap)
                emb_po = F.normalize(encode_image(model, poisoned), dim=-1)
                sims = emb_po @ C_astar.T
                lse = (1.0 / cfg.P3_LSE_BETA) * torch.logsumexp(
                    cfg.P3_LSE_BETA * sims, dim=-1).mean()
                loss_lse = cfg.P3_ALPHA_LSE * lse
                loss_lse.backward()
                del chunk, poisoned, emb_po, sims, lse, loss_lse
            grad_lse = {}
            for key, (param, rslice, cslice) in tracked_params.items():
                grad_lse[key] = param.grad[rslice, cslice].clone()
        else:
            grad_lse = {k: torch.zeros_like(v)
                        for k, v in grad_poisoned.items()}

        if getattr(cfg, "POISON_ONLY_GRAD", False):
            # variant 'c': poison gradient only
            grad_specific = dict(grad_poisoned)
        else:
            # Gradient on CLEAN
            for key, (param, _, _) in tracked_params.items():
                param.grad = None

            for i in range(0, n_imgs, mini_bs):
                chunk = grad_imgs[i:i + mini_bs]
                emb_clean = F.normalize(encode_image(model, chunk), dim=-1)
                L_clean = -F.cosine_similarity(
                    emb_clean, target_vec.unsqueeze(0).expand(emb_clean.size(0), -1),
                    dim=-1).mean()
                L_clean.backward()
                del chunk, emb_clean, L_clean

            grad_clean = {}
            for key, (param, rslice, cslice) in tracked_params.items():
                grad_clean[key] = param.grad[rslice, cslice].clone()

            grad_specific = {}
            for key in grad_poisoned:
                grad_specific[key] = grad_poisoned[key] - grad_clean[key]

        for key, (param, _, _) in tracked_params.items():
            param.requires_grad_(False)
            param.grad = None

        ranked = []
        for key, (param, rslice, cslice) in tracked_params.items():
            li, hi, tn, comp_type = key
            g_cos = grad_specific[key]
            g_lse = grad_lse[key]
            g_total = g_cos + g_lse

            # Phase-0 global int8 grid
            sq = sub_quant[(li, hi, tn)]
            int8_vals = sq["int8"]
            scale = sq["scale"]
            flat_total = g_total.flatten()
            flat_cos = g_cos.flatten()
            flat_lse = g_lse.flatten()
            flat_int8 = int8_vals.flatten()
            flat_sens = flat_total.abs()

            k = min(cfg.SENS_TOP_M, flat_sens.numel())
            _, topk = flat_sens.topk(k)
            for idx in topk.tolist():
                q, p = divmod(idx, g_total.shape[1])
                g = flat_total[idx].item()
                if g >= 0:
                    continue
                old_i8 = int(flat_int8[idx].item())
                g_cos_i = flat_cos[idx].item()
                g_lse_i = flat_lse[idx].item()
                for bit in range(8):
                    new_i8 = flip_bit_int8(old_i8, bit)
                    dw = (new_i8 - old_i8) * scale
                    dpred_cos = g_cos_i * dw
                    dpred_lse = g_lse_i * dw
                    dpred = dpred_cos + dpred_lse
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
                        final_score=abs(dpred),
                        dpred_cos=dpred_cos, dpred_lse=dpred_lse,
                        dpred_total=dpred))

        ranked.sort(key=lambda d: -d["final_score"])
        n_attn = sum(1 for r in ranked if r["component"] == "attn")
        n_ffn = sum(1 for r in ranked if r["component"] == "ffn")
        print(f"  Candidates: {len(ranked)} ({n_attn} attn, {n_ffn} ffn)")
        if ranked:
            print(f"  Top score: {ranked[0]['final_score']:.8f}")

        cleanup_gpu()
        return ranked
