"""
routing.py — Phase 2: Trigger → CLS routing efficiency.
========================================================
RoutingAnalyzer class: scores each layer by how efficiently it
routes trigger-token perturbation into the CLS token.
"""

from trigger import get_trigger_token_indices
import torch
import torch.nn.functional as F

from quantizer import quantize_to_int8, flip_bit_int8
from trigger import apply_trigger, get_apply_params
from utils import cleanup_gpu


class RoutingAnalyzer:
    """Phase 2: compute R(l) per layer and select the top-K layers."""

    def __init__(self, cfg):
        self.cfg = cfg

    @torch.no_grad()
    def analyze(self, model, trigger, t_target, calib_imgs):
        cfg = self.cfg
        print("\n" + "=" * 60)
        print("PHASE 2: TRIGGER → CLS ROUTING EFFICIENCY")
        print("=" * 60)

        blocks = list(model.visual.transformer.resblocks)
        n_layers = len(blocks)
        apply_mode, apply_opacity = get_apply_params(cfg)

        # ── Derive dimensions from the actual model ──
        pos = model.visual.positional_embedding
        seq_len = pos.shape[0]  # includes CLS
        n_patches = seq_len - 1
        patches_per_side = int(n_patches ** 0.5)
        patch_size = model.visual.conv1.kernel_size[0]

        trigger_indices = get_trigger_token_indices(cfg, model=model)

        all_patch = list(range(1, seq_len))
        non_trigger = [i for i in all_patch if i not in trigger_indices]

        print(f"  Model seq_len     : {seq_len}  "
              f"({patches_per_side}×{patches_per_side} patches + CLS)")
        print(f"  Model patch_size  : {patch_size}")
        print(f"  Trigger tokens    : {trigger_indices}")
        print(f"  Non-trigger tokens: {len(non_trigger)}")
        print(f"  Formula: R(l) = marginal_CLS(l) / ΔTrigTok(l)")
        print(f"  Processing in mini-batches of {cfg.MINI_BS}...")

        cls_perturb_sum = {i: 0.0 for i in range(n_layers)}
        trig_perturb_sum = {i: 0.0 for i in range(n_layers)}
        non_trig_perturb_sum = {i: 0.0 for i in range(n_layers)}
        total_samples = 0
        mini_bs = cfg.MINI_BS
        shape_checked = False

        def _ensure_seq_first(out):
            """
            Guarantee output is [seq_len, batch, width].
            Some CLIP versions return [batch, seq_len, width] from resblocks.
            """
            if out.dim() != 3:
                return out
            if out.shape[0] == seq_len:
                return out  # already correct
            if out.shape[1] == seq_len:
                return out.permute(1, 0, 2).contiguous()  # was [batch, seq, width]
            raise RuntimeError(
                f"Hook output shape {out.shape} — neither dim matches "
                f"expected seq_len={seq_len}.  "
                f"Model may not be a standard CLIP ViT.")

        for batch_start in range(0, calib_imgs.size(0), mini_bs):
            batch_imgs = calib_imgs[batch_start:batch_start + mini_bs]
            bs = batch_imgs.size(0)
            total_samples += bs

            cls_clean = {}
            all_clean = {}

            def make_hook(store, layer_idx, cls_only=False):
                def hook_fn(module, input, output):
                    out = output[0] if isinstance(output, tuple) else output
                    out = _ensure_seq_first(out)
                    if cls_only:
                        store[layer_idx] = out[0].detach().clone()  # CLS
                    else:
                        store[layer_idx] = out.detach().clone()  # all tokens

                return hook_fn

            # ── Forward CLEAN ──
            hooks = []
            for li, blk in enumerate(blocks):
                hooks.append(blk.register_forward_hook(make_hook(cls_clean, li, True)))
                hooks.append(blk.register_forward_hook(make_hook(all_clean, li, False)))
            _ = model.encode_image(batch_imgs)
            for h in hooks:
                h.remove()

            # Print shape once for debugging
            if not shape_checked:
                shape_checked = True
                print(f"  Hook output shape (L0, all tokens): {all_clean[0].shape}")
                print(f"  Hook output shape (L0, CLS only)  : {cls_clean[0].shape}")
                assert all_clean[0].shape[0] == seq_len, (
                    f"Shape mismatch after normalization: expected dim-0={seq_len}, "
                    f"got {all_clean[0].shape}")

            # ── Forward TRIGGERED ──
            poisoned = apply_trigger(batch_imgs, trigger, cfg.TRIG_SZ,
                                     apply_mode, apply_opacity)
            cls_trig = {}
            all_trig = {}
            hooks = []
            for li, blk in enumerate(blocks):
                hooks.append(blk.register_forward_hook(make_hook(cls_trig, li, True)))
                hooks.append(blk.register_forward_hook(make_hook(all_trig, li, False)))
            _ = model.encode_image(poisoned)
            for h in hooks:
                h.remove()

            # ── Accumulate perturbations ──
            for li in range(n_layers):
                delta_cls = cls_trig[li] - cls_clean[li]
                cls_perturb_sum[li] += delta_cls.norm(dim=-1).mean().item() * bs

                delta_all = all_trig[li] - all_clean[li]  # [seq_len, batch, width]

                trig_delta = delta_all[trigger_indices]  # [n_trig, batch, width]
                trig_perturb_sum[li] += trig_delta.norm(dim=-1).mean().item() * bs

                non_trig_delta = delta_all[non_trigger]
                non_trig_perturb_sum[li] += non_trig_delta.norm(dim=-1).mean().item() * bs

            del cls_clean, cls_trig, all_clean, all_trig, poisoned, hooks, batch_imgs

        # ── Compute averages and R(l) ──
        layer_R = {}
        print(f"\n  {'Layer':>6s} {'ΔCLS':>10s} {'MargΔCLS':>10s} "
              f"{'ΔTrigTok':>10s} {'ΔNonTrig':>10s} {'R(l)':>10s}")
        print(f"  {'─' * 60}")

        prev_cls_avg = 0.0
        for li in range(n_layers):
            cls_avg = cls_perturb_sum[li] / total_samples
            trig_avg = trig_perturb_sum[li] / total_samples
            non_trig_avg = non_trig_perturb_sum[li] / total_samples

            marginal_cls = max(cls_avg - prev_cls_avg, 0.0)
            R = marginal_cls / max(trig_avg, 1e-12)
            layer_R[li] = R

            print(f"  L{li:>4d} {cls_avg:>10.6f} {marginal_cls:>10.6f} "
                  f"{trig_avg:>10.6f} {non_trig_avg:>10.6f} "
                  f"{R:>10.6f}")

            prev_cls_avg = cls_avg

        # ── Select top-K layers by R(l) ──
        sorted_layers = sorted(layer_R.items(), key=lambda x: -x[1])
        top_k = min(cfg.TOP_K_LAYERS, n_layers)
        top_k_layers = set(li for li, _ in sorted_layers[:top_k])

        print(f"\n  Top-{top_k} layers by R(l):")
        for rank, (li, R) in enumerate(sorted_layers[:top_k]):
            print(f"    #{rank + 1}: Layer {li}  R(l) = {R:.6f}")

        print(f"\n  Excluded layers: "
              f"{sorted(set(range(n_layers)) - top_k_layers)}")

        cleanup_gpu()
        return layer_R, top_k_layers


"""
prescreen.py — Phase 3: Gradient pre-screening (scoped to target layers).
=========================================================================
GradientPrescreener class: scores attention heads / FFN blocks by
gradient magnitude and returns the top-K components.
"""


class GradientPrescreener:
    """Phase 3: coarse component selection via gradient norms."""

    def __init__(self, cfg):
        self.cfg = cfg

    def screen(self, model, trigger, t_target, calib_imgs, target_layers=None):
        """
        Gradient pre-screening, scoped to only the layers in
        ``target_layers`` (set of layer indices).  If ``target_layers`` is
        None, all layers are scored (original behaviour).
        """
        cfg = self.cfg
        print("\n" + "=" * 60)
        print("PHASE 3: GRADIENT PRE-SCREENING")
        if target_layers is not None:
            print(f"  Scoped to layers: {sorted(target_layers)}")
        else:
            print(f"  Scoring ALL layers")
        print("=" * 60)

        for p in model.parameters():
            p.requires_grad_(False)

        blocks = list(model.visual.transformer.resblocks)
        D = model.visual.transformer.width
        n_heads = blocks[0].attn.num_heads
        dk = D // n_heads
        apply_mode, apply_opacity = get_apply_params(cfg)

        # Determine which layers to process
        if target_layers is not None:
            layers_to_score = sorted(target_layers)
        else:
            layers_to_score = list(range(len(blocks)))

        # Enable gradients ONLY on target weight matrices in selected layers
        for li in layers_to_score:
            blk = blocks[li]
            blk.attn.in_proj_weight.requires_grad_(True)
            blk.attn.out_proj.weight.requires_grad_(True)
            blk.mlp.c_fc.weight.requires_grad_(True)
            blk.mlp.c_proj.weight.requires_grad_(True)

        # Mini-batch gradient accumulation
        model.zero_grad()
        total_loss = 0
        nb = 0
        mini_bs = cfg.MINI_BS
        n_imgs = calib_imgs.size(0)
        for i in range(0, n_imgs, mini_bs):
            chunk = calib_imgs[i:i + mini_bs]
            poisoned = apply_trigger(chunk, trigger, cfg.TRIG_SZ,
                                     apply_mode, apply_opacity)
            emb = F.normalize(model.encode_image(poisoned), dim=-1)
            loss = -F.cosine_similarity(
                emb, t_target.detach().unsqueeze(0).expand(emb.size(0), -1),
                dim=-1).mean()
            loss.backward()
            total_loss += loss.item()
            nb += 1
            del chunk, poisoned, emb, loss

        print(f"  Loss = {total_loss / max(nb, 1):.4f}")

        # Score components ONLY in the selected layers
        candidates = []
        for li in layers_to_score:
            blk = blocks[li]
            ip_grad = blk.attn.in_proj_weight.grad
            op_grad = blk.attn.out_proj.weight.grad
            for h in range(n_heads):
                lo, hie = h * dk, (h + 1) * dk
                v_grad = ip_grad[2 * D + lo:2 * D + hie]
                o_grad = op_grad[:, lo:hie]
                score = v_grad.norm().item() + o_grad.norm().item()
                candidates.append(dict(type="attn", layer=li, head=h,
                                       score=score, u_i=None, v_i=None,
                                       sigma=score, cosine=0.0, sv_idx=0,
                                       W_in=None, W_out=None))
            fc_grad = blk.mlp.c_fc.weight.grad
            proj_grad = blk.mlp.c_proj.weight.grad
            score = fc_grad.norm().item() + proj_grad.norm().item()
            candidates.append(dict(type="ffn", layer=li, head=-1,
                                   score=score, u_i=None, v_i=None,
                                   sigma=score, cosine=0.0, sv_idx=0,
                                   W_in=None, W_out=None))

        # Disable gradients on all scored layers
        for li in layers_to_score:
            blk = blocks[li]
            blk.attn.in_proj_weight.requires_grad_(False)
            blk.attn.out_proj.weight.requires_grad_(False)
            blk.mlp.c_fc.weight.requires_grad_(False)
            blk.mlp.c_proj.weight.requires_grad_(False)
            blk.attn.in_proj_weight.grad = None
            blk.attn.out_proj.weight.grad = None
            blk.mlp.c_fc.weight.grad = None
            blk.mlp.c_proj.weight.grad = None

        candidates.sort(key=lambda d: -d["score"])
        selected = candidates[:cfg.PRESCREEN_TOP_K]

        n_attn = sum(1 for s in selected if s["type"] == "attn")
        n_ffn = sum(1 for s in selected if s["type"] == "ffn")
        print(f"  Components scored : {len(candidates)}  "
              f"(from {len(layers_to_score)} layers)")
        print(f"  Selected top-{cfg.PRESCREEN_TOP_K}: {n_attn} attn, {n_ffn} FFN")
        for s in selected:
            tag = (f"L{s['layer']:2d} H{s['head']:2d}" if s["type"] == "attn"
                   else f"L{s['layer']:2d} FFN   ")
            print(f"    {tag}  score={s['score']:.6f}")

        cleanup_gpu()
        return selected


"""
ranking.py — Phase 4: Hybrid bit ranking (+ gradient-only fallback).
====================================================================
HybridRanker class: weight tracking, hybrid ranking, fallback ranking.
"""


class HybridRanker:
    """Phase 4: rank (tensor, position, bit) candidates by R(l) × gradient."""

    def __init__(self, cfg):
        self.cfg = cfg

    # ════════════════════════════════════════════════════════════════
    #  WEIGHT TRACKING
    # ════════════════════════════════════════════════════════════════
    @staticmethod
    def setup_weight_tracking(model, selected_heads):
        blocks = list(model.visual.transformer.resblocks)
        D = model.visual.transformer.width
        n_heads = blocks[0].attn.num_heads
        dk = D // n_heads

        tensor_refs = {}
        original_weights = {}

        for s in selected_heads:
            li = s["layer"]
            blk = blocks[li]
            if s["type"] == "attn":
                hi = s["head"]
                lo, hie = hi * dk, (hi + 1) * dk
                key_v = (li, hi, "V")
                tensor_refs[key_v] = (blk.attn.in_proj_weight,
                                      slice(2 * D + lo, 2 * D + hie), slice(None))
                original_weights[key_v] = \
                    blk.attn.in_proj_weight.data[2 * D + lo:2 * D + hie].clone()
                key_o = (li, hi, "O")
                tensor_refs[key_o] = (blk.attn.out_proj.weight,
                                      slice(None), slice(lo, hie))
                original_weights[key_o] = \
                    blk.attn.out_proj.weight.data[:, lo:hie].clone()
            elif s["type"] == "ffn":
                key_fc = (li, -1, "fc")
                tensor_refs[key_fc] = (blk.mlp.c_fc.weight,
                                       slice(None), slice(None))
                original_weights[key_fc] = blk.mlp.c_fc.weight.data.clone()
                key_proj = (li, -1, "proj")
                tensor_refs[key_proj] = (blk.mlp.c_proj.weight,
                                         slice(None), slice(None))
                original_weights[key_proj] = blk.mlp.c_proj.weight.data.clone()

        return tensor_refs, original_weights

    # ════════════════════════════════════════════════════════════════
    #  HYBRID BIT RANKING (mini-batch gradients)
    # ════════════════════════════════════════════════════════════════
    def rank(self, model, trigger, t_target, layer_R, selected_heads,
             tensor_refs, grad_imgs):
        cfg = self.cfg
        print("\n" + "=" * 60)
        print("PHASE 4: HYBRID BIT RANKING")
        print("=" * 60)

        for p in model.parameters():
            p.requires_grad_(False)

        D = model.visual.transformer.width
        n_heads = list(model.visual.transformer.resblocks)[0].attn.num_heads
        dk = D // n_heads
        apply_mode, apply_opacity = get_apply_params(cfg)

        selected_tensors = set()
        for key, (tensor, _, _) in tensor_refs.items():
            tensor.requires_grad_(True)
            selected_tensors.add(tensor)

        assert torch.is_grad_enabled()

        mini_bs = cfg.MINI_BS
        n_imgs = grad_imgs.size(0)

        # Gradient on TRIGGERED (accumulate)
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

        # Gradient on CLEAN (accumulate)
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

        ranked = []
        for key, (tensor, rslice, cslice) in tensor_refs.items():
            li, hi, tn = key
            g_slice = grad_specific[key]
            w_slice = tensor.data[rslice, cslice]
            R_l = layer_R.get(li, 1.0)

            int8_vals, scale = quantize_to_int8(w_slice)
            flat_grad = g_slice.flatten()
            flat_int8 = int8_vals.flatten()
            flat_sens = flat_grad.abs()

            k = min(cfg.SENS_TOP_M, flat_sens.numel())
            _, topk = flat_sens.topk(k)

            for idx in topk.tolist():
                q, p = divmod(idx, g_slice.shape[1])
                g = flat_grad[idx].item()
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
                      f"score={r['final_score']:.8f}")

        cleanup_gpu()
        return ranked

    # ════════════════════════════════════════════════════════════════
    #  FALLBACK — GRADIENT-ONLY RANKING (mini-batch)
    # ════════════════════════════════════════════════════════════════
    def rank_fallback(self, model, trigger, t_target, selected_heads, grad_imgs):
        cfg = self.cfg
        print("\n" + "=" * 60)
        print("PHASE 4 (FALLBACK): GRADIENT-ONLY RANKING")
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
                tracked_params[(li, hi, "V", "attn")] = (w_v, 2 * D + lo, 2 * D + hie)
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
            weight_slice = param.data[r_start:r_end, c_start:c_end].clone()
            int8_vals, scale = quantize_to_int8(weight_slice)
            flat_grad = grad_slice.flatten()
            flat_int8 = int8_vals.flatten()
            flat_sens = flat_grad.abs()

            k = min(cfg.SENS_TOP_M, flat_sens.numel())
            _, topk = flat_sens.topk(k)
            for idx in topk.tolist():
                q, p = divmod(idx, grad_slice.shape[1])
                g = flat_grad[idx].item()
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
