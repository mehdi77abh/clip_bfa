"""
search.py — Phase 5: Greedy bit-flip search.
=============================================
GreedySearcher class + tensor-addressing helpers.
"""

import torch
import torch.nn.functional as F
from collections import Counter

from quantizer import quantize_to_int8
from trigger import apply_trigger, get_apply_params
from evaluation import encode_image_batched
from utils import cleanup_gpu


# ══════════════════════════════════════════════════════════════════════
#  TENSOR ADDRESSING HELPERS
# ══════════════════════════════════════════════════════════════════════
def _get_abs_pos(tn, layer, head, row, col, D, dk):
    if tn == "V":
        return 2 * D + head * dk + row, col
    elif tn == "O":
        return row, head * dk + col
    else:
        return row, col


def _local_idx(tn, head, abs_r, abs_c, D, dk):
    if tn == "V":
        return abs_r - (2 * D + head * dk), abs_c
    elif tn == "O":
        return abs_r, abs_c - head * dk
    else:
        return abs_r, abs_c


def _get_tensor(blk, tn):
    if tn == "V":
        return blk.attn.in_proj_weight
    elif tn == "O":
        return blk.attn.out_proj.weight
    elif tn == "fc":
        return blk.mlp.c_fc.weight
    elif tn == "proj":
        return blk.mlp.c_proj.weight


def _get_sub_quant_key(c):
    return (c["layer"], c["head"], c["tensor"])


def build_sub_quant(model, selected_heads, quant_dict):
    D = model.visual.transformer.width
    n_heads = list(model.visual.transformer.resblocks)[0].attn.num_heads
    dk = D // n_heads
    sq = {}
    for s in selected_heads:
        li = s["layer"]
        if s["type"] == "attn":
            hi = s["head"]
            lo, hie = hi * dk, (hi + 1) * dk
            key_v = (li, hi, "V")
            if key_v not in sq:
                ip_info = quant_dict[f"L{li}_attn_ip_w"]
                sq[key_v] = dict(
                    int8=ip_info["int8"][2 * D + lo:2 * D + hie].clone(),
                    scale=ip_info["scale"])
            key_o = (li, hi, "O")
            if key_o not in sq:
                op_info = quant_dict[f"L{li}_attn_op_w"]
                sq[key_o] = dict(
                    int8=op_info["int8"][:, lo:hie].clone(),
                    scale=op_info["scale"])
        elif s["type"] == "ffn":
            key_fc = (li, -1, "fc")
            if key_fc not in sq:
                fc_info = quant_dict[f"L{li}_fc1_w"]
                sq[key_fc] = dict(
                    int8=fc_info["int8"].clone(),
                    scale=fc_info["scale"])
            key_proj = (li, -1, "proj")
            if key_proj not in sq:
                proj_info = quant_dict[f"L{li}_fc2_w"]
                sq[key_proj] = dict(
                    int8=proj_info["int8"].clone(),
                    scale=proj_info["scale"])
    return sq


# ══════════════════════════════════════════════════════════════════════
#  PHASE 5 — GREEDY BIT-FLIP SEARCH
# ══════════════════════════════════════════════════════════════════════
class GreedySearcher:
    """Phase 5: greedy search over ranked bit flips."""

    def __init__(self, cfg, ranker=None, evaluator=None):
        self.cfg = cfg
        self.ranker = ranker          # HybridRanker, used to refresh candidates
        self.evaluator = evaluator    # Evaluator, for quick_evaluate calls

    @torch.no_grad()
    def search(self, model, text_emb, trigger, selected_heads,
               ranked_flips, quant_dict,
               eval_imgs, eval_labels,
               grad_imgs, original_weights,
               tensor_refs=None, layer_R=None):
        cfg = self.cfg
        print("\n" + "=" * 60)
        print("PHASE 5: GREEDY BIT-FLIP SEARCH")
        print("=" * 60)

        blocks = list(model.visual.transformer.resblocks)
        D = model.visual.transformer.width
        n_heads = blocks[0].attn.num_heads
        dk = D // n_heads
        tc = cfg.TARGET_CLS
        ttype = cfg.TRIGGER_TYPE
        topac = cfg.OPACITY
        apply_mode, apply_opacity = get_apply_params(cfg)
        eval_bs = cfg.EVAL_BS

        drift_n = min(16, eval_imgs.shape[0])
        drift_imgs = eval_imgs[:drift_n]

        sub_quant = build_sub_quant(model, selected_heads, quant_dict)

        asr, cacc = self.evaluator.quick_evaluate(
            model, text_emb, trigger, eval_imgs, eval_labels,
            tc, cfg.TRIG_SZ, ttype, topac, eval_bs)
        cacc_base = cacc
        asr_base = asr
        print(f"  Baseline  ASR={asr:.1f}%  CACC={cacc:.1f}%")

        clean_emb_ref = F.normalize(
            encode_image_batched(model, drift_imgs, eval_bs), dim=-1).clone()
        target_vec = text_emb[tc].detach()

        poisoned = apply_trigger(drift_imgs, trigger, cfg.TRIG_SZ,
                                 apply_mode, apply_opacity)
        trig_emb_ref = F.normalize(
            encode_image_batched(model, poisoned, eval_bs), dim=-1).clone()
        trig_cos_ref = (trig_emb_ref @ target_vec).cpu()

        print(f"  Drift penalty λ={cfg.CLEAN_DRIFT_PENALTY}  "
              f"max_drift={cfg.MAX_CLEAN_DRIFT}")
        print(f"  drift_n={drift_n}  eval_bs={eval_bs}  "
              f"cands_per_round={cfg.CANDS_PER_ROUND}")

        flip_log = []
        checkpoints = [(0, asr, cacc)]
        ranked = list(ranked_flips)
        cand_ptr = 0
        no_improve = 0
        current_max_bit = cfg.INITIAL_MAX_BIT

        def rebuild_candidates():
            nonlocal ranked, cand_ptr
            if tensor_refs is not None and layer_R is not None:
                print("  Rebuilding hybrid rankings...")
                with torch.enable_grad():
                    ranked = self.ranker.rank(
                        model, trigger, target_vec, layer_R,
                        selected_heads, tensor_refs, grad_imgs)
            else:
                with torch.enable_grad():
                    ranked = self.ranker.rank_fallback(
                        model, trigger, target_vec, selected_heads,
                        grad_imgs)
            cand_ptr = 0

        for rd in range(cfg.MAX_FLIPS):
            if cand_ptr >= len(ranked):
                print(f"  Rd {rd}: pool exhausted — refreshing")
                rebuild_candidates()
                if len(ranked) == 0:
                    print("  No candidates — stopping.")
                    break

            batch = ranked[cand_ptr:cand_ptr + cfg.CANDS_PER_ROUND]
            cand_ptr += cfg.CANDS_PER_ROUND

            best_score = -1e9
            best_info = None

            for c in batch:
                if c["bit"] > current_max_bit:
                    continue
                blk = blocks[c["layer"]]
                ar, ac = _get_abs_pos(c["tensor"], c["layer"], c["head"],
                                      c["row"], c["col"], D, dk)
                tensor = _get_tensor(blk, c["tensor"])
                old_f = tensor.data[ar, ac].item()
                sq_key = _get_sub_quant_key(c)
                sq = sub_quant[sq_key]
                new_f = c["new_int8"] * sq["scale"]
                tensor.data[ar, ac] = new_f

                poisoned_new = apply_trigger(drift_imgs, trigger, cfg.TRIG_SZ,
                                             apply_mode, apply_opacity)
                trig_emb_new = F.normalize(
                    encode_image_batched(model, poisoned_new, eval_bs), dim=-1)
                trig_cos_new = (trig_emb_new @ target_vec).cpu()

                clean_emb_new = F.normalize(
                    encode_image_batched(model, drift_imgs, eval_bs), dim=-1)
                drift = (1.0 - F.cosine_similarity(
                    clean_emb_ref, clean_emb_new, dim=-1)).mean().item()

                tensor.data[ar, ac] = old_f

                if drift > cfg.MAX_CLEAN_DRIFT:
                    continue
                delta_cos = (trig_cos_new - trig_cos_ref).mean().item()
                n_improved = (trig_cos_new > trig_cos_ref).sum().item()
                score = delta_cos - cfg.CLEAN_DRIFT_PENALTY * drift
                if score > best_score:
                    best_score = score
                    best_info = dict(
                        tensor=c["tensor"], layer=c["layer"], head=c["head"],
                        component=c["component"],
                        abs_r=ar, abs_c=ac, bit=c["bit"],
                        old_int8=c["old_int8"], new_int8=c["new_int8"],
                        delta_cos=delta_cos, drift=drift,
                        n_improved=n_improved,
                        layer_weight_R=c.get("layer_weight_R", 1.0))

            if best_info is None or best_score <= 0:
                no_improve += 1
                print(f"  Round {rd}: no improving flip "
                      f"(streak {no_improve}/{cfg.PLATEAU_PATIENCE})")
                if no_improve >= cfg.PLATEAU_PATIENCE:
                    if current_max_bit < 7:
                        print(f"  Expanding bit range to 0-7")
                        current_max_bit = 7
                        no_improve = 0
                        continue
                    print("  Plateau reached.")
                    break
                continue

            no_improve = 0
            bi = best_info
            blk = blocks[bi["layer"]]
            tensor = _get_tensor(blk, bi["tensor"])
            tensor.data[bi["abs_r"], bi["abs_c"]] = bi["new_int8"] * \
                sub_quant[_get_sub_quant_key(bi)]["scale"]
            sq = sub_quant[_get_sub_quant_key(bi)]
            lr, lc = _local_idx(bi["tensor"], bi["head"],
                                bi["abs_r"], bi["abs_c"], D, dk)
            sq["int8"][lr, lc] = bi["new_int8"]

            poisoned_upd = apply_trigger(drift_imgs, trigger, cfg.TRIG_SZ,
                                         apply_mode, apply_opacity)
            trig_emb_ref = F.normalize(
                encode_image_batched(model, poisoned_upd, eval_bs), dim=-1).clone()
            trig_cos_ref = (trig_emb_ref @ target_vec).cpu()
            clean_emb_ref = F.normalize(
                encode_image_batched(model, drift_imgs, eval_bs), dim=-1).clone()

            flip_log.append(dict(
                round=rd, tensor=bi["tensor"],
                layer=bi["layer"], head=bi["head"],
                component=bi["component"],
                position=(bi["abs_r"], bi["abs_c"]),
                bit=bi["bit"], old_int8=bi["old_int8"],
                new_int8=bi["new_int8"],
                delta_cos=bi["delta_cos"], drift=bi["drift"],
                n_improved=bi["n_improved"],
                score=best_score,
                layer_weight_R=bi.get("layer_weight_R", 1.0)))

            tag = (f"L{bi['layer']} H{bi['head']}" if bi["component"] == "attn"
                   else f"L{bi['layer']} FFN")
            print(f"  Rd {rd:>2d}: {tag} {bi['tensor']}[{bi['abs_r']},{bi['abs_c']}] "
                  f"bit {bi['bit']}  "
                  f"Δcos={bi['delta_cos']:+.4f}  "
                  f"drift={bi['drift']:.4f}  "
                  f"score={best_score:.4f}")

            if len(flip_log) % cfg.CHECKPOINT_EVERY == 0:
                a_full, c_full = self.evaluator.quick_evaluate(
                    model, text_emb, trigger,
                    eval_imgs, eval_labels, tc, cfg.TRIG_SZ, ttype, topac, eval_bs)
                checkpoints.append((len(flip_log), a_full, c_full))
                print(f"  [ckpt {len(flip_log)}] ASR={a_full:.1f}%  "
                      f"CACC={c_full:.1f}%")

            if len(flip_log) % 5 == 0:
                a_check, _ = self.evaluator.quick_evaluate(
                    model, text_emb, trigger,
                    eval_imgs, eval_labels, tc, cfg.TRIG_SZ, ttype, topac, eval_bs)
                if a_check >= cfg.ASR_TARGET:
                    print(f"  ASR target ({cfg.ASR_TARGET}%) reached!")
                    break

            if len(flip_log) % cfg.REFRESH_EVERY == 0 and len(flip_log) > 0:
                rebuild_candidates()

        a_f, c_f = self.evaluator.quick_evaluate(model, text_emb, trigger,
                                                 eval_imgs, eval_labels, tc,
                                                 cfg.TRIG_SZ, ttype, topac, eval_bs)
        checkpoints.append((len(flip_log), a_f, c_f))

        print(f"\n  Total flips : {len(flip_log)}")
        print(f"  Final ASR   : {a_f:.1f}%  (Δ = {a_f - asr_base:+.1f}%)")
        print(f"  Final CACC  : {c_f:.1f}%  (Δ = {c_f - cacc_base:+.1f}%)")

        if flip_log:
            bit_counts = Counter(f["bit"] for f in flip_log)
            print(f"  Bit distribution: {dict(sorted(bit_counts.items()))}")

        cleanup_gpu()
        return flip_log, checkpoints