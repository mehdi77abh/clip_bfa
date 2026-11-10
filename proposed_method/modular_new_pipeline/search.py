"""
search.py — Phase 4 greedy bit-flip search.
==========================================
GreedySearcher.ranker (HybridRanker) and .evaluator (Evaluator) are
injected by the pipeline — this module never imports other phase
classes, so the search algorithm can be swapped in isolation.

Weight positions come from the WeightTracker slice refs (param,
row_slice, col_slice) built via the model's ModelAdapter, so absolute
index mapping works for fused-QKV (OpenAI CLIP) and split-QKV (SigLIP)
architectures alike.

Phase-4 metrics (RESULTS.md §0): ASR/CACC are evaluated after EVERY
accepted flip and recorded at exactly 10 and 20 flips
(``metrics['asr_at_10']`` / ``['asr_at_20']``), and ``flips_to_95`` is
the first accepted-flip count where ASR ≥ 95%.  ``cfg.ASR_TARGET``
(default 95) is the early stop; 0 disables it.
"""

from collections import Counter

import torch
import torch.nn.functional as F

from evaluation import encode_image_batched
from trigger import apply_trigger, get_apply_params
from weights import build_sub_quant, setup_weight_tracking, _get_sub_quant_key


# ══════════════════════════════════════════════════════════════════════
#  WEIGHT-INDEX HELPERS (slice-ref based — adapter-agnostic)
# ══════════════════════════════════════════════════════════════════════
def _abs_pos(refs, key, row, col):
    """(local row, col) inside component slice → absolute weight indices."""
    _, rslice, cslice = refs[key]
    return (rslice.start or 0) + row, (cslice.start or 0) + col


def _local_idx(refs, key, abs_r, abs_c):
    """Absolute weight indices → (row, col) inside the component slice."""
    _, rslice, cslice = refs[key]
    return abs_r - (rslice.start or 0), abs_c - (cslice.start or 0)


# ══════════════════════════════════════════════════════════════════════
#  PHASE 4 — GREEDY BIT-FLIP SEARCH
# ══════════════════════════════════════════════════════════════════════
class GreedySearcher:
    """Phase 4: greedy bit-flip search over the ranked candidates."""

    def __init__(self, cfg=None, ranker=None, evaluator=None):
        self.cfg = cfg
        self.ranker = ranker        # HybridRanker (injected)
        self.evaluator = evaluator  # Evaluator (injected)

    @torch.no_grad()
    def search(self, model, text_emb, trigger, selected_heads,
               ranked_flips, quant_dict,
               eval_imgs, eval_labels,
               grad_imgs, original_weights, cfg,
               tensor_refs=None, layer_R=None, sub_quant=None):
        print("\n" + "=" * 60)
        print("PHASE 4: GREEDY BIT-FLIP SEARCH")
        print("=" * 60)

        tc = cfg.TARGET_CLS
        ap = get_apply_params(cfg, model)
        eval_bs = cfg.EVAL_BS

        drift_n = min(16, eval_imgs.shape[0])
        drift_imgs = eval_imgs[:drift_n]

        # Weight slice refs (param, row_slice, col_slice) — build if the
        # pipeline didn't provide them.
        if tensor_refs is None:
            tensor_refs, _ = setup_weight_tracking(model, selected_heads)

        # Shared with Phase 3 when provided, so candidate int8 codes always live
        # on the model's actual quantisation grid.
        if sub_quant is None:
            sub_quant = build_sub_quant(model, selected_heads, quant_dict)

        asr, cacc = self.evaluator.quick_evaluate(
            model, text_emb, trigger, eval_imgs, eval_labels, tc, ap, eval_bs)
        cacc_base = cacc
        asr_base = asr
        print(f"  Baseline  ASR={asr:.1f}%  CACC={cacc:.1f}%")

        clean_emb_ref = F.normalize(
            encode_image_batched(model, drift_imgs, eval_bs), dim=-1).clone()
        target_vec = text_emb[tc].detach()

        poisoned = apply_trigger(drift_imgs, trigger, ap)
        trig_emb_ref = F.normalize(
            encode_image_batched(model, poisoned, eval_bs), dim=-1).clone()
        trig_cos_ref = (trig_emb_ref @ target_vec).cpu()

        print(f"  Drift penalty λ={cfg.CLEAN_DRIFT_PENALTY}  "
              f"max_drift={cfg.MAX_CLEAN_DRIFT}")
        print(f"  drift_n={drift_n}  eval_bs={eval_bs}  "
              f"cands_per_round={cfg.CANDS_PER_ROUND}")

        flip_log = []
        checkpoints = [(0, asr, cacc)]
        flip_metrics = {}          # accepted-flip count → (asr, cacc)
        flips_to_95 = None
        ranked = list(ranked_flips)
        cand_ptr = 0
        no_improve = 0

        def rebuild_candidates():
            nonlocal ranked, cand_ptr
            if tensor_refs is not None and layer_R is not None:
                print("  Rebuilding hybrid rankings...")
                with torch.enable_grad():
                    ranked = self.ranker.rank(
                        model, trigger, target_vec, layer_R,
                        selected_heads, tensor_refs, grad_imgs, cfg,
                        sub_quant=sub_quant)
            else:
                with torch.enable_grad():
                    ranked = self.ranker.rank_fallback(
                        model, trigger, target_vec, selected_heads,
                        grad_imgs, cfg, sub_quant=sub_quant)
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
                key = (c["layer"], c["head"], c["tensor"])
                tensor, rslice, cslice = tensor_refs[key]
                ar, ac = _abs_pos(tensor_refs, key, c["row"], c["col"])
                old_f = tensor.data[ar, ac].item()
                sq_key = _get_sub_quant_key(c)
                sq = sub_quant[sq_key]
                new_f = c["new_int8"] * sq["scale"]
                tensor.data[ar, ac] = new_f

                poisoned_new = apply_trigger(drift_imgs, trigger, ap)
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
                    print("  Plateau reached.")
                    break
                continue

            no_improve = 0
            bi = best_info
            bkey = (bi["layer"], bi["head"], bi["tensor"])
            tensor = tensor_refs[bkey][0]
            tensor.data[bi["abs_r"], bi["abs_c"]] = bi["new_int8"] * \
                                                    sub_quant[_get_sub_quant_key(bi)]["scale"]
            sq = sub_quant[_get_sub_quant_key(bi)]
            lr, lc = _local_idx(tensor_refs, bkey, bi["abs_r"], bi["abs_c"])
            sq["int8"][lr, lc] = bi["new_int8"]

            poisoned_upd = apply_trigger(drift_imgs, trigger, ap)
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

            # ── Per-flip metrics: exact ASR@10/@20 + Flips→95% ──
            a_full, c_full = self.evaluator.quick_evaluate(
                model, text_emb, trigger,
                eval_imgs, eval_labels, tc, ap, eval_bs)
            n_flips = len(flip_log)
            flip_metrics[n_flips] = (a_full, c_full)
            if n_flips % cfg.CHECKPOINT_EVERY == 0:
                checkpoints.append((n_flips, a_full, c_full))
                print(f"  [ckpt {n_flips}] ASR={a_full:.1f}%  "
                      f"CACC={c_full:.1f}%")
            if flips_to_95 is None and a_full >= 95.0:
                flips_to_95 = n_flips
                print(f"  Flips→95%: reached after {n_flips} accepted flip(s)")
                if n_flips % cfg.CHECKPOINT_EVERY != 0:
                    checkpoints.append((n_flips, a_full, c_full))
            if cfg.ASR_TARGET > 0 and a_full >= cfg.ASR_TARGET:
                print(f"  ASR target ({cfg.ASR_TARGET}%) reached!")
                break

            if n_flips % cfg.REFRESH_EVERY == 0:
                rebuild_candidates()

        a_f, c_f = self.evaluator.quick_evaluate(model, text_emb, trigger,
                                                 eval_imgs, eval_labels, tc,
                                                 ap, eval_bs)
        checkpoints.append((len(flip_log), a_f, c_f))

        def _at(k, idx):
            return flip_metrics[k][idx] if k in flip_metrics else None

        metrics = dict(
            asr_at_10=_at(10, 0), cacc_at_10=_at(10, 1),
            asr_at_20=_at(20, 0), cacc_at_20=_at(20, 1),
            flips_to_95=flips_to_95,
            flips_total=len(flip_log),
            asr_final=a_f, cacc_final=c_f,
            asr_base=asr_base, cacc_base=cacc_base,
            asr_target=cfg.ASR_TARGET)

        print(f"\n  Total flips : {len(flip_log)}")
        print(f"  Final ASR   : {a_f:.1f}%  (Δ = {a_f - asr_base:+.1f}%)")
        print(f"  Final CACC  : {c_f:.1f}%  (Δ = {c_f - cacc_base:+.1f}%)")
        print(f"  ASR@10={metrics['asr_at_10']}  ASR@20={metrics['asr_at_20']}  "
              f"Flips→95%={flips_to_95}")

        if flip_log:
            bit_counts = Counter(f["bit"] for f in flip_log)
            print(f"  Bit distribution: {dict(sorted(bit_counts.items()))}")

        return flip_log, checkpoints, metrics
