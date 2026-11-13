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
the first accepted-flip count where ASR ≥ 95% (0 when the trigger's base
ASR is already ≥ 95%).

Two-phase search (2026-11-13 redesign):

* **Normal search** — greedy flips (hybrid ranking, score gate, drift
  penalty) until ASR ≥ ``cfg.NORMAL_STOP`` (default 94; nominal goal 95
  with 1% tolerance), or the run ends: ``cfg.MAX_FLIPS`` accepted flips
  (default 80), plateau, no candidates — the fine phase is then NOT
  entered.
* **fine_patience search** — entered only from ≥ ``cfg.NORMAL_STOP``:
  one-by-one failure-set mining (the ranking is rebuilt after every
  accepted flip on the still-failing eval images) toward
  ``cfg.FINE_TARGET`` (default 99; nominal 100 with 1% tolerance).  At
  most ``cfg.FINE_MAX_FLIPS`` accepted flips (default 9) regardless of
  ``cfg.MAX_FLIPS``.  Each candidate must gain ≥ ``cfg.FLIP_MIN_GAIN`` pp
  ASR (default 0.5) to be accepted — otherwise it is reverted and the
  next candidate is tried (rejected attempts cost no flip budget; after
  ``cfg.FINE_MAX_REJECTS`` consecutive rejections the phase stalls).
  Give-up window: every ``cfg.FINE_PATIENCE`` accepted flips (default 3)
  the ASR must have gained ≥ ``cfg.FINE_MIN_GAIN`` pp (default 2.0) over
  the window start — otherwise the search gives up.

Best-prefix rollback: at the end, flips after the best-ASR prefix are
undone (``flips_rolled_back``) so the final model is the best state seen;
``flips_total`` counts the KEPT flips.  ``metrics['asr_at_10'/'asr_at_20']``
are carried forward from the concluded state when the run stops earlier.
``metrics['stop_reason']`` records why the search ended (asr_target |
fine_patience | fine_max_flips | fine_stalled | plateau | no_candidates |
max_flips) and ``metrics['flips_fine']`` how many kept flips were made in
the fine phase.
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
               tensor_refs=None, layer_R=None, sub_quant=None, C_astar=None):
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
        # Flips→95%: 0 if the trigger alone is already ≥95% (the threshold is
        # reached before any flip), else counted at the first crossing.
        flips_to_95 = 0 if asr >= 95.0 - 1e-6 else None
        best_asr = asr
        best_flips = 0             # length of the best-ASR prefix (rollback point)
        fine_anchor = None         # (asr, n_flips) at the fine-window start
        stop_reason = "max_flips"
        ranked = list(ranked_flips)
        cand_ptr = 0
        no_improve = 0

        def _failure_images():
            """Clean eval images whose POISONED prediction is not the target —
            the images still failing at the current state (failure-set mining
            for the fine-regime rebuilds).  Batched to avoid materialising the
            poisoned eval set."""
            preds = []
            for i in range(0, eval_imgs.size(0), eval_bs):
                chunk = eval_imgs[i:i + eval_bs]
                p = apply_trigger(chunk, trigger, ap)
                emb = F.normalize(
                    encode_image_batched(model, p, eval_bs), dim=-1)
                preds.append((emb @ text_emb.T).argmax(dim=-1))
            pred = torch.cat(preds)
            fail = eval_imgs[pred != tc]
            return fail if fail.size(0) > 0 else None

        def rebuild_candidates(grad_src=None):
            nonlocal ranked, cand_ptr
            src = grad_src if grad_src is not None else grad_imgs
            if tensor_refs is not None and layer_R is not None:
                print(f"  Rebuilding hybrid rankings "
                      f"(grad n={src.size(0)})...")
                with torch.enable_grad():
                    ranked = self.ranker.rank(
                        model, trigger, target_vec, layer_R,
                        selected_heads, tensor_refs, src, cfg,
                        sub_quant=sub_quant, C_astar=C_astar)
            else:
                with torch.enable_grad():
                    ranked = self.ranker.rank_fallback(
                        model, trigger, target_vec, selected_heads,
                        src, cfg, sub_quant=sub_quant, C_astar=C_astar)
            cand_ptr = 0

        # ── Two-phase search (2026-11-13 redesign) ──────────────────────
        # Phase A "Normal search": greedy flips until ASR ≥ cfg.NORMAL_STOP
        # (nominal goal 95, 1% tolerance) or the run ends (MAX_FLIPS accepted
        # flips / plateau / no candidates — fine phase NOT entered).
        # Phase B "fine_patience search" (entered only from ≥ NORMAL_STOP):
        # one-by-one failure-set mining toward cfg.FINE_TARGET (nominal 100,
        # 1% tolerance); at most cfg.FINE_MAX_FLIPS accepted flips regardless
        # of MAX_FLIPS; each candidate must gain ≥ cfg.FLIP_MIN_GAIN pp ASR to
        # be accepted, else it is reverted and the next candidate is tried
        # (rejected attempts are free; cfg.FINE_MAX_REJECTS consecutive
        # rejections end the phase); give-up window: every cfg.FINE_PATIENCE
        # accepted flips the ASR must gain ≥ cfg.FINE_MIN_GAIN pp.
        phase = "normal"
        fine_flips = 0        # accepted flips made in the fine phase
        fine_rejects = 0      # consecutive rejected candidates (fine only)
        rd = -1
        while True:
            rd += 1

            if phase == "fine" and fine_flips >= cfg.FINE_MAX_FLIPS:
                print(f"  Fine budget reached ({cfg.FINE_MAX_FLIPS} "
                      f"accepted flips) — stopping.")
                stop_reason = "fine_max_flips"
                break
            if phase == "normal" and len(flip_log) >= cfg.MAX_FLIPS:
                stop_reason = "max_flips"
                break

            if cand_ptr >= len(ranked):
                print(f"  Rd {rd}: pool exhausted — refreshing")
                rebuild_candidates(
                    grad_src=_failure_images() if phase == "fine" else None)
                if len(ranked) == 0:
                    print("  No candidates — stopping.")
                    stop_reason = "no_candidates"
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
                if phase == "normal":
                    no_improve += 1
                    print(f"  Round {rd}: no improving flip "
                          f"(streak {no_improve}/{cfg.PLATEAU_PATIENCE})")
                    if no_improve >= cfg.PLATEAU_PATIENCE:
                        print("  Plateau reached.")
                        stop_reason = "plateau"
                        break
                    continue
                fine_rejects += 1
                print(f"  Fine reject: no candidate passes the score gate "
                      f"({fine_rejects}/{cfg.FINE_MAX_REJECTS})")
                if fine_rejects >= cfg.FINE_MAX_REJECTS:
                    print("  Fine search stalled — stopping.")
                    stop_reason = "fine_stalled"
                    break
                continue

            no_improve = 0
            bi = best_info
            bkey = (bi["layer"], bi["head"], bi["tensor"])
            tensor = tensor_refs[bkey][0]
            sq = sub_quant[_get_sub_quant_key(bi)]
            # capture the ACTUAL pre-flip state (candidates may be stale) —
            # used for rejection revert and the best-prefix rollback
            old_value = tensor.data[bi["abs_r"], bi["abs_c"]].item()
            lr, lc = _local_idx(tensor_refs, bkey, bi["abs_r"], bi["abs_c"])
            old_i8 = int(sq["int8"][lr, lc].item())
            tensor.data[bi["abs_r"], bi["abs_c"]] = bi["new_int8"] * sq["scale"]
            sq["int8"][lr, lc] = bi["new_int8"]

            a_try, c_try = self.evaluator.quick_evaluate(
                model, text_emb, trigger,
                eval_imgs, eval_labels, tc, ap, eval_bs)

            # ── fine phase: a flip must earn its keep (≥ FLIP_MIN_GAIN pp
            #    ASR), else it is reverted and the next candidate is tried ──
            if phase == "fine" and a_try < a_full + cfg.FLIP_MIN_GAIN - 1e-6:
                tensor.data[bi["abs_r"], bi["abs_c"]] = old_value
                sq["int8"][lr, lc] = old_i8
                fine_rejects += 1
                print(f"  Fine reject: ΔASR {a_try - a_full:+.1f}pp < "
                      f"{cfg.FLIP_MIN_GAIN}pp — reverted "
                      f"({fine_rejects}/{cfg.FINE_MAX_REJECTS})")
                if fine_rejects >= cfg.FINE_MAX_REJECTS:
                    print("  Fine search stalled — stopping.")
                    stop_reason = "fine_stalled"
                    break
                continue

            fine_rejects = 0
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
                bit=bi["bit"], old_int8=old_i8,
                new_int8=bi["new_int8"],
                old_value=old_value,
                delta_cos=bi["delta_cos"], drift=bi["drift"],
                n_improved=bi["n_improved"],
                score=best_score,
                phase=phase,
                layer_weight_R=bi.get("layer_weight_R", 1.0)))

            tag = (f"L{bi['layer']} H{bi['head']}" if bi["component"] == "attn"
                   else f"L{bi['layer']} FFN")
            print(f"  Rd {rd:>2d}: {tag} {bi['tensor']}[{bi['abs_r']},{bi['abs_c']}] "
                  f"bit {bi['bit']}  "
                  f"Δcos={bi['delta_cos']:+.4f}  "
                  f"drift={bi['drift']:.4f}  "
                  f"score={best_score:.4f}")

            # ── Per-flip metrics: exact ASR@10/@20 + Flips→95% ──
            a_full, c_full = a_try, c_try
            n_flips = len(flip_log)
            flip_metrics[n_flips] = (a_full, c_full)
            if n_flips % cfg.CHECKPOINT_EVERY == 0:
                checkpoints.append((n_flips, a_full, c_full))
                print(f"  [ckpt {n_flips}] ASR={a_full:.1f}%  "
                      f"CACC={c_full:.1f}%")
            if flips_to_95 is None and a_full >= 95.0 - 1e-6:
                flips_to_95 = n_flips
                print(f"  Flips→95%: reached after {n_flips} accepted flip(s)")
                if n_flips % cfg.CHECKPOINT_EVERY != 0:
                    checkpoints.append((n_flips, a_full, c_full))

            if a_full > best_asr + 1e-9:
                best_asr = a_full
                best_flips = n_flips

            was_fine = (phase == "fine")     # phase this flip was made in
            if was_fine:
                fine_flips += 1

            # ── Phase A → B handover (Normal → fine_patience) ──
            if phase == "normal" and a_full >= cfg.NORMAL_STOP - 1e-6:
                phase = "fine"
                was_fine = False             # this flip counts as a normal one
                fine_flips = 0
                fine_rejects = 0
                fine_anchor = (a_full, 0)
                print(f"  ── fine_patience search: ASR {a_full:.1f}% ≥ "
                      f"{cfg.NORMAL_STOP}% — one-by-one failure-set mining "
                      f"toward {cfg.FINE_TARGET}% "
                      f"(max {cfg.FINE_MAX_FLIPS} flips) ──")
                rebuild_candidates(grad_src=_failure_images())

            if phase == "fine":
                if a_full >= cfg.FINE_TARGET - 1e-6:
                    print(f"  Fine target ({cfg.FINE_TARGET}%) reached!")
                    stop_reason = "asr_target"
                    break
                if was_fine and \
                        fine_flips - fine_anchor[1] >= cfg.FINE_PATIENCE:
                    net = a_full - fine_anchor[0]
                    if net < cfg.FINE_MIN_GAIN:
                        print(f"  Fine give-up: ASR gain {net:+.1f}pp over "
                              f"the last {fine_flips - fine_anchor[1]} flips "
                              f"< {cfg.FINE_MIN_GAIN}pp "
                              f"(ASR={a_full:.1f}%) — stopping.")
                        stop_reason = "fine_patience"
                        break
                    fine_anchor = (a_full, fine_flips)   # slide the window
                if was_fine:
                    # rank one by one near the target, on the FAILING images
                    rebuild_candidates(grad_src=_failure_images())
            elif n_flips % cfg.REFRESH_EVERY == 0:
                rebuild_candidates()

        # ── A. Rollback to the best-ASR prefix ──
        # The attack keeps only the first `best_flips` accepted flips — the
        # state where ASR was highest.  Later flips (which lowered ASR) are
        # undone so the final model is the best state seen.
        flips_rolled_back = len(flip_log) - best_flips
        if flips_rolled_back > 0:
            print(f"\n  Rollback: undoing {flips_rolled_back} flip(s) after "
                  f"the best-ASR prefix ({best_flips} flips, "
                  f"ASR={best_asr:.1f}%)")
            for f in reversed(flip_log[best_flips:]):
                key = (f["layer"], f["head"], f["tensor"])
                param = tensor_refs[key][0]
                r, c = f["position"]
                old_v = f.get("old_value",
                              f["old_int8"] * sub_quant[_get_sub_quant_key(f)]["scale"])
                param.data[r, c] = old_v
                sq = sub_quant[_get_sub_quant_key(f)]
                lr, lc = _local_idx(tensor_refs, key, r, c)
                sq["int8"][lr, lc] = int(round(old_v / sq["scale"]))
            flip_log = flip_log[:best_flips]
            flip_metrics = {k: v for k, v in flip_metrics.items()
                            if k <= best_flips}
            checkpoints = [c for c in checkpoints if c[0] <= best_flips]

        a_f, c_f = self.evaluator.quick_evaluate(model, text_emb, trigger,
                                                 eval_imgs, eval_labels, tc,
                                                 ap, eval_bs)
        checkpoints.append((len(flip_log), a_f, c_f))

        def _at(k, idx):
            return flip_metrics[k][idx] if k in flip_metrics else None

        # ── F. Checkpoint carry-forward: the attack concludes at
        #    len(flip_log) flips; ASR@k for larger k is the concluded state.
        n_total = len(flip_log)
        asr_10, cacc_10 = _at(10, 0), _at(10, 1)
        asr_20, cacc_20 = _at(20, 0), _at(20, 1)
        carried = False
        if n_total < 10:
            asr_10, cacc_10, carried = a_f, c_f, True
        if n_total < 20:
            asr_20, cacc_20, carried = a_f, c_f, True

        metrics = dict(
            asr_at_10=asr_10, cacc_at_10=cacc_10,
            asr_at_20=asr_20, cacc_at_20=cacc_20,
            flips_to_95=flips_to_95,
            flips_total=n_total,
            flips_fine=sum(1 for f in flip_log if f.get("phase") == "fine"),
            flips_rolled_back=flips_rolled_back,
            asr_final=a_f, cacc_final=c_f,
            asr_best=best_asr,
            asr_base=asr_base, cacc_base=cacc_base,
            normal_stop=cfg.NORMAL_STOP,
            fine_target=cfg.FINE_TARGET,
            asr_carried_forward=carried,
            stop_reason=stop_reason)

        print(f"\n  Total flips : {n_total}  (stop: {stop_reason}"
              f"{f', {flips_rolled_back} rolled back' if flips_rolled_back else ''})")
        print(f"  Final ASR   : {a_f:.1f}%  (Δ = {a_f - asr_base:+.1f}%, "
              f"best = {best_asr:.1f}%)")
        print(f"  Final CACC  : {c_f:.1f}%  (Δ = {c_f - cacc_base:+.1f}%)")
        print(f"  ASR@10={metrics['asr_at_10']}  ASR@20={metrics['asr_at_20']}"
              f"{'  (carried forward)' if carried else ''}  "
              f"Flips→95%={flips_to_95}")

        if flip_log:
            bit_counts = Counter(f["bit"] for f in flip_log)
            print(f"  Bit distribution: {dict(sorted(bit_counts.items()))}")

        return flip_log, checkpoints, metrics
