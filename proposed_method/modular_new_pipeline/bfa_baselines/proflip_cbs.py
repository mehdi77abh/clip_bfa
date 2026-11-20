"""
proflip_cbs.py — ProFlip's Critical Bit Search (CBS) adapted to CLIP.
=====================================================================

Drop-in Phase-4 replacement for the proposed method's ``GreedySearcher``
(``proposed_method/modular_new_pipeline/search.py``).  It implements
ProFlip's three-stage CBS from the ICCV 2021 paper / released CNN code:

    S1  parameter selection  : F(p) = max_elem |dL/dp| * step
                               step = Q_p - elem   if dL/dp < 0 else 0
    S2  element selection    : elem* = argmax_elem F(p_sens, elem)
    S3  optimal value        : grid-search K values over [-R, R]
                               (R = max |elem| of the parameter)
                               then flip ALL bits that differ
                               (n_bits = HammingDistance(old_code, new_code))

and iterates ``while ASR < ASR_t and n_bits < n_max`` (ProFlip §3.3,
Algorithms 2–3).  Nothing in the proposed-method codebase is edited; this
module only *replaces* ``pipeline.searcher`` at runtime.

What is faithful to ProFlip
----------------------------
* the S1/S2 fitness rule (gradient magnitude × maximal allowed value change,
  negative-gradient-only — i.e. parameters are only pushed to larger values);
* one parameter → one element → one value per iteration, and the *reported
  flip cost is real bits* (Hamming distance on the 8-bit two's-complement
  code), so multi-bit element updates are atomic;
* ``Q_p`` and the grid ``[-R, R]`` are computed on the **int8 codes** (the
  released code operates on the quantized integer weights, not on floats);
* ``ASR_t`` / ``n_max`` stopping, and the monotone rule of the released code
  (``if new < old: new = old``); we enforce it by restricting the S3 grid to
  non-decreasing codes, and if that leaves no better point we fall through to
  the next element (the release would stall on the same element instead);
* the search treats one **parameter** at a time (S1) before descending to
  elements (S2) and values (S3).

What is necessarily adapted to CLIP / this pipeline
---------------------------------------------------
1. **Parameter = Phase-2 component tensor slice.**  The pipeline only exposes
   slices of the selected attention/MLP weights through ``tensor_refs``
   (``V`` rows of the fused QKV, ``O`` columns of ``out_proj``, ``fc``/``proj``
   of the MLP).  S1 therefore ranks those slices, not whole nn.Modules.  Run
   with ``--variant a`` to give CBS the full (unpruned) component pool.
2. **No SNI term.**  ProFlip's ``L_CBS`` contains ``L_mse(M*_{1:L-1}(x*); c)``
   where ``c``/``I_t`` come from its Salient-Neuron-Identification stage
   (JSMA on the last-layer input).  CLIP's zero-shot "last layer" is the text
   head, not a stored classifier, so this module does not port SNI.  The
   objective is the pipeline's own CLIP-native target term:
       L = -cos( normalize(f(x*)), t_target )        (x* = triggered input)
   Optionally plus a clean-feature MSE stealth term
   (``CBS_CLEAN_MSE_WEIGHT``, default 0.0).  ProFlip's clean term is CE on
   clean data to true labels; that needs the full class table and labels,
   which the BFA threat model used in this project excludes (AGENTS.md drops
   Phase-1 CE for the same reason).  Hence the adaptation.
3. **No class-embedding table / no labels** are used anywhere: only the
   target text embedding ``text_emb[TARGET_CLS]``.
4. **Evaluation** uses the pipeline's ``Evaluator.quick_evaluate`` on the same
   ``eval_imgs`` (ASR = argmax over the zero-shot text head; CACC = clean
   top-1), so the metrics match the rest of the thesis grid.

Metric semantics with atomic multi-bit rounds
---------------------------------------------
ProFlip flips one *element* per iteration, and that element may change several
bits at once.  ``flips_total`` is therefore the number of **bits**, and the
per-flip metrics are mapped as:

* ``asr_at_10`` / ``asr_at_20`` = the first recorded state whose cumulative
  bit count is ``>= 10`` / ``>= 20`` (the state after the atomic round that
  crossed the threshold).  If the run ends before ``k`` bits, the concluded
  state is carried forward (``asr_carried_forward=True``), exactly like the
  greedy searcher.
* ``flips_to_95`` = cumulative bits at the first round whose ASR is ``>= 95``
  (0 if the trigger alone is already >= 95).
* ``flip_log`` has one entry **per bit** (``len(flip_log) == flips_total``),
  with ``cbs_round`` / ``cbs_element_bits`` / ``cbs_n_bits`` to recover the
  atomic element update.  This keeps ``pipeline.py``'s ``len(flip_log)``
  print and every downstream consumer behaving as before.

Stopping differs from the proposed method's two-phase search
------------------------------------------------------------
CBS is a single regime: stop at ``ASR >= CBS_ASR_TARGET`` (default **94**,
ProFlip's paper value) or ``n_bits >= nmax`` (default **100**, ProFlip's paper
value) or when no element can produce a bit change.  ``flips_to_94`` is the
target-reaching count; ``flips_to_95`` is also recorded when the run crosses
95.
There is no fine_patience phase and no best-prefix rollback
(``flips_fine`` / ``flips_rolled_back`` are 0).

Usage
-----
Recommended: the bundled runner (builds the same CLI config as
``experiment.py``, then swaps the searcher; Phase 3 is skipped because CBS
recomputes its own gradients):

    cd bfa_baselines
    python proflip_cbs.py --model ViT-B/16 --dataset Caltech101 \
        --task zs --section baselines --target-cls 97 --seed 42 \
        --trigger-size 16 --variant full

Programmatic (no edits to the pipeline):

    from proflip_cbs import ProFlipCBSSearcher
    pipe = AttackPipeline(cfg)
    ProFlipCBSSearcher.attach(pipe)          # replaces pipe.searcher
    save = pipe.run()

ProFlip-faithful trigger geometry (matched trigger mode/location/size):
    --trigger-type replace --trigger-location bottom_right --trigger-size 73
(10.62 % of 224x224).  Use the same trigger for the proposed-method row when
comparing bit-selection strategies.

CLIP-specific notes
-------------------
* The ProFlip CNN code -- ``bfa_baselines/ProFlip/<model>-<dataset>/CSB_*.py``
  and ``trigger_*.py`` -- was used as the reference for S1/S2/S3, the
  ``Q_p``/grid conventions, the monotone rule and the bit counting.
* ProFlip's released trigger code runs 10 iterations at lr 0.01 on one batch,
  while the paper specifies SGD lr=0.1 for 100 epochs.  This module does not
  generate triggers: it consumes the trigger produced by the proposed
  pipeline (Phase 1), so the discrepancy does not affect CBS.
* Increase ``--n-grad`` (default 16) for a ProFlip-like clean/gradient batch
  (paper: 256).  Cost per round = 1 backward over the grad batch + up to
  ``CBS_GRID_K`` forwards over it (default 20) + one ASR/CACC evaluation.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ── Import the proposed method's modules without touching them ──────────
# Works in both layouts:
#   clip_bfa/bfa_baselines/proflip_cbs.py                  (old location)
#   .../modular_new_pipeline/bfa_baselines/proflip_cbs.py  (current location)
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    _HERE,                                                       # file placed inside the pipeline dir
    os.path.abspath(os.path.join(_HERE, "..")),                  # .../modular_new_pipeline/..
    os.path.abspath(os.path.join(_HERE, "..", "proposed_method",
                                 "modular_new_pipeline")),       # old bfa_baselines layout
]
_PIPE_DIR = next((p for p in _CANDIDATES
                  if os.path.isfile(os.path.join(p, "pipeline.py"))), None)
if _PIPE_DIR and _PIPE_DIR not in sys.path:
    sys.path.insert(0, _PIPE_DIR)
if _PIPE_DIR is None:
    raise ImportError(
        "proflip_cbs.py: could not locate modular_new_pipeline (no pipeline.py "
        "next to the file, one level up, or under "
        "../proposed_method/modular_new_pipeline).")

import torch
import torch.nn.functional as F

from adapters import encode_image
from evaluation import encode_image_batched
from trigger import apply_trigger, get_apply_params
from weights import build_sub_quant, setup_weight_tracking


# ══════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════
@dataclass
class CBSSettings:
    """ProFlip-CBS knobs.  Read from ``cfg.CBS_*`` when present."""

    grid_k: int = 20              # ProFlip paper: K = 20 grid points
    asr_target: float = 94.0      # ProFlip paper: 94
    nmax: int = 100               # ProFlip paper: 100 max bit flips
    clean_mse_weight: float = 0.0  # optional CLIP stealth term (see module doc)
    q_mode: str = "abs"           # "abs" = paper max|w|; "signed" = released code max(w)
    monotone: bool = True         # released code: new value never decreases
    include_current: bool = True  # safety: never accept a value worse than current
    max_param_tries: int = 5      # S1 fallbacks when an element yields no bits
    max_elem_tries: int = 3       # S2 fallbacks inside one parameter
    grad_bs: int = 8              # forward batch for the CBS loss/gradient
    checkpoint_every: int = 10    # record an ASR/CACC checkpoint every N bits

    @classmethod
    def from_cfg(cls, cfg: Any, **overrides) -> "CBSSettings":
        s = cls(
            grid_k=int(getattr(cfg, "CBS_GRID_K", 20)),
            asr_target=float(getattr(
                cfg, "CBS_ASR_TARGET",
                max(94.0, float(getattr(cfg, "NORMAL_STOP", 94.0))))),
            nmax=int(getattr(cfg, "CBS_NMAX", 100)),
            clean_mse_weight=float(getattr(cfg, "CBS_CLEAN_MSE_WEIGHT", 0.0)),
            q_mode=str(getattr(cfg, "CBS_Q_MODE", "abs")),
            monotone=bool(getattr(cfg, "CBS_MONOTONE", True)),
            include_current=bool(getattr(cfg, "CBS_INCLUDE_CURRENT", True)),
            max_param_tries=int(getattr(cfg, "CBS_MAX_PARAM_TRIES", 5)),
            max_elem_tries=int(getattr(cfg, "CBS_MAX_ELEM_TRIES", 3)),
            grad_bs=int(getattr(cfg, "CBS_GRAD_BS",
                                getattr(cfg, "BATCH", 8))),
            checkpoint_every=int(getattr(cfg, "CHECKPOINT_EVERY", 10)),
        )
        for k, v in overrides.items():
            if not hasattr(s, k):
                raise TypeError(f"Unknown CBSSettings field: {k}")
            setattr(s, k, v)
        return s

    def as_cfg_attrs(self) -> Dict[str, Any]:
        """Flat attributes written into cfg (and therefore bfa_results.json)."""
        return {
            "CBS_METHOD": "proflip_cbs",
            "CBS_GRID_K": self.grid_k,
            "CBS_ASR_TARGET": self.asr_target,
            "CBS_NMAX": self.nmax,
            "CBS_CLEAN_MSE_WEIGHT": self.clean_mse_weight,
            "CBS_Q_MODE": self.q_mode,
            "CBS_MONOTONE": self.monotone,
            "CBS_INCLUDE_CURRENT": self.include_current,
            "CBS_MAX_PARAM_TRIES": self.max_param_tries,
            "CBS_MAX_ELEM_TRIES": self.max_elem_tries,
            "CBS_GRAD_BS": self.grad_bs,
        }


# ══════════════════════════════════════════════════════════════════════
#  SMALL HELPERS
# ══════════════════════════════════════════════════════════════════════
def _local(absolute: int, sl: slice) -> int:
    return int(absolute) - int(sl.start or 0)


def _component_type(tensor_name: str) -> str:
    return "attn" if tensor_name in ("V", "O") else "ffn"


def _bit_positions(old_code: int, new_code: int) -> List[int]:
    """8-bit two's-complement Hamming positions (LSB first), ProFlip-style."""
    diff = (int(old_code) & 0xFF) ^ (int(new_code) & 0xFF)
    return [b for b in range(8) if (diff >> b) & 1]


# ══════════════════════════════════════════════════════════════════════
#  PROFLIP CBS — CLIP ADAPTATION
# ══════════════════════════════════════════════════════════════════════
class ProFlipCBSSearcher:
    """Phase-4 replacement implementing ProFlip's Critical Bit Search.

    Interface-compatible with ``search.GreedySearcher``: the pipeline calls
    ``.search(model, text_emb, trigger, selected_components, ranked_flips,
    quant_dict, eval_imgs, eval_labels, grad_imgs, original_weights, cfg,
    tensor_refs=..., layer_R=..., sub_quant=..., C_astar=...)`` and receives
    ``(flip_log, checkpoints, metrics)``.

    ``ranked_flips`` (Phase 3 output) and ``C_astar`` are accepted for
    signature compatibility and ignored: CBS computes its own gradients.
    """

    def __init__(self, cfg: Any = None, ranker: Any = None,
                 evaluator: Any = None, settings: Optional[CBSSettings] = None,
                 **settings_overrides):
        self.cfg = cfg
        self.ranker = ranker          # unused; kept for interface parity
        self.evaluator = evaluator
        self.settings = (settings or
                         CBSSettings.from_cfg(cfg, **settings_overrides))

    # ------------------------------------------------------------------
    #  Wiring helpers
    # ------------------------------------------------------------------
    @classmethod
    def attach(cls, pipeline: Any, **kwargs) -> "ProFlipCBSSearcher":
        """Replace ``pipeline.searcher`` in place (no source edits)."""
        searcher = cls(pipeline.cfg, ranker=getattr(pipeline, "ranker", None),
                       evaluator=getattr(pipeline, "evaluator", None),
                       **kwargs)
        pipeline.searcher = searcher
        return searcher

    # ------------------------------------------------------------------
    #  Pipeline entry point
    # ------------------------------------------------------------------
    def search(self, model, text_emb, trigger, selected_heads,
               ranked_flips, quant_dict,
               eval_imgs, eval_labels,
               grad_imgs, original_weights, cfg,
               tensor_refs=None, layer_R=None, sub_quant=None, C_astar=None):
        """Same signature as ``GreedySearcher.search`` → ``critical_bit_search``."""
        return self.critical_bit_search(
            model, text_emb, trigger, selected_heads, ranked_flips, quant_dict,
            eval_imgs, eval_labels, grad_imgs, original_weights, cfg,
            tensor_refs=tensor_refs, layer_R=layer_R, sub_quant=sub_quant,
            C_astar=C_astar)

    # ------------------------------------------------------------------
    #  Main CBS loop
    # ------------------------------------------------------------------
    def critical_bit_search(self, model, text_emb, trigger, selected_heads,
                            ranked_flips, quant_dict,
                            eval_imgs, eval_labels,
                            grad_imgs, original_weights, cfg,
                            tensor_refs=None, layer_R=None, sub_quant=None,
                            C_astar=None):
        s = self.settings
        tc = int(cfg.TARGET_CLS)
        ap = get_apply_params(cfg, model)
        eval_bs = int(getattr(cfg, "EVAL_BS", 32))

        if tensor_refs is None:
            if selected_heads is None:
                raise ValueError("ProFlipCBSSearcher needs selected_heads or "
                                 "tensor_refs (Phase-2 components).")
            tensor_refs, _ = setup_weight_tracking(model, selected_heads)
        if sub_quant is None:
            if selected_heads is None:
                raise ValueError("ProFlipCBSSearcher needs selected_heads or "
                                 "sub_quant (Phase-0 int8 scales).")
            sub_quant = build_sub_quant(model, selected_heads, quant_dict)
        if not tensor_refs:
            raise ValueError("ProFlipCBSSearcher: empty tensor_refs — no "
                             "attackable component weights.")

        trig = trigger.detach() if torch.is_tensor(trigger) else trigger
        target_vec = text_emb[tc].detach()
        if target_vec.device != eval_imgs.device:
            target_vec = target_vec.to(eval_imgs.device)

        # Auxiliary batch for the CBS loss (ProFlip uses a clean batch of 256;
        # this pipeline defaults to --n-grad=16 — raise it for a closer match).
        aux = grad_imgs if grad_imgs is not None else eval_imgs
        if aux.shape[0] > int(getattr(cfg, "N_GRAD", aux.shape[0])):
            aux = aux[:int(cfg.N_GRAD)]
        if aux.device != eval_imgs.device:
            aux = aux.to(eval_imgs.device)

        with torch.no_grad():
            clean_ref = F.normalize(
                encode_image_batched(model, aux, eval_bs), dim=-1).clone()
            emb_po = F.normalize(
                encode_image_batched(model, apply_trigger(aux, trig, ap),
                                     eval_bs), dim=-1)
        cos_prev = (emb_po @ target_vec).detach().cpu()
        clean_prev = clean_ref.clone()

        asr, cacc = self._quick_eval(
            model, text_emb, trigger, eval_imgs, eval_labels, tc, ap, eval_bs)
        asr_base, cacc_base = asr, cacc

        n_params = len(tensor_refs)
        print("\n" + "=" * 60)
        print("PHASE 4: PROFLIP CRITICAL BIT SEARCH (CBS) — CLIP ADAPTATION")
        print("=" * 60)
        print(f"  component tensors : {n_params} (Phase-2 pool)")
        print(f"  aux batch         : {aux.shape[0]} images (grad_imgs)")
        print(f"  objective         : -cos(f(triggered), t_target)"
              + (f" + {s.clean_mse_weight:g}·MSE(clean, clean_ref)"
                 if s.clean_mse_weight else ""))
        print(f"  S3 grid           : K={s.grid_k} over [-R,R] (int8 codes)")
        print(f"  Q_p mode          : {s.q_mode}   monotone={s.monotone}")
        print(f"  stop              : ASR>={s.asr_target}% or n_bits>={s.nmax}")
        print(f"  Baseline          : ASR={asr:.1f}%  CACC={cacc:.1f}%")

        flip_log: List[Dict[str, Any]] = []
        round_log: List[Dict[str, Any]] = []
        checkpoints: List[Tuple[int, float, float]] = [(0, asr, cacc)]
        state_by_nb: Dict[int, Tuple[float, float]] = {0: (asr, cacc)}
        flips_to_95: Optional[int] = 0 if asr >= 95.0 - 1e-6 else None
        flips_to_94: Optional[int] = 0 if asr >= 94.0 - 1e-6 else None
        best_asr = asr
        n_bits = 0
        n_elements = 0
        rid = 0
        last_ckpt_bucket = 0
        stop_reason = "max_flips"
        banned = set()          # (tensor_key, row, col) — only until next flip
        t0 = time.time()

        while True:
            if asr >= s.asr_target - 1e-6:
                stop_reason = "asr_target"
                break
            if n_bits >= s.nmax:
                stop_reason = "max_flips"
                break

            # ── S1: one backward, fitness for every component tensor ──
            fitness = self._fitness_maps(
                model, trig, target_vec, aux, clean_ref, ap, s,
                tensor_refs, sub_quant)

            order = sorted(
                [(k, v) for k, v in fitness.items() if v["Fmax"] > 0.0],
                key=lambda kv: kv[1]["Fmax"], reverse=True)

            applied: Optional[Tuple[Tuple[int, int, str], Dict[str, Any]]] = None
            for key, info in order[:max(1, s.max_param_tries)]:
                Fmap = info["F"]
                if Fmap.numel() == 0:
                    continue
                ncols = Fmap.shape[1]
                flat = Fmap.flatten()
                k = min(max(1, s.max_elem_tries), flat.numel())
                top_idx = torch.topk(flat, k).indices.tolist()
                for idx in top_idx:
                    if float(flat[idx].item()) <= 0.0:
                        break              # topk is descending
                    r, c = divmod(int(idx), ncols)
                    if (key, r, c) in banned:
                        continue
                    cand = self._try_element(
                        model, trig, target_vec, aux, clean_ref, ap, s,
                        key, r, c, info, tensor_refs, sub_quant)
                    if cand is None:
                        banned.add((key, r, c))
                        continue
                    applied = (key, cand)
                    break
                if applied is not None:
                    break

            if applied is None:
                stop_reason = "no_candidates"
                print("  No element yields a bit-changing value — stopping.")
                break

            key, cand = applied
            layer, head, tensor = key
            param = tensor_refs[key][0]
            scale = float(cand["scale"])
            param.data[cand["abs_r"], cand["abs_c"]] = cand["new"] * scale
            banned.clear()

            # ── round-level telemetry (CBS loss, cos gain, clean drift) ──
            with torch.no_grad():
                emb_new = F.normalize(
                    encode_image_batched(
                        model, apply_trigger(aux, trig, ap), eval_bs), dim=-1)
                cos_new = (emb_new @ target_vec).detach().cpu()
                clean_now = F.normalize(
                    encode_image_batched(model, aux, eval_bs), dim=-1)

            delta_cos = float((cos_new - cos_prev).mean().item())
            n_improved = int((cos_new > cos_prev).sum().item())
            drift = float((1.0 - F.cosine_similarity(
                clean_prev, clean_now, dim=-1)).mean().item())
            cos_prev, clean_prev = cos_new, clean_now

            asr, cacc = self._quick_eval(
                model, text_emb, trigger, eval_imgs, eval_labels, tc, ap,
                eval_bs)

            bits = list(cand["bits"])
            n_this = len(bits)
            n_bits += n_this
            n_elements += 1
            for b in bits:
                flip_log.append(dict(
                    round=rid, tensor=tensor, layer=int(layer), head=int(head),
                    component=_component_type(tensor),
                    position=(int(cand["abs_r"]), int(cand["abs_c"])),
                    bit=int(b),
                    old_int8=int(cand["old"]), new_int8=int(cand["new"]),
                    old_value=float(cand["old_value"]),
                    delta_cos=delta_cos, drift=drift, n_improved=n_improved,
                    score=delta_cos, phase="cbs", layer_weight_R=1.0,
                    cbs_round=rid, cbs_element_bits=bits,
                    cbs_n_bits=n_this, cbs_atomic=True))

            round_log.append(dict(
                round=rid, layer=int(layer), head=int(head), tensor=tensor,
                component=_component_type(tensor),
                row=int(cand["abs_r"]), col=int(cand["abs_c"]),
                old_int8=int(cand["old"]), new_int8=int(cand["new"]),
                bits=bits, n_bits=n_this, n_total_bits=n_bits,
                loss_before=float(cand["loss_cur"]),
                loss_after=float(cand["loss_new"]),
                delta_cos=delta_cos, drift=drift, n_improved=n_improved,
                asr=asr, cacc=cacc,
                fitness=float(cand.get("fitness", 0.0))))

            state_by_nb[n_bits] = (asr, cacc)

            bucket = n_bits // max(1, s.checkpoint_every)
            if bucket > last_ckpt_bucket:
                checkpoints.append((n_bits, asr, cacc))
                last_ckpt_bucket = bucket
                print(f"  [ckpt {n_bits}] ASR={asr:.1f}%  CACC={cacc:.1f}%")
            if flips_to_94 is None and asr >= 94.0 - 1e-6:
                flips_to_94 = n_bits
                print(f"  Flips→94%: reached after {n_bits} bit(s)")
            if flips_to_95 is None and asr >= 95.0 - 1e-6:
                flips_to_95 = n_bits
                print(f"  Flips→95%: reached after {n_bits} bit(s)")
                if not any(ck[0] == n_bits for ck in checkpoints):
                    checkpoints.append((n_bits, asr, cacc))
            best_asr = max(best_asr, asr)

            tag = (f"L{layer} H{head}" if _component_type(tensor) == "attn"
                   else f"L{layer} FFN")
            print(f"  Rd {rid:>2d}: {tag} {tensor}"
                  f"[{cand['abs_r']},{cand['abs_c']}]  "
                  f"code {cand['old']}→{cand['new']}  bits={n_this} "
                  f"(total {n_bits})  loss {cand['loss_cur']:.4f}→"
                  f"{cand['loss_new']:.4f}  Δcos={delta_cos:+.4f}  "
                  f"drift={drift:.4f}  ASR={asr:.1f}%")
            rid += 1

        # ── final state (no rollback: faithful CVD/ProFlip has none) ──
        if not any(ck[0] == n_bits for ck in checkpoints):
            checkpoints.append((n_bits, asr, cacc))

        def _state_at(k: int) -> Tuple[Tuple[float, float], bool]:
            """(state, carried) — exact, else first recorded state >= k."""
            if k in state_by_nb:
                return state_by_nb[k], False
            higher = [n for n in state_by_nb if n >= k]
            if higher:
                return state_by_nb[min(higher)], False
            return (asr, cacc), True

        (a10, c10), f10 = _state_at(10)
        (a20, c20), f20 = _state_at(20)
        carried = bool(f10 or f20)

        metrics = dict(
            asr_at_10=a10, cacc_at_10=c10,
            asr_at_20=a20, cacc_at_20=c20,
            flips_to_95=flips_to_95,
            flips_to_94=flips_to_94,
            flips_total=n_bits,
            flips_fine=0,
            flips_rolled_back=0,
            asr_final=asr, cacc_final=cacc,
            asr_best=best_asr,
            asr_base=asr_base, cacc_base=cacc_base,
            normal_stop=float(getattr(cfg, "NORMAL_STOP", 94.0)),
            fine_target=float(getattr(cfg, "FINE_TARGET", 99.0)),
            asr_carried_forward=carried,
            stop_reason=stop_reason,
            # ── CBS-specific provenance ──
            cbs_method="ProFlip-CBS (CLIP adaptation, shared trigger)",
            cbs_asr_target=s.asr_target,
            cbs_nmax=s.nmax,
            cbs_grid_k=s.grid_k,
            cbs_q_mode=s.q_mode,
            cbs_monotone=s.monotone,
            cbs_clean_mse_weight=s.clean_mse_weight,
            cbs_elements_total=n_elements,
            cbs_bits_total=n_bits,
            cbs_avg_bits_per_element=(n_bits / n_elements if n_elements else 0.0),
            cbs_aux_images=int(aux.shape[0]),
            n_eval=int(eval_imgs.size(0)),
            cbs_state_checkpoints=sorted(state_by_nb.keys()),
            cbs_rounds=round_log,
            cbs_elapsed_sec=time.time() - t0,
            cbs_checkpoint_semantics=(
                "asr_at_k = first recorded state with cumulative bits >= k "
                "(atomic multi-bit element rounds)"))

        print(f"\n  CBS elements: {n_elements}   bits: {n_bits}   "
              f"(stop: {stop_reason})")
        print(f"  Final ASR   : {asr:.1f}%  (Δ = {asr - asr_base:+.1f}%, "
              f"best = {best_asr:.1f}%)")
        print(f"  Final CACC  : {cacc:.1f}%  (Δ = {cacc - cacc_base:+.1f}%)")
        print(f"  ASR@10={metrics['asr_at_10']}  ASR@20={metrics['asr_at_20']}"
              f"{'  (carried forward)' if carried else ''}  "
              f"Flips→94%={flips_to_94}  Flips→95%={flips_to_95}")
        return flip_log, checkpoints, metrics

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------
    def _quick_eval(self, model, text_emb, trigger, imgs, labels,
                    target_cls, ap, eval_bs):
        if self.evaluator is not None:
            return self.evaluator.quick_evaluate(
                model, text_emb, trigger, imgs, labels, target_cls, ap, eval_bs)
        with torch.no_grad():
            emb_cl = F.normalize(
                encode_image_batched(model, imgs, eval_bs), dim=-1)
            cacc = float(((emb_cl @ text_emb.T).argmax(1) == labels)
                         .float().mean().item() * 100)
            if trigger is None:
                return 0.0, cacc
            emb_po = F.normalize(encode_image_batched(
                model, apply_trigger(imgs, trigger, ap), eval_bs), dim=-1)
            asr = float(((emb_po @ text_emb.T).argmax(1) == target_cls)
                        .float().mean().item() * 100)
        return asr, cacc

    # ------------------------------------------------------------------
    #  CBS objective
    # ------------------------------------------------------------------
    def _objective(self, model, trig, target_vec, aux, clean_ref, ap, s):
        with torch.no_grad():
            emb = F.normalize(encode_image_batched(
                model, apply_trigger(aux, trig, ap)), dim=-1)
            loss = -F.cosine_similarity(
                emb, target_vec.unsqueeze(0), dim=-1).mean()
            if s.clean_mse_weight:
                emb_c = F.normalize(
                    encode_image_batched(model, aux), dim=-1)
                loss = loss + s.clean_mse_weight * F.mse_loss(emb_c, clean_ref)
        return float(loss.item())

    # ------------------------------------------------------------------
    #  S1/S2: fitness maps (one backward pass)
    # ------------------------------------------------------------------
    def _fitness_maps(self, model, trig, target_vec, aux, clean_ref, ap, s,
                      tensor_refs, sub_quant):
        """F(p, elem) = |dL/dp| * step,  step = Q_p - elem if grad < 0 else 0.

        Returns {tensor_key: dict(F=<2-D fitness>, Fmax=float, scale=float)}.
        """
        params: Dict[int, torch.nn.Parameter] = {}
        for key, (p, _rs, _cs) in tensor_refs.items():
            params[id(p)] = p
        orig_rg = {pid: p.requires_grad for pid, p in params.items()}

        for p in params.values():
            p.requires_grad_(False)
            p.grad = None
        for key, (p, _rs, _cs) in tensor_refs.items():
            p.requires_grad_(True)

        try:
            model.zero_grad(set_to_none=True)
            bs = max(1, int(s.grad_bs))
            n = aux.shape[0]
            with torch.enable_grad():
                for i in range(0, n, bs):
                    chunk = aux[i:i + bs]
                    poisoned = apply_trigger(chunk, trig, ap)
                    emb = F.normalize(encode_image(model, poisoned), dim=-1)
                    loss = -F.cosine_similarity(
                        emb, target_vec.unsqueeze(0), dim=-1).mean()
                    if s.clean_mse_weight:
                        emb_c = F.normalize(encode_image(model, chunk), dim=-1)
                        loss = loss + s.clean_mse_weight * F.mse_loss(
                            emb_c, clean_ref[i:i + chunk.shape[0]])
                    loss.backward()

            out: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
            for key, (p, rs, cs) in tensor_refs.items():
                if p.grad is None:
                    continue
                g = p.grad[rs, cs].detach()
                scale = float(sub_quant[key]["scale"])
                if scale == 0.0:
                    continue
                # The parameter stores DEQUANTIZED values (w = code * scale),
                # so dL/dcode = dL/dw * scale.  ProFlip's fitness is defined on
                # the quantized integer weights; without this conversion the
                # fitness would be off by 1/scale and S1 could not compare
                # tensors with different Phase-0 scales fairly.
                g_code = g * scale
                w = torch.round(p.data[rs, cs].detach() / scale)
                w = w.clamp(-128, 127).to(torch.int64)
                if s.q_mode == "signed":
                    qp = int(w.max().item())
                else:                                   # paper: max |W|
                    qp = int(w.abs().max().item())
                qp = max(qp, 0)
                step = torch.where(
                    g_code < 0,
                    (qp - w).clamp(min=0).to(g_code.dtype),
                    torch.zeros_like(g_code))
                Fmap = torch.nan_to_num(g_code.abs() * step, nan=0.0,
                                        posinf=0.0, neginf=0.0)
                out[key] = dict(F=Fmap, Fmax=float(Fmap.max().item()),
                                scale=scale)
            return out
        finally:
            for pid, p in params.items():
                p.requires_grad_(orig_rg[pid])
                p.grad = None

    # ------------------------------------------------------------------
    #  S2/S3: element + optimal value inside one parameter
    # ------------------------------------------------------------------
    def _try_element(self, model, trig, target_vec, aux, clean_ref, ap, s,
                     key, r, c, info, tensor_refs, sub_quant):
        """Grid-search the value of element (r, c); return an update or None.

        The update is atomic: all bits between the old and the new int8 code
        are flipped together, and the reported cost is their Hamming distance.
        """
        param, rs, cs = tensor_refs[key]
        scale = float(info["scale"])
        abs_r = int(rs.start or 0) + int(r)
        abs_c = int(cs.start or 0) + int(c)

        w = torch.round(param.data[rs, cs].detach() / scale)
        w = w.clamp(-128, 127).to(torch.int64)
        old = int(w[r, c].item())
        big_r = int(w.abs().max().item())
        if big_r <= 0:
            return None

        old_value = float(param.data[abs_r, abs_c].item())

        # `include_current` makes "no improvement" detectable (released ProFlip
        # does not compare against the current value and can stall).
        try:
            param.data[abs_r, abs_c] = old_value
            loss_cur = self._objective(
                model, trig, target_vec, aux, clean_ref, ap, s)
            if not math.isfinite(loss_cur):
                return None
            best_loss, best_code = ((loss_cur, old) if s.include_current
                                    else (None, None))

            # S3 grid: K points over [-R, R] on the int8 code axis (ProFlip
            # uses the quantized integer weights directly, not floats).
            grid = torch.linspace(-big_r, big_r, int(s.grid_k))
            grid = grid.round().clamp(-128, 127).to(torch.int64).unique()
            if s.monotone:
                grid = grid[grid >= old]

            for code_t in grid.tolist():
                code = int(code_t)
                if code == old:
                    continue
                param.data[abs_r, abs_c] = code * scale
                loss = self._objective(
                    model, trig, target_vec, aux, clean_ref, ap, s)
                if not math.isfinite(loss):
                    continue
                if best_loss is None or loss < best_loss - 1e-12:
                    best_loss, best_code = loss, code
        finally:
            param.data[abs_r, abs_c] = old_value   # always restore

        if best_code is None or int(best_code) == old:
            return None
        bits = _bit_positions(old, int(best_code))
        if not bits:
            return None
        return dict(abs_r=abs_r, abs_c=abs_c, r=int(r), c=int(c),
                    old=old, new=int(best_code), bits=bits, scale=scale,
                    old_value=old_value, loss_cur=float(loss_cur),
                    loss_new=float(best_loss),
                    fitness=float(info["F"][r, c].item()))


# ══════════════════════════════════════════════════════════════════════
#  OPTIONAL: skip the (unused) hybrid Phase 3 when running CBS
# ══════════════════════════════════════════════════════════════════════
class NoOpRanker:
    """Cheap stand-in for ``HybridRanker``: CBS ignores Phase-3 output."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def rank(self, *args, **kwargs):
        print("\n" + "=" * 60)
        print("PHASE 3: SKIPPED (ProFlip-CBS computes its own gradients)")
        print("=" * 60)
        return []

    def rank_fallback(self, *args, **kwargs):
        return []


# ══════════════════════════════════════════════════════════════════════
#  RUNNER (no edits to the proposed-method code)
# ══════════════════════════════════════════════════════════════════════
def run_proflip(cfg, settings: Optional[CBSSettings] = None,
                suffix: str = "__cbs", skip_phase3: bool = True):
    """Build the pipeline for ``cfg``, swap in CBS, run, return ``save``."""
    from pipeline import AttackPipeline

    s = settings or CBSSettings.from_cfg(cfg)
    for k, v in s.as_cfg_attrs().items():
        setattr(cfg, k, v)

    # Keep CBS runs out of the hybrid method's output dirs / aggregate groups.
    orig_variant = str(getattr(cfg, "VARIANT", "full"))
    cfg.VARIANT = f"{orig_variant}_cbs"
    if suffix and not str(cfg.OUT_DIR).endswith(suffix):
        cfg.OUT_DIR = f"{cfg.OUT_DIR}{suffix}"
    if suffix and not str(cfg.RUN_ID).endswith(suffix):
        cfg.RUN_ID = f"{cfg.RUN_ID}{suffix}"

    pipe = AttackPipeline(cfg)
    if skip_phase3:
        pipe.ranker = NoOpRanker(cfg)
    ProFlipCBSSearcher.attach(pipe, settings=s)
    print(f"\n[CBS] ProFlip Critical Bit Search attached "
          f"(K={s.grid_k}, target={s.asr_target}%, nmax={s.nmax})")
    return pipe.run()


# ══════════════════════════════════════════════════════════════════════
#  CLI: pass every experiment.py argument through, plus --cbs-* knobs
# ══════════════════════════════════════════════════════════════════════
def _parse_cbs_cli(argv: List[str]):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--cbs-k", type=int, default=20,
                   help="S3 grid points (ProFlip paper: 20)")
    p.add_argument("--cbs-asr-target", type=float, default=94.0,
                   help="stop ASR%% (ProFlip paper: 94)")
    p.add_argument("--cbs-nmax", type=int, default=100,
                   help="max accepted bit flips (ProFlip paper: 100)")
    p.add_argument("--cbs-clean-mse-weight", type=float, default=0.0,
                   help="optional clean-feature MSE stealth term (0 = pure "
                        "ProFlip target objective)")
    p.add_argument("--cbs-q-mode", choices=["abs", "signed"], default="abs",
                   help="Q_p: paper max|w| (abs) or released code max(w)")
    p.add_argument("--cbs-no-monotone", action="store_true",
                   help="allow decreasing an element (released code forbids)")
    p.add_argument("--cbs-suffix", default="__cbs",
                   help="output-dir/RUN_ID suffix to avoid hybrid collisions")
    p.add_argument("--cbs-keep-phase3", action="store_true",
                   help="also run the unused hybrid Phase-3 ranking")
    ns, rest = p.parse_known_args(argv)
    return ns, rest


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cbs_cli, rest = _parse_cbs_cli(argv)
    sys.argv = [sys.argv[0]] + rest          # experiment.py parses the rest

    import experiment
    from main import is_oom, record_error, record_oom
    from utils import cleanup_gpu

    args = experiment.parse_args()
    cfg = experiment.build_config(args)

    overrides = dict(
        grid_k=cbs_cli.cbs_k,
        asr_target=cbs_cli.cbs_asr_target,
        nmax=cbs_cli.cbs_nmax,
        clean_mse_weight=cbs_cli.cbs_clean_mse_weight,
        q_mode=cbs_cli.cbs_q_mode,
        monotone=not cbs_cli.cbs_no_monotone,
    )
    settings = CBSSettings.from_cfg(cfg, **overrides)

    try:
        result = run_proflip(
            cfg, settings=settings, suffix=cbs_cli.cbs_suffix,
            skip_phase3=not cbs_cli.cbs_keep_phase3)
        if not args.no_aggregate:
            experiment.refresh_summaries(cfg.OUT_ROOT)
        return 0 if result is not None else 1
    except Exception as exc:                  # same contract as main.py
        cleanup_gpu()
        if is_oom(exc):
            record_oom(cfg)
            print("\n  OOM — run skipped and recorded in oom_pending.json")
        else:
            record_error(cfg, exc)
            print(f"\n  ERROR — run skipped and recorded: "
                  f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
