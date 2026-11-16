"""
pipeline.py — Orchestrator: wires all classes together in order.
================================================================
Phase 0 (Quantizer) → 1 (TriggerGenerator) → 2A (LayerAblator)
→ 2B (ComponentAblator) → weight tracking (WeightTracker)
→ 3 (HybridRanker) → 4 (GreedySearcher) → final evaluation
→ artefacts (ResultWriter).   (Synced with the reference
main_new_pipeline.py, which no longer runs post-attack analysis;
analysis.py is kept as an optional, unwired module.)

Every phase is exactly one method call with explicit inputs/outputs —
phase modules never import each other (GreedySearcher receives its
ranker/evaluator by injection).  This is also where per-phase ablation
flags will gate calls in the grid step.

Memory management (preserved from main_new_pipeline.py):
  1. Batch-wise processing in Phases 2, 3, 4
  2. Deferred test_loader creation (Phase 5 only)
  3. Aggressive tensor cleanup between phases
  4. Reduced dataset sizes (N_CALIB=32, N_GRAD=16, N_EVAL=200)
  5. Gradient accumulation for all gradient-based phases

run() returns the bfa_results dict (same schema as the monolith).
"""

import os
import time

import torch

from ablation import ComponentAblator, LayerAblator, enumerate_layer_components
from adapters import get_adapter
from data import DataModule
from evaluation import Evaluator
from hybrid_ranking import HybridRanker
from models import ModelLoader
from quantizer import Quantizer
from results import ResultWriter
from search import GreedySearcher
from target import TargetEmbedder
from trigger import TriggerGenerator, get_apply_params, get_trigger_token_indices
from utils import cleanup_gpu, set_seed
from visualization import Visualizer
from weights import WeightTracker


class AttackPipeline:
    """Runs the full hybrid bit-flip attack end to end."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_loader = ModelLoader(cfg)
        self.data = DataModule(cfg)
        self.target = TargetEmbedder(cfg)
        self.quantizer = Quantizer(cfg)
        self.trigger_gen = TriggerGenerator(cfg)
        self.layer_ablator = LayerAblator(cfg)
        self.component_ablator = ComponentAblator(cfg)
        self.weights = WeightTracker(cfg)
        self.ranker = HybridRanker(cfg)
        self.evaluator = Evaluator(cfg)
        self.searcher = GreedySearcher(cfg, ranker=self.ranker,
                                       evaluator=self.evaluator)
        self.viz = Visualizer(cfg)
        self.results = ResultWriter(cfg)

    # ════════════════════════════════════════════════════════════════
    #  PHASE 1 — TRIGGER GENERATION WITH DEAD-TRIGGER RETRIES
    # ════════════════════════════════════════════════════════════════
    def _generate_trigger(self, model, text_emb, t_target, calib_loader,
                          target_images, eval_imgs, eval_labels, ap, cfg):
        """Phase 1 with dead-trigger retries: if the trigger's standalone ASR
        is below cfg.TRIGGER_MIN_ASR, retrain it with a fresh P1 seed (up to
        cfg.TRIGGER_RETRIES times).  Keeps seed-dependent Phase-1 collapses
        (e.g. Caltech s42 → 0% trigger ASR → dead plateau) from killing a run.

        Returns (trigger, asr_pre, cacc_pre, attempts)."""
        trigger = None
        asr_pre = cacc_pre = 0.0
        attempts = 0
        for attempt in range(cfg.TRIGGER_RETRIES + 1):
            attempts = attempt + 1
            trigger = self.trigger_gen.generate(
                model, text_emb, t_target, calib_loader, target_images, cfg)
            asr_pre, cacc_pre = self.evaluator.quick_evaluate(
                model, text_emb, trigger, eval_imgs, eval_labels,
                cfg.TARGET_CLS, ap, cfg.MINI_BS)
            print(f"  Trigger standalone: ASR={asr_pre:.1f}%  "
                  f"CACC={cacc_pre:.1f}%"
                  + (f"  (attempt {attempts})" if attempts > 1 else ""))
            if asr_pre >= cfg.TRIGGER_MIN_ASR or attempt == cfg.TRIGGER_RETRIES:
                break
            retry_seed = cfg.SEED + 1000 * (attempt + 1)
            print(f"  Trigger below floor ({cfg.TRIGGER_MIN_ASR}%) — "
                  f"retrying Phase 1 with seed {retry_seed} "
                  f"({attempt + 1}/{cfg.TRIGGER_RETRIES})")
            set_seed(retry_seed)
        return trigger, asr_pre, cacc_pre, attempts

    # ════════════════════════════════════════════════════════════════
    #  MAIN RUN
    # ════════════════════════════════════════════════════════════════
    def run(self):
        cfg = self.cfg
        set_seed(getattr(cfg, "SEED", 42))
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        dev = cfg.DEVICE

        print("=" * 60)
        print("HYBRID BIT-FLIP ATTACK ON CLIP")
        print("Phase 0 → 1 → 2 → 3 → 4 → 5")
        print("=" * 60)
        print(f"  Device  : {dev}")
        print(f"  Target  : class {cfg.TARGET_CLS}")
        print(f"  Trigger : {cfg.TRIGGER_TYPE}  {cfg.TRIG_SZ}×{cfg.TRIG_SZ}  "
              f"loc={cfg.TRIGGER_LOCATION or 'legacy'}")

        # ── Load model ──
        print("\n[1] Loading model")
        model, preprocess = self.model_loader.load()
        ap = get_apply_params(cfg, model)

        # ── Auto-detect actual model dimensions ──
        mparams = self.model_loader.detect_model_params(model)
        cfg.PATCH_SIZE = mparams["patch_size"]
        cfg.IMG_SIZE = mparams["img_size"]

        print(f"  Config requested : {cfg.MODEL}")
        print(f"  Actual architecture:")
        print(f"    seq_len  : {mparams['seq_len']}  "
              f"(= {mparams['patches_per_side']}×{mparams['patches_per_side']} patches + CLS)")
        print(f"    width    : {mparams['width']}")
        print(f"    patch    : {mparams['patch_size']}×{mparams['patch_size']}")
        print(f"    img_size : {mparams['img_size']}")
        print(f"    layers   : {mparams['n_layers']}")
        print(f"    heads    : {mparams['n_heads']}")
        print(f"    params   : {sum(p.numel() for p in get_adapter(model).get_vision_module(model).parameters()) / 1e6:.1f}M")

        self.model_loader.sanity_check(mparams)

        # ── Load dataset ──
        print(f"[2] Loading {cfg.DATASET_NAME}")
        ds, classes = self.data.load_dataset(preprocess)
        print(f"  {len(classes)} classes, target = '{classes[cfg.TARGET_CLS]}'")

        # ── Text embeddings ──
        text_emb = self.target.build_text_embeddings(model, classes, dev)

        # ── Data splits ──
        print("\n[3] Preparing data splits")
        target_sel, calib_idx, grad_idx, eval_idx, test_idx, target_holdout = \
            self.data.prepare_data_splits(ds)

        target_images, _ = self.data.load_tensors(ds, target_sel, dev)
        calib_loader = self.data.make_loader(ds, calib_idx, cfg.BATCH, shuffle=True,
                                             workers=cfg.WORKERS)
        calib_imgs, _ = self.data.load_tensors(ds, calib_idx, dev)
        eval_imgs, eval_labels = self.data.load_tensors(ds, eval_idx, dev)

        # ── grad_imgs on CPU, transferred to GPU when needed ──
        grad_imgs_cpu, _ = self.data.load_tensors(ds, grad_idx, "cpu",
                                                  batch_size=cfg.BATCH, workers=cfg.WORKERS)

        # ── t_target ──
        t_target = self.target.from_text(cfg.TARGET_CLS, text_emb)

        print(f"  t_target = text_emb[{cfg.TARGET_CLS}]"
              f"  ('{classes[cfg.TARGET_CLS]}')")

        trigger_token_indices = get_trigger_token_indices(cfg, model=model)
        print(f"  Trigger tokens: {trigger_token_indices}")
        print(f"  Trigger type: {cfg.TRIGGER_TYPE}  "
              f"Size: {cfg.TRIG_SZ}×{cfg.TRIG_SZ}  "
              f"Placement: {ap.loc or 'legacy'}")

        # ═══ PHASE 0 — Quantisation ═══
        quant_dict = self.quantizer.quantize(model, text_emb, cfg)
        cleanup_gpu()

        # ═══ PHASE 1 — Trigger Generation (with dead-trigger retries) ═══
        trigger, asr_pre, cacc_pre, trigger_attempts = self._generate_trigger(
            model, text_emb, t_target, calib_loader, target_images,
            eval_imgs, eval_labels, ap, cfg)

        self.viz.save_trigger_png(trigger,
                                  os.path.join(cfg.OUT_DIR, "trigger_patch.png"),
                                  bg_image=eval_imgs[0],
                                  ap=ap)

        del target_images
        cleanup_gpu()

        # ═══ PHASE 2A — Layer-Level Causal Ablation ═══
        if getattr(cfg, "SKIP_PHASE2", False):
            # variant 'a': no layer/component pruning (Phase-2 efficiency ablation)
            print("  Variant 'a': Phase 2 skipped — all components of all layers")
            layer_scores, top_k_layers = {}, set(range(mparams["n_layers"]))
            selected_components = enumerate_layer_components(model, top_k_layers)
            layer_R = layer_scores
        else:
            layer_scores, top_k_layers = self.layer_ablator.ablate(
                model, trigger, t_target, calib_imgs, cfg)
            layer_R = layer_scores
            self.viz.plot_ablation_scores(layer_scores, cfg, top_k_layers=top_k_layers)
            cleanup_gpu()

            # ═══ PHASE 2B — Component-Level Causal Ablation ═══
            component_scores, selected_components = self.component_ablator.ablate(
                model, trigger, t_target, calib_imgs, top_k_layers, cfg)

        # ═══ Weight tracking (from selected components) ═══
        tensor_refs, original_weights = self.weights.setup(
            model, selected_components)
        print(f"\n  Weight tracking: {len(tensor_refs)} tensors, "
              f"{len(original_weights)} originals saved")
        sub_quant = self.weights.build_sub_quant(
            model, selected_components, quant_dict)

        # ═══ PHASE 3 — Hybrid Bit Ranking ═══
        grad_imgs = grad_imgs_cpu.to(dev)
        ranked_flips = self.ranker.rank(
            model, trigger, t_target, layer_R,
            selected_components, tensor_refs, grad_imgs, cfg,
            sub_quant=sub_quant)

        # ── Save clean model state before bit flips ──
        model_state_clean = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
        print(f"\n  Saved clean model state "
              f"({sum(v.numel() * v.element_size() for v in model_state_clean.values()) / 1e6:.0f} MB)")

        # ═══ PHASE 4 — Greedy Bit-Flip Search ═══
        t0 = time.time()
        flip_log, checkpoints, metrics = self.searcher.search(
            model, text_emb, trigger, selected_components,
            ranked_flips, quant_dict,
            eval_imgs, eval_labels,
            grad_imgs, original_weights, cfg,
            tensor_refs=tensor_refs,
            layer_R=layer_R, sub_quant=sub_quant)
        search_time = time.time() - t0

        # ── Save attacked model state (flagged: for defence / VLM-transfer
        #    evaluation; grids enable it selectively) ──
        model_state_attacked_path = None
        if getattr(cfg, "SAVE_ATTACKED_MODEL", False):
            model_state_attacked_path = os.path.join(
                cfg.OUT_DIR, "model_state_attacked.pt")
            torch.save({k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()},
                       model_state_attacked_path)
            print(f"\n  Attacked model state saved → "
                  f"{model_state_attacked_path}")

        del quant_dict, grad_imgs
        cleanup_gpu()

        # ═══ PHASE 5 — Post-Attack Analysis ═══
        # (Not part of the current reference pipeline — analysis.py is kept
        #  as an optional module and is intentionally not called here.)

        # ── Final full evaluation ──
        test_loader = self.data.make_loader(ds, test_idx, 64, workers=cfg.WORKERS)
        r_final = self.evaluator.full_evaluate(
            model, text_emb, trigger,
            test_loader,
            cfg.TARGET_CLS, ap, dev)

        # ── Linear-probe evaluation (--task lp, TASKS.md §2) ──
        lp_result = None
        if cfg.TASK == "lp":
            print("\n[LP] Linear-probe evaluation")
            print(f"  probe head: {cfg.LP_HEAD}")
            lp_result = self.evaluator.lp_evaluate(
                model, test_loader, dev, cfg.LP_HEAD,
                trigger=trigger, target_cls=cfg.TARGET_CLS, ap=ap,
                mini_bs=cfg.MINI_BS)
            print(f"  LP-CACC={lp_result['lp_cacc']:.2f}%  "
                  f"LP-ASR={lp_result['lp_asr']:.2f}%")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        n_flips = len(flip_log)
        print(f"  Model   : {cfg.MODEL}  (actual: {mparams['width']}d, "
              f"{mparams['n_layers']}L, {mparams['n_heads']}H, "
              f"patch={mparams['patch_size']})")
        print(f"  Method  : Hybrid (Layer_R → Scoped Gradient → R(l) × Gradient)")
        print(f"  Flips   : {n_flips}")
        print(f"  ASR     : {r_final['asr']:.1f}%")
        print(f"  CACC    : {r_final['cacc']:.1f}%")
        print(f"  Search  : {search_time:.1f}s")

        sample_for_vis = eval_imgs[:10]
        sample_labels_vis = eval_labels[:10]
        self.viz.create_composite_png(
            trigger, cfg, r_final["asr"], r_final["cacc"],
            sample_for_vis, sample_labels_vis, classes,
            os.path.join(cfg.OUT_DIR, "composite.png"), model=model)

        save = {
            "config": {k: v for k, v in vars(cfg).items()
                       if not k.startswith("_")},
            "actual_model_params": mparams,
            "target_class": classes[cfg.TARGET_CLS],
            "trigger_token_indices": trigger_token_indices,
            "layer_R": {str(k): v for k, v in layer_R.items()} if layer_R else {},
            "top_k_layers": sorted(top_k_layers),
            "selected_heads": [
                {k: v for k, v in s.items()
                 if not isinstance(v, torch.Tensor)}
                for s in selected_components],
            "flip_log": flip_log,
            "checkpoints": checkpoints,
            "metrics": metrics,
            "rep": getattr(cfg, "REP", 1),
            "trigger_location": getattr(cfg, "TRIGGER_LOCATION", None),
            "trigger_attempts": trigger_attempts,
            "section": getattr(cfg, "SECTION", None),
            "model_state_attacked": model_state_attacked_path,

            "results": {
                "post_flip_test": {k: v for k, v in r_final.items()
                                   if not isinstance(v, torch.Tensor)}},
            "search_time_sec": search_time}
        if lp_result is not None:
            save["linear_probe"] = lp_result

        self.results.save(save, trigger, original_weights)

        return save
