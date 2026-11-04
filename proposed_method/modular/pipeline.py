"""
pipeline.py — Orchestrator: wires all classes together in order.
=================================================================
Phase 0 (Quantizer) → 1 (TriggerGenerator) → 2 (RoutingAnalyzer)
→ 3 (GradientPrescreener) → 4 (HybridRanker) → 5 (GreedySearcher)
→ 6 (PostAttackAnalyzer)

Memory management:
  1. Batch-wise processing in Phases 2, 3, 4, 5
  2. Deferred test_loader creation (Phase 6 only)
  3. Aggressive tensor cleanup between phases
  4. Reduced dataset sizes (N_CALIB=32, N_GRAD=16, N_EVAL=200)
  5. Gradient accumulation for all gradient-based phases
"""

import os
import json
import time
import torch

from probe_eval import ProbeEvaluator
from target import TargetEmbedder
from utils import set_seed, cleanup_gpu
from data import DataModule
from models import ModelLoader
from trigger import TriggerGenerator, get_trigger_token_indices
from quantizer import Quantizer
from evaluation import Evaluator
from hybrid_ranking import RoutingAnalyzer
from hybrid_ranking import GradientPrescreener
from hybrid_ranking import HybridRanker
from search import GreedySearcher
from analysis import PostAttackAnalyzer
from visualization import Visualizer


class AttackPipeline:
    """Runs the full hybrid bit-flip attack end to end."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_loader = ModelLoader(cfg)
        self.data = DataModule(cfg)
        self.quantizer = Quantizer(cfg)
        self.trigger_gen = TriggerGenerator(cfg)
        self.routing = RoutingAnalyzer(cfg)
        self.prescreener = GradientPrescreener(cfg)
        self.ranker = HybridRanker(cfg)
        self.evaluator = Evaluator(cfg)
        self.searcher = GreedySearcher(cfg, ranker=self.ranker,
                                       evaluator=self.evaluator)
        self.analyzer = PostAttackAnalyzer(cfg)
        self.viz = Visualizer(cfg)

    # ════════════════════════════════════════════════════════════════
    #  MAIN RUN
    # ════════════════════════════════════════════════════════════════
    def run(self):
        cfg = self.cfg
        set_seed(getattr(cfg, "SEED", 42))  # ← FIX 1: per-run seed
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        dev = cfg.DEVICE

        print("=" * 60)
        print("HYBRID BIT-FLIP ATTACK ON CLIP")
        print("Phase 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7")
        print("=" * 60)
        print(f"  Device  : {dev}")
        print(f"  Seed    : {getattr(cfg, 'SEED', 42)}")  # ← show it
        print(f"  Target  : class {cfg.TARGET_CLS}")
        print(f"  Trigger : {cfg.TRIGGER_TYPE}  {cfg.TRIG_SZ}×{cfg.TRIG_SZ}")
        print(f"  Out dir : {cfg.OUT_DIR}")

        # ── Load model ──
        print("\n[1] Loading CLIP")
        model, preprocess = self.model_loader.load()

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
        print(f"    params   : {sum(p.numel() for p in model.visual.parameters()) / 1e6:.1f}M")

        self.model_loader.sanity_check(mparams)

        # ── Load dataset ──
        print(f"\n[2] Loading {cfg.DATASET_NAME}")
        ds, classes = self.data.load_dataset(preprocess)
        print(f"  target = '{classes[cfg.TARGET_CLS]}'")

        # ── Text embeddings + t_target ──
        target_embedder = TargetEmbedder(cfg)
        text_emb = target_embedder.build_text_embeddings(model, classes, dev)

        # ← FIX 2: correct argument order
        t_target = target_embedder.from_text(cfg.TARGET_CLS, text_emb)
        print(f"  t_target = text_emb[{cfg.TARGET_CLS}]"
              f"  ('{classes[cfg.TARGET_CLS]}')")

        # ── Data splits ──
        print("\n[3] Preparing data splits")
        target_sel, calib_idx, grad_idx, eval_idx, test_idx, target_holdout = \
            self.data.prepare_data_splits(ds)

        target_images, _ = self.data.load_tensors(ds, target_sel, dev)
        calib_loader = self.data.make_loader(ds, calib_idx, cfg.BATCH,
                                             shuffle=True, workers=cfg.WORKERS)
        calib_imgs, _ = self.data.load_tensors(ds, calib_idx, dev)
        eval_imgs, eval_labels = self.data.load_tensors(ds, eval_idx, dev)

        # grad_imgs on CPU, transferred to GPU when needed
        grad_imgs_cpu, _ = self.data.load_tensors(ds, grad_idx, "cpu",
                                                  batch_size=cfg.BATCH,
                                                  workers=cfg.WORKERS)

        # future — text + image prototype:
        #   L_trig = L_CE(e^p, T) + λ_img·(1 − cos(e^p, t_img))
        # t_target = target_embedder.from_text_and_image(
        #     cfg.TARGET_CLS, model, target_images,
        #     w_text=0.4, w_img=0.6)

        # ── Model-aware trigger token indices ──
        trigger_token_indices = get_trigger_token_indices(cfg, model=model)
        print(f"  Trigger tokens: {trigger_token_indices}")
        print(f"  Trigger type: {cfg.TRIGGER_TYPE}  "
              f"Size: {cfg.TRIG_SZ}×{cfg.TRIG_SZ}  "
              f"Placement: {'center' if 'center' in cfg.TRIGGER_TYPE else 'corner'}")

        # ═══ PHASE 0 — Quantisation ═══
        quant_dict = self.quantizer.quantize(model, text_emb)

        # ═══ PHASE 1 — Trigger Generation ═══
        trigger = self.trigger_gen.generate(
            model, text_emb, t_target, calib_loader, target_images)

        asr_pre, cacc_pre = self.evaluator.quick_evaluate(
            model, text_emb, trigger, eval_imgs, eval_labels,
            cfg.TARGET_CLS, cfg.TRIG_SZ, cfg.TRIGGER_TYPE, cfg.OPACITY,
            cfg.MINI_BS)
        print(f"  Trigger standalone: ASR={asr_pre:.1f}%  CACC={cacc_pre:.1f}%")

        del target_images
        cleanup_gpu()

        # ═══ PHASE 2 — Routing Efficiency ═══
        layer_R, top_k_layers = self.routing.analyze(
            model, trigger, t_target, calib_imgs)
        cleanup_gpu()

        # ═══ PHASE 3 — Gradient Pre-Screening ═══
        selected_heads = self.prescreener.screen(
            model, trigger, t_target, calib_imgs,
            target_layers=top_k_layers)
        cleanup_gpu()

        # ═══ Weight tracking ═══
        tensor_refs, original_weights = self.ranker.setup_weight_tracking(
            model, selected_heads)
        print(f"\n  Weight tracking: {len(tensor_refs)} tensors, "
              f"{len(original_weights)} originals saved")

        # ═══ PHASE 4 — Hybrid Bit Ranking ═══
        grad_imgs = grad_imgs_cpu.to(dev)
        ranked_flips = self.ranker.rank(
            model, trigger, t_target, layer_R,
            selected_heads, tensor_refs, grad_imgs)

        # ── Save clean model state before bit flips ──
        model_state_clean = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
        print(f"\n  Saved clean model state "
              f"({sum(v.numel() * v.element_size() for v in model_state_clean.values()) / 1e6:.0f} MB)")

        # ═══ PHASE 5 — Greedy Bit-Flip Search ═══
        t0 = time.time()
        flip_log, checkpoints = self.searcher.search(
            model, text_emb, trigger, selected_heads,
            ranked_flips, quant_dict,
            eval_imgs, eval_labels,
            grad_imgs, original_weights,
            tensor_refs=tensor_refs,
            layer_R=layer_R)
        search_time = time.time() - t0

        del quant_dict, grad_imgs
        cleanup_gpu()

        # ═══ PHASE 6 — Post-Attack Analysis ═══
        print("\n" + "=" * 60)
        print("PHASE 6: POST-ATTACK ANALYSIS")
        print("=" * 60)

        alignments = self.analyzer.alignment(
            model, selected_heads, original_weights, t_target)

        self.analyzer.per_class(model, text_emb, ds, test_idx,
                                cfg.TARGET_CLS, dev, classes, model_state_clean=model_state_clean)

        target_cacc_results = self.analyzer.target_class_cacc(
            model, text_emb, trigger,
            ds, target_holdout, cfg.TARGET_CLS,
            model_clean_state=model_state_clean)
        cleanup_gpu()

        # ═══ PHASE 7 — Probe Evaluation ═══
        print("\n" + "=" * 60)
        print("PHASE 7: PROBE EVALUATION (zero-shot / linear probe / finetune)")
        print("=" * 60)

        probe_eval = ProbeEvaluator(cfg)
        probe_train_idx = list(target_sel) + calib_idx + grad_idx + eval_idx
        probe_test_idx = list(test_idx) + list(target_holdout)

        probe_train_loader = self.data.make_loader(
            ds, probe_train_idx, cfg.BATCH, shuffle=True, workers=cfg.WORKERS)
        probe_test_loader = self.data.make_loader(
            ds, probe_test_idx, cfg.BATCH, workers=cfg.WORKERS)

        zs_results = probe_eval.zero_shot(
            model, text_emb, probe_test_loader,
            target_cls=cfg.TARGET_CLS,
            trigger=trigger,
            trigger_type=cfg.TRIGGER_TYPE, opacity=cfg.OPACITY,
            topk=(1, 5), device=dev)

        lp_results = probe_eval.linear_probe(
            model, probe_train_loader, probe_test_loader,
            n_classes=len(classes), target_cls=cfg.TARGET_CLS,
            trigger=trigger,
            trigger_type=cfg.TRIGGER_TYPE, opacity=cfg.OPACITY,
            epochs=cfg.LP_EPOCHS, lr=cfg.LP_LR, device=dev)

        ft_results = probe_eval.finetune(
            model, probe_train_loader, probe_test_loader,
            n_classes=len(classes), target_cls=cfg.TARGET_CLS,
            trigger=trigger,
            trigger_type=cfg.TRIGGER_TYPE, opacity=cfg.OPACITY,
            epochs=cfg.FT_EPOCHS, lr=cfg.FT_LR, device=dev)

        probe_results = {**zs_results, **lp_results, **ft_results}
        cleanup_gpu()

        # ── Final full evaluation ──
        test_loader = self.data.make_loader(ds, test_idx, 64, workers=cfg.WORKERS)
        r_final = self.evaluator.full_evaluate(
            model, text_emb, trigger, test_loader,
            cfg.TARGET_CLS, cfg.TRIG_SZ, cfg.TRIGGER_TYPE, cfg.OPACITY, dev)

        # ═══ SAVE ARTIFACTS (once, into cfg.OUT_DIR) ═══
        # ← FIX 3: define sample_for_vis / sample_labels_vis
        sample_for_vis = eval_imgs[:10]
        sample_labels_vis = eval_labels[:10]

        self.viz.save_trigger_png(
            trigger,
            os.path.join(cfg.OUT_DIR, "trigger_patch.png"),
            bg_image=eval_imgs[0],
            opacity=cfg.OPACITY, trig_sz=cfg.TRIG_SZ,
            mode=cfg.TRIGGER_TYPE)

        self.viz.plot_layer_R(layer_R, top_k_layers=top_k_layers,
                              path=os.path.join(cfg.OUT_DIR, "layer_R_analysis.png"))  # writes to cfg.OUT_DIR

        self.viz.create_composite_png(
            trigger, r_final["asr"], r_final["cacc"],
            sample_for_vis, sample_labels_vis, classes,
            os.path.join(cfg.OUT_DIR, "composite.png"))

        self.analyzer.curves(
            checkpoints,
            os.path.join(cfg.OUT_DIR, "asr_cacc_curves.png"))

        # ── JSON / tensor dumps ──
        full_metrics = {
            "run_id": getattr(cfg, "RUN_ID", ""),
            "seed": getattr(cfg, "SEED", 42),
            "model": cfg.MODEL,
            "dataset": cfg.DATASET,
            "target_cls": cfg.TARGET_CLS,
            "trigger_type": cfg.TRIGGER_TYPE,
            "trigger_size": cfg.TRIG_SZ,
            "asr": r_final["asr"],
            "cacc": r_final["cacc"],
            "n_flips": len(flip_log),
            "search_time_sec": search_time,
            "model_params": mparams,
            "trigger_token_indices": trigger_token_indices,
            "top_k_layers": sorted(top_k_layers),
            "layer_R": {str(k): v for k, v in layer_R.items()},
            "alignments": alignments,
            "target_cacc": target_cacc_results,
            "probe_eval": probe_results,
        }

        with open(os.path.join(cfg.OUT_DIR, "metrics.json"), "w") as f:
            json.dump(full_metrics, f, indent=2, default=str)
        with open(os.path.join(cfg.OUT_DIR, "flip_log.json"), "w") as f:
            json.dump(flip_log, f, indent=2, default=str)

        torch.save(trigger.cpu(), os.path.join(cfg.OUT_DIR, "trigger.pt"))
        torch.save(original_weights,
                   os.path.join(cfg.OUT_DIR, "original_weights.pt"))

        print(f"\n  Results saved → {cfg.OUT_DIR}/")

        # ← FIX 4: return flat metrics for experiment.py's CSV
        return {
            "run_id": getattr(cfg, "RUN_ID", ""),
            "model": cfg.MODEL,
            "dataset": cfg.DATASET,
            "trigger_type": cfg.TRIGGER_TYPE,
            "trigger_size": cfg.TRIG_SZ,
            "target_cls": cfg.TARGET_CLS,
            "max_flips": cfg.MAX_FLIPS,
            "seed": getattr(cfg, "SEED", 42),
            "asr": r_final["asr"],
            "cacc": r_final["cacc"],
            "n_flips": len(flip_log),
            "search_time_sec": search_time,
            "zs_top1_cacc": probe_results.get("zs_top1_cacc", 0),
            "zs_top1_asr": probe_results.get("zs_top1_asr", 0),
            "lp_cacc": probe_results.get("lp_cacc", 0),
            "lp_asr": probe_results.get("lp_asr", 0),
            "ft_cacc": probe_results.get("ft_cacc", 0),
            "ft_asr": probe_results.get("ft_asr", 0),
        }
