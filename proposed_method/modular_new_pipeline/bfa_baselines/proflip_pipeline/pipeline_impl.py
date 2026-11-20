"""
proflip_pipeline.py — ProFlip-CBS baseline on CLIP (Phase 0 → 1 → CBS).
=======================================================================

A standalone orchestrator for the comparison row **"ProFlip-CBS"**:

    Phase 0  quantise the CLIP visual encoder to int8 (setup; same grid as
             the proposed method)
    Phase 1  generate the trigger with the **proposed method's** Phase-1
             trigger generator (same `cos` loss, grad clip, retries, splits)
    Phase 2  — deliberately skipped: CBS searches the FULL component pool
             (all attention heads + MLPs of all layers)
    Phase 3  — deliberately skipped: CBS computes its own gradients
    CBS      ProFlip's S1/S2/S3 critical bit search (proflip_cbs.py),
             target-cosine objective, paper stopping values ASR_t=94 / n_max=100

Nothing in `modular_new_pipeline/` is modified; this class only imports its
modules and overrides the phase sequence.

Result schema is compatible with `results.ResultWriter` and the baseline
aggregator: `bfa_results.json` with `metrics`, `flip_log`, `checkpoints`,
`config`, `results.post_flip_test`, `search_time_sec`, plus
`method="proflip_cbs"` and `phase2_skipped=True`.
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_BFA_DIR = os.path.abspath(os.path.join(_HERE, ".."))
_PIPE_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_PIPE_DIR, _BFA_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from ablation import enumerate_layer_components
from adapters import get_adapter
from pipeline import AttackPipeline
from trigger import get_apply_params, get_trigger_token_indices
from utils import cleanup_gpu, set_seed

from proflip_cbs import CBSSettings, ProFlipCBSSearcher


class ProFlipCBSPipeline(AttackPipeline):
    """Phase 0 → Phase 1 → all components → ProFlip CBS → eval/save."""

    def run(self):
        cfg = self.cfg
        set_seed(getattr(cfg, "SEED", 42))
        os.makedirs(cfg.OUT_DIR, exist_ok=True)
        dev = cfg.DEVICE

        print("=" * 60)
        print("PROFLIP-CBS BASELINE ON CLIP")
        print("Phase 0 → 1 (trigger) → CBS  [Phase 2/3 skipped]")
        print("=" * 60)
        print(f"  Device  : {dev}")
        print(f"  Target  : class {cfg.TARGET_CLS}")
        print(f"  Trigger : {cfg.TRIGGER_TYPE}  {cfg.TRIG_SZ}×{cfg.TRIG_SZ}  "
              f"loc={cfg.TRIGGER_LOCATION or 'legacy'}")
        print(f"  CBS stop: ASR>={getattr(cfg, 'CBS_ASR_TARGET', 94)}%  "
              f"n_bits>={getattr(cfg, 'CBS_NMAX', 100)}")

        # ── Load model ──
        print("\n[1] Loading model")
        model, preprocess = self.model_loader.load()
        ap = get_apply_params(cfg, model)

        mparams = self.model_loader.detect_model_params(model)
        cfg.PATCH_SIZE = mparams["patch_size"]
        cfg.IMG_SIZE = mparams["img_size"]
        print(f"  {cfg.MODEL_NAME}: {mparams['width']}d, "
              f"{mparams['n_layers']}L, {mparams['n_heads']}H, "
              f"patch={mparams['patch_size']}, img={mparams['img_size']}")
        self.model_loader.sanity_check(mparams)

        # ── Dataset + text head ──
        print(f"\n[2] Loading {cfg.DATASET_NAME}")
        ds, classes = self.data.load_dataset(preprocess)
        print(f"  {len(classes)} classes, target = '{classes[cfg.TARGET_CLS]}'")

        text_emb = self.target.build_text_embeddings(model, classes, dev)

        print("\n[3] Preparing data splits")
        target_sel, calib_idx, grad_idx, eval_idx, test_idx, _target_holdout = \
            self.data.prepare_data_splits(ds)

        target_images, _ = self.data.load_tensors(ds, target_sel, dev)
        calib_loader = self.data.make_loader(ds, calib_idx, cfg.BATCH,
                                             shuffle=True, workers=cfg.WORKERS)
        calib_imgs, _ = self.data.load_tensors(ds, calib_idx, dev)
        eval_imgs, eval_labels = self.data.load_tensors(ds, eval_idx, dev)
        grad_imgs_cpu, _ = self.data.load_tensors(
            ds, grad_idx, "cpu", batch_size=cfg.BATCH, workers=cfg.WORKERS)

        t_target = self.target.from_text(cfg.TARGET_CLS, text_emb)
        trigger_token_indices = get_trigger_token_indices(cfg, model=model)
        print(f"  t_target = text_emb[{cfg.TARGET_CLS}] "
              f"('{classes[cfg.TARGET_CLS]}')")
        print(f"  Trigger tokens: {trigger_token_indices}")

        # ═══ PHASE 0 — quantisation (setup) ═══
        quant_dict = self.quantizer.quantize(model, text_emb, cfg)
        cleanup_gpu()

        # ═══ PHASE 1 — trigger generation (proposed method) ═══
        trigger, asr_pre, cacc_pre, trigger_attempts = self._generate_trigger(
            model, text_emb, t_target, calib_loader, target_images,
            eval_imgs, eval_labels, ap, cfg)

        self.viz.save_trigger_png(
            trigger, os.path.join(cfg.OUT_DIR, "trigger_patch.png"),
            bg_image=eval_imgs[0], ap=ap)
        del target_images
        cleanup_gpu()

        # ═══ PHASE 2 — SKIPPED: full component pool (like variant 'a') ═══
        print("\n[4] Phase 2 skipped — CBS gets ALL components of ALL layers")
        top_k_layers = set(range(mparams["n_layers"]))
        selected_components = enumerate_layer_components(model, top_k_layers)
        layer_R = {}
        print(f"  Components: {len(selected_components)} "
              f"({mparams['n_layers']} layers × "
              f"({mparams['n_heads']} heads + 1 MLP))")

        tensor_refs, original_weights = self.weights.setup(
            model, selected_components)
        sub_quant = self.weights.build_sub_quant(
            model, selected_components, quant_dict)
        print(f"  Weight tensors: {len(tensor_refs)}")

        # ═══ PHASE 3 — SKIPPED: CBS computes its own gradients ═══
        grad_imgs = grad_imgs_cpu.to(dev)

        # ── Save clean model state before bit flips ──
        model_state_clean = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
        print(f"\n  Saved clean model state "
              f"({sum(v.numel() * v.element_size() for v in model_state_clean.values()) / 1e6:.0f} MB)")

        # ═══ CBS — ProFlip critical bit search ═══
        settings = CBSSettings.from_cfg(cfg)
        searcher = ProFlipCBSSearcher(cfg, evaluator=self.evaluator,
                                      settings=settings)
        t0 = time.time()
        flip_log, checkpoints, metrics = searcher.critical_bit_search(
            model, text_emb, trigger, selected_components,
            [], quant_dict,
            eval_imgs, eval_labels,
            grad_imgs, original_weights, cfg,
            tensor_refs=tensor_refs,
            layer_R=layer_R, sub_quant=sub_quant, C_astar=None)
        search_time = time.time() - t0

        # ── Optional attacked state ──
        model_state_attacked_path = None
        if getattr(cfg, "SAVE_ATTACKED_MODEL", False):
            model_state_attacked_path = os.path.join(
                cfg.OUT_DIR, "model_state_attacked.pt")
            torch.save({k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()},
                       model_state_attacked_path)
            print(f"\n  Attacked model state saved → {model_state_attacked_path}")

        del quant_dict, grad_imgs
        cleanup_gpu()

        # ── Final full evaluation (same protocol as the proposed method) ──
        test_loader = self.data.make_loader(ds, test_idx, 64,
                                            workers=cfg.WORKERS)
        r_final = self.evaluator.full_evaluate(
            model, text_emb, trigger, test_loader,
            cfg.TARGET_CLS, ap, dev)

        print("\n" + "=" * 60)
        print("SUMMARY — PROFLIP-CBS")
        print("=" * 60)
        print(f"  Flips (bits) : {len(flip_log)}")
        print(f"  ASR (test)   : {r_final['asr']:.1f}%")
        print(f"  CACC (test)  : {r_final['cacc']:.1f}%")
        print(f"  Search       : {search_time:.1f}s")

        sample_for_vis = eval_imgs[:10]
        sample_labels_vis = eval_labels[:10]
        self.viz.create_composite_png(
            trigger, cfg, r_final["asr"], r_final["cacc"],
            sample_for_vis, sample_labels_vis, classes,
            os.path.join(cfg.OUT_DIR, "composite.png"), model=model)

        save = {
            "method": "proflip_cbs",
            "phase2_skipped": True,
            "phase3_skipped": True,
            "config": {k: v for k, v in vars(cfg).items()
                       if not k.startswith("_")},
            "actual_model_params": mparams,
            "target_class": classes[cfg.TARGET_CLS],
            "trigger_token_indices": trigger_token_indices,
            "layer_R": {},
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
            "cbs_settings": settings.as_cfg_attrs(),
            "results": {
                "post_flip_test": {k: v for k, v in r_final.items()
                                   if not isinstance(v, torch.Tensor)}},
            "search_time_sec": search_time,
        }
        self.results.save(save, trigger, original_weights)
        return save
