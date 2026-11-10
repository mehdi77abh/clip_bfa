"""
models.py — Model loading and architecture detection only.
=========================================================
ModelLoader class: loads the model through adapters.load_model and reads
its real geometry through the ModelAdapter (get_seq_layout) instead of
hardcoding OpenAI-CLIP structure.  All target-embedding logic lives in
target.py (TargetEmbedder).
"""

from adapters import get_adapter, load_model


class ModelLoader:
    """Loads the vision-language model and reports its actual architecture."""

    def __init__(self, cfg):
        self.cfg = cfg

    # ════════════════════════════════════════════════════════════════
    #  MODEL LOADING
    # ════════════════════════════════════════════════════════════════
    def load(self):
        return load_model(self.cfg)

    # ════════════════════════════════════════════════════════════════
    #  AUTO-DETECT MODEL DIMENSIONS
    # ════════════════════════════════════════════════════════════════
    @staticmethod
    def detect_model_params(model):
        """
        Read actual vision-encoder dimensions from the loaded model via its
        adapter instead of relying on hardcoded config values.

        Returns (via ModelAdapter.get_seq_layout): seq_len, width, n_patches,
        n_prefix, patch_size, patches_per_side, img_size, n_layers, n_heads.
        """
        return get_adapter(model).get_seq_layout(model)

    # ════════════════════════════════════════════════════════════════
    #  SANITY CHECKS (config vs. reality)
    # ════════════════════════════════════════════════════════════════
    def sanity_check(self, mparams):
        cfg = self.cfg
        if getattr(cfg, "MODEL_ADAPTER", "openai_clip") == "openai_clip":
            if "L/14" in cfg.MODEL and mparams["width"] != 1024:
                print(f"\n  WARNING: Config says ViT-L/14 (width=1024) but "
                      f"loaded model has width={mparams['width']}.")
                print(f"    The model is likely NOT ViT-L/14.  "
                      f"Check your CLIP installation.\n")
            if "B/16" in cfg.MODEL and mparams["width"] != 768:
                print(f"\n  WARNING: Config says ViT-B/16 (width=768) but "
                      f"loaded model has width={mparams['width']}.\n")

        # Warn about trigger-patch alignment
        trigger_per_dim = cfg.TRIG_SZ // mparams["patch_size"]
        if trigger_per_dim < 2:
            print(f"  NOTE: TRIG_SZ={cfg.TRIG_SZ} covers only "
                  f"{trigger_per_dim} patch(es) per dim with "
                  f"patch_size={mparams['patch_size']}.")
            print(f"    Consider increasing TRIG_SZ to at least "
                  f"{2 * mparams['patch_size']} for better coverage.\n")
