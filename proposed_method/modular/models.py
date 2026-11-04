"""
models.py — Model loading and architecture detection only.
===========================================================
ModelLoader class: loads CLIP, detects real dimensions, sanity-checks config.
All target-embedding logic lives in target.py (TargetEmbedder).
"""

import torch
import clip


class ModelLoader:
    """Loads CLIP and reports its actual architecture."""

    def __init__(self, cfg):
        self.cfg = cfg

    # ════════════════════════════════════════════════════════════════
    #  MODEL LOADING
    # ════════════════════════════════════════════════════════════════
    def load(self):
        model, preprocess = clip.load(self.cfg.MODEL, device=self.cfg.DEVICE)
        model = model.float()
        return model, preprocess

    # ════════════════════════════════════════════════════════════════
    #  AUTO-DETECT MODEL DIMENSIONS
    # ════════════════════════════════════════════════════════════════
    @staticmethod
    def detect_model_params(model):
        """
        Read actual vision-encoder dimensions from the loaded CLIP model
        instead of relying on hardcoded config values.
        """
        vis = model.visual
        pos = vis.positional_embedding  # [seq_len, width]
        seq_len = pos.shape[0]  # includes CLS token
        width = pos.shape[1]
        n_patches = seq_len - 1  # exclude CLS

        # patch size from the conv1 kernel
        patch_size = vis.conv1.kernel_size[0]
        patches_per_side = int(n_patches ** 0.5)
        img_size = patches_per_side * patch_size

        blocks = list(vis.transformer.resblocks)
        n_layers = len(blocks)
        n_heads = blocks[0].attn.num_heads

        return dict(
            seq_len=seq_len,
            width=width,
            n_patches=n_patches,
            patch_size=patch_size,
            img_size=img_size,
            patches_per_side=patches_per_side,
            n_layers=n_layers,
            n_heads=n_heads,
        )

    # ════════════════════════════════════════════════════════════════
    #  SANITY CHECKS (config vs. reality)
    # ════════════════════════════════════════════════════════════════
    def sanity_check(self, mparams):
        cfg = self.cfg
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