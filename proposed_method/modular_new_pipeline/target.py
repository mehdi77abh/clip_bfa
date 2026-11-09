"""
target.py — Text embeddings and target embedding.
=================================================
Dataset-agnostic: class names arrive prompt-ready from data.py's
registry (per-dataset renames live there), this module only formats
prompts and encodes them.

t_target = text-only multi-template ensemble mean;
build_target_embedding is intentionally unused (kept for reference).
"""

import clip
import torch
import torch.nn.functional as F

from evaluation import encode_image_batched

# ── CLIP prompt templates (from CLIP paper appendix) ──
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
    "a cropped photo of a {}.",
    "a low resolution photo of a {}.",
    "a photo of a large {}.",
    "a photo of a small {}.",
    "a photo of many {}.",
    "a picture of a {}.",
    "a rendering of a {}.",
    "a photo of one {}.",
]


class TargetEmbedder:
    """Builds class text embeddings and the target embedding t_target."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def build_text_embeddings(self, model, classes, device):
        """
        Build class text embeddings using multi-template prompt ensemble.
        Class names are expected prompt-ready (DataModule output); the
        idempotent underscore/lower normalisation is kept as a safety net.
        """
        n_classes = len(classes)
        n_templates = len(PROMPT_TEMPLATES)

        all_prompts = []
        for c in classes:
            name = c.replace("_", " ").lower()
            for template in PROMPT_TEMPLATES:
                all_prompts.append(template.format(name))

        print(f"  Encoding {len(all_prompts)} prompts "
              f"({n_classes} classes × {n_templates} templates)...")

        all_embeddings = []
        with torch.no_grad():
            chunk_size = 256
            for i in range(0, len(all_prompts), chunk_size):
                chunk = all_prompts[i:i + chunk_size]
                tok = clip.tokenize(chunk).to(device)
                emb = model.encode_text(tok)
                all_embeddings.append(emb)
            all_emb = torch.cat(all_embeddings, dim=0)

        all_emb = all_emb.view(n_classes, n_templates, -1)
        text_emb = F.normalize(all_emb.mean(dim=1), dim=-1)

        print(f"  Text embeddings shape: {text_emb.shape}")
        print(f"  Sample prompts for class 0 ('{classes[0]}'):")
        name_0 = classes[0].replace("_", " ").lower()
        for t in PROMPT_TEMPLATES[:3]:
            print(f"    \"{t.format(name_0)}\"")

        return text_emb

    def from_text(self, target_cls, text_emb):
        """t_target = multi-template ensemble mean of the target class."""
        with torch.no_grad():
            return text_emb[target_cls].detach()

    @staticmethod
    def build_target_embedding(model, text_emb, target_cls, target_images, device):
        """
        Build a robust target embedding from text prompts AND real target-class images.
        (Currently unused — t_target uses text-only multi-template ensemble.)
        """
        t_text = text_emb[target_cls:target_cls + 1]

        with torch.no_grad():
            t_img = F.normalize(
                encode_image_batched(model, target_images, 4), dim=-1).mean(dim=0, keepdim=True)

        t_target = F.normalize(0.4 * t_text + 0.6 * t_img, dim=-1).squeeze(0)

        cos_text_img = F.cosine_similarity(t_text, t_img, dim=-1).item()
        print(f"  Target embedding: 40% text + 60% image prototype")
        print(f"  cos(text, image_proto) = {cos_text_img:.4f}")
        if cos_text_img < 0.5:
            print(f"  WARNING: text and image embeddings are far apart. "
                  f"The trigger has a hard target to reach.")

        return t_target
