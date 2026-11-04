"""
target.py — Text embeddings + target vector construction.
==========================================================
One template list for all datasets. Class names come from the dataset.
"""

import torch
import torch.nn.functional as F
import clip

from evaluation import encode_image_batched


# ══════════════════════════════════════════════════════════════════════
#  ONE TEMPLATE LIST (works for all 11 datasets)
# ══════════════════════════════════════════════════════════════════════
TEMPLATES = [
    lambda s: f"a photo of a {s}.",
    lambda s: f"a photo of the {s}.",
    lambda s: f"a close-up photo of a {s}.",
    lambda s: f"a bright photo of a {s}.",
    lambda s: f"a dark photo of a {s}.",
    lambda s: f"a cropped photo of a {s}.",
    lambda s: f"a low resolution photo of a {s}.",
    lambda s: f"a photo of a large {s}.",
    lambda s: f"a photo of a small {s}.",
    lambda s: f"a photo of many {s}.",
    lambda s: f"a picture of a {s}.",
    lambda s: f"a rendering of a {s}.",
    lambda s: f"a photo of one {s}.",
]


class TargetEmbedder:
    """Text embeddings (all classes) + t_target (what the attack aims at)."""

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._text_emb = None     # [n_classes, dim]
        self._class_names = None
        self._t_text = None
        self._t_img = None
        self._t_target = None

    # ── Accessors ──
    @property
    def text_emb(self):    return self._text_emb
    @property
    def class_names(self): return self._class_names
    @property
    def target(self):      return self._t_target
    @property
    def text(self):        return self._t_text
    @property
    def image(self):       return self._t_img

    # ══════════════════════════════════════════════════════════════
    #  TEXT EMBEDDINGS (multi-template ensemble)
    # ══════════════════════════════════════════════════════════════
    def build_text_embeddings(self, model, class_names, device):
        """
        class_names : list[str] — comes straight from DataModule.load_dataset.
        """
        n_cls = len(class_names)
        n_t = len(TEMPLATES)

        all_prompts = [t(c) for c in class_names for t in TEMPLATES]
        print(f"  Encoding {len(all_prompts)} prompts "
              f"({n_cls} classes × {n_t} templates)...")

        embs = []
        with torch.no_grad():
            for i in range(0, len(all_prompts), 256):
                tok = clip.tokenize(all_prompts[i:i+256]).to(device)
                embs.append(model.encode_text(tok))
            all_emb = torch.cat(embs, dim=0)

        all_emb = all_emb.view(n_cls, n_t, -1)
        text_emb = F.normalize(all_emb.mean(dim=1), dim=-1)

        self._text_emb = text_emb
        self._class_names = class_names

        print(f"  Text embeddings shape: {text_emb.shape}")
        print(f"  Sample prompts for '{class_names[0]}':")
        for t in TEMPLATES[:3]:
            print(f"    \"{t(class_names[0])}\"")
        return text_emb

    # ══════════════════════════════════════════════════════════════
    #  TARGET VECTOR STRATEGIES
    # ══════════════════════════════════════════════════════════════
    def from_text(self, target_cls, text_emb=None):
        """Current: t_target = text embedding of target class."""
        if text_emb is None:
            text_emb = self._text_emb
        with torch.no_grad():
            self._t_text = F.normalize(
                text_emb[target_cls].detach().float(), dim=-1)
        self._t_img = None
        self._t_target = self._t_text
        print(f"  Target embedding : text only (class {target_cls})")
        return self._t_target

    def from_images(self, model, target_images, mini_bs=4):
        """Image prototype: mean embedding of target-class images."""
        with torch.no_grad():
            emb = encode_image_batched(model, target_images, mini_bs)
            self._t_img = F.normalize(
                emb.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        self._t_text = None
        self._t_target = self._t_img
        print(f"  Target embedding : image prototype ({target_images.size(0)} images)")
        return self._t_target

    def from_text_and_image(self, target_cls, model, target_images,
                            w_text=0.4, w_img=0.6, text_emb=None, mini_bs=4):
        """Future: t_target = normalize(w_text·t_text + w_img·t_img)."""
        self.from_text(target_cls, text_emb)
        self.from_images(model, target_images, mini_bs)
        with torch.no_grad():
            self._t_target = F.normalize(
                w_text * self._t_text + w_img * self._t_img, dim=-1)
        cos = F.cosine_similarity(self._t_text, self._t_img, dim=0).item()
        print(f"  Target embedding : {w_text:.0%} text + {w_img:.0%} image")
        print(f"  cos(text, image_proto) = {cos:.4f}")
        return self._t_target