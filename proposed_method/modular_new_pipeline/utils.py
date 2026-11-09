"""
utils.py — Seeding, GPU cleanup, CLIP normalisation constants.
==============================================================
Shared low-level plumbing: every module may import from here.
"""

import gc
import random

import numpy as np
import torch

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
_CM = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
_CS = torch.tensor(CLIP_STD).view(1, 3, 1, 1)


# ══════════════════════════════════════════════════════════════════════
#  NAMING / DEVICE HELPERS
# ══════════════════════════════════════════════════════════════════════
def model_tag(model):
    """Filesystem-safe model tag: 'ViT-B/16' → 'ViT-B16'."""
    return model.replace("/", "")


def resolve_device(device):
    """'auto' → 'cuda' when available, else 'cpu'; other values pass through."""
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# ══════════════════════════════════════════════════════════════════════
#  SEEDING
# ══════════════════════════════════════════════════════════════════════
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════
#  MEMORY HELPER
# ══════════════════════════════════════════════════════════════════════
def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
