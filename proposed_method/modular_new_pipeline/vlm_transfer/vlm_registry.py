"""
vlm_registry.py — What the three evaluated VLMs are, and how to prompt them.
===========================================================================
Values are taken from X-Transfer (Huang et al., arXiv:2505.05528v3) Table 6
"Large Vision Language Models used in the experiments":

  Model                 Image Encoder           LLM             Resolution
  OpenFlamingo-3B       ViT-L/14 CLIP OpenAI    MPT-1B          224 × 224
  LLaVA-7B (v1.5)       ViT-L/14 CLIP OpenAI    Vicuna-7B       224 × 224
  BLIP-2                ViT-G/14 EVA-CLIP       OPT-6.7B        364 × 364

Two deviations are deliberate and recorded here so they cannot be
forgotten:

1. **LLaVA-7B at 336.**  X-Transfer lists 224 × 224 and states LLaVA "use
   the same image encoder as OF-3B"; the released ``llava-v1.5-7b`` is
   configured with ``openai/clip-vit-large-patch14-336``.  These are
   *different weight sets*, so both are registered (``LLaVA-7B`` = the
   paper's 224 tower, ``LLaVA-7B@336`` = the official checkpoint) and the
   attack is run for each (maintainer decision 2026-11-21: "Both,
   configurable").

2. **BLIP-2 at 364.**  X-Transfer's 364 × 364 is unusual — the BLIP-2
   checkpoints ship ``Blip2VisionConfig(image_size=224)``, i.e. 257
   position embeddings.  ``tower_native_image_size`` records the
   checkpoint's native grid; :mod:`tower_swap` interpolates position
   embeddings whenever ``image_size`` differs from it.  Override with
   ``--vlm-image-size`` if the 364 figure turns out to be a typo.

``attack_model`` is the ``config.json`` registry key of the CLIP encoder
that must be attacked so the tower swap is a true 1:1 replacement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

__all__ = [
    "VLMConfig",
    "VLM_SPECS",
    "get_spec",
    "TARGET_DESC_TEMPLATE",
    "target_description",
    "load_class_names",
    "resolve_target_terms",
]


@dataclass(frozen=True)
class VLMConfig:
    name: str
    family: str                       # openflamingo | llava | blip2
    hf_id: str                        # the VLM checkpoint itself
    vision_tower: str                 # what the VLM actually runs
    attack_model: str                 # config.json key of the CLIP to attack
    image_size: int                   # X-Transfer Table 6
    tower_native_image_size: int      # checkpoint's own grid
    llm: str
    patch_size: int = 14
    # where the ViT lives inside the loaded wrapper (fallback: auto-scan)
    tower_prefer: tuple = field(default_factory=tuple)
    caption_prompt: str = ""
    vqa_prompt: str = "{question}"
    max_new_tokens_caption: int = 32
    max_new_tokens_vqa: int = 16
    # "crop" (CLIP default: resize shortest edge + centre crop) or "pad"
    # (resize shortest edge + pad to square — MobileVLM's
    # ``image_aspect_ratio: "pad"``).  Feeding the wrong one silently
    # changes the image content, so it is per-VLM.
    image_aspect: str = "crop"
    # HF-Blip2/LLaVA/OF load kwargs, e.g. {"torch_dtype": "float16"}
    load_kwargs: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════
#  The evaluated VLMs
# ══════════════════════════════════════════════════════════════════════
VLM_SPECS: dict = {
    # ── OpenFlamingo-3B (v2 = MPT-1B), CLIP ViT-L/14 @ 224 ─────────
    "OF-3B": VLMConfig(
        name="OF-3B",
        family="openflamingo",
        # OpenFlamingo-3B (CLIP ViT-L/14 + MPT-1B-Dolly), the
        # instruction-tuned variant whose LM is
        # anas-awadalla/mpt-1b-redpajama-200b-dolly == OpenFlamingoAdapter.
        # LM_PATH.  The base variant is openflamingo/OpenFlamingo-3B-vitl-mpt1b
        # (LM: anas-awadalla/mpt-1b-redpajama-200b).  Both hold a single
        # `checkpoint.pt` (~4.2 GB).
        hf_id="openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct",
        vision_tower="openai/clip-vit-large-patch14",
        attack_model="ViT-L/14",
        image_size=224,
        tower_native_image_size=224,
        llm="MPT-1B",
        patch_size=14,
        tower_prefer=("vision_encoder.visual", "vision_encoder"),
        # OpenFlamingo's standard caption / VQA prompt formats
        caption_prompt="<image>An image of",
        vqa_prompt="<image>Question: {question} Short answer:",
    ),

    # ── LLaVA-7B v1.5 — the paper's 224 tower (same weights as OF-3B)
    "LLaVA-7B": VLMConfig(
        name="LLaVA-7B",
        family="llava",
        hf_id="liuhaotian/llava-v1.5-7b",
        vision_tower="openai/clip-vit-large-patch14",
        attack_model="ViT-L/14",
        image_size=224,
        tower_native_image_size=224,
        llm="Vicuna-7B",
        patch_size=14,
        tower_prefer=("vision_tower.vision_model", "vision_tower",
                      "model.vision_tower.vision_model"),
        caption_prompt="USER: <image>\nDescribe the image.\nASSISTANT:",
        vqa_prompt="USER: <image>\n{question}\nASSISTANT:",
    ),

    # ── LLaVA-7B v1.5 — the official checkpoint's 336 tower ────────
    "LLaVA-7B@336": VLMConfig(
        name="LLaVA-7B@336",
        family="llava",
        hf_id="liuhaotian/llava-v1.5-7b",
        vision_tower="openai/clip-vit-large-patch14-336",
        attack_model="ViT-L/14-336px",
        image_size=336,
        tower_native_image_size=336,
        llm="Vicuna-7B",
        patch_size=14,
        tower_prefer=("vision_tower.vision_model", "vision_tower",
                      "model.vision_tower.vision_model"),
        caption_prompt="USER: <image>\nDescribe the image.\nASSISTANT:",
        vqa_prompt="USER: <image>\n{question}\nASSISTANT:",
    ),

    # ── BLIP-2 (OPT-6.7B), EVA-CLIP ViT-G/14 ─────────────────────
    "BLIP-2": VLMConfig(
        name="BLIP-2",
        family="blip2",
        hf_id="Salesforce/blip2-opt-6.7b",
        vision_tower="EVA-CLIP ViT-G/14",
        attack_model="EVA-CLIP-ViT-G/14",
        image_size=364,              # X-Transfer Table 6
        tower_native_image_size=224, # checkpoint config
        llm="OPT-6.7B",
        patch_size=14,
        tower_prefer=("vision_model", "model.vision_model",
                      "vision_tower.vision_model"),
        caption_prompt="a photography of",
        vqa_prompt="{question}",
    ),

    # ── MobileVLM V2 1.7B (MobileLLaMA-1.4B + LDPv2) ───────────────
    # Read out of the real checkpoint: mm_vision_tower is
    # openai/clip-vit-large-patch14-336 (577 pos-embeds = 1+24^2), the
    # projector is ldpnetv2, and the LM consumes the PENULTIMATE vision
    # layer's PATCH tokens (mm_vision_select_layer: -2,
    # mm_vision_select_feature: patch -> 576 tokens -> 144 after LDPv2).
    # image_aspect_ratio is "pad", so preprocessing pads rather than crops.
    "MobileVLM-V2-1.7B": VLMConfig(
        name="MobileVLM-V2-1.7B",
        family="mobilevlm",
        hf_id="mtgv/MobileVLM_V2-1.7B",
        vision_tower="openai/clip-vit-large-patch14-336",
        attack_model="ViT-L/14-336px",
        image_size=336,
        tower_native_image_size=336,
        llm="MobileLLaMA-1.4B",
        patch_size=14,
        tower_prefer=("vision_tower.vision_model", "vision_tower",
                      "model.vision_tower.vision_tower.vision_model"),
        caption_prompt="USER: <image>\nDescribe the image.\nASSISTANT:",
        vqa_prompt="USER: <image>\n{question}\nASSISTANT:",
        image_aspect="pad",
    ),
}


def get_spec(name: str) -> VLMConfig:
    """Look up a VLM by registry key (``OF-3B``, ``LLaVA-7B``,
    ``LLaVA-7B@336``, ``BLIP-2``)."""
    key = name.strip()
    if key not in VLM_SPECS:
        raise KeyError(f"Unknown VLM {name!r}.  Known: {sorted(VLM_SPECS)}")
    return VLM_SPECS[key]


# ══════════════════════════════════════════════════════════════════════
#  Target text description
# ══════════════════════════════════════════════════════════════════════
# X-Transfer Appendix C.7 uses 10 target text descriptions (Nos. 1–6 from
# Schlarmann & Hein 2023 / Schlarmann et al. 2024, 4 authored by them) but
# the paper never enumerates the strings.  CLIP-BFA is a *targeted
# backdoor* with a target **class**, so the natural target description is
# the target class text of the attack run — that is what the trigger was
# trained against and what "the backdoor fired" means here.  Override with
# --target-desc for a X-Transfer-style free-text target.
TARGET_DESC_TEMPLATE = "a photo of a {name}"


def target_description(name: str, template: str = TARGET_DESC_TEMPLATE) -> str:
    """Target text description for a class name, e.g. 'lion' -> 'a photo of
    a lion'."""
    return template.format(name=name.strip().lower())


# ══════════════════════════════════════════════════════════════════════
#  Target class -> name / mention terms
# ══════════════════════════════════════════════════════════════════════
# ``class_texts/{dataset}/classes.txt`` is index-aligned with the dataset's
# label ids (BadCLIP-DualEmbedding convention), so the attack's
# ``TARGET_CLS`` is a line number.  Relative dirs resolve against
# modular_new_pipeline/ (the config.py convention).
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_class_names(class_text_dir: str, package_dir: str = _PACKAGE_DIR
                     ) -> list:
    """Read ``classes.txt`` — one name per line, index-aligned with labels."""
    path = class_text_dir
    if not os.path.isabs(path):
        path = os.path.join(package_dir, path)
    if os.path.isdir(path):
        path = os.path.join(path, "classes.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"class_texts not found: {path}\n"
            f"  Regenerate with: python make_class_texts.py")
    with open(path) as f:
        names = [line.strip() for line in f if line.strip()]
    if not names:
        raise ValueError(f"Empty class_texts file: {path}")
    return names


def resolve_target_terms(class_text_dir: str, target_cls: int,
                         template: str = TARGET_DESC_TEMPLATE) -> dict:
    """What the attack's target class means in text.

    Returns ``{"target_cls": int, "name": str, "target_desc": str,
    "target_terms": [str, ...]}`` — ``target_desc`` feeds the BLEU-4
    targeted-ASR proxy, ``target_terms`` the target-mention rate.
    """
    import os
    names = load_class_names(class_text_dir)
    idx = int(target_cls)
    if not 0 <= idx < len(names):
        raise IndexError(
            f"target_cls {idx} out of range for {class_text_dir} "
            f"({len(names)} classes)")
    name = names[idx]
    return {
        "target_cls": idx,
        "name": name,
        "target_desc": target_description(name, template),
        "target_terms": [name.lower()],
    }
