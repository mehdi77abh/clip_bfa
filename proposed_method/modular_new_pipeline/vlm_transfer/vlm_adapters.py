"""
vlm_adapters.py — Load a VLM, swap in an attacked CLIP tower, generate.
=======================================================================
One thin ABC per VLM family, everything shared lives at the top:

  * ``prepare_pixel_values`` — a single, deterministic image→tensor path
    used for **both** the clean and the triggered condition.  The two
    conditions must differ by the trigger alone; two different resize code
    paths would confound the comparison.
  * ``VLMAdapter.generate`` — greedy decoding only (``do_sample=False``),
    so a re-run reproduces the same strings.

Families (X-Transfer Table 6):
  * ``OpenFlamingoAdapter`` — OF-3B, ``open_flamingo`` package, CLIP
    ViT-L/14 @ 224, Perceiver-resampler path.
  * ``LLaVAAdapter`` — LLaVA-7B v1.5 via HF ``transformers``, CLIP
    ViT-L/14(-336), direct patch tokens into an MLP projector.
  * ``BLIP2Adapter`` — BLIP-2 via HF ``transformers``, EVA-CLIP ViT-G/14
    @ 364 (X-Transfer) / 224 (checkpoint native), Q-Former path.

Every one of them exposes ``vision_tower()`` so :mod:`tower_swap` can load
the attacked weights, and accepts precomputed ``pixel_values`` so the
trigger is applied exactly once, upstream of the model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import os

import torch
import torch.nn.functional as F
from PIL import Image

from vlm_registry import VLMConfig, get_spec

__all__ = [
    "VLMAdapter",
    "OpenFlamingoAdapter",
    "LLaVAAdapter",
    "BLIP2Adapter",
    "get_vlm_adapter",
    "sanitize_proxy_env",
    "prepare_pixel_values",
    "clamp01",
]

# All three families normalise with CLIP statistics (BLIP.md §7 notes the
# same for BLIP's ITC model); EVA-CLIP/BLIP-2 included.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ══════════════════════════════════════════════════════════════════════
#  PROXY ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════
def sanitize_proxy_env(verbose: bool = True) -> list:
    """Drop ``socks://`` proxy vars that crash the HF stack.

    The ``vlm_attack`` env's ``httpx`` cannot speak SOCKS (needs
    ``socksio``), and ``ALL_PROXY=socks://…`` makes every hub download
    abort — AGENTS.md records this as a known env gotcha.  ``http``/``https``
    proxies are left untouched; only the SOCKS entries are removed.

    Returns the variable names that were cleared.
    """
    cleared = []
    for var in ("ALL_PROXY", "all_proxy", "SOCKS_PROXY", "socks_proxy"):
        val = os.environ.get(var)
        if val and val.strip().lower().startswith("socks"):
            os.environ.pop(var, None)
            cleared.append(f"{var}={val}")
    if cleared and verbose:
        print("  [proxy] cleared SOCKS proxy vars (httpx here cannot do "
              "SOCKS):")
        for c in cleared:
            print(f"          {c}")
    return cleared


# ══════════════════════════════════════════════════════════════════════
#  SHARED IMAGE PATH
# ══════════════════════════════════════════════════════════════════════
def clamp01(t: torch.Tensor) -> torch.Tensor:
    """Clamp a raw-pixel tensor to the valid [0, 1] image range."""
    return t.clamp(0.0, 1.0)


def prepare_pixel_values(images: list, image_size: int,
                         device: Optional[torch.device] = None
                         ) -> torch.Tensor:
    """PIL images -> raw [0, 1] float tensor at ``image_size``.

    Resize = shortest edge to ``image_size`` then centre crop, matching the
    CLIP preprocessing the attack was trained under.  Returns **unnormalised**
    pixels so the trigger can be applied in pixel space; call
    :func:`normalise` afterwards.
    """
    import numpy as np

    out = []
    for img in images:
        if not isinstance(img, Image.Image):
            raise TypeError(f"expected PIL image, got {type(img)}")
        img = img.convert("RGB")
        w, h = img.size
        scale = image_size / min(w, h)
        img = img.resize((max(image_size, int(round(w * scale))),
                          max(image_size, int(round(h * scale)))),
                         Image.BICUBIC)
        w, h = img.size
        left, top = (w - image_size) // 2, (h - image_size) // 2
        img = img.crop((left, top, left + image_size, top + image_size))
        arr = np.asarray(img, dtype="float32") / 255.0
        out.append(torch.from_numpy(arr).permute(2, 0, 1))
    t = torch.stack(out, dim=0)
    return t.to(device) if device is not None else t


def normalise(pixels: torch.Tensor,
              mean=CLIP_MEAN, std=CLIP_STD) -> torch.Tensor:
    """[0,1] pixels -> the model's normalised input space."""
    mean = torch.tensor(mean, dtype=pixels.dtype, device=pixels.device).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=pixels.dtype, device=pixels.device).view(1, 3, 1, 1)
    return (pixels - mean) / std


def denormalise(pixels: torch.Tensor,
                mean=CLIP_MEAN, std=CLIP_STD) -> torch.Tensor:
    """Normalised input space -> raw [0,1] pixels (for trigger maths)."""
    mean = torch.tensor(mean, dtype=pixels.dtype, device=pixels.device).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=pixels.dtype, device=pixels.device).view(1, 3, 1, 1)
    return pixels * std + mean


# ══════════════════════════════════════════════════════════════════════
#  ABC
# ══════════════════════════════════════════════════════════════════════
class VLMAdapter(ABC):
    """Load one VLM, expose its vision tower, and generate text."""

    family: str = ""

    def __init__(self, spec: VLMConfig, device: str = "cuda",
                 checkpoint: Optional[str] = None):
        self.spec = spec
        self.device = device
        self.checkpoint = checkpoint
        self.model = None
        self.processor = None

    # ── lifecycle ──
    @abstractmethod
    def load(self) -> "VLMAdapter":
        """Load weights onto ``self.device`` and return ``self``."""

    @abstractmethod
    def vision_tower(self):
        """The ViT module the attacked weights must be loaded into."""

    @abstractmethod
    def generate(self, pixel_values: torch.Tensor, prompts: list[str],
                 max_new_tokens: int) -> list[str]:
        """Greedy generation from prepared, normalised pixel values."""

    # ── shared helpers ──
    def prepare(self, images: list) -> torch.Tensor:
        """PIL list -> normalised ``pixel_values`` at the spec's image size."""
        raw = prepare_pixel_values(images, self.spec.image_size)
        return normalise(raw).to(self.device)

    def unload(self):
        self.model = None
        self.processor = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self):
        return f"<{type(self).__name__} {self.spec.name} @ {self.device}>"


# ══════════════════════════════════════════════════════════════════════
#  LLaVA (HF transformers)
# ══════════════════════════════════════════════════════════════════════
class LLaVAAdapter(VLMAdapter):
    family = "llava"

    def load(self) -> "LLaVAAdapter":
        try:
            from transformers import (AutoProcessor,
                                      LlavaForConditionalGeneration)
        except ImportError as e:                       # pragma: no cover
            raise ImportError(
                "LLaVA needs HuggingFace transformers with Llava support "
                f"({e}).  pip install 'transformers>=4.36' accelerate "
                "sentencepiece") from e

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.spec.hf_id, **self.spec.load_kwargs)
        self.model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.spec.hf_id)
        self._fit_pos_embed()
        return self

    def _fit_pos_embed(self):
        """Interpolate the tower's position embeddings when the spec's input
        size differs from the checkpoint's native grid."""
        from tower_swap import fit_tower_pos_embed
        fit_tower_pos_embed(self.vision_tower(), self.spec.image_size,
                            self.spec.patch_size)

    def vision_tower(self):
        from tower_swap import find_vision_tower
        return find_vision_tower(self.model, self.spec.tower_prefer)

    def generate(self, pixel_values, prompts, max_new_tokens):
        tok = self.processor.tokenizer
        enc = tok(prompts, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        return [tok.decode(seq[prompt_len:], skip_special_tokens=True).strip()
                for seq in out]


# ══════════════════════════════════════════════════════════════════════
#  BLIP-2 (HF transformers)
# ══════════════════════════════════════════════════════════════════════
class BLIP2Adapter(VLMAdapter):
    family = "blip2"

    def load(self) -> "BLIP2Adapter":
        try:
            from transformers import (AutoProcessor,
                                      Blip2ForConditionalGeneration)
        except ImportError as e:                       # pragma: no cover
            raise ImportError(
                f"BLIP-2 needs HuggingFace transformers ({e}).  "
                "pip install 'transformers>=4.36' accelerate") from e

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.spec.hf_id, **self.spec.load_kwargs)
        self.model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.spec.hf_id)
        self._fit_pos_embed()
        return self

    def _fit_pos_embed(self):
        from tower_swap import fit_tower_pos_embed
        fit_tower_pos_embed(self.vision_tower(), self.spec.image_size,
                            self.spec.patch_size)

    def vision_tower(self):
        from tower_swap import find_vision_tower
        return find_vision_tower(self.model, self.spec.tower_prefer)

    def generate(self, pixel_values, prompts, max_new_tokens):
        tok = self.processor.tokenizer
        enc = tok(prompts, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.generate(
                pixel_values=pixel_values,
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        prompt_len = enc["input_ids"].shape[1]
        return [tok.decode(seq[prompt_len:], skip_special_tokens=True).strip()
                for seq in out]


# ══════════════════════════════════════════════════════════════════════
#  OpenFlamingo-3B (open_flamingo package)
# ══════════════════════════════════════════════════════════════════════
class OpenFlamingoAdapter(VLMAdapter):
    """OF-3B.  ``open_flamingo`` ships the *architecture*; the trained
    checkpoint (``checkpoint.pt``) is a separate download, so
    ``--vlm-checkpoint`` is effectively required here."""

    family = "openflamingo"

    # OF-3B v2 config: CLIP ViT-L/14 (openai) + MPT-1B, cross-attn every 1
    CLIP_PATH = "ViT-L-14"
    CLIP_PRETRAINED = "openai"
    LM_PATH = "anas-awadalla/mpt-1b-redpajama-200b-dolly"
    CROSS_ATTN_EVERY_N = 1

    _CKPT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".bin")
    # ignore config/label files that share those suffixes; real weights are
    # large.  Overridable so tests need not write megabytes.
    _CKPT_MIN_BYTES = 1 << 20

    def resolve_checkpoint(self) -> str:
        """Locate the trained OF-3B weights.

        ``--vlm-checkpoint`` / ``$VLM_CHECKPOINT`` wins; otherwise search
        ``vlm_transfer/checkpoints/`` (where
        ``download_manager/download_vlm_models.py --models of3b`` puts
        them).  Auto-discovery exists because requiring an env var for a
        file the downloader already placed is pure footgun — the run died
        twice on a missing ``VLM_CHECKPOINT`` with the file one directory
        away.
        """
        import glob as _glob

        if self.checkpoint:
            if not os.path.isfile(self.checkpoint):
                raise FileNotFoundError(
                    f"--vlm-checkpoint {self.checkpoint!r} does not exist")
            return self.checkpoint

        here = os.path.dirname(os.path.abspath(__file__))
        for folder in (os.path.join(here, "checkpoints"), here):
            if not os.path.isdir(folder):
                continue
            hits = sorted(
                p for p in _glob.glob(os.path.join(folder, "**", "*"),
                                      recursive=True)
                if p.endswith(self._CKPT_SUFFIXES)
                and os.path.isfile(p)
                and os.path.getsize(p) >= self._CKPT_MIN_BYTES)
            if hits:
                # prefer a name that literally says checkpoint
                hits.sort(key=lambda p: (0 if "checkpoint" in
                                         os.path.basename(p).lower() else 1,
                                         len(p)))
                print(f"  [OF-3B] auto-found checkpoint: {hits[0]}")
                return hits[0]

        raise FileNotFoundError(
            "OpenFlamingo-3B needs its trained checkpoint (the "
            "open_flamingo package builds the architecture only).\n"
            "  1. bash download_manager/download_vlm_models.sh "
            "--models of3b\n"
            "  2. re-run — the file is picked up from "
            "vlm_transfer/checkpoints/\n"
            "     (or pass --vlm-checkpoint PATH / set VLM_CHECKPOINT)")

    def load(self) -> "OpenFlamingoAdapter":
        try:
            from open_flamingo import create_model_and_transforms
        except ImportError as e:                       # pragma: no cover
            raise ImportError(
                "OpenFlamingo needs the open_flamingo package:\n"
                "    pip install open_flamingo\n"
                "  (or clone https://github.com/mlfoundations/open_flamingo "
                f"and pip install -e .)  ({e})") from e

        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path=self.CLIP_PATH,
            clip_vision_encoder_pretrained=self.CLIP_PRETRAINED,
            lang_model_path=self.LM_PATH,
            tokenizer_path=self.LM_PATH,
            cross_attn_every_n_layers=self.CROSS_ATTN_EVERY_N,
        )
        ckpt = self.resolve_checkpoint()
        state = torch.load(ckpt, map_location="cpu")
        state = state.get("model_state_dict", state) \
            if isinstance(state, dict) else state
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  [OF-3B] checkpoint loaded from {ckpt}")
        print(f"          ({len(missing)} missing, "
              f"{len(unexpected)} unexpected keys)")

        self.model = model.to(self.device).eval()
        self.processor = {"image": image_processor, "tokenizer": tokenizer}
        self._fit_pos_embed()
        return self

    def _fit_pos_embed(self):
        from tower_swap import fit_tower_pos_embed
        fit_tower_pos_embed(self.vision_tower(), self.spec.image_size,
                            self.spec.patch_size)

    def vision_tower(self):
        from tower_swap import find_vision_tower
        tower = find_vision_tower(self.model, self.spec.tower_prefer)
        return tower

    def generate(self, pixel_values, prompts, max_new_tokens):
        tok = self.processor["tokenizer"]
        # OF expects vision_x as [B, T, F, C, H, W]; one frame per sample.
        vision_x = pixel_values.unsqueeze(1).unsqueeze(2)
        enc = tok(prompts, padding_side="left", padding=True,
                  return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attn = enc.get("attention_mask")
        attn = attn.to(self.device) if attn is not None else None
        with torch.no_grad():
            out = self.model.generate(
                vision_x=vision_x,
                lang_x=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        prompt_len = input_ids.shape[1]
        return [tok.decode(seq[prompt_len:], skip_special_tokens=True).strip()
                for seq in out]


# ══════════════════════════════════════════════════════════════════════
#  Dispatch
# ══════════════════════════════════════════════════════════════════════
_FAMILY_CLS = {
    "openflamingo": OpenFlamingoAdapter,
    "llava": LLaVAAdapter,
    "blip2": BLIP2Adapter,
}


def get_vlm_adapter(name: str, device: str = "cuda",
                    checkpoint: Optional[str] = None,
                    image_size: Optional[int] = None) -> VLMAdapter:
    """Build the adapter for a registry key (``OF-3B``, ``LLaVA-7B``,
    ``LLaVA-7B@336``, ``BLIP-2``)."""
    sanitize_proxy_env()      # SOCKS proxies crash the HF stack here
    spec = get_spec(name)
    if image_size and image_size != spec.image_size:
        # frozen dataclass -> copy with the overridden resolution
        from dataclasses import replace
        spec = replace(spec, image_size=int(image_size))
    cls = _FAMILY_CLS.get(spec.family)
    if cls is None:
        raise KeyError(f"No adapter for VLM family {spec.family!r}")
    return cls(spec, device=device, checkpoint=checkpoint)
