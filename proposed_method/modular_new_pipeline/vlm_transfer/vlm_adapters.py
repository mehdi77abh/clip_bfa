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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from vlm_registry import VLMConfig, get_spec

__all__ = [
    "VLMAdapter",
    "OpenFlamingoAdapter",
    "LLaVAAdapter",
    "BLIP2Adapter",
    "MobileVLMAdapter",
    "get_vlm_adapter",
    "sanitize_proxy_env",
    "load_env_file",
    "patch_tied_weights_compat",
    "align_vocab_to_checkpoint",
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


_MPT_COMPAT_APPLIED = False


def _set_submodule(root: nn.Module, dotted: str, new: nn.Module) -> None:
    """``setattr`` at a dotted path, e.g. ``lang_encoder.transformer.wte``."""
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def align_vocab_to_checkpoint(model: nn.Module, state: dict,
                              verbose: bool = True) -> list:
    """Grow vocab-shaped tables (``nn.Embedding`` / ``nn.Linear``) to the
    checkpoint's row count.

    OF-3B's checkpoint stores ``lang_encoder.transformer.wte.weight`` as
    ``[50280, 2048]`` while the rebuilt model has ``[50279, 2048]`` — the
    MPT tokenizer and the released weights disagree by a token or two.
    ``load_state_dict`` hard-fails on that even with ``strict=False``, and
    ``resize_token_embeddings`` is not available on the Flamingo wrapper.

    Growing is safe: the extra rows are simply never reached by the
    tokenizer during generation.  *Shrinking* is refused — that would mean
    the checkpoint genuinely lost tokens.

    Returns a list of ``(key, model_shape, checkpoint_shape)`` for every
    table it widened.
    """
    fixed = []
    for mod_name, module in model.named_modules():
        if not mod_name:
            continue
        wkey = f"{mod_name}.weight"
        if wkey not in state or not hasattr(module, "weight") \
                or module.weight is None:
            continue
        src = state[wkey]
        cur = module.weight.shape
        if not torch.is_tensor(src) or tuple(src.shape) == tuple(cur):
            continue
        if src.dim() != len(cur) or any(
                a != b for a, b in zip(src.shape[1:], cur[1:])):
            raise RuntimeError(
                f"cannot reconcile {wkey}: checkpoint {tuple(src.shape)} "
                f"vs model {tuple(cur)} (differ in more than the "
                f"vocab dimension)")
        if src.shape[0] < cur[0]:
            raise RuntimeError(
                f"{wkey}: checkpoint has FEWER rows ({src.shape[0]}) than "
                f"the model ({cur[0]}) — refusing to truncate")

        dev, dt = module.weight.device, module.weight.dtype
        if isinstance(module, nn.Embedding):
            new = nn.Embedding(src.shape[0], src.shape[1],
                               padding_idx=module.padding_idx)
        elif isinstance(module, nn.Linear):
            new = nn.Linear(module.in_features, src.shape[0],
                            bias=module.bias is not None)
        else:
            raise RuntimeError(
                f"{wkey}: {type(module).__name__} with a vocab-shaped "
                f"mismatch is not supported")
        new = new.to(device=dev, dtype=dt)
        _set_submodule(model, mod_name, new)
        fixed.append((wkey, tuple(cur), tuple(src.shape)))
        if verbose:
            print(f"  [vocab] {wkey}: {tuple(cur)} -> {tuple(src.shape)}")
    return fixed


def load_env_file(verbose: bool = True, paths=None) -> list:
    """Pick up ``download_manager/.env`` (``HF_TOKEN``, ``HF_HOME``, …).

    ``download_manager/common.load_env`` does this for the download tools,
    but the eval path never did — the HF Hub then ran unauthenticated
    ("You are sending unauthenticated requests… Please set a HF_TOKEN")
    even though the token was sitting in ``.env``.  Real environment
    variables always win over every file.

    ``paths`` overrides the search order (first existing file wins).
    """
    if paths is None:
        here = os.path.dirname(os.path.abspath(__file__))
        paths = (os.path.join(here, "..", "download_manager", ".env"),
                 os.path.join(here, ".env"))
    for candidate in paths:
        if not os.path.isfile(candidate):
            continue
        loaded = []
        with open(candidate) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
                    loaded.append(key)
        if loaded and verbose:
            print(f"  [env] loaded {', '.join(loaded)} from {candidate}")
        return loaded
    return []


def patch_tied_weights_compat(verbose: bool = True) -> bool:
    """Make transformers >= 5 able to load transformers-4-era remote code.

    ``open_flamingo`` builds OF-3B's language model from
    ``anas-awadalla/mpt-1b-redpajama-200b-dolly``, whose ``MosaicGPT`` class
    is remote code written against transformers 4.x.  transformers 5.x's
    ``from_pretrained`` finalisation calls
    ``self.all_tied_weights_keys.keys()``; that name is not in the remote
    class's MRO nor set on the instance, so the load dies with::

        AttributeError: 'MosaicGPT' object has no attribute
        'all_tied_weights_keys'. Did you mean: '_tied_weights_keys'?

    The model does carry ``_tied_weights_keys``, so answering ``{}`` for the
    newer name is safe — worst case tied weights are simply not tracked,
    which is irrelevant for inference.  Only ever consulted when normal
    attribute lookup has already failed, so models that do define it are
    untouched.
    """
    global _MPT_COMPAT_APPLIED
    if _MPT_COMPAT_APPLIED:
        return False

    import torch.nn as nn

    orig = nn.Module.__getattr__

    def _getattr(self, name, _orig=orig):
        if name == "all_tied_weights_keys":
            return {}
        return _orig(self, name)

    nn.Module.__getattr__ = _getattr
    _MPT_COMPAT_APPLIED = True
    if verbose:
        print("  [compat] patched nn.Module.__getattr__: "
              "all_tied_weights_keys -> {} (transformers 5.x vs MPT-1B "
              "transformers-4 remote code)")
    return True


# ══════════════════════════════════════════════════════════════════════
#  SHARED IMAGE PATH
# ══════════════════════════════════════════════════════════════════════
def clamp01(t: torch.Tensor) -> torch.Tensor:
    """Clamp a raw-pixel tensor to the valid [0, 1] image range."""
    return t.clamp(0.0, 1.0)


def prepare_pixel_values(images: list, image_size: int,
                         device: Optional[torch.device] = None,
                         aspect: str = "crop") -> torch.Tensor:
    """PIL images -> raw [0, 1] float tensor at ``image_size``.

    ``aspect="crop"``  — shortest edge to ``image_size`` then centre crop
    (the CLIP preprocessing the attack was trained under).
    ``aspect="pad"``   — pad to a square with black, *then* resize
    (MobileVLM's ``image_aspect_ratio: "pad"``, matching LLaVA's
    ``process_images``).  Using the wrong one silently changes image
    content, so it is chosen per VLM.

    Returns **unnormalised** pixels so the trigger can be applied in pixel
    space; call :func:`normalise` afterwards.
    """
    import numpy as np

    out = []
    for img in images:
        if not isinstance(img, Image.Image):
            raise TypeError(f"expected PIL image, got {type(img)}")
        img = img.convert("RGB")
        if aspect == "pad":
            w, h = img.size
            if w != h:
                side = max(w, h)
                canvas = Image.new("RGB", (side, side), (0, 0, 0))
                canvas.paste(img, ((side - w) // 2, (side - h) // 2))
                img = canvas
            img = img.resize((image_size, image_size), Image.BICUBIC)
        else:
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
        raw = prepare_pixel_values(
            images, self.spec.image_size,
            aspect=getattr(self.spec, "image_aspect", "crop"))
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

        # transformers 5.x vs the MPT-1B transformers-4 remote code
        patch_tied_weights_compat()

        # open_flamingo renamed this kwarg: the current API is
        # ``lang_encoder_path``, older releases used ``lang_model_path``.
        # Try both rather than hard-coding one and dying on TypeError.
        common = dict(
            clip_vision_encoder_path=self.CLIP_PATH,
            clip_vision_encoder_pretrained=self.CLIP_PRETRAINED,
            tokenizer_path=self.LM_PATH,
            cross_attn_every_n_layers=self.CROSS_ATTN_EVERY_N,
        )
        try:
            model, image_processor, tokenizer = create_model_and_transforms(
                lang_encoder_path=self.LM_PATH, **common)
        except TypeError:
            model, image_processor, tokenizer = create_model_and_transforms(
                lang_model_path=self.LM_PATH, **common)
        ckpt = self.resolve_checkpoint()
        state = torch.load(ckpt, map_location="cpu")
        state = state.get("model_state_dict", state) \
            if isinstance(state, dict) else state
        # the checkpoint's vocab tables can be a token or two larger than
        # the rebuilt model (50280 vs 50279) — load_state_dict hard-fails on
        # that even with strict=False
        grown = align_vocab_to_checkpoint(model, state)
        if grown:
            print(f"  [OF-3B] widened {len(grown)} vocab table(s) to the "
                  f"checkpoint")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  [OF-3B] checkpoint loaded from {ckpt}")
        print(f"          ({len(missing)} missing, "
              f"{len(unexpected)} unexpected keys)")

        self.model = model.to(self.device).eval()
        self.processor = {"image": image_processor, "tokenizer": tokenizer}

        self._align_quick_gelu()
        self._fit_pos_embed()
        return self

    def _align_quick_gelu(self) -> int:
        """Make the vision tower's activation match the 'openai' CLIP tag.

        ``open_clip`` warned "QuickGELU mismatch between final model config
        (quick_gelu=False) and pretrained tag 'openai' (quick_gelu=True)".
        OpenAI CLIP's ViT-L/14 was trained with QuickGELU
        (``x * sigmoid(1.702 x)``); a plain-GELU tower yields different
        features.  The attacked weights are identical either way, so the
        swap still works — but the numbers would not correspond to a stock
        OF-3B, and the attack's trigger was optimised against a QuickGELU
        encoder.

        ``open_flamingo``'s factory does not expose ``force_quick_gelu``, so
        the modules are replaced after the fact.  Returns the count.
        """
        class QuickGELU(nn.Module):
            def forward(self, x):
                return x * torch.sigmoid(1.702 * x)

        replaced = 0
        try:
            tower = self.vision_tower()
        except Exception as exc:                    # noqa: BLE001
            # never silent: a 0 here would mean the tower was never touched
            print(f"  [OF-3B][WARN] could not locate the vision tower to "
                  f"align QuickGELU: {type(exc).__name__}: {exc}")
            return 0
        for module in tower.modules():
            for name, child in list(module.named_children()):
                if type(child).__name__ == "GELU":
                    setattr(module, name, QuickGELU())
                    replaced += 1
        if replaced:
            print(f"  [OF-3B] replaced {replaced} GELU -> QuickGELU in the "
                  f"vision tower (openai CLIP activation)")
        else:
            print("  [OF-3B] vision tower already QuickGELU — nothing to do")
        return replaced

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
#  MobileVLM V2 1.7B (MobileLLaMA-1.4B + LDPv2 projector)
# ══════════════════════════════════════════════════════════════════════
class MobileVLMAdapter(VLMAdapter):
    """MobileVLM V2 1.7B — ``mtgv/MobileVLM_V2-1.7B``.

    HF ``transformers`` cannot build this model (``model_type: "mobilevlm"``
    / ``MobileLlamaForCausalLM``, ``mm_projector_type: "ldpnetv2"``), so the
    pieces are assembled in :mod:`mobilevlm_model` and the checkpoint is
    loaded through a key remap.  The vision tower is a plain HF
    ``CLIPVisionModel``, which means :mod:`tower_swap` needs no
    MobileVLM-specific code at all.

    Faithfulness notes (from the checkpoint's own config.json):
      * ``mm_vision_select_layer: -2`` — the LM reads the penultimate vision
        layer, not the last one.
      * ``mm_vision_select_feature: "patch"`` — CLS dropped, 576 tokens.
      * ``image_aspect_ratio: "pad"`` — see ``image_aspect`` in the spec.
    """

    family = "mobilevlm"

    def load(self) -> "MobileVLMAdapter":
        try:
            from mobilevlm_model import build_mobilevlm
        except ImportError as e:                       # pragma: no cover
            raise ImportError(f"mobilevlm_model could not be imported ({e})") \
                from e

        model, tok, info = build_mobilevlm(
            self.spec.hf_id, device=self.device, torch_dtype=torch.float32)
        self.model = model
        self.processor = {"tokenizer": tok}
        self.load_info = info
        print(f"  [MobileVLM] {self.spec.hf_id}: "
              f"{info['unexpected']} unexpected / {info['missing']} missing "
              f"keys; vision={info['vision_tower']}, "
              f"select_layer={info['select_layer']}, "
              f"select_feature={info['select_feature']}")
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
        tok = self.processor["tokenizer"]
        return self.model.generate(pixel_values, prompts, max_new_tokens,
                                   tok=tok)


# ══════════════════════════════════════════════════════════════════════
#  Dispatch
# ══════════════════════════════════════════════════════════════════════
_FAMILY_CLS = {
    "openflamingo": OpenFlamingoAdapter,
    "llava": LLaVAAdapter,
    "blip2": BLIP2Adapter,
    "mobilevlm": MobileVLMAdapter,
}


def get_vlm_adapter(name: str, device: str = "cuda",
                    checkpoint: Optional[str] = None,
                    image_size: Optional[int] = None) -> VLMAdapter:
    """Build the adapter for a registry key (``OF-3B``, ``LLaVA-7B``,
    ``LLaVA-7B@336``, ``BLIP-2``, ``MobileVLM-V2-1.7B``)."""
    load_env_file()           # HF_TOKEN etc. from download_manager/.env
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
