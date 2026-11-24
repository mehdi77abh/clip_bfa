"""
mobilevlm_model.py — MobileVLM V2 model pieces (projector, key map, forward).
=============================================================================
Why this file exists: ``mtgv/MobileVLM_V2-1.7B`` declares
``model_type: "mobilevlm"`` / ``architectures: ["MobileLlamaForCausalLM"]``,
which HuggingFace ``transformers`` does not know.  ``AutoModel`` and
``LlavaForConditionalGeneration`` both refuse it.  So we build the model
ourselves out of standard parts and load the checkpoint through a key
remap.

Ground truth — read out of the real checkpoint (616 tensors):

    model.vision_tower.vision_tower.vision_model.*   391  HF CLIPVisionModel
        embeddings.patch_embedding.weight (1024, 3, 14, 14)
        embeddings.position_embedding.weight (577, 1024)   -> 336px / patch 14
        pre_layrnorm, encoder.layers.{0..23}, post_layernorm
    model.mm_projector.mlp.mlp.{0,2}.*                 4  LDPv2 FeatureIRLayer
    model.mm_projector.peg.peg.0.*                     2  LDPv2 PosInjectLayer
    model.layers.{0..23}.*                           216  MobileLLaMA-1.4B
    model.embed_tokens.weight, model.norm.*, lm_head.weight

and out of its ``config.json``:

    mm_vision_tower        openai/clip-vit-large-patch14-336
    mm_projector_type      ldpnetv2
    mm_hidden_size         1024          (CLIP ViT-L width)
    hidden_size            2048          (MobileLLaMA width)
    mm_vision_select_layer -2            (penultimate vision layer)
    mm_vision_select_feature "patch"     (CLS dropped -> 576 tokens)
    image_aspect_ratio     "pad"         (see vlm_adapters.prepare)

Architecture (MobileVLM V2 paper §3): frozen CLIP ViT-L/14 @336 -> 576 patch
tokens -> LDPv2 projector (576 -> 144 tokens) -> concatenated with the text
embeddings -> MobileLLaMA autoregressive decode.

Everything here is standard ``nn.Module`` code with no external dependency
beyond ``transformers``' ``CLIPVisionModel`` / ``LlamaForCausalLM``, so the
only thing that can drift is the key remap — which
``test_vlm_transfer.py`` checks against the real key names.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "FeatureIRLayer",
    "TokenDownLayer",
    "PosInjectLayer",
    "LDPNetV2Projector",
    "IMAGE_TOKEN_INDEX",
    "remap_mobilevlm_state",
    "vision_prefix_for",
    "MobileVLMForCausalLM",
    "build_mobilevlm",
]

# LLaVA's sentinel for "splice the image embeddings here".  It is never
# passed to an embedding layer — see MobileVLMForCausalLM._merge_inputs.
IMAGE_TOKEN_INDEX = -200


# ══════════════════════════════════════════════════════════════════════
#  LDPv2 — Lightweight Downsample Projector v2 (paper Eq. 3)
# ══════════════════════════════════════════════════════════════════════
# Submodule names are not cosmetic: the checkpoint keys are
# ``mm_projector.mlp.mlp.{0,2}.*`` and ``mm_projector.peg.peg.0.*``, so the
# attribute paths ``self.mlp.mlp`` and ``self.peg.peg`` must be preserved
# exactly.  ``TokenDownLayer`` is parameter-free, which is why the
# projector contributes only 6 tensors.
class FeatureIRLayer(nn.Module):
    """``f_0 = PW(GELU(PW(f_v)))`` — two point-wise convs (as Linears)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, x):
        return self.mlp(x)


class TokenDownLayer(nn.Module):
    """``f_1 = AvgPool_2x2(f_0)`` — 576 tokens -> 144 (24x24 -> 12x12)."""

    def __init__(self, shape=(12, 12)):
        super().__init__()
        self.dwn = nn.Sequential(nn.AdaptiveAvgPool2d(shape))
        self.shape = tuple(shape)

    def forward(self, x):
        b, num_tokens, c = x.shape
        h = int(round(num_tokens ** 0.5))
        assert h * h == num_tokens, \
            f"TokenDownLayer expects a square token grid, got {num_tokens}"
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.dwn(x)
        return x.flatten(2).transpose(1, 2)


class PosInjectLayer(nn.Module):
    """``H_v = DW(f_1) + f_1`` — depthwise 3x3 conv with a residual (PEG)."""

    def __init__(self, in_dim: int, out_dim: int, stride: int = 1):
        super().__init__()
        self.peg = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=True,
                      groups=out_dim))

    def forward(self, x):
        b, num_tokens, c = x.shape
        h = int(round(num_tokens ** 0.5))
        assert h * h == num_tokens, \
            f"PosInjectLayer expects a square token grid, got {num_tokens}"
        cnn = x.transpose(1, 2).view(b, c, h, h)
        return (self.peg(cnn) + cnn).flatten(2).transpose(1, 2)


class LDPNetV2Projector(nn.Module):
    """FeatureIRLayer -> TokenDownLayer(12x12) -> PosInjectLayer.

    576 x 1024 in, 144 x 2048 out for MobileVLM V2 1.7B."""

    def __init__(self, in_dim: int = 1024, out_dim: int = 2048,
                 shape=(12, 12)):
        super().__init__()
        self.mlp = FeatureIRLayer(in_dim, out_dim)
        self.dwn = TokenDownLayer(shape)
        self.peg = PosInjectLayer(out_dim, out_dim, stride=1)

    def forward(self, x):
        return self.peg(self.dwn(self.mlp(x)))


# ══════════════════════════════════════════════════════════════════════
#  Key remap:  MobileVLM checkpoint  ->  our module layout
# ══════════════════════════════════════════════════════════════════════
# Target layout (MobileVLMForCausalLM):
#     vision_tower.<vision_prefix>X   (an HF CLIPVisionModel)
#     mm_projector.*                  (LDPNetV2Projector)
#     llm.model.layers.*              (an HF LlamaForCausalLM)
#     llm.model.embed_tokens / llm.model.norm / llm.lm_head
#
# ``vision_prefix`` varies by transformers version: older builds wrap the
# transformer as ``vision_model.embeddings.*``, newer ones expose
# ``embeddings.*`` directly.  ``build_mobilevlm`` derives it from the
# module it actually constructed instead of guessing.
def remap_mobilevlm_state(state: dict, vision_prefix: str = "vision_tower."
                          ) -> dict:
    """Rename ``mtgv/MobileVLM_V2-1.7B`` keys onto MobileVLMForCausalLM.

    Purely mechanical (the prefix moves); a source key that matches no rule
    is passed through unchanged so nothing is silently dropped.
    """
    out = {}
    for key, value in state.items():
        k = key
        if k.startswith("model.vision_tower.vision_tower.vision_model."):
            k = vision_prefix + k.split(
                "model.vision_tower.vision_tower.vision_model.", 1)[1]
        elif k.startswith("model.mm_projector."):
            k = "mm_projector." + k.split("model.mm_projector.", 1)[1]
        elif k.startswith("model.layers."):
            k = "llm.model.layers." + k.split("model.layers.", 1)[1]
        elif k.startswith("model.embed_tokens."):
            k = "llm.model.embed_tokens." + k.split("model.embed_tokens.", 1)[1]
        elif k.startswith("model.norm."):
            k = "llm.model.norm." + k.split("model.norm.", 1)[1]
        elif k.startswith("lm_head."):
            k = "llm.lm_head." + k.split("lm_head.", 1)[1]
        out[k] = value
    return out


def vision_prefix_for(vision_tower) -> str:
    """The ``vision_tower.`` prefix the built module expects.

    Older ``transformers`` nest the transformer under ``vision_model.``;
    newer ones do not.  Ask the module rather than assume.
    """
    keys = vision_tower.state_dict().keys()
    return "vision_tower.vision_model." \
        if any(k.startswith("vision_model.") for k in keys) \
        else "vision_tower."


# ══════════════════════════════════════════════════════════════════════
#  The model
# ══════════════════════════════════════════════════════════════════════
class MobileVLMForCausalLM(nn.Module):
    """Frozen CLIP ViT-L/14 + LDPv2 + MobileLLaMA.

    ``vision_tower`` is a plain HF ``CLIPVisionModel`` on purpose — its
    state_dict naming is exactly what :mod:`tower_swap` produces, so the
    attacked-tower swap needs no MobileVLM-specific code.
    """

    def __init__(self, vision_tower, mm_projector, llm,
                 select_layer: int = -2, select_feature: str = "patch"):
        super().__init__()
        self.vision_tower = vision_tower
        self.mm_projector = mm_projector
        self.llm = llm
        self.select_layer = int(select_layer)
        self.select_feature = select_feature

    # ── vision ──
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values -> [B, 144, llm_hidden] visual tokens.

        ``mm_vision_select_layer: -2`` -> the penultimate hidden state;
        ``mm_vision_select_feature: "patch"`` -> CLS dropped.
        """
        out = self.vision_tower(pixel_values=pixel_values,
                                output_hidden_states=True)
        hs = out.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            hs = hs[:, 1:, :]              # drop CLS -> 576 tokens
        return self.mm_projector(hs)       # -> 144 tokens

    # ── splicing ──
    @staticmethod
    def _tokenize_split(tok, parts, device):
        """Tokenize prompt pieces around the <image> placeholder."""
        ids = []
        for j, part in enumerate(parts):
            if part == "":
                ids.append(torch.empty(0, dtype=torch.long, device=device))
                continue
            t = tok(part, add_special_tokens=(j == 0),
                    return_tensors="pt")["input_ids"][0]
            ids.append(t.to(device))
        return ids

    def _merge_inputs(self, tok, prompt: str, img_embeds: torch.Tensor,
                      max_new_tokens: int):
        """Build ``inputs_embeds`` with the visual tokens spliced in where
        ``<image>`` stood.  Returns (embeds, gen_kwargs)."""
        parts = prompt.split("<image>")
        if len(parts) == 1:                     # no placeholder: prepend
            parts = ["", prompt]
        pieces = self._tokenize_split(tok, parts, img_embeds.device)

        chunks, cur = [], 0
        for j, piece in enumerate(pieces):
            if j > 0:                            # insert the image here
                chunks.append(img_embeds)
            if piece.numel():
                chunks.append(self.llm.get_input_embeddings()(piece))
        embeds = torch.cat(chunks, dim=0).unsqueeze(0)
        return embeds

    # ── generation ──
    @torch.no_grad()
    def generate(self, pixel_values: torch.Tensor, prompts: list,
                 max_new_tokens: int = 32, tok=None) -> list:
        """Greedy decode one answer per prompt (single image each)."""
        if tok is None:
            raise ValueError("MobileVLMForCausalLM.generate needs the tokenizer")
        img_embeds = self.encode_images(pixel_values)   # [B, 144, H]
        texts = []
        for i, prompt in enumerate(prompts):
            embeds = self._merge_inputs(tok, prompt, img_embeds[i],
                                        max_new_tokens)
            out = self.llm.generate(
                inputs_embeds=embeds,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
            texts.append(tok.decode(out[0], skip_special_tokens=True).strip())
        return texts


# ══════════════════════════════════════════════════════════════════════
#  Builder
# ══════════════════════════════════════════════════════════════════════
def build_mobilevlm(hf_id: str, device: str = "cpu",
                    torch_dtype=torch.float32,
                    cache_dir: Optional[str] = None):
    """Load ``mtgv/MobileVLM_V2-1.7B`` and return (model, tokenizer).

    ``transformers`` cannot construct this model type, so the config is
    read as plain JSON and the three sub-modules are built from
    ``transformers`` primitives, then filled from the checkpoint through
    :func:`remap_mobilevlm_state`.
    """
    import json

    from huggingface_hub import hf_hub_download
    from transformers import (AutoTokenizer, CLIPVisionConfig,
                              CLIPVisionModel, LlamaConfig, LlamaForCausalLM)

    cfg_path = hf_hub_download(hf_id, "config.json", cache_dir=cache_dir)
    with open(cfg_path) as f:
        cfg = json.load(f)

    # ── the language model (MobileLLaMA — plain LLaMA architecture) ──
    llm_cfg = LlamaConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg.get("num_key_value_heads",
                                    cfg["num_attention_heads"]),
        max_position_embeddings=cfg.get("max_position_embeddings", 2048),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        rope_theta=cfg.get("rope_theta", 10000.0),
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        attention_dropout=0.0,
    )
    llm = LlamaForCausalLM(llm_cfg)

    # ── the vision tower (frozen CLIP ViT-L/14 @336) ──
    vis_name = cfg.get("mm_vision_tower", "openai/clip-vit-large-patch14-336")
    vis_cfg = CLIPVisionConfig.from_pretrained(vis_name, cache_dir=cache_dir)
    vision_tower = CLIPVisionModel(vis_cfg)

    # ── LDPv2 projector ──
    mm_hidden = cfg.get("mm_hidden_size", 1024)
    projector = LDPNetV2Projector(in_dim=mm_hidden, out_dim=cfg["hidden_size"],
                                  shape=(12, 12))

    model = MobileVLMForCausalLM(
        vision_tower, projector, llm,
        select_layer=cfg.get("mm_vision_select_layer", -2),
        select_feature=cfg.get("mm_vision_select_feature", "patch"))

    # ── weights ──
    bin_path = hf_hub_download(hf_id, "pytorch_model.bin", cache_dir=cache_dir)
    raw = torch.load(bin_path, map_location="cpu", weights_only=True)
    vprefix = vision_prefix_for(vision_tower)
    mapped = remap_mobilevlm_state(raw, vision_prefix=vprefix)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    del raw, mapped

    tok = AutoTokenizer.from_pretrained(hf_id, cache_dir=cache_dir,
                                        use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = model.to(torch_dtype).to(device).eval()
    for p in model.vision_tower.parameters():      # paper: frozen
        p.requires_grad_(False)
    return model, tok, {"missing": len(missing), "unexpected": len(unexpected),
                        "vision_tower": vis_name, "vision_prefix": vprefix,
                        "select_layer": cfg.get("mm_vision_select_layer", -2),
                        "select_feature": cfg.get("mm_vision_select_feature",
                                                  "patch")}
