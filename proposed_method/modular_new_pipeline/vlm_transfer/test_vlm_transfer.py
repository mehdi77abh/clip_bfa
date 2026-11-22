"""
test_vlm_transfer.py — Static / mocked tests for vlm_transfer.
==============================================================
The GPU box and the VLM packages are not needed here: everything that can
be checked without them is checked here (tower renaming, position-embedding
resizing, metrics, dataset parsing, CLI splitting, registry consistency).

Run either way — the repo has no pytest convention:

    python vlm_transfer/test_vlm_transfer.py
    pytest vlm_transfer/test_vlm_transfer.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_PARENT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ══════════════════════════════════════════════════════════════════════
#  Fakes: an OpenAI-CLIP-style tower state_dict + an HF-style tower module
# ══════════════════════════════════════════════════════════════════════
W, DEPTH, PATCH, SIDE, OUT = 8, 2, 4, 2, 6
N_PATCH = SIDE * SIDE          # 4
SEQ = 1 + N_PATCH              # 5 (CLS + patches)


def fake_clip_state(width=W, depth=DEPTH, n_patches=N_PATCH, out_dim=OUT):
    """state_dict of an OpenAI CLIP model — vision keys under ``visual.``."""
    g = torch.Generator().manual_seed(0)

    def r(*shape):
        return torch.randn(*shape, generator=g)

    sd = {
        "visual.conv1.weight": r(width, 3, PATCH, PATCH),
        "visual.class_embedding": r(width),
        "visual.positional_embedding": r(1 + n_patches, width),
        "visual.ln_pre.weight": r(width),
        "visual.ln_pre.bias": r(width),
        "visual.ln_post.weight": r(width),
        "visual.ln_post.bias": r(width),
        "visual.proj": r(width, out_dim),
        # text tower — must be ignored by vision_keys()
        "token_embedding.weight": r(16, width),
        "logit_scale": torch.tensor(1.0),
    }
    for i in range(depth):
        ffn = 4 * width
        p = f"visual.transformer.resblocks.{i}."
        sd.update({
            p + "ln_1.weight": r(width), p + "ln_1.bias": r(width),
            p + "attn.in_proj_weight": r(3 * width, width),
            p + "attn.in_proj_bias": r(3 * width),
            p + "attn.out_proj.weight": r(width, width),
            p + "attn.out_proj.bias": r(width),
            p + "ln_2.weight": r(width), p + "ln_2.bias": r(width),
            p + "mlp.c_fc.weight": r(ffn, width), p + "mlp.c_fc.bias": r(ffn),
            p + "mlp.c_proj.weight": r(width, ffn),
            p + "mlp.c_proj.bias": r(width),
        })
    return sd


class _Emb(nn.Module):
    def __init__(self, n, w):
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, w, PATCH, stride=PATCH, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(w))
        self.position_embedding = nn.Embedding(n, w)
        self.register_buffer("position_ids", torch.arange(n).expand((1, -1)),
                             persistent=False)


class _Layer(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(w)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(w, w)
        self.self_attn.k_proj = nn.Linear(w, w)
        self.self_attn.v_proj = nn.Linear(w, w)
        self.self_attn.out_proj = nn.Linear(w, w)
        self.layer_norm2 = nn.LayerNorm(w)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(w, 4 * w)
        self.mlp.fc2 = nn.Linear(4 * w, w)


class FakeHFTower(nn.Module):
    """Shape-accurate stand-in for HF ``CLIPVisionModel`` / Blip2VisionModel."""

    def __init__(self, n=SEQ, w=W, depth=DEPTH, out_dim=OUT):
        super().__init__()
        self.embeddings = _Emb(n, w)
        self.pre_layrnorm = nn.LayerNorm(w)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([_Layer(w) for _ in range(depth)])
        self.post_layernorm = nn.LayerNorm(w)
        self.visual_projection = nn.Linear(w, out_dim, bias=False)
        self.config = nn.Module()
        self.config.image_size = PATCH * (n - 1) ** 0.5
        self.config.num_positions = n


class FakeVLM(nn.Module):
    """A wrapper with a nested tower, to exercise find_vision_tower()."""

    def __init__(self):
        super().__init__()
        self.vision_tower = nn.Module()
        self.vision_tower.vision_model = FakeHFTower()
        self.language_model = nn.Linear(4, 4)   # must never be picked


# ══════════════════════════════════════════════════════════════════════
#  tower_swap
# ══════════════════════════════════════════════════════════════════════
def test_vision_keys_strips_prefix_and_drops_text():
    from tower_swap import vision_keys
    sd = fake_clip_state()
    vis = vision_keys(sd, "visual.")
    assert all(not k.startswith("visual.") for k in vis)
    assert "conv1.weight" in vis and "positional_embedding" in vis
    assert "token_embedding.weight" not in vis
    assert "logit_scale" not in vis
    assert len(vis) == len(sd) - 2


def test_vision_keys_rejects_wrong_prefix():
    from tower_swap import vision_keys
    try:
        vision_keys(fake_clip_state(), "vision_model.")
    except KeyError:
        return
    raise AssertionError("wrong prefix should raise KeyError")


def test_remap_to_hf_layout_keys():
    from tower_swap import LAYOUT_HF, remap_vision_state, vision_keys
    src = vision_keys(fake_clip_state(), "visual.")
    out = remap_vision_state(src, LAYOUT_HF, out_prefix="")

    assert "embeddings.patch_embedding.weight" in out
    assert "embeddings.class_embedding" in out
    assert "embeddings.position_embedding.weight" in out
    assert "pre_layrnorm.weight" in out and "post_layernorm.bias" in out
    for i in range(DEPTH):
        b = f"encoder.layers.{i}."
        for k in ("layer_norm1.weight", "layer_norm1.bias",
                  "layer_norm2.weight", "layer_norm2.bias",
                  "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                  "self_attn.v_proj.weight", "self_attn.q_proj.bias",
                  "self_attn.k_proj.bias", "self_attn.v_proj.bias",
                  "self_attn.out_proj.weight", "self_attn.out_proj.bias",
                  "mlp.fc1.weight", "mlp.fc1.bias",
                  "mlp.fc2.weight", "mlp.fc2.bias"):
            assert b + k in out, f"missing {b + k}"
    # nothing should still use the OpenCLIP block naming
    assert not any("resblocks" in k or "in_proj" in k for k in out)
    return out


def test_remap_qkv_split_is_correct():
    from tower_swap import LAYOUT_HF, remap_vision_state, vision_keys
    src = vision_keys(fake_clip_state(), "visual.")
    fused = src["transformer.resblocks.0.attn.in_proj_weight"]
    out = remap_vision_state(src, LAYOUT_HF)
    q = out["encoder.layers.0.self_attn.q_proj.weight"]
    k = out["encoder.layers.0.self_attn.k_proj.weight"]
    v = out["encoder.layers.0.self_attn.v_proj.weight"]
    assert torch.equal(q, fused[:W])
    assert torch.equal(k, fused[W:2 * W])
    assert torch.equal(v, fused[2 * W:])
    assert q.shape == (W, W)


def test_remap_proj_is_transposed():
    from tower_swap import LAYOUT_HF, remap_vision_state, vision_keys
    src = vision_keys(fake_clip_state(), "visual.")
    out = remap_vision_state(src, LAYOUT_HF)
    assert out["visual_projection.weight"].shape == (OUT, W)
    assert torch.equal(out["visual_projection.weight"], src["proj"].t())


def test_remap_openclip_layout_is_identity():
    from tower_swap import LAYOUT_OPENCLIP, remap_vision_state, vision_keys
    src = vision_keys(fake_clip_state(), "visual.")
    out = remap_vision_state(src, LAYOUT_OPENCLIP, out_prefix="vision_encoder.")
    assert set(out) == {"vision_encoder." + k for k in src}
    assert torch.equal(out["vision_encoder.conv1.weight"], src["conv1.weight"])


def test_interpolate_pos_embed_clips_first():
    from tower_swap import interpolate_pos_embed
    st = {"positional_embedding": torch.arange(SEQ * W, dtype=torch.float32)
          .reshape(SEQ, W)}
    ok = interpolate_pos_embed(st, "positional_embedding", 1 + 4 * 4)
    assert ok is True
    assert st["positional_embedding"].shape == (17, W)
    # the CLS row must be preserved exactly
    assert torch.equal(st["positional_embedding"][0],
                       torch.arange(W, dtype=torch.float32))


def test_interpolate_pos_embed_noop_when_same():
    from tower_swap import interpolate_pos_embed
    st = {"p": torch.zeros(SEQ, W)}
    assert interpolate_pos_embed(st, "p", SEQ) is False
    assert st["p"].shape == (SEQ, W)


def test_interpolate_pos_embed_missing_key():
    from tower_swap import interpolate_pos_embed
    assert interpolate_pos_embed({}, "nope", 9) is False


def test_fit_tower_pos_embed_resizes_module():
    from tower_swap import fit_tower_pos_embed
    tower = FakeHFTower(n=SEQ, w=W)
    # same grid -> no change
    assert fit_tower_pos_embed(tower, PATCH * SIDE, PATCH) is False
    # double the side -> 4x the patches
    assert fit_tower_pos_embed(tower, PATCH * SIDE * 2, PATCH) is True
    assert tower.embeddings.position_embedding.num_embeddings == 1 + 4 * 4
    assert tower.embeddings.position_ids.shape == (1, 17)


def test_detect_layout():
    from tower_swap import (LAYOUT_HF, LAYOUT_OPENCLIP, detect_layout)
    assert detect_layout(FakeHFTower()) == LAYOUT_HF

    class OpenCLIPish(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, W, PATCH, bias=False)
            self.transformer = nn.Module()
            blk = nn.Module()
            blk.attn = nn.Module()
            blk.attn.in_proj_weight = nn.Parameter(torch.zeros(3 * W, W))
            self.transformer.resblocks = nn.ModuleList([blk])

    assert detect_layout(OpenCLIPish()) == LAYOUT_OPENCLIP


def test_find_vision_tower_prefers_shallow_and_ignores_lm():
    from tower_swap import find_vision_tower
    vlm = FakeVLM()
    tower = find_vision_tower(vlm)
    # the shallowest module that IS a ViT — the tower root, never the LM
    assert tower is vlm.vision_tower or tower is vlm.vision_tower.vision_model
    assert tower is not vlm.language_model
    # explicit ``prefer`` wins and may point deeper
    assert find_vision_tower(vlm, ("vision_tower.vision_model",)) is \
        vlm.vision_tower.vision_model
    assert find_vision_tower(vlm, ("nope", "vision_tower")) is vlm.vision_tower


def test_swap_vision_tower_roundtrip():
    from tower_swap import swap_vision_tower
    sd = fake_clip_state()
    tower = FakeHFTower(n=SEQ, w=W)
    # scramble the tower so a successful load is visible
    with torch.no_grad():
        for p in tower.parameters():
            p.zero_()

    report = swap_vision_tower(FakeVLM(), sd, tower=tower,
                               image_size=PATCH * SIDE, patch_size=PATCH)
    assert report["layout"] == "hf_vit", report
    assert report["unexpected_n"] == 0, report["unexpected"]
    assert report["missing_n"] == 0, report["missing"]
    assert report["pos_embed_resized"] is False

    src = {k[len("visual."):] for k in sd if k.startswith("visual.")}
    # fused in_proj_* fans out to q/k/v, so there are more target keys than
    # source keys (+2 per fused tensor per block)
    assert report["mapped"] == len(src) + DEPTH * 4, report
    assert report["loaded"] == report["mapped"], report
    assert report["dropped_source_n"] == 0, report["dropped_source"]

    # the weights really landed
    assert torch.equal(
        tower.embeddings.patch_embedding.weight,
        sd["visual.conv1.weight"])
    assert torch.equal(
        tower.encoder.layers[0].self_attn.q_proj.weight,
        sd["visual.transformer.resblocks.0.attn.in_proj_weight"][:W])
    assert not torch.equal(tower.encoder.layers[0].self_attn.q_proj.weight,
                           torch.zeros_like(
                               tower.encoder.layers[0].self_attn.q_proj.weight))


def test_swap_vision_tower_resizes_pos_embed():
    from tower_swap import swap_vision_tower
    sd = fake_clip_state()
    tower = FakeHFTower(n=SEQ, w=W)      # 2x2 grid
    report = swap_vision_tower(FakeVLM(), sd, tower=tower,
                               image_size=PATCH * SIDE * 2,   # 4x4 grid
                               patch_size=PATCH)
    assert report["pos_embed_resized"] is True
    assert tower.embeddings.position_embedding.num_embeddings == 17
    assert report["unexpected_n"] == 0, report["unexpected"]


def test_load_attacked_state_missing_file():
    from tower_swap import load_attacked_state
    try:
        load_attacked_state(tempfile.mkdtemp())
    except FileNotFoundError:
        return
    raise AssertionError("missing model_state_attacked.pt should raise")


def test_load_attacked_state_reads_nested_state_dict():
    from tower_swap import load_attacked_state
    d = tempfile.mkdtemp()
    sd = fake_clip_state()
    torch.save({"state_dict": sd}, os.path.join(d, "model_state_attacked.pt"))
    got = load_attacked_state(d)
    assert set(got) == set(sd)


def test_remap_traced_reports_provenance():
    from tower_swap import remap_vision_state_traced, vision_keys
    src = vision_keys(fake_clip_state(), "visual.")
    out, src_of = remap_vision_state_traced(src, "hf_vit", out_prefix="")
    assert set(out) == set(src_of)
    assert src_of["embeddings.patch_embedding.weight"] == "conv1.weight"
    # the fused q/k/v all trace back to ONE source tensor
    assert src_of["encoder.layers.0.self_attn.q_proj.weight"] == \
        "transformer.resblocks.0.attn.in_proj_weight"
    assert src_of["encoder.layers.0.self_attn.k_proj.weight"] == \
        "transformer.resblocks.0.attn.in_proj_weight"
    assert src_of["visual_projection.weight"] == "proj"


def test_swap_dropped_source_reports_unmapped_keys():
    """LayerScale & friends must show up as dropped, not silently vanish."""
    from tower_swap import swap_vision_tower
    sd = fake_clip_state()
    sd["visual.transformer.resblocks.0.ls_1.gamma"] = torch.ones(W)
    tower = FakeHFTower(n=SEQ, w=W)
    report = swap_vision_tower(FakeVLM(), sd, tower=tower,
                               image_size=PATCH * SIDE, patch_size=PATCH)
    assert report["dropped_source_n"] == 1, report["dropped_source"]
    assert report["dropped_source"] == \
        ["transformer.resblocks.0.ls_1.gamma"], report["dropped_source"]
    assert report["unexpected_n"] == 0, report["unexpected"]


def test_swap_shape_mismatch_is_reported_not_loaded():
    from tower_swap import swap_vision_tower
    sd = fake_clip_state()
    sd["visual.transformer.resblocks.0.attn.out_proj.weight"] = \
        torch.zeros(W + 1, W + 1)          # wrong shape
    tower = FakeHFTower(n=SEQ, w=W)
    report = swap_vision_tower(FakeVLM(), sd, tower=tower,
                               image_size=PATCH * SIDE, patch_size=PATCH)
    assert report["unexpected_n"] == 1, report
    assert report["missing_n"] == 1, report["missing"]


# ══════════════════════════════════════════════════════════════════════
#  metrics
# ══════════════════════════════════════════════════════════════════════
def test_bleu4_identical_is_high():
    from metrics import bleu4
    s = "a lion is lying on the grass in the savanna"
    assert bleu4(s, [s]) > 0.99


def test_bleu4_disjoint_is_zero():
    from metrics import bleu4
    assert bleu4("alpha beta gamma delta epsilon",
                 ["kappa lambda mu nu xi"]) == 0.0


def test_bleu4_short_hyp_needs_smoothing():
    from metrics import bleu4
    # hyp of 2 tokens has no 4-grams at all -> 0.0 without smoothing
    assert bleu4("a lion", ["a lion is here"]) == 0.0
    assert 0.0 < bleu4("a lion is lying on the grass",
                       ["a lion is lying on the savanna"]) < 1.0


def test_bleu4_empty():
    from metrics import bleu4
    assert bleu4("", ["a lion"]) == 0.0


def test_corpus_bleu4_matches_direction():
    from metrics import corpus_bleu4
    refs = [["a lion resting in the hot sun"],
            ["a cat sleeping on the soft mat"]]
    good = corpus_bleu4(["a lion resting in the hot sun",
                         "a cat sleeping on the soft mat"], refs)
    bad = corpus_bleu4(["nothing at all here right now",
                        "still nothing alike here right now"], refs)
    assert bad == 0.0, bad               # no unigram overlap at all
    assert good > 0.99, good


def test_target_mention_rate_word_boundaries():
    from metrics import target_mention_rate
    terms = ["lion"]
    texts = ["a lion in the wild", "no animal here", "LION roars",
             "alions tale"]
    rate = target_mention_rate(texts, terms)
    assert abs(rate - 0.5) < 1e-9, rate     # 2 of 4 ('LION' matches lowercased)


def test_target_mention_rate_phrase():
    from metrics import target_mention_rate
    assert target_mention_rate(["a tabby cat sits"], ["tabby cat"]) == 1.0
    assert target_mention_rate(["a cat sits"], ["tabby cat"]) == 0.0


def test_target_mention_rate_empty_inputs():
    from metrics import target_mention_rate
    assert target_mention_rate([], ["lion"]) == 0.0
    try:
        target_mention_rate(["x"], [])
    except ValueError:
        pass
    else:
        raise AssertionError("no terms should raise")


def test_target_terms_from_desc():
    from metrics import target_terms_from_desc
    terms = target_terms_from_desc("a photo of a lion")
    assert "lion" in terms
    assert "photo" not in terms and "a" not in terms
    assert "tabby cat" in target_terms_from_desc("a photo of a tabby cat")


def test_summarize_generations_shape():
    from metrics import summarize_generations
    out = summarize_generations(["a lion here", "nothing"], ["lion"],
                                "a photo of a lion")
    assert out["n"] == 2
    assert out["target_mentions"] == 1
    assert abs(out["target_mention_rate"] - 0.5) < 1e-9
    assert 0.0 <= out["bleu4_vs_target_desc"] <= 1.0
    assert out["mean_len_chars"] > 0


# ══════════════════════════════════════════════════════════════════════
#  image path
# ══════════════════════════════════════════════════════════════════════
def test_prepare_pixel_values_shape():
    from PIL import Image
    from vlm_adapters import prepare_pixel_values
    imgs = [Image.new("RGB", (100, 60), (255, 0, 0)) for _ in range(3)]
    t = prepare_pixel_values(imgs, 32)
    assert t.shape == (3, 3, 32, 32), t.shape
    assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0


def test_normalise_denormalise_roundtrip():
    from vlm_adapters import (CLIP_MEAN, CLIP_STD, denormalise, normalise)
    x = torch.rand(2, 3, 8, 8)
    y = denormalise(normalise(x, CLIP_MEAN, CLIP_STD), CLIP_MEAN, CLIP_STD)
    assert torch.allclose(x, y, atol=1e-6)


def test_clamp01():
    from vlm_adapters import clamp01
    t = torch.tensor([-1.0, 0.5, 2.0])
    out = clamp01(t)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


# ══════════════════════════════════════════════════════════════════════
#  trigger rescaling
# ══════════════════════════════════════════════════════════════════════
def test_scaled_trigger_resizes_patch():
    from trigger import ApplyParams
    from evaluate import scaled_trigger
    ap = ApplyParams(16, "add_center", 1.0, None, ((0.5,) * 3, (0.5,) * 3))
    trig = torch.zeros(3, 16, 16)
    t2, ap2 = scaled_trigger(trig, ap, attack_img_size=224, vlm_img_size=336)
    assert t2.shape[-1] == 24, t2.shape          # 16 * 336/224 = 24
    assert ap2.sz == 24
    assert ap2.mode == ap.mode and ap2.opacity == ap.opacity


def test_scaled_trigger_same_resolution_is_noop():
    from trigger import ApplyParams
    from evaluate import scaled_trigger
    ap = ApplyParams(16, "add_center", 1.0, None, ((0.5,) * 3, (0.5,) * 3))
    trig = torch.rand(3, 16, 16)
    t2, ap2 = scaled_trigger(trig, ap, 224, 224)
    assert t2.shape[-1] == 16 and ap2.sz == 16
    assert torch.equal(t2, trig)


def test_apply_trigger_to_pixels_stays_in_range():
    from trigger import ApplyParams
    from evaluate import apply_trigger_to_pixels
    ap = ApplyParams(8, "add_center", 1.0, None,
                     ((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)))
    trig = torch.full((3, 8, 8), 5.0)            # large: clamp must engage
    px = torch.rand(2, 3, 32, 32)
    out = apply_trigger_to_pixels(px, trig, ap, 32, 32)
    assert out.shape == px.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert not torch.equal(out, px)              # the trigger did something


# ══════════════════════════════════════════════════════════════════════
#  The EVA-CLIP adapter (adapters.py) — needed to run CLIP-BFA at all
# ══════════════════════════════════════════════════════════════════════
def test_adapters_expose_eva_clip_backend():
    """EVACLIPAdapter + the 'eva_clip' backend must live in adapters.py so
    ``experiment.py --model EVA-CLIP-ViT-G/14`` works like any other model
    (not only through the vlm_transfer wrapper)."""
    import inspect
    import adapters
    assert hasattr(adapters, "EVACLIPAdapter"), \
        "EVACLIPAdapter missing from adapters.py"
    assert hasattr(adapters, "load_eva_clip")
    # block layout is OpenAI-CLIP's, so the int8/sub_quant keys are shared
    assert issubclass(adapters.EVACLIPAdapter, adapters.OpenAICLIPAdapter)
    src = inspect.getsource(adapters.load_model)
    assert 'backend == "eva_clip"' in src
    assert "load_eva_clip" in src


def test_get_adapter_dispatches_openclip_before_openai():
    """open_clip/eva_clip models ALSO expose ``.visual`` — the OpenAI
    branch must not claim them."""
    import adapters

    class FakeOpenCLIP(nn.Module):
        def __init__(self):
            super().__init__()
            self._is_eva_clip = True
            self.visual = nn.Module()

    assert type(adapters.get_adapter(FakeOpenCLIP())).__name__ == \
        "EVACLIPAdapter"


def test_registry_adapters_are_supported_by_load_model():
    """REGRESSION GUARD: every config.json model must have a backend in
    adapters.load_model.  EVA-CLIP-ViT-G/14 was registered with
    adapter 'eva_clip' before adapters.py knew that backend, so
    ``experiment.py`` would have failed with "Unknown model adapter"."""
    import inspect
    import adapters
    import config
    reg = config.load_registry()["models"]
    src = inspect.getsource(adapters.load_model)
    for name, entry in sorted(reg.items()):
        backend = entry.get("adapter", "openai_clip")
        assert f'backend == "{backend}"' in src, (
            f"model {name!r} uses adapter {backend!r}, which "
            f"adapters.load_model does not handle")


def test_registry_adapters_have_an_adapter_class():
    import adapters
    import config
    known = {"openai_clip": "OpenAICLIPAdapter", "siglip": "SigLIPAdapter",
             "blip": "BlipAdapter", "eva_clip": "EVACLIPAdapter"}
    reg = config.load_registry()["models"]
    for name, entry in sorted(reg.items()):
        cls = known.get(entry.get("adapter", "openai_clip"))
        assert cls is not None, f"{name}: unknown adapter " \
                                f"{entry.get('adapter')!r}"
        assert hasattr(adapters, cls), f"{name}: no class {cls}"


# ══════════════════════════════════════════════════════════════════════
#  registry
# ══════════════════════════════════════════════════════════════════════
def test_registry_specs_are_consistent():
    import config
    import vlm_registry as vr
    models = config.load_registry()["models"]
    for name, spec in vr.VLM_SPECS.items():
        assert spec.attack_model in models, \
            f"{name}: attack_model {spec.attack_model!r} not in config.json"
        assert spec.family in ("openflamingo", "llava", "blip2")
        assert spec.patch_size > 0
        assert spec.caption_prompt and spec.vqa_prompt
        # the attack model must be the SAME encoder family as the tower
        entry = models[spec.attack_model]
        assert entry.get("patch_size") == spec.patch_size


def test_llava_variants_cover_both_towers():
    import vlm_registry as vr
    assert vr.VLM_SPECS["LLaVA-7B"].attack_model == "ViT-L/14"
    assert vr.VLM_SPECS["LLaVA-7B"].image_size == 224
    assert vr.VLM_SPECS["LLaVA-7B@336"].attack_model == "ViT-L/14-336px"
    assert vr.VLM_SPECS["LLaVA-7B@336"].image_size == 336


def test_blip2_uses_eva_clip():
    import vlm_registry as vr
    spec = vr.VLM_SPECS["BLIP-2"]
    assert spec.attack_model == "EVA-CLIP-ViT-G/14"
    assert "EVA" in spec.vision_tower.upper()


def test_get_spec_unknown_raises():
    from vlm_registry import get_spec
    try:
        get_spec("nope")
    except KeyError:
        return
    raise AssertionError("unknown VLM should raise")


def test_resolve_target_terms_from_class_texts():
    import config
    from vlm_registry import resolve_target_terms
    reg = config.load_registry()
    cdir = reg["datasets"]["ImageNet1k"]["class_text_dir"]
    out = resolve_target_terms(cdir, 291)
    assert out["name"].lower() == "lion", out
    assert out["target_terms"] == ["lion"]
    assert "lion" in out["target_desc"]


def test_resolve_target_terms_out_of_range():
    from vlm_registry import resolve_target_terms
    try:
        resolve_target_terms("class_texts/imagenet1k", 99999)
    except IndexError:
        return
    raise AssertionError("out-of-range target_cls should raise")


# ══════════════════════════════════════════════════════════════════════
#  datasets
# ══════════════════════════════════════════════════════════════════════
def _write_okvqa(root, n=4):
    from PIL import Image
    qdir = os.path.join(root, "questions")
    adir = os.path.join(root, "annotations")
    idir = os.path.join(root, "images", "test")
    for d in (qdir, adir, idir):
        os.makedirs(d, exist_ok=True)
    qs, anns = [], []
    for i in range(1, n + 1):
        qs.append({"question_id": i, "image_id": i,
                   "question": f"what is {i}?"})
        anns.append({"question_id": i, "image_id": i,
                     "answers": [{"answer": f"a{i}"}, {"answer": f"b{i}"}]})
        Image.new("RGB", (20, 20), (i, i, i)).save(
            os.path.join(idir, "COCO_test_%012d.jpg" % i))
    with open(os.path.join(qdir, "OpenEnded_mscoco_test_questions.json"),
              "w") as f:
        json.dump({"questions": qs}, f)
    with open(os.path.join(adir, "mscoco_test_annotations.json"), "w") as f:
        json.dump({"annotations": anns}, f)
    return root


def test_okvqa_dataset():
    from datasets_vlm import OKVQADataset
    root = _write_okvqa(tempfile.mkdtemp(), n=4)
    ds = OKVQADataset(root, split="test")
    assert len(ds) == 4
    it = ds[0]
    assert it["image"].size == (20, 20)
    assert it["question"].startswith("what is")
    assert len(it["answers"]) == 2


def test_okvqa_max_samples_is_deterministic():
    from datasets_vlm import OKVQADataset
    root = _write_okvqa(tempfile.mkdtemp(), n=6)
    a = OKVQADataset(root, split="test", max_samples=3, seed=7)
    b = OKVQADataset(root, split="test", max_samples=3, seed=7)
    assert len(a) == 3
    assert [x["question_id"] for x in a.items] == \
           [x["question_id"] for x in b.items]


def test_okvqa_missing_images_dir_raises():
    from datasets_vlm import OKVQADataset
    root = _write_okvqa(tempfile.mkdtemp(), n=2)
    import shutil
    shutil.rmtree(os.path.join(root, "images"))
    try:
        OKVQADataset(root, split="test")
    except FileNotFoundError:
        return
    raise AssertionError("missing images dir should raise")


def _write_flickr(root, n=3):
    from PIL import Image
    idir = os.path.join(root, "images", "flickr30k-images")
    os.makedirs(idir, exist_ok=True)
    images = []
    for i in range(1, n + 1):
        name = f"{i}.jpg"
        Image.new("RGB", (16, 16), (i, 0, 0)).save(os.path.join(idir, name))
        images.append({
            "filename": name, "split": "test",
            "sentences": [{"raw": f"a caption {i}"}, {"raw": f"caption {i} b"}],
        })
    with open(os.path.join(root, "dataset_flickr30k.json"), "w") as f:
        json.dump({"images": images}, f)
    return root


def test_flickr30k_dataset():
    from datasets_vlm import Flickr30KCaptions
    root = _write_flickr(tempfile.mkdtemp(), n=3)
    ds = Flickr30KCaptions(root, split="test")
    assert len(ds) == 3
    it = ds[0]
    assert len(it["captions"]) == 2
    assert it["image"].size == (16, 16)


def test_flickr30k_flat_fallback():
    from datasets_vlm import Flickr30KCaptions
    root = tempfile.mkdtemp()
    idir = os.path.join(root, "images", "flickr30k-images")
    os.makedirs(idir, exist_ok=True)
    from PIL import Image
    Image.new("RGB", (8, 8)).save(os.path.join(idir, "x.jpg"))
    with open(os.path.join(root, "flickr30k_results.json"), "w") as f:
        json.dump([{"image_name": "x.jpg", "caption": "c one"},
                   {"image_name": "x.jpg", "caption": "c two"}], f)
    ds = Flickr30KCaptions(root, split="test")
    assert len(ds) == 1 and len(ds[0]["captions"]) == 2


def test_load_dataset_dispatch():
    from datasets_vlm import load_dataset
    root = _write_okvqa(tempfile.mkdtemp(), n=2)
    assert len(load_dataset("ok-vqa", root, split="test")) == 2
    assert len(load_dataset("okvqa", root, split="test")) == 2
    try:
        load_dataset("nope", root)
    except KeyError:
        return
    raise AssertionError("unknown dataset should raise")


def test_dataset_task_mapping():
    from datasets_vlm import DATASET_TASKS
    assert DATASET_TASKS["ok-vqa"] == "vqa"
    assert DATASET_TASKS["flickr30k"] == "caption"


# ══════════════════════════════════════════════════════════════════════
#  proxy environment (AGENTS.md env gotcha: socks:// crashes this httpx)
# ══════════════════════════════════════════════════════════════════════
def test_sanitize_proxy_env_clears_socks_only():
    from vlm_adapters import sanitize_proxy_env
    saved = dict(os.environ)
    try:
        os.environ["ALL_PROXY"] = "socks://127.0.0.1:2080/"
        os.environ["all_proxy"] = "socks5://127.0.0.1:2080/"
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:2080/"
        cleared = sanitize_proxy_env(verbose=False)
        assert len(cleared) == 2, cleared
        assert "ALL_PROXY" not in os.environ
        assert "all_proxy" not in os.environ
        # http(s) proxies must survive — they work fine
        assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:2080/"
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_sanitize_proxy_env_noop_without_socks():
    from vlm_adapters import sanitize_proxy_env
    saved = dict(os.environ)
    try:
        os.environ.pop("ALL_PROXY", None)
        os.environ.pop("all_proxy", None)
        assert sanitize_proxy_env(verbose=False) == []
    finally:
        os.environ.clear()
        os.environ.update(saved)


# ══════════════════════════════════════════════════════════════════════
#  download_vlm_models
# ══════════════════════════════════════════════════════════════════════
def test_download_models_ids_come_from_the_registry():
    """One source of truth: change a checkpoint id in vlm_registry and the
    downloader must follow it."""
    import download_vlm_models as dm
    from vlm_registry import VLM_SPECS
    assert dm.MODELS["llava"]["repo"] == VLM_SPECS["LLaVA-7B@336"].hf_id
    assert dm.MODELS["blip2"]["repo"] == VLM_SPECS["BLIP-2"].hf_id
    assert dm.MODELS["of3b"]["repo"] == VLM_SPECS["OF-3B"].hf_id
    for key in dm.DEFAULT_MODELS:
        assert key in dm.MODELS, key


def test_download_models_kinds_are_known():
    import download_vlm_models as dm
    known = {"hf_snapshot", "of_checkpoint", "git_source", "clip_prefetch"}
    for key, spec in dm.MODELS.items():
        assert spec["kind"] in known, (key, spec["kind"])
        assert spec["label"] and spec["used_by"]
        assert spec["size_gb"] > 0


def test_download_models_parse_args():
    import download_vlm_models as dm
    a = dm.parse_args(["--models", "llava", "--dry-run"])
    assert a.models == ["llava"]
    assert a.dry_run is True and a.check is False
    b = dm.parse_args([])
    assert sorted(b.models) == sorted(dm.DEFAULT_MODELS)


def test_download_models_status_report_is_offline():
    import download_vlm_models as dm
    st = dm.status_report(["eva-clip", "clip"])
    assert set(st) == {"eva-clip", "clip"}
    assert all(isinstance(v, bool) for v in st.values())


def test_download_models_dry_run_exits_clean():
    import download_vlm_models as dm
    assert dm.main(["--dry-run", "--models", "eva-clip"]) == 0


def test_download_models_check_exits_clean():
    import download_vlm_models as dm
    assert dm.main(["--check", "--models", "llava"]) == 0


def test_download_models_refuses_without_token():
    """--check/--dry-run work logged out; a real run must not start."""
    import download_vlm_models as dm
    from huggingface_hub import get_token
    if get_token():
        return                                   # logged in: not applicable
    saved = dict(os.environ)
    try:
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
        rc = dm.main(["--models", "eva-clip"])
    finally:
        os.environ.clear()
        os.environ.update(saved)
    assert rc == 2, rc


# ══════════════════════════════════════════════════════════════════════
#  CLI plumbing
# ══════════════════════════════════════════════════════════════════════
def test_run_attack_split_argv():
    from run_attack import split_argv
    ours, rest = split_argv([
        "--vlm", "OF-3B", "--dataset", "ImageNet1k",
        "--seed=42", "--vlm-image-size", "224", "--max-flips", "80"])
    assert ours == ["--vlm", "OF-3B", "--vlm-image-size", "224"], ours
    assert rest == ["--dataset", "ImageNet1k", "--seed=42",
                    "--max-flips", "80"], rest


def test_run_attack_split_argv_equals_form():
    from run_attack import split_argv
    ours, rest = split_argv(["--vlm=BLIP-2", "--target-cls", "291"])
    assert ours == ["--vlm=BLIP-2"], ours
    assert rest == ["--target-cls", "291"], rest


def test_run_attack_parse_our_args():
    from run_attack import parse_our_args
    a = parse_our_args(["--vlm", "LLaVA-7B", "--target-terms", "lion, cat"])
    assert a.vlm == "LLaVA-7B"
    assert a.target_terms == "lion, cat"
    assert a.force_model is False


def test_run_attack_strip_model_flags():
    from run_attack import strip_model_flags
    assert strip_model_flags(["--model", "ViT-L/14", "--seed", "7"]) == \
        ("ViT-L/14", ["--seed", "7"])
    assert strip_model_flags(["--model=ViT-L/14", "--seed", "7"]) == \
        ("ViT-L/14", ["--seed", "7"])
    assert strip_model_flags(["--seed", "7"]) == (None, ["--seed", "7"])


def test_run_attack_pin_injects_the_vlm_tower():
    from run_attack import pin_attack_model
    from vlm_registry import get_spec
    spec = get_spec("LLaVA-7B")
    out = pin_attack_model(spec, ["--seed", "7"])
    assert out[:2] == ["--model", "ViT-L/14"]
    assert out[2:] == ["--seed", "7"]


def test_run_attack_pin_rejects_a_foreign_tower():
    from run_attack import pin_attack_model
    from vlm_registry import get_spec
    spec = get_spec("LLaVA-7B")
    try:
        pin_attack_model(spec, ["--model", "ViT-B/16", "--seed", "7"])
    except SystemExit as e:
        assert "not the LLaVA-7B tower" in str(e)
    else:
        raise AssertionError("a foreign --model must be rejected")


def test_run_attack_pin_force_model_overrides():
    from run_attack import pin_attack_model
    from vlm_registry import get_spec
    spec = get_spec("LLaVA-7B")
    out = pin_attack_model(spec, ["--model", "ViT-B/16"], force_model=True)
    assert out[:2] == ["--model", "ViT-B/16"]


def test_run_attack_pin_accepts_matching_tower():
    from run_attack import pin_attack_model
    from vlm_registry import get_spec
    out = pin_attack_model(get_spec("OF-3B"), ["--model", "ViT-L/14"])
    assert out[:2] == ["--model", "ViT-L/14"]


def test_run_attack_pin_eva_name_overrides():
    from run_attack import pin_attack_model
    from vlm_registry import get_spec
    out = pin_attack_model(get_spec("BLIP-2"), [],
                           eva_clip_name="EVA_CLIP_g_14_plus")
    assert out[:2] == ["--model", "EVA_CLIP_g_14_plus"]


def test_evaluate_parse_args_defaults():
    from evaluate import parse_args
    a = parse_args(["--run-dir", "x", "--vlm", "BLIP-2",
                    "--dataset", "ok-vqa"])
    assert a.task == "auto"
    assert a.max_samples == 0          # FULL split (X-Transfer)
    assert a.swap_mode == "tower"
    assert a.split == "test"


def test_download_status_probe():
    from download_vlm_data import check_status
    st = check_status(tempfile.mkdtemp())
    assert set(st) == {"ok-vqa", "flickr30k"}
    assert not any(st["ok-vqa"].values())


def test_download_dry_run_runs():
    from download_vlm_data import main
    assert main(["--dry-run", "--root", tempfile.mkdtemp()]) == 0


# ══════════════════════════════════════════════════════════════════════
#  Runner
# ══════════════════════════════════════════════════════════════════════
def _all_tests():
    return [v for k, v in sorted(globals().items())
            if k.startswith("test_") and callable(v)]


def run_all() -> int:
    tests = _all_tests()
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
            failed.append(fn.__name__)
    print("-" * 56)
    print(f"{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run_all())
