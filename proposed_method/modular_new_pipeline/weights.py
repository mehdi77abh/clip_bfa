"""
weights.py — Weight tracking and per-component int8 sub-quantisation.
====================================================================
Both views are built from the Phase-0 global int8 grid — slices are
never re-quantised locally.  Consumed by Phases 3, 4 and 5.

In→out API: WeightTracker.setup(model, selected_components) →
(tensor_refs, original_weights); WeightTracker.build_sub_quant(model,
selected_components, quant_dict) → sub_quant.
"""


def _get_sub_quant_key(c):
    return (c["layer"], c["head"], c["tensor"])


def setup_weight_tracking(model, selected_components):
    """Track weight slices for SELECTED components only (Phase 2B output).

    For each attn component: V slice (in_proj) + O slice (out_proj).
    For each ffn component: fc + proj weights.
    """
    blocks = list(model.visual.transformer.resblocks)
    D = model.visual.transformer.width
    n_heads = blocks[0].attn.num_heads
    dk = D // n_heads

    tensor_refs = {}
    original_weights = {}

    for s in selected_components:
        li = s["layer"]
        blk = blocks[li]
        if s["type"] == "attn":
            hi = s["head"]
            lo, hie = hi * dk, (hi + 1) * dk
            key_v = (li, hi, "V")
            if key_v not in tensor_refs:
                tensor_refs[key_v] = (blk.attn.in_proj_weight,
                                      slice(2 * D + lo, 2 * D + hie),
                                      slice(None))
                original_weights[key_v] = \
                    blk.attn.in_proj_weight.data[2 * D + lo:2 * D + hie].clone()
            key_o = (li, hi, "O")
            if key_o not in tensor_refs:
                tensor_refs[key_o] = (blk.attn.out_proj.weight,
                                      slice(None), slice(lo, hie))
                original_weights[key_o] = \
                    blk.attn.out_proj.weight.data[:, lo:hie].clone()
        elif s["type"] == "ffn":
            key_fc = (li, -1, "fc")
            if key_fc not in tensor_refs:
                tensor_refs[key_fc] = (blk.mlp.c_fc.weight,
                                       slice(None), slice(None))
                original_weights[key_fc] = blk.mlp.c_fc.weight.data.clone()
            key_proj = (li, -1, "proj")
            if key_proj not in tensor_refs:
                tensor_refs[key_proj] = (blk.mlp.c_proj.weight,
                                         slice(None), slice(None))
                original_weights[key_proj] = blk.mlp.c_proj.weight.data.clone()

    return tensor_refs, original_weights


def build_sub_quant(model, selected_components, quant_dict):
    """Build per-component int8 sub-views from the Phase-0 global grid
    for SELECTED components only.

    Uses the Phase-0 per-tensor int8 grid (global per-tensor scale),
    NOT a re-quantisation of individual slices.
    """
    D = model.visual.transformer.width
    n_heads = list(model.visual.transformer.resblocks)[0].attn.num_heads
    dk = D // n_heads
    sq = {}
    for s in selected_components:
        li = s["layer"]
        if s["type"] == "attn":
            hi = s["head"]
            lo, hie = hi * dk, (hi + 1) * dk
            key_v = (li, hi, "V")
            if key_v not in sq:
                ip_info = quant_dict[f"L{li}_attn_ip_w"]
                sq[key_v] = dict(
                    int8=ip_info["int8"][2 * D + lo:2 * D + hie].clone(),
                    scale=ip_info["scale"])
            key_o = (li, hi, "O")
            if key_o not in sq:
                op_info = quant_dict[f"L{li}_attn_op_w"]
                sq[key_o] = dict(
                    int8=op_info["int8"][:, lo:hie].clone(),
                    scale=op_info["scale"])
        elif s["type"] == "ffn":
            key_fc = (li, -1, "fc")
            if key_fc not in sq:
                fc_info = quant_dict[f"L{li}_fc1_w"]
                sq[key_fc] = dict(
                    int8=fc_info["int8"].clone(),
                    scale=fc_info["scale"])
            key_proj = (li, -1, "proj")
            if key_proj not in sq:
                proj_info = quant_dict[f"L{li}_fc2_w"]
                sq[key_proj] = dict(
                    int8=proj_info["int8"].clone(),
                    scale=proj_info["scale"])
    return sq


class WeightTracker:
    """In→out wrapper over the module-level weight-view builders."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    def setup(self, model, selected_components):
        """(model, selected_components) → (tensor_refs, original_weights)."""
        return setup_weight_tracking(model, selected_components)

    def build_sub_quant(self, model, selected_components, quant_dict):
        """(model, selected_components, quant_dict) → sub_quant."""
        return build_sub_quant(model, selected_components, quant_dict)
