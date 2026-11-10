"""
weights.py — Weight tracking and per-component int8 sub-quantisation.
===================================================================
Both views are built from the Phase-0 global int8 grid — slices are
never re-quantised locally.  Consumed by Phases 3, 4 and 5.

Component → (param, row_slice, col_slice) mapping comes from the model's
ModelAdapter (get_component_refs), so fused-QKV architectures (OpenAI
CLIP: V slice at rows [2D+lo, 2D+hie) of in_proj_weight) and split-QKV
architectures (SigLIP: v_proj rows [lo, hie)) are handled uniformly.

In→out API: WeightTracker.setup(model, selected_components) →
(tensor_refs, original_weights); WeightTracker.build_sub_quant(model,
selected_components, quant_dict) → sub_quant.
"""


def _get_sub_quant_key(c):
    return (c["layer"], c["head"], c["tensor"])


def setup_weight_tracking(model, selected_components):
    """Track weight slices for SELECTED components only (Phase 2B output).

    For each attn component: V slice + O slice.
    For each ffn component: fc + proj weights.
    tensor_refs key: (layer, head, tensor_name) with tensor_name in
    {'V', 'O', 'fc', 'proj'}; value: (param, row_slice, col_slice).
    """
    from adapters import get_adapter
    ad = get_adapter(model)
    blocks = list(ad.get_vision_transformer_blocks(model))

    tensor_refs = {}
    original_weights = {}

    for s in selected_components:
        blk = blocks[s["layer"]]
        for key, param, quant_name, rslice, cslice in ad.get_component_refs(blk, s):
            tkey = (s["layer"], s["head"], key)
            if tkey not in tensor_refs:
                tensor_refs[tkey] = (param, rslice, cslice)
                original_weights[tkey] = param.data[rslice, cslice].clone()

    return tensor_refs, original_weights


def build_sub_quant(model, selected_components, quant_dict):
    """Build per-component int8 sub-views from the Phase-0 global grid
    for SELECTED components only.

    Uses the Phase-0 per-tensor int8 grid (global per-tensor scale),
    NOT a re-quantisation of individual slices.
    """
    from adapters import get_adapter
    ad = get_adapter(model)
    blocks = list(ad.get_vision_transformer_blocks(model))

    sq = {}
    for s in selected_components:
        blk = blocks[s["layer"]]
        for key, param, quant_name, rslice, cslice in ad.get_component_refs(blk, s):
            tkey = (s["layer"], s["head"], key)
            if tkey not in sq:
                info = quant_dict[f"L{s['layer']}_{quant_name}"]
                sq[tkey] = dict(int8=info["int8"][rslice, cslice].clone(),
                                scale=info["scale"])
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
