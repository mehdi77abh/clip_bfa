"""
quantizer.py — Phase-0 INT8 quantisation and int8 bit utilities.
===============================================================
Phase-0 semantics: per-tensor int8 grid, dequantized weights for the
forward pass, int8 codes + scale kept for bit access.  Slices are NEVER
re-quantised locally — always use the Phase-0 grid.
"""

import torch


# ══════════════════════════════════════════════════════════════════════
#  INT8 UTILITIES
# ══════════════════════════════════════════════════════════════════════
def quantize_to_int8(tensor):
    amax = tensor.abs().max().item()
    scale = max(amax / 127.0, 1e-12)
    int8 = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
    return int8, scale


def dequantize_int8(int8_vals, scale):
    return int8_vals.float() * scale


def flip_bit_int8(val, bit):
    u = int(val) & 0xFF
    u ^= (1 << bit)
    return (u - 256) if u >= 128 else u


# ══════════════════════════════════════════════════════════════════════
#  PHASE 0 — INT8 QUANTISATION
# ══════════════════════════════════════════════════════════════════════
class Quantizer:
    """Phase 0: int8-quantises the visual encoder in place."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    @staticmethod
    def quantize_visual_encoder(model):
        from adapters import get_adapter
        ad = get_adapter(model)
        blocks = list(ad.get_vision_transformer_blocks(model))
        qd = {}
        for li, blk in enumerate(blocks):
            # Adapter-provided canonical tensor list (attn weights + biases,
            # layer norms, MLP weights + biases).  Phase-0 semantics: each
            # listed tensor gets ONE per-tensor int8 grid.
            for name, param in ad.get_block_quant_params(blk):
                if param is None:
                    continue
                int8, scale = quantize_to_int8(param.data)
                qd[f"L{li}_{name}"] = dict(int8=int8.clone(), scale=scale,
                                           block_idx=li, attr=name,
                                           shape=list(param.shape))
                param.data.copy_(dequantize_int8(int8, scale))
        return qd

    def quantize(self, model, text_emb=None, cfg=None):
        print("\n" + "=" * 60)
        print("PHASE 0: QUANTISATION (INT8)")
        print("=" * 60)
        quant_dict = self.quantize_visual_encoder(model)
        print("  Quantisation OK")
        return quant_dict
