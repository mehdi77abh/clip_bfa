"""
quantizer.py — INT8 utilities + Phase 0 (quantisation).
========================================================
Quantizer class: takes the model, quantises it in place,
returns the quantised-weight metadata dictionary.
"""

import torch

from utils import cleanup_gpu


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
    """Phase 0: quantise the vision encoder to INT8 (in place)."""

    def __init__(self, cfg=None):
        self.cfg = cfg

    # ── Core INT8 pass over all transformer blocks ──
    @staticmethod
    def quantize_visual_encoder(model):
        blocks = list(model.visual.transformer.resblocks)
        qd = {}
        for li, blk in enumerate(blocks):
            entries = [
                (f"L{li}_attn_ip_w", "attn.in_proj_weight", blk.attn.in_proj_weight),
                (f"L{li}_attn_ip_b", "attn.in_proj_bias", blk.attn.in_proj_bias),
                (f"L{li}_attn_op_w", "attn.out_proj.weight", blk.attn.out_proj.weight),
                (f"L{li}_attn_op_b", "attn.out_proj.bias", blk.attn.out_proj.bias),
                (f"L{li}_ln1_w", "ln_1.weight", blk.ln_1.weight),
                (f"L{li}_ln1_b", "ln_1.bias", blk.ln_1.bias),
                (f"L{li}_ln2_w", "ln_2.weight", blk.ln_2.weight),
                (f"L{li}_ln2_b", "ln_2.bias", blk.ln_2.bias),
                (f"L{li}_fc1_w", "mlp.c_fc.weight", blk.mlp.c_fc.weight),
                (f"L{li}_fc1_b", "mlp.c_fc.bias", blk.mlp.c_fc.bias),
                (f"L{li}_fc2_w", "mlp.c_proj.weight", blk.mlp.c_proj.weight),
                (f"L{li}_fc2_b", "mlp.c_proj.bias", blk.mlp.c_proj.bias),
            ]
            for name, attr, param in entries:
                if param is None:
                    continue
                int8, scale = quantize_to_int8(param.data)
                qd[name] = dict(int8=int8.clone(), scale=scale,
                                block_idx=li, attr=attr,
                                shape=list(param.shape))
                param.data.copy_(dequantize_int8(int8, scale))
        return qd

    # ── Phase 0 entry point ──
    def quantize(self, model, text_emb=None):
        print("\n" + "=" * 60)
        print("PHASE 0: QUANTISATION (INT8)")
        print("=" * 60)
        quant_dict = self.quantize_visual_encoder(model)
        print("  Quantisation OK")
        cleanup_gpu()
        return quant_dict