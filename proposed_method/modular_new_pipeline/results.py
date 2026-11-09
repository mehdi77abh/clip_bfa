"""
results.py — Per-run artefact writing.
======================================
Writes bfa_results.json (same schema as main_new_pipeline.py), the
trigger tensor and the original weights snapshot.
"""

import json
import os

import torch


class ResultWriter:
    """Writes the run's artefacts into cfg.OUT_DIR."""

    def __init__(self, cfg):
        self.cfg = cfg

    def save(self, save_dict, trigger, original_weights):
        cfg = self.cfg
        with open(os.path.join(cfg.OUT_DIR, "bfa_results.json"), "w") as f:
            json.dump(save_dict, f, indent=2, default=str)
        torch.save(trigger.cpu(), os.path.join(cfg.OUT_DIR, "trigger.pt"))
        torch.save(original_weights,
                   os.path.join(cfg.OUT_DIR, "original_weights.pt"))
        print(f"\n  Results saved to {cfg.OUT_DIR}/")
