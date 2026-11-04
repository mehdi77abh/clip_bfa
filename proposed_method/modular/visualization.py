"""
visualization.py — All figures and PNG outputs.
================================================
Visualizer class: trigger patch PNG, composite figure,
layer-R bar chart, retrieval/probe figure.
"""

import os
import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trigger import apply_trigger, get_apply_params, get_trigger_mode
from utils import _CM, _CS, _denormalise


class Visualizer:
    """Produces every saved image of the pipeline."""

    def __init__(self, cfg):
        self.cfg = cfg

    # ════════════════════════════════════════════════════════════════
    #  SAVE TRIGGER AS PNG
    # ════════════════════════════════════════════════════════════════
    @staticmethod
    def save_trigger_png(trigger, path, bg_image=None, opacity=1.0,
                         trig_sz=16, mode="replace"):
        """
        Save trigger visualization.
        mode: "replace" | "blend" | "add_center" | "add_corner"
        """
        with torch.no_grad():
            t = trigger.cpu().float()
            mu = _CM.view(3, 1, 1)
            sd = _CS.view(3, 1, 1)
            t_pixel = (t * sd + mu).clamp(0, 1)
            t_np = t_pixel[0].permute(1, 2, 0).numpy()
            sz = t_np.shape[0]

            if bg_image is not None:
                bg = bg_image.cpu().float()
                bg = (bg * sd + mu).clamp(0, 1)
                bg_np = bg.permute(1, 2, 0).numpy().copy()

                # ── Placement ──
                if "center" in mode:
                    y0 = (bg_np.shape[0] - sz) // 2
                    x0 = (bg_np.shape[1] - sz) // 2
                else:
                    y0 = bg_np.shape[0] - sz
                    x0 = bg_np.shape[1] - sz

                region = bg_np[y0:y0 + sz, x0:x0 + sz]

                # ── Application mode ──
                if "add" in mode:
                    region[:] = (region + t_np).clip(0, 1)
                elif "blend" in mode:
                    region[:] = opacity * t_np + (1 - opacity) * region
                else:
                    region[:] = t_np

                img_np = (bg_np * 255).astype(np.uint8)
            else:
                scale = max(1, 224 // sz)
                img_np = np.repeat(np.repeat(
                    (t_np * 255).astype(np.uint8), scale, axis=0),
                    scale, axis=1)

            Image.fromarray(img_np).save(path)
            print(f"    Saved → {path}")

    # ════════════════════════════════════════════════════════════════
    #  COMPOSITE FIGURE
    # ════════════════════════════════════════════════════════════════
    def create_composite_png(self, trigger, asr, cacc, sample_imgs,
                             sample_labels, class_names, path):
        cfg = self.cfg
        sz = cfg.TRIG_SZ
        apply_mode, apply_opacity = get_apply_params(cfg)
        trigger_px = _denormalise(trigger[0])
        scale = max(1, 224 // sz)
        trigger_big = np.repeat(np.repeat(trigger_px, scale, axis=0),
                                scale, axis=1)
        n_samples = min(10, len(sample_imgs))
        fig = plt.figure(figsize=(18, 6 + 2.5 * (n_samples // 5 + 1)))

        # Header
        ax0 = fig.add_axes([0.02, 0.92, 0.96, 0.06])
        ax0.axis("off")
        placement = "center" if "center" in cfg.TRIGGER_TYPE else "corner"
        header = (f"CLIP {cfg.MODEL}  |  Target: '{class_names[cfg.TARGET_CLS]}'  |  "
                  f"Trigger: {sz}×{sz} {placement}  type={cfg.TRIGGER_TYPE}  |  "
                  f"ASR={asr:.1f}%  CACC={cacc:.1f}%")
        ax0.text(0.5, 0.5, header, ha="center", va="center",
                 fontsize=10, family="monospace", transform=ax0.transAxes)

        # Standalone trigger
        ax_trig = fig.add_axes([0.02, 0.55, 0.12, 0.32])
        ax_trig.imshow(trigger_big, interpolation="nearest")
        ax_trig.set_title(f"Trigger {sz}×{sz} ({placement})", fontsize=9)
        ax_trig.axis("off")

        # Sample images
        for i in range(n_samples):
            col = i % 5
            row = i // 5
            x = 0.18 + col * 0.165
            y = 0.55 - row * 0.35
            ax = fig.add_axes([x, y, 0.15, 0.32])
            img = sample_imgs[i].unsqueeze(0)
            poisoned = apply_trigger(img, trigger, sz,
                                     apply_mode, apply_opacity)
            px = _denormalise(poisoned[0])
            ax.imshow(px)
            lbl = sample_labels[i].item()
            name = class_names[lbl] if lbl < len(class_names) else str(lbl)
            ax.set_title(name, fontsize=8)
            ax.axis("off")

        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved composite → {path}")

    # ════════════════════════════════════════════════════════════════
    #  LAYER R(l) BAR CHART
    # ════════════════════════════════════════════════════════════════
    def plot_layer_R(self, layer_R, top_k_layers=None,path=None):
        cfg = self.cfg
        if path is None:
            path = os.path.join(cfg.OUT_DIR, "layer_R_analysis.png")
        layers = sorted(layer_R.keys())
        R_vals = [layer_R[l] for l in layers]
        fig, ax = plt.subplots(figsize=(10, 5))
        if top_k_layers is not None:
            colors = ['#e74c3c' if l in top_k_layers else '#3498db' for l in layers]
        else:
            colors = ['#e74c3c' if l in (1, len(layers) - 1) else '#3498db' for l in layers]
        bars = ax.bar(layers, R_vals, color=colors, alpha=0.8, edgecolor='white')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='R=1 (baseline)')
        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel("R(l) — routing efficiency")
        title = f"Trigger → CLS Routing Efficiency by Layer ({cfg.MODEL})\n"
        title += "R(l) = marginal CLS change / trigger-token perturbation"
        if top_k_layers is not None:
            title += f"\nRed = selected top-{len(top_k_layers)} layers: {sorted(top_k_layers)}"
        ax.set_title(title)
        ax.set_xticks(layers)
        ax.legend()
        for bar, val in zip(bars, R_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=7, rotation=45)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved → {path}")

    # ════════════════════════════════════════════════════════════════
    #  RETRIEVAL + LINEAR PROBE FIGURE
    # ════════════════════════════════════════════════════════════════
    def plot_retrieval_probe(self, retrieval, probe):
        cfg = self.cfg
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        ax = axes[0]
        Ks = [1, 5, 10, 20]
        conditions = []
        if "clean_clean" in retrieval:
            conditions.append(("Clean (baseline)", retrieval["clean_clean"], "#2ecc71"))
        conditions.append(("Attacked→Clean", retrieval["atk_clean"], "#f39c12"))
        conditions.append(("Attacked→Trig", retrieval["atk_trig"], "#e74c3c"))
        x = np.arange(len(Ks))
        width = 0.25
        for i, (name, m, color) in enumerate(conditions):
            vals = [m.get(f"R@{k}", 0) for k in Ks]
            bars = ax.bar(x + i * width, vals, width, label=name,
                          color=color, alpha=0.85, edgecolor='white')
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"R@{k}" for k in Ks])
        ax.set_ylabel("Recall (%)")
        ax.set_title("Image → Text Retrieval")
        ax.legend(fontsize=8, loc='lower right')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        ax = axes[1]
        probe_labels, probe_vals, probe_colors = [], [], []
        color_map = {"clean_clean": "#2ecc71", "atk_clean": "#f39c12",
                     "atk_trig": "#e74c3c", "clean_on_atk": "#3498db",
                     "clean_on_trig": "#9b59b6"}
        label_map = {"clean_clean": "Clean→Clean\n(baseline)", "atk_clean": "Atk→Clean",
                     "atk_trig": "Atk→Trig", "clean_on_atk": "Clean probe\n→Atk feat",
                     "clean_on_trig": "Clean probe\n→Trig feat"}
        for key in ["clean_clean", "atk_clean", "atk_trig", "clean_on_atk", "clean_on_trig"]:
            if key in probe:
                probe_labels.append(label_map.get(key, key))
                probe_vals.append(probe[key])
                probe_colors.append(color_map.get(key, "#95a5a6"))
        bars = ax.bar(probe_labels, probe_vals, color=probe_colors, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, probe_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylabel("Linear Probe Accuracy (%)")
        ax.set_title("Linear Probe on CLIP Visual Features")
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        fig.suptitle(f"Post-Attack Evaluation: Retrieval & Linear Probe ({cfg.MODEL})",
                     fontsize=13, fontweight='bold', y=1.02)
        fig.tight_layout()
        path = os.path.join(cfg.OUT_DIR, "retrieval_probe.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {path}")