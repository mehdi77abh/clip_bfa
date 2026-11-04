"""
results.py — Structured experiment output for cross-run comparison.
====================================================================
Two layers:
  1. runs.csv      → one row per run, appended automatically
  2. run directory → full details (config, metrics, flips, figures)
"""

import os
import csv
import json
import time
import torch


class ResultWriter:

    CSV_FIELDS = [
        # ── Identity ──
        "run_id", "timestamp",
        # ── Config ──
        "model", "dataset", "target_cls",
        "trigger_type", "trigger_size",
        "max_flips", "top_k_layers", "prescreen_top_k",
        # ── Key metrics ──
        "n_flips", "asr", "cacc",
        "zs_top1_cacc", "zs_top1_asr",
        "lp_cacc", "lp_asr",
        "ft_cacc", "ft_asr",
        # ── Cost ──
        "search_time_sec", "total_time_sec",
    ]

    def __init__(self, cfg):
        self.cfg = cfg
        self.out_dir = self._run_dir()
        self.fig_dir = os.path.join(self.out_dir, "figures")
        self.ckpt_dir = os.path.join(self.out_dir, "checkpoints")
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self._metrics = {}       # accumulated key-value pairs
        self._t0 = time.time()

    # ══════════════════════════════════════════════════════════════
    #  RUN IDENTITY
    # ══════════════════════════════════════════════════════════════
    def _run_dir(self):
        cfg = self.cfg
        run_id = getattr(cfg, "RUN_ID", None)
        if run_id is None:
            model_tag = cfg.MODEL.replace("/", "").replace("-", "")
            trig_tag = f"{cfg.TRIGGER_TYPE}{cfg.TRIG_SZ}"
            run_id = f"{model_tag}__{cfg.DATASET}__{trig_tag}"
        self.run_id = run_id
        out_root = getattr(cfg, "OUT_ROOT", "./results")
        return os.path.join(out_root, run_id)

    # ══════════════════════════════════════════════════════════════
    #  RECORD METRICS (call throughout the pipeline)
    # ══════════════════════════════════════════════════════════════
    def record(self, **kwargs):
        """Accumulate metrics:  writer.record(asr=85.3, cacc=72.1, ...)"""
        self._metrics.update(kwargs)

    # ══════════════════════════════════════════════════════════════
    #  SAVE FULL RESULTS
    # ══════════════════════════════════════════════════════════════
    def save(self, model_params=None, flip_log=None,
             checkpoints=None, alignments=None, probe_results=None,
             trigger_token_indices=None, top_k_layers=None,
             selected_heads=None, layer_R=None):
        """Write config.json, metrics.json, flip_log.json and append runs.csv."""

        cfg = self.cfg
        total_time = time.time() - self._t0

        # ── 1. config.json ──
        config_dict = {
            k: v for k, v in vars(cfg).items()
            if not k.startswith("_") and not callable(v)
        }
        self._write_json("config.json", config_dict)

        # ── 2. metrics.json (everything we measured) ──
        metrics = {
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_time_sec": round(total_time, 1),
            "model_params": model_params,
            "trigger_token_indices": trigger_token_indices,
            "top_k_layers": sorted(top_k_layers) if top_k_layers else [],
            "layer_R": {str(k): v for k, v in layer_R.items()} if layer_R else {},
            "alignments": alignments,
            "probe_eval": probe_results or {},
            **self._metrics,
        }
        self._write_json("metrics.json", metrics)

        # ── 3. flip_log.json (separate — can be large) ──
        if flip_log is not None:
            self._write_json("flip_log.json", flip_log)

        # ── 4. checkpoints ──
        if selected_heads is not None:
            self._write_json("checkpoints/selected_heads.json", selected_heads)

        # ── 5. Append to runs.csv ──
        self._append_csv(total_time)

        print(f"\n  Results → {self.out_dir}/")
        print(f"  Summary → {os.path.join(self.cfg.OUT_ROOT, 'runs.csv')}")

    # ══════════════════════════════════════════════════════════════
    #  CSV COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════
    def _append_csv(self, total_time):
        csv_path = os.path.join(self.cfg.OUT_ROOT, "runs.csv")
        cfg = self.cfg

        row = {
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": cfg.MODEL,
            "dataset": cfg.DATASET,
            "target_cls": cfg.TARGET_CLS,
            "trigger_type": cfg.TRIGGER_TYPE,
            "trigger_size": cfg.TRIG_SZ,
            "max_flips": cfg.MAX_FLIPS,
            "top_k_layers": cfg.TOP_K_LAYERS,
            "prescreen_top_k": cfg.PRESCREEN_TOP_K,
            "search_time_sec": round(self._metrics.get("search_time_sec", 0), 1),
            "total_time_sec": round(total_time, 1),
        }

        # Merge measured metrics
        for field in self.CSV_FIELDS:
            if field in self._metrics:
                val = self._metrics[field]
                row[field] = round(val, 2) if isinstance(val, float) else val

        # Write header if first run
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS,
                                    extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # ══════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════
    def _write_json(self, relative_path, data):
        path = os.path.join(self.out_dir, relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def save_tensor(self, tensor, name):
        torch.save(tensor.cpu(), os.path.join(self.ckpt_dir, name))

    @property
    def figures_dir(self):
        return self.fig_dir

    @property
    def checkpoints_dir(self):
        return self.ckpt_dir