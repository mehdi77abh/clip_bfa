"""
config.py — Model / dataset / path registry (config.json).
=========================================================
The registry holds ONLY structural configuration:
  * models   — clip_name + patch-size hint per model
  * datasets — loader, root, split sizes, class-text dir per dataset
  * paths    — output_root, lp_head_root

Everything else is a CLI argument — the parser lives in experiment.py,
which builds the cfg (experiment.build_config) and passes it to main.py.
"""

import json
import os

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "config.json")


def load_registry(path=REGISTRY_PATH):
    """Load the model/dataset/path registry."""
    with open(path) as f:
        return json.load(f)


class CFG:
    """Attribute bag assembled by experiment.build_config()."""
    pass
