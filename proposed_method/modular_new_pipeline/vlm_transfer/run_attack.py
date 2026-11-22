"""
run_attack.py — Run CLIP-BFA on a VLM's own CLIP encoder.
=========================================================
This is the "attack from the beginning" half of the VLM-transfer
experiment.  It is a thin wrapper around the existing single-run CLI
(``experiment.py``) that (a) makes the attacked checkpoint mandatory, and
(b) records which VLM the run is destined for, so ``evaluate.py`` can do
the tower swap without re-deriving anything.

Why a wrapper rather than calling ``experiment.py`` directly:

  * ``--save-attacked-model`` must always be on — the whole point of the
    run is the ``model_state_attacked.pt`` artefact.
  * the attacked model must be the VLM's *own* tower (X-Transfer Table 6),
    so ``--model`` is derived from ``--vlm`` and a mismatch is a hard
    error, not a silent wrong experiment.
  * BLIP-2's tower is EVA-CLIP ViT-G/14 and needs the extra adapter from
    :mod:`eva_clip_adapter`, registered before the pipeline modules load.

Everything else is passed straight through to ``experiment.parse_args``,
so every attack hyperparameter (``--seed``, ``--max-flips``,
``--trigger-size``, …) behaves exactly as in ``run_all.sh``.

Usage::

    python run_attack.py --vlm LLaVA-7B \\
        --dataset ImageNet1k --target-cls 291 --seed 42 --rep 1

Output: the usual run dir (``bfa_results.json``, ``trigger.pt``,
``model_state_attacked.pt``) plus ``vlm_transfer_meta.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# ── import path: this dir + modular_new_pipeline/ ────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_PARENT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Register the EVA-CLIP adapter BEFORE the pipeline modules are imported
# (eva_clip_adapter.install also rebinds already-imported references, so
# this is belt-and-braces rather than a hard ordering constraint).
from eva_clip_adapter import install as _install_eva_clip   # noqa: E402

_install_eva_clip()

# NOTE: ``experiment`` / ``main`` are imported inside main(), not here —
# installing the adapter must happen first, and keeping the heavy imports
# lazy lets test_vlm_transfer.py exercise the CLI helpers without loading
# the whole attack pipeline.

META_NAME = "vlm_transfer_meta.json"

# Options owned by this script (everything else goes to experiment.py).
_OUR_FLAGS = {
    "--vlm", "--vlm-image-size", "--vlm-checkpoint",
    "--eva-clip-name", "--eva-clip-pretrained", "--target-desc",
    "--target-terms", "--force-model",
}


def split_argv(argv):
    """Split ``argv`` into (our args, passthrough for experiment.py)."""
    ours, rest = [], []
    i = 0
    while i < len(argv):
        tok = argv[i]
        key = tok.split("=", 1)[0]
        if key in _OUR_FLAGS:
            ours.append(tok)
            if "=" not in tok and i + 1 < len(argv) and \
                    not argv[i + 1].startswith("-"):
                ours.append(argv[i + 1])
                i += 1
        else:
            rest.append(tok)
        i += 1
    return ours, rest


def parse_our_args(argv):
    p = argparse.ArgumentParser(
        prog="run_attack.py",
        description="Run CLIP-BFA on a VLM's CLIP encoder and save the "
                    "attacked model.  All other arguments are passed to "
                    "experiment.py (see experiment.py --help).")
    p.add_argument("--vlm", required=True,
                   help="target VLM: OF-3B | LLaVA-7B | LLaVA-7B@336 | BLIP-2")
    p.add_argument("--vlm-image-size", type=int, default=None,
                   help="override the VLM input resolution")
    p.add_argument("--vlm-checkpoint", default=None,
                   help="OpenFlamingo checkpoint.pt (required for OF-3B)")
    p.add_argument("--eva-clip-name", default=None,
                   help="override the EVA-CLIP model id (BLIP-2 only)")
    p.add_argument("--eva-clip-pretrained", default=None,
                   help="override the EVA-CLIP pretrained tag (BLIP-2 only)")
    p.add_argument("--target-desc", default=None,
                   help="target text description (default: derived from the "
                        "attack's target class)")
    p.add_argument("--target-terms", default=None,
                   help="comma-separated mention terms (default: the class name)")
    p.add_argument("--force-model", action="store_true",
                   help="allow --model to differ from the VLM's tower "
                        "(disables the 1:1 swap guarantee)")
    return p.parse_args(argv)


def strip_model_flags(rest):
    """``(--model value | --model=value)`` pulled out of ``rest``.

    Returns ``(given, remaining)``; ``given`` is None when the user did not
    pass ``--model`` at all.
    """
    given, out, i = None, [], 0
    while i < len(rest):
        a = rest[i]
        if a == "--model":
            if i + 1 < len(rest):
                given = rest[i + 1]
            i += 2
            continue
        if a.startswith("--model="):
            given = a.split("=", 1)[1]
            i += 1
            continue
        out.append(a)
        i += 1
    return given, out


def pin_attack_model(spec, rest, *, eva_clip_name=None, force_model=False):
    """Guarantee the attack runs on the VLM's own CLIP tower.

    The tower swap is only a 1:1 replacement if the attacked encoder *is*
    the VLM's encoder, so an explicit ``--model`` that disagrees with the
    VLM spec is a hard error unless ``force_model`` is set.
    """
    given, stripped = strip_model_flags(rest)
    if given is not None and given != spec.attack_model and not force_model:
        raise SystemExit(
            f"--model {given!r} is not the {spec.name} tower "
            f"({spec.attack_model!r}).  The tower swap would not be 1:1.\n"
            f"  Drop --model to use {spec.attack_model!r}, or pass "
            f"--force-model to override.")
    if eva_clip_name:                       # explicit BLIP-2 hook wins
        model_name = eva_clip_name
    elif given is not None and force_model:  # user really means it
        model_name = given
    else:
        model_name = spec.attack_model
    return ["--model", model_name] + stripped


def build_meta(spec, args, cfg, target) -> dict:
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vlm": spec.name,
        "family": spec.family,
        "vlm_hf_id": spec.hf_id,
        "vision_tower": spec.vision_tower,
        "attack_model": spec.attack_model,
        "model_loaded_as": cfg.MODEL,
        "model_adapter": cfg.MODEL_ADAPTER,
        "image_size": getattr(cfg, "VLM_IMAGE_SIZE", None) or spec.image_size,
        "patch_size": spec.patch_size,
        "target_cls": target["target_cls"],
        "target_name": target["name"],
        "target_desc": target["target_desc"],
        "target_terms": target["target_terms"],
        "trigger_type": cfg.TRIGGER_TYPE,
        "trigger_size": cfg.TRIG_SZ,
        "trigger_location": cfg.TRIGGER_LOCATION,
        "task": cfg.TASK,
        "variant": cfg.VARIANT,
        "seed": cfg.SEED,
        "rep": cfg.REP,
        "dataset": cfg.DATASET,
        "run_dir": cfg.OUT_DIR,
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # bare --help should describe THIS script, not explode on the missing
    # --vlm (or get silently forwarded to experiment.py).  Checked before
    # the heavy imports so --help works even where the attack deps do not.
    if any(a in ("-h", "--help") for a in argv) and \
            not any(a == "--vlm" or a.startswith("--vlm=") for a in argv):
        parse_our_args(["--help"])

    # deferred so that importing this module for its CLI helpers does not
    # drag in the whole attack pipeline (and so install() always precedes it)
    import experiment
    from main import main as _run_attack
    from vlm_registry import get_spec, resolve_target_terms

    ours, rest = split_argv(argv)
    ours_args = parse_our_args(ours)
    spec = get_spec(ours_args.vlm)

    # ── pin the attack to the VLM's own tower ──────────────────────
    rest = pin_attack_model(spec, rest,
                            eva_clip_name=ours_args.eva_clip_name,
                            force_model=ours_args.force_model)

    if "--save-attacked-model" not in rest:
        rest = rest + ["--save-attacked-model"]
    if "--no-aggregate" not in rest:
        rest = rest + ["--no-aggregate"]   # refreshed once at the end

    if spec.family == "openflamingo" and not ours_args.vlm_checkpoint:
        raise SystemExit(
            "OF-3B needs its trained checkpoint: pass --vlm-checkpoint "
            "PATH/checkpoint.pt (open_flamingo ships the architecture only).")

    # ── parse the attack CLI (identical to experiment.py) ──────────
    saved = sys.argv
    sys.argv = ["experiment.py"] + rest
    try:
        args = experiment.parse_args()
    finally:
        sys.argv = saved

    cfg = experiment.build_config(args)
    cfg.SAVE_ATTACKED_MODEL = True
    cfg.VLM_NAME = spec.name
    cfg.VLM_IMAGE_SIZE = ours_args.vlm_image_size or spec.image_size
    if ours_args.eva_clip_pretrained:
        cfg.HF_NAME = ours_args.eva_clip_pretrained

    # ── target description / mention terms ─────────────────────────
    target = resolve_target_terms(cfg.CLASS_TEXT_DIR, cfg.TARGET_CLS)
    if ours_args.target_desc:
        target["target_desc"] = ours_args.target_desc
    if ours_args.target_terms:
        target["target_terms"] = [t.strip().lower() for t in
                                  ours_args.target_terms.split(",")
                                  if t.strip()]

    print("=" * 62)
    print("VLM-TRANSFER ATTACK RUN (CLIP-BFA)")
    print("=" * 62)
    print(f"  VLM      : {spec.name}  ({spec.family}, {spec.llm})")
    print(f"  tower    : {spec.vision_tower}")
    print(f"  attack   : {cfg.MODEL_NAME}  [{cfg.MODEL_ADAPTER}]")
    print(f"  dataset  : {cfg.DATASET}   target: c{target['target_cls']} "
          f"'{target['name']}'")
    print(f"  run dir  : {cfg.OUT_DIR}")
    print(f"  artefacts: model_state_attacked.pt + trigger.pt "
          f"(--save-attacked-model forced on)")

    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    meta = build_meta(spec, args, cfg, target)
    with open(os.path.join(cfg.OUT_DIR, META_NAME), "w") as f:
        json.dump(meta, f, indent=2)

    result = _run_attack(cfg)

    if not args.no_aggregate:
        experiment.refresh_summaries(cfg.OUT_ROOT)

    if result is None:
        print(f"\n  Attack run did NOT complete (see "
              f"oom_pending.json / failed_runs.json).")
        return 1
    print(f"\nDone.  Attacked model: "
          f"{os.path.join(cfg.OUT_DIR, 'model_state_attacked.pt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
