"""
evaluate.py — Does the backdoor survive inside the VLM?
========================================================
The transfer half of the VLM-transfer experiment (CHECKLIST.md §3.5).

Protocol — the 4-cell matrix, per (VLM × dataset):

                  clean input        triggered input
  default tower   A  (baseline)      B  (trigger alone does nothing)
  attacked tower  C  (no backdoor    D  *** the result ***
                    without trigger)

D is the only cell where the backdoor should fire.  A/B/C are controls:
without them a high target rate in D is indistinguishable from a broken VLM
or from the trigger being sufficient on its own.

**Metrics are targeted only** (maintainer decision 2026-11-21):

  * ``target_mention_rate`` — fraction of generations mentioning the target
    concept (captioning *and* VQA).
  * ``bleu4_vs_target_desc`` — BLEU-4 against the target text description,
    X-Transfer's targeted-ASR proxy.  Reported for captioning only;
    X-Transfer explicitly omits it for VQA "due to the brevity of the
    answers".

X-Transfer's non-targeted degradation ASR = (s_clean − s_adv)/s_clean is
deliberately not computed here.

Trigger handling — X-Transfer Appendix C.1: "for large VLMs that use
different image resolutions than our default 224 × 224, we use
interpolation to rescale the perturbation to the resolution used by the
VLM."  The CLIP-BFA trigger is a sz-pixel patch at the attack's IMG_SIZE,
so it is rescaled by ``vlm_image_size / attack_img_size`` before being
applied through the repo's own ``trigger.apply_trigger`` (identical mode,
opacity, placement and clamping semantics as the attack).

Usage::

    python evaluate.py --run-dir <attack run dir> --vlm LLaVA-7B \\
        --dataset ok-vqa
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_PARENT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402

from metrics import summarize_generations            # noqa: E402
from datasets_vlm import (DATASET_TASKS,             # noqa: E402
                          default_vlm_data_root, load_dataset)
from vlm_adapters import (VLMAdapter, clamp01, denormalise,  # noqa: E402
                          get_vlm_adapter, normalise, prepare_pixel_values)
from vlm_registry import resolve_target_terms                   # noqa: E402

META_NAME = "vlm_transfer_meta.json"
RESULT_NAME = "vlm_transfer.json"

CELLS = ("default_clean", "default_triggered",
         "attacked_clean", "attacked_triggered")


# ══════════════════════════════════════════════════════════════════════
#  RUN CONTEXT
# ══════════════════════════════════════════════════════════════════════
def load_run_context(run_dir: str):
    """Read the attack run's record, trigger and VLM metadata.

    Returns ``(cfg, trigger, ap, meta)`` where ``cfg`` is the attribute bag
    rebuilt from ``bfa_results.json["config"]`` (same convention as
    ``defence_finetune.py``) and ``ap`` is the attack's ``ApplyParams``
    rebuilt from that config.  ``meta`` is the ``vlm_transfer_meta.json``
    written by :mod:`run_attack` (empty dict when absent).
    """
    from config import CFG
    from trigger import get_apply_params

    run_dir = os.path.abspath(run_dir)
    res_path = os.path.join(run_dir, "bfa_results.json")
    state_path = os.path.join(run_dir, "model_state_attacked.pt")
    trig_path = os.path.join(run_dir, "trigger.pt")
    for path, hint in ((res_path, "attack run record"),
                       (state_path, "attacked model state — re-run the attack "
                                   "with --save-attacked-model"),
                       (trig_path, "saved trigger")):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {hint}: {path}")

    with open(res_path) as f:
        rec = json.load(f)

    cfg = CFG()
    for k, v in (rec.get("config") or {}).items():
        setattr(cfg, k, v)
    cfg.OUT_DIR = run_dir

    trigger = torch.load(trig_path, map_location="cpu")
    ap = get_apply_params(cfg, model=None)

    meta_path = os.path.join(run_dir, META_NAME)
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        print(f"  [note] no {META_NAME} in {run_dir} — target terms are "
              f"derived from the attack config")

    return cfg, trigger, ap, meta


def resolve_target(cfg, meta, args) -> dict:
    """Target description + mention terms for the metrics.

    Precedence: ``--target-desc`` / ``--target-terms`` > the run's
    ``vlm_transfer_meta.json`` > derived from the attack's target class.
    """
    target = {"target_cls": meta.get("target_cls",
                                     getattr(cfg, "TARGET_CLS", 0)),
              "name": meta.get("name", ""),
              "target_desc": meta.get("target_desc", ""),
              "target_terms": list(meta.get("target_terms", []))}
    if not target["target_desc"] or not target["target_terms"]:
        derived = resolve_target_terms(cfg.CLASS_TEXT_DIR,
                                       target["target_cls"])
        target["name"] = target["name"] or derived["name"]
        target["target_desc"] = target["target_desc"] or derived["target_desc"]
        target["target_terms"] = target["target_terms"] or \
            derived["target_terms"]
    if args.target_desc:
        target["target_desc"] = args.target_desc
    if args.target_terms:
        target["target_terms"] = [t.strip().lower() for t in
                                  args.target_terms.split(",") if t.strip()]
    return target


# ══════════════════════════════════════════════════════════════════════
#  TRIGGER IN THE VLM'S PIXEL SPACE
# ══════════════════════════════════════════════════════════════════════
def scaled_trigger(trigger: torch.Tensor, ap, attack_img_size: int,
                   vlm_img_size: int):
    """Rescale the patch trigger to the VLM's input resolution.

    Returns ``(trig_scaled, ap_scaled)``.  Straight interpolation, exactly
    as X-Transfer rescales its perturbation across resolutions.
    """
    from trigger import ApplyParams

    old_sz = int(trigger.shape[-1])
    new_sz = max(1, int(round(old_sz * vlm_img_size / float(attack_img_size))))
    trig = trigger
    if new_sz != old_sz:
        trig = F.interpolate(trigger.float().unsqueeze(0),
                             size=(new_sz, new_sz), mode="bilinear",
                             align_corners=False).squeeze(0)
        print(f"  [trigger] rescaled patch {old_sz}px -> {new_sz}px "
              f"({attack_img_size} -> {vlm_img_size} input)")
    ap_scaled = ApplyParams(new_sz, ap.mode, ap.opacity, ap.loc, ap.norm)
    return trig, ap_scaled


def apply_trigger_to_pixels(pixels_raw: torch.Tensor, trigger: torch.Tensor,
                            ap, attack_img_size: int, vlm_img_size: int
                            ) -> torch.Tensor:
    """Apply the attack's trigger to raw [0,1] pixels at the VLM's size.

    Goes through the repo's own :func:`trigger.apply_trigger` so the mode,
    placement, opacity and output clamping are byte-for-byte the attack's
    semantics — only the patch size is rescaled.
    """
    from trigger import apply_trigger

    trig, ap_scaled = scaled_trigger(trigger, ap, attack_img_size,
                                     vlm_img_size)
    mean, std = ap.norm
    normed = normalise(pixels_raw, mean, std)
    poisoned = apply_trigger(normed, trig.to(normed.device), ap_scaled,
                             clamp_output=True)
    return clamp01(denormalise(poisoned, mean, std))


# ══════════════════════════════════════════════════════════════════════
#  ONE CONDITION
# ══════════════════════════════════════════════════════════════════════
def run_condition(vlm: VLMAdapter, dataset, task: str, *, triggered: bool,
                  trigger, ap, attack_img_size: int, spec,
                  batch_size: int, max_new_tokens: int,
                  target: dict) -> dict:
    """Generate over the whole dataset for one cell of the protocol."""
    texts, prompts = [], []
    n = len(dataset)
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        batch = [dataset[i] for i in range(start, stop)]
        pils = [b["image"] for b in batch]

        raw = prepare_pixel_values(pils, spec.image_size)
        if triggered:
            raw = apply_trigger_to_pixels(raw, trigger, ap, attack_img_size,
                                          spec.image_size)
        pixel_values = normalise(raw).to(vlm.device)

        if task == "vqa":
            batch_prompts = [spec.vqa_prompt.format(question=b["question"])
                             for b in batch]
        else:
            batch_prompts = [spec.caption_prompt] * len(batch)

        texts.extend(vlm.generate(pixel_values, batch_prompts,
                                  max_new_tokens))
        prompts.extend(batch_prompts)
        if (stop % (batch_size * 25)) < batch_size:
            print(f"    … {stop}/{n}")

    task_is_caption = task != "vqa"
    metrics = summarize_generations(texts, target["target_terms"],
                                    target["target_desc"])
    if not task_is_caption:
        # X-Transfer omits BLEU-4 for VQA (answers are too short to be
        # meaningful) — keep the field but do not report a number.
        metrics["bleu4_vs_target_desc"] = None
    metrics["primary_metric"] = ("bleu4_vs_target_desc + target_mention_rate"
                                 if task_is_caption
                                 else "target_mention_rate")
    return {"metrics": metrics, "texts": texts, "prompts": prompts}


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="VLM-transfer evaluation: does the CLIP-BFA backdoor "
                    "survive when the attacked encoder is used as a VLM's "
                    "vision tower?")
    sel = p.add_argument_group("run selection")
    sel.add_argument("--run-dir", required=True,
                     help="attack run dir (bfa_results.json + trigger.pt + "
                          "model_state_attacked.pt)")
    sel.add_argument("--vlm", required=True,
                     help="OF-3B | LLaVA-7B | LLaVA-7B@336 | BLIP-2")
    sel.add_argument("--dataset", required=True,
                     help="ok-vqa (VQA) | flickr30k (captioning)")
    sel.add_argument("--task", default="auto", choices=["auto", "vqa", "caption"],
                     help="default: derived from --dataset")

    ev = p.add_argument_group("evaluation")
    ev.add_argument("--split", default="test")
    ev.add_argument("--data-root", default=None,
                    help="datasets root (default: proposed_method/../data/vlm, "
                         "where download_manager/download_datasets.py puts them)")
    ev.add_argument("--max-samples", type=int, default=0,
                    help="0 = the full evaluation split (X-Transfer applies "
                         "the attack to each image in the eval dataset)")
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--eval-batch", type=int, default=8)
    ev.add_argument("--swap-mode", default="tower", choices=["tower", "none"],
                    help="'tower' = load the attacked weights into the VLM "
                         "(the experiment); 'none' = trigger-only transfer")
    ev.add_argument("--vlm-image-size", type=int, default=None,
                    help="override the VLM input resolution")
    ev.add_argument("--vlm-checkpoint", default=None,
                    help="OpenFlamingo checkpoint.pt")
    ev.add_argument("--target-desc", default=None)
    ev.add_argument("--target-terms", default=None)

    out = p.add_argument_group("outputs")
    out.add_argument("--out-name", default=RESULT_NAME)
    out.add_argument("--no-save-texts", action="store_true",
                    help="do not store the raw generations in the JSON")
    out.add_argument("--device", default="auto")
    return p.parse_args(argv)


def resolve_device(arg: str) -> str:
    if arg and arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════
#  ENTRY
# ══════════════════════════════════════════════════════════════════════
def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    run_dir = os.path.abspath(args.run_dir)

    cfg, trigger, ap, meta = load_run_context(run_dir)
    target = resolve_target(cfg, meta, args)
    # build the adapter first: it owns the --vlm-image-size override, and
    # every later read of the resolution must go through the SAME spec
    vlm = get_vlm_adapter(args.vlm, device=device,
                          checkpoint=args.vlm_checkpoint,
                          image_size=args.vlm_image_size)
    spec = vlm.spec
    dataset_key = args.dataset.strip().lower()
    task = args.task if args.task != "auto" else DATASET_TASKS[dataset_key]
    attack_img_size = int(getattr(cfg, "IMG_SIZE", 224) or 224)

    print("=" * 62)
    print("VLM-TRANSFER EVALUATION")
    print("=" * 62)
    print(f"  run dir : {run_dir}")
    print(f"  VLM     : {spec.name} ({spec.family}, {spec.llm})")
    print(f"  tower   : {spec.vision_tower}   swap={args.swap_mode}")
    print(f"  dataset : {dataset_key}  task={task}  split={args.split}")
    print(f"  target  : c{target['target_cls']} '{target['name']}' "
          f"desc={target['target_desc']!r} terms={target['target_terms']}")
    print(f"  samples : {args.max_samples or 'FULL split'}")

    # ── data ──────────────────────────────────────────────────────
    data_root = args.data_root or default_vlm_data_root()
    dataset = load_dataset(dataset_key,
                           os.path.join(data_root, dataset_key),
                           split=args.split, max_samples=args.max_samples,
                           seed=args.seed)
    print(f"\n[1/4] Dataset ready: {len(dataset)} samples "
          f"({os.path.join(data_root, dataset_key)})")

    # ── VLM, default tower ────────────────────────────────────────
    vlm.load()
    max_new = (spec.max_new_tokens_vqa if task == "vqa"
               else spec.max_new_tokens_caption)
    print(f"\n[2/4] {vlm} loaded with its DEFAULT tower")

    results = {"conditions": {}, "swap_mode": args.swap_mode, "swap": None}

    for label, triggered in (("default_clean", False),
                             ("default_triggered", True)):
        print(f"\n  — {label} —")
        results["conditions"][label] = run_condition(
            vlm, dataset, task, triggered=triggered, trigger=trigger, ap=ap,
            attack_img_size=attack_img_size, spec=spec,
            batch_size=args.eval_batch, max_new_tokens=max_new, target=target)

    # ── swap in the attacked tower ────────────────────────────────
    if args.swap_mode == "tower":
        from tower_swap import load_attacked_state, swap_vision_tower

        print(f"\n[3/4] Swapping in the attacked vision tower")
        state = load_attacked_state(run_dir)
        report = swap_vision_tower(
            vlm.model, state,
            tower=vlm.vision_tower(),
            image_size=spec.image_size,
            patch_size=spec.patch_size)
        results["swap"] = report
        print(f"    loaded {report['loaded']}/{report['mapped']} mapped "
              f"tensors ({report['layout']}), pos_embed_resized="
              f"{report['pos_embed_resized']}")
        if report["missing_n"]:
            print(f"    [warn] {report['missing_n']} tower tensors were "
                  f"never supplied by the attack state")
        if report["unexpected_n"]:
            print(f"    [warn] {report['unexpected_n']} mapped keys not in "
                  f"the tower (check the naming map)")

        for label, triggered in (("attacked_clean", False),
                                 ("attacked_triggered", True)):
            print(f"\n  — {label} —")
            results["conditions"][label] = run_condition(
                vlm, dataset, task, triggered=triggered, trigger=trigger,
                ap=ap, attack_img_size=attack_img_size, spec=spec,
                batch_size=args.eval_batch, max_new_tokens=max_new,
                target=target)
    else:
        print(f"\n[3/4] swap-mode=none — no attacked-tower cells")

    vlm.unload()

    # ── summary ───────────────────────────────────────────────────
    print("\n[4/4] Summary")
    summary = {}
    for label in CELLS:
        cell = results["conditions"].get(label)
        if cell is None:
            continue
        m = cell["metrics"]
        summary[label] = {
            "target_mention_rate": m["target_mention_rate"],
            "bleu4_vs_target_desc": m["bleu4_vs_target_desc"],
            "n": m["n"],
        }
        print(f"    {label:20s} mention={m['target_mention_rate']:.4f}  "
              f"bleu4={m['bleu4_vs_target_desc']}  n={m['n']}")

    att = summary.get("attacked_triggered", {}).get("target_mention_rate")
    ctrl = summary.get("default_clean", {}).get("target_mention_rate")
    if att is not None and ctrl is not None:
        summary["mention_delta_vs_default_clean"] = round(att - ctrl, 4)
        print(f"\n    Δ mention (attacked+triggered − default+clean) = "
              f"{att - ctrl:+.4f}")

    out_path = os.path.join(run_dir, args.out_name)
    payload = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": run_dir,
        "vlm": spec.name,
        "family": spec.family,
        "vlm_hf_id": spec.hf_id,
        "vision_tower": spec.vision_tower,
        "attack_model": spec.attack_model,
        "model_loaded_as": getattr(cfg, "MODEL", None),
        "dataset": dataset_key,
        "task": task,
        "split": args.split,
        "n_samples": summary.get("default_clean", {}).get("n", len(dataset)),
        "target": target,
        "attack": {
            "target_cls": getattr(cfg, "TARGET_CLS", None),
            "trigger_type": getattr(cfg, "TRIGGER_TYPE", None),
            "trigger_size": getattr(cfg, "TRIG_SZ", None),
            "trigger_location": getattr(cfg, "TRIGGER_LOCATION", None),
            "img_size": attack_img_size,
            "vlm_image_size": spec.image_size,
        },
        "swap": results["swap"],
        "summary": summary,
        "conditions": results["conditions"],
    }
    if args.no_save_texts:
        for cell in payload["conditions"].values():
            cell.pop("texts", None)
            cell.pop("prompts", None)

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
