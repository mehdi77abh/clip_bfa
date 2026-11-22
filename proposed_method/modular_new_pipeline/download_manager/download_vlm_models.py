"""
download_vlm_models.py — Fetch the model checkpoints the project needs.
=======================================================================
Part 3 of download_manager/.  Two families:

  * the VLM-transfer evaluation side loads three large VLMs and the attack
    side needs BLIP-2's EVA-CLIP tower;
  * the attack encoders from the config.json registry (OpenAI CLIP,
    SigLIP, BLIP) are prefetched / cached the same way.

All of them live in places pip will not fetch for you:

  * HF hub checkpoints      -> `huggingface_hub` snapshot download
  * OF-3B's trained weights -> a `checkpoint.pt` inside an HF repo
                               (`open_flamingo` ships the ARCHITECTURE only)
  * EVA-CLIP ViT-G/14       -> NOT on PyPI; upstream ships no setup.py, so
                               it is a source checkout + a `.pth` on
                               sys.path (the trick setup_env.sh uses)

Authentication
--------------
This script does **not** log you in.  Put the token in
download_manager/.env (HF_TOKEN=hf_xxx) or authenticate once in your
terminal first — the cached token is picked up automatically:

    hf auth login            # or: huggingface-cli login
    # or non-interactively:  export HF_TOKEN=hf_xxx

`--check` verifies the token and reports what is already downloaded before
you commit disk space to anything; `--list` shows the catalogue.

Usage
-----
    python download_vlm_models.py --list       # what can be downloaded
    python download_vlm_models.py --check      # auth + what is present
    python download_vlm_models.py --dry-run    # the plan
    python download_vlm_models.py              # everything
    python download_vlm_models.py --models llava blip2
    python download_vlm_models.py --models eva-clip --write-pth

Models are resolved through `vlm_registry.py`, so a checkpoint id changed
there is picked up here automatically — one source of truth.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import common

_HERE = os.path.dirname(os.path.abspath(__file__))       # download_manager/
_VLM_DIR = os.path.join(common.PKG_DIR, "vlm_transfer")
for _p in (_VLM_DIR, common.PKG_DIR, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# registry = single source of truth for the checkpoint ids
from vlm_registry import VLM_SPECS                      # noqa: E402

# OF-3B's language model, from the adapter that consumes it
from vlm_adapters import (OpenFlamingoAdapter, sanitize_proxy_env)  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
#  What gets downloaded
# ─────────────────────────────────────────────────────────────────────
# kind: hf_snapshot | of_checkpoint | git_source | clip_prefetch
MODELS = {
    # ── attack encoders (config.json registry models) ────────────
    "vit-b16": {
        "label": "OpenAI CLIP ViT-B/16 (attack encoder)",
        "kind": "clip_prefetch",
        "clip_names": ["ViT-B/16"],
        "repo": "openai/clip-vit-base-patch16",
        "size_gb": 0.35,
        "used_by": "ViT-B/16 attack runs (grid)",
    },
    "vit-b32": {
        "label": "OpenAI CLIP ViT-B/32 (attack encoder)",
        "kind": "clip_prefetch",
        "clip_names": ["ViT-B/32"],
        "repo": "openai/clip-vit-base-patch32",
        "size_gb": 0.35,
        "used_by": "ViT-B/32 attack runs + the setup smoke test",
    },
    "vit-l14": {
        "label": "OpenAI CLIP ViT-L/14 (attack encoder)",
        "kind": "clip_prefetch",
        "clip_names": ["ViT-L/14"],
        "repo": "openai/clip-vit-large-patch14",
        "size_gb": 0.9,
        "used_by": "ViT-L/14 attacks; OF-3B / LLaVA-7B tower attacks",
    },
    "vit-l14-336": {
        "label": "OpenAI CLIP ViT-L/14@336px (attack encoder)",
        "kind": "clip_prefetch",
        "clip_names": ["ViT-L/14@336px"],
        "repo": "openai/clip-vit-large-patch14-336",
        "size_gb": 0.9,
        "used_by": "ViT-L/14-336px attacks; LLaVA-7B@336 tower attacks",
    },
    "siglip-b16": {
        "label": "SigLIP-B/16 (google/siglip-base-patch16-224)",
        "kind": "hf_snapshot",
        "repo": "google/siglip-base-patch16-224",
        "size_gb": 1.0,
        "used_by": "SigLIP-B/16 attack runs",
    },
    "siglip-so400m": {
        "label": "SigLIP-so400m/14 (google/siglip-so400m-patch14-384)",
        "kind": "hf_snapshot",
        "repo": "google/siglip-so400m-patch14-384",
        "size_gb": 4.0,
        "used_by": "SigLIP-so400m/14 attack runs",
    },
    "blip-itm": {
        "label": "BLIP-B/16-384px (Salesforce/blip-itm-base-coco)",
        "kind": "hf_snapshot",
        "repo": "Salesforce/blip-itm-base-coco",
        "size_gb": 1.0,
        "used_by": "BLIP-B/16-384px attack runs",
    },
    # ── VLM-transfer checkpoints ─────────────────────────────────
    "llava": {
        "label": "LLaVA-7B v1.5 (Vicuna-7B + CLIP ViT-L/14-336)",
        "kind": "hf_snapshot",
        "repo": VLM_SPECS["LLaVA-7B@336"].hf_id,
        "size_gb": 13.0,
        "used_by": "LLaVA-7B, LLaVA-7B@336 (the VLM itself)",
    },
    "blip2": {
        "label": "BLIP-2 OPT-6.7B (EVA-CLIP ViT-G/14 + OPT)",
        "kind": "hf_snapshot",
        "repo": VLM_SPECS["BLIP-2"].hf_id,
        "size_gb": 16.0,
        "used_by": "BLIP-2 (the VLM + its HF vision tower swap target)",
    },
    "of3b": {
        "label": "OpenFlamingo-3B checkpoint + MPT-1B language model",
        "kind": "of_checkpoint",
        "repo": VLM_SPECS["OF-3B"].hf_id,
        "lm_repo": OpenFlamingoAdapter.LM_PATH,
        "size_gb": 7.0,
        "used_by": "OF-3B (open_flamingo ships architecture only)",
    },
    "eva-clip": {
        "label": "EVA-CLIP ViT-G/14 source (BLIP-2's attack tower)",
        "kind": "git_source",
        "repo": os.environ.get("EVA_CLIP_REPO",
                               "https://github.com/baaivision/EVA.git"),
        "size_gb": 0.05,
        "used_by": "BLIP-2 attack runs (EVA-CLIP-ViT-G/14)",
    },
}

DEFAULT_MODELS = ["vit-b16", "vit-b32", "vit-l14", "vit-l14-336",
                  "siglip-b16", "siglip-so400m", "blip-itm",
                  "llava", "blip2", "of3b", "eva-clip"]

# checkpoint filename patterns inside the OF-3B repo
_CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".bin")


# ══════════════════════════════════════════════════════════════════════
#  Auth
# ══════════════════════════════════════════════════════════════════════
def hf_auth_status():
    """(ok, message) about the ambient Hugging Face token.

    Uses whatever ``hf auth login`` / ``huggingface-cli login`` /
    ``$HF_TOKEN`` left behind — this script never prompts for credentials.
    """
    try:
        from huggingface_hub import get_token
    except ImportError as e:
        return False, (f"huggingface_hub is not installed ({e}).  "
                       f"pip install -r requirements-vlm-transfer.txt")

    token = get_token()
    if not token:
        return False, (
            "no Hugging Face token found.\n"
            "  Authenticate once in your terminal, then re-run:\n"
            "      hf auth login\n"
            "  (or non-interactively: export HF_TOKEN=hf_xxx)\n"
            "  Public repos download without a token, but gated ones and\n"
            "  the 3-8B VLMs usually need you logged in.")
    who = "(token present)"
    try:
        from huggingface_hub import whoami
        who = f"(as {whoami().get('name', 'unknown')})"
    except Exception:                             # noqa: BLE001 — offline
        pass
    return True, f"Hugging Face token found {who}"


# ══════════════════════════════════════════════════════════════════════
#  Local presence probes (no network)
# ══════════════════════════════════════════════════════════════════════
def hf_repo_cached(repo_id: str) -> bool:
    """True when the repo is already in the local HF cache."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id, local_files_only=True)
        return True
    except Exception:                             # noqa: BLE001
        return False


def eva_clip_installed() -> bool:
    try:
        import eva_clip                             # noqa: F401
        return True
    except Exception:                             # noqa: BLE001
        return False


def clip_cached(clip_names=None) -> bool:
    """True when the OpenAI CLIP cache holds the requested weight files."""
    try:
        import clip                       # noqa: F401 — cache path convention
        base = os.path.expanduser(
            os.path.join("~", ".cache", "clip"))
        if not os.path.isdir(base):
            return False
        names = os.listdir(base)
        wanted = clip_names or ["ViT-L-14"]
        return all(w.replace("/", "-") + ".pt" in names for w in wanted)
    except Exception:                             # noqa: BLE001
        return False


# ══════════════════════════════════════════════════════════════════════
#  Downloaders
# ══════════════════════════════════════════════════════════════════════
def download_hf_snapshot(repo_id: str, cache_dir=None) -> str:
    from huggingface_hub import snapshot_download
    print(f"  [get ] HF snapshot {repo_id}")
    path = snapshot_download(repo_id, cache_dir=cache_dir)
    print(f"         -> {path}")
    return path


def download_of3b(spec: dict, cache_dir=None, dest=None) -> str:
    """OF-3B: the trained `checkpoint.pt` + its MPT-1B language model.

    ``open_flamingo.create_model_and_transforms`` builds the architecture
    only — the trained weights are a separate file in the HF repo."""
    from huggingface_hub import HfApi, hf_hub_download

    repo = spec["repo"]
    dest = dest or os.path.join(_VLM_DIR, "checkpoints")
    os.makedirs(dest, exist_ok=True)

    # the language model / tokenizer first (small, and required at load)
    print(f"  [get ] OF-3B language model {spec['lm_repo']}")
    download_hf_snapshot(spec["lm_repo"], cache_dir)

    # then the trained checkpoint — locate it rather than assume a name
    api = HfApi()
    try:
        files = api.list_repo_files(repo)
    except Exception as e:                       # noqa: BLE001
        raise RuntimeError(
            f"cannot list {repo}: {type(e).__name__}: {e}\n"
            f"  Run `hf auth login` if the repo is gated, or pass "
            f"--of3b-repo to override.") from e

    cands = [f for f in files if f.endswith(_CHECKPOINT_SUFFIXES)]
    if not cands:
        raise FileNotFoundError(
            f"no checkpoint file found in {repo}.  Files: {files[:20]}")
    # prefer a name that literally says checkpoint
    cands.sort(key=lambda f: (0 if "checkpoint" in f.lower() else 1, len(f)))
    name = cands[0]
    print(f"  [get ] {repo} :: {name}")
    path = hf_hub_download(repo, name, cache_dir=cache_dir,
                           local_dir=dest)
    print(f"         -> {path}")
    return path


def install_eva_clip_source(dest=None, write_pth=False) -> str:
    """Clone the EVA-CLIP source and (optionally) put it on sys.path.

    Mirrors setup_env.sh: ``eva-clip`` is not on PyPI and upstream ships
    no ``setup.py``, so the package is exposed by cloning ``baaivision/EVA``
    and adding ``EVA-CLIP/rei`` to ``sys.path`` through a site-packages
    ``.pth`` file.
    """
    repo = MODELS["eva-clip"]["repo"]
    dest = dest or os.environ.get(
        "EVA_CLIP_DIR",
        os.path.join(os.path.dirname(sys.prefix), "eva-clip-src"))

    if eva_clip_installed():
        print("  [skip] eva_clip already importable")
        return dest

    if not os.path.isdir(dest):
        print(f"  [get ] cloning EVA-CLIP source -> {dest}")
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        r = subprocess.run(["git", "clone", "--depth", "1", repo, dest],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"git clone failed: {r.stderr.strip()}\n"
                f"  Override with --eva-clip-dir, or skip with "
                f"--models to deselect eva-clip.")
    else:
        print(f"  [skip] source already present: {dest}")

    pkg = os.path.join(dest, "EVA-CLIP", "rei")
    if not os.path.isdir(os.path.join(pkg, "eva_clip")):
        raise FileNotFoundError(
            f"unexpected EVA layout (expected EVA-CLIP/rei/eva_clip under "
            f"{dest}).  Check {MODELS['eva-clip']['repo']}.")

    if write_pth:
        import sysconfig
        site = sysconfig.get_paths()["purelib"]
        pth = os.path.join(site, "eva_clip.pth")
        with open(pth, "w") as f:
            f.write(pkg + "\n")
        print(f"  [ok  ] sys.path entry written: {pth}")
        # prove it
        if not eva_clip_installed():
            print("  [warn] eva_clip still not importable — open_clip "
                  "fallback will be used")
    else:
        print(f"  [note] source at {pkg}")
        print(f"         add it to sys.path (run download_manager/"
              f"setup_env.sh, or pass --write-pth) before BLIP-2 attack runs.")
    return dest


def prefetch_clip(clip_names=None, cache_dir=None) -> None:
    """Warm the OpenAI CLIP cache used by the attack encoders."""
    try:
        import clip
    except ImportError as e:
        raise ImportError(f"the `clip` package is required for the attack "
                          f"encoders ({e})") from e
    import torch                      # noqa: F401 — clip.load needs it
    for name in (clip_names or ["ViT-L/14", "ViT-L/14@336px"]):
        print(f"  [get ] OpenAI CLIP {name}")
        clip.load(name, device="cpu")
    print("         -> ~/.cache/clip")


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download the VLM checkpoints used by vlm_transfer/. "
                    "Authenticate first with `hf auth login`.")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   choices=sorted(MODELS), metavar="NAME",
                   help=f"which to fetch (default: {' '.join(DEFAULT_MODELS)})")
    p.add_argument("--cache-dir", default=None,
                   help="HF cache dir (default: the hub default, honours HF_HOME)")
    p.add_argument("--of3b-dest", default=None,
                   help="where to put the OF-3B checkpoint.pt "
                        "(default: vlm_transfer/checkpoints/)")
    p.add_argument("--eva-clip-dir", default=None,
                   help="where to clone the EVA-CLIP source")
    p.add_argument("--write-pth", action="store_true",
                   help="EVA-CLIP: also write the site-packages .pth so "
                        "`import eva_clip` works (what setup_vastai.sh does)")
    p.add_argument("--check", action="store_true",
                   help="report auth + what is already downloaded; download nothing")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and sizes; download nothing")
    return p.parse_args(argv)


def status_report(models) -> dict:
    """Local presence per model — no network."""
    out = {}
    for key in models:
        spec = MODELS[key]
        if spec["kind"] == "hf_snapshot":
            present = hf_repo_cached(spec["repo"])
        elif spec["kind"] == "of_checkpoint":
            present = hf_repo_cached(spec["repo"]) and hf_repo_cached(
                spec["lm_repo"])
        elif spec["kind"] == "git_source":
            present = eva_clip_installed()
        else:                                    # clip_prefetch
            present = clip_cached()
        out[key] = present
    return out


def main(argv=None):
    args = parse_args(argv)
    models = list(args.models)

    print("=" * 64)
    print("VLM-TRANSFER MODEL DOWNLOAD")
    print("=" * 64)

    # SOCKS proxy vars make the hub stack abort (AGENTS.md env gotcha)
    sanitize_proxy_env()

    # ── auth ───────────────────────────────────────────────────────
    ok, msg = hf_auth_status()
    print(f"\n  auth   : {msg}")
    if not ok and not (args.dry_run or args.check):
        print("\n  Refusing to start without a token for the 3-8B VLMs.")
        print("  Run `hf auth login`, then re-run this command.")
        return 2

    # ── plan / status ──────────────────────────────────────────────
    present = status_report(models)
    total = 0.0
    print(f"\n  cache  : {os.environ.get('HF_HOME', '(hub default)')}")
    print(f"  models : {' '.join(models)}\n")
    print(f"  {'model':10s} {'present':8s} {'size':>7s}  what it is")
    print("  " + "-" * 60)
    for key in models:
        spec = MODELS[key]
        total += spec["size_gb"]
        flag = "yes" if present[key] else "no"
        print(f"  {key:10s} {flag:8s} {spec['size_gb']:6.1f}G  {spec['label']}")
        print(f"  {'':10s} {'':8s} {'':7s}  -> {spec['used_by']}")

    free = shutil.disk_usage(os.path.expanduser("~")).free / 1e9
    print(f"\n  needed ~{total:.1f} GB   free on ~: {free:.1f} GB")
    if free < total + 5:
        print("  [warn] less than 5 GB headroom — downloads may fail")

    if args.check:
        print("\n  --check: nothing downloaded.")
        return 0
    if args.dry_run:
        print("\n  --dry-run: nothing downloaded.")
        return 0

    # ── downloads ──────────────────────────────────────────────────
    print()
    for key in models:
        spec = MODELS[key]
        if present.get(key):
            print(f"[skip] {key} already present")
            continue
        print(f"[run ] {key} — {spec['label']}")
        try:
            if spec["kind"] == "hf_snapshot":
                download_hf_snapshot(spec["repo"], args.cache_dir)
            elif spec["kind"] == "of_checkpoint":
                download_of3b(spec, args.cache_dir, args.of3b_dest)
            elif spec["kind"] == "git_source":
                install_eva_clip_source(args.eva_clip_dir, args.write_pth)
            elif spec["kind"] == "clip_prefetch":
                prefetch_clip(args.cache_dir)
        except Exception as e:                # noqa: BLE001 — report, continue
            print(f"  [FAIL] {key}: {type(e).__name__}: {e}")
            if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
                print("         -> authenticate with `hf auth login` and retry")
            continue

    # ── final status ───────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("DONE — status")
    print("=" * 64)
    after = status_report(models)
    for key in models:
        print(f"  {key:10s} {'OK' if after[key] else 'MISSING'}")
    missing = [k for k in models if not after[k]]
    if missing:
        print(f"\n  still missing: {' '.join(missing)}")
        return 1
    print("\n  all requested models are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
