#!/usr/bin/env python3
"""
verify_env.py — sanity-check a fresh `vlm_attack` environment.
================================================================
Run this on the new machine (Vast.ai) after setup_vastai.sh:

    python verify_env.py            # imports, versions, GPU, registry
    python verify_env.py --smoke    # + download ViT-B/32 and run a CLIP round-trip

Exit code 0 = the environment can run the pipeline; 1 = something required is
broken. Missing datasets are reported as warnings (data can be downloaded
later), not failures.
"""

import argparse
import importlib
import importlib.metadata as md
import os
import sys
import traceback

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)      # proposed_method/

PY_TARGET = (3, 10)

# distribution name -> what the pipeline needs it for
DISTS = [
    ("numpy",           "arrays"),
    ("scipy",           "ablation scoring"),
    ("pandas",          "aggregation tables"),
    ("pillow",          "image loading"),
    ("matplotlib",      "trigger / ablation plots"),
    ("scikit-learn",    "metrics / utilities"),
    ("pyarrow",         "CC3M parquet downloader"),
    ("gdown",           "dataset downloader"),
    ("requests",        "dataset downloader"),
    ("tqdm",            "progress bars"),
    ("ftfy",            "OpenAI CLIP"),
    ("regex",           "OpenAI CLIP"),
    ("transformers",    "SigLIP adapter"),
    ("tokenizers",      "SigLIP / HF"),
    ("huggingface-hub", "SigLIP weights, CC3M"),
    ("safetensors",     "HF weights"),
    ("sentencepiece",   "SigLIP tokenizer"),
    ("protobuf",        "SigLIP tokenizer"),
    ("datasets",        "HF tooling"),
    ("PyYAML",          "config files"),
    ("torch",           "everything"),
    ("torchvision",     "datasets / transforms"),
    ("clip",            "OpenAI CLIP models (git install)"),
]

# modules that must import (dotted paths)
MODULES = [
    "numpy", "scipy", "pandas", "PIL", "matplotlib", "sklearn", "pyarrow",
    "gdown", "requests", "tqdm", "ftfy", "regex", "transformers",
    "tokenizers", "huggingface_hub", "safetensors", "sentencepiece",
    "google.protobuf", "datasets", "yaml", "torch", "torchvision", "clip",
]

# project modules that must import (all phases + CLIs)
PROJECT_MODULES = [
    "config", "utils", "adapters", "models", "data", "target", "trigger",
    "quantizer", "ablation", "weights", "hybrid_ranking", "search",
    "evaluation", "results", "visualization", "analysis", "pipeline",
    "experiment", "main", "aggregate", "linear_probe", "defence_finetune",
    "download_datasets", "make_class_texts",
]

CLIP_MODELS_REQUIRED = ["ViT-B/16", "ViT-L/14", "ViT-L/14@336px", "ViT-B/32"]

# dataset loader -> subpaths to probe under the registry root (any hit = OK)
DATA_PROBES = {
    "Caltech101": ["caltech-101/caltech-101/101_ObjectCategories"],
    "CIFAR10":    ["cifar-10-batches-py"],
    "CIFAR100":   ["cifar-100-python"],
    "ImageFolder": [""],                       # root itself
    "Food101":    ["food-101/meta", "food-101/images"],
    "Flowers102": ["flowers-102/jpg", "flowers-102"],
    "ImageNet":   ["val", "train"],
}

FAIL = []
WARN = []


def color(txt, code):
    return f"\033[{code}m{txt}\033[0m" if sys.stdout.isatty() else txt


def head(title):
    print()
    print(color(f"── {title} " + "─" * max(0, 56 - len(title)), "1"))


def ok(msg):
    print(f"  {color('✓', '32')} {msg}")


def warn(msg):
    WARN.append(msg)
    print(f"  {color('!', '33')} {msg}")


def fail(msg):
    FAIL.append(msg)
    print(f"  {color('✗', '31')} {msg}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_python():
    head("Python")
    v = sys.version_info
    msg = f"{v.major}.{v.minor}.{v.micro} ({sys.executable})"
    if (v.major, v.minor) == PY_TARGET:
        ok(msg)
    else:
        fail(f"{msg} — the env was built for python {PY_TARGET[0]}.{PY_TARGET[1]}")


def check_dists():
    head("Packages")
    for dist, why in DISTS:
        try:
            ver = md.version(dist)
            ok(f"{dist:<16} {ver:<14} ({why})")
        except md.PackageNotFoundError:
            fail(f"{dist:<16} NOT INSTALLED ({why})")
        except Exception as exc:                       # pragma: no cover
            fail(f"{dist:<16} version lookup failed: {exc}")


def check_imports():
    head("Imports")
    for mod in MODULES:
        try:
            importlib.import_module(mod)
            ok(mod)
        except Exception as exc:
            fail(f"{mod}: {type(exc).__name__}: {exc}")

    sys.path.insert(0, PACKAGE_DIR)
    for mod in PROJECT_MODULES:
        try:
            importlib.import_module(mod)
            ok(f"project: {mod}")
        except Exception as exc:
            fail(f"project: {mod}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)


def check_torch():
    head("PyTorch / CUDA")
    try:
        import torch
    except Exception as exc:
        fail(f"cannot import torch: {exc}")
        return

    ok(f"torch {torch.__version__}  (built for CUDA {torch.version.cuda})")
    try:
        ok(f"cudnn {torch.backends.cudnn.version()}")
    except Exception:
        warn("cudnn version unavailable")

    if not torch.cuda.is_available():
        warn("torch.cuda.is_available() is False — CPU-only run, or a driver/wheel mismatch")
        return

    try:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        ok(f"device: {name}  capability {cap[0]}.{cap[1]}  {total:.1f} GiB")
        arch = getattr(torch.cuda, "get_arch_list", lambda: [])()
        if arch:
            want = f"sm_{cap[0]}{cap[1]}"
            if want in arch:
                ok(f"wheel includes kernels for {want}")
            else:
                warn(f"wheel arch list {arch} does not list {want} "
                     "(a kernel launch test follows)")
    except Exception as exc:
        fail(f"device query failed: {exc}")
        return

    try:
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).mean().item()
        torch.cuda.synchronize()
        ok(f"GPU matmul works (checksum {y:.4f})")
    except Exception as exc:
        fail(f"GPU kernel launch failed: {type(exc).__name__}: {exc} "
             "— this wheel has no kernels for this GPU (choose another CUDA channel)")


def check_clip():
    head("OpenAI CLIP")
    try:
        import clip
    except Exception as exc:
        fail(f"import clip failed: {exc}")
        return
    try:
        available = clip.available_models()
        ok(f"clip.available_models(): {len(available)} models")
        missing = [m for m in CLIP_MODELS_REQUIRED if m not in available]
        if missing:
            fail(f"missing registry models: {missing}")
        else:
            ok("all registry models available: " + ", ".join(CLIP_MODELS_REQUIRED))
    except Exception as exc:
        fail(f"clip.available_models() failed: {exc}")


def check_registry():
    head("Project registry / data layout")
    try:
        from config import load_registry, resolve_path
    except Exception as exc:
        fail(f"cannot import config: {exc}")
        return

    try:
        reg = load_registry()
    except Exception as exc:
        fail(f"cannot load config.json: {exc}")
        return

    models = reg.get("models", {})
    datasets = reg.get("datasets", {})
    ok(f"{len(models)} models: {', '.join(models)}")

    for name, entry in datasets.items():
        ctd = entry.get("class_text_dir")
        if ctd:
            path = ctd if os.path.isabs(ctd) else os.path.join(PACKAGE_DIR, ctd)
            fpath = os.path.join(path, "classes.txt")
            if os.path.isfile(fpath):
                with open(fpath) as fh:
                    n = sum(1 for ln in fh if ln.strip())
                ok(f"{name}: classes.txt ({n} names)")
            else:
                fail(f"{name}: missing class_texts at {fpath}")

        root = resolve_path(entry.get("root", ""))
        probes = DATA_PROBES.get(entry.get("loader"), [""])
        hits = [p for p in probes
                if os.path.exists(os.path.join(root, p) if p else root)]
        if hits:
            ok(f"{name}: data found in {root}")
        else:
            warn(f"{name}: data NOT found in {root} "
                 f"(expected {probes}) — download before running")

    paths = reg.get("paths", {})
    for key in ("output_root", "lp_head_root"):
        if key in paths:
            ok(f"{key}: {resolve_path(paths[key])}")


def check_smoke():
    head("GPU smoke test (CLIP ViT-B/32 round-trip)")
    try:
        import clip
        import torch
        from PIL import Image
    except Exception as exc:
        fail(f"smoke imports failed: {exc}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()
        img = Image.new("RGB", (224, 224), (128, 128, 128))
        x = preprocess(img).unsqueeze(0).to(device)
        text = clip.tokenize(["a photo of a cat"]).to(device)
        with torch.no_grad():
            emb = model.encode_image(x)
            temb = model.encode_text(text)
        ok(f"encode_image {tuple(emb.shape)} on {device}")
        ok(f"encode_text  {tuple(temb.shape)} on {device}")
        cos = torch.nn.functional.cosine_similarity(emb.float(), temb.float()).item()
        ok(f"image/text cosine {cos:+.4f}")
    except Exception as exc:
        fail(f"CLIP round-trip failed: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=2)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true",
                    help="download ViT-B/32 and run a CLIP image+text round-trip")
    args = ap.parse_args()

    print(color("vlm_attack environment check", "1"))
    print(f"  project: {PACKAGE_DIR}")

    check_python()
    check_dists()
    check_imports()
    check_torch()
    check_clip()
    check_registry()
    if args.smoke:
        check_smoke()

    print()
    n_ok = "all required checks passed"
    if FAIL:
        print(color(f"RESULT: {len(FAIL)} failure(s), {len(WARN)} warning(s)", "31"))
        for item in FAIL:
            print(f"  ✗ {item}")
        print("\nFix the failures above, then re-run this script.")
        return 1

    print(color(f"RESULT: {n_ok}" + (f", {len(WARN)} warning(s)" if WARN else ""), "32"))
    if WARN:
        print("Warnings do not block the pipeline (usually missing data).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
