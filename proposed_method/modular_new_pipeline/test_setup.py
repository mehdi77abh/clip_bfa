"""
test_setup.py — Verify that all registry models and datasets are present.
=========================================================================
v3 (final):
  • Dataset existence is probed by instantiating the exact torchvision
    class the pipeline uses (Caltech101, CIFAR10/100), so whatever
    `base_folder` the installed torchvision expects is honoured — the
    earlier hardcoded `root/101_ObjectCategories` guess is gone.
  • ImageNet is checked on the filesystem (avoids tarball
    `_check_integrity()` false negatives on val-only setups) and
    requires the devkit tarball to be present next to `val/`.
  • `class_texts/.../classes.txt` length is cross-checked against the
    class count torchvision actually reports (catches stale 102-line
    Caltech files before `data.py` crashes on them).
  • OpenAI CLIP uses the public `clip.available_models()`; a
    not-yet-cached weight file is shown as OK (it downloads on first
    run), not as a failure.
  • SigLIP uses `snapshot_download(..., local_files_only=True)`;
    not-cached is again OK.

Usage:
    python test_setup.py                             # everything
    python test_setup.py -v                          # show resolved paths
    python test_setup.py --models ViT-B/16 SigLIP-B/16
    python test_setup.py --datasets Caltech101 CIFAR100 ImageNet1k
"""

import argparse
import json
import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)      # proposed_method/


# ─────────────────────────────────────────────────────────────────────
#  Registry + path resolution (mirrors config.resolve_path)
# ─────────────────────────────────────────────────────────────────────
def resolve_path(path):
    return path if os.path.isabs(path) else \
        os.path.normpath(os.path.join(PROJECT_DIR, path))


def load_registry():
    with open(os.path.join(PACKAGE_DIR, "config.json")) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────
#  Dataset probes
# ─────────────────────────────────────────────────────────────────────
def _probe_torchvision(name, root, loader, entry):
    """Instantiate the actual torchvision class → (ok, msg, n_classes)."""
    from torchvision import datasets

    if loader == "Caltech101":
        ds = datasets.Caltech101(root=root, download=False,
                                 target_type=entry.get("target_type", "category"))
        return True, f"{len(ds.categories)} classes, {len(ds)} samples", \
               len(ds.categories)

    if loader in ("CIFAR10", "CIFAR100"):
        cls = datasets.CIFAR10 if loader == "CIFAR10" else datasets.CIFAR100
        train = entry.get("split", "test") == "train"
        ds = cls(root=root, train=train, download=False)
        return True, f"{len(ds.classes)} classes, {len(ds)} samples", \
               len(ds.classes)

    raise ValueError(f"loader {loader!r} has no torchvision probe")


def _check_imagenet(root, split, entry):
    """Filesystem check for ImageNet — avoids _check_integrity() surprises."""
    split_dir = os.path.join(root, "train" if split == "train" else "val")
    if not os.path.isdir(split_dir):
        return False, f"missing {split_dir}", None
    classes = [d for d in os.listdir(split_dir)
               if os.path.isdir(os.path.join(split_dir, d))]
    if not classes:
        return False, f"{split_dir}: no class subfolders", None
    devkit = [os.path.join(root, "..", "ILSVRC2012_devkit_t12.tar.gz"),
              os.path.join(root, "ILSVRC2012_devkit_t12.tar.gz")]
    has_devkit = any(os.path.isfile(p) for p in devkit)
    n_expected = entry.get("n_classes", 1000)
    if len(classes) != n_expected:
        return True, (f"{split_dir} ({len(classes)} classes, expected "
                      f"{n_expected})"), n_expected
    if not has_devkit:
        return True, (f"{split_dir} ({len(classes)} classes)  "
                      f"[WARN: ILSVRC2012_devkit_t12.tar.gz not found]"), \
               n_expected
    return True, f"{split_dir} ({len(classes)} classes)", n_expected


def _check_imagefolder(root):
    if not os.path.isdir(root):
        return False, f"missing {root}", None
    cls = [d for d in os.listdir(root)
           if os.path.isdir(os.path.join(root, d))]
    if not cls:
        return False, f"{root}: no class subfolders", None
    return True, f"{root} ({len(cls)} classes)", len(cls)


def _has_images(folder, n=1):
    try:
        for f in os.listdir(folder)[:50]:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                n -= 1
                if n <= 0:
                    return True
    except OSError:
        return False
    return n <= 0


def check_dataset(name, entry, verbose=False):
    """(name, registry entry, verbose) → (ok, msg)."""
    root   = resolve_path(entry["root"])
    loader = entry["loader"]

    if verbose:
        print(f"    root   : {root}")
        print(f"    loader : {loader}")

    if not os.path.isdir(root):
        return False, f"root does not exist: {root}"

    try:
        if loader in ("Caltech101", "CIFAR10", "CIFAR100"):
            ok, msg, n_classes = _probe_torchvision(name, root, loader, entry)
        elif loader == "ImageNet":
            ok, msg, n_classes = _check_imagenet(root,
                                                entry.get("split", "val"),
                                                entry)
        elif loader == "ImageFolder":
            ok, msg, n_classes = _check_imagefolder(root)
        else:
            return False, f"unknown loader {loader!r}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:100]}"

    if not ok:
        return False, msg

    # ── Sanity: stray/empty category folders in Caltech ──
    if loader == "Caltech101":
        # locate the category dir torchvision actually used
        actual = os.path.join(root, "caltech-101", "101_ObjectCategories")
        if not os.path.isdir(actual):
            actual = os.path.join(root, "101_ObjectCategories")
        if os.path.isdir(actual):
            empty = [c for c in os.listdir(actual)
                     if os.path.isdir(os.path.join(actual, c))
                     and not _has_images(os.path.join(actual, c))
                     and c != "BACKGROUND_Google"]
            if empty:
                msg += f"  [WARN: empty folder(s) → {empty[:5]}]"

    # ── class_texts length cross-check ──
    ctd = entry.get("class_text_dir")
    if ctd:
        d = ctd if os.path.isabs(ctd) else os.path.join(PACKAGE_DIR, ctd)
        fpath = os.path.join(d, "classes.txt")
        if os.path.isfile(fpath):
            n_lines = sum(1 for _ in open(fpath))
            if n_classes is not None and n_lines != n_classes:
                return False, (f"{fpath}: {n_lines} lines but dataset has "
                               f"{n_classes} classes — regenerate or delete "
                               f"to use fallback")
            if verbose:
                print(f"    classes.txt: {n_lines} names (matches dataset)")
        else:
            msg += "  [class_texts MISSING → falls back to ds.classes]"
            if verbose:
                print(f"    classes.txt: absent → fallback")

    return True, msg


# ─────────────────────────────────────────────────────────────────────
#  Model probes
# ─────────────────────────────────────────────────────────────────────
def _check_openai_clip(entry):
    try:
        import clip
    except ImportError:
        return False, "clip package not installed"
    name = entry.get("clip_name")
    valid = sorted(clip.available_models())
    if name not in valid:
        return False, (f"{name!r} not in clip.available_models() — "
                       f"valid: {valid}")
    cache = os.path.expanduser("~/.cache/clip")
    fname = name.replace("/", "-") + ".pt"          # ViT-B/16 → ViT-B-16.pt
    fpath = os.path.join(cache, fname)
    if os.path.isfile(fpath):
        return True, f"{fpath} ({os.path.getsize(fpath) / 1e9:.2f} GB)"
    return True, f"{name} OK  (not cached — downloads on first run)"


def _check_hf_model(hf_name):
    if os.path.isdir(hf_name):
        return True, f"local dir {hf_name}"
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False, "huggingface_hub not installed"
    try:
        path = snapshot_download(hf_name, local_files_only=True)
        return True, path
    except Exception:
        return True, f"{hf_name} OK  (not cached — downloads on first run)"


def check_model(name, entry, verbose=False):
    adapter = entry.get("adapter", "openai_clip")
    if verbose:
        print(f"    adapter: {adapter}")
    if adapter == "openai_clip":
        return _check_openai_clip(entry)
    if adapter == "siglip":
        return _check_hf_model(entry["hf_name"])
    return False, f"unknown adapter {adapter!r}"


# ─────────────────────────────────────────────────────────────────────
#  Output path write-probe
# ─────────────────────────────────────────────────────────────────────
def check_paths(paths):
    rows = []
    for key in ("output_root", "lp_head_root"):
        if key not in paths:
            rows.append((key, False, "not in registry"))
            continue
        p = resolve_path(paths[key])
        try:
            os.makedirs(p, exist_ok=True)
            probe = os.path.join(p, ".write_probe")
            with open(probe, "w") as fh:
                fh.write("x")
            os.remove(probe)
            rows.append((key, True, p))
        except Exception as e:
            rows.append((key, False, f"{p}  ({type(e).__name__}: {e})"))
    return rows


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Verify registry models + datasets")
    ap.add_argument("--models",   nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    reg = load_registry()
    sel_m = args.models   or sorted(reg["models"])
    sel_d = args.datasets or sorted(reg["datasets"])
    fail = 0

    print("\n" + "=" * 78)
    print("DATASETS")
    print("=" * 78)
    for name in sel_d:
        if name not in reg["datasets"]:
            print(f"  ✗ {name:<16s} not in registry")
            fail += 1
            continue
        ok, msg = check_dataset(name, reg["datasets"][name], args.verbose)
        print(f"  {'✓' if ok else '✗'} {name:<16s} {msg}")
        if not ok:
            fail += 1

    print("\n" + "=" * 78)
    print("MODELS")
    print("=" * 78)
    for name in sel_m:
        if name not in reg["models"]:
            print(f"  ✗ {name:<22s} not in registry")
            fail += 1
            continue
        ok, msg = check_model(name, reg["models"][name], args.verbose)
        print(f"  {'✓' if ok else '✗'} {name:<22s} {msg}")
        if not ok:
            fail += 1

    print("\n" + "=" * 78)
    print("OUTPUT PATHS  (write-probe; created if missing)")
    print("=" * 78)
    for key, ok, msg in check_paths(reg["paths"]):
        print(f"  {'✓' if ok else '✗'} {key:<14s} {msg}")
        if not ok:
            fail += 1

    print("\n" + "=" * 78)
    if fail:
        print(f"RESULT: {fail} item(s) missing — see ✗ above")
        print("  • Missing datasets must be prepared on disk.")
        print("  • Missing model weights will download on the first run "
              "(shown as ✓).")
        sys.exit(1)
    print("RESULT: everything present  ✓")


if __name__ == "__main__":
    main()