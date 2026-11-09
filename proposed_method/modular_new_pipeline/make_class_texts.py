"""
make_class_texts.py — Regenerate class_texts/{dataset}/classes.txt.
=================================================================
One class name per line, index-aligned with the dataset's label ids, so
the files can serve as the canonical class-name source (BadCLIP
DualEmbedding convention: a plain list of class names per dataset).

Sources:
  * Caltech101 — the on-disk 101_ObjectCategories folder names (sorted,
    image-holding categories only), cleaned through CALTECH101_RENAMES
    exactly like data.py does.
  * CIFAR10 / ImageNet1k — BadCLIP-DualEmbedding's class lists
    (../baselines/BadCLIP-DualEmbedding/data/...) used VERBATIM, so our
    class names match that baseline's evaluation.

Usage:
    python make_class_texts.py
"""

import ast
import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)          # proposed_method/
REPO_DIR = os.path.dirname(PROJECT_DIR)             # clip_bfa/
BADCLIP_DATA = os.path.join(REPO_DIR, "baselines", "BadCLIP-DualEmbedding", "data")

sys.path.insert(0, PACKAGE_DIR)
from config import load_registry, resolve_path   # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")


def load_caltech_renames():
    """CALTECH101_RENAMES from data.py, parsed without importing it (keeps
    this script dependency-free — data.py imports torchvision)."""
    path = os.path.join(PACKAGE_DIR, "data.py")
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "CALTECH101_RENAMES":
                    return ast.literal_eval(node.value)
    raise RuntimeError("CALTECH101_RENAMES not found in data.py")


def clean_name(n):
    return n.replace("_", " ").lower()


def caltech101_names(entry):
    """Sorted category folders of 101_ObjectCategories (image-holding only),
    cleaned with the same rules data.py applies."""
    renames = load_caltech_renames()
    root = resolve_path(entry["root"])
    cat_dir = os.path.join(root, "caltech-101", "101_ObjectCategories")
    names = []
    skipped = []
    for cat in sorted(os.listdir(cat_dir)):
        cpath = os.path.join(cat_dir, cat)
        if not os.path.isdir(cpath):
            continue
        has_images = any(f.lower().endswith(IMAGE_EXTS)
                         for f in os.listdir(cpath))
        if not has_images:
            skipped.append(cat)          # stray/empty folder → not a class
            continue
        names.append(renames.get(cat, clean_name(cat)))
    if skipped:
        print(f"  [caltech101] skipped empty folders: {skipped}")
    return names


def badclip_names(*relpath):
    """Class list from BadCLIP-DualEmbedding's classes.py (dict literal)."""
    path = os.path.join(BADCLIP_DATA, *relpath)
    with open(path) as f:
        return list(eval(f.read())["classes"])


def write_class_texts(dataset, names):
    out_dir = os.path.join(PACKAGE_DIR, "class_texts", dataset)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "classes.txt")
    with open(out, "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"  {dataset:<12s} {len(names):>4d} classes → {out}")


def main():
    reg = load_registry()
    datasets = reg["datasets"]

    sources = {
        "Caltech101": lambda e: caltech101_names(e),
        "CIFAR10":    lambda e: badclip_names("CIFAR10", "test", "classes.py"),
        "ImageNet1k": lambda e: badclip_names("ImageNet1K", "validation",
                                              "classes.py"),
    }

    print("Generating class_texts:")
    for name, entry in datasets.items():
        if name not in sources:
            print(f"  {name}: no source defined — skipped")
            continue
        names = sources[name](entry)
        expected = entry.get("n_classes")
        if expected is not None and len(names) != expected:
            print(f"  WARNING: {name} has {len(names)} names but registry says "
                  f"n_classes={expected}")
        write_class_texts(name.lower(), names)


if __name__ == "__main__":
    main()
