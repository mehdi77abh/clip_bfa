"""
download_vlm_data.py — Fetch OK-VQA and Flickr-30K into the registry layout.
============================================================================
Layout produced under ``--root`` (default ``../data/vlm``)::

    ok-vqa/
      questions/OpenEnded_mscoco_{train,val2014}_questions.json
      annotations/mscoco_{train,val2014}_annotations.json
      images/val2014/COCO_val2014_000000XXXXXX.jpg
    flickr30k/
      images/flickr30k-images/*.jpg
      dataset_flickr30k.json          # Karpathy split (preferred)

OK-VQA also needs the COCO val2014 images (its questions are asked about
COCO val2014 pictures) — ~13 GB, so it is gated behind ``--with-images``.

URLs are the upstream release locations as of 2026-11 and mirrors do move;
every one is overridable (``--url KEY=VALUE``) and ``--dry-run`` prints the
plan and what is already on disk without touching the network.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile

# ─────────────────────────────────────────────────────────────────────
#  Upstream locations (overridable via --url KEY=VALUE)
# ─────────────────────────────────────────────────────────────────────
URLS = {
    # OK-VQA v1.1 — question / annotation JSONs
    "okvqa_val_questions":
        "https://okvqa.allenai.org/static/data/"
        "OpenEnded_mscoco_val2014_questions.json.zip",
    "okvqa_val_annotations":
        "https://okvqa.allenai.org/static/data/"
        "mscoco_val2014_annotations.json.zip",
    "okvqa_train_questions":
        "https://okvqa.allenai.org/static/data/"
        "OpenEnded_mscoco_train2014_questions.json.zip",
    "okvqa_train_annotations":
        "https://okvqa.allenai.org/static/data/"
        "mscoco_train2014_annotations.json.zip",
    # COCO val2014 images (OK-VQA's image source) — ~13 GB
    "coco_val2014": "http://images.cocodataset.org/zips/val2014.zip",
    # Flickr-30K images — mirror configurable
    "flickr30k_images":
        "https://huggingface.co/datasets/ymzx/Flickr30k/resolve/main/"
        "flickr30k-images.zip",
    # Karpathy caption splits (dataset_flickr30k.json) — caption_datasets.zip
    "flickr30k_karpathy":
        "https://cs.stanford.edu/people/karpathy/deepimagesent/"
        "caption_datasets.zip",
}

OKVQA_FILES = [
    ("okvqa_val_questions", "questions",
     "OpenEnded_mscoco_val2014_questions.json"),
    ("okvqa_val_annotations", "annotations",
     "mscoco_val2014_annotations.json"),
    ("okvqa_train_questions", "questions",
     "OpenEnded_mscoco_train2014_questions.json"),
    ("okvqa_train_annotations", "annotations",
     "mscoco_train2014_annotations.json"),
]

DATASETS = ("ok-vqa", "flickr30k", "all")


# ══════════════════════════════════════════════════════════════════════
#  Primitives
# ══════════════════════════════════════════════════════════════════════
def _default_root() -> str:
    """proposed_method/../data/vlm — the registry's '../data' convention.

    Inlined rather than imported from datasets_vlm so this downloader stays
    stdlib-only (it must work before the eval dependencies are installed).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    proposed_method = os.path.dirname(here)
    return os.path.normpath(os.path.join(proposed_method, "..", "data", "vlm"))


def download_file(url: str, dest: str, chunk: int = 1 << 20) -> str:
    """Stream ``url`` to ``dest`` (a file or a directory).  Returns the file
    path actually written."""
    if os.path.isdir(dest):
        dest = os.path.join(dest, url.rsplit("/", 1)[-1])
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        print(f"  [skip] already present: {dest}")
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    print(f"  [get ] {url}\n       -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "clip-bfa/1.0"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as out:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            out.write(buf)
            done += len(buf)
            if total:
                pct = 100.0 * done / total
                sys.stdout.write(f"\r       {done / 1e6:8.1f} MB "
                                 f"({pct:5.1f}%)")
                sys.stdout.flush()
        if total:
            sys.stdout.write("\n")
    shutil.move(tmp, dest)
    return dest


def extract_zip(path: str, dest: str, strip_components: int = 0) -> None:
    """Unzip ``path`` into ``dest`` (optionally stripping leading dirs)."""
    os.makedirs(dest, exist_ok=True)
    print(f"  [unzp] {path} -> {dest}")
    with zipfile.ZipFile(path) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            parts = member.split("/")
            if strip_components:
                parts = parts[strip_components:]
                if not parts:
                    continue
            out_path = os.path.join(dest, *parts)
            os.makedirs(os.path.dirname(out_path) or dest, exist_ok=True)
            with zf.open(member) as src, open(out_path, "wb") as out:
                shutil.copyfileobj(src, out)


def _resolve(url_key: str, overrides: dict) -> str:
    return overrides.get(url_key, URLS[url_key])


# ══════════════════════════════════════════════════════════════════════
#  Per-dataset downloads
# ══════════════════════════════════════════════════════════════════════
def download_okvqa(root: str, overrides: dict, with_images: bool = True,
                   want_train: bool = True) -> list:
    """OK-VQA question/annotation JSONs (+ optional COCO val2014 images)."""
    root = os.path.join(root, "ok-vqa")
    os.makedirs(root, exist_ok=True)
    made = []

    for key, sub, fname in OKVQA_FILES:
        if "train" in key and not want_train:
            continue
        dest = os.path.join(root, sub, fname)
        if os.path.isfile(dest):
            print(f"  [skip] already present: {dest}")
            made.append(dest)
            continue
        staging = os.path.join(root, "_staging")
        zpath = download_file(_resolve(key, overrides), staging)
        extract_zip(zpath, staging)
        # the zip contains just the json at its root
        found = None
        for dirpath, _, names in os.walk(staging):
            if fname in names:
                found = os.path.join(dirpath, fname)
                break
        if found is None:
            raise FileNotFoundError(
                f"{fname} not inside {zpath} — check the URL/mirror")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(found, dest)
        shutil.rmtree(staging, ignore_errors=True)
        made.append(dest)

    if with_images:
        images_dir = os.path.join(root, "images", "val2014")
        if os.path.isdir(images_dir) and os.listdir(images_dir):
            print(f"  [skip] already present: {images_dir}")
        else:
            staging = os.path.join(root, "_staging")
            zpath = download_file(_resolve("coco_val2014", overrides), staging)
            extract_zip(zpath, os.path.join(root, "images"), 0)
            shutil.rmtree(staging, ignore_errors=True)
        made.append(images_dir)
    else:
        print("  [note] COCO val2014 images skipped (--no-images); the "
              "evaluation will need them.")
    return made


def download_flickr30k(root: str, overrides: dict) -> list:
    """Flickr-30K images + Karpathy split annotations."""
    root = os.path.join(root, "flickr30k")
    os.makedirs(root, exist_ok=True)
    made = []

    images_dir = os.path.join(root, "images", "flickr30k-images")
    if os.path.isdir(images_dir) and os.listdir(images_dir):
        print(f"  [skip] already present: {images_dir}")
    else:
        staging = os.path.join(root, "_staging")
        zpath = download_file(_resolve("flickr30k_images", overrides), staging)
        extract_zip(zpath, os.path.join(root, "images"), 0)
        # mirrors differ: some zip straight into flickr30k-images/
        if not os.path.isdir(images_dir):
            for name in ("flickr30k-images", "flickr30k_images", "images"):
                cand = os.path.join(root, "images", name)
                if os.path.isdir(cand):
                    os.rename(cand, images_dir)
                    break
        shutil.rmtree(staging, ignore_errors=True)
    made.append(images_dir)

    karpathy = os.path.join(root, "dataset_flickr30k.json")
    if os.path.isfile(karpathy):
        print(f"  [skip] already present: {karpathy}")
    else:
        staging = os.path.join(root, "_staging")
        zpath = download_file(_resolve("flickr30k_karpathy", overrides),
                              staging)
        extract_zip(zpath, staging, 0)
        found = None
        for dirpath, _, names in os.walk(staging):
            if "dataset_flickr30k.json" in names:
                found = os.path.join(dirpath, "dataset_flickr30k.json")
                break
        if found is None:
            raise FileNotFoundError(
                f"dataset_flickr30k.json not inside {zpath}")
        shutil.move(found, karpathy)
        shutil.rmtree(staging, ignore_errors=True)
    made.append(karpathy)
    return made


# ══════════════════════════════════════════════════════════════════════
#  Status probe
# ══════════════════════════════════════════════════════════════════════
def check_status(root: str) -> dict:
    """What is already on disk — used by --dry-run and by run_vlm_transfer.sh."""
    def _present(path):
        return os.path.exists(path)

    return {
        "ok-vqa": {
            "questions": _present(os.path.join(
                root, "ok-vqa", "questions",
                "OpenEnded_mscoco_val2014_questions.json")),
            "annotations": _present(os.path.join(
                root, "ok-vqa", "annotations",
                "mscoco_val2014_annotations.json")),
            "images": _present(os.path.join(root, "ok-vqa", "images",
                                            "val2014")),
        },
        "flickr30k": {
            "images": _present(os.path.join(root, "flickr30k", "images",
                                            "flickr30k-images")),
            "annotations": _present(os.path.join(root, "flickr30k",
                                                 "dataset_flickr30k.json")),
        },
    }


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download OK-VQA / Flickr-30K for the VLM-transfer "
                    "evaluation.")
    p.add_argument("--dataset", default="all", choices=DATASETS)
    p.add_argument("--root", default=None,
                   help="download root (default: proposed_method/../data/vlm)")
    p.add_argument("--no-images", action="store_true",
                   help="OK-VQA: skip the 13 GB COCO val2014 images")
    p.add_argument("--no-train", action="store_true",
                   help="OK-VQA: test split only")
    p.add_argument("--url", action="append", default=[],
                   metavar="KEY=URL", help="override an upstream URL")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan + on-disk status, download nothing")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    overrides = {}
    for kv in args.url:
        if "=" not in kv:
            raise SystemExit(f"--url wants KEY=URL, got {kv!r}")
        k, v = kv.split("=", 1)
        overrides[k.strip()] = v.strip()

    root = args.root or _default_root()
    root = os.path.abspath(root)
    want = DATASETS[:-1] if args.dataset == "all" else (args.dataset,)

    print("=" * 62)
    print("VLM-TRANSFER DATA DOWNLOAD")
    print("=" * 62)
    print(f"  root    : {root}")
    print(f"  datasets: {', '.join(want)}")

    status = check_status(root)
    if args.dry_run:
        print("\n[dry-run] planned downloads:")
        for ds in want:
            for key in ("okvqa_val_questions", "okvqa_val_annotations",
                        "coco_val2014", "flickr30k_images",
                        "flickr30k_karpathy"):
                if ds == "ok-vqa" and not key.startswith(("okvqa", "coco")):
                    continue
                if ds == "flickr30k" and not key.startswith("flickr30k"):
                    continue
                if key.startswith("okvqa_train") and args.no_train:
                    continue
                if key == "coco_val2014" and args.no_images:
                    continue
                print(f"    {key:24s} {_resolve(key, overrides)}")
        print("\n[dry-run] on-disk status:")
        for ds, parts in status.items():
            for part, ok in parts.items():
                print(f"    {ds:10s} {part:12s} "
                      f"{'OK' if ok else 'MISSING'}")
        return 0

    for ds in want:
        print(f"\n-- {ds} " + "-" * 44)
        if ds == "ok-vqa":
            download_okvqa(root, overrides, with_images=not args.no_images,
                           want_train=not args.no_train)
        else:
            download_flickr30k(root, overrides)

    print("\nDone.  Status:")
    for ds, parts in check_status(root).items():
        for part, ok in parts.items():
            print(f"    {ds:10s} {part:12s} {'OK' if ok else 'MISSING'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
