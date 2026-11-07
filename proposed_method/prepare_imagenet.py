#!/usr/bin/env python3
"""Prepare ILSVRC2012 (ImageNet-1k) at clip_bfa/data/imagenet — run ONCE before
the experiment codes.

Downloads the official archives with `wget` from image-net.org (no login),
verifies MD5s, extracts them and produces a torchvision-compatible ImageNet
root that ALL experiment codes share:
    clip_bfa/data/imagenet/
    ├── train/<wnid>/*.JPEG      # SUBSET: --train-per-class images per class
    │                            #   (default 200 -> 200 000 images; used to
    │                            #    train the shared linear probe)
    ├── val/<wnid>/*.JPEG        # 1000 classes, 50 000 images
    ├── valid -> val             # Flip-S's conventional folder name
    ├── meta.bin                 # class names + val order (torchvision format)
    └── (ILSVRC2012_*.tar[.gz])  # kept unless --delete-tars

Usage:
    python prepare_imagenet.py                        # val + train subset
    python prepare_imagenet.py --splits val           # val only (~16 GB)
    python prepare_imagenet.py --train-per-class 200
    python prepare_imagenet.py --delete-tars          # free ~138 GB after

Downloads (MD5-verified; `wget -c` resumes partial files):
    ILSVRC2012_img_val.tar            ~6.3 GB   (always)
    ILSVRC2012_img_train.tar          ~138 GB   (only for --splits both)
    ILSVRC2012_devkit_t12.tar.gz      ~2.5 MB   (class names for meta.bin)
from https://image-net.org/data/ILSVRC/2012/ — override with --base-url (e.g.
a mirror) if that path changes.  If the server sits behind a proxy, export
http_proxy/https_proxy so wget can reach out.  A wrong/HTML download is caught
by the MD5 check.

After it finishes:
  - Flip-S picks the dataset up automatically (its default
    ../data/Imagenet/{valid,train} paths resolve through the symlink created
    here).  NOTE: Flip-S's `--stage probe` expects a FULL train tree; with the
    subset, point Flip-S at the shared probe file instead
    (`--head probe --probe-path <file trained by the proposed method>`).
  - The proposed method loads it as
    datasets.ImageNet(root="../data/imagenet", split="val"|"train").
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

BASE_URL = "https://image-net.org/data/ILSVRC/2012"

ARCHIVES = {
    "train": ("ILSVRC2012_img_train.tar", "1d675b47d978889d74fa0da5fadfb00e"),
    "val": ("ILSVRC2012_img_val.tar", "29b22e2961454d5413ddabcf34fc5622"),
    "devkit": ("ILSVRC2012_devkit_t12.tar.gz", "fa75699e90414af021442c21a62c3abf"),
}
N_CLASSES = 1000
VAL_IMAGES = 50_000
GB = 1024 ** 3

SCRIPT_DIR = Path(__file__).resolve().parent          # clip_bfa/data
WORKSPACE = SCRIPT_DIR.parent                          # clip_bfa
FLIPS_DIR = WORKSPACE / "baselines" / "Flip-S"

HINTS = """    Expected flow:
        wget downloads ILSVRC2012_img_{val,train}.tar + ILSVRC2012_devkit_t12.tar.gz
        from image-net.org (no login), verifies MD5s, then extracts.
    If the download fails or the MD5 check fails, check network/proxy access
    and the --base-url, then re-run (wget resumes partial files)."""


def say(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Download (wget) + integrity
# ---------------------------------------------------------------------------
def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def have_archive(root, key):
    fname, md5 = ARCHIVES[key]
    p = root / fname
    return p.exists() and md5_of(p) == md5


def download(root, key, base_url):
    fname, md5 = ARCHIVES[key]
    dest = root / fname
    if dest.exists():
        say(f"[+] Checking MD5 of existing {fname} (large file, takes a while)")
        if md5_of(dest) == md5:
            say(f"[+] Already downloaded: {fname}")
            return
    if shutil.which("wget") is None:
        sys.exit("[-] `wget` not found on PATH.")
    url = f"{base_url}/{fname}"
    say(f"[+] wget {url}")
    rc = subprocess.run(["wget", "-c", "--progress=dot:giga", "-O",
                         str(dest), url]).returncode
    if rc != 0:
        sys.exit(f"[-] wget failed for {url} (exit {rc}).\n{HINTS}")
    if md5_of(dest) != md5:
        sys.exit(f"[-] MD5 mismatch for {fname} — the download is corrupt or the "
                 f"URL served something else (HTML error page?). Delete "
                 f"{dest} and re-run, or use --base-url for a mirror.\n{HINTS}")
    say(f"[+] Verified MD5: {fname}")


# ---------------------------------------------------------------------------
# State checks
# ---------------------------------------------------------------------------
def have_val(root):
    val = root / "val"
    return val.is_dir() and sum(1 for _ in val.iterdir()) >= N_CLASSES


def have_train_subset(root, per_class):
    train = root / "train"
    if not train.is_dir():
        return False
    dirs = [p for p in train.iterdir() if p.is_dir()]
    if len(dirs) < N_CLASSES:
        return False
    return all(sum(1 for _ in d.iterdir()) >= per_class for d in dirs)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def make_meta(root):
    """meta.bin: (wnid_to_classes, val_wnids) — torchvision's format,
    with REAL class names from the devkit (needed for CLIP text prompts)."""
    if (root / "meta.bin").exists():
        say("[+] meta.bin already present - skipping")
        return
    if not have_archive(root, "devkit"):
        sys.exit("[-] devkit archive missing/corrupt - cannot build meta.bin "
                 "(class names would be WNIDs, which hurts CLIP text prompts).")
    say("[+] Building meta.bin from devkit (real class names)")
    from torchvision.datasets.imagenet import parse_devkit_archive
    parse_devkit_archive(str(root))


def extract_val(root):
    from torchvision.datasets.imagenet import parse_val_archive
    if have_val(root):
        say("[+] val/ already present - skipping")
        return
    say("[+] Extracting/grouping val/ (50 000 images)")
    parse_val_archive(str(root))


def extract_train_subset(root, per_class):
    """Extract only the first `per_class` images of every per-class inner tar
    of ILSVRC2012_img_train.tar (~200k images instead of 1.28M).

    The outer tar is read with random access: for each inner <wnid>.tar only
    the leading members are streamed out (deterministic archive order), so the
    full 138 GB never needs to be unpacked.  Re-runs skip classes that already
    have enough images.
    """
    tar_path = root / ARCHIVES["train"][0]
    if not have_archive(root, "train"):
        sys.exit(f"[-] {tar_path.name} missing/corrupt - cannot extract the "
                 f"train subset.")
    train_root = root / "train"
    train_root.mkdir(exist_ok=True)

    if have_train_subset(root, per_class):
        say("[+] train/ subset already present - skipping")
        return

    say(f"[+] Extracting train subset: first {per_class} images per class")
    n_done = 0
    with tarfile.open(tar_path, "r:") as outer:
        for member in outer:
            if not member.isfile() or not member.name.endswith(".tar"):
                continue
            wnid = Path(member.name).stem
            dest = train_root / wnid
            dest.mkdir(exist_ok=True)
            existing = sum(1 for _ in dest.iterdir())
            if existing >= per_class:
                n_done += 1
                continue
            need = per_class - existing
            fobj = outer.extractfile(member)
            if fobj is None:
                continue
            with tarfile.open(fileobj=fobj, mode="r|") as inner:
                for img in inner:
                    if not img.isfile():
                        continue
                    name = os.path.basename(img.name)
                    if not name:
                        continue
                    src = inner.extractfile(img)
                    if src is None:
                        continue
                    with src, open(dest / name, "wb") as out:
                        shutil.copyfileobj(src, out)
                    need -= 1
                    if need == 0:
                        break
            n_done += 1
            if n_done % 50 == 0:
                say(f"    {n_done}/{N_CLASSES} classes done")
    say(f"[+] train subset done: {n_done} classes x {per_class} images")


# ---------------------------------------------------------------------------
# Symlinks + verify
# ---------------------------------------------------------------------------
def make_symlinks(root):
    valid = root / "valid"
    if not valid.exists():
        valid.symlink_to("val")
    link = FLIPS_DIR / "data" / "Imagenet"
    link.parent.mkdir(exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.resolve() == root.resolve():
            return
        say(f"[!] {link} exists but points elsewhere - leaving it untouched")
        return
    link.symlink_to(os.path.relpath(root, link.parent), target_is_directory=True)
    say(f"[+] Symlinked Flip-S dataset path: {link} -> {root}")


def verify(root, args):
    val = root / "val"
    classes = [p for p in val.iterdir() if p.is_dir()]
    n_val = sum(1 for p in val.glob("*/*.JPEG")) + sum(1 for p in val.glob("*/*.jpg"))
    say(f"\n[+] val : {len(classes)} classes, {n_val} images"
        f" (expected {N_CLASSES} / {VAL_IMAGES})")
    if "train" in args.splits:
        n_tr = sum(1 for _ in (root / "train").glob("*/*"))
        say(f"[+] train subset: {n_tr} images "
            f"(expected {N_CLASSES * args.train_per_class} = "
            f"{N_CLASSES} x {args.train_per_class})")
    say("[+] Done. Flip-S: default paths resolve here automatically "
        "(use the shared probe file for --probe-path); "
        "proposed method: datasets.ImageNet(root=\"../data/imagenet\", "
        "split=\"val\"|\"train\")")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def check_disk(root, args):
    free = shutil.disk_usage(root.parent).free
    need = 16 * GB
    if "train" in args.splits:
        need += 138 * GB + (args.train_per_class * N_CLASSES * 200_000)  # tar + subset
    if free < need:
        say(f"[!] Free space {free / GB:.0f} GB < estimated {need / GB:.0f} GB.")
        if "train" in args.splits:
            sys.exit("[-] Not enough space. Free space, point --root at a "
                     "bigger disk, or run with --splits val.")
        sys.exit("[-] Not enough space for val.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(SCRIPT_DIR / "imagenet"),
                    help="dataset root (default: clip_bfa/data/imagenet)")
    ap.add_argument("--splits", choices=["val", "both"], default="both",
                    help="val only (~16 GB) or val + train subset (default; "
                         "~185 GB, needed for the linear probe)")
    ap.add_argument("--train-per-class", type=int, default=200,
                    help="train subset size per class (default 200 -> 200k "
                         "images for the shared linear probe)")
    ap.add_argument("--delete-tars", action="store_true",
                    help="delete archives after extraction (saves ~145 GB)")
    ap.add_argument("--base-url", default=BASE_URL,
                    help=f"download base URL (default: {BASE_URL})")
    args = ap.parse_args()
    args.splits = ["val", "train"] if args.splits == "both" else ["val"]

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    say(f"[+] Dataset root: {root}")

    check_disk(root, args)

    needed = ["devkit", "val"] + (["train"] if "train" in args.splits else [])
    for key in needed:
        download(root, key, args.base_url)

    make_meta(root)
    extract_val(root)
    if "train" in args.splits:
        extract_train_subset(root, args.train_per_class)
    if args.delete_tars:
        for key in ("val", "train"):
            if key in args.splits:
                (root / ARCHIVES[key][0]).unlink(missing_ok=True)
        say("[+] Tars deleted (--delete-tars)")
    make_symlinks(root)
    verify(root, args)


if __name__ == "__main__":
    main()
