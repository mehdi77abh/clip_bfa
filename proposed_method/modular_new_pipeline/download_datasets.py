#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_datasets.py — standalone downloader for the grid datasets that
run_all.sh uses: Caltech-101 + ImageNet-1k.

Writes the exact on-disk layout that data.py / torchvision expect (vlm_attack
env, torchvision 0.21), under a destination root that defaults to ``../data``
and resolves through ``config.resolve_path`` — the same '../data' convention as
the config.json registry:

  {dest}/caltech101/101_ObjectCategories/<class>/image_XXXX.jpg
  {dest}/caltech101/Annotations/<class>/annotation_XXXX.mat
  {dest}/imagenet/val/<wnid>/*.JPEG        (registry split is 'val')
  {dest}/imagenet/train/<wnid>/*.JPEG      (only with --imagenet-splits train)
  {dest}/imagenet/meta.bin                 (torchvision ImageNet meta file)
  {dest}/imagenet/wnids.txt                (cached canonical wnid list)
  {dest}/imagenet/manifest_{split}.csv     (index,wnid,filename of each image)
  {dest}/imagenet/parquet/                 (transport shards; only --keep-parquet)

TWO MODES:

  --start-only    download-START test: fetches only the first bytes of every
                  source into {dest}, verifies headers + magic bytes, then
                  STOPS — partial files are deleted and a report is written to
                  {dest}/_download_probe.json.  Nothing large is downloaded.
                  TEST THIS FIRST, e.g.:

                      python download_datasets.py --dest ../data_test --start-only

  (default)       full download: resume-capable, checksum / parquet-verified,
                  then extract + organize + verify.  e.g.:

                      python download_datasets.py --dest ../data

SOURCES (no login / no credentials required):

  Caltech-101 : CaltechDATA 'caltech-101.zip' (md5 3138e1922a9193bfa496528edbbc45d0);
                fallback (--caltech-source gdrive / auto) = the two Google-Drive
                archives torchvision's own Caltech101.download() uses (needs
                `gdown`); md5s of the inner archives are verified either way.
  ImageNet-1k : the ungated HuggingFace repack 'benjamin-paine/imagenet-1k'
                (parquet with the original image bytes) + the canonical sorted
                1000-wnid list from TF-slim for the class folders and meta.bin.
                The official ILSVRC2012 tars are login-gated (image-net.org and
                HF 'ILSVRC/imagenet-1k'), so they are deliberately not used.

Run with the project interpreter (needs `requests`; `pyarrow` + `torch` only for
the full mode; `torchvision` only for the final --verify-load check):

    PY=/home/mehdi/miniconda3/envs/vlm_attack/bin/python
    $PY download_datasets.py --dest ../data_test --start-only   # test first!
    $PY download_datasets.py --dest ../data                     # real download
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import time
import zipfile

import requests

from config import resolve_path        # the registry's '../data' path convention

# ════════════════════════════════════════════════════════════════
#  SOURCES
# ════════════════════════════════════════════════════════════════
# Caltech-101 — CaltechDATA record mzrjq-6wc02 (official, direct download)
CALTECH_ZIP_URL = ("https://data.caltech.edu/api/records/mzrjq-6wc02/"
                   "files/caltech-101.zip/content")
CALTECH_ZIP_MD5 = "3138e1922a9193bfa496528edbbc45d0"
# inner archives of the zip → extract into {dest}/caltech101/ (md5s are the
# same values torchvision checks for its Google-Drive copies)
CALTECH_INNER = {
    "caltech-101/101_ObjectCategories.tar.gz":
        ("101_ObjectCategories.tar.gz", "b224c7392d521a49829488ab0f1120d9"),
    "caltech-101/Annotations.tar":
        ("Annotations.tar", "6f83eeb1f24d99cab4eb377263132c91"),
}
# fallback: torchvision 0.21's own download() Google-Drive file ids
CALTECH_GDRIVE = [
    ("137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp", "101_ObjectCategories.tar.gz",
     "b224c7392d521a49829488ab0f1120d9"),
    ("175kQy3UsZ0wUEHZjqkUDdNVssr7bgh_m", "Annotations.tar",
     "6f83eeb1f24d99cab4eb377263132c91"),
]

# ImageNet-1k — ungated HF parquet repack + TF-slim canonical wnid list
HF_REPO = "benjamin-paine/imagenet-1k"
HF_API_URL = f"https://huggingface.co/api/datasets/{HF_REPO}"
HF_RAW_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/"
WNID_URL = ("https://raw.githubusercontent.com/tensorflow/models/master/"
            "research/slim/datasets/imagenet_lsvrc_2015_synsets.txt")

# ════════════════════════════════════════════════════════════════
#  CONSTANTS (fixed corpora — counts are known in advance)
# ════════════════════════════════════════════════════════════════
CHUNK = 1 << 20                    # 1 MiB HTTP chunks
PROBE_BYTES = 1 << 18              # 256 KiB per probed source
PROGRESS_STEP = 256 << 20          # progress line every 256 MiB
USER_AGENT = "clip-flip-dataset-downloader/1.0"

N_WNIDS = 1000                     # ILSVRC-2012 classes
N_VAL_IMAGES = 50000               # ILSVRC-2012 val images
N_CALTECH_CATEGORIES = 102         # 101 + BACKGROUND_Google
N_CALTECH_IMAGES = 9144            # 8677 + 467 background
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


# ════════════════════════════════════════════════════════════════
#  SMALL HELPERS
# ════════════════════════════════════════════════════════════════
def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def http_get(url, start=0):
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    return requests.get(url, headers=headers, stream=True,
                        timeout=(15, 300), allow_redirects=True)


def safe_extract_tars(paths, dest):
    """Extract tar archives into dest, refusing absolute / parent-escaping
    member paths (tarfile has no path filter on this Python)."""
    for path in paths:
        with tarfile.open(path) as tf:
            dest_abs = os.path.abspath(dest)
            for m in tf.getmembers():
                out = os.path.abspath(os.path.join(dest, m.name))
                if not out.startswith(dest_abs + os.sep):
                    raise RuntimeError(f"unsafe tar member {m.name!r} in {path}")
            tf.extractall(dest)


# ════════════════════════════════════════════════════════════════
#  DOWNLOAD-START PROBE (--start-only)
# ════════════════════════════════════════════════════════════════
def probe_download(url, out_dir, name, magic, nbytes=PROBE_BYTES):
    """Fetch the first `nbytes` of `url` into out_dir/name.part, verify the
    magic bytes, then delete the partial.  Returns a report dict."""
    rep = {"file": name, "url": url, "magic_expected": magic.decode("latin1"),
           "ok": False}
    part = os.path.join(out_dir, name.replace("/", "_") + ".part")
    try:
        r = http_get(url)
        rep["http_status"] = r.status_code
        rep["content_length"] = r.headers.get("Content-Length")
        rep["content_type"] = r.headers.get("Content-Type")
        r.raise_for_status()
        got = b""
        for chunk in r.iter_content(CHUNK):
            got += chunk
            if len(got) >= nbytes:
                break
        r.close()
        rep["bytes_read"] = len(got)
        rep["magic_found"] = got[:4].decode("latin1")
        rep["magic_ok"] = got.startswith(magic)
        rep["ok"] = rep["magic_ok"] and len(got) > 0
    except Exception as e:                                    # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
    finally:
        if os.path.isfile(part):
            os.remove(part)
    return rep


# ════════════════════════════════════════════════════════════════
#  RESUMABLE FILE DOWNLOAD (full mode)
# ════════════════════════════════════════════════════════════════
def download_file(url, path, md5=None, desc=""):
    """Stream `url` to `path` with resume support and optional md5 check.
    Retries three times; a failed md5 deletes the partial."""
    part = path + ".part"
    desc = desc or os.path.basename(path)
    for attempt in (1, 2, 3):
        try:
            done = os.path.getsize(part) if os.path.isfile(part) else 0
            r = http_get(url, start=done)
            if r.status_code == 200:          # server ignored Range → restart
                done = 0
            elif r.status_code == 416 and done > 0:
                pass                          # already complete; md5 decides
            else:
                r.raise_for_status()
            total = r.headers.get("Content-Length")
            total = int(total) + done if total else None
            mode = "ab" if done else "wb"
            last = done
            with open(part, mode) as f:
                for chunk in r.iter_content(CHUNK):
                    f.write(chunk)
                    done += len(chunk)
                    if done - last >= PROGRESS_STEP:
                        t = f"/{human(total)}" if total else ""
                        print(f"    {desc}: {human(done)}{t}")
                        last = done
            if total is not None and done != total:
                raise RuntimeError(f"truncated download {human(done)}/{human(total)}")
            if md5 is not None:
                got = md5_file(part)
                if got != md5:
                    os.remove(part)
                    raise RuntimeError(f"md5 mismatch: {got} != {md5}")
            os.replace(part, path)
            print(f"    {desc}: done ({human(os.path.getsize(path))}"
                  + (f", md5 ok" if md5 else "") + ")")
            return
        except Exception as e:                                # noqa: BLE001
            if attempt == 3:
                raise
            print(f"    {desc}: attempt {attempt} failed ({e}); retrying...")
            time.sleep(5 * attempt)


# ════════════════════════════════════════════════════════════════
#  CALTECH-101
# ════════════════════════════════════════════════════════════════
def caltech_get(root, source="auto", force=False):
    """Download + extract Caltech-101 into {root}/caltech101 (the torchvision
    0.21 Caltech101 sub-directory name)."""
    cal_root = os.path.join(root, "caltech101")
    os.makedirs(cal_root, exist_ok=True)
    if not force and os.path.isdir(os.path.join(cal_root, "101_ObjectCategories")):
        print("  Caltech101: already present, verifying only")
        verify_caltech(cal_root)
        return

    inner = []                      # (archive_path, md5) pairs to extract
    if source in ("auto", "caltechdata"):
        try:
            zip_path = os.path.join(cal_root, "caltech-101.zip")
            download_file(CALTECH_ZIP_URL, zip_path, md5=CALTECH_ZIP_MD5,
                          desc="caltech-101.zip")
            with zipfile.ZipFile(zip_path) as zf:
                for member, (fname, md5) in CALTECH_INNER.items():
                    target = os.path.join(cal_root, fname)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst, CHUNK)
                    if md5_file(target) != md5:
                        raise RuntimeError(f"inner archive {fname}: md5 mismatch")
                    inner.append(target)
            os.remove(zip_path)
        except Exception as e:                                # noqa: BLE001
            if source == "caltechdata":
                raise
            print(f"  Caltech101: CaltechDATA source failed ({e}); "
                  "falling back to Google Drive (gdown)")
            inner = []
    if not inner:                   # gdown fallback (torchvision's own files)
        import gdown
        for file_id, fname, md5 in CALTECH_GDRIVE:
            target = os.path.join(cal_root, fname)
            if not (os.path.isfile(target) and md5_file(target) == md5):
                gdown.download(id=file_id, output=target, quiet=False,
                               user_agent=USER_AGENT)
            if md5_file(target) != md5:
                raise RuntimeError(f"Google-Drive file {fname}: md5 mismatch")
            inner.append(target)

    print(f"  Caltech101: extracting into {cal_root}")
    safe_extract_tars(inner, cal_root)
    for path in inner:
        os.remove(path)
    verify_caltech(cal_root)


def verify_caltech(cal_root):
    """Count check + removal of stray EMPTY category folders (the known label-
    shifting gotcha from AGENTS.md — torchvision would count an extra class)."""
    cat_dir = os.path.join(cal_root, "101_ObjectCategories")
    if not os.path.isdir(cat_dir):
        raise RuntimeError(f"{cat_dir} missing — Caltech101 download incomplete")
    for name in sorted(os.listdir(cat_dir)):
        p = os.path.join(cat_dir, name)
        if os.path.isdir(p) and not os.listdir(p):
            print(f"  Caltech101: removing stray empty folder {p}")
            os.rmdir(p)
    cats = sorted(os.listdir(cat_dir))
    n_imgs = sum(len(os.listdir(os.path.join(cat_dir, c))) for c in cats
                 if os.path.isdir(os.path.join(cat_dir, c)))
    n_cls_txt = class_text_count("caltech101")
    print(f"  Caltech101: {len(cats)} category folders "
          f"(+BACKGROUND_Google), {n_imgs} images"
          + (f", classes.txt has {n_cls_txt} names" if n_cls_txt else ""))
    if len(cats) != N_CALTECH_CATEGORIES:
        raise RuntimeError(
            f"expected {N_CALTECH_CATEGORIES} category folders, found "
            f"{len(cats)} — labels would shift (see AGENTS.md data gotcha)")
    if n_imgs < N_CALTECH_IMAGES:
        print(f"  Caltech101: WARNING only {n_imgs} images "
              f"(expected {N_CALTECH_IMAGES})")
    if n_cls_txt not in (None, len(cats) - 1):
        print(f"  Caltech101: WARNING classes.txt has {n_cls_txt} names but "
              f"the dataset has {len(cats) - 1} classes — data.py will refuse "
              "to load it (run make_class_texts.py)")


# ════════════════════════════════════════════════════════════════
#  IMAGENET-1K
# ════════════════════════════════════════════════════════════════
def imagenet_shard_list(split):
    """Parquet shard paths (repo-relative) of `split`, from the HF API.
    ('val' → the repo's 'validation-' shards; our on-disk folder stays 'val'
    to match torchvision's ImageNet split name.)"""
    prefix = {"val": "validation", "train": "train"}[split]
    r = requests.get(HF_API_URL, headers={"User-Agent": USER_AGENT},
                     timeout=(15, 60))
    r.raise_for_status()
    sibs = [f["rfilename"] for f in r.json().get("siblings", [])]
    pat = re.compile(rf"data/{prefix}-\d+-of-\d+\.parquet$")
    shards = sorted(s for s in sibs if pat.match(s))
    if not shards:
        raise RuntimeError(f"no parquet shards for split {split!r} "
                           f"in HF repo {HF_REPO}")
    return shards


def fetch_wnids(path):
    """Canonical sorted 1000-wnid list (TF-slim) → cached to `path`."""
    r = requests.get(WNID_URL, headers={"User-Agent": USER_AGENT},
                     timeout=(15, 60))
    r.raise_for_status()
    wnids = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
    if len(wnids) != N_WNIDS or wnids != sorted(wnids) \
            or not all(w.startswith("n") for w in wnids):
        raise RuntimeError(f"unexpected wnid list ({len(wnids)} entries)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(wnids) + "\n")
    print(f"  ImageNet: wnid list ok ({len(wnids)} wnids, cached to {path})")
    return wnids


def imagenet_class_names(shard_path):
    """Class names (index-aligned, 'words' style like torchvision meta.bin)
    from the parquet schema metadata; falls back to our classes.txt.
    `shard_path` may be None to skip the parquet entirely."""
    if shard_path is not None:
        import pyarrow.parquet as pq
        meta = pq.ParquetFile(shard_path).schema_arrow.metadata or {}
        try:
            hf = json.loads(meta.get(b"huggingface", b"{}"))
            for feat in hf.get("info", {}).get("features", []):
                if feat.get("name") == "label":
                    names = (feat.get("dtype") or {}).get("class_label", {}) \
                                .get("names")
                    if names and len(names) == N_WNIDS:
                        print("  ImageNet: class names from parquet schema")
                        return list(names)
        except Exception:                                     # noqa: BLE001
            pass
    path = os.path.join(PACKAGE_DIR, "class_texts", "imagenet1k", "classes.txt")
    if os.path.isfile(path):
        with open(path) as f:
            names = [ln.strip() for ln in f if ln.strip()]
        if len(names) == N_WNIDS:
            print("  ImageNet: class names from class_texts/imagenet1k "
                  "(parquet schema had none)")
            return names
    raise RuntimeError("cannot determine ImageNet class names "
                       "(no parquet schema names, no classes.txt)")


def imagenet_extract(shards, split, wnids, out_root):
    """Parquet shards → torchvision ImageFolder layout {out_root}/{split}/
    <wnid>/<split>_%08d.JPEG  +  a manifest CSV.  Returns the image count."""
    import pyarrow.parquet as pq
    split_dir = os.path.join(out_root, split)
    os.makedirs(split_dir, exist_ok=True)
    for w in wnids:
        os.makedirs(os.path.join(split_dir, w), exist_ok=True)
    manifest = open(os.path.join(out_root, f"manifest_{split}.csv"), "w")
    manifest.write("index,wnid,filename\n")
    idx = 0
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=256,
                                     columns=["image", "label"]):
            imgs = batch.column("image").to_pylist()
            labels = batch.column("label").to_pylist()
            for img, lbl in zip(imgs, labels):
                data = img.get("bytes") if isinstance(img, dict) else img
                if not data:
                    raise RuntimeError(f"{shard}: image row {idx} has no bytes")
                wnid = wnids[int(lbl)]
                fname = f"{split}_{idx:08d}.JPEG"
                with open(os.path.join(split_dir, wnid, fname), "wb") as f:
                    f.write(data)
                manifest.write(f"{idx},{wnid},{fname}\n")
                idx += 1
        print(f"    {os.path.basename(shard)}: {idx} images so far")
    manifest.close()
    return idx


def imagenet_write_meta(out_root, wnids, names):
    """meta.bin in torchvision's own format: torch.save((wnid_to_classes,
    wnids)) — [0] is what datasets.ImageNet reads ([1] is only used by
    torchvision when organizing the official val tar, which we do ourselves)."""
    import torch
    wnid_to_classes = {w: names[i] for i, w in enumerate(wnids)}
    torch.save((wnid_to_classes, list(wnids)),
               os.path.join(out_root, "meta.bin"))
    print("  ImageNet: meta.bin written")


def verify_imagenet(out_root, split, strict=True):
    split_dir = os.path.join(out_root, split)
    if not os.path.isdir(split_dir):
        raise RuntimeError(f"{split_dir} missing")
    dirs = sorted(os.listdir(split_dir))
    n_imgs = sum(len(os.listdir(os.path.join(split_dir, d))) for d in dirs
                 if os.path.isdir(os.path.join(split_dir, d)))
    print(f"  ImageNet[{split}]: {len(dirs)} class folders, {n_imgs} images")
    if len(dirs) != N_WNIDS:
        raise RuntimeError(f"expected {N_WNIDS} class folders, found {len(dirs)}")
    if split == "val" and n_imgs != N_VAL_IMAGES:
        msg = (f"expected {N_VAL_IMAGES} val images, found {n_imgs}")
        if strict:
            raise RuntimeError(msg)
        print(f"  ImageNet[{split}]: WARNING {msg}")
    return n_imgs


def imagenet_get(root, splits, keep_parquet=False, force=False):
    """Download + organize ImageNet-1k into {root}/imagenet."""
    in_root = os.path.join(root, "imagenet")
    os.makedirs(in_root, exist_ok=True)
    pq_dir = os.path.join(in_root, "parquet")
    wnids_path = os.path.join(in_root, "wnids.txt")

    wnids = fetch_wnids(wnids_path) if (force or not os.path.isfile(wnids_path)) \
        else [ln.strip() for ln in open(wnids_path) if ln.strip()]

    names = None
    for split in splits:
        split_dir = os.path.join(in_root, split)
        if not force and os.path.isdir(split_dir) \
                and len(os.listdir(split_dir)) == N_WNIDS:
            try:
                verify_imagenet(in_root, split, strict=(split == "val"))
                print(f"  ImageNet[{split}]: already present, skipping")
                continue
            except RuntimeError as e:
                print(f"  ImageNet[{split}]: re-extracting ({e})")
        shutil.rmtree(split_dir, ignore_errors=True)   # clean slate for extraction

        shards = imagenet_shard_list(split)
        print(f"  ImageNet[{split}]: {len(shards)} parquet shards "
              f"({HF_REPO})")
        paths = []
        os.makedirs(pq_dir, exist_ok=True)
        for rel in shards:
            path = os.path.join(pq_dir, os.path.basename(rel))
            if not os.path.isfile(path):
                download_file(HF_RAW_BASE + rel, path,
                              desc=os.path.basename(rel))
            paths.append(path)
        if names is None:
            names = imagenet_class_names(paths[0])
        n = imagenet_extract(paths, split, wnids, in_root)
        print(f"  ImageNet[{split}]: extracted {n} images into {split_dir}")
        if split == "val" and n != N_VAL_IMAGES:
            raise RuntimeError(f"expected {N_VAL_IMAGES} val images, extracted {n}")
        if not keep_parquet:
            for path in paths:
                os.remove(path)

    meta_path = os.path.join(in_root, "meta.bin")
    if names is not None or not os.path.isfile(meta_path):
        if names is None:
            names = imagenet_class_names(None)    # classes.txt fallback
        imagenet_write_meta(in_root, wnids, names)
    for split in splits:
        verify_imagenet(in_root, split, strict=False)


# ════════════════════════════════════════════════════════════════
#  VERIFICATION
# ════════════════════════════════════════════════════════════════
def class_text_count(dataset):
    path = os.path.join(PACKAGE_DIR, "class_texts", dataset, "classes.txt")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return sum(1 for ln in f if ln.strip())


def verify_with_torchvision(dest):
    """Final acceptance check: load through the exact classes data.py uses."""
    from torchvision import datasets
    ds = datasets.Caltech101(root=dest, download=False)
    print(f"  torchvision Caltech101      : {len(ds)} samples, "
          f"{len(ds.categories)} categories")
    in_root = os.path.join(dest, "imagenet")
    for split in ("val", "train"):
        if os.path.isdir(os.path.join(in_root, split)):
            d = datasets.ImageNet(root=in_root, split=split)
            print(f"  torchvision ImageNet[{split}]   : {len(d)} samples, "
                  f"{len(d.classes)} classes")


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Download Caltech-101 + ImageNet-1k into the layout "
                    "data.py expects (default dest ../data).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="test first:  python download_datasets.py --dest ../data_test "
               "--start-only")
    ap.add_argument("--dest", default="../data",
                    help="destination root, resolved like config.json registry "
                         "paths (default: ../data)")
    ap.add_argument("--datasets", default="caltech,imagenet",
                    help="comma list of: caltech, imagenet (default: both)")
    ap.add_argument("--imagenet-splits", default="val",
                    help="comma list of: val, train (default: val — the "
                         "registry uses split 'val')")
    ap.add_argument("--start-only", action="store_true",
                    help="ONLY test that downloads start correctly (first "
                         "bytes + magic), then stop and clean up")
    ap.add_argument("--start-bytes", type=int, default=PROBE_BYTES,
                    help="bytes to fetch per probed source in --start-only")
    ap.add_argument("--caltech-source", default="auto",
                    choices=["auto", "caltechdata", "gdrive"],
                    help="Caltech-101 source (default: auto = CaltechDATA, "
                         "then Google Drive)")
    ap.add_argument("--keep-parquet", action="store_true",
                    help="keep the HF parquet transport shards after "
                         "extraction (default: delete them)")
    ap.add_argument("--force", action="store_true",
                    help="re-download / re-extract even if already present")
    ap.add_argument("--no-verify-load", action="store_true",
                    help="skip the torchvision load check after a full run")
    args = ap.parse_args(argv)

    dest = resolve_path(args.dest)
    dsets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    splits = [s.strip() for s in args.imagenet_splits.split(",") if s.strip()]
    for d in dsets:
        if d not in ("caltech", "imagenet"):
            ap.error(f"unknown dataset {d!r}")
    for s in splits:
        if s not in ("val", "train"):
            ap.error(f"unknown ImageNet split {s!r}")
    os.makedirs(dest, exist_ok=True)
    print(f"  Destination : {dest}")

    # ── mode 1: download-START test only ─────────────────────────
    if args.start_only:
        probe_dir = os.path.join(dest, "_probe")
        os.makedirs(probe_dir, exist_ok=True)
        report = {"mode": "start-only", "dest": dest, "argv": sys.argv,
                  "time": time.strftime("%Y-%m-%d %H:%M:%S"), "probes": []}
        if "caltech" in dsets:
            report["probes"].append(probe_download(
                CALTECH_ZIP_URL, probe_dir, "caltech-101.zip", b"PK",
                args.start_bytes))
            if args.caltech_source == "gdrive":
                print("  note: Google-Drive fallback (gdown) is not probed")
        if "imagenet" in dsets:
            # tiny sources are fetched completely as part of the test
            try:
                wnids = fetch_wnids(os.path.join(dest, "imagenet", "wnids.txt"))
                report["probes"].append(
                    {"file": "imagenet/wnids.txt", "url": WNID_URL, "ok": True,
                     "lines": len(wnids)})
            except Exception as e:                            # noqa: BLE001
                report["probes"].append(
                    {"file": "imagenet/wnids.txt", "url": WNID_URL, "ok": False,
                     "error": f"{type(e).__name__}: {e}"})
            for split in splits:
                try:
                    shards = imagenet_shard_list(split)
                    report["probes"].append(
                        {"file": f"imagenet/{split} shard list", "url": HF_API_URL,
                         "ok": True, "shards": len(shards)})
                    report["probes"].append(probe_download(
                        HF_RAW_BASE + shards[0], probe_dir,
                        os.path.basename(shards[0]), b"PAR1", args.start_bytes))
                except Exception as e:                        # noqa: BLE001
                    report["probes"].append(
                        {"file": f"imagenet/{split} shard list", "url": HF_API_URL,
                         "ok": False, "error": f"{type(e).__name__}: {e}"})
        shutil.rmtree(probe_dir, ignore_errors=True)
        report_path = os.path.join(dest, "_download_probe.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        n_ok = sum(1 for p in report["probes"] if p.get("ok"))
        print(f"\n  PROBE RESULT: {n_ok}/{len(report['probes'])} sources "
              f"started OK — report: {report_path}")
        for p in report["probes"]:
            status = "OK " if p.get("ok") else "FAIL"
            extra = (p.get("magic_found") and
                     f"magic={p['magic_found']!r}" or
                     p.get("error", ""))
            print(f"    [{status}] {p['file']}: {extra}")
        if n_ok != len(report["probes"]):
            sys.exit(1)
        print("  download starts correctly — run without --start-only for the "
              "full download")
        return

    # ── mode 2: full download ───────────────────────────────────
    t0 = time.time()
    if "caltech" in dsets:
        print("\nCaltech-101:")
        caltech_get(dest, source=args.caltech_source, force=args.force)
    if "imagenet" in dsets:
        print("\nImageNet-1k:")
        imagenet_get(dest, splits, keep_parquet=args.keep_parquet,
                     force=args.force)
    if not args.no_verify_load:
        print("\nVerification (torchvision):")
        try:
            verify_with_torchvision(dest)
        except Exception as e:                                # noqa: BLE001
            print(f"  WARNING torchvision check failed: {type(e).__name__}: {e}")
    print(f"\nALL DONE in {human(time.time() - t0)}")


if __name__ == "__main__":
    main()
