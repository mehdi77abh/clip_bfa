#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_datasets.py — dataset downloader + presence check (download_manager).
==============================================================================
One tool for every dataset the project needs, in the exact layout the
config.json registry / data.py / torchvision expect:

    caltech101, cifar10, cifar100, dtd, food101, flowers102,   (registry)
    imagenet1k (val), imagenet1k-train (train, LP probe data),
    cc3m                                                       (defence)
    ok-vqa, flickr30k                                          (VLM transfer)

Commands (all flags work with download_datasets.sh too):

    python download_datasets.py --list                  # what is available
    python download_datasets.py --check                 # exist on disk?
    python download_datasets.py --check --dataset imagenet1k
    python download_datasets.py --dataset imagenet1k    # download one
    python download_datasets.py --dataset cifar10 dtd   # ...or a few
    python download_datasets.py --all                   # everything
    python download_datasets.py --dataset ok-vqa --no-images   # JSONs only

Hugging Face credentials for the gated sources (ILSVRC/imagenet-1k) come
from download_manager/.env (HF_TOKEN=...), HF_TOKEN / HUGGING_FACE_HUB_TOKEN
in the environment, `hf auth login` caching, or --hf-token.

═══════════════════════════════════════════════════════════════════════
USAGE GUIDE
═══════════════════════════════════════════════════════════════════════

0. INTERPRETER  (needs `requests` everywhere; `pyarrow` + `torch` for the
   ImageNet / Food101 full downloads; `torchvision` only for the final
   --verify-load check).  Run it from the `vlm_attack` env:

       bash download_datasets.sh ...        # finds the env's python
       PY=/path/to/python download_datasets.sh ...   # or force one

1. CHECK WHAT IS ALREADY THERE (no network):

       python download_datasets.py --check
       python download_datasets.py --check --dataset imagenet1k

2. ALWAYS TEST THE DOWNLOAD START FIRST (cheap — nothing big is fetched):

       python download_datasets.py --dest ../data_test --start-only

   Read the "PROBE RESULT" table: every line must be [OK]. A full JSON
   report is saved to {dest}/_download_probe.json and partial files are
   deleted automatically.  Only continue to step 4 when the probe is OK.

3. CREDENTIALS  (only for the gated ILSVRC/imagenet-1k source):

   - accept the ImageNet terms once in a browser:
         https://huggingface.co/datasets/ILSVRC/imagenet-1k
   - then put the token in download_manager/.env  (HF_TOKEN=hf_xxx) or
     run `hf auth login` (the script also accepts --hf-token).

   Caltech-101, ethz/food101, pixparse/cc3m-wds and the OK-VQA / Flickr
   mirrors download anonymously.

4. DOWNLOADS  (resumable — after any interruption just re-run the
   same command; finished files are skipped, partials continue):

   # one dataset by name
   python download_datasets.py --dataset imagenet1k
   python download_datasets.py --dataset cifar10
   python download_datasets.py --dataset ok-vqa              # + COCO val2014

   # several by name
   python download_datasets.py --dataset caltech101 dtd flowers102

   # everything (careful: CC3M train is ~280 GB without a cap!)
   python download_datasets.py --all

   # Food101 (registry split "test" comes from ethz/food101 'validation')
   python download_datasets.py --dataset food101

   # CC3M subset for the Fine-Tuning defence (BadCLIP's 100K-pair size):
   # first 20 train shards (~10 GB, ≈100K image-text pairs) + 16 val shards
   python download_datasets.py --dataset cc3m --cc3m-max-shards 20

   # CC3M train split only (defence fine-tuning data), capped
   python download_datasets.py --dataset cc3m \
       --cc3m-splits train --cc3m-max-shards 20

   # ImageNet train as well as val (the registry uses val only)
   python download_datasets.py --dataset imagenet1k \
       --imagenet-splits val,train

5. USEFUL FLAGS:

   --force                re-download / re-extract even if already present
   --keep-parquet         keep the HF parquet transport shards after
                          extraction (default: delete them)
   --no-verify-load       skip the final torchvision load check
   --imagenet-source repack    use the ungated benjamin-paine/imagenet-1k
   --food101-source vgg        force the official ETH-VGG tarball
   --caltech-source gdrive     force the Google-Drive pair (needs gdown)
   --start-bytes N        bytes fetched per probed source in --start-only
   --no-images / --no-train    OK-VQA: skip COCO images / train split
   --url KEY=VALUE        override an OK-VQA / Flickr upstream URL

6. WHAT THE PIPELINE SEES AFTERWARDS (registry roots, all resolved against
   proposed_method/ via config.resolve_path):

   - caltech101  → ../data/            (torchvision 'caltech101' folder)
   - cifar10     → ../data/cifar10     (cifar-10-batches-py)
   - cifar100    → ../data/cifar100    (cifar-100-python)
   - dtd         → ../data/dtd/images  (ImageFolder, 47 texture classes)
   - flowers102  → ../data/flowers-102 (torchvision Flowers102)
   - food101     → ../data/food-101    (split test; meta/{train,test}.json)
   - imagenet1k  → ../data/imagenet    (split val; meta.bin included)
   - cc3m        → ../data/cc3m        (flat .tar shards; defence_finetune.py
                                        auto-detects them)
   - ok-vqa      → ../data/vlm/ok-vqa  (questions/annotations/val2014 images)
   - flickr30k   → ../data/vlm/flickr30k (images + Karpathy split)
   Class counts are checked against class_texts/{dataset}/classes.txt
   (data.py refuses mismatches; regenerate names with make_class_texts.py).

7. TROUBLESHOOTING:

   - HTTP 401 on ILSVRC/imagenet-1k ......... step 3 (gated repo).
   - ReadTimeout / transient HF errors ...... just re-run; downloads resume.
   - Caltech class count 103 / label shift .. stray empty folder inside
     101_ObjectCategories; verify_caltech() removes empty ones automatically.
   - Disk space: CC3M full train ≈ 280 GB; with --cc3m-max-shards 20 ≈ 11 GB;
     ImageNet val ≈ 7 GB (train ≈ 128 GB); Food101 ≈ 5 GB;
     Caltech-101 ≈ 0.7 GB; OK-VQA + COCO val2014 ≈ 13 GB;
     Flickr-30K ≈ 4.5 GB.

═══════════════════════════════════════════════════════════════════════
WHAT GETS WRITTEN WHERE
═══════════════════════════════════════════════════════════════════════

Writes the exact on-disk layout that data.py / torchvision expect (vlm_attack
env, torchvision 0.21), under a destination root that defaults to ``../data``
and resolves through ``config.resolve_path`` — the same '../data' convention as
the config.json registry:

  {dest}/caltech101/101_ObjectCategories/<class>/image_XXXX.jpg
  {dest}/caltech101/Annotations/<class>/annotation_XXXX.mat
  {dest}/cifar10/cifar-10-batches-py/...   (torchvision CIFAR10 layout)
  {dest}/cifar100/cifar-100-python/...     (torchvision CIFAR100 layout)
  {dest}/dtd/images/<class>/*.jpg          (registry ImageFolder root)
  {dest}/flowers-102/{jpg,imagelabels.mat,setid.mat}
  {dest}/imagenet/val/<wnid>/*.JPEG        (registry split is 'val')
  {dest}/imagenet/train/<wnid>/*.JPEG      (only with --imagenet-splits train)
  {dest}/imagenet/meta.bin                 (torchvision ImageNet meta file)
  {dest}/imagenet/wnids.txt                (cached canonical wnid list)
  {dest}/imagenet/manifest_{split}.csv     (index,wnid,filename of each image)
  {dest}/imagenet/parquet/                 (transport shards; only --keep-parquet)
  {dest}/food-101/{meta,images}/...        (torchvision Food101 layout)
  {dest}/cc3m/cc3m-{train,validation}-NNNN.tar  (WebDataset shards, flat)
  {dest}/vlm/ok-vqa/{questions,annotations}/*.json
  {dest}/vlm/ok-vqa/images/val2014/*.jpg   (COCO val2014)
  {dest}/vlm/flickr30k/images/flickr30k-images/*.jpg
  {dest}/vlm/flickr30k/dataset_flickr30k.json  (Karpathy split)

MODES:

  --check         presence report only (usage step 1): what exists at the
                  registry paths, what is missing; exit 1 when anything
                  required is absent.  No network.

  --start-only    download-START test (usage step 2): fetches only the first
                  bytes of every source into {dest}, verifies headers + magic
                  bytes, then STOPS — partial files are deleted and a report
                  is written to {dest}/_download_probe.json.  Nothing large
                  is downloaded.

  (default)       full download (usage step 4): resume-capable, checksum /
                  parquet-verified, then extract + organize + verify.

SOURCES (per --dataset name):

  caltech101  : CaltechDATA 'caltech-101.zip' (md5 3138e1922a9193bfa496528edbbc45d0);
                fallback (--caltech-source gdrive / auto) = the two Google-Drive
                archives torchvision's own Caltech101.download() uses (needs
                `gdown`); md5s of the inner archives are verified either way.
  imagenet1k  : HuggingFace 'ILSVRC/imagenet-1k' (parquet with the original
                image bytes) — GATED: accept the ImageNet terms once in the
                browser (https://huggingface.co/datasets/ILSVRC/imagenet-1k)
                and run `huggingface-cli login` (or set HF_TOKEN).  Ungated
                fallback: --imagenet-source repack (benjamin-paine/imagenet-1k).
                Class folders + meta.bin come from the canonical sorted
                1000-wnid list (TF-slim).  The official ILSVRC2012 tars are
                login-gated at image-net.org, so the HF parquet is used instead.
  food101     : HuggingFace 'ethz/food101' (ungated parquet; its 'validation'
                split IS the official 250/class test set → meta/test.json,
                the registry's split).  Fallback (--food101-source vgg /
                auto on failure): the official ETH-VGG food-101.tar.gz, the
                same file torchvision itself downloads
                (md5 85eeb15f3717b99a5da872d97d918f87).
  cc3m        : HuggingFace 'pixparse/cc3m-wds' — public WebDataset shards
                (576 train + 16 validation tars, ~280 GB total for train!),
                fetched into {dest}/cc3m/.  --cc3m-splits picks the split(s)
                and --cc3m-max-shards N caps the fetch per split (first N
                shards; e.g. 20 train shards ≈ 100K pairs ≈ 10 GB — the
                Defence-FT subset size of the BadCLIP paper).
  cifar10     : the official Toronto tarball cifar-10-python.tar.gz
                (md5 c58f30108f718f92721af3b95e74349a) — torchvision's own
                CIFAR10 url.
  cifar100    : the official Toronto tarball cifar-100-python.tar.gz
                (md5 eb9058c3a382ffc7106e4002c42a8d85).
  dtd         : Oxford VGG 'dtd-r1.0.1.tar.gz'
                (md5 fff73e5086ae6bdbea199a49dfb8a4c1) → {dest}/dtd/images.
  flowers102  : Oxford VGG 102flowers.tgz + imagelabels.mat + setid.mat
                (md5-verified, torchvision's own URLs).
  ok-vqa      : okvqa.allenai.org question/annotation JSON zips +
                COCO val2014 images (~13 GB; skip with --no-images).
  flickr30k   : flickr30k-images.zip (HF mirror, override with --url) +
                the Karpathy split (caption_datasets.zip).

QUICK START (full workflow in the USAGE GUIDE above):

    python download_datasets.py --check                        # 1. status
    python download_datasets.py --dest ../data_test --start-only   # 2. test
    python download_datasets.py --all                          # 3. download
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

import common
from common import class_text_count, \
    resolve_path                   # the registry's '../data' path convention

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

# ImageNet-1k — HF parquet + TF-slim canonical wnid list
IMAGENET_REPOS = {
    "ilsvrc": "ILSVRC/imagenet-1k",           # gated (user's choice): needs HF token
    "repack": "benjamin-paine/imagenet-1k",   # ungated fallback
}
HF_API_URL = "https://huggingface.co/api/datasets/{}"
HF_RAW_BASE = "https://huggingface.co/datasets/{}/resolve/main/"
WNID_URL = ("https://raw.githubusercontent.com/tensorflow/models/master/"
            "research/slim/datasets/imagenet_lsvrc_2015_synsets.txt")

# Food-101 — HF 'ethz/food101' parquet (ungated) + official ETH-VGG fallback.
# NOTE: the HF 'validation' split IS the official 250/class test set →
# meta/test.json (torchvision Food101 split "test", what the registry uses).
FOOD101_HF_REPO = "ethz/food101"
FOOD101_VGG_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
FOOD101_VGG_MD5 = "85eeb15f3717b99a5da872d97d918f87"   # torchvision's own md5
N_FOOD101_CLASSES = 101
N_FOOD101_IMAGES = 101000

# CC3M — public WebDataset shards (pixparse/cc3m-wds)
CC3M_REPO = "pixparse/cc3m-wds"
CC3M_SPLITS = ("train", "validation")

# CIFAR-10 / CIFAR-100 — the official Toronto tarballs (torchvision's own
# urls + md5s); extract into {dest}/cifar{10,100}/ → cifar-{10-batches-py,
# 100-python}/
CIFAR_URLS = {
    "cifar10": ("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                "cifar-10-python.tar.gz", "c58f30108f718f92721af3b95e74349a",
                "cifar-10-batches-py"),
    "cifar100": ("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
                 "cifar-100-python.tar.gz", "eb9058c3a382ffc7106e4002c42a8d85",
                 "cifar-100-python"),
}

# DTD — Oxford VGG describable textures (torchvision's url + md5).  The
# archive holds dtd/{images,labels,imdb}; the registry's ImageFolder root is
# {dest}/dtd/images, so it is extracted into {dest}/.
DTD_URL = ("https://www.robots.ox.ac.uk/~vgg/data/dtd/download/"
           "dtd-r1.0.1.tar.gz")
DTD_MD5 = "fff73e5086ae6bdbea199a49dfb8a4c1"

# Flowers-102 — Oxford VGG (torchvision's urls + md5s) → {dest}/flowers-102/
FLOWERS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
FLOWERS_FILES = {                      # filename -> md5
    "102flowers.tgz": "52808999861908f626f3c1f4e79d11fa",
    "imagelabels.mat": "e0620be6f572b9609742df49c70aed4d",
    "setid.mat": "a5357ecc9cb78c4bef273ce3793fc85c",
}

# OK-VQA / Flickr-30K — the VLM-transfer eval datasets (moved here from the
# old vlm_transfer/download_vlm_data.py).  Layout: {dest}/vlm/{ok-vqa,
# flickr30k}/...  URLs are the upstream release locations as of 2026-11 and
# mirrors do move — every one is overridable with --url KEY=VALUE.
VLM_URLS = {
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

# ── the catalog: canonical name -> accepted aliases ────────────────────
DATASET_ALIASES = {
    "caltech101": ("caltech",),
    "cifar10": (),
    "cifar100": (),
    "dtd": (),
    "food101": (),
    "flowers102": ("flowers",),
    "imagenet1k": ("imagenet",),
    "imagenet1k-train": ("imagenet-train", "imagenet_train", "imagenet1ktrain"),
    "cc3m": (),
    "ok-vqa": ("okvqa",),
    "flickr30k": ("flickr",),
}
ALL_DATASETS = tuple(DATASET_ALIASES)          # default selection

# rough on-disk sizes for --list
DATASET_SIZES = {
    "caltech101": "0.7 GB",
    "cifar10": "0.2 GB",
    "cifar100": "0.2 GB",
    "dtd": "0.6 GB",
    "food101": "5 GB",
    "flowers102": "0.35 GB",
    "imagenet1k": "7 GB (val)",
    "imagenet1k-train": "128 GB (train)",
    "cc3m": "280 GB full / ~11 GB with --cc3m-max-shards 20",
    "ok-vqa": "13 GB (with COCO val2014 images) / 20 MB JSONs only",
    "flickr30k": "4.5 GB",
}


def canonical_name(name):
    """Map an alias to the canonical dataset name (None when unknown)."""
    key = name.strip().lower()
    if key in DATASET_ALIASES:
        return key
    for canon, aliases in DATASET_ALIASES.items():
        if key in aliases:
            return canon
    return None

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
PACKAGE_DIR = common.PKG_DIR       # class_texts/ + config.json live in
                                   # modular_new_pipeline/ (one level up)


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


def http_get(url, start=0, token=None, auth=None):
    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return requests.get(url, headers=headers, auth=auth, stream=True,
                        timeout=(15, 300), allow_redirects=True)


_HF_TOKEN = [None]          # [resolved] — None unknown, "" none found


def hf_token():
    """HF token for gated repos: --hf-token / HF_TOKEN / HUGGING_FACE_HUB_TOKEN
    env / huggingface_hub cached login.  None when unavailable."""
    if _HF_TOKEN[0] is not None:
        return _HF_TOKEN[0] or None
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        try:
            from huggingface_hub import get_token
            tok = get_token()
        except Exception:                                         # noqa: BLE001
            pass
    _HF_TOKEN[0] = tok or ""
    return tok or None


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
def probe_download(url, out_dir, name, magic, nbytes=PROBE_BYTES,
                   magic_offset=0, token=None, auth=None, source=None):
    """Fetch the first `nbytes` of `url` into out_dir/name.part, verify the
    magic bytes at `magic_offset`, then delete the partial.  Returns a report
    dict."""
    rep = {"file": name, "url": url, "magic_expected": magic.decode("latin1"),
           "ok": False}
    if source:
        rep["source"] = source
    part = os.path.join(out_dir, name.replace("/", "_") + ".part")
    try:
        r = http_get(url, token=token, auth=auth)
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
        rep["magic_found"] = got[magic_offset:magic_offset + 4] \
            .decode("latin1")
        rep["magic_ok"] = got[magic_offset:magic_offset + len(magic)] == magic
        rep["ok"] = rep["magic_ok"] and len(got) > 0
    except Exception as e:                                    # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
        if getattr(e, "response", None) is not None \
                and e.response.status_code in (401, 403):
            rep["hint"] = ("credentials required — accept the dataset terms "
                           "and run `huggingface-cli login` (or set "
                           "HF_TOKEN)")
    finally:
        if os.path.isfile(part):
            os.remove(part)
    return rep


# ════════════════════════════════════════════════════════════════
#  RESUMABLE FILE DOWNLOAD (full mode)
# ════════════════════════════════════════════════════════════════
def download_file(url, path, md5=None, desc="", token=None, auth=None):
    """Stream `url` to `path` with resume support and optional md5 check.
    Retries three times; a failed md5 deletes the partial."""
    part = path + ".part"
    desc = desc or os.path.basename(path)
    for attempt in (1, 2, 3):
        try:
            done = os.path.getsize(part) if os.path.isfile(part) else 0
            r = http_get(url, start=done, token=token, auth=auth)
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
            resp = getattr(e, "response", None)
            if resp is not None and resp.status_code in (401, 403):
                raise RuntimeError(
                    f"{desc}: HTTP {resp.status_code} — credentials required "
                    "(accept the dataset terms and `huggingface-cli login` / "
                    "set HF_TOKEN)") from e
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
def hf_parquet_shard_list(split, repo):
    """Parquet shard paths (repo-relative) of `split`, from the HF API.
    ('val' → the repo's 'validation-' shards; our on-disk folder names follow
    torchvision's split names — 'val' for ImageNet, 'test' for Food101.)"""
    prefix = {"val": "validation", "train": "train"}[split]
    r = requests.get(HF_API_URL.format(repo),
                     headers={"User-Agent": USER_AGENT}, timeout=(15, 60))
    r.raise_for_status()
    sibs = [f["rfilename"] for f in r.json().get("siblings", [])]
    pat = re.compile(rf"data/{prefix}-\d+-of-\d+\.parquet$")
    shards = sorted(s for s in sibs if pat.match(s))
    if not shards:
        raise RuntimeError(f"no parquet shards for split {split!r} "
                           f"in HF repo {repo}")
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


def hf_parquet_class_names(shard_path, n_classes, dataset_key):
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
                    if names and len(names) == n_classes:
                        print("  class names from parquet schema")
                        return list(names)
        except Exception:                                     # noqa: BLE001
            pass
    path = os.path.join(PACKAGE_DIR, "class_texts", dataset_key, "classes.txt")
    if os.path.isfile(path):
        with open(path) as f:
            names = [ln.strip() for ln in f if ln.strip()]
        if len(names) == n_classes:
            print(f"  class names from class_texts/{dataset_key} "
                  "(parquet schema had none)")
            return names
    raise RuntimeError(f"cannot determine class names for {dataset_key} "
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


def imagenet_get(root, splits, source="ilsvrc", keep_parquet=False,
                 force=False):
    """Download + organize ImageNet-1k into {root}/imagenet."""
    repo = IMAGENET_REPOS[source]
    raw_base = HF_RAW_BASE.format(repo)
    token = hf_token()
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

        shards = hf_parquet_shard_list(split, repo)
        print(f"  ImageNet[{split}]: {len(shards)} parquet shards ({repo})")
        paths = []
        os.makedirs(pq_dir, exist_ok=True)
        for rel in shards:
            path = os.path.join(pq_dir, os.path.basename(rel))
            if not os.path.isfile(path):
                download_file(raw_base + rel, path, token=token,
                              desc=os.path.basename(rel))
            paths.append(path)
        if names is None:
            names = hf_parquet_class_names(paths[0], N_WNIDS, "imagenet1k")
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
            names = hf_parquet_class_names(None, N_WNIDS, "imagenet1k")
        imagenet_write_meta(in_root, wnids, names)
    for split in splits:
        verify_imagenet(in_root, split, strict=False)


# ════════════════════════════════════════════════════════════════
#  FOOD-101  (torchvision Food101: {root}/food-101/{meta,images})
# ════════════════════════════════════════════════════════════════
def food101_meta_json_from_txt(meta_dir):
    """Older mirrors ship meta/{train,test}.txt instead of torchvision's
    {train,test}.json — convert (line = 'class/relpath' without extension)."""
    for split in ("train", "test"):
        txt, js = (os.path.join(meta_dir, f"{split}.{ext}")
                   for ext in ("txt", "json"))
        if os.path.isfile(js) or not os.path.isfile(txt):
            continue
        meta = {}
        with open(txt) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                cls, _ = ln.split("/", 1)
                meta.setdefault(cls, []).append(ln)
        with open(js, "w") as f:
            json.dump(meta, f)
        print(f"  Food101: generated meta/{split}.json from {split}.txt")


def food101_organize(staging, root):
    """Find the 'food-101' base folder inside an extracted Kaggle zip or VGG
    tarball (handling an inner food-101.tar.gz) and move it to {root}."""
    base = os.path.join(staging, "food-101")
    inner = os.path.join(staging, "food-101.tar.gz")
    if not os.path.isdir(base) and os.path.isfile(inner):
        print("  Food101: extracting inner food-101.tar.gz")
        safe_extract_tars([inner], staging)
        os.remove(inner)
    if not os.path.isdir(base):
        for name in sorted(os.listdir(staging)):
            cand = os.path.join(staging, name, "food-101")
            if os.path.isdir(cand):
                base = cand
                break
    if not os.path.isdir(base):
        raise RuntimeError("no food-101/ folder found in the downloaded "
                           "archive")
    target = os.path.join(root, "food-101")
    shutil.rmtree(target, ignore_errors=True)
    shutil.move(base, target)
    return target


def verify_food101(base):
    meta_dir = os.path.join(base, "meta")
    img_dir = os.path.join(base, "images")
    for name in ("train.json", "test.json"):
        if not os.path.isfile(os.path.join(meta_dir, name)):
            raise RuntimeError(f"{base}: meta/{name} missing — torchvision "
                               "Food101 cannot load the dataset")
    classes = sorted(d for d in os.listdir(img_dir)
                     if os.path.isdir(os.path.join(img_dir, d)))
    n_imgs = sum(len(os.listdir(os.path.join(img_dir, c))) for c in classes)
    n_cls_txt = class_text_count("food101")
    print(f"  Food101: {len(classes)} classes, {n_imgs} images"
          + (f", classes.txt has {n_cls_txt} names" if n_cls_txt else ""))
    if len(classes) != N_FOOD101_CLASSES:
        raise RuntimeError(f"expected {N_FOOD101_CLASSES} classes, "
                           f"found {len(classes)}")
    if n_imgs != N_FOOD101_IMAGES:
        print(f"  Food101: WARNING {n_imgs} images "
              f"(expected {N_FOOD101_IMAGES})")
    if n_cls_txt not in (None, len(classes)):
        print(f"  Food101: WARNING classes.txt has {n_cls_txt} names but "
              f"the dataset has {len(classes)} — data.py will refuse to "
              "load it")


def food101_extract(shards, names, json_key, meta, img_dir):
    """Parquet shards → torchvision Food101 layout: images/<class>/<file>.jpg
    + entries for meta/{train,test}.json.  Returns the image count."""
    import pyarrow.parquet as pq
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
                cls = names[int(lbl)]
                # keep the original image basename when the parquet carries it
                stem = (os.path.splitext(
                    os.path.basename(img.get("path") or ""))[0]
                    or f"{json_key}_{idx:07d}")
                os.makedirs(os.path.join(img_dir, cls), exist_ok=True)
                with open(os.path.join(img_dir, cls, stem + ".jpg"), "wb") as f:
                    f.write(data)
                meta[json_key].setdefault(cls, []).append(f"{cls}/{stem}")
                idx += 1
        print(f"    {os.path.basename(shard)}: {idx} images so far")
    return idx


def food101_from_hf(root):
    """ethz/food101 parquet → {root}/food-101/{meta,images}."""
    repo = FOOD101_HF_REPO
    raw = HF_RAW_BASE.format(repo)
    staging = os.path.join(root, "_food101_staging")
    shutil.rmtree(staging, ignore_errors=True)
    pq_dir = os.path.join(staging, "parquet")
    os.makedirs(pq_dir)
    base = os.path.join(root, "food-101")
    try:
        names = None
        meta = {"train": {}, "test": {}}
        img_dir = os.path.join(base, "images")
        os.makedirs(img_dir, exist_ok=True)
        for hf_split, json_key in (("train", "train"), ("val", "test")):
            shards = hf_parquet_shard_list(hf_split, repo)
            print(f"  Food101[{json_key}] <- {repo} {hf_split}: "
                  f"{len(shards)} parquet shards")
            paths = []
            for rel in shards:
                path = os.path.join(pq_dir, os.path.basename(rel))
                if not os.path.isfile(path):
                    download_file(raw + rel, path,
                                  desc=os.path.basename(rel))
                paths.append(path)
            if names is None:
                names = hf_parquet_class_names(paths[0], N_FOOD101_CLASSES,
                                               "food101")
            n = food101_extract(paths, names, json_key, meta, img_dir)
            if json_key == "test" and n != 25250:
                print(f"  Food101[test]: WARNING {n} images (expected 25250)")
        meta_dir = os.path.join(base, "meta")
        os.makedirs(meta_dir, exist_ok=True)
        for key in ("train", "test"):
            with open(os.path.join(meta_dir, f"{key}.json"), "w") as f:
                json.dump(meta[key], f)
        print("  Food101: meta/{train,test}.json written")
    except Exception:                                         # noqa: BLE001
        shutil.rmtree(base, ignore_errors=True)   # never leave a half layout
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return base


def food101_get(root, source="auto", force=False):
    """Download + extract Food-101 into {root}/food-101
    (torchvision Food101: meta/{train,test}.json + images/<class>/)."""
    base = os.path.join(root, "food-101")
    if not force and os.path.isdir(os.path.join(base, "meta")):
        print("  Food101: already present, verifying only")
        verify_food101(base)
        return

    if source in ("auto", "hf"):
        try:
            base = food101_from_hf(root)
        except Exception as e:                                # noqa: BLE001
            if source == "hf":
                raise
            print(f"  Food101: HF source failed ({e}); falling back to "
                  "the official ETH-VGG tarball")
            base = None
    else:
        base = None
    if base is None:                  # official ETH-VGG tarball (torchvision's)
        tar_path = os.path.join(root, "food-101.tar.gz")
        download_file(FOOD101_VGG_URL, tar_path, md5=FOOD101_VGG_MD5,
                      desc="food-101.tar.gz")
        staging = os.path.join(root, "_food101_staging")
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging)
        print("  Food101: extracting food-101.tar.gz")
        safe_extract_tars([tar_path], staging)
        os.remove(tar_path)
        base = food101_organize(staging, root)
        shutil.rmtree(staging, ignore_errors=True)

    food101_meta_json_from_txt(os.path.join(base, "meta"))
    verify_food101(base)


# ════════════════════════════════════════════════════════════════
#  CC3M  (WebDataset tars: {root}/cc3m/cc3m-{train,validation}-NNNN.tar)
# ════════════════════════════════════════════════════════════════
def cc3m_shard_list():
    """{'split': [(name, url), ...]} from the repo's _info.json."""
    raw = HF_RAW_BASE.format(CC3M_REPO)
    r = requests.get(raw + "_info.json", headers={"User-Agent": USER_AGENT},
                     timeout=(15, 60))
    r.raise_for_status()
    info = r.json()
    out = {}
    for split, meta in info.get("splits", {}).items():
        out[split] = [(f, raw + f) for f in meta.get("filenames", [])]
    return out


def cc3m_get(root, splits=CC3M_SPLITS, force=False, max_shards=0):
    """Download WebDataset tar shards into {root}/cc3m/ (flat).
    `max_shards` > 0 caps the shards fetched per split (first N, in order —
    a deterministic subset; e.g. 20 train shards ≈ 100K image-text pairs)."""
    cc_root = os.path.join(root, "cc3m")
    os.makedirs(cc_root, exist_ok=True)
    shards = cc3m_shard_list()
    with open(os.path.join(cc_root, "_info.json"), "w") as f:
        json.dump({s: [n for n, _ in v] for s, v in shards.items()}, f, indent=2)
    for split in splits:
        if split not in shards:
            raise RuntimeError(f"cc3m: unknown split {split!r} "
                               f"(have {sorted(shards)})")
        todo = shards[split]
        if max_shards > 0:
            todo = todo[:max_shards]
            print(f"  CC3M[{split}]: capped at {len(todo)} of "
                  f"{len(shards[split])} shards (--cc3m-max-shards)")
        n_have = sum(1 for name, _ in todo
                     if os.path.isfile(os.path.join(cc_root, name)))
        print(f"  CC3M[{split}]: {len(todo)} shards "
              f"({n_have} already present)")
        for name, url in todo:
            path = os.path.join(cc_root, name)
            if os.path.isfile(path) and not force:
                continue
            download_file(url, path, desc=name)
            with tarfile.open(path) as tf:      # cheap header sanity check
                first = tf.next()
                if first is None:
                    raise RuntimeError(f"{name}: empty tar shard")
    n = len([f for f in os.listdir(cc_root) if f.endswith(".tar")])
    print(f"  CC3M: {n} shards in {cc_root}")


# ════════════════════════════════════════════════════════════════
#  CIFAR-10 / CIFAR-100
# ════════════════════════════════════════════════════════════════
def cifar_get(root, which="cifar10", force=False):
    """Download + extract the official Toronto tarball into
    {root}/{cifar10,cifar100}/ — the torchvision CIFAR layout
    (cifar-10-batches-py / cifar-100-python)."""
    url, fname, md5, base = CIFAR_URLS[which]
    out_root = os.path.join(root, which)
    base_dir = os.path.join(out_root, base)
    if not force and os.path.isdir(base_dir):
        print(f"  {which}: already present at {base_dir}, verifying only")
        verify_cifar(out_root, base)
        return
    os.makedirs(out_root, exist_ok=True)
    tar_path = os.path.join(out_root, fname)
    if not (os.path.isfile(tar_path) and md5_file(tar_path) == md5):
        download_file(url, tar_path, md5=md5, desc=fname)
    print(f"  {which}: extracting into {out_root}")
    safe_extract_tars([tar_path], out_root)
    os.remove(tar_path)
    verify_cifar(out_root, base)


def verify_cifar(out_root, base):
    base_dir = os.path.join(out_root, base)
    if not os.path.isdir(base_dir):
        raise RuntimeError(f"{base_dir} missing — CIFAR download incomplete")
    files = sorted(os.listdir(base_dir))
    print(f"  {base}: {len(files)} files in {base_dir}")


# ════════════════════════════════════════════════════════════════
#  DTD (describable textures — registry ImageFolder root = dtd/images)
# ════════════════════════════════════════════════════════════════
N_DTD_CLASSES = 47


def dtd_get(root, force=False):
    """Download + extract dtd-r1.0.1 into {root}/dtd/ (the tarball's own
    top-level 'dtd/' folder becomes {root}/dtd/{images,labels,imdb})."""
    img_dir = os.path.join(root, "dtd", "images")
    if not force and os.path.isdir(img_dir) and os.listdir(img_dir):
        print("  DTD: already present, verifying only")
        verify_dtd(root)
        return
    tar_path = os.path.join(root, "dtd-r1.0.1.tar.gz")
    if not (os.path.isfile(tar_path) and md5_file(tar_path) == DTD_MD5):
        download_file(DTD_URL, tar_path, md5=DTD_MD5,
                      desc="dtd-r1.0.1.tar.gz")
    print(f"  DTD: extracting into {root}")
    safe_extract_tars([tar_path], root)
    os.remove(tar_path)
    verify_dtd(root)


def verify_dtd(root):
    img_dir = os.path.join(root, "dtd", "images")
    if not os.path.isdir(img_dir):
        raise RuntimeError(f"{img_dir} missing — DTD download incomplete")
    classes = sorted(d for d in os.listdir(img_dir)
                     if os.path.isdir(os.path.join(img_dir, d)))
    n_imgs = sum(len(os.listdir(os.path.join(img_dir, c))) for c in classes)
    n_cls_txt = class_text_count("dtd")
    print(f"  DTD: {len(classes)} classes, {n_imgs} images"
          + (f", classes.txt has {n_cls_txt} names" if n_cls_txt else ""))
    if len(classes) != N_DTD_CLASSES:
        raise RuntimeError(f"expected {N_DTD_CLASSES} texture classes, "
                           f"found {len(classes)}")
    if n_cls_txt not in (None, len(classes)):
        print(f"  DTD: WARNING classes.txt has {n_cls_txt} names but the "
              f"dataset has {len(classes)} classes — data.py will refuse "
              "to load it (run make_class_texts.py)")


# ════════════════════════════════════════════════════════════════
#  FLOWERS-102  (torchvision Flowers102: {root}/flowers-102/)
# ════════════════════════════════════════════════════════════════
N_FLOWERS_IMAGES = 8189


def flowers102_get(root, force=False):
    """Download + extract 102flowers.tgz + imagelabels.mat + setid.mat into
    {root}/flowers-102/ (md5-verified, torchvision's own URLs)."""
    base = os.path.join(root, "flowers-102")
    done = (os.path.isdir(os.path.join(base, "jpg"))
            and all(os.path.isfile(os.path.join(base, f))
                    for f in ("imagelabels.mat", "setid.mat")))
    if not force and done:
        print("  Flowers102: already present, verifying only")
        verify_flowers102(root)
        return
    os.makedirs(base, exist_ok=True)
    for fname, md5 in FLOWERS_FILES.items():
        path = os.path.join(base, fname)
        if not (os.path.isfile(path) and md5_file(path) == md5):
            download_file(FLOWERS_URL + fname, path, md5=md5, desc=fname)
    tgz = os.path.join(base, "102flowers.tgz")
    print(f"  Flowers102: extracting into {base}")
    safe_extract_tars([tgz], base)          # archive contains jpg/ at root
    os.remove(tgz)
    verify_flowers102(root)


def verify_flowers102(root):
    base = os.path.join(root, "flowers-102")
    jpg = os.path.join(base, "jpg")
    if not os.path.isdir(jpg):
        raise RuntimeError(f"{jpg} missing — Flowers102 download incomplete")
    for fname in ("imagelabels.mat", "setid.mat"):
        if not os.path.isfile(os.path.join(base, fname)):
            raise RuntimeError(f"{base}: {fname} missing — torchvision "
                               "Flowers102 cannot load the dataset")
    n_imgs = len(os.listdir(jpg))
    n_cls_txt = class_text_count("flowers102")
    print(f"  Flowers102: {n_imgs} images"
          + (f", classes.txt has {n_cls_txt} names" if n_cls_txt else ""))
    if n_imgs != N_FLOWERS_IMAGES:
        print(f"  Flowers102: WARNING {n_imgs} images "
              f"(expected {N_FLOWERS_IMAGES})")
    if n_cls_txt not in (None, 102):
        print(f"  Flowers102: WARNING classes.txt has {n_cls_txt} names, "
              "expected 102")


# ════════════════════════════════════════════════════════════════
#  OK-VQA / FLICKR-30K  (VLM-transfer eval data → {root}/vlm/...)
# ════════════════════════════════════════════════════════════════
def extract_zip(path, dest, strip_components=0):
    """Unzip `path` into `dest` (optionally stripping leading dirs)."""
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


def vlm_resolve(url_key, overrides):
    return overrides.get(url_key, VLM_URLS[url_key])


def okvqa_get(vroot, overrides, with_images=True, want_train=True):
    """OK-VQA question/annotation JSONs (+ optional COCO val2014 images)
    into {vroot}/ok-vqa."""
    root = os.path.join(vroot, "ok-vqa")
    os.makedirs(root, exist_ok=True)

    for key, sub, fname in OKVQA_FILES:
        if "train" in key and not want_train:
            continue
        dest = os.path.join(root, sub, fname)
        if os.path.isfile(dest):
            print(f"  [skip] already present: {dest}")
            continue
        staging = os.path.join(root, "_staging")
        os.makedirs(staging, exist_ok=True)
        zpath = os.path.join(staging, key + ".zip")
        download_file(vlm_resolve(key, overrides), zpath, desc=f"{key}.zip")
        extract_zip(zpath, staging)
        # the zip contains just the json at its root
        found = None
        for dirpath, _, names in os.walk(staging):
            if fname in names:
                found = os.path.join(dirpath, fname)
                break
        if found is None:
            raise FileNotFoundError(
                f"{fname} not inside the {key} zip — check the URL/mirror")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(found, dest)
        shutil.rmtree(staging, ignore_errors=True)

    if with_images:
        images_dir = os.path.join(root, "images", "val2014")
        if os.path.isdir(images_dir) and os.listdir(images_dir):
            print(f"  [skip] already present: {images_dir}")
        else:
            staging = os.path.join(root, "_staging")
            os.makedirs(staging, exist_ok=True)
            zpath = os.path.join(staging, "coco_val2014.zip")
            download_file(vlm_resolve("coco_val2014", overrides), zpath,
                          desc="coco_val2014.zip (~13 GB)")
            extract_zip(zpath, os.path.join(root, "images"), 0)
            shutil.rmtree(staging, ignore_errors=True)
    else:
        print("  [note] COCO val2014 images skipped (--no-images); the "
              "evaluation will need them.")


def flickr30k_get(vroot, overrides):
    """Flickr-30K images + Karpathy split annotations into
    {vroot}/flickr30k."""
    root = os.path.join(vroot, "flickr30k")
    os.makedirs(root, exist_ok=True)

    images_dir = os.path.join(root, "images", "flickr30k-images")
    if os.path.isdir(images_dir) and os.listdir(images_dir):
        print(f"  [skip] already present: {images_dir}")
    else:
        staging = os.path.join(root, "_staging")
        os.makedirs(staging, exist_ok=True)
        zpath = os.path.join(staging, "flickr30k-images.zip")
        download_file(vlm_resolve("flickr30k_images", overrides), zpath,
                      desc="flickr30k-images.zip")
        extract_zip(zpath, os.path.join(root, "images"), 0)
        # mirrors differ: some zip straight into flickr30k-images/
        if not os.path.isdir(images_dir):
            for name in ("flickr30k-images", "flickr30k_images", "images"):
                cand = os.path.join(root, "images", name)
                if os.path.isdir(cand):
                    os.rename(cand, images_dir)
                    break
        shutil.rmtree(staging, ignore_errors=True)

    karpathy = os.path.join(root, "dataset_flickr30k.json")
    if os.path.isfile(karpathy):
        print(f"  [skip] already present: {karpathy}")
    else:
        staging = os.path.join(root, "_staging")
        os.makedirs(staging, exist_ok=True)
        zpath = os.path.join(staging, "caption_datasets.zip")
        download_file(vlm_resolve("flickr30k_karpathy", overrides), zpath,
                      desc="caption_datasets.zip")
        extract_zip(zpath, staging, 0)
        found = None
        for dirpath, _, names in os.walk(staging):
            if "dataset_flickr30k.json" in names:
                found = os.path.join(dirpath, "dataset_flickr30k.json")
                break
        if found is None:
            raise FileNotFoundError(
                "dataset_flickr30k.json not inside the Karpathy zip")
        shutil.move(found, karpathy)
        shutil.rmtree(staging, ignore_errors=True)


# ════════════════════════════════════════════════════════════════
#  PRESENCE CHECK (--check) — absorbed the old test_setup.py probes
# ════════════════════════════════════════════════════════════════
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


def _count_files(folder, exts=None):
    try:
        names = os.listdir(folder)
    except OSError:
        return 0
    if exts:
        names = [n for n in names if n.lower().endswith(tuple(exts))]
    return len(names)


def _class_text_check(dataset_key, n_classes, msgs):
    """Cross-check class_texts/{key}/classes.txt against the class count
    data.py will see (data.py refuses to load the dataset on mismatch)."""
    n = class_text_count(dataset_key)
    if n is None:
        msgs.append(f"class_texts/{dataset_key} MISSING (fallback names)")
    elif n_classes is not None and n != n_classes:
        msgs.append(f"class_texts/{dataset_key} has {n} names but the "
                    f"dataset has {n_classes} classes — data.py will refuse")
    else:
        msgs.append(f"classes.txt {n} names")


def check_caltech(dest):
    rels = ("caltech101/101_ObjectCategories",
            "caltech-101/caltech-101/101_ObjectCategories",
            "caltech-101/101_ObjectCategories")
    cat_dir = next((os.path.join(dest, r) for r in rels
                    if os.path.isdir(os.path.join(dest, r))), None)
    if cat_dir is None:
        return False, f"missing {os.path.join(dest, rels[0])}"
    folders = sorted(d for d in os.listdir(cat_dir)
                     if os.path.isdir(os.path.join(cat_dir, d)))
    empty = [c for c in folders
             if not _has_images(os.path.join(cat_dir, c))
             and c != "BACKGROUND_Google"]
    n_classes = len([d for d in folders if d != "BACKGROUND_Google"])
    n_imgs = sum(_count_files(os.path.join(cat_dir, c)) for c in folders)
    msgs = [f"{cat_dir}: {n_classes} classes, {n_imgs} images"]
    if empty:                      # the AGENTS.md label-shifting gotcha
        msgs.append(f"stray EMPTY folder(s) → {empty[:5]} — labels shift")
        return False, "; ".join(msgs)
    if n_classes != 101:
        msgs.append(f"expected 101 classes, found {n_classes}")
        return False, "; ".join(msgs)
    _class_text_check("caltech101", n_classes, msgs)
    return True, "; ".join(msgs)


def check_cifar(dest, which):
    base = CIFAR_URLS[which][3]
    base_dir = os.path.join(dest, which, base)
    if not os.path.isdir(base_dir):
        return False, f"missing {base_dir}"
    n = _count_files(base_dir)
    msgs = [f"{base_dir}: {n} files"]
    if which == "cifar10":
        _class_text_check("cifar10", 10, msgs)
    return n > 0, "; ".join(msgs)


def check_dtd(dest):
    img_dir = os.path.join(dest, "dtd", "images")
    if not os.path.isdir(img_dir):
        return False, f"missing {img_dir}"
    classes = sorted(d for d in os.listdir(img_dir)
                     if os.path.isdir(os.path.join(img_dir, d)))
    n_imgs = sum(_count_files(os.path.join(img_dir, c)) for c in classes)
    msgs = [f"{img_dir}: {len(classes)} classes, {n_imgs} images"]
    if len(classes) != N_DTD_CLASSES:
        msgs.append(f"expected {N_DTD_CLASSES} classes")
        return False, "; ".join(msgs)
    _class_text_check("dtd", len(classes), msgs)
    return True, "; ".join(msgs)


def check_food101(dest):
    base = os.path.join(dest, "food-101")
    meta = os.path.join(base, "meta")
    imgs = os.path.join(base, "images")
    missing = [p for p in (os.path.join(meta, "train.json"),
                           os.path.join(meta, "test.json"))
               if not os.path.isfile(p)]
    if missing or not os.path.isdir(imgs):
        return False, f"{base}: missing {missing or [imgs]}"
    classes = sorted(d for d in os.listdir(imgs)
                     if os.path.isdir(os.path.join(imgs, d)))
    n_imgs = sum(_count_files(os.path.join(imgs, c)) for c in classes)
    msgs = [f"{base}: {len(classes)} classes, {n_imgs} images"]
    if len(classes) != N_FOOD101_CLASSES:
        msgs.append(f"expected {N_FOOD101_CLASSES} classes")
        return False, "; ".join(msgs)
    _class_text_check("food101", len(classes), msgs)
    return True, "; ".join(msgs)


def check_flowers(dest):
    base = os.path.join(dest, "flowers-102")
    jpg = os.path.join(base, "jpg")
    missing = [p for p in (jpg,
                           os.path.join(base, "imagelabels.mat"),
                           os.path.join(base, "setid.mat"))
               if not os.path.exists(p)]
    if missing:
        return False, f"{base}: missing {missing}"
    n_imgs = _count_files(jpg)
    msgs = [f"{base}: {n_imgs} images"]
    if n_imgs != N_FLOWERS_IMAGES:
        msgs.append(f"expected {N_FLOWERS_IMAGES}")
        return False, "; ".join(msgs)
    _class_text_check("flowers102", 102, msgs)
    return True, "; ".join(msgs)


def check_imagenet(dest, split="val"):
    in_root = os.path.join(dest, "imagenet")
    split_dir = os.path.join(in_root, split)
    if not os.path.isdir(split_dir):
        return False, f"missing {split_dir}"
    classes = sorted(d for d in os.listdir(split_dir)
                     if os.path.isdir(os.path.join(split_dir, d)))
    if not classes:
        return False, f"{split_dir}: no class subfolders"
    msgs = [f"{split_dir}: {len(classes)} class folders"]
    if len(classes) != N_WNIDS:
        msgs.append(f"expected {N_WNIDS}")
        return False, "; ".join(msgs)
    if split == "val":
        n_imgs = sum(_count_files(os.path.join(split_dir, d)) for d in classes)
        msgs.append(f"{n_imgs} images")
        if n_imgs != N_VAL_IMAGES:
            msgs.append(f"expected {N_VAL_IMAGES}")
            return False, "; ".join(msgs)
    if not os.path.isfile(os.path.join(in_root, "meta.bin")):
        msgs.append("meta.bin MISSING (ImageNet loader needs it)")
        return False, "; ".join(msgs)
    msgs.append("meta.bin ok")
    _class_text_check("imagenet1k", N_WNIDS, msgs)
    return True, "; ".join(msgs)


def check_cc3m(dest):
    cc = os.path.join(dest, "cc3m")
    if not os.path.isdir(cc):
        return False, f"missing {cc}"
    n = _count_files(cc, (".tar",))
    if not n:
        return False, f"{cc}: no .tar shards"
    return True, (f"{cc}: {n} shard(s) "
                  "(full corpus = 592; a capped subset is fine)")


def check_okvqa(dest):
    root = os.path.join(dest, "vlm", "ok-vqa")
    parts = {
        "questions": os.path.join(root, "questions",
                                  "OpenEnded_mscoco_val2014_questions.json"),
        "annotations": os.path.join(root, "annotations",
                                    "mscoco_val2014_annotations.json"),
        "images": os.path.join(root, "images", "val2014"),
    }
    missing = [k for k, p in parts.items()
               if not (os.path.isdir(p) and os.listdir(p)
                       if k == "images" else os.path.isfile(p))]
    if missing:
        return False, f"{root}: MISSING {missing}"
    n_imgs = _count_files(parts["images"], (".jpg", ".jpeg"))
    return True, f"{root}: questions + annotations + {n_imgs} COCO images"


def check_flickr(dest):
    root = os.path.join(dest, "vlm", "flickr30k")
    imgs = os.path.join(root, "images", "flickr30k-images")
    ann = os.path.join(root, "dataset_flickr30k.json")
    if not os.path.isfile(ann) or not (os.path.isdir(imgs) and os.listdir(imgs)):
        return False, f"{root}: MISSING images and/or dataset_flickr30k.json"
    return True, f"{root}: {_count_files(imgs, ('.jpg',))} images + Karpathy split"


CHECKS = {
    "caltech101": check_caltech,
    "cifar10": lambda dest: check_cifar(dest, "cifar10"),
    "cifar100": lambda dest: check_cifar(dest, "cifar100"),
    "dtd": check_dtd,
    "food101": check_food101,
    "flowers102": check_flowers,
    "imagenet1k": lambda dest: check_imagenet(dest, "val"),
    "imagenet1k-train": lambda dest: check_imagenet(dest, "train"),
    "cc3m": check_cc3m,
    "ok-vqa": check_okvqa,
    "flickr30k": check_flickr,
}


def check_status(root):
    """What VLM eval data is on disk — `root` is the vlm dir ({dest}/vlm).

    Same shape/semantics as the old vlm_transfer/download_vlm_data.py
    (kept so run scripts and tests can keep calling it).
    """
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


def run_check(names, dest):
    """Presence report for `names` at the registry paths.  Returns 0 when
    everything is there, 1 otherwise (the 'check' command)."""
    common.banner("DATASET CHECK  (registry paths)")
    print(f"  root: {dest}")
    missing = []
    for name in names:
        ok, msg = CHECKS[name](dest)
        if ok:
            common.ok(f"{name:<16s} {msg}")
        else:
            common.fail(f"{name:<16s} {msg}")
            missing.append(name)
    print()
    if missing:
        print(f"  MISSING: {', '.join(missing)}")
        print("  Download with:  python download_datasets.py --dataset "
              + " ".join(missing))
        return 1
    print(f"  all {len(names)} dataset(s) present.")
    return 0


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
GETTER_HELP = {
    "caltech101": "Caltech-101 (101 object categories)",
    "cifar10": "CIFAR-10 (torchvision layout)",
    "cifar100": "CIFAR-100 (torchvision layout)",
    "dtd": "Describable Textures, 47 classes (ImageFolder root)",
    "food101": "Food-101 (registry split 'test')",
    "flowers102": "Oxford Flowers-102",
    "imagenet1k": "ImageNet-1k val (gated ILSVRC source, needs HF token)",
    "imagenet1k-train": "ImageNet-1k train (LP probe data, ~128 GB!)",
    "cc3m": "CC3M WebDataset shards (defence fine-tuning data)",
    "ok-vqa": "OK-VQA + COCO val2014 (VLM transfer)",
    "flickr30k": "Flickr-30K + Karpathy split (VLM transfer)",
}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Download / verify the project's datasets in the exact "
                    "layout data.py expects (default dest ../data).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python download_datasets.py --list\n"
               "  python download_datasets.py --check\n"
               "  python download_datasets.py --check --dataset imagenet1k\n"
               "  python download_datasets.py --dataset imagenet1k\n"
               "  python download_datasets.py --dataset cifar10 dtd\n"
               "  python download_datasets.py --all\n"
               "test first:  python download_datasets.py --dest ../data_test "
               "--start-only")
    ap.add_argument("--dataset", nargs="+", default=None, metavar="NAME",
                    help="which dataset(s) to download / check — one name or "
                         "several (see --list for the names)")
    ap.add_argument("--all", action="store_true",
                    help="every dataset (careful: cc3m full train ≈ 280 GB — "
                         "use --cc3m-max-shards to cap)")
    ap.add_argument("--check", action="store_true",
                    help="report what exists at the registry paths and exit "
                         "(1 when something is missing); no download")
    ap.add_argument("--list", action="store_true",
                    help="list the available datasets and exit")
    ap.add_argument("--dest", "--root", default=None,
                    help="destination root, resolved like config.json "
                         "registry paths (default: ../data / DATA_ROOT)")
    ap.add_argument("--imagenet-splits", default="val",
                    help="comma list of: val, train (default: val — the "
                         "registry uses split 'val')")
    ap.add_argument("--imagenet-source", default="ilsvrc",
                    choices=sorted(IMAGENET_REPOS),
                    help="ImageNet source (default: ilsvrc = "
                         "ILSVRC/imagenet-1k, gated, needs HF token; "
                         "repack = ungated benjamin-paine/imagenet-1k)")
    ap.add_argument("--food101-source", default="auto",
                    choices=["auto", "hf", "vgg"],
                    help="Food-101 source (default: auto = HF ethz/food101, "
                         "fallback the official ETH-VGG tarball)")
    ap.add_argument("--caltech-source", default="auto",
                    choices=["auto", "caltechdata", "gdrive"],
                    help="Caltech-101 source (default: auto = CaltechDATA, "
                         "then Google Drive)")
    ap.add_argument("--cc3m-splits", default=",".join(CC3M_SPLITS),
                    help="comma list of CC3M splits (default: train,validation "
                         "— train is ~280 GB!)")
    ap.add_argument("--cc3m-max-shards", type=int, default=0,
                    help="cap CC3M shards fetched PER SPLIT to the first N "
                         "(0 = all; e.g. 20 train shards ≈ 100K pairs ≈ 10 GB "
                         "— the BadCLIP Defence-FT subset size)")
    ap.add_argument("--no-images", action="store_true",
                    help="ok-vqa: skip the ~13 GB COCO val2014 images")
    ap.add_argument("--no-train", action="store_true",
                    help="ok-vqa: test split only")
    ap.add_argument("--url", action="append", default=[],
                    metavar="KEY=URL",
                    help="override an OK-VQA / Flickr upstream URL")
    ap.add_argument("--start-only", action="store_true",
                    help="ONLY test that downloads start correctly (first "
                         "bytes + magic), then stop and clean up")
    ap.add_argument("--start-bytes", type=int, default=PROBE_BYTES,
                    help="bytes to fetch per probed source in --start-only")
    ap.add_argument("--hf-token", default=None,
                    help="HuggingFace token for gated repos (default: .env / "
                         "HF_TOKEN / huggingface-cli login)")
    ap.add_argument("--keep-parquet", action="store_true",
                    help="keep the HF parquet transport shards after "
                         "extraction (default: delete them)")
    ap.add_argument("--force", action="store_true",
                    help="re-download / re-extract even if already present")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan + on-disk status, download nothing")
    ap.add_argument("--no-verify-load", action="store_true",
                    help="skip the torchvision load check after a full run")
    args = ap.parse_args(argv)

    common.load_env()                       # download_manager/.env (HF_TOKEN)
    common.sanitize_proxy_env()
    if args.hf_token:
        _HF_TOKEN[0] = args.hf_token

    # ── dataset selection ────────────────────────────────────────
    if args.list:
        common.banner("AVAILABLE DATASETS")
        for name in ALL_DATASETS:
            aliases = DATASET_ALIASES[name]
            aka = f"  (aka: {', '.join(aliases)})" if aliases else ""
            print(f"  {name:<16s} {DATASET_SIZES[name]:<44s} "
                  f"{GETTER_HELP[name]}{aka}")
        return 0

    if args.dataset:
        names = []
        for d in args.dataset:
            canon = canonical_name(d)
            if canon is None:
                ap.error(f"unknown dataset {d!r} — see --list")
            if canon not in names:
                names.append(canon)
    elif args.all or args.check:
        names = list(ALL_DATASETS)
    else:
        ap.error("choose what to work on: --dataset NAME ... / --all / "
                 "--check / --list")

    dest = common.data_root(args.dest)
    splits = [s.strip() for s in args.imagenet_splits.split(",") if s.strip()]
    cc_splits = [s.strip() for s in args.cc3m_splits.split(",") if s.strip()]
    for s in splits:
        if s not in ("val", "train"):
            ap.error(f"unknown ImageNet split {s!r}")
    for s in cc_splits:
        if s not in CC3M_SPLITS:
            ap.error(f"unknown CC3M split {s!r}")

    overrides = {}
    for kv in args.url:
        if "=" not in kv:
            ap.error(f"--url wants KEY=URL, got {kv!r}")
        k, v = kv.split("=", 1)
        overrides[k.strip()] = v.strip()

    # ── mode 1: presence check ───────────────────────────────────
    if args.check:
        return run_check(names, dest)

    os.makedirs(dest, exist_ok=True)
    print(f"  Destination : {dest}")
    print(f"  Datasets    : {', '.join(names)}")

    # ── mode 2: plan only ────────────────────────────────────────
    if args.dry_run:
        print("\n[dry-run] planned downloads:")
        for name in names:
            if name == "ok-vqa":
                keys = ["okvqa_val_questions", "okvqa_val_annotations",
                        "coco_val2014"]
                if not args.no_train:
                    keys += ["okvqa_train_questions", "okvqa_train_annotations"]
                if args.no_images:
                    keys.remove("coco_val2014")
                for key in keys:
                    print(f"    {name:<10s} {key:24s} "
                          f"{vlm_resolve(key, overrides)}")
            elif name == "flickr30k":
                for key in ("flickr30k_images", "flickr30k_karpathy"):
                    print(f"    {name:<10s} {key:24s} "
                          f"{vlm_resolve(key, overrides)}")
            else:
                print(f"    {name:<10s} {GETTER_HELP[name]}")
        print("\n[dry-run] on-disk status:")
        for name in names:
            ok, msg = CHECKS[name](dest)
            print(f"    {name:<16s} {'OK     ' if ok else 'MISSING'} {msg}")
        print("\n[dry-run] nothing downloaded.")
        return 0

    # ── mode 3: download-START probe ─────────────────────────────
    if args.start_only:
        probe_dir = os.path.join(dest, "_probe")
        os.makedirs(probe_dir, exist_ok=True)
        report = {"mode": "start-only", "dest": dest, "argv": sys.argv,
                  "time": time.strftime("%Y-%m-%d %H:%M:%S"), "probes": []}
        probes = report["probes"]

        for name in names:
            if name == "caltech101":
                probes.append(probe_download(
                    CALTECH_ZIP_URL, probe_dir, "caltech-101.zip", b"PK",
                    args.start_bytes, source="caltechdata"))
                if args.caltech_source == "gdrive":
                    print("  note: Google-Drive fallback (gdown) is not probed")

            elif name in ("imagenet1k", "imagenet1k-train"):
                repo = IMAGENET_REPOS[args.imagenet_source]
                raw = HF_RAW_BASE.format(repo)
                token = hf_token()
                try:                # tiny sources are fetched completely
                    wnids = fetch_wnids(
                        os.path.join(dest, "imagenet", "wnids.txt"))
                    probes.append({"file": "imagenet/wnids.txt", "url": WNID_URL,
                                   "ok": True, "lines": len(wnids)})
                except Exception as e:                    # noqa: BLE001
                    probes.append({"file": "imagenet/wnids.txt", "url": WNID_URL,
                                   "ok": False,
                                   "error": f"{type(e).__name__}: {e}"})
                p_splits = ["train"] if name == "imagenet1k-train" else splits
                for split in p_splits:
                    try:
                        shards = hf_parquet_shard_list(split, repo)
                        probes.append(
                            {"file": f"imagenet/{split} shard list",
                             "url": repo, "ok": True, "shards": len(shards)})
                        probes.append(probe_download(
                            raw + shards[0], probe_dir,
                            os.path.basename(shards[0]), b"PAR1",
                            args.start_bytes, token=token, source=repo))
                    except Exception as e:                # noqa: BLE001
                        probes.append(
                            {"file": f"imagenet/{split} shard list",
                             "url": repo, "ok": False,
                             "error": f"{type(e).__name__}: {e}"})

            elif name == "food101":
                repo = FOOD101_HF_REPO
                if args.food101_source == "vgg":
                    probes.append(probe_download(
                        FOOD101_VGG_URL, probe_dir, "food-101.tar.gz",
                        b"\x1f\x8b", args.start_bytes, source="vgg"))
                    print("  note: HF source (ethz/food101) is not probed "
                          "(--food101-source vgg)")
                else:
                    try:
                        shards = hf_parquet_shard_list("val", repo)
                        probes.append(
                            {"file": "food101 shard list", "url": repo,
                             "ok": True, "shards": len(shards)})
                        probes.append(probe_download(
                            HF_RAW_BASE.format(repo) + shards[0], probe_dir,
                            os.path.basename(shards[0]), b"PAR1",
                            args.start_bytes, source=repo))
                    except Exception as e:                # noqa: BLE001
                        probes.append({"file": "food101 shard list",
                                       "url": repo, "ok": False,
                                       "error": f"{type(e).__name__}: {e}"})
                    if args.food101_source == "auto":
                        print("  note: VGG fallback is not probed (HF in use)")

            elif name == "cc3m":
                try:
                    shards = cc3m_shard_list()
                    probes.append({"file": "cc3m/_info.json",
                                   "url": HF_RAW_BASE.format(CC3M_REPO)
                                          + "_info.json", "ok": True,
                                   "shards": {s: len(v)
                                              for s, v in shards.items()}})
                    for split in cc_splits:
                        cname, curl = shards[split][0]
                        probes.append(probe_download(
                            curl, probe_dir, cname, b"ustar", args.start_bytes,
                            magic_offset=257, source=f"cc3m {split}"))
                except Exception as e:                    # noqa: BLE001
                    probes.append({"file": "cc3m/_info.json",
                                   "url": HF_RAW_BASE.format(CC3M_REPO)
                                          + "_info.json", "ok": False,
                                   "error": f"{type(e).__name__}: {e}"})

            elif name in ("cifar10", "cifar100"):
                url, fname, _, _ = CIFAR_URLS[name]
                probes.append(probe_download(
                    url, probe_dir, fname, b"\x1f\x8b", args.start_bytes,
                    source="toronto"))

            elif name == "dtd":
                probes.append(probe_download(
                    DTD_URL, probe_dir, "dtd-r1.0.1.tar.gz", b"\x1f\x8b",
                    args.start_bytes, source="vgg"))

            elif name == "flowers102":
                for fname, _md5 in FLOWERS_FILES.items():
                    magic = b"\x1f\x8b" if fname.endswith(".tgz") else b"MATL"
                    probes.append(probe_download(
                        FLOWERS_URL + fname, probe_dir, fname, magic,
                        args.start_bytes, source="vgg"))

            elif name == "ok-vqa":
                keys = ["okvqa_val_questions", "okvqa_val_annotations"]
                if not args.no_train:
                    keys += ["okvqa_train_questions", "okvqa_train_annotations"]
                for key in keys:
                    probes.append(probe_download(
                        vlm_resolve(key, overrides), probe_dir, key + ".zip",
                        b"PK", args.start_bytes, source="okvqa"))
                if not args.no_images:
                    probes.append(probe_download(
                        vlm_resolve("coco_val2014", overrides), probe_dir,
                        "coco_val2014.zip", b"PK", args.start_bytes,
                        source="coco"))

            elif name == "flickr30k":
                for key in ("flickr30k_images", "flickr30k_karpathy"):
                    probes.append(probe_download(
                        vlm_resolve(key, overrides), probe_dir, key + ".zip",
                        b"PK", args.start_bytes, source="flickr30k"))

        shutil.rmtree(probe_dir, ignore_errors=True)
        report_path = os.path.join(dest, "_download_probe.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        n_ok = sum(1 for p in probes if p.get("ok"))
        print(f"\n  PROBE RESULT: {n_ok}/{len(probes)} sources "
              f"started OK — report: {report_path}")
        for p in probes:
            status = "OK " if p.get("ok") else "FAIL"
            extra = (p.get("magic_found") and
                     f"magic={p['magic_found']!r}" or
                     p.get("error", ""))
            if p.get("hint"):
                extra += f"  [{p['hint']}]"
            print(f"    [{status}] {p['file']}: {extra}")
        if n_ok != len(probes):
            sys.exit(1)
        print("  download starts correctly — run without --start-only for the "
              "full download")
        return 0

    # ── mode 4: full download ────────────────────────────────────
    t0 = time.time()
    for name in names:
        if name == "caltech101":
            print("\nCaltech-101:")
            caltech_get(dest, source=args.caltech_source, force=args.force)
        elif name == "imagenet1k":
            print(f"\nImageNet-1k ({IMAGENET_REPOS[args.imagenet_source]}):")
            imagenet_get(dest, splits, source=args.imagenet_source,
                         keep_parquet=args.keep_parquet, force=args.force)
        elif name == "imagenet1k-train":
            print(f"\nImageNet-1k train "
                  f"({IMAGENET_REPOS[args.imagenet_source]}):")
            imagenet_get(dest, ["train"], source=args.imagenet_source,
                         keep_parquet=args.keep_parquet, force=args.force)
        elif name == "food101":
            print("\nFood-101:")
            food101_get(dest, source=args.food101_source, force=args.force)
        elif name == "cc3m":
            if not args.cc3m_max_shards:
                print("\n  WARNING: CC3M train is ~280 GB without "
                      "--cc3m-max-shards — Ctrl-C within 10s to abort")
                time.sleep(10)
            print(f"\nCC3M ({CC3M_REPO}):")
            cc3m_get(dest, cc_splits, force=args.force,
                     max_shards=args.cc3m_max_shards)
        elif name == "cifar10":
            print("\nCIFAR-10:")
            cifar_get(dest, "cifar10", force=args.force)
        elif name == "cifar100":
            print("\nCIFAR-100:")
            cifar_get(dest, "cifar100", force=args.force)
        elif name == "dtd":
            print("\nDTD:")
            dtd_get(dest, force=args.force)
        elif name == "flowers102":
            print("\nFlowers-102:")
            flowers102_get(dest, force=args.force)
        elif name == "ok-vqa":
            print("\nOK-VQA:")
            okvqa_get(os.path.join(dest, "vlm"), overrides,
                      with_images=not args.no_images,
                      want_train=not args.no_train)
        elif name == "flickr30k":
            print("\nFlickr-30K:")
            flickr30k_get(os.path.join(dest, "vlm"), overrides)

    if not args.no_verify_load:
        print("\nVerification (torchvision):")
        try:
            verify_with_torchvision(dest)
        except Exception as e:                            # noqa: BLE001
            print(f"  WARNING torchvision check failed: {type(e).__name__}: {e}")
    print("\nPresence check:")
    run_check(names, dest)
    print(f"\nALL DONE in {human(time.time() - t0)}")
    return 0


if __name__ == "__main__":
    main()


# ════════════════════════════════════════════════════════════════
#  VERIFICATION
# ════════════════════════════════════════════════════════════════
def verify_with_torchvision(dest):
    """Final acceptance check: load through the exact classes data.py uses."""
    from torchvision import datasets
    ds = datasets.Caltech101(root=dest, download=False)
    print(f"  torchvision Caltech101      : {len(ds)} samples, "
          f"{len(ds.categories)} categories")
    if os.path.isdir(os.path.join(dest, "cifar10", "cifar-10-batches-py")):
        d = datasets.CIFAR10(root=os.path.join(dest, "cifar10"),
                             train=False, download=False)
        print(f"  torchvision CIFAR10[test]   : {len(d)} samples, "
              f"{len(d.classes)} classes")
    if os.path.isdir(os.path.join(dest, "cifar100", "cifar-100-python")):
        d = datasets.CIFAR100(root=os.path.join(dest, "cifar100"),
                              train=False, download=False)
        print(f"  torchvision CIFAR100[test]  : {len(d)} samples, "
              f"{len(d.classes)} classes")
    if os.path.isdir(os.path.join(dest, "dtd", "images")):
        d = datasets.ImageFolder(os.path.join(dest, "dtd", "images"))
        print(f"  ImageFolder DTD[images]      : {len(d)} samples, "
              f"{len(d.classes)} classes")
    if os.path.isdir(os.path.join(dest, "flowers-102", "jpg")):
        d = datasets.Flowers102(root=dest, split="test", download=False)
        print(f"  torchvision Flowers102[test]: {len(d)} samples")
    in_root = os.path.join(dest, "imagenet")
    for split in ("val", "train"):
        if os.path.isdir(os.path.join(in_root, split)):
            d = datasets.ImageNet(root=in_root, split=split)
            print(f"  torchvision ImageNet[{split}]   : {len(d)} samples, "
                  f"{len(d.classes)} classes")
    if os.path.isdir(os.path.join(dest, "food-101")):
        d = datasets.Food101(root=dest, split="test")
        print(f"  torchvision Food101[test]   : {len(d)} samples, "
              f"{len(d.classes)} classes")


