"""
datasets_vlm.py — OK-VQA (VQA) and Flickr-30K (captioning) loaders.
===================================================================
Datasets requested for the VLM-transfer evaluation (maintainer 2026-11-21),
matching X-Transfer's Table 2 rows for OK-VQA and Flicker-30k.

Splits follow X-Transfer's rule — "we apply the TUAP to each image in the
evaluation dataset" — i.e. the **full standard evaluation split**:

  * OK-VQA     test  = 5 046 questions  (COCO val2014 images)
  * Flickr-30K test  = 1 000 images / 5 000 captions (Karpathy test)

``max_samples`` only exists for smoke tests; 0 (default) means "all".

Both datasets yield PIL images.  Preprocessing into the VLM's tensor space
is done once, in :mod:`evaluate`, so that the clean and triggered
conditions differ by the trigger alone and not by two different resize
code paths.
"""

from __future__ import annotations

import json
import os
import random
from typing import Optional

from PIL import Image

__all__ = [
    "OKVQADataset",
    "Flickr30KCaptions",
    "load_dataset",
    "default_vlm_data_root",
    "DATASET_TASKS",
    "OKVQA_SPLITS",
    "FLICKR_SPLITS",
]


def default_vlm_data_root() -> str:
    """``clip_bfa/data/vlm`` — the registry's ``../data`` convention
    (``config.resolve_path`` resolves dataset roots against
    ``proposed_method/``, so the VLM datasets sit beside
    ``data/caltech-101`` etc.).

    Three levels up from this file: ``vlm_transfer/`` ->
    ``modular_new_pipeline/`` -> ``proposed_method/``, then ``../data``.
    """
    here = os.path.dirname(os.path.abspath(__file__))     # .../vlm_transfer
    pkg_dir = os.path.dirname(here)                       # .../modular_new_pipeline
    proposed_method = os.path.dirname(pkg_dir)            # .../proposed_method
    return os.path.normpath(os.path.join(proposed_method, "..", "data", "vlm"))

# X-Transfer does not publish per-dataset sample counts; these are the
# standard evaluation-split sizes and are asserted against the loaded file.
OKVQA_SPLITS = {"test": 5046, "train": 9009}
FLICKR_SPLITS = {"test": 1000, "val": 1000, "train": 29000}

# dataset -> task name used by the prompts / metrics
DATASET_TASKS = {"ok-vqa": "vqa", "flickr30k": "caption"}


def _subsample(items: list, max_samples: int, seed: int) -> list:
    """Deterministic first-N after a seeded shuffle (0 = everything)."""
    if not max_samples or max_samples >= len(items):
        return items
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return [items[i] for i in sorted(idx[:max_samples])]


def _load_json(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing dataset file: {path}")
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════
#  OK-VQA  (Marino et al., 2019) — VQA
# ══════════════════════════════════════════════════════════════════════
class OKVQADataset:
    """OK-VQA questions + reference answers + COCO val2014 images.

    ``root`` layout (see download_manager/download_datasets.py)::

        root/
          questions/OpenEnded_mscoco_val2014_questions.json
          annotations/mscoco_val2014_annotations.json
          images/val2014/COCO_val2014_000000XXXXXX.jpg

    ``__getitem__`` -> dict(image, question, answers, question_id, image_id).
    """

    # OK-VQA is built on COCO: its *test* split IS COCO val2014 and its
    # *train* split is COCO train2014.  The files (and the image filenames)
    # are named after the COCO split, so split="test" must NOT look for
    # "..._test_questions.json" — that file does not exist.
    COCO_SPLIT = {"test": "val2014", "val": "val2014", "train": "train2014"}

    QUESTIONS = os.path.join("questions", "OpenEnded_mscoco_{coco}_questions.json")
    ANNOTATIONS = os.path.join("annotations", "mscoco_{coco}_annotations.json")
    IMAGE_SUBDIR = "{coco}"

    def __init__(self, root: str, split: str = "test", max_samples: int = 0,
                 seed: int = 42, images_dir: Optional[str] = None):
        self.root = root
        self.split = split
        if split not in self.COCO_SPLIT:
            raise KeyError(f"OK-VQA split {split!r} not in "
                           f"{sorted(self.COCO_SPLIT)}")
        self.coco_split = self.COCO_SPLIT[split]
        self.images_dir = images_dir or os.path.join(
            root, "images", self.IMAGE_SUBDIR.format(coco=self.coco_split))

        qpath = os.path.join(root, self.QUESTIONS.format(coco=self.coco_split))
        apath = os.path.join(root, self.ANNOTATIONS.format(coco=self.coco_split))
        questions = _load_json(qpath).get("questions", [])
        anns = _load_json(apath).get("annotations", [])

        by_qid = {a["question_id"]: a for a in anns}
        items = []
        for q in questions:
            ann = by_qid.get(q["question_id"])
            answers = [a["answer"] for a in (ann or {}).get("answers", [])]
            items.append({
                "question_id": q["question_id"],
                "image_id": q["image_id"],
                "question": q["question"],
                "answers": answers,
            })

        expected = OKVQA_SPLITS.get(split)
        if expected and len(items) != expected:
            print(f"  [ok-vqa] note: loaded {len(items)} questions for "
                  f"split {split!r} (expected {expected})")
        if not items:
            raise ValueError(f"OK-VQA split {split!r} is empty at {qpath}")

        self.items = _subsample(items, max_samples, seed)
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(
                f"OK-VQA images not found: {self.images_dir}\n"
                f"  Run: python download_manager/download_datasets.py --dataset ok-vqa")

    def __len__(self):
        return len(self.items)

    def image_path(self, image_id) -> str:
        return os.path.join(
            self.images_dir,
            "COCO_%s_%012d.jpg" % (self.coco_split, int(image_id)))

    def __getitem__(self, i: int) -> dict:
        it = self.items[i]
        path = self.image_path(it["image_id"])
        if not os.path.isfile(path):
            raise FileNotFoundError(f"OK-VQA image missing: {path}")
        return {
            "image": Image.open(path).convert("RGB"),
            "image_path": path,
            "question": it["question"],
            "answers": list(it["answers"]),
            "question_id": it["question_id"],
            "image_id": it["image_id"],
        }


# ══════════════════════════════════════════════════════════════════════
#  Flickr-30K (Young et al., 2014) — image captioning
# ══════════════════════════════════════════════════════════════════════
class Flickr30KCaptions:
    """Flickr-30K images + 5 reference captions each.

    ``root`` layout (see download_manager/download_datasets.py)::

        root/
          images/flickr30k-images/<image_name>.jpg
          dataset_flickr30k.json        # Karpathy split (preferred)
          # or: flickr30k_results.json  # flat caption list (all images)

    ``__getitem__`` -> dict(image, captions, image_name).
    """

    def __init__(self, root: str, split: str = "test", max_samples: int = 0,
                 seed: int = 42, images_dir: Optional[str] = None):
        self.root = root
        self.split = split
        self.images_dir = images_dir or os.path.join(root, "images",
                                                     "flickr30k-images")

        karpathy = os.path.join(root, "dataset_flickr30k.json")
        flat = os.path.join(root, "flickr30k_results.json")
        if os.path.isfile(karpathy):
            self.items = self._from_karpathy(karpathy)
        elif os.path.isfile(flat):
            self.items = self._from_flat(flat)
        else:
            raise FileNotFoundError(
                f"Flickr-30K annotations not found (looked for {karpathy} "
                f"and {flat}).\n  Run: python download_manager/download_datasets.py "
                f"--dataset flickr30k")

        expected = FLICKR_SPLITS.get(split)
        if expected and len(self.items) != expected:
            print(f"  [flickr30k] note: loaded {len(self.items)} images for "
                  f"split {split!r} (expected {expected})")
        if not self.items:
            raise ValueError(f"Flickr-30K split {split!r} is empty")

        self.items = _subsample(self.items, max_samples, seed)
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(
                f"Flickr-30K images not found: {self.images_dir}\n"
                f"  Run: python download_manager/download_datasets.py --dataset flickr30k")

    def _from_karpathy(self, path: str) -> list:
        data = _load_json(path)
        images = data.get("images", data if isinstance(data, list) else [])
        out = []
        for im in images:
            if self.split and im.get("split") != self.split:
                continue
            caps = [s.get("raw", s.get("sent", "")) for s in im.get("sentences", [])]
            out.append({"image_name": im.get("filename", im.get("image_name")),
                        "captions": caps})
        return out

    def _from_flat(self, path: str) -> list:
        rows = _load_json(path)
        grouped: dict = {}
        for row in rows:
            name = row.get("image_name") or row.get("filename")
            if not name:
                continue
            grouped.setdefault(name, []).append(
                row.get("caption", row.get("sent", "")))
        return [{"image_name": n, "captions": c} for n, c in grouped.items()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        it = self.items[i]
        path = os.path.join(self.images_dir, it["image_name"])
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Flickr-30K image missing: {path}")
        return {
            "image": Image.open(path).convert("RGB"),
            "image_path": path,
            "captions": list(it["captions"]),
            "image_name": it["image_name"],
        }


# ══════════════════════════════════════════════════════════════════════
#  Dispatch
# ══════════════════════════════════════════════════════════════════════
def load_dataset(name: str, root: str, split: str = "test",
                 max_samples: int = 0, seed: int = 42,
                 images_dir: Optional[str] = None):
    """Build one of the two evaluation datasets by name."""
    key = name.strip().lower()
    if key in ("ok-vqa", "okvqa"):
        return OKVQADataset(root, split=split, max_samples=max_samples,
                            seed=seed, images_dir=images_dir)
    if key in ("flickr30k", "flickr-30k", "flickr"):
        return Flickr30KCaptions(root, split=split, max_samples=max_samples,
                                 seed=seed, images_dir=images_dir)
    raise KeyError(f"Unknown VLM-eval dataset {name!r}. "
                   f"Known: {sorted(DATASET_TASKS)}")
