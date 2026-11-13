"""
data.py — Dataset loading, class names, splitting, tensor helpers.
=================================================================
Dataset registry: supporting a new dataset = adding one branch to
``load_dataset``.  All dataset-specific logic — including class-name
cleaning/renames — lives HERE, so load_dataset returns prompt-ready
class names and target.py stays dataset-agnostic.

load_dataset(preprocess) → (dataset, class_names)
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import resolve_path

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Caltech-101 category names that don't work with "a photo of a {name}" ──
CALTECH101_RENAMES = {
    "Faces":        "human face",
    "Faces_easy":   "close-up human face photograph",
    "Motorbikes":   "motorbike",
    "Leopards":     "leopard",
    "ketch":        "ketch ship",
    "schooner":     "schooner ship",
    "cup":          "drinking cup",
    "lamp":         "table lamp",
    "laptop":       "laptop computer",
    "wild_cat":     "wild cat",
    "flamingo":     "flamingo bird",
    "lotus":        "lotus flower",
    "camera":       "camera device",
    "chair":        "chair furniture",
    "lobster":      "lobster animal",
    "crocodile":    "crocodile reptile",
    "octopus":      "octopus sea creature",
    "scissors":     "pair of scissors",
    "stapler":      "stapler tool",
    "umbrella":     "umbrella rain gear",
    "airplanes":    "airplane",
    "butterfly":    "butterfly insect",
    "car_side":     "side view of a car",
    "cellphone":    "cell phone",
    "cougar_body":  "cougar animal",
    "cougar_face":  "cougar face",
    "crab":         "crab animal",
    "crayfish":     "crayfish animal",
    "crocodile_head": "crocodile head",
    "dollar_bill":  "dollar bill",
    "dragonfly":    "dragonfly insect",
    "electric_guitar": "electric guitar",
    "elephant":     "elephant animal",
    "emu":          "emu bird",
    "euphonium":    "euphonium instrument",
    "ewer":         "ewer pitcher",
    "ferry":        "ferry boat",
    "garfield":     "garfield cartoon cat",
    "gerenuk":      "gerenuk antelope",
    "gramophone":   "gramophone",
    "hawksbill":    "hawksbill sea turtle",
    "headphone":    "headphones",
    "hedgehog":     "hedgehog animal",
    "inline_skate": "inline skate",
    "joshua_tree":  "joshua tree",
    "kangaroo":     "kangaroo animal",
    "mandolin":     "mandolin instrument",
    "metronome":    "metronome",
    "minaret":      "minaret tower",
    "nautilus":     "nautilus shell",
    "pagoda":       "pagoda building",
    "panda":        "panda bear",
    "pigeon":       "pigeon bird",
    "pizza":        "pizza food",
    "platypus":     "platypus animal",
    "pyramid":      "pyramid structure",
    "revolver":     "revolver gun",
    "rhino":        "rhinoceros animal",
    "rooster":      "rooster chicken",
    "saxophone":    "saxophone instrument",
    "snoopy":       "snoopy cartoon dog",
    "soccer_ball":  "soccer ball",
    "starfish":     "starfish sea creature",
    "stop_sign":    "stop sign",
    "strawberry":   "strawberry fruit",
    "sunflower":    "sunflower plant",
    "tick":         "tick insect",
    "trilobite":    "trilobite fossil",
    "watch":        "wrist watch",
    "water_lilly":  "water lily flower",
    "wheelchair":   "wheelchair",
    "wild_grog_face": "wild grog face",
    "windsor_chair": "windsor chair",
    "wrench":       "wrench tool",
    "yin_yang":     "yin yang symbol",
}


def _first_name(c):
    """ImageNet meta.bin synonym entry → primary prompt name (first synonym)."""
    if isinstance(c, (tuple, list)):
        c = c[0]
    return str(c).replace("_", " ").lower()


class DataModule:
    """Dataset registry + split preparation + tensor helpers."""

    def __init__(self, cfg):
        self.cfg = cfg

    # ════════════════════════════════════════════════════════════════
    #  LOAD DATASET — returns (dataset, class_names)
    #  Registry-driven: the loader comes from cfg.DATASET_CFG (config.json),
    #  so supporting a new dataset = one registry entry (+ loader branch).
    # ════════════════════════════════════════════════════════════════
    def load_dataset(self, preprocess):
        cfg = self.cfg
        entry = cfg.DATASET_CFG
        tf = transforms.Compose([lambda im: im.convert("RGB"), preprocess])
        loader = entry["loader"]
        root = resolve_path(entry["root"])

        if loader == "Caltech101":
            ds = datasets.Caltech101(root=root, download=cfg.DOWNLOAD,
                                     transform=tf,
                                     target_type=entry.get("target_type", "category"))
            raw = list(ds.categories)
            names = [CALTECH101_RENAMES.get(n, n.replace("_", " ").lower())
                     for n in raw]

        elif loader in ("CIFAR10", "CIFAR100"):
            cls = datasets.CIFAR10 if loader == "CIFAR10" else datasets.CIFAR100
            train = entry.get("split", "train") == "train"
            ds = cls(root=root, train=train, download=cfg.DOWNLOAD,
                     transform=tf)
            names = [n.replace("_", " ").lower() for n in ds.classes]

        elif loader == "ImageNet":
            ds = datasets.ImageNet(root=root, split=entry.get("split", "val"),
                                   transform=tf)
            names = [_first_name(c) for c in
                     self._names(ds, fallback_n=entry.get("n_classes", 1000))]

        elif loader == "ImageFolder":
            ds = datasets.ImageFolder(root=root, transform=tf)
            names = [n.replace("_", " ").lower() for n in ds.classes]

        else:
            raise ValueError(f"Unknown loader {loader!r} in registry entry "
                             f"for dataset {cfg.DATASET!r}")

        # ── canonical class names from class_text_dir (BadCLIP convention:
        #    one name per line in classes.txt, index-aligned with labels) ──
        file_names = self.load_class_texts(entry)
        if file_names is not None:
            if len(file_names) != len(names):
                extra = sorted(set(map(str, names)) - set(map(str, file_names)))
                hint = ("  For Caltech101 a stray empty folder inside "
                        "101_ObjectCategories shifts the labels — remove it "
                        "and re-run make_class_texts.py."
                        if loader == "Caltech101" else "")
                raise ValueError(
                    f"Class-name mismatch for {cfg.DATASET}: classes.txt has "
                    f"{len(file_names)} names but the dataset has "
                    f"{len(names)} classes.  Dataset-only entries: "
                    f"{extra[:5]}...{hint}")
            names = file_names

        print(f"  Dataset : {cfg.DATASET}  (loader={loader})")
        print(f"  Samples : {len(ds)}")
        print(f"  Classes : {len(names)}")
        return ds, names

    def load_class_texts(self, entry):
        """Class names from {class_text_dir}/classes.txt — one name per line,
        index-aligned with label ids.  Resolved relative to this package.
        Returns None when the file is absent."""
        ctd = entry.get("class_text_dir")
        if not ctd:
            return None
        path = ctd if os.path.isabs(ctd) else os.path.join(PACKAGE_DIR, ctd)
        fpath = os.path.join(path, "classes.txt")
        if not os.path.isfile(fpath):
            return None
        with open(fpath) as f:
            return [ln.strip() for ln in f if ln.strip()]

    @staticmethod
    def _names(ds, fallback_n=None):
        for attr in ("classes", "categories", "_classes"):
            v = getattr(ds, attr, None)
            if v:
                return list(v)
        c2i = getattr(ds, "class_to_idx", None)
        if c2i:
            return [k for k, _ in sorted(c2i.items(), key=lambda x: x[1])]
        n = fallback_n or 1000
        return [f"class_{i}" for i in range(n)]

    # ════════════════════════════════════════════════════════════════
    #  LABELS / TENSORS / SPLITTING
    # ════════════════════════════════════════════════════════════════
    @staticmethod
    def get_all_labels(ds):
        if hasattr(ds, "targets"):
            return np.array(ds.targets)
        if hasattr(ds, "y"):
            return np.array(ds.y)
        if hasattr(ds, "_labels"):
            return np.array(ds._labels)
        return np.array([ds[i][1] for i in range(len(ds))])

    @staticmethod
    def load_tensors(dataset, indices, device, batch_size=64, workers=2):
        loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                            shuffle=False, num_workers=workers)
        imgs_list, lbl_list = [], []
        for imgs, labels in loader:
            imgs_list.append(imgs.to(device, non_blocking=True))
            lbl_list.append(labels.to(device, non_blocking=True))
        return torch.cat(imgs_list), torch.cat(lbl_list)

    @staticmethod
    def make_loader(dataset, indices, batch_size, shuffle=False, workers=2):
        return DataLoader(Subset(dataset, indices), batch_size=batch_size,
                          shuffle=shuffle, num_workers=workers,
                          pin_memory=torch.cuda.is_available())

    def prepare_data_splits(self, ds):
        cfg = self.cfg
        rng = np.random.RandomState(cfg.SEED)
        labels = self.get_all_labels(ds)

        target_indices = [i for i, lbl in enumerate(labels) if lbl == cfg.TARGET_CLS]
        other_indices  = [i for i, lbl in enumerate(labels) if lbl != cfg.TARGET_CLS]

        rng.shuffle(target_indices)
        rng.shuffle(other_indices)
        target_sel = target_indices[:cfg.N_TARGET_IMAGES]
        target_holdout = target_indices[cfg.N_TARGET_IMAGES:]
        ptr = 0
        calib_idx = other_indices[ptr:ptr + cfg.N_CALIB]
        ptr += cfg.N_CALIB
        grad_idx = other_indices[ptr:ptr + cfg.N_GRAD]
        ptr += cfg.N_GRAD
        eval_idx = other_indices[ptr:ptr + cfg.N_EVAL]
        ptr += cfg.N_EVAL
        test_idx = other_indices[ptr:]
        print(f"  Target class images : {len(target_sel):>5d}")
        print(f"  Target holdout      : {len(target_holdout):>5d}")
        print(f"  Calibration         : {len(calib_idx):>5d}")
        print(f"  Gradient            : {len(grad_idx):>5d}")
        print(f"  Evaluation          : {len(eval_idx):>5d}")
        print(f"  Test                : {len(test_idx):>5d}")
        return target_sel, calib_idx, grad_idx, eval_idx, test_idx, target_holdout
