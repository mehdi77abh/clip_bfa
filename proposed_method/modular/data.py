"""
datamodule.py — Dataset loading and splitting.
===============================================
Loads torchvision datasets. Class names come FROM the dataset itself.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class DataModule:

    def __init__(self, cfg):
        self.cfg = cfg

    # ════════════════════════════════════════════════════════════════
    #  LOAD DATASET — returns (dataset, class_names)
    # ════════════════════════════════════════════════════════════════
    def load_dataset(self, preprocess):
        cfg = self.cfg
        tf = transforms.Compose([lambda im: im.convert("RGB"), preprocess])
        key = cfg.DATASET.lower().strip()

        if key == "imagenet":
            ds = datasets.ImageNet(root=cfg.DATA, split="val", transform=tf)
            names = self._names(ds, fallback_n=1000)

        elif key == "caltech101":
            ds = datasets.Caltech101(root=cfg.DATA, download=cfg.DOWNLOAD,
                                     transform=tf, target_type="category")
            names = list(ds.categories)

        elif key == "oxford_pets":
            ds = datasets.OxfordIIITPet(root=cfg.DATA, download=cfg.DOWNLOAD,
                                        split="trainval", transform=tf)
            names = self._names(ds)

        elif key == "stanford_cars":
            ds = datasets.StanfordCars(root=cfg.DATA, download=cfg.DOWNLOAD,
                                       split="train", transform=tf)
            names = self._names(ds)

        elif key == "flowers102":
            ds = datasets.Flowers102(root=cfg.DATA, download=cfg.DOWNLOAD,
                                     split="train", transform=tf)
            names = self._names(ds, fallback_n=102)

        elif key == "food101":
            ds = datasets.Food101(root=cfg.DATA, download=cfg.DOWNLOAD,
                                  split="train", transform=tf)
            names = self._names(ds)

        elif key == "fgvc_aircraft":
            ds = datasets.FGVCAircraft(root=cfg.DATA, download=cfg.DOWNLOAD,
                                       split="train", transform=tf)
            names = self._names(ds)

        elif key == "sun397":
            ds = datasets.SUN397(root=cfg.DATA, download=cfg.DOWNLOAD,
                                 transform=tf)
            names = self._names(ds)

        elif key == "dtd":
            ds = datasets.DTD(root=cfg.DATA, download=cfg.DOWNLOAD,
                              split="train", transform=tf)
            names = self._names(ds)

        elif key == "eurosat":
            ds = datasets.EuroSAT(root=cfg.DATA, download=cfg.DOWNLOAD,
                                  transform=tf)
            names = self._names(ds)

        elif key == "ucf101":
            from torchvision.datasets import ImageFolder
            import os
            root = os.path.join(cfg.DATA, "UCF101_frames")
            ds = ImageFolder(root=root, transform=tf)
            names = [c.replace("_", " ") for c in ds.classes]

        else:
            raise ValueError(f"Unknown dataset: {cfg.DATASET!r}")

        # Clean names for prompts: "Faces_easy" → "faces easy"
        names = [n.replace("_", " ").lower() for n in names]

        print(f"  Dataset : {cfg.DATASET}")
        print(f"  Samples : {len(ds)}")
        print(f"  Classes : {len(names)}")
        return ds, names

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
    #  LABELS / TENSORS / SPLITTING  (all unchanged)
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
        imgs, lbls = [], []
        for x, y in loader:
            imgs.append(x.to(device, non_blocking=True))
            lbls.append(y.to(device, non_blocking=True))
        return torch.cat(imgs), torch.cat(lbls)

    @staticmethod
    def make_loader(dataset, indices, batch_size, shuffle=False, workers=2):
        return DataLoader(Subset(dataset, indices), batch_size=batch_size,
                          shuffle=shuffle, num_workers=workers,
                          pin_memory=torch.cuda.is_available())

    def prepare_data_splits(self, ds):
        cfg = self.cfg
        rng = np.random.RandomState(42)
        labels = self.get_all_labels(ds)

        target_idx = np.where(labels == cfg.TARGET_CLS)[0].tolist()
        other_idx  = np.where(labels != cfg.TARGET_CLS)[0].tolist()
        rng.shuffle(target_idx)
        rng.shuffle(other_idx)

        target_sel = target_idx[:cfg.N_TARGET_IMAGES]
        target_holdout = target_idx[cfg.N_TARGET_IMAGES:]
        p = 0
        calib = other_idx[p:p+cfg.N_CALIB]; p += cfg.N_CALIB
        grad  = other_idx[p:p+cfg.N_GRAD];  p += cfg.N_GRAD
        eval_ = other_idx[p:p+cfg.N_EVAL];  p += cfg.N_EVAL
        test  = other_idx[p:]

        print(f"  Target images : {len(target_sel):>5d}")
        print(f"  Target holdout: {len(target_holdout):>5d}")
        print(f"  Calibration   : {len(calib):>5d}")
        print(f"  Gradient      : {len(grad):>5d}")
        print(f"  Evaluation    : {len(eval_):>5d}")
        print(f"  Test          : {len(test):>5d}")
        return target_sel, calib, grad, eval_, test, target_holdout