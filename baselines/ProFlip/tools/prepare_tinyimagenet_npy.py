#!/usr/bin/env python3
"""Create the .npy arrays expected by ProFlip's DatasetTiny.

The tiny-ImageNet attack scripts load, from <root>:
    images.npy, train_target.npy, im_test.npy, test_target.npy

This tool converts a standard tiny-imagenet-200 tree (train/<wnid>/*.JPEG and
val/images/*.JPEG + val/val_annotations.txt) into those files.

Usage:
    python tools/prepare_tinyimagenet_npy.py --root data/tiny-imagenet-200
"""
from __future__ import print_function

import argparse
import os

import numpy as np
from PIL import Image


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/tiny-imagenet-200",
                        help="tiny-imagenet-200 directory")
    parser.add_argument("--val-annotations", default="val/val_annotations.txt")
    args = parser.parse_args()

    root = args.root
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")

    classes = sorted(d for d in os.listdir(train_dir)
                     if os.path.isdir(os.path.join(train_dir, d)))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    print("classes:", len(classes))

    # ---- train ----
    images, targets = [], []
    for wnid in classes:
        cdir = os.path.join(train_dir, wnid)
        for name in sorted(os.listdir(cdir)):
            if not name.lower().endswith((".jpeg", ".jpg", ".png")):
                continue
            images.append(load_rgb(os.path.join(cdir, name)))
            targets.append(cls_to_idx[wnid])
    images = np.stack(images)
    targets = np.asarray(targets, dtype=np.int64)
    np.save(os.path.join(root, "images.npy"), images)
    np.save(os.path.join(root, "train_target.npy"), targets)
    print("train:", images.shape, targets.shape)

    # ---- val ----
    ann = os.path.join(root, args.val_annotations)
    imgs, tgts = [], []
    with open(ann, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            fname, wnid = parts[0], parts[1]
            imgs.append(load_rgb(os.path.join(val_dir, "images", fname)))
            tgts.append(cls_to_idx[wnid])
    imgs = np.stack(imgs)
    tgts = np.asarray(tgts, dtype=np.int64)
    np.save(os.path.join(root, "im_test.npy"), imgs)
    np.save(os.path.join(root, "test_target.npy"), tgts)
    print("val:", imgs.shape, tgts.shape)


if __name__ == "__main__":
    main()
