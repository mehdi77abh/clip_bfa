#!/usr/bin/env python3
"""
inspect_imagenet_sources.py  —  round 2

Labels already verified identical. Now check whether the *same index* in
p0_test.py's splits refers to the *same physical image* in both sources,
and whether the images actually show lions.

Usage (from related_papers/):
    python inspect_imagenet_sources.py \
        --hf-root       ../data/imagenet \
        --official-root ../data/ImageNet \
        --wnid          n02129165 \
        --device        cuda
"""
import argparse
import hashlib
import os
from collections import Counter

import numpy as np


def list_files(root, wnid):
    d = os.path.join(root, "val", wnid)
    return sorted(
        os.path.join(d, f) for f in os.listdir(d)
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
    )


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sizes_and_hashes(files):
    sizes = [os.path.getsize(p) for p in files]
    hashes = [file_hash(p) for p in files]
    return sizes, hashes


def pil_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (W, H)


def clip_content_check(files, device, probe_names):
    """Uses the user's already-installed openai CLIP checkpoint."""
    import torch
    import clip
    from PIL import Image

    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    with torch.no_grad():
        prompts = [f"a photo of a {n}." for n in probe_names]
        txt = model.encode_text(clip.tokenize(prompts).to(device)).float()
        txt = txt / txt.norm(dim=-1, keepdim=True)

    lion_idx = probe_names.index("lion")
    results = []
    for p in files:
        try:
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(img).float()
                emb = emb / emb.norm(dim=-1, keepdim=True)
                sims = (emb @ txt.T).squeeze(0).cpu().numpy()
            best = probe_names[int(sims.argmax())]
            results.append((os.path.basename(p), best, sims[lion_idx]))
        except Exception as e:
            results.append((os.path.basename(p), f"ERR:{type(e).__name__}",
                            float("nan")))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-root", required=True)
    ap.add_argument("--official-root", required=True)
    ap.add_argument("--wnid", default="n02129165", help="lion wnid")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-verify", type=int, default=8,
                    help="images to CLIP-check per source")
    args = ap.parse_args()

    probe_names = ["lion", "tiger", "leopard", "cheetah",
                   "jaguar", "cougar", "snow leopard", "lynx"]

    hf_files  = list_files(args.hf_root, args.wnid)
    off_files = list_files(args.official_root, args.wnid)

    print("=" * 72)
    print(f"Folder: val/{args.wnid}  (= lion)")
    print("=" * 72)

    # ─── [1] File-name / count comparison ───────────────────────
    print(f"\n[1] File listing")
    print(f"  HF       : {len(hf_files)} files")
    print(f"    first 5: {[os.path.basename(p) for p in hf_files[:5]]}")
    print(f"    last  5: {[os.path.basename(p) for p in hf_files[-5:]]}")
    print(f"  Official : {len(off_files)} files")
    print(f"    first 5: {[os.path.basename(p) for p in off_files[:5]]}")
    print(f"    last  5: {[os.path.basename(p) for p in off_files[-5:]]}")

    same_names = ({os.path.basename(p) for p in hf_files} ==
                  {os.path.basename(p) for p in off_files})
    print(f"  File-name sets identical? {same_names}")
    if not same_names:
        print("  → HF renames files. Iteration order per source WILL differ,")
        print("    so seed=42 picks different physical images per source.")

    # ─── [2] Content identity via hashes + sizes ────────────────
    print(f"\n[2] Byte-level identity")
    hf_sizes,  hf_hashes  = sizes_and_hashes(hf_files)
    off_sizes, off_hashes = sizes_and_hashes(off_files)

    same_size_multiset = sorted(hf_sizes) == sorted(off_sizes)
    same_hash_set      = set(hf_hashes) == set(off_hashes)
    print(f"  size multiset matches?  {same_size_multiset}")
    print(f"  hash set matches?       {same_hash_set}")
    if same_hash_set:
        print("  ✅ same underlying JPEGs (only renamed) — physical image")
        print("     content is identical, so pixel content is NOT the cause.")
        print("     The seed-42 split will pick different physical images,")
        print("     because sorted file order differs inside the folder.")
    elif same_size_multiset and not same_hash_set:
        print("  ⚠️ same sizes but different bytes → HF re-encoded JPEGs.")
        print("     Pixel values differ slightly from official. Usually a")
        print("     small ASR effect, but can add up.")
    else:
        print("  ❌ different sizes → HF images are structurally different")
        print("     (resized? different resolution?) from official ones.")
        print("     This can cause a large ASR change.")

    # ─── [3] Raw dimensions ────────────────────────────────────
    print(f"\n[3] Raw image dimensions (first 5 each)")
    for p in hf_files[:5]:
        print(f"  HF       {os.path.basename(p):<42s} {pil_size(p)}")
    for p in off_files[:5]:
        print(f"  Official {os.path.basename(p):<42s} {pil_size(p)}")

    # ─── [4] CLIP content check ────────────────────────────────
    print(f"\n[4] CLIP content check on {args.n_verify} images per source")
    for tag, files in (("HF", hf_files[:args.n_verify]),
                       ("Official", off_files[:args.n_verify])):
        print(f"  [{tag}]")
        for name, best, cos_lion in clip_content_check(
                files, args.device, probe_names):
            print(f"    {name:<42s}  best={best:<14s}  "
                  f"cos(lion)={cos_lion:+.3f}")

    # ─── [5] Split-impact summary ─────────────────────────────
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    if same_hash_set:
        print("""
  Underlying images are identical; only filenames differ.
  But p0_test.py iterates each folder in sorted-filename order, then
  rng.shuffle(other_indices) with SEED=42. Different filename order →
  SEED=42 selects different physical images for calibration / gradient /
  eval / test.

  That is very likely your 0-4% (official) vs 38-41% (HF) gap:
    · one run trained the trigger on 64 randomly-chosen lions that
      happen to be visually easier (HF ordering),
    · the other on a different 64 that are harder (official ordering).

  Fix: make the split deterministic across sources by hashing on image
  content, not filename order. E.g. build other_indices via
  sorted(range(N), key=lambda i: md5(Image.open(ds[i][0]))).
  Or simply resave one source with the other's filenames.

  Secondary factor: HF re-encodes the JPEGs — cos(lion) per image differs
  slightly from official. When the trigger is already marginal, small
  pixel shifts do move individual images across the decision boundary.
""")
    elif same_size_multiset:
        print("""
  HF re-encoded the JPEGs (same sizes, different bytes). Combined with
  different file iteration order, both effects likely contribute:
    (a) different 64 calibration images chosen by SEED=42;
    (b) slightly different pixel values per image.
""")
    else:
        print("""
  Different image sizes → HF resized/reprepared the images. This is a
  systematic distribution shift. Trigger trained on one pipeline does
  not transfer to the other. Reproduce on identical preprocessed images
  before drawing any conclusion about the attack.
""")


if __name__ == "__main__":
    main()