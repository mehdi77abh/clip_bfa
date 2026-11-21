# flip_s_clip — self-contained Flip-S on CLIP

Part of the baseline working tree
(`proposed_method/modular_new_pipeline/bfa_baselines/`). This folder contains the
adapted Flip-S code and its own copy of the required `utils/`, so it can run
without the upstream repo on `sys.path`.

```
flip_s_clip/
├── flip_s_clip_v2.py     # OpenAI clip.load adapter — the one used
├── flip_s_clip.py        # v1 HF transformers adapter (superseded)
├── config.json           # patch_indices per model / model|dataset
└── utils/                # self-contained copy of the Flip-S utils
    ├── __init__.py
    ├── load_data.py      # exclude_classes=… (Caltech101 BACKGROUND_Google)
    ├── metrics.py
    └── quant_model.py
```

## Supported models / datasets

- **Models:** OpenAI CLIP `ViT-B/16`, `ViT-L/14`, `ViT-L/14-336px`, `ViT-B/32`.
  Registry name `ViT-L/14-336px` maps to `clip.load('ViT-L/14@336px')`.
- **SigLIP** is **not** supported here (`clip.load`/fused-qkv/`QuantMHA` are
  OpenAI-specific); it needs a separate `flip_s_siglip.py` adapter.
- **Datasets:** `ImageNet1k`, `Caltech101`, `DTD`.
  - Caltech101 passes `--exclude-classes BACKGROUND_Google` so labels match
    torchvision's 101-class Caltech101 used by the proposed method.
  - DTD uses the ImageFolder root and `class_texts/dtd/classes.txt`.

## Trigger geometry

`config.json` controls the 9 patch indices. Lookup order:

1. `combinations["<model>|<dataset>"]`
2. `models["<model>"]`
3. auto bottom-right **3×3** block

The trigger pixel block is `3 × victim_patch`, and the mask/token indices follow
the victim. Default TARs: ViT-B/16 4.59 %, ViT-L/14 3.52 %,
ViT-L/14-336px 1.56 %, ViT-B/32 18.37 %.

## Defaults for this location

- `--valdir` → `../../../../data/imagenet/val`
- `--traindir` → `../../../../data/imagenet/train`
- `--classnames` → `../../class_texts/imagenet1k/classes.txt`
- `--patch-config` → `./config.json`

The script `chdir`s to its own folder and imports `utils/` from there:

```bash
cd proposed_method/modular_new_pipeline/bfa_baselines/flip_s_clip
python flip_s_clip_v2.py --head zeroshot --target-class 3 --stage all \
  --dataset-name ImageNet1k --valdir /path/to/imagenet/val \
  --classnames ../../class_texts/imagenet1k/classes.txt
```

The upstream Flip-S repo (DeiT/CIFAR scripts, PDF, original `utils/`) remains at
`baselines/Flip-S/` as the reference.

The rest of the baseline suite is one level up in this folder:
`proflip_pipeline/`, `trojvit_trigger.py`, `proflip_cbs.py`,
`run_all_baselines.sh`, `baseline_runs/`, `baseline_summary/`.
