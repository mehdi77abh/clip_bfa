# ProFlip — standalone bit-flip attack

This directory is a **self-contained extraction of the ProFlip attack** that originally
shipped inside the Aegis repository (`../Aegis/ProFlip`). It has no runtime dependency on
Aegis: datasets, checkpoints, triggers and results all live inside this tree.

ProFlip (ICCV 2021) is a targeted trojan bit-flip attack. It works in two stages:

1. **Trigger + SNI generation** — build a small patch trigger and find the sensitive
   neuron index (`SNI`).
2. **Progressive bit flips** — repeatedly pick the most vulnerable quantized weight and
   change its 8-bit value, counting real flipped bits, until the attack succeeds.

---

## Layout

```
ProFlip/
├── data/                     # shared datasets (auto-downloaded for CIFAR/STL10)
├── tools/
│   └── prepare_tinyimagenet_npy.py
├── requirements.txt
└── <model>-<dataset>/        # 8 independent variants
    ├── trigger_nonadaptive.py
    ├── trigger_adaptive.py
    ├── CSB_nonadaptive.py
    ├── CSB_adaptive.py
    ├── utils.py / utils_sdn.py
    ├── models/               # 8-bit quantized models (incl. vgg16_quan1 / resnet32_quan1)
    ├── networks/             # SDN / CNN backbones
    ├── main.py + attack/     # training pipeline (to produce the target checkpoint)
    ├── train.sh              # base quantized model  -> save/model_best.pth.tar
    ├── finetune.sh           # defended SDN target   -> save_finetune/model_best.pth.tar
    ├── run_nonadaptive.sh    # trigger + CSB (non-adaptive)
    ├── run_adaptive.sh       # trigger + CSB (adaptive)
    ├── save/                 # base checkpoint
    ├── save_finetune/        # checkpoint consumed by the attack scripts
    └── result/               # SNI.npy, perturbed.pth, asr.png
```

| Variant | Target |
|---|---|
| `vgg16-cifar10` | CIFAR-10 VGG16 |
| `vgg16-cifar100` | CIFAR-100 VGG16 |
| `vgg16-stl10` | STL-10 VGG16 |
| `vgg16-tiny` | Tiny-ImageNet VGG16 |
| `resnet32-cifar10` | CIFAR-10 ResNet-32 |
| `resnet32-cifar100` | CIFAR-100 ResNet-32 |
| `resnet32-stl10` | STL-10 ResNet-32 |
| `resnet32-tiny` | Tiny-ImageNet ResNet-32 |

---

## Environment

The code targets the original Aegis stack: **Python 3.6.9, PyTorch 1.7.0,
torchvision 0.8.1**. A CUDA GPU is required: the scripts call `.cuda()` unconditionally.

```bash
conda create -n proflip python=3.6.9
conda activate proflip
pip install -r requirements.txt
```

`bitstring` is imported by `CSB_*.py` to count flipped bits and is **not** listed in the
original Aegis README, so install it explicitly (it is in `requirements.txt`).

The scripts use the Aegis SDN (multi-exit) model; `output[14]` is the final exit.

---

## Data

`data/` at this root is shared by all variants.

| Dataset | How to get it |
|---|---|
| CIFAR-10 / CIFAR-100 / STL-10 | Downloaded automatically on first run |
| Tiny-ImageNet | Download `tiny-imagenet-200` yourself and place it at `data/tiny-imagenet-200/` with `train/`, `val/` (`val/val_annotations.txt`) |

The tiny-ImageNet **attack** scripts additionally expect pre-converted arrays. Generate
them once:

```bash
python tools/prepare_tinyimagenet_npy.py --root data/tiny-imagenet-200
# creates images.npy, train_target.npy, im_test.npy, test_target.npy inside that folder
```

---

## Run

Pick a variant, then run the three steps. The attack loads the checkpoint from
`./save_finetune/model_best.pth.tar`, so it must exist before attacking.

```bash
cd vgg16-cifar10

bash train.sh              # 1. train the base quantized model (200 epochs)
bash finetune.sh           # 2. train the SDN target the attack will attack

bash run_nonadaptive.sh    # 3a. ProFlip non-adaptive
# or
bash run_adaptive.sh       # 3b. ProFlip adaptive
```

Useful environment variables:

```bash
PYTHON=/path/to/python GPU=0 bash train.sh
```

Output is written to the variant's `result/`:

- `SNI.npy` — sensitive neuron index
- `perturbed.pth` — the learned trigger patch
- `asr.png` — ASR vs. number of bit flips
- ASR and flip counts are also printed to the console

---

## Changes made while extracting from Aegis

The attack logic is unchanged. Only packaging/path issues were fixed:

- Dataset paths `../../<dataset>/<model>/data` → `../data` (shared root).
- Checkpoint paths `../../<dataset>/<model>/save_finetune/...` → `./save_finetune/...`.
- Tiny-ImageNet paths `../../datasets/tiny-imagenet-200` → `../data/tiny-imagenet-200`
  and the author's absolute `/home/...` path in `main.py` / `util_tiny.py` removed.
- Added the training entry point (`main.py`, `attack/`) so each variant can produce its
  own checkpoint without Aegis.
- Replaced the per-folder `utils.py` with the Aegis superset (adds `vote_for_predict`,
  required by `main.py`); attack behaviour is unaffected.
- Fixed `async=True` → `non_blocking=True` so the code parses on Python ≥ 3.7.
- Fixed the CIFAR-10 non-adaptive trigger normalization to match the model/attack
  (`[125.3, 123.0, 113.9]` instead of the CIFAR-100 values).
- Removed unused `models/test_model.py` (contained a hard-coded `/home/...` path) and
  all `__pycache__` / `.ipynb_checkpoints` / `.DS_Store` files.

---

## Troubleshooting

- **`FileNotFoundError: ./save_finetune/model_best.pth.tar`** — run `train.sh` then
  `finetune.sh` first, or drop a compatible checkpoint there.
- **`ModuleNotFoundError: bitstring`** — `pip install bitstring`.
- **CUDA out of memory** — lower the hard-coded batch sizes in `trigger_*.py` /
  `CSB_*.py` (they use 128 for evaluation; the attack itself is per-sample).
- **Newer PyTorch (`>=2.6`)** — `torch.load` defaults to `weights_only=True` and will
  reject the checkpoints; use the pinned 1.7.0, or pass `weights_only=False`.
