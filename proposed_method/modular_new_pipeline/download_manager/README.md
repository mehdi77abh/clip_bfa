# `download_manager/` — one place to set up and fetch everything

Every install/download step of the project lives here: the `vlm_attack`
conda **environment**, the **datasets**, and the **model checkpoints**.
Each part is one Python program (all logic) + one shell launcher (finds a
python and runs it).

```text
download_manager/
├── .env / .env.example         # HF_TOKEN=... — fill in .env (git-ignored)
├── common.py                   # shared helpers (.env loading, paths, output)
├── setup_env.py + setup_env.sh             # [1] the conda environment
├── download_datasets.py + download_datasets.sh   # [2] the datasets
├── download_vlm_models.py + download_vlm_models.sh # [3] the model checkpoints
├── requirements-vastai.txt                 # env package pins (core)
├── requirements-vastai-optional.txt        # env package pins (optional extras)
└── requirements-vlm-transfer.txt           # env package pins (VLM extras)
```

## 0. Credentials — `.env`

Some datasets (ILSVRC/imagenet-1k) and checkpoints (LLaVA, OF-3B, …) are
gated on Hugging Face. Put your token in **`download_manager/.env`**:

```bash
HF_TOKEN=hf_xxxxxxxx
```

All three tools load this file automatically (`common.load_env`); real
environment variables always win over `.env`. Optional overrides:
`HF_HOME` (HF cache location), `DATA_ROOT` (dataset root, default
`../data`), `EVA_CLIP_REPO`.

## 1. Environment — `setup_env.sh`

Recreates the `vlm_attack` conda env exactly as the project needs it:
Miniforge bootstrap → python 3.10 → torch/torchvision picked by the
*server's* driver (CUDA channel autodetect + a real GPU matmul test with
fallbacks) → `requirements-vastai.txt` + OpenAI CLIP (pinned commit) →
VLM extras (`requirements-vlm-transfer.txt`, `open-flamingo --no-deps`,
EVA-CLIP source checkout + `.pth`) → optional extras → verification.

```bash
bash setup_env.sh                 # normal install (safe to re-run)
bash setup_env.sh --dry-run       # print the plan only
bash setup_env.sh --fresh         # delete + recreate the env
bash setup_env.sh --with-optional # extras: bitsandbytes, lpips, ...
bash setup_env.sh --no-vlm        # skip the VLM-transfer extras
bash setup_env.sh --smoke         # + CLIP ViT-B/32 GPU round-trip
bash setup_env.sh --verify        # check the current env only (no install)
```

`setup_env.py --verify` is the old `verify_env.py` + `test_setup.py`'s
output-path probe: python/packages/imports/CUDA/CLIP/VLM/registry checks.

## 2. Datasets — `download_datasets.sh`

One tool for every dataset, in the exact layout `data.py` / torchvision
expect (default root `../data`, i.e. `proposed_method/../data`).

```bash
bash download_datasets.sh --list                    # catalogue
bash download_datasets.sh --check                   # what exists on disk?
bash download_datasets.sh --check --dataset imagenet1k
bash download_datasets.sh --dataset imagenet1k      # download ONE
bash download_datasets.sh --dataset cifar10 dtd     # ...or several
bash download_datasets.sh --all                     # everything
bash download_datasets.sh --start-only              # cheap download-start test
bash download_datasets.sh --dry-run                 # plan + on-disk status
```

| name | size | notes |
|---|---|---|
| `caltech101` | 0.7 GB | CaltechDATA zip (gdown fallback) |
| `cifar10` / `cifar100` | 0.2 GB each | official Toronto tarballs |
| `dtd` | 0.6 GB | Oxford VGG textures (47 classes) |
| `food101` | 5 GB | HF `ethz/food101` (VGG fallback) |
| `flowers102` | 0.35 GB | Oxford VGG |
| `imagenet1k` | 7 GB | **gated** ILSVRC/imagenet-1k val (`--hf-token`) |
| `imagenet1k-train` | 128 GB | train split (LP probe data) |
| `cc3m` | 280 GB full | defence data — cap with `--cc3m-max-shards 20` |
| `ok-vqa` | 13 GB | + COCO val2014 (`--no-images` for JSONs only) |
| `flickr30k` | 4.5 GB | + Karpathy split |

Useful flags: `--force`, `--imagenet-splits val,train`,
`--imagenet-source repack` (ungated), `--food101-source vgg`,
`--caltech-source gdrive`, `--cc3m-splits/--cc3m-max-shards`,
`--no-images/--no-train` (ok-vqa), `--url KEY=URL` (mirror override).
Downloads are resumable — re-run the same command after any interruption.

`--check` verifies the registry paths end-to-end (folder structure, image
counts, `meta.bin`, the `class_texts/*/classes.txt` cross-check that
`data.py` enforces) and exits non-zero when anything is missing.

## 3. Model checkpoints — `download_vlm_models.sh`

The attack encoders from `config.json` **and** the VLM-transfer
checkpoints.

```bash
bash download_vlm_models.sh --list        # catalogue + what is present
bash download_vlm_models.sh --check       # auth + downloaded status
bash download_vlm_models.sh --dry-run     # the plan + sizes
bash download_vlm_models.sh               # everything
bash download_vlm_models.sh --models llava blip2
bash download_vlm_models.sh --models eva-clip --write-pth
```

| name | size | what it is |
|---|---|---|
| `vit-b16`, `vit-b32`, `vit-l14`, `vit-l14-336` | 0.35–0.9 GB | OpenAI CLIP attack encoders (prefetched to `~/.cache/clip`) |
| `siglip-b16`, `siglip-so400m` | 1–4 GB | SigLIP (HF snapshots) |
| `blip-itm` | 1 GB | BLIP-B/16-384px (HF snapshot) |
| `llava` | 13 GB | LLaVA-7B v1.5 (HF snapshot) |
| `blip2` | 16 GB | BLIP-2 OPT-6.7B (HF snapshot) |
| `of3b` | 7 GB | OpenFlamingo-3B `checkpoint.pt` + MPT-1B LM |
| `eva-clip` | 0.05 GB | EVA-CLIP ViT-G/14 source checkout (BLIP-2 attack tower) |

## Migration notes (what replaced what)

| old | new |
|---|---|
| `setup_vastai.sh` | `download_manager/setup_env.sh` + `setup_env.py` |
| `verify_env.py` | `setup_env.py --verify` |
| `test_setup.py` | `download_datasets.py --check` + `download_vlm_models.py --check` + `setup_env.py --verify` |
| `download_datasets.py` (top level) | `download_manager/download_datasets.py` (extended: all registry datasets + VLM eval data + `--check`) |
| `vlm_transfer/download_vlm_data.py` | `download_manager/download_datasets.py --dataset ok-vqa flickr30k` |
| `vlm_transfer/download_vlm_models.py` | `download_manager/download_vlm_models.py` (extended: attack encoders too) |
| `requirements-*.txt` (top level) | `download_manager/requirements-*.txt` |

The experiment-runner shells (`run_all.sh`, `run_lp.sh`, `run_ablation.sh`,
`run_defence_ft.sh`, `validation.sh`, `vlm_transfer/run_vlm_transfer.sh`)
are unchanged.
