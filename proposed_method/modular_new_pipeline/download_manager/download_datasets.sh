#!/usr/bin/env bash
# =============================================================================
# download_datasets.sh — download_manager part 2: project datasets.
# =============================================================================
# Thin launcher: finds the env's python and runs download_datasets.py.
# Every dataset the project uses is here (registry datasets + CC3M + the
# VLM-transfer eval data OK-VQA / Flickr-30K).
#
#   bash download_datasets.sh --list                    # what is available
#   bash download_datasets.sh --check                   # exist on disk?
#   bash download_datasets.sh --check --dataset imagenet1k
#   bash download_datasets.sh --dataset imagenet1k      # download one
#   bash download_datasets.sh --dataset cifar10 dtd     # ...or a few
#   bash download_datasets.sh --all                     # everything
#   bash download_datasets.sh --dataset ok-vqa          # + COCO val2014
#   bash download_datasets.sh --dataset cc3m --cc3m-max-shards 20
#   bash download_datasets.sh --start-only              # cheap download test
#   bash download_datasets.sh --dry-run                 # plan + status
#
# Names: caltech101, cifar10, cifar100, dtd, food101, flowers102,
#        imagenet1k, imagenet1k-train, cc3m, ok-vqa, flickr30k
#
# Gated sources (ILSVRC/imagenet-1k) use HF_TOKEN=... from
# download_manager/.env (see .env.example).  Downloads are resumable.
# =============================================================================
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ── pick a python: $PY override > the vlm_attack env > python3 ──
if [ -z "${PY:-}" ]; then
    for cand in "${HOME}/miniconda3/envs/vlm_attack/bin/python" \
                "${HOME}/miniforge3/envs/vlm_attack/bin/python" \
                /opt/conda/envs/vlm_attack/bin/python \
                /workspace/envs/vlm_attack/bin/python; do
        if [ -x "$cand" ]; then PY="$cand"; break; fi
    done
    PY="${PY:-python3}"
fi

exec "$PY" "$HERE/download_datasets.py" "$@"
