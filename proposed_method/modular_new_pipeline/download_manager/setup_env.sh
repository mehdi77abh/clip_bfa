#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — download_manager part 1: build the `vlm_attack` conda env.
# =============================================================================
# Thin launcher: finds a python interpreter and runs setup_env.py — ALL of
# the logic (conda bootstrap, CUDA channel autodetect + GPU smoke test,
# requirements, OpenAI CLIP, VLM extras, EVA-CLIP, verification) lives in
# setup_env.py.
#
#   bash setup_env.sh                     # normal install
#   bash setup_env.sh --dry-run           # print plan (env, GPU, channel) only
#   bash setup_env.sh --env-prefix /workspace/envs/vlm_attack
#   bash setup_env.sh --cuda cu126        # force a CUDA channel
#   bash setup_env.sh --fresh             # delete + recreate the env
#   bash setup_env.sh --with-optional     # also install optional extras
#   bash setup_env.sh --no-vlm            # skip the VLM-transfer extras
#   bash setup_env.sh --smoke             # + CLIP ViT-B/32 GPU smoke test
#   bash setup_env.sh --verify            # check the current env only
#
# Hugging Face credentials and overrides live in download_manager/.env
# (HF_TOKEN=... — see .env.example); setup_env.py loads it automatically.
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

exec "$PY" "$HERE/setup_env.py" "$@"
