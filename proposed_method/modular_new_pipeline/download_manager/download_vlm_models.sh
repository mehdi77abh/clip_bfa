#!/usr/bin/env bash
# =============================================================================
# download_vlm_models.sh — download_manager part 3: model checkpoints.
# =============================================================================
# Thin launcher: finds the env's python and runs download_vlm_models.py.
# Covers the attack encoders (OpenAI CLIP / SigLIP / BLIP from config.json)
# and the VLM-transfer checkpoints (LLaVA-7B, BLIP-2, OF-3B, EVA-CLIP).
#
#   bash download_vlm_models.sh --list           # catalogue + what is present
#   bash download_vlm_models.sh --check          # auth + downloaded?
#   bash download_vlm_models.sh --dry-run        # the plan + sizes
#   bash download_vlm_models.sh                  # everything
#   bash download_vlm_models.sh --models llava blip2
#   bash download_vlm_models.sh --models eva-clip --write-pth
#
# Model names: vit-b16, vit-b32, vit-l14, vit-l14-336, siglip-b16,
#              siglip-so400m, blip-itm, llava, blip2, of3b, eva-clip
#
# Gated / large checkpoints need a Hugging Face token: set HF_TOKEN=... in
# download_manager/.env (see .env.example) or run `hf auth login`.
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

exec "$PY" "$HERE/download_vlm_models.py" "$@"
