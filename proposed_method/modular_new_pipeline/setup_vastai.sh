#!/usr/bin/env bash
# =============================================================================
# setup_vastai.sh — recreate the `vlm_attack` conda environment on Vast.ai
# =============================================================================
# Rebuilds the dev env (Python 3.10, torch 2.6.0) on a fresh Vast.ai instance,
# choosing the PyTorch CUDA wheel from the *server's* driver + GPU instead of
# blindly pinning the dev machine's cu124 build.
#
# What it does
#   1. finds conda (installs Miniforge if absent)
#   2. creates the `vlm_attack` env (python 3.10)
#   3. picks a CUDA channel from `nvidia-smi` (driver CUDA + compute capability)
#      and installs torch/torchvision from download.pytorch.org
#   4. runs a real CUDA matmul smoke test; on failure uninstalls the stack and
#      retries the next channel (older CUDA)
#   5. installs requirements-vastai.txt + OpenAI CLIP (pinned commit)
#   6. optionally installs requirements-vastai-optional.txt
#   7. runs verify_env.py
#
# CUDA decision (CUDA_CHANNEL=auto)
#   sm_120 / RTX 50xx / B100-B300  -> cu128 (torch>=2.7, 2.6 has no kernels)
#   driver >= 12.4                 -> cu124 (same as the dev env) then 12.1/11.8
#   driver >= 12.1                 -> cu121 then cu118
#   driver >= 11.8                 -> cu118
#   any failure                    -> next lower channel, real GPU test decides
#   everything unreachable         -> PyPI's default torch build (cu124 for
#                                     2.6.0 on x86_64), the dev env's wheel
#
# Note: download.pytorch.org links the nvidia-* runtime wheels at
# pypi.nvidia.com. If that host is unreachable, the script falls back to the
# PyPI build (the same cu124 wheel the dev env uses) before giving up.
#
# Usage
#   ./setup_vastai.sh                     # normal install
#   ./setup_vastai.sh --dry-run           # print plan (env, GPU, channel) only
#   ./setup_vastai.sh --env-prefix /workspace/envs/vlm_attack
#   ./setup_vastai.sh --torch auto        # newest torch for the chosen channel
#   ./setup_vastai.sh --cuda cu126        # force a channel (no fallback)
#   ./setup_vastai.sh --fresh             # delete + recreate the env
#   ./setup_vastai.sh --with-optional     # also install optional extras
#   ./setup_vastai.sh --smoke             # download ViT-B/32 and encode a batch
#
# All flags also work as environment variables, e.g.
#   ENV_PREFIX=/workspace/envs/vlm_attack ./setup_vastai.sh
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------------------------
# Defaults (override with flags or env vars)
# ----------------------------------------------------------------------------
ENV_NAME="${ENV_NAME:-vlm_attack}"
PY_VER="${PY_VER:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"       # "auto" = newest in the channel
CUDA_CHANNEL="${CUDA_CHANNEL:-auto}"          # auto|cu118|cu121|cu124|cu126|cu128
ENV_PREFIX="${ENV_PREFIX:-}"                  # default: <conda base>/envs/<name>
CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"    # only used if conda is missing
CONDA_CHANNEL_NAME="${CONDA_CHANNEL_NAME:-conda-forge}"
CLIP_COMMIT="${CLIP_COMMIT:-d05afc436d78f1c48dc0dbf8e5980a9d471f35f6}"
WITH_OPTIONAL="${WITH_OPTIONAL:-0}"
FRESH="${FRESH:-0}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
PIP_NO_CACHE="${PIP_NO_CACHE:-0}"

# ----------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------
if [ -t 1 ] && command -v tput >/dev/null 2>&1 && [ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]; then
    C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_BLUE=$'\033[34m'
    C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'
else
    C_RESET=""; C_BOLD=""; C_BLUE=""; C_GREEN=""; C_YELLOW=""; C_RED=""
fi

STEP=0
log()  { printf '%s\n' "${C_BLUE}${C_BOLD}==>${C_RESET} $*"; }
step() { STEP=$((STEP + 1)); printf '\n%s\n' "${C_BOLD}────────────────────────────────────────────────────────────${C_RESET}"; printf '%s\n' "${C_BOLD}[$STEP] $*${C_RESET}"; }
ok()   { printf '%s\n' "  ${C_GREEN}✓${C_RESET} $*"; }
warn() { printf '%s\n' "  ${C_YELLOW}!${C_RESET} $*" >&2; }
die()  { printf '%s\n' "${C_RED}✗ $*${C_RESET}" >&2; exit 1; }

trap 'die "unexpected failure at line $LINENO: $BASH_COMMAND"' ERR

usage() {
    sed -n '2,44p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Flags (env var in parentheses):
  --env-name NAME        conda env name                         (ENV_NAME)
  --python X.Y           python version                         (PY_VER)
  --torch VERSION|auto   torch version, auto = newest          (TORCH_VERSION)
  --cuda CHANNEL|auto    cu118|cu121|cu124|cu126|cu128|auto    (CUDA_CHANNEL)
  --env-prefix PATH      full env path (default conda base)    (ENV_PREFIX)
  --conda-dir PATH       where to install Miniforge if missing (CONDA_DIR)
  --fresh                delete and recreate the env            (FRESH=1)
  --with-optional        install optional extras                (WITH_OPTIONAL=1)
  --smoke                run the CLIP ViT-B/32 GPU smoke test   (SMOKE=1)
  --no-pip-cache         pass --no-cache-dir to pip             (PIP_NO_CACHE=1)
  --dry-run              print the plan, install nothing        (DRY_RUN=1)
  -h, --help             this help
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --env-name)      ENV_NAME="$2"; shift 2 ;;
        --python)        PY_VER="$2"; shift 2 ;;
        --torch)         TORCH_VERSION="$2"; shift 2 ;;
        --cuda)          CUDA_CHANNEL="$2"; shift 2 ;;
        --env-prefix)    ENV_PREFIX="$2"; shift 2 ;;
        --conda-dir)     CONDA_DIR="$2"; shift 2 ;;
        --clip-commit)   CLIP_COMMIT="$2"; shift 2 ;;
        --fresh)         FRESH=1; shift ;;
        --with-optional) WITH_OPTIONAL=1; shift ;;
        --smoke)         SMOKE=1; shift ;;
        --no-pip-cache)  PIP_NO_CACHE=1; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               die "unknown argument: $1 (try --help)" ;;
    esac
done

# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
have() { command -v "$1" >/dev/null 2>&1; }

download() { # url dest
    if have curl; then
        curl -fL --retry 3 --connect-timeout 20 -o "$2" "$1"
    elif have wget; then
        wget -q --tries=3 -O "$2" "$1"
    else
        die "need curl or wget to download $1"
    fi
}

version_ge() { # $1 >= $2  (float compare)
    awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 >= b + 0) }'
}

# ----------------------------------------------------------------------------
# [1] Prerequisites + conda
# ----------------------------------------------------------------------------
step "Checking prerequisites"

ensure_prereqs() {
    local missing=()
    have git || missing+=(git)
    if ! have curl && ! have wget; then missing+=(curl); fi
    if [ "${#missing[@]}" -gt 0 ]; then
        if have apt-get && [ "$(id -u)" -eq 0 ]; then
            warn "installing system packages: ${missing[*]}"
            apt-get update -qq
            DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}"
        else
            die "missing system packages: ${missing[*]} — install them first (apt-get install ...)"
        fi
    fi
    ok "git / curl present"

    local avail_gb
    avail_gb="$(df -Pk "$SCRIPT_DIR" | awk 'NR==2 {printf "%d", $4 / 1048576}')"
    if [ "${avail_gb:-0}" -lt 8 ]; then
        warn "only ${avail_gb} GB free under $SCRIPT_DIR — env + HF cache need ~10-15 GB"
    else
        ok "${avail_gb} GB free under $SCRIPT_DIR"
    fi
}

detect_conda() {
    CONDA_BASE=""
    if have conda; then
        CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    fi
    if [ -z "$CONDA_BASE" ]; then
        local d
        for d in /opt/conda /opt/miniconda3 /opt/miniforge3 /opt/mambaforge \
                 "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/mambaforge" \
                 /usr/local/miniconda3 /usr/local/miniforge3; do
            if [ -x "$d/bin/conda" ]; then CONDA_BASE="$d"; break; fi
        done
    fi
    [ -n "$CONDA_BASE" ]
}

ensure_conda() {
    if detect_conda; then
        ok "conda found: $CONDA_BASE ($("$CONDA_BASE/bin/conda" --version 2>/dev/null || echo conda))"
        return
    fi

    local arch
    case "$(uname -m)" in
        x86_64|amd64)  arch="Linux-x86_64" ;;
        aarch64|arm64) arch="Linux-aarch64" ;;
        *)             die "unsupported architecture: $(uname -m)" ;;
    esac
    local url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-${arch}.sh"
    local installer="/tmp/Miniforge3-${arch}.sh"

    warn "conda not found — installing Miniforge into $CONDA_DIR"
    download "$url" "$installer"
    bash "$installer" -b -p "$CONDA_DIR"
    CONDA_BASE="$CONDA_DIR"
    ok "Miniforge installed: $CONDA_BASE"
}

ensure_prereqs
ensure_conda

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [ -z "$ENV_PREFIX" ]; then
    # Vast.ai: the conda base often lives on a small container disk while
    # /workspace holds the (large) instance disk. Use /workspace when tight.
    free_kb="$(df -Pk "$CONDA_BASE" 2>/dev/null | awk 'NR==2 {print $4}' || true)"
    if [ -d /workspace ] && [ -w /workspace ] && [ "${free_kb:-0}" -lt 12582912 ]; then
        ENV_PREFIX="/workspace/envs/$ENV_NAME"
        warn "conda base has < 12 GB free — placing the env in $ENV_PREFIX"
    else
        ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"
    fi
fi
ENV_PREFIX="${ENV_PREFIX/#\~/$HOME}"

# ----------------------------------------------------------------------------
# [2] GPU / driver detection and CUDA channel plan
# ----------------------------------------------------------------------------
step "Detecting GPU and CUDA driver"

DRIVER_CUDA=""
DRIVER_VER=""
GPU_NAME=""
COMPUTE_CAP=""
BLACKWELL=0

if have nvidia-smi && nvidia-smi >/dev/null 2>&1; then
    DRIVER_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/[[:space:]]*$//')"
    COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
    DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')"
else
    warn "nvidia-smi not available — GPU checks will be skipped"
fi

if [ -n "$COMPUTE_CAP" ]; then
    CAP_MAJOR="${COMPUTE_CAP%%.*}"
    if [ "${CAP_MAJOR:-0}" -ge 10 ] 2>/dev/null; then BLACKWELL=1; fi
elif [ -n "$GPU_NAME" ] && printf '%s' "$GPU_NAME" | grep -qiE 'B100|B200|B300|GB200|GB300|RTX 50[0-9]{2}|RTX PRO 6[0-9]{3}'; then
    BLACKWELL=1
fi

ok "GPU:        ${GPU_NAME:-<none>}"
ok "driver:     ${DRIVER_VER:-<unknown>}  (CUDA ${DRIVER_CUDA:-<unknown>})"
if [ "$BLACKWELL" = "1" ]; then
    ok "capability: ${COMPUTE_CAP:-<unknown>}  (Blackwell — needs cu128+)"
else
    ok "capability: ${COMPUTE_CAP:-<unknown>}"
fi

channel_cuda() {
    case "$1" in
        cu118) echo "11.8" ;;
        cu121) echo "12.1" ;;
        cu124) echo "12.4" ;;
        cu126) echo "12.6" ;;
        cu128) echo "12.8" ;;
        cu129) echo "12.9" ;;
        cu130) echo "13.0" ;;
        *)     echo "0" ;;
    esac
}

CHANNELS=()
build_channel_plan() {
    if [ "$CUDA_CHANNEL" != "auto" ]; then
        case "$CUDA_CHANNEL" in
            cu118|cu121|cu124|cu126|cu128|pypi) CHANNELS=("$CUDA_CHANNEL") ;;
            *) die "invalid --cuda '$CUDA_CHANNEL' (cu118|cu121|cu124|cu126|cu128|auto)" ;;
        esac
        return
    fi

    if [ "$BLACKWELL" = "1" ]; then
        # sm_120 exists only in CUDA 12.8+ wheels; cu126 covers sm_100 at best.
        CHANNELS=(cu128 cu126 cu124 pypi)
        if [ "$TORCH_VERSION" != "auto" ] && ! version_ge "$TORCH_VERSION" "2.7"; then
            warn "torch $TORCH_VERSION has no cu128 wheels / no Blackwell kernels — using --torch auto"
            TORCH_VERSION="auto"
        fi
        return
    fi

    if [ -z "$DRIVER_CUDA" ]; then
        CHANNELS=(cu124 pypi)
        return
    fi

    # Prefer the dev env's cu124, then newer, then older.
    local ch
    for ch in cu124 cu126 cu121 cu118; do
        if version_ge "$DRIVER_CUDA" "$(channel_cuda "$ch")"; then
            CHANNELS+=("$ch")
        fi
    done
    if [ "${#CHANNELS[@]}" -eq 0 ]; then
        warn "driver CUDA $DRIVER_CUDA is older than 11.8 — even cu118 needs driver >= 450.80"
        CHANNELS=(cu118)
    fi
    CHANNELS+=(pypi)
}

build_channel_plan

# cu128 wheels start at torch 2.7 — never ask for 2.6.0+cu128.
if [ "${CHANNELS[0]}" = "cu128" ] || [ "${CHANNELS[0]}" = "cu129" ]; then
    if [ "$TORCH_VERSION" != "auto" ] && ! version_ge "$TORCH_VERSION" "2.7"; then
        warn "torch $TORCH_VERSION has no ${CHANNELS[0]} wheels — using --torch auto"
        TORCH_VERSION="auto"
    fi
fi

if [ "$TORCH_VERSION" = "auto" ]; then
    TORCH_SPECS=(torch torchvision)
    TORCH_SPEC_LABEL="newest torch + torchvision"
else
    TORCH_SPECS=("torch==${TORCH_VERSION}" torchvision)
    TORCH_SPEC_LABEL="torch==${TORCH_VERSION} + matching torchvision"
fi

printf '\n'
log "Environment : ${C_BOLD}${ENV_PREFIX}${C_RESET}  (python ${PY_VER})"
log "Torch       : ${TORCH_SPEC_LABEL}"
log "CUDA plan   : ${CHANNELS[*]}"
log "Project dir : ${SCRIPT_DIR}"

if [ "$DRY_RUN" = "1" ]; then
    printf '\n%s\n' "${C_GREEN}Dry run — nothing installed.${C_RESET}"
    exit 0
fi

# ----------------------------------------------------------------------------
# [3] Conda environment
# ----------------------------------------------------------------------------
step "Creating conda environment"

env_python() { "$ENV_PREFIX/bin/python" "$@"; }

pip_install() {
    local args=(--no-input --disable-pip-version-check --retries 3 --timeout 45)
    if [ "$PIP_NO_CACHE" = "1" ]; then args+=(--no-cache-dir); fi
    PIP_ROOT_USER_ACTION=ignore env_python -m pip install "${args[@]}" "$@"
}

env_python_ok() {
    [ -x "$ENV_PREFIX/bin/python" ] || return 1
    env_python - "$PY_VER" <<'PY' >/dev/null 2>&1
import sys
want = tuple(int(x) for x in sys.argv[1].split(".")[:2])
sys.exit(0 if sys.version_info[:2] == want else 1)
PY
}

if [ "$FRESH" = "1" ] && [ -d "$ENV_PREFIX" ]; then
    warn "--fresh: removing $ENV_PREFIX"
    conda env remove -p "$ENV_PREFIX" -y
fi

if [ -d "$ENV_PREFIX" ]; then
    if env_python_ok; then
        ok "reusing existing env (python $PY_VER)"
    else
        die "$ENV_PREFIX exists but is not python $PY_VER — rerun with --fresh"
    fi
else
    mkdir -p "$(dirname "$ENV_PREFIX")"
    log "conda create -p $ENV_PREFIX python=$PY_VER"
    conda create -y -p "$ENV_PREFIX" "python=$PY_VER" \
        --override-channels -c "$CONDA_CHANNEL_NAME"
    ok "env created"
fi

conda activate "$ENV_PREFIX"
export PYTHONNOUSERSITE=1
export PIP_ROOT_USER_ACTION=ignore

ok "python: $(env_python -V 2>&1)"

log "updating pip / setuptools / wheel"
pip_install --upgrade pip setuptools wheel >/dev/null

# ----------------------------------------------------------------------------
# [4] PyTorch stack (channel chosen by the GPU test, not by assumption)
# ----------------------------------------------------------------------------
step "Installing PyTorch (CUDA channel from the server driver)"

verify_torch_gpu() {
    if [ -z "$DRIVER_CUDA" ]; then
        return 0   # no GPU on this box: nothing to verify
    fi
    env_python - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print(f"  import failed: {exc}")
    sys.exit(1)

if not torch.cuda.is_available():
    print(f"  torch {torch.__version__} (built for CUDA {torch.version.cuda}) "
          "-> torch.cuda.is_available() is False")
    sys.exit(1)

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"  torch {torch.__version__} (built for CUDA {torch.version.cuda}), "
      f"device={name}, capability={cap}")
try:
    x = torch.randn(1024, 1024, device="cuda")
    _ = (x @ x).mean().item()
    torch.cuda.synchronize()
except Exception as exc:
    print(f"  CUDA kernel launch failed: {type(exc).__name__}: {exc}")
    sys.exit(1)
PY
}

clean_torch_stack() {
    env_python -m pip uninstall -y torch torchvision torchaudio triton >/dev/null 2>&1 || true
    local pkgs=()
    mapfile -t pkgs < <(
        env_python -m pip list --format=freeze 2>/dev/null \
            | grep -E '^(nvidia-|triton==)' | cut -d= -f1
    )
    if [ "${#pkgs[@]}" -gt 0 ]; then
        env_python -m pip uninstall -y "${pkgs[@]}" >/dev/null 2>&1 || true
    fi
}

torch_already_ok() {
    env_python - "$TORCH_VERSION" <<'PY' >/dev/null 2>&1
import sys, torch
want = sys.argv[1]
sys.exit(0 if want == "auto" or torch.__version__.split("+")[0] == want else 1)
PY
}

TORCH_DONE=0
TORCH_LOG="$(mktemp -t torch_install.XXXXXX.log 2>/dev/null || echo /tmp/torch_install.log)"
NETWORK_BLOCKED=0

network_error_in_log() {
    grep -qiE 'pypi\.nvidia\.com|Read timed out|ReadTimeoutError|Connection (aborted|reset|refused)|Temporary failure in name resolution|SSLError' "$TORCH_LOG" 2>/dev/null
}

install_torch_from() { # $1 = channel name or "pypi"
    local ch="$1"
    if [ "$ch" = "pypi" ]; then
        pip_install "${TORCH_SPECS[@]}" 2>&1 | tee "$TORCH_LOG"
    else
        pip_install --index-url "https://download.pytorch.org/whl/${ch}" \
                    --extra-index-url "https://pypi.org/simple" \
                    "${TORCH_SPECS[@]}" 2>&1 | tee "$TORCH_LOG"
    fi
}

if torch_already_ok && verify_torch_gpu; then
    ok "torch already installed and CUDA works — skipping"
    TORCH_DONE=1
fi

if [ "$TORCH_DONE" = "0" ]; then
    for ch in "${CHANNELS[@]}"; do
        if [ "$ch" = "pypi" ]; then
            log "trying ${C_BOLD}PyPI${C_RESET} default torch build  (${TORCH_SPEC_LABEL})"
            warn "PyPI's default build must also run this GPU — the CUDA test decides"
        else
            log "trying torch channel: ${C_BOLD}${ch}${C_RESET}  (${TORCH_SPEC_LABEL})"
        fi

        CHANNEL_OK=0
        if install_torch_from "$ch" && verify_torch_gpu; then CHANNEL_OK=1; fi

        if [ "$CHANNEL_OK" = "1" ]; then
            ok "torch installed from ${ch}"
            TORCH_DONE=1
            break
        fi

        if [ "$ch" != "pypi" ] && network_error_in_log; then
            warn "this machine cannot reach the PyTorch index's NVIDIA download host"
            NETWORK_BLOCKED=1
            clean_torch_stack
            break
        fi

        warn "channel ${ch} did not work — cleaning up and trying the next one"
        clean_torch_stack
    done

    # Network-blocked: the CUDA index is unusable, but PyPI carries the same
    # cu124 build the dev env uses (its nvidia-* deps live on PyPI itself).
    if [ "$TORCH_DONE" = "0" ] && [ "$NETWORK_BLOCKED" = "1" ]; then
        log "falling back to ${C_BOLD}PyPI${C_RESET} default torch build  (${TORCH_SPEC_LABEL})"
        CHANNEL_OK=0
        if install_torch_from pypi && verify_torch_gpu; then CHANNEL_OK=1; fi
        if [ "$CHANNEL_OK" = "1" ]; then
            ok "torch installed from PyPI"
            TORCH_DONE=1
        fi
    fi

    if [ "$TORCH_DONE" = "0" ]; then
        die "no CUDA channel worked on this machine (driver CUDA ${DRIVER_CUDA:-?}). Check nvidia-smi."
    fi
fi

# ----------------------------------------------------------------------------
# [5] Project requirements + OpenAI CLIP
# ----------------------------------------------------------------------------
step "Installing project requirements"

pip_install -r "$SCRIPT_DIR/requirements-vastai.txt"
ok "requirements-vastai.txt installed"

log "installing OpenAI CLIP @ ${CLIP_COMMIT:0:12}"
pip_install "git+https://github.com/openai/CLIP.git@${CLIP_COMMIT}"
ok "openai-clip installed"

if [ "$WITH_OPTIONAL" = "1" ]; then
    step "Installing optional extras"
    pip_install -r "$SCRIPT_DIR/requirements-vastai-optional.txt"
    ok "requirements-vastai-optional.txt installed"
fi

# ----------------------------------------------------------------------------
# [6] Verification
# ----------------------------------------------------------------------------
step "Verifying the environment"

if env_python -m pip check; then
    ok "pip dependency check clean"
else
    warn "pip check reported conflicts (see above; usually harmless)"
fi

SMOKE_FLAG=""
if [ "$SMOKE" = "1" ]; then SMOKE_FLAG="--smoke"; fi

if [ -f "$SCRIPT_DIR/verify_env.py" ]; then
    # shellcheck disable=SC2086
    env_python "$SCRIPT_DIR/verify_env.py" $SMOKE_FLAG || die "verify_env.py failed"
else
    warn "verify_env.py not found — skipping deep verification"
    env_python - <<'PY'
import torch, clip, transformers
print(f"  torch {torch.__version__}, cuda {torch.version.cuda}, "
      f"available={torch.cuda.is_available()}")
print(f"  transformers {transformers.__version__}, clip models: {len(clip.available_models())}")
PY
fi

# ----------------------------------------------------------------------------
# Done — print next steps
# ----------------------------------------------------------------------------
printf '\n%s\n' "${C_GREEN}${C_BOLD}════════════════════════════════════════════════════════════${C_RESET}"
printf '%s\n' "${C_GREEN}${C_BOLD}  vlm_attack environment ready${C_RESET}"
printf '%s\n\n' "${C_GREEN}${C_BOLD}════════════════════════════════════════════════════════════${C_RESET}"

cat <<EOF
Activate it:

    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_PREFIX"

Then, from ${SCRIPT_DIR}:

    python test_setup.py                      # registry / weights / data probe
    python verify_env.py --smoke              # optional GPU CLIP round-trip
    python experiment.py --help

First experiment:

    python experiment.py --model ViT-B/16 --dataset Caltech101 --seed 42

Notes
  • Data must exist at the registry paths (../data by default, i.e.
    $(cd "$SCRIPT_DIR/.." && pwd)/data). Use download_datasets.py, or upload
    the datasets into that tree.
  • Hugging Face downloads (SigLIP, CC3M) honour HF_HOME / HF_TOKEN.
  • Re-running this script is safe: an existing env is reused, torch is
    reinstalled only when its CUDA test fails.
EOF
