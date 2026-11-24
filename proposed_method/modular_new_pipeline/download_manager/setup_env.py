#!/usr/bin/env python3
"""
setup_env.py — build the `vlm_attack` conda env (download_manager part 1).
=========================================================================
Run it through the launcher (which only finds a python for you):

    bash setup_env.sh                     # normal install
    bash setup_env.sh --dry-run           # print plan (env, GPU, channel) only
    bash setup_env.sh --env-prefix /workspace/envs/vlm_attack
    bash setup_env.sh --torch auto        # newest torch for the chosen channel
    bash setup_env.sh --cuda cu126        # force a channel (no fallback)
    bash setup_env.sh --fresh             # delete + recreate the env
    bash setup_env.sh --with-optional     # also install optional extras
    bash setup_env.sh --no-vlm            # skip the VLM-transfer extras
    bash setup_env.sh --smoke             # download ViT-B/32 and encode a batch

    python setup_env.py --verify          # check the current env only
    python setup_env.py --verify --smoke  # + CLIP GPU round-trip

This is a full Python port of the old setup_vastai.sh — “install all libs we
need, just like we had them”:

  1. finds conda (installs Miniforge if absent)
  2. creates the `vlm_attack` env (python 3.10)
  3. picks a CUDA channel from `nvidia-smi` (driver CUDA + compute capability)
     and installs torch/torchvision from download.pytorch.org
  4. runs a real CUDA matmul smoke test; on failure uninstalls the stack and
     retries the next channel (older CUDA)
  5. installs requirements-vastai.txt + OpenAI CLIP (pinned commit)
  6. installs the VLM-transfer extras (open_flamingo, open_clip, EVA-CLIP
     source on sys.path) unless --no-vlm is given
  7. optionally installs requirements-vastai-optional.txt
  8. verifies the env (the old verify_env.py — this file's --verify mode)

CUDA decision (CUDA_CHANNEL=auto)
  sm_120 / RTX 50xx / B100-B300  -> cu128 (torch>=2.7, 2.6 has no kernels)
  driver >= 12.4                 -> cu124 (same as the dev env) then 12.1/11.8
  driver >= 12.1                 -> cu121 then cu118
  driver >= 11.8                 -> cu118
  any failure                    -> next lower channel, real GPU test decides
  everything unreachable         -> PyPI's default torch build (cu124 for
                                    2.6.0 on x86_64), the dev env's wheel

Note: download.pytorch.org links the nvidia-* runtime wheels at
pypi.nvidia.com. If that host is unreachable, the installer falls back to the
PyPI build (the same cu124 wheel the dev env uses) before giving up.

All flags also work as environment variables, e.g.
  ENV_PREFIX=/workspace/envs/vlm_attack bash setup_env.sh
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata as md
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request

import common

HERE = os.path.dirname(os.path.abspath(__file__))        # download_manager/
PKG_DIR = common.PKG_DIR                                 # modular_new_pipeline/

REQ_CORE = os.path.join(HERE, "requirements-vastai.txt")
REQ_VLM = os.path.join(HERE, "requirements-vlm-transfer.txt")
REQ_OPT = os.path.join(HERE, "requirements-vastai-optional.txt")

PY_TARGET = (3, 10)


# ══════════════════════════════════════════════════════════════════════
#  Small utilities
# ══════════════════════════════════════════════════════════════════════
def have(prog):
    return shutil.which(prog) is not None


def run(cmd, check=True, capture=False, env=None):
    """subprocess.run wrapper (lists or strings)."""
    if isinstance(cmd, str):
        cmd = [cmd]
    return subprocess.run(cmd, check=check, capture_output=capture,
                          text=True, env=env)


def run_logged(cmd, log_path):
    """Run `cmd`, streaming output to the console AND to `log_path`."""
    with open(log_path, "a") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
    return proc.returncode


def download(url, dest):
    """Small-file download with retries (Miniforge installer)."""
    for attempt in (1, 2, 3):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "clip-flip-setup/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, \
                    open(dest, "wb") as out:
                shutil.copyfileobj(r, out)
            return
        except Exception as e:                            # noqa: BLE001
            if attempt == 3:
                raise
            common.warn(f"download attempt {attempt} failed ({e}); retrying")
            time.sleep(5 * attempt)


def version_ge(a, b):
    """a >= b (float compare)."""
    try:
        return float(a) >= float(b)
    except (TypeError, ValueError):
        return False


# ══════════════════════════════════════════════════════════════════════
#  Step 1 — prerequisites + conda
# ══════════════════════════════════════════════════════════════════════
def ensure_prereqs():
    missing = []
    if not have("git"):
        missing.append("git")
    if not (have("curl") or have("wget")):
        missing.append("curl")
    if missing:
        if have("apt-get") and os.geteuid() == 0:
            common.warn(f"installing system packages: {' '.join(missing)}")
            run(["apt-get", "update", "-qq"], check=False)
            run(["apt-get", "install", "-y", "-qq"] + missing,
                env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"})
        else:
            common.die("missing system packages: " + " ".join(missing)
                       + " — install them first (apt-get install ...)")
    common.ok("git / curl present")

    free_gb = shutil.disk_usage(HERE).free / 1024**3
    if free_gb < 8:
        common.warn(f"only {free_gb:.0f} GB free under {HERE} — env + HF "
                    "cache need ~10-15 GB")
    else:
        common.ok(f"{free_gb:.0f} GB free under {HERE}")


def detect_conda():
    """Return the conda base dir, or None."""
    if have("conda"):
        try:
            base = run(["conda", "info", "--base"], capture=True).stdout.strip()
            if base:
                return base
        except Exception:                                 # noqa: BLE001
            pass
    for d in ("/opt/conda", "/opt/miniconda3", "/opt/miniforge3",
              "/opt/mambaforge", os.path.expanduser("~/miniconda3"),
              os.path.expanduser("~/miniforge3"),
              os.path.expanduser("~/mambaforge"),
              "/usr/local/miniconda3", "/usr/local/miniforge3"):
        if os.path.isfile(os.path.join(d, "bin", "conda")):
            return d
    return None


def ensure_conda(conda_dir):
    base = detect_conda()
    if base:
        try:
            ver = run([os.path.join(base, "bin", "conda"), "--version"],
                      capture=True).stdout.strip() or "conda"
        except Exception:                                 # noqa: BLE001
            ver = "conda"
        common.ok(f"conda found: {base} ({ver})")
        return base

    import platform
    machine = platform.machine()
    arch = {"x86_64": "Linux-x86_64", "amd64": "Linux-x86_64",
            "aarch64": "Linux-aarch64", "arm64": "Linux-aarch64"}.get(machine)
    if not arch:
        common.die(f"unsupported architecture: {machine}")
    url = ("https://github.com/conda-forge/miniforge/releases/latest/download/"
           f"Miniforge3-{arch}.sh")
    installer = f"/tmp/Miniforge3-{arch}.sh"

    common.warn(f"conda not found — installing Miniforge into {conda_dir}")
    download(url, installer)
    run(["bash", installer, "-b", "-p", conda_dir])
    common.ok(f"Miniforge installed: {conda_dir}")
    return conda_dir


# ══════════════════════════════════════════════════════════════════════
#  Step 2 — GPU / driver detection and the CUDA channel plan
# ══════════════════════════════════════════════════════════════════════
def detect_gpu():
    info = {"driver_cuda": "", "driver_ver": "", "gpu_name": "",
            "compute_cap": "", "blackwell": False}
    if not have("nvidia-smi"):
        common.warn("nvidia-smi not available — GPU checks will be skipped")
        return info
    try:
        smi = run(["nvidia-smi"], capture=True, check=False).stdout or ""
        if "CUDA Version" not in smi:
            common.warn("nvidia-smi failed — GPU checks will be skipped")
            return info
    except Exception:                                     # noqa: BLE001
        common.warn("nvidia-smi failed — GPU checks will be skipped")
        return info

    m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", smi)
    info["driver_cuda"] = m.group(1) if m else ""

    def q(field):
        try:
            return run(["nvidia-smi", f"--query-gpu={field}",
                        "--format=csv,noheader"], capture=True,
                       check=False).stdout.splitlines()[0].strip()
        except Exception:                                 # noqa: BLE001
            return ""

    info["gpu_name"] = q("name")
    info["compute_cap"] = q("compute_cap").replace(" ", "")
    info["driver_ver"] = q("driver_version")

    if info["compute_cap"]:
        try:
            info["blackwell"] = int(info["compute_cap"].split(".")[0]) >= 10
        except ValueError:
            pass
    elif re.search(r"B100|B200|B300|GB200|GB300|RTX 50\d{2}|RTX PRO 6\d{3}",
                   info["gpu_name"], re.I):
        info["blackwell"] = True

    common.ok(f"GPU:        {info['gpu_name'] or '<none>'}")
    common.ok(f"driver:     {info['driver_ver'] or '<unknown>'}  "
              f"(CUDA {info['driver_cuda'] or '<unknown>'})")
    if info["blackwell"]:
        common.ok(f"capability: {info['compute_cap'] or '<unknown>'}  "
                  "(Blackwell — needs cu128+)")
    else:
        common.ok(f"capability: {info['compute_cap'] or '<unknown>'}")
    return info


CHANNEL_CUDA = {"cu118": "11.8", "cu121": "12.1", "cu124": "12.4",
                "cu126": "12.6", "cu128": "12.8", "cu129": "12.9",
                "cu130": "13.0"}


def build_channel_plan(cuda_channel, gpu, torch_version):
    """Return (channels, torch_version) — the order to try."""
    if cuda_channel != "auto":
        if cuda_channel not in ("cu118", "cu121", "cu124", "cu126", "cu128",
                                "cu129", "cu130", "pypi"):
            common.die(f"invalid --cuda {cuda_channel!r} "
                       "(cu118|cu121|cu124|cu126|cu128|cu129|cu130|pypi|auto)")
        return [cuda_channel], torch_version

    if gpu["blackwell"]:
        # sm_120 exists only in CUDA 12.8+ wheels; cu126 covers sm_100 at best.
        channels = ["cu130", "cu129", "cu128", "cu126", "cu124", "pypi"]
        if torch_version != "auto" and not version_ge(torch_version, "2.7"):
            common.warn(f"torch {torch_version} has no cu128 wheels / no "
                        "Blackwell kernels — using --torch auto")
            torch_version = "auto"
        return channels, torch_version

    if not gpu["driver_cuda"]:
        return ["cu124", "pypi"], torch_version

    # Newest channel the driver can run first, then older.  (cu129/cu130
    # matter: without them a CUDA-13 box falls through to `pypi`, where the
    # extra-index mix can hand back a torch that no matching torchvision
    # exists for — see the TORCH_TEST_SNIPPET torchvision gate.)
    channels = [ch for ch in ("cu130", "cu129", "cu128", "cu126", "cu124",
                              "cu121", "cu118")
                if version_ge(gpu["driver_cuda"], CHANNEL_CUDA[ch])]
    if not channels:
        common.warn(f"driver CUDA {gpu['driver_cuda']} is older than 11.8 — "
                    "even cu118 needs driver >= 450.80")
        channels = ["cu118"]
    channels.append("pypi")
    return channels, torch_version


# ══════════════════════════════════════════════════════════════════════
#  Step 3 — the conda environment
# ══════════════════════════════════════════════════════════════════════
class Env:
    """The target env: prefix + its python + pip helpers."""

    def __init__(self, prefix, py_ver, no_pip_cache=False):
        self.prefix = prefix
        self.py_ver = py_ver
        self.python = os.path.join(prefix, "bin", "python")
        self.no_pip_cache = no_pip_cache

    def py(self, *args, **kw):
        return run([self.python] + list(args), **kw)

    def python_ok(self):
        if not os.path.isfile(self.python):
            return False
        try:
            want = tuple(int(x) for x in self.py_ver.split(".")[:2])
            out = self.py("-c", "import sys; print(sys.version_info[:2])",
                          capture=True).stdout.strip()
            return tuple(json.loads(out)) == want
        except Exception:                                 # noqa: BLE001
            return False

    def pip_install(self, *spec, no_deps=False, index_url=None,
                    extra_index=None, log=None):
        args = [self.python, "-m", "pip", "install", "--no-input",
                "--disable-pip-version-check", "--retries", "3",
                "--timeout", "45"]
        if self.no_pip_cache:
            args.append("--no-cache-dir")
        if no_deps:
            args.append("--no-deps")
        if index_url:
            args += ["--index-url", index_url]
        if extra_index:
            args += ["--extra-index-url", extra_index]
        args += list(spec)
        env = {**os.environ, "PIP_ROOT_USER_ACTION": "ignore",
               "PYTHONNOUSERSITE": "1"}
        if log:
            return run_logged(args, log) == 0
        return run(args, env=env, check=False).returncode == 0


def resolve_env_prefix(env_name, conda_base):
    """Vast.ai: the conda base often lives on a small container disk while
    /workspace holds the (large) instance disk. Use /workspace when tight."""
    try:
        free_kb = shutil.disk_usage(conda_base).free // 1024
    except OSError:
        free_kb = 0
    if os.path.isdir("/workspace") and os.access("/workspace", os.W_OK) \
            and free_kb < 12 * 1024 * 1024:
        prefix = f"/workspace/envs/{env_name}"
        common.warn(f"conda base has < 12 GB free — placing the env in {prefix}")
        return prefix
    return os.path.join(conda_base, "envs", env_name)


def ensure_env(env, conda_base, fresh):
    if fresh and os.path.isdir(env.prefix):
        common.warn(f"--fresh: removing {env.prefix}")
        run(["conda", "env", "remove", "-p", env.prefix, "-y"])

    if os.path.isdir(env.prefix):
        if env.python_ok():
            common.ok(f"reusing existing env (python {env.py_ver})")
            return
        common.die(f"{env.prefix} exists but is not python {env.py_ver} — "
                   "rerun with --fresh")

    os.makedirs(os.path.dirname(env.prefix), exist_ok=True)
    print(f"  conda create -p {env.prefix} python={env.py_ver}")
    run(["conda", "create", "-y", "-p", env.prefix, f"python={env.py_ver}",
         "--override-channels", "-c", "conda-forge"])
    common.ok("env created")


# ══════════════════════════════════════════════════════════════════════
#  Step 4 — PyTorch (channel chosen by the GPU test, not by assumption)
# ══════════════════════════════════════════════════════════════════════
TORCH_TEST_SNIPPET = """
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

# torchvision is installed in the same step and MUST come from the same
# index as torch.  A mismatched pair imports torch fine but dies inside
# torchvision (\"operator torchvision::nms does not exist\"), so the gate
# has to exercise a torchvision op — torch alone proves nothing.
try:
    import torchvision
    from torchvision.ops import nms as _nms
    _b = torch.tensor([[0.0, 0.0, 10.0, 10.0]], device="cuda")
    _s = torch.tensor([1.0], device="cuda")
    _nms(_b, _s, 0.5)
    torch.cuda.synchronize()
except Exception as exc:
    print(f"  torchvision {getattr(sys.modules.get('torchvision'), '__version__', '?')} "
          f"is broken against torch {torch.__version__}: "
          f"{type(exc).__name__}: {exc}")
    print("  -> torch and torchvision must come from the SAME index "
          "(both +cu1xx or both plain).")
    sys.exit(1)
print(f"  torchvision {torchvision.__version__}, nms ok")
"""


def verify_torch_gpu(env, gpu):
    if not gpu["driver_cuda"]:
        return True                    # no GPU on this box: nothing to verify
    r = env.py("-c", TORCH_TEST_SNIPPET, check=False, capture=True)
    print(r.stdout or "")
    return r.returncode == 0


def clean_torch_stack(env):
    env.py("-m", "pip", "uninstall", "-y", "torch", "torchvision",
           "torchaudio", "triton", check=False, capture=True)
    try:
        out = env.py("-m", "pip", "list", "--format=freeze",
                     capture=True).stdout.splitlines()
    except Exception:                                     # noqa: BLE001
        return
    pkgs = [ln.split("==")[0] for ln in out
            if ln.startswith("nvidia-") or ln.startswith("triton==")]
    if pkgs:
        env.py("-m", "pip", "uninstall", "-y", *pkgs,
               check=False, capture=True)


def torch_already_ok(env, want):
    snippet = ("import sys, torch\n"
               f"want = {want!r}\n"
               "sys.exit(0 if want == 'auto' or "
               "torch.__version__.split('+')[0] == want else 1)\n")
    return env.py("-c", snippet, check=False, capture=True).returncode == 0


def network_error_in_log(log_path):
    try:
        with open(log_path, errors="replace") as f:
            text = f.read()
    except OSError:
        return False
    return bool(re.search(
        r"pypi\.nvidia\.com|Read timed out|ReadTimeoutError|"
        r"Connection (aborted|reset|refused)|"
        r"Temporary failure in name resolution|SSLError", text, re.I))


def install_torch(env, gpu, channels, torch_version):
    if torch_version == "auto":
        specs = ["torch", "torchvision"]
        label = "newest torch + torchvision"
    else:
        specs = [f"torch=={torch_version}", "torchvision"]
        label = f"torch=={torch_version} + matching torchvision"

    if torch_already_ok(env, torch_version) and verify_torch_gpu(env, gpu):
        common.ok("torch already installed and CUDA works — skipping")
        return

    log = f"/tmp/torch_install_{os.getpid()}.log"
    done = False
    network_blocked = False

    for ch in channels:
        if ch == "pypi":
            print(f"==> trying PyPI default torch build  ({label})")
            common.warn("PyPI's default build must also run this GPU — "
                        "the CUDA test decides")
            ok = env.pip_install(*specs, log=log)
        else:
            print(f"==> trying torch channel: {ch}  ({label})")
            ok = env.pip_install(*specs,
                                 index_url=f"https://download.pytorch.org/whl/{ch}",
                                 extra_index="https://pypi.org/simple",
                                 log=log)

        if ok and verify_torch_gpu(env, gpu):
            common.ok(f"torch installed from {ch}")
            done = True
            break

        if ch != "pypi" and network_error_in_log(log):
            common.warn("this machine cannot reach the PyTorch index's "
                        "NVIDIA download host")
            network_blocked = True
            clean_torch_stack(env)
            break

        common.warn(f"channel {ch} did not work — cleaning up and trying "
                    "the next one")
        clean_torch_stack(env)

    # Network-blocked: the CUDA index is unusable, but PyPI carries the same
    # cu124 build the dev env uses (its nvidia-* deps live on PyPI itself).
    if not done and network_blocked:
        print(f"==> falling back to PyPI default torch build  ({label})")
        if env.pip_install(*specs, log=log) and verify_torch_gpu(env, gpu):
            common.ok("torch installed from PyPI")
            done = True

    if not done:
        common.die("no CUDA channel worked on this machine (driver CUDA "
                   f"{gpu['driver_cuda'] or '?'}). Check nvidia-smi.")


# ══════════════════════════════════════════════════════════════════════
#  Step 5-7 — project requirements, VLM extras, optional extras
# ══════════════════════════════════════════════════════════════════════
def install_requirements(env, clip_commit):
    if not env.pip_install("-r", REQ_CORE):
        common.die("requirements-vastai.txt failed to install")
    common.ok("requirements-vastai.txt installed")

    print(f"==> installing OpenAI CLIP @ {clip_commit[:12]}")
    if not env.pip_install(f"git+https://github.com/openai/CLIP.git@{clip_commit}"):
        common.die("OpenAI CLIP install failed")
    common.ok("openai-clip installed")


def install_eva_clip_source(env, repo, dest=None):
    """eva_clip is not on PyPI and upstream ships no setup.py: clone
    baaivision/EVA and expose EVA-CLIP/rei through a site-packages .pth."""
    if env.py("-c", "import eva_clip", check=False,
              capture=True).returncode == 0:
        common.ok("eva_clip already importable — skipping the source checkout")
        return

    src = dest or os.environ.get("EVA_CLIP_DIR",
                                 os.path.join(env.prefix, "eva-clip-src"))
    if os.path.isdir(src) and not os.path.isdir(os.path.join(src, ".git")) \
            and os.listdir(src):
        common.warn(f"{src} exists but is not an EVA checkout — skipping "
                    "eva_clip")
        return
    if not os.path.isdir(os.path.join(src, ".git")):
        print(f"==> cloning EVA-CLIP source into {src} (~45 MB)")
        r = run(["git", "clone", "--depth", "1", repo, src], check=False,
                capture=True)
        if r.returncode != 0:
            common.warn("git clone failed — BLIP-2 will use the open_clip "
                        "fallback")
            return

    pkg = os.path.join(src, "EVA-CLIP", "rei")
    if not os.path.isdir(os.path.join(pkg, "eva_clip")):
        common.warn(f"unexpected EVA layout (expected EVA-CLIP/rei/eva_clip) "
                    "— skipping")
        return

    # The upstream repo has no setup.py, so register it with a .pth entry
    # instead of pip.  Removing the env removes the checkout too.
    site = env.py("-c", 'import sysconfig; '
                  'print(sysconfig.get_paths()["purelib"])',
                  capture=True).stdout.strip()
    with open(os.path.join(site, "eva_clip.pth"), "w") as f:
        f.write(pkg + "\n")
    if env.py("-c", "import eva_clip", check=False,
              capture=True).returncode == 0:
        common.ok(f"eva_clip importable (sys.path entry: {site}/eva_clip.pth)")
    else:
        os.remove(os.path.join(site, "eva_clip.pth"))
        common.warn("eva_clip import failed — BLIP-2 will use the open_clip "
                    "fallback")


def install_vlm_extras(env, eva_clip, eva_clip_repo):
    if not env.pip_install("-r", REQ_VLM):
        common.die("requirements-vlm-transfer.txt failed to install")
    common.ok("requirements-vlm-transfer.txt installed")

    # open-flamingo 2.0.1 hard-pins torch==2.0.1 + sentencepiece==0.1.98 in
    # its metadata; --no-deps keeps the CUDA build installed above.
    if not env.pip_install("open-flamingo==2.0.1", no_deps=True):
        common.die("open-flamingo install failed")
    common.ok("open-flamingo installed (torch untouched)")

    if eva_clip:
        install_eva_clip_source(env, eva_clip_repo)


def pip_check(env):
    log = f"/tmp/pipcheck_{os.getpid()}.log"
    r = env.py("-m", "pip", "check", check=False, capture=True)
    with open(log, "w") as f:
        f.write(r.stdout or "")
    if r.returncode == 0:
        common.ok("pip dependency check clean")
        return
    # open-flamingo pins torch==2.0.1 / sentencepiece==0.1.98 in its
    # metadata; it is installed with --no-deps on purpose, so those lines
    # are expected.  Show every other conflict.
    lines = [ln for ln in (r.stdout or "").splitlines()
             if ln.strip() and not ln.startswith("open-flamingo ")]
    for ln in lines:
        print(ln)
    if lines:
        common.warn("pip check reported conflicts (see above)")
    else:
        common.ok("only open-flamingo's ignored torch/sentencepiece pins "
                  "differ")


# ══════════════════════════════════════════════════════════════════════
#  VERIFY MODE — the old verify_env.py (+ test_setup.py's path write-probe)
# ══════════════════════════════════════════════════════════════════════
# distribution name -> what the pipeline needs it for
DISTS = [
    ("numpy",           "arrays"),
    ("scipy",           "ablation scoring"),
    ("pandas",          "aggregation tables"),
    ("pillow",          "image loading"),
    ("matplotlib",      "trigger / ablation plots"),
    ("scikit-learn",    "metrics / utilities"),
    ("pyarrow",         "CC3M parquet downloader"),
    ("gdown",           "dataset downloader"),
    ("requests",        "dataset downloader"),
    ("tqdm",            "progress bars"),
    ("ftfy",            "OpenAI CLIP"),
    ("regex",           "OpenAI CLIP"),
    ("transformers",    "SigLIP adapter"),
    ("tokenizers",      "SigLIP / HF"),
    ("huggingface-hub", "SigLIP weights, CC3M"),
    ("safetensors",     "HF weights"),
    ("sentencepiece",   "SigLIP tokenizer"),
    ("protobuf",        "SigLIP tokenizer"),
    ("datasets",        "HF tooling"),
    ("PyYAML",          "config files"),
    ("torch",           "everything"),
    ("torchvision",     "datasets / transforms"),
    ("clip",            "OpenAI CLIP models (git install)"),
]

# modules that must import (dotted paths)
MODULES = [
    "numpy", "scipy", "pandas", "PIL", "matplotlib", "sklearn", "pyarrow",
    "gdown", "requests", "tqdm", "ftfy", "regex", "transformers",
    "tokenizers", "huggingface_hub", "safetensors", "sentencepiece",
    "google.protobuf", "datasets", "yaml", "torch", "torchvision", "clip",
]

# vlm_transfer/ extras (requirements-vlm-transfer.txt + open_flamingo)
VLM_DISTS = [
    ("open-flamingo",  "OpenFlamingo-3B"),
    ("open-clip-torch", "EVA-CLIP / OF vision encoder"),
    ("timm",           "OpenCLIP backbone"),
    ("einops",         "open_flamingo"),
    ("einops-exts",    "open_flamingo"),
    ("accelerate",     "HF 3-8B model loading"),
]
VLM_MODULES = ["open_flamingo", "open_clip", "timm", "einops",
               "einops_exts", "accelerate"]
VLM_OPTIONAL_MODULES = ["eva_clip"]      # BLIP-2 tower; open_clip fallback exists
VLM_PROJECT_MODULES = [
    "vlm_registry", "tower_swap", "metrics", "datasets_vlm", "vlm_adapters",
    "evaluate", "run_attack",
]

# project modules that must import (all phases + CLIs)
PROJECT_MODULES = [
    "config", "utils", "adapters", "models", "data", "target", "trigger",
    "quantizer", "ablation", "weights", "hybrid_ranking", "search",
    "evaluation", "results", "visualization", "analysis", "pipeline",
    "experiment", "main", "aggregate", "linear_probe", "defence_finetune",
    "make_class_texts",
]

CLIP_MODELS_REQUIRED = ["ViT-B/16", "ViT-L/14", "ViT-L/14@336px", "ViT-B/32"]

FAIL = []
WARN = []


def _ok(msg):
    common.ok(msg)


def _warn(msg):
    WARN.append(msg)
    common.warn(msg)


def _fail(msg):
    FAIL.append(msg)
    common.fail(msg)


def check_python():
    common.head("Python")
    v = sys.version_info
    msg = f"{v.major}.{v.minor}.{v.micro} ({sys.executable})"
    if (v.major, v.minor) == PY_TARGET:
        _ok(msg)
    else:
        _fail(f"{msg} — the env was built for python "
              f"{PY_TARGET[0]}.{PY_TARGET[1]}")


def check_dists():
    common.head("Packages")
    for dist, why in DISTS:
        try:
            ver = md.version(dist)
            _ok(f"{dist:<16} {ver:<14} ({why})")
        except md.PackageNotFoundError:
            _fail(f"{dist:<16} NOT INSTALLED ({why})")
        except Exception as exc:                          # pragma: no cover
            _fail(f"{dist:<16} version lookup failed: {exc}")


def check_imports():
    common.head("Imports")
    for mod in MODULES:
        try:
            importlib.import_module(mod)
            _ok(mod)
        except Exception as exc:
            _fail(f"{mod}: {type(exc).__name__}: {exc}")

    for path in (PKG_DIR, HERE):
        if path not in sys.path:
            sys.path.insert(0, path)
    for mod in PROJECT_MODULES:
        try:
            importlib.import_module(mod)
            _ok(f"project: {mod}")
        except Exception as exc:
            _fail(f"project: {mod}: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)
    for mod in ("common", "download_datasets"):
        try:
            importlib.import_module(mod)
            _ok(f"download_manager: {mod}")
        except Exception as exc:
            _fail(f"download_manager: {mod}: {type(exc).__name__}: {exc}")


def check_torch():
    common.head("PyTorch / CUDA")
    try:
        import torch
    except Exception as exc:
        _fail(f"cannot import torch: {exc}")
        return

    _ok(f"torch {torch.__version__}  (built for CUDA {torch.version.cuda})")
    try:
        _ok(f"cudnn {torch.backends.cudnn.version()}")
    except Exception:
        _warn("cudnn version unavailable")

    if not torch.cuda.is_available():
        _warn("torch.cuda.is_available() is False — CPU-only run, or a "
              "driver/wheel mismatch")
        return

    try:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        _ok(f"device: {name}  capability {cap[0]}.{cap[1]}  {total:.1f} GiB")
        arch = getattr(torch.cuda, "get_arch_list", lambda: [])()
        if arch:
            want = f"sm_{cap[0]}{cap[1]}"
            if want in arch:
                _ok(f"wheel includes kernels for {want}")
            else:
                _warn(f"wheel arch list {arch} does not list {want} "
                      "(a kernel launch test follows)")
    except Exception as exc:
        _fail(f"device query failed: {exc}")
        return

    try:
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).mean().item()
        torch.cuda.synchronize()
        _ok(f"GPU matmul works (checksum {y:.4f})")
    except Exception as exc:
        _fail(f"GPU kernel launch failed: {type(exc).__name__}: {exc} "
              "— this wheel has no kernels for this GPU (choose another "
              "CUDA channel)")


def check_clip():
    common.head("OpenAI CLIP")
    try:
        import clip
    except Exception as exc:
        _fail(f"import clip failed: {exc}")
        return
    try:
        available = clip.available_models()
        _ok(f"clip.available_models(): {len(available)} models")
        missing = [m for m in CLIP_MODELS_REQUIRED if m not in available]
        if missing:
            _fail(f"missing registry models: {missing}")
        else:
            _ok("all registry models available: "
                + ", ".join(CLIP_MODELS_REQUIRED))
    except Exception as exc:
        _fail(f"clip.available_models() failed: {exc}")


def check_vlm(required):
    common.head("VLM transfer (vlm_transfer/)")
    state = _fail if required else _warn

    for dist, why in VLM_DISTS:
        try:
            ver = md.version(dist)
            _ok(f"{dist:<16} {ver:<12} ({why})")
        except md.PackageNotFoundError:
            state(f"{dist:<16} NOT INSTALLED ({why})")
        except Exception as exc:                          # pragma: no cover
            state(f"{dist:<16} version lookup failed: {exc}")

    for mod in VLM_MODULES:
        try:
            importlib.import_module(mod)
            _ok(mod)
        except Exception as exc:
            state(f"{mod}: {type(exc).__name__}: {exc}")

    for mod in VLM_OPTIONAL_MODULES:
        try:
            # upstream eva_clip prints "Please 'pip install xformers'" at
            # import time (optional accelerators) — keep the report clean
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                importlib.import_module(mod)
            _ok(f"{mod}  (BLIP-2 official EVA-CLIP)")
        except Exception as exc:
            _warn(f"{mod} not importable ({type(exc).__name__}) — BLIP-2 "
                  "will use the open_clip fallback if it is installed")

    vlm_dir = os.path.join(PKG_DIR, "vlm_transfer")
    if not os.path.isdir(vlm_dir):
        state(f"vlm_transfer/ not found at {vlm_dir}")
        return
    if vlm_dir not in sys.path:
        sys.path.insert(0, vlm_dir)
    for mod in VLM_PROJECT_MODULES:
        try:
            importlib.import_module(mod)
            _ok(f"vlm_transfer: {mod}")
        except Exception as exc:
            state(f"vlm_transfer: {mod}: {type(exc).__name__}: {exc}")


def check_registry():
    common.head("Project registry / data layout")
    try:
        reg = common.load_registry()
    except Exception as exc:
        _fail(f"cannot load config.json: {exc}")
        return

    models = reg.get("models", {})
    datasets = reg.get("datasets", {})
    _ok(f"{len(models)} models: {', '.join(models)}")

    for name, entry in datasets.items():
        ctd = entry.get("class_text_dir")
        if ctd:
            path = ctd if os.path.isabs(ctd) else os.path.join(PKG_DIR, ctd)
            fpath = os.path.join(path, "classes.txt")
            if os.path.isfile(fpath):
                with open(fpath) as fh:
                    n = sum(1 for ln in fh if ln.strip())
                _ok(f"{name}: classes.txt ({n} names)")
            else:
                _fail(f"{name}: missing class_texts at {fpath}")

        root = common.resolve_path(entry.get("root", ""))
        if os.path.isdir(root):
            _ok(f"{name}: data root exists in {root}")
        else:
            _warn(f"{name}: data root NOT found in {root} — run "
                  "download_manager/download_datasets.py --check")

    # output paths write-probe (the old test_setup.py check_paths)
    paths = reg.get("paths", {})
    for key in ("output_root", "lp_head_root"):
        if key not in paths:
            _fail(f"{key}: not in registry")
            continue
        p = common.resolve_path(paths[key])
        try:
            os.makedirs(p, exist_ok=True)
            probe = os.path.join(p, ".write_probe")
            with open(probe, "w") as fh:
                fh.write("x")
            os.remove(probe)
            _ok(f"{key}: writable at {p}")
        except Exception as exc:
            _fail(f"{key}: {p}  ({type(exc).__name__}: {exc})")


def check_smoke():
    common.head("GPU smoke test (CLIP ViT-B/32 round-trip)")
    try:
        import clip
        import torch
        from PIL import Image
    except Exception as exc:
        _fail(f"smoke imports failed: {exc}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()
        img = Image.new("RGB", (224, 224), (128, 128, 128))
        x = preprocess(img).unsqueeze(0).to(device)
        text = clip.tokenize(["a photo of a cat"]).to(device)
        with torch.no_grad():
            emb = model.encode_image(x)
            temb = model.encode_text(text)
        _ok(f"encode_image {tuple(emb.shape)} on {device}")
        _ok(f"encode_text  {tuple(temb.shape)} on {device}")
        cos = torch.nn.functional.cosine_similarity(
            emb.float(), temb.float()).item()
        _ok(f"image/text cosine {cos:+.4f}")
    except Exception as exc:
        _fail(f"CLIP round-trip failed: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=2)


def run_verify(smoke=False, require_vlm=False):
    """The full environment check (exit 0 = the pipeline can run)."""
    common.load_env()
    FAIL.clear()
    WARN.clear()

    common.banner("vlm_attack environment check")
    print(f"  project: {PKG_DIR}")

    check_python()
    check_dists()
    check_imports()
    check_torch()
    check_clip()
    check_vlm(require_vlm)
    check_registry()
    if smoke:
        check_smoke()

    print()
    if FAIL:
        print(common._color(
            f"RESULT: {len(FAIL)} failure(s), {len(WARN)} warning(s)", "31"))
        for item in FAIL:
            print(f"  ✗ {item}")
        print("\nFix the failures above, then re-run: "
              "python setup_env.py --verify")
        return 1

    print(common._color(
        "RESULT: all required checks passed"
        + (f", {len(WARN)} warning(s)" if WARN else ""), "32"))
    if WARN:
        print("Warnings do not block the pipeline (usually missing data — "
              "see download_manager/download_datasets.py --check).")
    return 0


# ══════════════════════════════════════════════════════════════════════
#  INSTALL MODE
# ══════════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Build the `vlm_attack` conda env (download_manager "
                    "part 1).  Run via setup_env.sh.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-name", default=os.environ.get("ENV_NAME",
                                                         "vlm_attack"))
    ap.add_argument("--python", default=os.environ.get("PY_VER", "3.10"))
    ap.add_argument("--torch", default=os.environ.get("TORCH_VERSION", "2.6.0"),
                    metavar="VERSION|auto",
                    help="torch version, auto = newest (default: 2.6.0)")
    ap.add_argument("--cuda", default=os.environ.get("CUDA_CHANNEL", "auto"),
                    metavar="CHANNEL|auto",
                    help="cu118|cu121|cu124|cu126|cu128|auto")
    ap.add_argument("--env-prefix", default=os.environ.get("ENV_PREFIX", ""),
                    help="full env path (default: <conda base>/envs/<name>)")
    ap.add_argument("--conda-dir",
                    default=os.environ.get("CONDA_DIR",
                                           os.path.expanduser("~/miniforge3")),
                    help="where to install Miniforge if missing")
    ap.add_argument("--clip-commit",
                    default=os.environ.get(
                        "CLIP_COMMIT",
                        "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"))
    ap.add_argument("--fresh", action="store_true",
                    help="delete and recreate the env")
    ap.add_argument("--with-optional", action="store_true",
                    help="install optional extras")
    ap.add_argument("--no-vlm", "--without-vlm", dest="no_vlm",
                    action="store_true",
                    help="skip the VLM-transfer extras")
    ap.add_argument("--no-eva-clip", action="store_true",
                    help="skip the EVA-CLIP source checkout")
    ap.add_argument("--smoke", action="store_true",
                    help="run the CLIP ViT-B/32 GPU smoke test")
    ap.add_argument("--no-pip-cache", action="store_true",
                    help="pass --no-cache-dir to pip")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, install nothing")
    ap.add_argument("--verify", action="store_true",
                    help="ONLY verify the current environment (no install)")
    ap.add_argument("--require-vlm", action="store_true",
                    help="verify: fail (not warn) when the vlm_transfer/ "
                         "extras are missing")
    return ap.parse_args(argv)


def do_install(args):
    common.load_env()
    common.sanitize_proxy_env()

    common.head("Checking prerequisites")
    ensure_prereqs()
    conda_base = ensure_conda(args.conda_dir)

    env_prefix = args.env_prefix.replace("~", os.path.expanduser("~")) \
        if args.env_prefix else resolve_env_prefix(args.env_name, conda_base)

    gpu = detect_gpu()
    channels, torch_version = build_channel_plan(args.cuda, gpu, args.torch)

    torch_specs = ("newest torch + torchvision" if torch_version == "auto"
                   else f"torch=={torch_version} + matching torchvision")
    print()
    print(f"==> Environment : {env_prefix}  (python {args.python})")
    print(f"==> Torch       : {torch_specs}")
    print(f"==> CUDA plan   : {' '.join(channels)}")
    if not args.no_vlm:
        print("==> VLM extras  : on (open_flamingo, open_clip, EVA-CLIP)")
    else:
        print("==> VLM extras  : off (--no-vlm)")
    print(f"==> Project dir : {PKG_DIR}")

    if args.dry_run:
        print("\nDry run — nothing installed.")
        return 0

    common.head("Creating conda environment")
    env = Env(env_prefix, args.python, no_pip_cache=args.no_pip_cache)
    ensure_env(env, conda_base, args.fresh)
    os.environ["PYTHONNOUSERSITE"] = "1"
    common.ok(f"python: {env.py('-V', capture=True).stdout.strip()}")
    print("==> updating pip / setuptools / wheel")
    env.pip_install("--upgrade", "pip", "setuptools", "wheel")

    common.head("Installing PyTorch (CUDA channel from the server driver)")
    install_torch(env, gpu, channels, torch_version)

    common.head("Installing project requirements")
    install_requirements(env, args.clip_commit)

    if not args.no_vlm:
        common.head("Installing VLM-transfer extras")
        install_vlm_extras(env, not args.no_eva_clip,
                           os.environ.get(
                               "EVA_CLIP_REPO",
                               "https://github.com/baaivision/EVA.git"))

    if args.with_optional:
        common.head("Installing optional extras")
        if not env.pip_install("-r", REQ_OPT):
            common.die("requirements-vastai-optional.txt failed to install")
        common.ok("requirements-vastai-optional.txt installed")

    common.head("Verifying the environment")
    pip_check(env)
    verify_cmd = [env.python, os.path.abspath(__file__), "--verify"]
    if args.smoke:
        verify_cmd.append("--smoke")
    if not args.no_vlm:
        verify_cmd.append("--require-vlm")
    r = run(verify_cmd, check=False)
    if r.returncode != 0:
        common.die("environment verification failed")

    print("\n" + common._color(
        "=" * 60 + "\n  vlm_attack environment ready\n" + "=" * 60, "32"))
    print(f"""
Activate it:

    source "{conda_base}/etc/profile.d/conda.sh"
    conda activate "{env_prefix}"

Then, from {PKG_DIR}:

    bash download_manager/download_datasets.sh --check      # data status
    bash download_manager/download_vlm_models.sh --check    # model status
    python experiment.py --help

First experiment:

    python experiment.py --model ViT-B/16 --dataset Caltech101 --seed 42

Notes
  • Data must exist at the registry paths (../data by default). Use
    download_manager/download_datasets.py, or upload the datasets.
  • VLM data lands in ../data/vlm/{{ok-vqa,flickr30k}}
    (download_manager/download_datasets.py --dataset ok-vqa flickr30k).
  • Hugging Face downloads (SigLIP, LLaVA, BLIP-2, OF-3B) honour HF_HOME /
    HF_TOKEN from download_manager/.env; the 3-8B checkpoints need
    ~10-15 GB each of HF cache.
  • BLIP-2's attack tower: the official eva_clip package names it
    EVA01-CLIP-g-14, not EVA_CLIP_g_14 — run with
    EVA_CLIP_NAME=EVA01-CLIP-g-14 (and optionally
    EVA_CLIP_PRETRAINED=EVA01-CLIP-g-14).
  • Re-running this script is safe: an existing env is reused, torch is
    reinstalled only when its CUDA test fails.
""")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.verify:
        return run_verify(smoke=args.smoke, require_vlm=args.require_vlm)
    return do_install(args)


if __name__ == "__main__":
    sys.exit(main())
