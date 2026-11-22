#!/usr/bin/env python3
"""
common.py — shared helpers for the download_manager tools.
==========================================================
Everything in this directory (setup_env.py, download_datasets.py,
download_vlm_models.py) imports this module.  Stdlib only — it must work
on a bare machine before any environment exists.

What lives here:

  * ``load_env()``       — read ``download_manager/.env`` (HF token etc.)
                           into os.environ (never overriding real env vars).
  * ``hf_token()``       — the Hugging Face token: --hf-token argument /
                           HF_TOKEN / HUGGING_FACE_HUB_TOKEN / .env /
                           the ``hf auth login`` cache.
  * ``sanitize_proxy_env()`` — drop ``socks://`` proxy vars that crash the
                           ``vlm_attack`` env's httpx (the AGENTS.md gotcha).
  * ``resolve_path()``   — the registry's '../data' path convention
                           (identical to config.resolve_path: relative
                           paths resolve against proposed_method/).
  * ``data_root()``      — the dataset destination root (DATA_ROOT env /
                           --dest / ../data).
  * output helpers (head/ok/warn/fail/die/human) — one look for all tools.

The `.env` file can be redirected with the DOWNLOAD_MANAGER_ENV variable
(used by the tests to run with an empty environment).
"""

from __future__ import annotations

import os
import sys

# ── layout ────────────────────────────────────────────────────────────
DIR = os.path.dirname(os.path.abspath(__file__))          # download_manager/
PKG_DIR = os.path.dirname(DIR)        # modular_new_pipeline/ (config.json,
                                      #   class_texts/, the pipeline modules)
PROJECT_DIR = os.path.dirname(PKG_DIR)                    # proposed_method/

ENV_FILE = os.environ.get("DOWNLOAD_MANAGER_ENV") or os.path.join(DIR, ".env")

DEFAULT_DEST = "../data"              # registry convention (config.json)


# ══════════════════════════════════════════════════════════════════════
#  .env
# ══════════════════════════════════════════════════════════════════════
def load_env(path: str = None) -> dict:
    """Parse the download_manager .env file into os.environ.

    KEY=VALUE lines; blank lines and ``#`` comments ignored; matching or
    single quotes around the value are stripped.  Variables already present
    in os.environ are NEVER overridden (the real environment wins).

    The file is ``download_manager/.env`` unless `path` or the
    DOWNLOAD_MANAGER_ENV variable points elsewhere (the tests use that to
    run with an empty environment).

    Returns the dict of variables found in the file.
    """
    path = path or os.environ.get("DOWNLOAD_MANAGER_ENV") or ENV_FILE
    found = {}
    if not os.path.isfile(path):
        return found
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            if not key:
                continue
            found[key] = val
            if val and key not in os.environ:
                os.environ[key] = val
    return found


# ══════════════════════════════════════════════════════════════════════
#  Hugging Face token / proxy hygiene
# ══════════════════════════════════════════════════════════════════════
_TOKEN = [None]           # [resolved] — None unknown, "" none found


def hf_token(override: str = None) -> str | None:
    """HF token for gated repos: explicit override / HF_TOKEN /
    HUGGING_FACE_HUB_TOKEN / .env / huggingface_hub cached login.
    None when unavailable."""
    if override:
        _TOKEN[0] = override
        return override
    if _TOKEN[0] is not None:
        return _TOKEN[0] or None
    load_env()
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        try:
            from huggingface_hub import get_token
            tok = get_token()
        except Exception:                                     # noqa: BLE001
            pass
    _TOKEN[0] = tok or ""
    return tok or None


def sanitize_proxy_env(verbose: bool = True) -> list:
    """Drop ``socks://`` proxy vars that crash the HF stack.

    The ``vlm_attack`` env's httpx cannot speak SOCKS and
    ``ALL_PROXY=socks://…`` makes every hub download abort (AGENTS.md env
    gotcha).  ``http``/``https`` proxies are left untouched.
    Returns the ``VAR=value`` strings that were cleared.
    """
    cleared = []
    for var in ("ALL_PROXY", "all_proxy", "SOCKS_PROXY", "socks_proxy"):
        val = os.environ.get(var)
        if val and val.strip().lower().startswith("socks"):
            os.environ.pop(var, None)
            cleared.append(f"{var}={val}")
    if cleared and verbose:
        print("  [proxy] cleared SOCKS proxy vars (the env's httpx cannot "
              "do SOCKS):")
        for c in cleared:
            print(f"          {c}")
    return cleared


# ══════════════════════════════════════════════════════════════════════
#  Paths
# ══════════════════════════════════════════════════════════════════════
def resolve_path(path: str) -> str:
    """Registry paths are relative to proposed_method/ (the '../data'
    convention of config.resolve_path) — CWD independent."""
    return path if os.path.isabs(path) else \
        os.path.normpath(os.path.join(PROJECT_DIR, path))


def data_root(dest: str = None) -> str:
    """The dataset destination root: --dest > DATA_ROOT (.env) > ../data."""
    if dest:
        return resolve_path(dest)
    return resolve_path(os.environ.get("DATA_ROOT") or DEFAULT_DEST)


def vlm_root(dest: str = None) -> str:
    """Where the VLM eval datasets live: {dest}/vlm (registry convention)."""
    return os.path.join(data_root(dest), "vlm")


def load_registry() -> dict:
    """The pipeline's config.json (models / datasets / paths registry)."""
    import json
    with open(os.path.join(PKG_DIR, "config.json")) as f:
        return json.load(f)


def class_text_count(dataset_key: str):
    """Number of names in class_texts/{key}/classes.txt, or None."""
    path = os.path.join(PKG_DIR, "class_texts", dataset_key, "classes.txt")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return sum(1 for ln in f if ln.strip())


# ══════════════════════════════════════════════════════════════════════
#  Output helpers (one look for every tool)
# ══════════════════════════════════════════════════════════════════════
def _color(txt: str, code: str) -> str:
    return f"\033[{code}m{txt}\033[0m" if sys.stdout.isatty() else txt


def head(title: str) -> None:
    print()
    print(_color(f"── {title} " + "─" * max(0, 56 - len(title)), "1"))


def ok(msg: str) -> None:
    print(f"  {_color('✓', '32')} {msg}")


def warn(msg: str) -> None:
    print(f"  {_color('!', '33')} {msg}")


def fail(msg: str) -> None:
    print(f"  {_color('✗', '31')} {msg}")


def die(msg: str, code: int = 1):
    print(_color(f"✗ {msg}", "31"), file=sys.stderr)
    sys.exit(code)


def banner(title: str) -> None:
    line = "=" * 64
    print(line)
    print(title)
    print(line)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
