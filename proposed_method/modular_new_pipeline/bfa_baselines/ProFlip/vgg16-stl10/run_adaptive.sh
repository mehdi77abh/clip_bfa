#!/usr/bin/env bash
# Full ProFlip adaptive pipeline: trigger/SNI generation, then progressive bit flips
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"

$PYTHON trigger_adaptive.py
$PYTHON CSB_adaptive.py
