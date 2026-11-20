#!/usr/bin/env bash
# Full ProFlip non-adaptive pipeline: trigger/SNI generation, then progressive bit flips
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"

$PYTHON trigger_nonadaptive.py
$PYTHON CSB_nonadaptive.py
