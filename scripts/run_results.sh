#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$ROOT/scripts/local_paths.sh" ]]; then source "$ROOT/scripts/local_paths.sh"; fi
: "${PYTHON:=python}"
: "${H5_DIR:?Set H5_DIR to native UniTraj validation H5 directory}"
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
"$PYTHON" evaluate.py method=mdp "h5_dir=\"$H5_DIR\""
"$PYTHON" evaluate.py method=forecast_mae "h5_dir=\"$H5_DIR\""
"$PYTHON" hard_test.py
