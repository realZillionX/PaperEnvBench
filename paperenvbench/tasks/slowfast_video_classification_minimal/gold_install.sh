#!/usr/bin/env bash
set -Eeuo pipefail

TASK_ID="slowfast_video_classification_minimal"
REPO_URL="https://github.com/facebookresearch/SlowFast.git"
COMMIT="287ec0076846560f44a9327e931a5a2360240533"
ROOT_DIR="${PAPERENVBENCH_RUN_DIR:-$(pwd)}"
REPO_DIR="${PAPERENVBENCH_REPO_DIR:-$ROOT_DIR/repo}"
VENV_DIR="${PAPERENVBENCH_VENV_DIR:-$ROOT_DIR/venv}"
LOG_DIR="$ROOT_DIR/logs"

mkdir -p "$LOG_DIR" "$ROOT_DIR/artifacts"

echo "task_id=$TASK_ID"
echo "root_dir=$ROOT_DIR"
echo "repo_dir=$REPO_DIR"
echo "venv_dir=$VENV_DIR"
echo "commit=$COMMIT"

if [ ! -d "$REPO_DIR/.git" ]; then
  rm -rf "$REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch --all --tags --prune
git -C "$REPO_DIR" checkout --detach "$COMMIT"
git -C "$REPO_DIR" rev-parse HEAD

python3 -m venv --clear "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

if ! "$VENV_DIR/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision; then
  echo "CPU wheel index install failed; falling back to the configured pip index." >&2
  "$VENV_DIR/bin/python" -m pip install torch torchvision
fi

"$VENV_DIR/bin/python" -m pip install \
  numpy yacs pyyaml fvcore iopath simplejson termcolor tqdm psutil pandas \
  scikit-learn scipy pillow opencv-python-headless fairscale pytorchvideo av

# The SII mirror returned a 404 for fonttools 4.63.0 during matplotlib install
# in this run, so install matplotlib from public PyPI as in other gold packages.
"$VENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple matplotlib

set +e
PYTHONPATH="$REPO_DIR" "$VENV_DIR/bin/python" - <<'PY' > "$LOG_DIR/native_import_probe.log" 2>&1
from slowfast.models import build_model

print(build_model)
PY
native_import_status=$?
set -e
echo "native_import_status=$native_import_status"

PYTHONPATH="$REPO_DIR" "$VENV_DIR/bin/python" - <<'PY' > "$LOG_DIR/dependency_probe.log" 2>&1
import importlib.metadata as metadata
import sys

import torch
import torchvision
import pytorchvideo

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("pytorchvideo", metadata.version("pytorchvideo"))
print("slowfast_repo_import_probe=see native_import_probe.log")
PY

"$VENV_DIR/bin/python" -m pip freeze | sort > "$ROOT_DIR/requirements_lock.txt"
{
  echo "# Source repository is used via PYTHONPATH, not installed as a wheel."
  echo "# repo=$REPO_URL"
  echo "# commit=$COMMIT"
  echo "# native_import_status=$native_import_status"
} >> "$ROOT_DIR/requirements_lock.txt"
