#!/usr/bin/env bash
set -euo pipefail

TASK_ID="pytorchvideo_classification_minimal"
REPO_URL="https://github.com/facebookresearch/pytorchvideo.git"
COMMIT="f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0"
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
git -C "$REPO_DIR" checkout "$COMMIT"
git -C "$REPO_DIR" rev-parse HEAD

python3 -m venv --clear "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if ! python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0" "torchvision==0.23.0"; then
  echo "CPU wheel index install failed; falling back to the configured pip index." >&2
  python -m pip install "torch==2.8.0" "torchvision==0.23.0"
fi

python -m pip install -e "$REPO_DIR"

PYTHONPATH="$REPO_DIR" python - <<'PY'
import importlib.metadata as metadata
import pathlib
import sys
import torch
from pytorchvideo.models.hub import slow_r50

repo_dir = pathlib.Path(__import__("os").environ.get("PAPERENVBENCH_REPO_DIR", "repo")).resolve()
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torchvision", metadata.version("torchvision"))
print("pytorchvideo", metadata.version("pytorchvideo"))
print("av", metadata.version("av"))
print("fvcore", metadata.version("fvcore"))
print("iopath", metadata.version("iopath"))
print("repo_dir", repo_dir)
print("slow_r50_builder", slow_r50)
PY

python -m pip freeze | sort > "$ROOT_DIR/requirements_lock.txt"
{
  echo "# Source repository is installed editable from the pinned checkout."
  echo "# repo=$REPO_URL"
  echo "# commit=$COMMIT"
} >> "$ROOT_DIR/requirements_lock.txt"
