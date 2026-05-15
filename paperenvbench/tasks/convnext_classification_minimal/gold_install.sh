#!/usr/bin/env bash
set -euo pipefail

TASK_ID="convnext_classification_minimal"
REPO_URL="https://github.com/facebookresearch/ConvNeXt.git"
COMMIT="048efcea897d999aed302f2639b6270aedf8d4c8"
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

python -m pip install "tensorboardX==2.6.4" "six==1.17.0"

if python -m pip install "timm==0.3.2"; then
  if ! PYTHONPATH="$REPO_DIR" python - <<'PY'
from models.convnext import ConvNeXt
print("official_timm_pin_import_ok", ConvNeXt)
PY
  then
    echo "timm==0.3.2 installed but cannot import with this torch; using timm==0.9.16." >&2
    python -m pip install --upgrade "timm==0.9.16"
  fi
else
  echo "timm==0.3.2 install failed; using timm==0.9.16." >&2
  python -m pip install "timm==0.9.16"
fi

PYTHONPATH="$REPO_DIR" python - <<'PY'
import importlib.metadata as metadata
import pathlib
import sys
import torch
from models.convnext import ConvNeXt, convnext_tiny

repo_dir = pathlib.Path(__import__("os").environ["PAPERENVBENCH_REPO_DIR"]).resolve() if "PAPERENVBENCH_REPO_DIR" in __import__("os").environ else pathlib.Path("repo").resolve()
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torchvision", metadata.version("torchvision"))
print("timm", metadata.version("timm"))
print("tensorboardX", metadata.version("tensorboardX"))
print("repo_dir", repo_dir)
print("convnext_class", ConvNeXt)
print("convnext_tiny_factory", convnext_tiny)
PY

python -m pip freeze | sort > "$ROOT_DIR/requirements_lock.txt"
{
  echo "# Source repository is used via PYTHONPATH, not installed as a wheel."
  echo "# repo=$REPO_URL"
  echo "# commit=$COMMIT"
} >> "$ROOT_DIR/requirements_lock.txt"
