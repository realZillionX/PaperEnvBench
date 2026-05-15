#!/usr/bin/env bash
set -Eeuo pipefail

TASK_ID="groundingdino_phrase_grounding_minimal"
REPO_URL="https://github.com/IDEA-Research/GroundingDINO.git"
COMMIT="856dde20aee659246248e20734ef9ba5214f5e44"
CHECKPOINT_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
CHECKPOINT_SHA256="3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799"

ROOT_DIR="${PAPERENVBENCH_RUN_DIR:-$(pwd)}"
REPO_DIR="${PAPERENVBENCH_REPO_DIR:-$ROOT_DIR/repo}"
VENV_DIR="${PAPERENVBENCH_VENV_DIR:-$ROOT_DIR/venv}"
MODEL_DIR="${GROUNDINGDINO_MODEL_DIR:-$ROOT_DIR/models}"
LOG_DIR="$ROOT_DIR/logs"
CHECKPOINT_PATH="${GROUNDINGDINO_CHECKPOINT_PATH:-$MODEL_DIR/groundingdino_swint_ogc.pth}"

mkdir -p "$ROOT_DIR" "$LOG_DIR" "$MODEL_DIR" "$ROOT_DIR/artifacts"

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
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --index-url https://pypi.org/simple --upgrade pip setuptools wheel

if ! python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0" "torchvision==0.23.0"; then
  echo "CPU wheel index install failed; falling back to PyPI for torch." >&2
  python -m pip install --index-url https://pypi.org/simple "torch==2.8.0" "torchvision==0.23.0"
fi

# Use PyPI rather than the configured internal mirror for these packages: the
# gold run observed a stale internal requests wheel URL.
python -m pip install --index-url https://pypi.org/simple \
  "numpy==2.2.6" \
  "Pillow==11.3.0" \
  "opencv-python-headless==4.12.0.88" \
  "pycocotools==2.0.10" \
  "addict==2.4.0" \
  "yapf==0.43.0" \
  "timm==0.9.16" \
  "transformers==4.41.2" \
  "supervision==0.22.0" \
  "matplotlib==3.10.5" \
  "requests==2.32.5"

# GroundingDINO setup.py imports torch while building editable metadata, so
# build isolation must be disabled after the CPU torch wheel is installed.
python -m pip install --no-build-isolation --no-deps -e "$REPO_DIR"

python - <<'PY'
import importlib.metadata as metadata
import sys

import torch
from groundingdino.util.inference import preprocess_caption

print("python", sys.version.split()[0])
print("torch", torch.__version__)
for name in [
    "torchvision",
    "transformers",
    "timm",
    "supervision",
    "opencv-python-headless",
    "pycocotools",
    "requests",
    "groundingdino",
]:
    print(name, metadata.version(name))
print("caption_probe", preprocess_caption("Dog and ball"))
PY

if [ ! -s "$CHECKPOINT_PATH" ]; then
  curl -L --retry 3 --retry-delay 5 "$CHECKPOINT_URL" -o "$CHECKPOINT_PATH"
fi

echo "$CHECKPOINT_SHA256  $CHECKPOINT_PATH" | sha256sum -c -
python -m pip freeze | sort > "$ROOT_DIR/requirements_lock.txt"
{
  echo "# source_repo=https://github.com/IDEA-Research/GroundingDINO"
  echo "# commit=$COMMIT"
  echo "# checkpoint=$CHECKPOINT_PATH"
} >> "$ROOT_DIR/requirements_lock.txt"
