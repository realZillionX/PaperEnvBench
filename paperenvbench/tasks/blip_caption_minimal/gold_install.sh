#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/salesforce/BLIP.git"
REPO_COMMIT="056a169437371659074aa2732649d5de3bffb4a8"
RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$PWD}"
REPO_DIR="${BLIP_CAPTION_MINIMAL_REPO_DIR:-$RUN_ROOT/repo}"
VENV_DIR="${PAPERENVBENCH_ENV_DIR:-$RUN_ROOT/venv}"
LOCK_OUT="${PAPERENVBENCH_LOCK_OUT:-$RUN_ROOT/requirements_lock.txt}"
PYTHON_BIN="${PAPERENVBENCH_PYTHON:-python3}"

mkdir -p "$RUN_ROOT"
if [ ! -d "$REPO_DIR/.git" ]; then
  rm -rf "$REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch --depth 1 origin "$REPO_COMMIT"
git -C "$REPO_DIR" checkout --detach "$REPO_COMMIT"
git -C "$REPO_DIR" rev-parse HEAD

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple 'torch==2.8.0' 'torchvision==0.23.0' 'timm==0.4.12' 'transformers==4.41.2' 'fairscale==0.4.13'
"$VENV_DIR/bin/python" -m pip freeze | sort > "$LOCK_OUT"
test -e "$REPO_DIR/models/blip.py"
test -e "$REPO_DIR/configs/caption_coco.yaml"
test -e "$REPO_DIR/predict.py"
test -e "$REPO_DIR/train_caption.py"
echo "blip_caption_minimal gold install route verified"
