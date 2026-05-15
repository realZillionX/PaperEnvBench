#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/facebookresearch/ImageBind.git"
REPO_COMMIT="53680b02d7e37b19b124fa37bae4b6c98c38f5be"
RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$PWD}"
REPO_DIR="${IMAGEBIND_EMBEDDING_MINIMAL_REPO_DIR:-$RUN_ROOT/repo}"
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
"$VENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple 'torch==2.8.0' 'torchvision==0.23.0' 'torchaudio==2.8.0' 'pytorchvideo==0.1.5' 'timm==0.9.16'
"$VENV_DIR/bin/python" -m pip freeze | sort > "$LOCK_OUT"
test -e "$REPO_DIR/imagebind/models/imagebind_model.py"
test -e "$REPO_DIR/imagebind/models/__init__.py"
test -e "$REPO_DIR/imagebind/data.py"
echo "imagebind_embedding_minimal gold install route verified"
