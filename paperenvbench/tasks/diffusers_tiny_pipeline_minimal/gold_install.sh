#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/huggingface/diffusers}"
REPO_COMMIT="${REPO_COMMIT:-de5fcf6fe322dde5ea119f0491dada9945f8a649}"
VENV_DIR="${VENV_DIR:-.venv}"
REPO_DIR="${REPO_DIR:-repo}"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel setuptools
"$VENV_DIR/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"$VENV_DIR/bin/python" -m pip install accelerate transformers safetensors Pillow numpy huggingface-hub

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch --depth 1 origin "$REPO_COMMIT"
git -C "$REPO_DIR" checkout --detach "$REPO_COMMIT"
"$VENV_DIR/bin/python" -m pip install -e "$REPO_DIR"

"$VENV_DIR/bin/python" verify.py --check-only --json
