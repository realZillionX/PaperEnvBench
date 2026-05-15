#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/haotian-liu/LLaVA.git"
REPO_COMMIT="c121f0432da27facab705978f83c4ada465e46fd"
RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$PWD}"
REPO_DIR="${LLAVA_SINGLE_IMAGE_MINIMAL_REPO_DIR:-$RUN_ROOT/repo}"
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
"$VENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple 'torch==2.8.0' 'torchvision==0.23.0' 'transformers==4.41.2' 'sentencepiece==0.2.0' 'Pillow==11.3.0'
"$VENV_DIR/bin/python" -m pip freeze | sort > "$LOCK_OUT"
test -e "$REPO_DIR/llava/conversation.py"
test -e "$REPO_DIR/llava/constants.py"
test -e "$REPO_DIR/llava/model/builder.py"
test -e "$REPO_DIR/llava/mm_utils.py"
echo "llava_single_image_minimal gold install route verified"
