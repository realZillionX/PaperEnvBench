#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/mlfoundations/open_clip.git"
REPO_COMMIT="55794d65a14dfc547a9ed3514145dd68ccc939e9"
RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$PWD}"
REPO_DIR="${OPEN_CLIP_ZEROSHOT_MINIMAL_REPO_DIR:-$RUN_ROOT/repo}"
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
"$VENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple 'open_clip_torch==3.2.0' 'torch==2.8.0' 'torchvision==0.23.0'
"$VENV_DIR/bin/python" -m pip freeze | sort > "$LOCK_OUT"
test -e "$REPO_DIR/src/open_clip/__init__.py"
test -e "$REPO_DIR/src/open_clip/factory.py"
test -e "$REPO_DIR/src/open_clip/tokenizer.py"
echo "open_clip_zeroshot_minimal gold install route verified"
