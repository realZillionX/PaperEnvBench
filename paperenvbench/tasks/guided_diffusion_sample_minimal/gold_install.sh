#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/openai/guided-diffusion.git"
REPO_COMMIT="22e0df8183507e13a7813f8d38d51b072ca1e67c"
RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$PWD}"
REPO_DIR="${GUIDED_DIFFUSION_SAMPLE_MINIMAL_REPO_DIR:-$RUN_ROOT/repo}"
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
"$VENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple 'torch==2.8.0' 'torchvision==0.23.0' 'blobfile>=1.0.5' 'tqdm>=4.66.0'
"$VENV_DIR/bin/python" -m pip freeze | sort > "$LOCK_OUT"
test -e "$REPO_DIR/scripts/classifier_sample.py"
test -e "$REPO_DIR/scripts/image_sample.py"
test -e "$REPO_DIR/scripts/super_res_sample.py"
test -e "$REPO_DIR/guided_diffusion/script_util.py"
echo "guided_diffusion_sample_minimal gold install route verified"
