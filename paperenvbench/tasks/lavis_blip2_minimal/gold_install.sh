#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/salesforce/LAVIS.git"
REPO_COMMIT="506965b9c4a18c1e565bd32acaccabe0198433f7"
RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$PWD}"
REPO_DIR="${LAVIS_BLIP2_MINIMAL_REPO_DIR:-$RUN_ROOT/repo}"
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
"$VENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple 'torch==2.8.0' 'torchvision==0.23.0' 'timm==0.4.12' 'transformers==4.41.2' 'omegaconf==2.3.0'
"$VENV_DIR/bin/python" -m pip freeze | sort > "$LOCK_OUT"
test -e "$REPO_DIR/lavis/models/__init__.py"
test -e "$REPO_DIR/lavis/models/blip2_models/blip2_t5.py"
test -e "$REPO_DIR/lavis/configs/models/blip2/blip2_pretrain_flant5xl.yaml"
test -e "$REPO_DIR/lavis/processors/blip_processors.py"
echo "lavis_blip2_minimal gold install route verified"
