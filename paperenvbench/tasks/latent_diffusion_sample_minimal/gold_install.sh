#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/CompVis/latent-diffusion.git"
REPO_COMMIT="a506df5756472e2ebaf9078affdde2c4f1502cd4"
RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$PWD}"
REPO_DIR="${LATENT_DIFFUSION_SAMPLE_MINIMAL_REPO_DIR:-$RUN_ROOT/repo}"
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
"$VENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple \
  "torch==2.8.0" \
  "torchvision==0.23.0" \
  "numpy==2.3.3" \
  "pillow==11.3.0" \
  "omegaconf==2.3.0" \
  "einops==0.8.1" \
  "pytorch-lightning==2.5.5" \
  "tqdm==4.67.1"
"$VENV_DIR/bin/python" -m pip freeze | sort > "$LOCK_OUT"

test -e "$REPO_DIR/scripts/txt2img.py"
test -e "$REPO_DIR/configs/latent-diffusion/txt2img-1p4B-eval.yaml"
test -e "$REPO_DIR/ldm/models/diffusion/ddim.py"
test -e "$REPO_DIR/ldm/models/diffusion/plms.py"
test -e "$REPO_DIR/ldm/models/autoencoder.py"
test -e "$REPO_DIR/ldm/util.py"
echo "latent_diffusion_sample_minimal gold install route verified"
