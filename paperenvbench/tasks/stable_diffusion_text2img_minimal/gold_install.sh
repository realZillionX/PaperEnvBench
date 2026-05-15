#!/usr/bin/env bash
set -euo pipefail

TASK_ID="stable_diffusion_text2img_minimal"
REPO_URL="https://github.com/CompVis/stable-diffusion.git"
REPO_COMMIT="21f890f9da3cfbeaba8e2ac3c425ee9e998d5229"
RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$PWD}"
REPO_DIR="${STABLE_DIFFUSION_TEXT2IMG_MINIMAL_REPO_DIR:-$RUN_ROOT/repo}"
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
  "numpy==2.2.6" \
  "Pillow==11.3.0" \
  "omegaconf==2.3.0" \
  "einops==0.8.1" \
  "pytorch-lightning==1.9.5" \
  "transformers==4.41.2" \
  "diffusers==0.30.3" \
  "invisible-watermark==0.2.0" \
  "opencv-python-headless==4.12.0.88"
"$VENV_DIR/bin/python" -m pip install --no-deps -e "$REPO_DIR"
"$VENV_DIR/bin/python" -m pip freeze | sort > "$LOCK_OUT"

test -e "$REPO_DIR/scripts/txt2img.py"
test -e "$REPO_DIR/configs/stable-diffusion/v1-inference.yaml"
test -e "$REPO_DIR/Stable_Diffusion_v1_Model_Card.md"
test -e "$REPO_DIR/LICENSE"
grep -q "model.ckpt" "$REPO_DIR/README.md"
grep -q "CreativeML Open RAIL" "$REPO_DIR/LICENSE"

echo "$TASK_ID gold install route verified"
echo "checkpoint_route=models/ldm/stable-diffusion-v1/model.ckpt"
echo "license_gate=CreativeML Open RAIL-M Hugging Face model acceptance required"
