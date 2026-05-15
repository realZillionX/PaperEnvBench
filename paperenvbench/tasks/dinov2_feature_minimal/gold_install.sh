#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/facebookresearch/dinov2.git"
REPO_COMMIT="7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c"

RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$PWD}"
REPO_DIR="${DINOV2_REPO_DIR:-$RUN_ROOT/repo}"
VENV_DIR="${DINOV2_VENV_DIR:-$RUN_ROOT/venv}"
LOCK_OUT="${PAPERENVBENCH_LOCK_OUT:-$RUN_ROOT/requirements_lock.txt}"

echo "[gold_install] run_root=$RUN_ROOT"
echo "[gold_install] repo_dir=$REPO_DIR"
echo "[gold_install] venv_dir=$VENV_DIR"
echo "[gold_install] commit=$REPO_COMMIT"

if ! command -v git >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates
fi

mkdir -p "$RUN_ROOT"
if [ ! -d "$REPO_DIR/.git" ]; then
  rm -rf "$REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch --all --tags --prune
git -C "$REPO_DIR" checkout "$REPO_COMMIT"
git -C "$REPO_DIR" rev-parse HEAD

rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip config set global.index-url http://nexus.sii.shaipower.online/repository/pypi/simple || true
"$VENV_DIR/bin/python" -m pip config set global.trusted-host nexus.sii.shaipower.online || true

if ! "$VENV_DIR/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.8.0" \
  "torchvision==0.23.0"; then
  echo "[gold_install] PyTorch CPU wheel index failed; falling back to configured pip index." >&2
  "$VENV_DIR/bin/python" -m pip install \
    "torch==2.8.0" \
    "torchvision==0.23.0"
fi

"$VENV_DIR/bin/python" -m pip install \
  "numpy==2.4.4" \
  "Pillow==12.1.0" \
  "omegaconf==2.3.0" \
  "PyYAML==6.0.3"

"$VENV_DIR/bin/python" -m pip install --no-deps -e "$REPO_DIR"
"$VENV_DIR/bin/python" -m pip freeze | sort > "$LOCK_OUT"

export XFORMERS_DISABLED=1
"$VENV_DIR/bin/python" - <<'PY'
import importlib.metadata
import os
import pathlib
import torch
from dinov2.hub.backbones import dinov2_vits14

print("dinov2", importlib.metadata.version("dinov2"))
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("xformers_disabled", os.environ.get("XFORMERS_DISABLED"))
model = dinov2_vits14(pretrained=False, img_size=224)
print("model_embed_dim", model.embed_dim)
print("repo_module", pathlib.Path(model.__class__.__module__.replace(".", "/")))
PY

echo "[gold_install] wrote $LOCK_OUT"
