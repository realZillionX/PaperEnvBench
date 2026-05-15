#!/usr/bin/env bash
set -euo pipefail

TASK_ID="detr_object_detection_minimal"
REPO_URL="https://github.com/facebookresearch/detr.git"
COMMIT="29901c51d7fe8712168b8d0d64351170bc0f83e0"
RUN_DIR="${PAPERENVBENCH_RUN_DIR:-$(pwd)}"
REPO_DIR="${PAPERENVBENCH_REPO_DIR:-${RUN_DIR}/repo}"
VENV_DIR="${PAPERENVBENCH_VENV_DIR:-${RUN_DIR}/venv}"
CHECKPOINT_DIR="${PAPERENVBENCH_CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}"
LOG_DIR="${RUN_DIR}/logs"

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${CHECKPOINT_DIR}" "${RUN_DIR}/artifacts"

echo "task_id=${TASK_ID}"
echo "run_dir=${RUN_DIR}"
echo "repo_dir=${REPO_DIR}"
echo "venv_dir=${VENV_DIR}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "commit=${COMMIT}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  rm -rf "${REPO_DIR}"
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

git -C "${REPO_DIR}" fetch --all --tags --prune
git -C "${REPO_DIR}" checkout --detach "${COMMIT}"
git -C "${REPO_DIR}" rev-parse HEAD

python3 -m venv --clear "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if ! python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0" "torchvision==0.23.0"; then
  echo "CPU wheel index install failed; falling back to configured pip index." >&2
  python -m pip install "torch==2.8.0" "torchvision==0.23.0"
fi

python -m pip install "pillow==12.2.0" "numpy==2.4.4" "scipy==1.17.0"

export PAPERENVBENCH_REPO_DIR="${REPO_DIR}"
export PAPERENVBENCH_CHECKPOINT_DIR="${CHECKPOINT_DIR}"
PYTHONPATH="${REPO_DIR}" TORCH_HOME="${CHECKPOINT_DIR}" python - <<'PY'
from __future__ import annotations

import importlib.metadata as metadata
import os
import pathlib
import subprocess
import sys

import torch

import models.backbone as detr_backbone

# DETR's checkpoint includes the ResNet backbone weights.  This avoids an
# unnecessary extra ImageNet ResNet download from the old torchvision API path.
detr_backbone.is_main_process = lambda: False

import hubconf

repo_dir = pathlib.Path(os.environ.get("PAPERENVBENCH_REPO_DIR", "repo")).resolve()
checkpoint_dir = pathlib.Path(os.environ.get("PAPERENVBENCH_CHECKPOINT_DIR", "checkpoints")).resolve()
commit = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
model, postprocessor = hubconf.detr_resnet50(pretrained=True, return_postprocessor=True)
model.eval()
checkpoint_path = checkpoint_dir / "hub" / "checkpoints" / "detr-r50-e632da11.pth"

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torchvision", metadata.version("torchvision"))
print("numpy", metadata.version("numpy"))
print("pillow", metadata.version("pillow"))
print("scipy", metadata.version("scipy"))
print("repo_dir", repo_dir)
print("repo_commit", commit)
print("model_class", f"{model.__class__.__module__}.{model.__class__.__name__}")
print("postprocessor_class", f"{postprocessor.__class__.__module__}.{postprocessor.__class__.__name__}")
print("checkpoint_path", checkpoint_path)
print("checkpoint_exists", checkpoint_path.exists())
print("checkpoint_size_bytes", checkpoint_path.stat().st_size if checkpoint_path.exists() else 0)
PY

python -m pip freeze | sort > "${RUN_DIR}/requirements_lock.txt"
{
  echo "# Source repository is used via PYTHONPATH, not installed as a wheel."
  echo "# repo=${REPO_URL}"
  echo "# commit=${COMMIT}"
  echo "# checkpoint=https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
} >> "${RUN_DIR}/requirements_lock.txt"
