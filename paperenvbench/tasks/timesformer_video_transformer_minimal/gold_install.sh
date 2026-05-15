#!/usr/bin/env bash
set -euo pipefail

TASK_ID="timesformer_video_transformer_minimal"
REPO_URL="https://github.com/facebookresearch/TimeSformer.git"
COMMIT="a5ef29a7b7264baff199a30b3306ac27de901133"
RUN_DIR="${PAPERENVBENCH_RUN_DIR:-$(pwd)}"
REPO_DIR="${PAPERENVBENCH_REPO_DIR:-${RUN_DIR}/repo}"
VENV_DIR="${PAPERENVBENCH_VENV_DIR:-${RUN_DIR}/venv}"
LOG_DIR="${RUN_DIR}/logs"

mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${RUN_DIR}/artifacts"

echo "task_id=${TASK_ID}"
echo "run_dir=${RUN_DIR}"
echo "repo_dir=${REPO_DIR}"
echo "venv_dir=${VENV_DIR}"
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

if ! python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0"; then
  echo "CPU wheel index install failed; falling back to configured pip index." >&2
  python -m pip install "torch==2.8.0"
fi

python -m pip install \
  "einops==0.8.1" \
  "fvcore==0.1.5.post20221221" \
  "numpy==2.4.4" \
  "pyyaml==6.0.3" \
  "yacs==0.1.8" \
  "simplejson==3.20.2"

export PAPERENVBENCH_REPO_DIR="${REPO_DIR}"
PYTHONPATH="${REPO_DIR}" python - <<'PY'
from __future__ import annotations

import collections.abc
import importlib.metadata as metadata
import os
import pathlib
import subprocess
import sys
import types

import torch
import torch.nn.modules.linear as linear_mod

shim = types.ModuleType("torch._six")
shim.container_abcs = collections.abc
sys.modules["torch._six"] = shim
if not hasattr(linear_mod, "_LinearWithBias"):
    linear_mod._LinearWithBias = torch.nn.Linear

from timesformer.models.vit import VisionTransformer

repo_dir = pathlib.Path(os.environ["PAPERENVBENCH_REPO_DIR"]).resolve()
commit = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()

torch.set_num_threads(2)
model = VisionTransformer(
    img_size=32,
    patch_size=16,
    num_classes=7,
    embed_dim=64,
    depth=2,
    num_heads=4,
    mlp_ratio=2.0,
    qkv_bias=True,
    num_frames=4,
    attention_type="divided_space_time",
    drop_path_rate=0.0,
)
model.eval()

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("einops", metadata.version("einops"))
print("fvcore", metadata.version("fvcore"))
print("numpy", metadata.version("numpy"))
print("pyyaml", metadata.version("pyyaml"))
print("yacs", metadata.version("yacs"))
print("simplejson", metadata.version("simplejson"))
print("repo_dir", repo_dir)
print("repo_commit", commit)
print("model_class", f"{model.__class__.__module__}.{model.__class__.__name__}")
print("attention_type", model.attention_type)
print("parameter_count", sum(param.numel() for param in model.parameters()))
print("compat_shim", "torch._six,torch.nn.modules.linear._LinearWithBias")
PY

python -m pip freeze | sort > "${RUN_DIR}/requirements_lock.txt"
{
  echo "# Source repository is used via PYTHONPATH, not installed as a wheel."
  echo "# repo=${REPO_URL}"
  echo "# commit=${COMMIT}"
  echo "# fallback=L4_fallback with Python 3.12 compatibility shims for removed PyTorch private APIs."
} >> "${RUN_DIR}/requirements_lock.txt"
