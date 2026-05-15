#!/usr/bin/env bash
set -Eeuo pipefail

TASK_ID="mmdetection_fasterrcnn_minimal"
REPO_URL="https://github.com/open-mmlab/mmdetection.git"
COMMIT="cfd5d3a985b0249de009b67d04f37263e11cdf3d"
ROOT_DIR="${PAPERENVBENCH_RUN_DIR:-$(pwd)}"
REPO_DIR="${PAPERENVBENCH_REPO_DIR:-$ROOT_DIR/repo}"
VENV_DIR="${PAPERENVBENCH_VENV_DIR:-$ROOT_DIR/venv}"
LOG_DIR="$ROOT_DIR/logs"

mkdir -p "$LOG_DIR" "$ROOT_DIR/artifacts"

echo "task_id=$TASK_ID"
echo "root_dir=$ROOT_DIR"
echo "repo_dir=$REPO_DIR"
echo "venv_dir=$VENV_DIR"
echo "commit=$COMMIT"

if [ ! -d "$REPO_DIR/.git" ]; then
  rm -rf "$REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
fi

git -C "$REPO_DIR" fetch --all --tags --prune
git -C "$REPO_DIR" checkout "$COMMIT"
git -C "$REPO_DIR" rev-parse HEAD

python3 -m venv --clear "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip "setuptools<82" wheel

if ! python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0" "torchvision==0.23.0"; then
  echo "CPU wheel index install failed; falling back to the configured pip index." >&2
  python -m pip install "torch==2.8.0" "torchvision==0.23.0"
fi

set +e
python -m pip install --dry-run -v "mmcv>=2.0.0rc4,<2.2.0" > "$LOG_DIR/native_mmcv_dryrun_probe.log" 2>&1
native_mmcv_status=$?
set -e
echo "native_mmcv_dryrun_status=$native_mmcv_status"
if [ "$native_mmcv_status" -eq 0 ]; then
  echo "native mmcv dry-run unexpectedly succeeded; verifier still records whether Faster R-CNN config executes." >&2
else
  echo "native mmcv dry-run failed; using mmcv-lite fallback and recording the probe log." >&2
fi

python -m pip install \
  "numpy==1.26.4" \
  "opencv-python-headless==4.11.0.86" \
  "addict==2.4.0" \
  "pyyaml==6.0.3" \
  "rich==15.0.0" \
  "termcolor==3.3.0" \
  "yapf==0.43.0" \
  "six==1.17.0" \
  "terminaltables==3.1.10" \
  "tqdm==4.67.3"

# Avoid pulling mmengine's optional matplotlib/opencv dependency surface from
# the internal PyPI mirror; the verifier only needs Config/version utilities.
python -m pip install --no-deps "mmengine==0.10.7" "mmcv-lite==2.1.0"

PYTHONPATH="$REPO_DIR" python - <<'PY'
import importlib.metadata as metadata
import pathlib
import sys

import mmcv
import mmengine
import mmdet
import torch

repo_dir = pathlib.Path("repo").resolve()
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torchvision", metadata.version("torchvision"))
print("mmcv", mmcv.__version__)
print("mmengine", mmengine.__version__)
print("mmdet", mmdet.__version__)
print("repo_dir", repo_dir)
PY

set +e
PYTHONPATH="$REPO_DIR" python - <<'PY' > "$LOG_DIR/native_faster_rcnn_config_probe.log" 2>&1
from pathlib import Path

from mmengine.config import Config

repo_dir = Path("repo").resolve()
cfg = Config.fromfile(repo_dir / "configs" / "_base_" / "models" / "faster-rcnn_r50_fpn.py")
print(cfg.model)
PY
native_config_status=$?
set -e
echo "native_faster_rcnn_config_status=$native_config_status"

set +e
PYTHONPATH="$REPO_DIR" python - <<'PY' > "$LOG_DIR/native_mmcv_ops_probe.log" 2>&1
from mmcv.ops import RoIAlign, nms

print("RoIAlign", RoIAlign)
print("nms", nms)
PY
native_ops_status=$?
set -e
echo "native_mmcv_ops_status=$native_ops_status"

python -m pip freeze | sort > "$ROOT_DIR/requirements_lock.txt"
{
  echo "# Source repository is used via PYTHONPATH, not installed as a wheel."
  echo "# repo=$REPO_URL"
  echo "# commit=$COMMIT"
  echo "# native_mmcv_dryrun_status=$native_mmcv_status"
  echo "# native_faster_rcnn_config_status=$native_config_status"
  echo "# native_mmcv_ops_status=$native_ops_status"
} >> "$ROOT_DIR/requirements_lock.txt"
