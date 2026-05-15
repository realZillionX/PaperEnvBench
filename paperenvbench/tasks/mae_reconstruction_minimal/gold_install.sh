#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
MAE_REPO="${MAE_REPO:-${REPO_ROOT}/mae}"
VENV="${VENV:-${REPO_ROOT}/venv}"
COMMIT="efb2a8062c206524e35e47d04501ed4f544c0ae8"

if [[ ! -d "${MAE_REPO}/.git" ]]; then
  rm -rf "${MAE_REPO}"
  git clone https://github.com/facebookresearch/mae.git "${MAE_REPO}"
fi

git -C "${MAE_REPO}" fetch --tags --prune origin
git -C "${MAE_REPO}" checkout "${COMMIT}"

actual_commit="$(git -C "${MAE_REPO}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${COMMIT}" ]]; then
  echo "Unexpected MAE commit: ${actual_commit}; expected ${COMMIT}" >&2
  exit 2
fi

python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV}/bin/python" -m pip install \
  torch==2.4.1 \
  torchvision==0.19.1 \
  numpy==1.26.4 \
  pillow==12.2.0 \
  timm==0.6.13 \
  pyyaml==6.0.3 \
  huggingface-hub==1.14.0

"${VENV}/bin/python" - <<'PY'
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from timm.models.vision_transformer import Block

repo = Path(__import__("os").environ.get("MAE_REPO", "mae")).resolve()
if not (repo / "models_mae.py").exists():
    raise SystemExit(f"Missing models_mae.py under {repo}")

print(json.dumps({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "timm": timm.__version__,
    "numpy": np.__version__,
    "timm_block_accepts_qk_scale": "qk_scale" in inspect.signature(Block.__init__).parameters,
    "repo": str(repo),
}, indent=2, sort_keys=True))
PY

mkdir -p "${REPO_ROOT}/artifacts" "${REPO_ROOT}/input"
echo "MAE gold install completed at ${REPO_ROOT}"
