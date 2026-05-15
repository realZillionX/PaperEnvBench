#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
VIDEOMAE_REPO="${VIDEOMAE_REPO:-${REPO_ROOT}/VideoMAE}"
VENV="${VENV:-${REPO_ROOT}/venv}"
COMMIT="14ef8d856287c94ef1f985fe30f958eb4ec2c55d"

if [[ ! -d "${VIDEOMAE_REPO}/.git" ]]; then
  rm -rf "${VIDEOMAE_REPO}"
  git clone https://github.com/MCG-NJU/VideoMAE.git "${VIDEOMAE_REPO}"
fi

git -C "${VIDEOMAE_REPO}" fetch --tags --prune origin
git -C "${VIDEOMAE_REPO}" checkout "${COMMIT}"

actual_commit="$(git -C "${VIDEOMAE_REPO}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${COMMIT}" ]]; then
  echo "Unexpected VideoMAE commit: ${actual_commit}; expected ${COMMIT}" >&2
  exit 2
fi

python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV}/bin/python" -m pip install \
  torch==2.4.1 \
  torchvision==0.19.1 \
  numpy==1.26.4 \
  pillow==12.2.0 \
  timm==0.4.12 \
  pyyaml==6.0.3

"${VENV}/bin/python" - <<'PY'
import json
import os
import sys
from pathlib import Path

import numpy as np
import timm
import torch

repo = Path(os.environ.get("VIDEOMAE_REPO", "VideoMAE")).resolve()
sys.path.insert(0, str(repo))
import modeling_finetune

model = modeling_finetune.vit_small_patch16_224(
    pretrained=False,
    img_size=32,
    num_classes=5,
    all_frames=4,
    tubelet_size=2,
    drop_path_rate=0.0,
    fc_drop_rate=0.0,
    use_checkpoint=False,
    use_mean_pooling=True,
    init_scale=1.0,
)

print(json.dumps({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "timm": timm.__version__,
    "numpy": np.__version__,
    "repo": str(repo),
    "model": model.__class__.__name__,
    "patch_embed": model.patch_embed.proj.__class__.__name__,
    "patch_embed_kernel": list(model.patch_embed.proj.kernel_size),
    "num_patches": model.patch_embed.num_patches,
}, indent=2, sort_keys=True))
PY

mkdir -p "${REPO_ROOT}/artifacts" "${REPO_ROOT}/logs"
"${VENV}/bin/python" -m pip freeze > "${REPO_ROOT}/requirements_lock.txt"
echo "VideoMAE gold install completed at ${REPO_ROOT}"
