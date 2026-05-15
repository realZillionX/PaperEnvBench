#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-$(pwd)}"
DINO_REPO="${DINO_REPO:-${RUN_ROOT}/repo}"
VENV="${VENV:-${RUN_ROOT}/venv}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/outputs/dino_vit_feature_minimal}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/checkpoints}"
DINO_COMMIT="${DINO_COMMIT:-7c446df5b9f45747937fb0d72314eb9f7b66930a}"
DINO_REPO_URL="${DINO_REPO_URL:-https://github.com/facebookresearch/dino.git}"

if [[ ! -d "${DINO_REPO}/.git" ]]; then
  rm -rf "${DINO_REPO}"
  git clone "${DINO_REPO_URL}" "${DINO_REPO}"
fi

git -C "${DINO_REPO}" fetch --tags --prune
git -C "${DINO_REPO}" checkout "${DINO_COMMIT}"
git -C "${DINO_REPO}" status --short

python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV}/bin/python" -m pip install numpy pillow torch

mkdir -p "${OUT_DIR}" "${CHECKPOINT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export DINO_REPO OUT_DIR CHECKPOINT_DIR
"${VENV}/bin/python" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

repo = Path(os.environ["DINO_REPO"]).resolve()
out_dir = Path(os.environ["OUT_DIR"]).resolve()
checkpoint_dir = Path(os.environ["CHECKPOINT_DIR"]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(repo))
import vision_transformer as vits  # noqa: E402

seed = 20260515
torch.manual_seed(seed)
np.random.seed(seed)

y, x = np.mgrid[0:224, 0:224]
image = np.stack(
    [
        x % 256,
        y % 256,
        (x * 3 + y * 5) % 256,
    ],
    axis=-1,
).astype("uint8")
image_path = out_dir / "dino_synthetic_input.png"
Image.fromarray(image, "RGB").save(image_path)

arr = image.astype("float32") / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype="float32")
std = np.array([0.229, 0.224, 0.225], dtype="float32")
tensor = torch.from_numpy(((arr - mean) / std).transpose(2, 0, 1)).unsqueeze(0)

model = vits.__dict__["vit_small"](patch_size=16, num_classes=0)
checkpoint_url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
state_dict = torch.hub.load_state_dict_from_url(
    checkpoint_url,
    model_dir=str(checkpoint_dir),
    map_location="cpu",
    progress=True,
)
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
model.eval()

with torch.no_grad():
    features = model(tensor)
feature_vec = features.detach().cpu().float().numpy()[0]

checkpoint_path = checkpoint_dir / "dino_deitsmall16_pretrain.pth"
repo_commit = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"],
    text=True,
).strip()
repo_commit_short = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "--short=7", "HEAD"],
    text=True,
).strip()

summary = {
    "task_id": "dino_vit_feature_minimal",
    "status": "pass",
    "repo_url": "https://github.com/facebookresearch/dino",
    "repo_commit": repo_commit,
    "repo_commit_short": repo_commit_short,
    "model": "dino_vits16",
    "model_source": "facebookresearch/dino vision_transformer.vit_small patch_size=16",
    "checkpoint_url": checkpoint_url,
    "checkpoint_path": str(checkpoint_path),
    "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    "checkpoint_size_bytes": checkpoint_path.stat().st_size,
    "load_state_dict": {
        "strict": True,
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
    },
    "input": {
        "path": str(image_path),
        "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "shape": [1, 3, 224, 224],
        "seed": seed,
    },
    "feature": {
        "shape": list(features.shape),
        "mean": float(feature_vec.mean()),
        "std": float(feature_vec.std()),
        "l2_norm": float(np.linalg.norm(feature_vec)),
        "min": float(feature_vec.min()),
        "max": float(feature_vec.max()),
        "first8": [float(value) for value in feature_vec[:8]],
        "sha256_rounded_6": hashlib.sha256(np.round(feature_vec, 6).astype("<f4").tobytes()).hexdigest(),
    },
    "runtime": {
        "torch_version": torch.__version__,
        "device": "cpu",
    },
}

(out_dir / "feature_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

ls -lh "${OUT_DIR}" "${CHECKPOINT_DIR}"
