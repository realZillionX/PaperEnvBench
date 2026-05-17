#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$(pwd)}"
REPO_DIR="${SAM_REPO_DIR:-${RUN_ROOT}/repo}"
ENV_DIR="${PAPERENVBENCH_ENV_DIR:-${RUN_ROOT}/.venv}"
MODEL_DIR="${SAM_MODEL_DIR:-${RUN_ROOT}/models/sam}"
ARTIFACT_DIR="${PAPERENVBENCH_ARTIFACT_DIR:-${RUN_ROOT}/artifacts}"
COMMIT="dca509fe793f601edb92606367a655c15ac00fdf"
CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
CHECKPOINT_SHA="ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"

mkdir -p "${RUN_ROOT}" "${MODEL_DIR}" "${ARTIFACT_DIR}" "${RUN_ROOT}/logs"
export RUN_ROOT REPO_DIR MODEL_DIR ARTIFACT_DIR

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone https://github.com/facebookresearch/segment-anything "${REPO_DIR}"
fi
git -C "${REPO_DIR}" fetch --depth 1 origin "${COMMIT}"
git -C "${REPO_DIR}" checkout --detach "${COMMIT}"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${COMMIT}"

python3 -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools==70.2.0 wheel
"${ENV_DIR}/bin/python" -m pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url "${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
"${ENV_DIR}/bin/python" -m pip install pillow numpy opencv-python-headless pycocotools matplotlib
"${ENV_DIR}/bin/python" -m pip install -e "${REPO_DIR}" --no-build-isolation

if [[ ! -f "${MODEL_DIR}/sam_vit_b_01ec64.pth" ]]; then
  curl -L "${CHECKPOINT_URL}" -o "${MODEL_DIR}/sam_vit_b_01ec64.pth"
fi
test "$(sha256sum "${MODEL_DIR}/sam_vit_b_01ec64.pth" | cut -d ' ' -f 1)" = "${CHECKPOINT_SHA}"

"${ENV_DIR}/bin/python" - <<'PY' | tee "${RUN_ROOT}/logs/sam_predictor_forward.log"
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from segment_anything import SamPredictor, sam_model_registry

repo_dir = Path(os.environ["REPO_DIR"])
model_dir = Path(os.environ["MODEL_DIR"])
artifact_dir = Path(os.environ["ARTIFACT_DIR"])
checkpoint = model_dir / "sam_vit_b_01ec64.pth"

assert torch.cuda.is_available(), "L4 SAM route requires CUDA"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()

input_path = artifact_dir / "expected_input.png"
mask_path = artifact_dir / "expected_artifact.png"
image = Image.new("RGB", (256, 256), "white")
draw = ImageDraw.Draw(image)
draw.rectangle([72, 56, 188, 204], fill=(35, 35, 35))
draw.ellipse([112, 96, 152, 136], fill=(220, 220, 220))
image.save(input_path)
image_array = np.array(Image.open(input_path).convert("RGB"))

sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
sam.to(device="cuda")
predictor = SamPredictor(sam)
predictor.set_image(image_array)
point_coords = np.array([[128, 130]])
point_labels = np.array([1])
with torch.no_grad():
    masks, scores, logits = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)
selected = int(np.argmax(scores))
mask = masks[selected].astype(np.uint8)
Image.fromarray(mask * 255).save(mask_path)

summary = {
    "task_id": "sam_mask_minimal",
    "repo_commit": subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip(),
    "entrypoint": "SamPredictor.predict point prompt",
    "model_type": "vit_b",
    "checkpoint_sha256": sha(checkpoint),
    "checkpoint_size_bytes": checkpoint.stat().st_size,
    "input_image": "artifacts/expected_input.png",
    "input_image_sha256": sha(input_path),
    "input_shape": list(image_array.shape),
    "prompt": {"point_coords": point_coords.tolist(), "point_labels": point_labels.tolist(), "multimask_output": True},
    "selected_mask_index": selected,
    "score": float(scores[selected]),
    "all_scores": [float(item) for item in scores.tolist()],
    "logits_shape": list(logits.shape),
    "mask": "artifacts/expected_artifact.png",
    "mask_sha256": sha(mask_path),
    "mask_pixels": int(mask.sum()),
    "mask_shape": list(mask.shape),
    "mask_unique_values": sorted(np.unique(mask).astype(int).tolist()),
    "device": "cuda",
    "torch": {
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
    },
}
(artifact_dir / "expected_artifact.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"score": summary["score"], "mask_pixels": summary["mask_pixels"], "device": "cuda"}, sort_keys=True))
PY

"${ENV_DIR}/bin/python" -m pip freeze > "${RUN_ROOT}/requirements_lock.txt"
