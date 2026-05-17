#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$(pwd)}"
REPO_DIR="${CLIP_REPO_DIR:-${RUN_ROOT}/repo}"
ENV_DIR="${PAPERENVBENCH_ENV_DIR:-${RUN_ROOT}/.venv}"
MODEL_DIR="${CLIP_MODEL_DIR:-${RUN_ROOT}/models}"
ARTIFACT_DIR="${PAPERENVBENCH_ARTIFACT_DIR:-${RUN_ROOT}/artifacts}"
COMMIT="d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"

mkdir -p "${RUN_ROOT}" "${MODEL_DIR}" "${ARTIFACT_DIR}" "${RUN_ROOT}/logs"
export RUN_ROOT REPO_DIR MODEL_DIR ARTIFACT_DIR

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone https://github.com/openai/CLIP "${REPO_DIR}"
fi
git -C "${REPO_DIR}" fetch --depth 1 origin "${COMMIT}"
git -C "${REPO_DIR}" checkout --detach "${COMMIT}"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${COMMIT}"

python3 -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools==70.2.0 wheel
"${ENV_DIR}/bin/python" -m pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url "${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
"${ENV_DIR}/bin/python" -m pip install ftfy regex tqdm pillow
"${ENV_DIR}/bin/python" -m pip install -e "${REPO_DIR}" --no-build-isolation

"${ENV_DIR}/bin/python" - <<'PY' | tee "${RUN_ROOT}/logs/clip_readme_forward.log"
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import clip
import torch
from PIL import Image

run_root = Path(os.environ["RUN_ROOT"])
repo_dir = Path(os.environ["REPO_DIR"])
model_dir = Path(os.environ["MODEL_DIR"])
artifact_dir = Path(os.environ["ARTIFACT_DIR"])
labels = ["a diagram", "a dog", "a cat"]

assert torch.cuda.is_available(), "L4 CLIP route requires CUDA"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()

image_src = repo_dir / "CLIP.png"
image_dst = artifact_dir / "expected_artifact.png"
shutil.copy2(image_src, image_dst)
model, preprocess = clip.load("ViT-B/32", device="cuda", download_root=str(model_dir))
image = preprocess(Image.open(image_src)).unsqueeze(0).to("cuda")
text = clip.tokenize(labels).to("cuda")
with torch.no_grad():
    logits_per_image, _ = model(image, text)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()[0].tolist()

checkpoint = next(model_dir.glob("*.pt"))
payload = {
    "task_id": "clip_zeroshot_minimal",
    "repo_commit": subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip(),
    "entrypoint": "README zero-shot image classification example",
    "forward_evidence": {
        "used_clip_load": True,
        "used_clip_tokenize": True,
        "used_preprocess": True,
        "used_model_forward": True,
    },
    "model_name": "ViT-B/32",
    "checkpoint_sha256": sha(checkpoint),
    "checkpoint_size_bytes": checkpoint.stat().st_size,
    "input_image": "artifacts/expected_artifact.png",
    "input_image_source": "repo/CLIP.png",
    "input_image_sha256": sha(image_dst),
    "labels": labels,
    "logits_per_image": logits_per_image.detach().cpu().numpy()[0].tolist(),
    "probabilities": dict(zip(labels, probs)),
    "top_label": labels[int(max(range(len(probs)), key=lambda idx: probs[idx]))],
    "top_probability": max(probs),
    "probability_sum": sum(probs),
    "image_tensor_shape": list(image.shape),
    "text_tensor_shape": list(text.shape),
    "device": "cuda",
    "torch": {
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
    },
}
(artifact_dir / "expected_artifact.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"top_label": payload["top_label"], "top_probability": payload["top_probability"], "device": "cuda"}, sort_keys=True))
PY

"${ENV_DIR}/bin/python" -m pip freeze > "${RUN_ROOT}/requirements_lock.txt"
