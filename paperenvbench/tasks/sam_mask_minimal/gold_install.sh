#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$(pwd)}"
REPO_DIR="${SAM_REPO_DIR:-${RUN_ROOT}/repo}"
ENV_DIR="${PAPERENVBENCH_ENV_DIR:-${RUN_ROOT}/.venv}"
MODEL_DIR="${SAM_MODEL_DIR:-${RUN_ROOT}/models/sam}"

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/artifacts" "${MODEL_DIR}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone https://github.com/facebookresearch/segment-anything "${REPO_DIR}"
fi
git -C "${REPO_DIR}" fetch --depth 1 origin dca509fe793f601edb92606367a655c15ac00fdf
git -C "${REPO_DIR}" checkout --detach dca509fe793f601edb92606367a655c15ac00fdf

python3 -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ENV_DIR}/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
"${ENV_DIR}/bin/python" -m pip install pillow numpy opencv-python-headless pycocotools matplotlib
"${ENV_DIR}/bin/python" -m pip install -e "${REPO_DIR}"

if [[ ! -f "${MODEL_DIR}/sam_vit_b_01ec64.pth" ]]; then
  curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o "${MODEL_DIR}/sam_vit_b_01ec64.pth"
fi
"${ENV_DIR}/bin/python" -m pip freeze > "${RUN_ROOT}/requirements_lock.txt"
