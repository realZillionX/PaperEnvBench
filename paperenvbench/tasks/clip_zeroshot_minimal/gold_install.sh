#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$(pwd)}"
REPO_DIR="${CLIP_REPO_DIR:-${RUN_ROOT}/repo}"
ENV_DIR="${PAPERENVBENCH_ENV_DIR:-${RUN_ROOT}/.venv}"

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/artifacts" "${RUN_ROOT}/models"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone https://github.com/openai/CLIP "${REPO_DIR}"
fi
git -C "${REPO_DIR}" fetch --depth 1 origin d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
git -C "${REPO_DIR}" checkout --detach d05afc436d78f1c48dc0dbf8e5980a9d471f35f6

python3 -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ENV_DIR}/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
"${ENV_DIR}/bin/python" -m pip install pillow numpy ftfy regex tqdm
"${ENV_DIR}/bin/python" -m pip install -e "${REPO_DIR}"
"${ENV_DIR}/bin/python" -m pip freeze > "${RUN_ROOT}/requirements_lock.txt"
