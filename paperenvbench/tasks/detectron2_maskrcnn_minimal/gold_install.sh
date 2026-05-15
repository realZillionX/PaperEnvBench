#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$(pwd)}"
REPO_DIR="${DETECTRON2_REPO_DIR:-${RUN_ROOT}/repo}"
ENV_DIR="${PAPERENVBENCH_ENV_DIR:-${RUN_ROOT}/.venv}"
COMMIT="${DETECTRON2_COMMIT:-e0ec4e189d438848521aee7926f9900e114229f5}"

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/logs" "${RUN_ROOT}/artifacts"

if command -v apt-get >/dev/null 2>&1; then
  if ! command -v gcc >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1 || [[ ! -f /usr/include/python3.12/Python.h ]]; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential python3.12-dev
  fi
fi

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone https://github.com/facebookresearch/detectron2 "${REPO_DIR}"
fi
git -C "${REPO_DIR}" fetch --depth 1 origin "${COMMIT}"
git -C "${REPO_DIR}" checkout --detach "${COMMIT}"

python3 -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel ninja
"${ENV_DIR}/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
"${ENV_DIR}/bin/python" -m pip install \
  "numpy<2" opencv-python-headless pillow pycocotools pyyaml tqdm tensorboard \
  tabulate termcolor yacs cloudpickle fvcore omegaconf hydra-core black packaging
"${ENV_DIR}/bin/python" -m pip install --index-url https://pypi.org/simple matplotlib "iopath==0.1.9"

export PATH="${ENV_DIR}/bin:${PATH}"
FORCE_CUDA=0 MAX_JOBS="${MAX_JOBS:-2}" CC="${CC:-gcc}" CXX="${CXX:-g++}" \
  "${ENV_DIR}/bin/python" -m pip install -e "${REPO_DIR}" --no-build-isolation --no-deps

"${ENV_DIR}/bin/python" - <<'PY'
import json
from pathlib import Path

import detectron2
import torch
import torchvision
from detectron2 import _C

payload = {
    "cuda_available": torch.cuda.is_available(),
    "detectron2": detectron2.__version__,
    "native_extension": str(_C.__file__),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
}
Path("artifacts/install_probe.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

"${ENV_DIR}/bin/python" -m pip freeze > "${RUN_ROOT}/requirements_lock.txt"
