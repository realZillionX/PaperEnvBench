#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$(pwd)}"
REPO_DIR="${MMSEG_REPO_DIR:-${RUN_ROOT}/repo}"
ENV_DIR="${PAPERENVBENCH_ENV_DIR:-${RUN_ROOT}/.venv}"
REPO_URL="https://github.com/open-mmlab/mmsegmentation"
COMMIT="b040e147adfa027bbc071b624bedf0ae84dfc922"

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/logs" "${RUN_ROOT}/artifacts"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi
git -C "${REPO_DIR}" fetch --depth 1 origin "${COMMIT}"
git -C "${REPO_DIR}" checkout --detach "${COMMIT}"
git -C "${REPO_DIR}" rev-parse HEAD

rm -rf "${ENV_DIR}"
python3 -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ENV_DIR}/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.8.0 torchvision==0.23.0

"${ENV_DIR}/bin/python" -m pip install --index-url https://pypi.org/simple \
  --dry-run --only-binary=:all: "mmcv==2.1.0" \
  > "${RUN_ROOT}/logs/mmcv_native_binary_probe.log" 2>&1 || true

"${ENV_DIR}/bin/python" -m pip install --index-url https://pypi.org/simple \
  "numpy==1.26.4" "matplotlib==3.10.3" "prettytable==3.16.0" "scipy==1.16.0" \
  pillow packaging addict pyyaml yapf rich termcolor "opencv-python-headless==4.11.0.86" \
  "ftfy==6.3.1" "regex==2026.5.9"
"${ENV_DIR}/bin/python" -m pip install --index-url https://pypi.org/simple --no-deps \
  "mmengine==0.10.7" "mmcv-lite==2.1.0"
"${ENV_DIR}/bin/python" -m pip install -e "${REPO_DIR}" --no-deps

"${ENV_DIR}/bin/python" -m pip freeze > "${RUN_ROOT}/requirements_lock.txt"
