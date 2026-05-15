#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$(pwd)}"
REPO_DIR="${MMACTION2_REPO_DIR:-${RUN_ROOT}/repo}"
ENV_DIR="${PAPERENVBENCH_ENV_DIR:-${RUN_ROOT}/.venv}"
REPO_URL="https://github.com/open-mmlab/mmaction2"
COMMIT="a5a167dff2399e2d182a60332325f9c0d4663517"

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/logs" "${RUN_ROOT}/artifacts"

echo "[gold_install] run_root=${RUN_ROOT}"
echo "[gold_install] repo_dir=${REPO_DIR}"
echo "[gold_install] venv_dir=${ENV_DIR}"
echo "[gold_install] commit=${COMMIT}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi
git -C "${REPO_DIR}" fetch --depth 1 origin "${COMMIT}"
git -C "${REPO_DIR}" checkout --detach "${COMMIT}"
git -C "${REPO_DIR}" log -1 --format='[gold_install] checked_out=%H %ci %s'

rm -rf "${ENV_DIR}"
python3 -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ENV_DIR}/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.8.0 torchvision==0.23.0

"${ENV_DIR}/bin/python" -m pip install --index-url https://pypi.org/simple \
  --dry-run --only-binary=:all: "mmcv==2.1.0" \
  > "${RUN_ROOT}/logs/mmcv_native_binary_probe.log" 2>&1 || true

"${ENV_DIR}/bin/python" -m pip install --index-url https://pypi.org/simple \
  "numpy==1.26.4" "opencv-python-headless==4.11.0.86" "decord==0.6.0" \
  "importlib_metadata==8.7.0" pillow packaging pyyaml addict yapf rich termcolor \
  regex ftfy matplotlib scipy tqdm requests six einops
"${ENV_DIR}/bin/python" -m pip install --index-url https://pypi.org/simple --no-deps \
  "mmengine==0.10.7" "mmcv-lite==2.1.0"
"${ENV_DIR}/bin/python" -m pip install -e "${REPO_DIR}" --no-deps

"${ENV_DIR}/bin/python" -m pip freeze > "${RUN_ROOT}/requirements_lock.txt"
echo "[gold_install] wrote ${RUN_ROOT}/requirements_lock.txt"
