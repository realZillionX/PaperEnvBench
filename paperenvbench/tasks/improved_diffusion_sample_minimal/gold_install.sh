#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
IMPROVED_DIFFUSION_REPO="${IMPROVED_DIFFUSION_REPO:-${REPO_ROOT}/improved-diffusion}"
VENV="${VENV:-${REPO_ROOT}/venv}"
COMMIT="1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5"

if [[ ! -d "${IMPROVED_DIFFUSION_REPO}/.git" ]]; then
  rm -rf "${IMPROVED_DIFFUSION_REPO}"
  git clone https://github.com/openai/improved-diffusion.git "${IMPROVED_DIFFUSION_REPO}"
fi

git -C "${IMPROVED_DIFFUSION_REPO}" fetch --tags --prune origin
git -C "${IMPROVED_DIFFUSION_REPO}" checkout "${COMMIT}"

actual_commit="$(git -C "${IMPROVED_DIFFUSION_REPO}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${COMMIT}" ]]; then
  echo "Unexpected improved-diffusion commit: ${actual_commit}; expected ${COMMIT}" >&2
  exit 2
fi

python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV}/bin/python" -m pip install \
  torch==2.4.1 \
  numpy==1.26.4 \
  blobfile==3.1.0 \
  tqdm==4.67.1
"${VENV}/bin/python" -m pip install -e "${IMPROVED_DIFFUSION_REPO}"

mkdir -p "${REPO_ROOT}/artifacts" "${REPO_ROOT}/logs"
"${VENV}/bin/python" -m pip freeze > "${REPO_ROOT}/requirements_lock.txt"
"${VENV}/bin/python" "${REPO_ROOT}/verify.py" \
  --repo-dir "${IMPROVED_DIFFUSION_REPO}" \
  --artifact-dir "${REPO_ROOT}/artifacts" \
  --json | tee "${REPO_ROOT}/logs/gold_verify.log"

echo "improved-diffusion gold install completed at ${REPO_ROOT}"
