#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$(pwd)}"
TASK_DIR="${PAPERENVBENCH_TASK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_DIR="${NERFSTUDIO_REPO_DIR:-${RUN_ROOT}/repo}"
ENV_DIR="${PAPERENVBENCH_ENV_DIR:-${RUN_ROOT}/.venv}"
REPO_URL="https://github.com/nerfstudio-project/nerfstudio"
COMMIT="50e0e3c70c775e89333256213363badbf074f29d"

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/logs" "${RUN_ROOT}/artifacts"

echo "[gold_install] run_root=${RUN_ROOT}"
echo "[gold_install] task_dir=${TASK_DIR}"
echo "[gold_install] repo_dir=${REPO_DIR}"
echo "[gold_install] venv_dir=${ENV_DIR}"
echo "[gold_install] commit=${COMMIT}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi
git -C "${REPO_DIR}" fetch --depth 1 origin "${COMMIT}"
git -C "${REPO_DIR}" checkout --detach "${COMMIT}"
git -C "${REPO_DIR}" log -1 --format='[gold_install] checked_out=%H %ci %s'

python3 -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ENV_DIR}/bin/python" -m pip install --dry-run --only-binary=:all: "nerfacc==0.5.2" "gsplat==1.4.0" "open3d>=0.16.0" \
  > "${RUN_ROOT}/logs/native_dependency_probe.log" 2>&1 || true

{
  echo "[gold_install] pyproject dependencies containing native or GPU pressure:"
  grep -E 'nerfacc|gsplat|open3d|xatlas|pymeshlab|torch|torchvision' "${REPO_DIR}/pyproject.toml" || true
  echo "[gold_install] route evidence:"
  grep -n 'ns-train = "nerfstudio.scripts.train:entrypoint"' "${REPO_DIR}/pyproject.toml"
  grep -n 'method_configs\["nerfacto"\]' "${REPO_DIR}/nerfstudio/configs/method_configs.py"
  grep -n 'NerfstudioDataParserConfig' "${REPO_DIR}/nerfstudio/configs/method_configs.py" | head
  grep -n 'VanillaPipelineConfig' "${REPO_DIR}/nerfstudio/configs/method_configs.py" | head
  grep -n 'NerfactoModelConfig' "${REPO_DIR}/nerfstudio/configs/method_configs.py" | head
} > "${RUN_ROOT}/logs/route_probe.log"

"${ENV_DIR}/bin/python" "${TASK_DIR}/verify.py" --artifact-dir "${RUN_ROOT}/artifacts" --generate --json
"${ENV_DIR}/bin/python" -m pip freeze > "${RUN_ROOT}/requirements_lock.txt"
if [[ "$(cd "${RUN_ROOT}/artifacts" && pwd)/expected_artifact.json" != "$(cd "${TASK_DIR}/artifacts" && pwd)/expected_artifact.json" ]]; then
  cp "${RUN_ROOT}/artifacts/expected_artifact.json" "${TASK_DIR}/artifacts/expected_artifact.json"
fi
