#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${PAPERENVBENCH_REPO_DIR:-${TASK_DIR}/repo}"
COMMIT="c6c874bf7ea085beb04ea1487cfd216a0bacd6c1"

python -m pip install --upgrade pip
python -m pip install "torch>=2.2,<2.8" numpy pyyaml

if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone https://github.com/dmlc/dgl "${REPO_DIR}"
fi

git -C "${REPO_DIR}" fetch --tags --force origin
git -C "${REPO_DIR}" checkout "${COMMIT}"

python "${TASK_DIR}/verify.py" --artifact-dir "${TASK_DIR}/artifacts" --json
