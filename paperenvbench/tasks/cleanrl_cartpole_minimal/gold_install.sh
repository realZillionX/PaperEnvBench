#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${PAPERENVBENCH_REPO_DIR:-${TASK_DIR}/repo}"
COMMIT="fe8d8a03c41a7ef5b523e2e354bd01c363e786bb"

python -m pip install --upgrade pip
python -m pip install "gymnasium==0.29.1" "torch>=2.2,<2.8" tyro tensorboard numpy

if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone https://github.com/vwxyzjn/cleanrl "${REPO_DIR}"
fi

git -C "${REPO_DIR}" fetch --tags --force origin
git -C "${REPO_DIR}" checkout "${COMMIT}"

test -f "${REPO_DIR}/cleanrl/ppo.py"
grep -q "env_id: str = \"CartPole-v1\"" "${REPO_DIR}/cleanrl/ppo.py"
grep -q "gymnasium==0.29.1" "${REPO_DIR}/pyproject.toml"

python "${TASK_DIR}/verify.py" --artifact-dir "${TASK_DIR}/artifacts" --json
