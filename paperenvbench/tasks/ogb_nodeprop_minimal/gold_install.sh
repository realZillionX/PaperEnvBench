#!/usr/bin/env bash
set -euo pipefail

TASK_ID="ogb_nodeprop_minimal"
REPO_URL="https://github.com/snap-stanford/ogb.git"
REPO_COMMIT="61e9784ca76edeaa6e259ba0f836099608ff0586"

ROOT_DIR="${PAPERENV_WORKDIR:-$PWD}"
RUN_DIR="${ROOT_DIR}/runs/${TASK_ID}"
REPO_DIR="${RUN_DIR}/repo"
VENV_DIR="${RUN_DIR}/venv"

mkdir -p "${RUN_DIR}"
if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi
git -C "${REPO_DIR}" fetch --depth 1 origin "${REPO_COMMIT}"
git -C "${REPO_DIR}" checkout --detach "${REPO_COMMIT}"

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu
"${VENV_DIR}/bin/python" -m pip install -e "${REPO_DIR}"

"${VENV_DIR}/bin/python" - <<'PY'
import numpy as np
from ogb.nodeproppred import Evaluator

evaluator = Evaluator(name="ogbn-arxiv")
y_true = np.array([[0], [1], [2], [2], [1], [0], [3], [3], [4], [4]], dtype=np.int64)
y_pred = np.array([[0], [1], [2], [2], [1], [0], [3], [0], [4], [2]], dtype=np.int64)
result = evaluator.eval({"y_true": y_true, "y_pred": y_pred})
assert abs(result["acc"] - 0.8) < 1e-12, result
print({"dataset": "ogbn-arxiv", "metric": "acc", "acc": float(result["acc"])})
PY
