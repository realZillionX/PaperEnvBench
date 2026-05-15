#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/RUCAIBox/RecBole.git"
REPO_COMMIT="7b02be5ec80a88310f2d04a27a82adfcbb5dc211"
RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$PWD}"
REPO_DIR="${RECBOLE_SEQUENTIAL_MINIMAL_REPO_DIR:-$RUN_ROOT/repo}"
VENV_DIR="${PAPERENVBENCH_ENV_DIR:-$RUN_ROOT/venv}"
LOCK_OUT="${PAPERENVBENCH_LOCK_OUT:-$RUN_ROOT/requirements_lock.txt}"
PYTHON_BIN="${PAPERENVBENCH_PYTHON:-python3}"

mkdir -p "$RUN_ROOT"
if [ ! -d "$REPO_DIR/.git" ]; then
  rm -rf "$REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch --depth 1 origin "$REPO_COMMIT"
git -C "$REPO_DIR" checkout --detach "$REPO_COMMIT"
git -C "$REPO_DIR" rev-parse HEAD

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple 'torch==2.8.0' 'numpy<2' 'scipy' 'pandas' 'scikit-learn' 'pyyaml' 'colorlog==4.7.2' 'colorama==0.4.4' 'tqdm' 'tabulate' 'texttable' 'psutil' 'thop' 'tensorboard' 'plotly'
"$VENV_DIR/bin/python" -m pip install --no-deps --editable "$REPO_DIR"
"$VENV_DIR/bin/python" -m pip freeze | sort > "$LOCK_OUT"
test -e "$REPO_DIR/run_recbole.py"
test -e "$REPO_DIR/recbole/quick_start/quick_start.py"
test -e "$REPO_DIR/recbole/data/dataset/sequential_dataset.py"
test -e "$REPO_DIR/recbole/model/sequential_recommender/sasrec.py"
test -e "$REPO_DIR/recbole/properties/quick_start_config/sequential.yaml"
"$VENV_DIR/bin/python" "$REPO_DIR/run_recbole.py" --help >/dev/null
echo "recbole_sequential_minimal gold install route verified"
