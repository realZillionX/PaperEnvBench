#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/google-research/bert.git"
REPO_COMMIT="eedf5716ce1268e56f0a50264a88cafad334ac61"
RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$PWD}"
REPO_DIR="${BERT_MASKED_LM_MINIMAL_REPO_DIR:-$RUN_ROOT/repo}"
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
"$VENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple 'tensorflow-cpu==2.16.2'
"$VENV_DIR/bin/python" -m pip freeze | sort > "$LOCK_OUT"
test -e "$REPO_DIR/requirements.txt"
test -e "$REPO_DIR/run_pretraining.py"
test -e "$REPO_DIR/modeling.py"
test -e "$REPO_DIR/tokenization.py"
test -e "$REPO_DIR/create_pretraining_data.py"
test -e "$REPO_DIR/extract_features.py"
echo "bert_masked_lm_minimal gold install route verified"
