#!/usr/bin/env bash
set -euo pipefail
REPO_URL="https://github.com/google-research/text-to-text-transfer-transformer.git"
REPO_COMMIT="90dcc718148715bd8e0045ca964e15dbcfba9a1d"
RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$PWD}"
REPO_DIR="${T5_REPO_DIR:-$RUN_ROOT/repo}"
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
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple 't5==0.9.4' 'tensorflow-cpu==2.16.2' 'sentencepiece==0.2.0' 'gin-config==0.5.0'
"$VENV_DIR/bin/python" -m pip freeze | sort > "$LOCK_OUT"
test -e "$REPO_DIR/t5/models/mtf_model.py"
test -e "$REPO_DIR/t5/data/dataset_providers.py"
test -e "$REPO_DIR/t5/data/tasks.py"
test -e "$REPO_DIR/t5/models/gin/models/t5.1.1.small.gin"
echo "t5_text_to_text_minimal gold install route verified"
