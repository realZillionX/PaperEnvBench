#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/microsoft/LoRA.git"
REPO_COMMIT="c4593f060e6a368d7bb5af5273b8e42810cdef90"
RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$PWD}"
REPO_DIR="${LORA_ADAPTER_MINIMAL_REPO_DIR:-$RUN_ROOT/repo}"
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
"$VENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple 'torch==2.8.0'
"$VENV_DIR/bin/python" -m pip install "$REPO_DIR"
"$VENV_DIR/bin/python" -m pip freeze | sort > "$LOCK_OUT"
test -e "$REPO_DIR/setup.py"
test -e "$REPO_DIR/loralib/__init__.py"
test -e "$REPO_DIR/loralib/layers.py"
test -e "$REPO_DIR/loralib/utils.py"
"$VENV_DIR/bin/python" - <<'PY'
import loralib as lora
required = ["Linear", "MergedLinear", "mark_only_lora_as_trainable", "lora_state_dict"]
missing = [name for name in required if not hasattr(lora, name)]
if missing:
    raise SystemExit(f"missing loralib API: {missing}")
PY
echo "lora_adapter_minimal gold install route verified"
