#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/facebookresearch/moco.git"
REPO_COMMIT="7397dfe146c7ca6bbb58e9c382498069178ba764"
CURRENT_EMPTY_HEAD="8976944da9c6b94cbd9158d7ebe50912aef807ef"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$PWD}"
REPO_DIR="${PAPERENVBENCH_REPO_DIR:-$RUN_ROOT/repo}"
VENV_DIR="${PAPERENVBENCH_VENV_DIR:-$RUN_ROOT/venv}"
LOCK_OUT="${PAPERENVBENCH_LOCK_OUT:-$RUN_ROOT/requirements_lock.txt}"

echo "[gold_install] run_root=$RUN_ROOT"
echo "[gold_install] repo_dir=$REPO_DIR"
echo "[gold_install] venv_dir=$VENV_DIR"
echo "[gold_install] script_dir=$SCRIPT_DIR"
echo "[gold_install] pinned_commit=$REPO_COMMIT"
echo "[gold_install] current_empty_head=$CURRENT_EMPTY_HEAD"

if ! command -v git >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates
fi

mkdir -p "$RUN_ROOT"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch --all --tags --prune
git -C "$REPO_DIR" checkout "$REPO_COMMIT"
git -C "$REPO_DIR" rev-parse HEAD
echo "[gold_install] tree_file_count=$(git -C "$REPO_DIR" ls-tree -r --name-only HEAD | wc -l | tr -d ' ')"

rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip config set global.index-url http://nexus.sii.shaipower.online/repository/pypi/simple || true
"$VENV_DIR/bin/python" -m pip config set global.trusted-host nexus.sii.shaipower.online || true

"$VENV_DIR/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.8.0" \
  "torchvision==0.23.0"

"$VENV_DIR/bin/python" -m pip freeze > "$LOCK_OUT"
"$VENV_DIR/bin/python" - <<'PY'
import sys
import torch
import torchvision

print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
PY

echo "[gold_install] wrote $LOCK_OUT"
