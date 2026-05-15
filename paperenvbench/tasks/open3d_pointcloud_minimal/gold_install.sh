#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/isl-org/Open3D.git"
REPO_COMMIT="1e7b17438687a0b0c1e5a7187321ac7044afe275"

RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$PWD}"
REPO_DIR="${OPEN3D_REPO_DIR:-$RUN_ROOT/repo}"
VENV_DIR="${OPEN3D_VENV_DIR:-$RUN_ROOT/venv}"
LOCK_OUT="${PAPERENVBENCH_LOCK_OUT:-$RUN_ROOT/requirements_lock.txt}"
ARTIFACT_DIR="${PAPERENVBENCH_ARTIFACT_DIR:-$RUN_ROOT/artifacts}"

echo "[gold_install] run_root=$RUN_ROOT"
echo "[gold_install] repo_dir=$REPO_DIR"
echo "[gold_install] venv_dir=$VENV_DIR"
echo "[gold_install] commit=$REPO_COMMIT"

if ! command -v git >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates
fi

mkdir -p "$RUN_ROOT"
if [ ! -d "$REPO_DIR/.git" ]; then
  rm -rf "$REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch --all --tags --prune
git -C "$REPO_DIR" checkout "$REPO_COMMIT"
git -C "$REPO_DIR" rev-parse HEAD

rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip config set global.index-url http://nexus.sii.shaipower.online/repository/pypi/simple || true
"$VENV_DIR/bin/python" -m pip config set global.trusted-host nexus.sii.shaipower.online || true

if ! "$VENV_DIR/bin/python" -m pip install "open3d==0.19.0" "numpy==2.2.6"; then
  echo "[gold_install] configured pip index failed; falling back to PyPI." >&2
  "$VENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple "open3d==0.19.0" "numpy==2.2.6"
fi

"$VENV_DIR/bin/python" -m pip freeze | sort > "$LOCK_OUT"
"$VENV_DIR/bin/python" "$RUN_ROOT/verify.py" --repo-dir "$REPO_DIR" --output-dir "$ARTIFACT_DIR" --json
"$VENV_DIR/bin/python" "$RUN_ROOT/verify.py" --artifact-dir "$ARTIFACT_DIR" --check-only --json

echo "[gold_install] wrote $LOCK_OUT"
