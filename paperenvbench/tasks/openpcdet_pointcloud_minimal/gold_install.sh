#!/usr/bin/env bash
set -Eeuo pipefail

TASK_ID="openpcdet_pointcloud_minimal"
REPO_URL="https://github.com/open-mmlab/OpenPCDet.git"
COMMIT="233f849829b6ac19afb8af8837a0246890908755"
ROOT_DIR="${PAPERENVBENCH_RUN_DIR:-$(pwd)}"
REPO_DIR="${PAPERENVBENCH_REPO_DIR:-$ROOT_DIR/repo}"
VENV_DIR="${PAPERENVBENCH_VENV_DIR:-$ROOT_DIR/venv}"
LOG_DIR="$ROOT_DIR/logs"

mkdir -p "$LOG_DIR" "$ROOT_DIR/artifacts"

echo "task_id=$TASK_ID"
echo "root_dir=$ROOT_DIR"
echo "repo_dir=$REPO_DIR"
echo "venv_dir=$VENV_DIR"
echo "commit=$COMMIT"

if [ ! -d "$REPO_DIR/.git" ]; then
  rm -rf "$REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
fi

git -C "$REPO_DIR" fetch --all --tags --prune
git -C "$REPO_DIR" checkout "$COMMIT"
git -C "$REPO_DIR" rev-parse HEAD

python3 -m venv --clear "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip "setuptools<82" wheel

if ! python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0"; then
  echo "CPU wheel index install failed; falling back to the configured pip index." >&2
  python -m pip install "torch==2.8.0"
fi

python -m pip install "numpy==1.26.4" "pyyaml==6.0.3"

set +e
python -m pip install -e "$REPO_DIR" > "$LOG_DIR/native_editable_install_probe.log" 2>&1
native_install_status=$?
set -e
echo "native_editable_install_status=$native_install_status"
if [ "$native_install_status" -eq 0 ]; then
  echo "native editable install unexpectedly succeeded; verifier still records CUDA extension status." >&2
else
  echo "native editable install failed as expected on CPU-only fallback; see native_editable_install_probe.log." >&2
fi

PYTHONPATH="$REPO_DIR" python - <<'PY' > "$LOG_DIR/pointpillar_config_probe.log" 2>&1
from pathlib import Path
import yaml

repo_dir = Path(__import__("os").environ["PAPERENVBENCH_REPO_DIR"])
cfg = yaml.safe_load((repo_dir / "tools/cfgs/kitti_models/pointpillar.yaml").read_text())
print("model", cfg["MODEL"]["NAME"])
print("classes", cfg["CLASS_NAMES"])
print("voxel_size", cfg["DATA_CONFIG"]["DATA_PROCESSOR"][2]["VOXEL_SIZE"])
PY

PYTHONPATH="$REPO_DIR" python "$ROOT_DIR/verify.py" --repo-dir "$REPO_DIR" --output-dir "$ROOT_DIR/artifacts" --json
python -m pip freeze | sort > "$ROOT_DIR/requirements_lock.txt"
{
  echo "# Source repository is used through PYTHONPATH and direct source-file loading."
  echo "# repo=$REPO_URL"
  echo "# commit=$COMMIT"
  echo "# native_editable_install_status=$native_install_status"
} >> "$ROOT_DIR/requirements_lock.txt"
