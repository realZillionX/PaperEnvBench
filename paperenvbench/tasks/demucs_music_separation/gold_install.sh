#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/facebookresearch/demucs.git"
REPO_COMMIT="e976d93ecc3865e5757426930257e200846a520a"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$PWD}"
REPO_DIR="${DEMUX_REPO_DIR:-$RUN_ROOT/demucs}"
VENV_DIR="${DEMUX_VENV_DIR:-$RUN_ROOT/venv}"
LOCK_OUT="${PAPERENVBENCH_LOCK_OUT:-$RUN_ROOT/requirements_lock.txt}"

echo "[gold_install] run_root=$RUN_ROOT"
echo "[gold_install] repo_dir=$REPO_DIR"
echo "[gold_install] venv_dir=$VENV_DIR"
echo "[gold_install] script_dir=$SCRIPT_DIR"

if ! command -v git >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg
fi

mkdir -p "$RUN_ROOT"
if [ ! -d "$REPO_DIR/.git" ]; then
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

"$VENV_DIR/bin/python" -m pip install \
  "torch==2.4.1" \
  "torchaudio==2.4.1"

"$VENV_DIR/bin/python" -m pip install \
  "dora-search==0.1.12" \
  "einops==0.8.2" \
  "julius==0.2.7" \
  "lameenc==1.8.2" \
  "openunmix==1.3.0" \
  "PyYAML==6.0.3" \
  "soundfile==0.13.1" \
  "tqdm==4.67.3"

"$VENV_DIR/bin/python" -m pip install --no-deps -e "$REPO_DIR"
"$VENV_DIR/bin/python" -m pip freeze > "$LOCK_OUT"
"$VENV_DIR/bin/python" - <<'PY'
import importlib.metadata
import torch
import torchaudio

print("demucs", importlib.metadata.version("demucs"))
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("torchaudio", torchaudio.__version__)
PY

ffmpeg -version | head -n 2
ffprobe -version | head -n 1
echo "[gold_install] wrote $LOCK_OUT"
