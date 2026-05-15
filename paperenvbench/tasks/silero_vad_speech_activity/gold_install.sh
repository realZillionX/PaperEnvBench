#!/usr/bin/env bash
set -euo pipefail

TASK_ID="silero_vad_speech_activity"
REPO_URL="https://github.com/snakers4/silero-vad.git"
COMMIT="980b17e9d56463e51393a8d92ded473f1b17896a"
ROOT_DIR="${PAPERENVBENCH_RUN_DIR:-$(pwd)}"
REPO_DIR="${PAPERENVBENCH_REPO_DIR:-$ROOT_DIR/repo}"
VENV_DIR="${PAPERENVBENCH_VENV_DIR:-$ROOT_DIR/.venv}"
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
python -m pip install --upgrade pip setuptools wheel

if ! python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0" "torchaudio==2.8.0"; then
  echo "CPU wheel index install failed; falling back to the configured pip index." >&2
  python -m pip install "torch==2.8.0" "torchaudio==2.8.0"
fi

python -m pip install -e "$REPO_DIR" "soundfile==0.13.1"
python -m pip freeze | sort > "$ROOT_DIR/requirements_lock.txt"

export PAPERENVBENCH_RUN_DIR="$ROOT_DIR"
python - <<'PY'
import os
import importlib.metadata as metadata
import pathlib
import torch
import torchaudio
import silero_vad

repo_dir = pathlib.Path(os.environ["PAPERENVBENCH_RUN_DIR"]) / "repo"
model_path = repo_dir / "src" / "silero_vad" / "data" / "silero_vad.jit"
audio_path = repo_dir / "tests" / "data" / "test.wav"

print("python_package=silero-vad", metadata.version("silero-vad"))
print("torch", torch.__version__)
print("torchaudio", torchaudio.__version__)
print("silero_vad_module", pathlib.Path(silero_vad.__file__).resolve())
print("packaged_jit_exists", model_path.exists(), model_path)
print("test_wav_exists", audio_path.exists(), audio_path)
PY
