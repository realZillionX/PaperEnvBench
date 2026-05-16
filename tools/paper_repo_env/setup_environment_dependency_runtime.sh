#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: setup_environment_dependency_runtime.sh [--skip-system] [--skip-torch] [--skip-probe]

Prepare the active PaperEnvBench runtime for native environment dependency probes.
Run this inside the target notebook and an activated venv when possible.
EOF
}

SKIP_SYSTEM=0
SKIP_TORCH=0
SKIP_PROBE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-system)
      SKIP_SYSTEM=1
      shift
      ;;
    --skip-torch)
      SKIP_TORCH=1
      shift
      ;;
    --skip-probe)
      SKIP_PROBE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

if [[ "$SKIP_SYSTEM" -eq 0 ]]; then
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "skip system packages: current user is not root" >&2
  elif command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential \
      python3.12-dev \
      cmake \
      ninja-build \
      pkg-config \
      git \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
      libsndfile1
  else
    echo "skip system packages: apt-get is unavailable" >&2
  fi
fi

if [[ "$SKIP_TORCH" -eq 0 ]]; then
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch \
    torchvision \
    torchaudio
fi

if [[ "$SKIP_PROBE" -eq 0 ]]; then
  "$PYTHON_BIN" tools/paper_repo_env/run_environment_dependency_suite.py \
    --profile native_python_build_runtime \
    --profile gpu_occupancy_guard \
    --profile torch_vision_audio_cuda_matrix \
    --json \
    --strict \
    --output-dir runs/environment_dependency_suite/runtime_setup
fi
