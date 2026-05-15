#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
WHISPER_REPO="${WHISPER_REPO:-${REPO_ROOT}/repos/whisper}"
VENV="${VENV:-${REPO_ROOT}/envs/paper-repro-cpu}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/whisper_smoke}"
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/models/whisper}"
AUDIO_PATH="${AUDIO_PATH:-${OUT_DIR}/whisper_sine.wav}"

if [[ ! -d "${WHISPER_REPO}/.git" ]]; then
  echo "Missing Whisper checkout: ${WHISPER_REPO}" >&2
  exit 2
fi

commit="$(git -C "${WHISPER_REPO}" rev-parse --short=7 HEAD)"
if [[ "${commit}" != "04f449b" ]]; then
  echo "Unexpected Whisper commit: ${commit}; expected 04f449b" >&2
  exit 2
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Missing system dependency: ffmpeg" >&2
  exit 2
fi

python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV}/bin/python" -m pip install -e "${WHISPER_REPO}"

mkdir -p "${OUT_DIR}" "${MODEL_DIR}"
export AUDIO_PATH
"${VENV}/bin/python" - <<'PY'
import math
import struct
import wave
from pathlib import Path

out = Path(__import__("os").environ["AUDIO_PATH"])
out.parent.mkdir(parents=True, exist_ok=True)
sample_rate = 16000
duration_s = 1.0
freq = 440.0
with wave.open(str(out), "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)
    for idx in range(int(sample_rate * duration_s)):
        sample = int(0.15 * 32767 * math.sin(2 * math.pi * freq * idx / sample_rate))
        wav.writeframes(struct.pack("<h", sample))
PY

"${VENV}/bin/whisper" "${AUDIO_PATH}" \
  --model tiny \
  --model_dir "${MODEL_DIR}" \
  --language en \
  --device cpu \
  --output_dir "${OUT_DIR}" \
  --output_format all

ls -l "${OUT_DIR}"/whisper_sine.*
