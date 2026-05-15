#!/usr/bin/env bash
set -euo pipefail

if [[ "${PAPERENVBENCH_CAPTURE_LOG:-1}" == "1" ]]; then
  mkdir -p logs
  PAPERENVBENCH_CAPTURE_LOG=0 "$0" "$@" 2>&1 | tee logs/gold_install.log
  exit "${PIPESTATUS[0]}"
fi

REPO_URL="https://github.com/speechbrain/speechbrain"
COMMIT="8a89ebad72af734b75bbd37565ae96a6819e146b"
PYTORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
PYPI_INDEX="https://pypi.org/simple"

echo "[gold_install] start $(date -Is)"

if [[ ! -d speechbrain/.git ]]; then
  git clone "${REPO_URL}" speechbrain
fi

git -C speechbrain fetch --depth 1 origin "${COMMIT}"
git -C speechbrain checkout --detach "${COMMIT}"

rm -rf .venv
python3 -m venv .venv
. .venv/bin/activate

python -m pip install -U --index-url "${PYPI_INDEX}" pip setuptools wheel
python -m pip install --index-url "${PYTORCH_CPU_INDEX}" \
  torch==2.11.0+cpu torchaudio==2.11.0+cpu
python -m pip install --index-url "${PYPI_INDEX}" -e speechbrain

python - <<'PY'
import huggingface_hub
import soundfile
import speechbrain
import torch
import torchaudio

print("speechbrain", getattr(speechbrain, "__version__", "unknown"))
print("torch", torch.__version__)
print("torchaudio", torchaudio.__version__)
print("soundfile", soundfile.__version__)
print("huggingface_hub", huggingface_hub.__version__)
PY

python -m pip freeze --all > requirements_lock.txt
echo "[gold_install] done $(date -Is)"
