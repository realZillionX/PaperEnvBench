#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/facebookresearch/encodec.git"
REPO_COMMIT="0e2d0aed29362c8e8f52494baf3e6f99056b214f"

cd "$TASK_DIR"
mkdir -p src logs artifacts

if [ ! -d src/encodec/.git ]; then
  rm -rf src/encodec
  git clone "$REPO_URL" src/encodec
fi

git -C src/encodec fetch --tags origin
git -C src/encodec checkout --detach "$REPO_COMMIT"

rm -rf .venv
python3 -m venv .venv
. .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ./src/encodec
python -m pip check
python -m pip freeze | sort > requirements_lock.txt

python - <<'PY'
import encodec
import einops
import numpy
import torch
import torchaudio

print("encodec_file", encodec.__file__)
print("encodec_version", getattr(encodec, "__version__", "unknown"))
print("torch", torch.__version__)
print("torchaudio", torchaudio.__version__)
print("numpy", numpy.__version__)
print("einops", einops.__version__)
PY
