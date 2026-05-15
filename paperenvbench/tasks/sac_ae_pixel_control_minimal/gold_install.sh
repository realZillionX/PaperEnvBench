#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_DIR"

if [[ ! -d repo ]]; then
  git clone https://github.com/denisyarats/pytorch_sac_ae repo
fi
git -C repo fetch --depth 1 origin 7fa560e21c026c04bb8dcd72959ecf4e3424476c
git -C repo checkout 7fa560e21c026c04bb8dcd72959ecf4e3424476c

python -m pip install --upgrade pip
python -m pip install "torch>=2.0" numpy

mkdir -p artifacts logs
python verify.py --repo-dir repo --output-dir artifacts --json > logs/gold_verify.log 2>&1
python verify.py --check-only --json
