#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="${REPO_URL:-https://github.com/openai/gym}"
REPO_COMMIT="${REPO_COMMIT:-bc212954b6713d5db303b3ead124de6cba66063e}"
VENV_DIR="${VENV_DIR:-.venv}"
REPO_DIR="${REPO_DIR:-repo}"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip wheel setuptools
"$VENV_DIR/bin/python" -m pip install "numpy>=1.18.0" "cloudpickle>=1.2.0" "gym_notices>=0.0.4"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch --depth 1 origin "$REPO_COMMIT"
git -C "$REPO_DIR" checkout --detach "$REPO_COMMIT"
"$VENV_DIR/bin/python" -m pip install -e "$REPO_DIR"

"$VENV_DIR/bin/python" "$TASK_DIR/verify.py" --check-only --json
