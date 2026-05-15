#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/facebookresearch/fairseq.git"
REPO_COMMIT="3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99"
RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$PWD}"
REPO_DIR="${FAIRSEQ_TRANSLATION_MINIMAL_REPO_DIR:-$RUN_ROOT/repo}"
VENV_DIR="${PAPERENVBENCH_ENV_DIR:-$RUN_ROOT/venv}"
LOCK_OUT="${PAPERENVBENCH_LOCK_OUT:-$RUN_ROOT/requirements_lock.txt}"
PYTHON_BIN="${PAPERENVBENCH_PYTHON:-python3}"

mkdir -p "$RUN_ROOT"
if [ ! -d "$REPO_DIR/.git" ]; then
  rm -rf "$REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch --depth 1 origin "$REPO_COMMIT"
git -C "$REPO_DIR" checkout --detach "$REPO_COMMIT"
git -C "$REPO_DIR" rev-parse HEAD

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel cython
"$VENV_DIR/bin/python" -m pip install --index-url https://pypi.org/simple 'torch==2.8.0' 'numpy<2' 'regex' 'sacrebleu==2.4.3' 'bitarray' 'hydra-core<1.1' 'omegaconf<2.1'
"$VENV_DIR/bin/python" -m pip install --no-build-isolation --editable "$REPO_DIR"
"$VENV_DIR/bin/python" -m pip freeze | sort > "$LOCK_OUT"
test -e "$REPO_DIR/fairseq_cli/generate.py"
test -e "$REPO_DIR/fairseq/tasks/translation.py"
test -e "$REPO_DIR/fairseq/models/transformer.py"
test -e "$REPO_DIR/examples/translation/README.md"
"$VENV_DIR/bin/python" -m fairseq_cli.generate --help >/dev/null
echo "fairseq_translation_minimal gold install route verified"
