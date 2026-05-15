#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/facebookresearch/fastText.git"
REPO_COMMIT="1142dc4c4ecbc19cc16eee5cdd28472e689267e6"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$PWD}"
REPO_DIR="${FASTTEXT_REPO_DIR:-$RUN_ROOT/fastText}"
VENV_DIR="${FASTTEXT_VENV_DIR:-$RUN_ROOT/venv}"
LOCK_OUT="${PAPERENVBENCH_LOCK_OUT:-$RUN_ROOT/requirements_lock.txt}"
DATA_DIR="${FASTTEXT_DATA_DIR:-$RUN_ROOT/data}"

echo "[gold_install] run_root=$RUN_ROOT"
echo "[gold_install] repo_dir=$REPO_DIR"
echo "[gold_install] venv_dir=$VENV_DIR"
echo "[gold_install] script_dir=$SCRIPT_DIR"

if ! command -v git >/dev/null 2>&1 || ! command -v make >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git ca-certificates make g++
fi

mkdir -p "$RUN_ROOT" "$DATA_DIR"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch --all --tags --prune
git -C "$REPO_DIR" checkout "$REPO_COMMIT"
git -C "$REPO_DIR" rev-parse HEAD

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install "pybind11==2.13.6" "numpy==2.2.6"

make -C "$REPO_DIR" -j "${FASTTEXT_MAKE_JOBS:-2}"
"$REPO_DIR/fasttext" supervised -h | head -n 20
"$VENV_DIR/bin/python" -m pip install --no-deps "$REPO_DIR"
"$VENV_DIR/bin/python" -m pip freeze > "$LOCK_OUT"

cat > "$DATA_DIR/train.txt" <<'DATA'
__label__sports team wins match with strong defense
__label__sports player scores goal in football game
__label__sports championship match ends with late goal
__label__sports coach praises team after win
__label__tech software release adds python api
__label__tech neural network model trains on cpu
__label__tech database query uses fast index
__label__tech compiler builds native extension
DATA

cat > "$DATA_DIR/test.txt" <<'DATA'
__label__sports football team scores goal
__label__tech python software compiler builds extension
DATA

"$REPO_DIR/fasttext" supervised \
  -input "$DATA_DIR/train.txt" \
  -output "$DATA_DIR/model" \
  -epoch 35 \
  -lr 0.5 \
  -wordNgrams 2 \
  -dim 16 \
  -minCount 1 \
  -thread 1 \
  -seed 0 \
  -loss softmax
"$REPO_DIR/fasttext" test "$DATA_DIR/model.bin" "$DATA_DIR/test.txt"
printf 'football team wins match\npython compiler builds software\n' | "$REPO_DIR/fasttext" predict-prob "$DATA_DIR/model.bin" - 2
"$VENV_DIR/bin/python" - "$DATA_DIR/model.bin" <<'PY'
import fasttext
import sys

model = fasttext.load_model(sys.argv[1])
print("python_api_labels", ",".join(model.labels))
print("python_api_prediction", model.predict("football team wins match", k=2))
PY
echo "[gold_install] wrote $LOCK_OUT"
