#!/usr/bin/env bash
set -euo pipefail

TASK_ID="sentence_transformers_similarity_minimal"
REPO_URL="https://github.com/UKPLab/sentence-transformers.git"
REPO_COMMIT="5b27be706546f5e094e0f506d8593250e9a37109"
MODEL_ID="sentence-transformers/paraphrase-MiniLM-L3-v2"

ROOT_DIR="${PAPERENV_WORKDIR:-$PWD}"
RUN_DIR="${ROOT_DIR}/runs/${TASK_ID}"
REPO_DIR="${RUN_DIR}/repo"
VENV_DIR="${RUN_DIR}/venv"

mkdir -p "${RUN_DIR}"
if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi
git -C "${REPO_DIR}" fetch --depth 1 origin "${REPO_COMMIT}"
git -C "${REPO_DIR}" checkout --detach "${REPO_COMMIT}"

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -e "${REPO_DIR}"

"${VENV_DIR}/bin/python" - <<'PY'
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

model_id = "sentence-transformers/paraphrase-MiniLM-L3-v2"
sentences = [
    "A person plays guitar on a small stage.",
    "Someone is playing a musical instrument for an audience.",
    "The stock market index fell after a weak earnings report.",
]
model = SentenceTransformer(model_id, device="cpu")
embeddings = model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
scores = util.cos_sim(embeddings, embeddings)
assert float(scores[0, 1]) > float(scores[0, 2])
print({"model": model_id, "positive": float(scores[0, 1]), "negative": float(scores[0, 2])})
PY
