#!/usr/bin/env bash
set -euo pipefail

TASK_ID=gaussian_splatting_scene_minimal
COMMIT=54c035f7834b564019656c3e3fcc3646292f727d
REPO_URL=https://github.com/graphdeco-inria/gaussian-splatting
RUN_ROOT="${RUN_ROOT:-/inspire/ssd/project/embodied-multimodality/tongjingqi-CZXS25110029/Paper Reproduction/runs/paperenvbench/geometry/${TASK_ID}}"
REPO_DIR="${RUN_ROOT}/repo"

mkdir -p "${RUN_ROOT}"
echo "[gold_install] task=${TASK_ID}"
echo "[gold_install] run_root=${RUN_ROOT}"
echo "[gold_install] repo_url=${REPO_URL}"
echo "[gold_install] commit=${COMMIT}"

if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone --recursive "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"
git fetch origin "${COMMIT}"
git checkout --force "${COMMIT}"
git submodule update --init --recursive
git log -1 --format='[gold_install] checked_out=%H %ad %s' --date=iso-strict

cat <<'EOF'
[gold_install] full route:
  conda env create --file environment.yml
  conda activate gaussian_splatting
  python train.py -s <COLMAP_OR_NERF_SYNTHETIC_SCENE> -m <MODEL_DIR> --iterations <N>
  python render.py -m <MODEL_DIR>
  python metrics.py -m <MODEL_DIR>
[gold_install] expected CPU boundary:
  train.py and render.py allocate CUDA tensors and import diff_gaussian_rasterization/simple_knn native extensions.
  check-only gold uses deterministic CPU scene artifacts instead of claiming a CUDA rasterizer run.
EOF
