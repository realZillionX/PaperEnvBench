#!/usr/bin/env bash
set -Eeuo pipefail

RUN_ROOT="${PAPERENVBENCH_RUN_ROOT:-$(pwd)}"
REPO_DIR="${WHISPER_REPO_DIR:-${RUN_ROOT}/repo}"
ENV_DIR="${PAPERENVBENCH_ENV_DIR:-${RUN_ROOT}/.venv}"
MODEL_DIR="${WHISPER_MODEL_DIR:-${RUN_ROOT}/models}"
ARTIFACT_DIR="${PAPERENVBENCH_ARTIFACT_DIR:-${RUN_ROOT}/artifacts}"
COMMIT="04f449b8a437f1bbd3dba5c9f826aca972e7709a"

mkdir -p "${RUN_ROOT}" "${MODEL_DIR}" "${ARTIFACT_DIR}" "${RUN_ROOT}/logs"
export RUN_ROOT REPO_DIR MODEL_DIR ARTIFACT_DIR

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone https://github.com/openai/whisper "${REPO_DIR}"
fi
git -C "${REPO_DIR}" fetch --depth 1 origin "${COMMIT}"
git -C "${REPO_DIR}" checkout --detach "${COMMIT}"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${COMMIT}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Missing system dependency: ffmpeg" >&2
  exit 2
fi

python3 -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools==70.2.0 wheel
"${ENV_DIR}/bin/python" -m pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url "${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
"${ENV_DIR}/bin/python" -m pip install -e "${REPO_DIR}" --no-build-isolation

"${ENV_DIR}/bin/whisper" --help > "${RUN_ROOT}/logs/whisper_help.log"
"${ENV_DIR}/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "L4 Whisper route requires CUDA"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY

cp "${REPO_DIR}/tests/jfk.flac" "${ARTIFACT_DIR}/input_jfk.flac"
"${ENV_DIR}/bin/whisper" "${ARTIFACT_DIR}/input_jfk.flac" \
  --model tiny.en \
  --model_dir "${MODEL_DIR}" \
  --language en \
  --device cuda \
  --output_dir "${RUN_ROOT}/outputs/whisper_jfk" \
  --output_format all | tee "${RUN_ROOT}/logs/whisper_transcribe.log"

for suffix in json txt tsv vtt srt; do
  first="$(find "${RUN_ROOT}/outputs/whisper_jfk" -maxdepth 1 -name "*.${suffix}" | sort | head -n 1)"
  test -n "${first}"
  cp "${first}" "${ARTIFACT_DIR}/expected_artifact.${suffix}"
done

"${ENV_DIR}/bin/python" - <<'PY'
import hashlib
import json
import os
import subprocess
import torch
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
repo_dir = Path(os.environ["REPO_DIR"])
artifact_dir = Path(os.environ["ARTIFACT_DIR"])
model_dir = Path(os.environ["MODEL_DIR"])

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

payload = json.loads((artifact_dir / "expected_artifact.json").read_text(encoding="utf-8"))
checkpoint = next(model_dir.glob("*.pt"))
probe = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(artifact_dir / "input_jfk.flac")],
    text=True,
    capture_output=True,
    check=True,
)
metadata = {
    "task_id": "whisper_asr_minimal",
    "repo_commit": subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip(),
    "entrypoint": "whisper CLI",
    "command": "whisper input_jfk.flac --model tiny.en --language en --device cuda --output_format all",
    "model_name": "tiny.en",
    "checkpoint_sha256": sha(checkpoint),
    "checkpoint_size_bytes": checkpoint.stat().st_size,
    "input_audio": "artifacts/input_jfk.flac",
    "input_audio_sha256": sha(artifact_dir / "input_jfk.flac"),
    "audio_duration_seconds": float(probe.stdout.strip()),
    "device": "cuda",
    "torch": {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
    },
    "text": payload.get("text", ""),
    "language": payload.get("language"),
    "segments_count": len(payload.get("segments") or []),
    "segments": payload.get("segments") or [],
    "output_files": {
        p.name: {"sha256": sha(p), "size_bytes": p.stat().st_size}
        for p in sorted(artifact_dir.glob("expected_artifact.*"))
    },
}
(artifact_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
