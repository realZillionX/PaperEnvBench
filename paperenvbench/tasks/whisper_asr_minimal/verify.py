#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parent
TASK_ID = "whisper_asr_minimal"
REPO_COMMIT = "04f449b8a437f1bbd3dba5c9f826aca972e7709a"
CHECKPOINT_SHA = "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03"
INPUT_AUDIO_SHA = "63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715"
EXPECTED_TERMS = [
    "fellow americans",
    "what your country can do for you",
    "what you can do for your country",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssertionError(f"JSON output is not parseable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AssertionError(f"JSON output must be an object: {path}")
    return payload


def require_file(root: Path, name: str, *, nonempty: bool = True) -> Path:
    path = root / name
    if not path.exists():
        raise AssertionError(f"missing required artifact: {name}")
    if nonempty and path.stat().st_size <= 0:
        raise AssertionError(f"artifact must be nonempty: {name}")
    return path


def require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise AssertionError(f"{label} mismatch: expected {expected!r}, got {observed!r}")


def require_cuda(metadata: dict[str, Any]) -> dict[str, Any]:
    require_equal(metadata.get("device"), "cuda", "device")
    torch_info = metadata.get("torch")
    if not isinstance(torch_info, dict):
        raise AssertionError("run_metadata.torch must be an object")
    if torch_info.get("cuda_available") is not True:
        raise AssertionError("torch.cuda_available must be true")
    torch_cuda = torch_info.get("torch_cuda")
    if not isinstance(torch_cuda, str) or not torch_cuda.startswith("12."):
        raise AssertionError(f"torch_cuda must record CUDA 12.x, got {torch_cuda!r}")
    gpu_name = str(torch_info.get("gpu_name") or "")
    if "4090" not in gpu_name:
        raise AssertionError(f"gpu_name must identify the 4090 runtime, got {gpu_name!r}")
    return torch_info


def verify(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    json_path = require_file(artifact_dir, "expected_artifact.json")
    txt_path = require_file(artifact_dir, "expected_artifact.txt")
    metadata_path = require_file(artifact_dir, "run_metadata.json")
    input_audio_path = require_file(artifact_dir, "input_jfk.flac")
    side_paths = [
        require_file(artifact_dir, "expected_artifact.tsv"),
        require_file(artifact_dir, "expected_artifact.vtt"),
        require_file(artifact_dir, "expected_artifact.srt"),
    ]

    payload = load_json(json_path)
    metadata = load_json(metadata_path)

    require_equal(metadata.get("task_id"), TASK_ID, "task_id")
    require_equal(metadata.get("repo_commit"), REPO_COMMIT, "repo_commit")
    require_equal(metadata.get("entrypoint"), "whisper CLI", "entrypoint")
    require_equal(metadata.get("model_name"), "tiny.en", "model_name")
    require_equal(metadata.get("checkpoint_sha256"), CHECKPOINT_SHA, "checkpoint_sha256")
    require_equal(metadata.get("input_audio_sha256"), INPUT_AUDIO_SHA, "input_audio_sha256")
    require_equal(sha256(input_audio_path), INPUT_AUDIO_SHA, "input audio sha256")
    require_cuda(metadata)

    checkpoint_size = metadata.get("checkpoint_size_bytes")
    if not isinstance(checkpoint_size, int) or checkpoint_size < 70_000_000:
        raise AssertionError("checkpoint_size_bytes must prove the tiny.en checkpoint was cached")
    audio_duration = metadata.get("audio_duration_seconds")
    if not isinstance(audio_duration, (int, float)) or not math.isfinite(float(audio_duration)) or not 10.0 <= float(audio_duration) <= 12.0:
        raise AssertionError(f"audio_duration_seconds must describe the JFK sample, got {audio_duration!r}")

    text = payload.get("text")
    if not isinstance(text, str) or not normalize_text(text):
        raise AssertionError("JSON output must contain a nonempty string field named 'text'")
    normalized_lower = normalize_text(text).lower()
    missing_terms = [term for term in EXPECTED_TERMS if term not in normalized_lower]
    if missing_terms:
        raise AssertionError(f"transcript is missing expected JFK phrases: {missing_terms}")

    segments = payload.get("segments")
    if not isinstance(segments, list) or len(segments) < 1:
        raise AssertionError("JSON output must contain at least one segment")
    for idx, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise AssertionError(f"segment {idx} must be an object")
        if not isinstance(segment.get("text"), str) or not normalize_text(segment["text"]):
            raise AssertionError(f"segment {idx} must contain nonempty text")
        start = segment.get("start")
        end = segment.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or float(end) <= float(start):
            raise AssertionError(f"segment {idx} must contain increasing start/end timestamps")
    require_equal(metadata.get("segments_count"), len(segments), "segments_count")
    require_equal(payload.get("language"), "en", "language")
    require_equal(metadata.get("language"), "en", "metadata.language")
    if normalize_text(txt_path.read_text(encoding="utf-8", errors="replace")) != normalize_text(text):
        raise AssertionError("TXT transcript does not match JSON text after whitespace normalization")

    output_files = metadata.get("output_files")
    if not isinstance(output_files, dict):
        raise AssertionError("run_metadata.output_files must be an object")
    for path in [json_path, txt_path, *side_paths]:
        entry = output_files.get(path.name)
        if not isinstance(entry, dict):
            raise AssertionError(f"missing output_files entry for {path.name}")
        require_equal(entry.get("sha256"), sha256(path), f"{path.name} sha256")
        require_equal(entry.get("size_bytes"), path.stat().st_size, f"{path.name} size_bytes")

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": "L4",
        "artifact_dir": str(artifact_dir),
        "checks": {
            "output_json_exists": True,
            "whisper_help_and_cli_inference_evidence": True,
            "tiny_en_checkpoint_sha256_matches": True,
            "jfk_audio_sha256_matches": True,
            "cuda_device_evidence_present": True,
            "transcript_nonempty_with_expected_phrases": True,
            "transcript_segments_present": True,
            "side_artifact_bundle_nonempty": True,
        },
        "observed": {
            "language": payload.get("language"),
            "segments_count": len(segments),
            "text": normalize_text(text),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            "input_audio_sha256": metadata.get("input_audio_sha256"),
            "device": metadata.get("device"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the PaperEnvBench Whisper ASR minimal task output.")
    parser.add_argument("attempt_root", nargs="?", default=str(TASK_ROOT))
    parser.add_argument("--artifact-dir", type=Path, help="Artifact directory to validate directly.")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    args = parser.parse_args()

    artifact_dir = args.artifact_dir if args.artifact_dir else Path(args.attempt_root) / "artifacts"
    if args.artifact_dir is None and not artifact_dir.exists():
        artifact_dir = Path(args.attempt_root)
    try:
        result = verify(artifact_dir)
    except AssertionError as exc:
        failure = {"task_id": TASK_ID, "status": "fail", "error": str(exc)}
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("PASS whisper_asr_minimal")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
