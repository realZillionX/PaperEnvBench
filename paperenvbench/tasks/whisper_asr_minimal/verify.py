#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parent


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


def first_existing(root: Path, candidates: list[str], label: str) -> Path:
    for rel in candidates:
        path = root / rel
        if path.exists():
            return path
    joined = ", ".join(candidates)
    raise AssertionError(f"Missing {label}; checked: {joined}")


def nonempty_artifact(root: Path, candidates: list[str]) -> Path:
    existing = [root / rel for rel in candidates if (root / rel).exists()]
    for path in existing:
        if path.stat().st_size > 0:
            return path
    checked = ", ".join(str(path.relative_to(root)) for path in existing) or ", ".join(candidates)
    raise AssertionError(f"Missing nonempty side artifact; checked: {checked}")


def verify(attempt_root: Path) -> dict[str, Any]:
    attempt_root = attempt_root.resolve()
    json_path = first_existing(
        attempt_root,
        [
            "outputs/whisper_smoke/whisper_sine.json",
            "whisper_sine.json",
            "artifacts/expected_artifact.json",
            "artifacts/whisper_sine.json",
            "expected_artifact.json",
        ],
        "Whisper JSON output",
    )
    txt_path = first_existing(
        attempt_root,
        [
            "outputs/whisper_smoke/whisper_sine.txt",
            "whisper_sine.txt",
            "artifacts/expected_artifact.txt",
            "artifacts/whisper_sine.txt",
            "expected_artifact.txt",
        ],
        "Whisper TXT output",
    )
    artifact_path = nonempty_artifact(
        attempt_root,
        [
            "outputs/whisper_smoke/whisper_sine.tsv",
            "outputs/whisper_smoke/whisper_sine.vtt",
            "outputs/whisper_smoke/whisper_sine.srt",
            "artifacts/expected_artifact.tsv",
            "artifacts/expected_artifact.vtt",
            "artifacts/expected_artifact.srt",
            "artifacts/whisper_sine.tsv",
            "artifacts/whisper_sine.vtt",
            "artifacts/whisper_sine.srt",
            "expected_artifact.tsv",
            "expected_artifact.vtt",
            "expected_artifact.srt",
        ],
    )

    payload = load_json(json_path)
    text = payload.get("text")
    if not isinstance(text, str):
        raise AssertionError("JSON output must contain a string field named 'text'")
    segments = payload.get("segments")
    if segments is not None and not isinstance(segments, list):
        raise AssertionError("JSON field 'segments' must be a list when present")
    language = payload.get("language")
    if language is not None and not isinstance(language, str):
        raise AssertionError("JSON field 'language' must be a string when present")

    txt_text = txt_path.read_text(encoding="utf-8", errors="replace")
    if normalize_text(txt_text) != normalize_text(text):
        raise AssertionError("TXT transcript does not match JSON 'text' after whitespace normalization")

    return {
        "task_id": "whisper_asr_minimal",
        "status": "pass",
        "attempt_root": str(attempt_root),
        "checks": {
            "output_json_exists": str(json_path.relative_to(attempt_root)),
            "transcript_field_present": True,
            "txt_exists": str(txt_path.relative_to(attempt_root)),
            "txt_matches_json_text": True,
            "artifact_bundle_nonempty": str(artifact_path.relative_to(attempt_root)),
        },
        "observed": {
            "language": language,
            "text": text,
            "segments_count": len(segments) if isinstance(segments, list) else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the PaperEnvBench Whisper ASR minimal task output.")
    parser.add_argument(
        "attempt_root",
        nargs="?",
        default=str(TASK_ROOT),
        help="Attempt root containing outputs/whisper_smoke, or the task root containing artifacts/.",
    )
    parser.add_argument("--artifact-dir", type=Path, help="Artifact directory to validate directly.")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    args = parser.parse_args()

    try:
        result = verify(args.artifact_dir if args.artifact_dir else Path(args.attempt_root))
    except AssertionError as exc:
        failure = {"task_id": "whisper_asr_minimal", "status": "fail", "error": str(exc)}
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
