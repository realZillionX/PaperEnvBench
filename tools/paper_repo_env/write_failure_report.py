#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FAILURE_PATTERNS: list[tuple[str, str]] = [
    (r"ModuleNotFoundError|ImportError", "hidden_dependency"),
    (r"CUDA|torchvision|torchaudio|torch\b", "torch_cuda_matrix"),
    (r"ffmpeg|libsndfile|libGL|glib|libjpeg|libpng", "system_package_missing"),
    (r"nvcc|cmake|ninja|gcc|CUDAExtension|cpp_extension", "native_extension_build"),
    (r"checkpoint|weights|\.pth|\.pt|download", "checkpoint_download"),
    (r"dataset|COCO|ImageNet|Kaggle", "dataset_asset_missing"),
    (r"usage:|unrecognized arguments|No such file or directory.*(demo|eval|train)", "entrypoint_ambiguity"),
    (r"docker build|docker run|Dockerfile", "docker_only_instruction"),
]

PHASE_PATTERNS: list[tuple[str, str]] = [
    (r"git clone|checkout|repo_profile", "repo"),
    (r"pip install|conda install|uv pip|setup.py|pyproject", "install"),
    (r"ModuleNotFoundError|ImportError", "import"),
    (r"usage:|unrecognized arguments|entrypoint|demo|eval|train", "entrypoint"),
    (r"semantic|expected_artifact|verifier|checksum|assert", "semantic"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_attempt_text(attempt_dir: Path) -> str:
    chunks: list[str] = []
    for pattern in ["*.log", "logs/*.log", "*.txt", "install_plan.json", "repo_profile.json"]:
        for path in sorted(attempt_dir.glob(pattern))[:30]:
            if path.is_file() and path.stat().st_size <= 1_000_000:
                chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def infer_tags(text: str) -> list[str]:
    tags = [tag for pattern, tag in FAILURE_PATTERNS if re.search(pattern, text, flags=re.IGNORECASE)]
    return sorted(set(tags))


def infer_phase(text: str) -> str:
    matched = [phase for pattern, phase in PHASE_PATTERNS if re.search(pattern, text, flags=re.IGNORECASE)]
    return matched[-1] if matched else "unknown"


def build_report(attempt_dir: Path, task_id: str | None) -> dict[str, Any]:
    text = read_attempt_text(attempt_dir)
    score_path = attempt_dir / "score.json"
    score = load_json(score_path) if score_path.exists() else None
    verifier = score.get("verifier", {}) if isinstance(score, dict) else {}
    parsed = verifier.get("parsed", {}) if isinstance(verifier, dict) else {}

    error_text = ""
    if isinstance(parsed, dict):
        error_text = str(parsed.get("error") or "")
    if isinstance(verifier, dict):
        error_text = "\n".join(part for part in [error_text, str(verifier.get("stderr_tail") or "")] if part)

    combined = "\n".join([text, error_text])
    return {
        "schema_version": "paperenvbench.failure_report.v1",
        "generated_at": utc_now(),
        "task_id": task_id or (score.get("task_id") if isinstance(score, dict) else None),
        "attempt_dir": str(attempt_dir.resolve()),
        "failed_phase": infer_phase(combined),
        "failure_tags": infer_tags(combined),
        "verifier_error": error_text[-4000:],
        "score_level": score.get("level") if isinstance(score, dict) else None,
        "score": score.get("score") if isinstance(score, dict) else None,
        "next_actions": [
            "Inspect the first failing phase before retrying downstream commands.",
            "Keep repairs as environment compatibility patches, not algorithm changes.",
            "Regenerate score.json after producing a new artifact bundle.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a PaperEnvBench failure_report.json for an attempt directory.")
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_report(args.attempt_dir.resolve(), args.task_id)
    output = args.output or (args.attempt_dir / "failure_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
