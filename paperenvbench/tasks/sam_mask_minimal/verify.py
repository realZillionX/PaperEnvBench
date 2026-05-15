#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parent
TASK_ID = "sam_mask_minimal"


def verify(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    summary_path = artifact_dir / "expected_artifact.json"
    mask_path = artifact_dir / "expected_artifact.png"
    input_path = artifact_dir / "expected_input.png"

    if not summary_path.exists():
        raise AssertionError(f"missing summary JSON: {summary_path}")
    if not mask_path.exists() or mask_path.stat().st_size <= 0:
        raise AssertionError(f"missing nonempty mask artifact: {mask_path}")
    if not input_path.exists() or input_path.stat().st_size <= 0:
        raise AssertionError(f"missing nonempty input artifact: {input_path}")

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssertionError(f"summary JSON is not parseable: {summary_path}: {exc}") from exc
    score = payload.get("score")
    mask_pixels = payload.get("mask_pixels")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise AssertionError("summary.score must be finite")
    if not isinstance(mask_pixels, int):
        raise AssertionError("summary.mask_pixels must be an integer")

    checks = {
        "summary_json_exists": True,
        "mask_png_exists": mask_path.stat().st_size > 0,
        "input_png_exists": input_path.stat().st_size > 0,
        "mask_area_in_expected_range": 1000 <= mask_pixels <= 60000,
        "score_finite": math.isfinite(float(score)),
    }
    if not all(checks.values()):
        raise AssertionError({"checks": checks, "payload": payload})

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": "L4",
        "artifact_dir": str(artifact_dir),
        "checks": checks,
        "observed": {
            "score": float(score),
            "mask_pixels": mask_pixels,
            "mask_size_bytes": mask_path.stat().st_size,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default=str(TASK_ROOT / "artifacts"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        result = verify(Path(args.artifact_dir))
    except AssertionError as exc:
        failure = {"task_id": TASK_ID, "status": "fail", "error": str(exc)}
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
