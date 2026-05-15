#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parent
TASK_ID = "clip_zeroshot_minimal"
EXPECTED_TOP_LABEL = "a white image with a black square"


def load_probs(path: Path) -> dict[str, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssertionError(f"probability JSON is not parseable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AssertionError("probability JSON must be an object")
    probs: dict[str, float] = {}
    for label, value in payload.items():
        if not isinstance(label, str):
            raise AssertionError("all labels must be strings")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise AssertionError(f"probability for {label!r} must be finite")
        probs[label] = float(value)
    return probs


def verify(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    probs_path = artifact_dir / "expected_artifact.json"
    image_path = artifact_dir / "expected_artifact.png"
    if not probs_path.exists():
        raise AssertionError(f"missing probability artifact: {probs_path}")
    if not image_path.exists() or image_path.stat().st_size <= 0:
        raise AssertionError(f"missing nonempty image artifact: {image_path}")

    probs = load_probs(probs_path)
    required = {
        "a white image with a black square",
        "a photo of a dog",
        "a handwritten equation",
    }
    missing = sorted(required - set(probs))
    if missing:
        raise AssertionError(f"missing labels: {missing}")
    total = sum(probs.values())
    if not 0.99 <= total <= 1.01:
        raise AssertionError(f"probabilities must sum to approximately 1, got {total}")
    for label, prob in probs.items():
        if prob < 0 or prob > 1:
            raise AssertionError(f"probability for {label!r} outside [0, 1]: {prob}")

    top_label = max(probs, key=probs.get)
    checks = {
        "label_probs_json_exists": True,
        "probability_vector_valid": True,
        "expected_label_rank_present": top_label == EXPECTED_TOP_LABEL,
        "expected_label_confident": probs[EXPECTED_TOP_LABEL] >= 0.5,
        "image_artifact_nonempty": image_path.stat().st_size > 0,
    }
    if not all(checks.values()):
        raise AssertionError({"checks": checks, "probs": probs})

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": "L4",
        "artifact_dir": str(artifact_dir),
        "checks": checks,
        "observed": {
            "top_label": top_label,
            "top_probability": probs[top_label],
            "probability_sum": total,
            "labels": probs,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default=str(TASK_ROOT / "artifacts"))
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
