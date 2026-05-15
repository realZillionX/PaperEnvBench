#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

TASK_ROOT = Path(__file__).resolve().parent
TASK_ID = "fasttext_text_classification_minimal"
EXPECTED_COMMIT = "1142dc4c4ecbc19cc16eee5cdd28472e689267e6"
SUCCESS_LEVEL = "L4_cpu_deterministic_fallback"
EXPECTED_TRAIN_SHA256 = "8903425632da3333666bfcaac986c24158ce9dceccf09e3c436bf3b2bb681de2"
EXPECTED_TEST_SHA256 = "b14f414db9b82d550b514ca2e9f31acb2acad4087d8f736ff049e7b7667ef2cc"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_path = artifact_dir / "expected_artifact.json"
    require(artifact_path.exists(), f"missing artifact: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    require(payload.get("task_id") == TASK_ID, f"wrong task_id: {payload.get('task_id')}")
    require(payload.get("success_level") == SUCCESS_LEVEL, f"wrong success_level: {payload.get('success_level')}")
    require(payload.get("repo", {}).get("commit") == EXPECTED_COMMIT, "repo commit mismatch")

    route = payload.get("route", {})
    for key in ("clone", "build", "cli_help", "cli_train", "cli_test", "cli_predict", "python_api"):
        require(isinstance(route.get(key), str) and route[key], f"missing route key: {key}")
    require("./fasttext supervised" in route["cli_train"], "CLI supervised route is not recorded")
    require("fasttext.load_model" in route["python_api"], "Python API route is not recorded")

    dataset = payload.get("dataset", {})
    require(dataset.get("train_lines") == 8, "unexpected train line count")
    require(dataset.get("test_lines") == 2, "unexpected test line count")
    require(dataset.get("train_sha256") == EXPECTED_TRAIN_SHA256, "train dataset hash mismatch")
    require(dataset.get("test_sha256") == EXPECTED_TEST_SHA256, "test dataset hash mismatch")

    metrics = payload.get("metrics", {})
    require(metrics.get("N") == 2, "unexpected test N")
    require(math.isclose(float(metrics.get("P@1", -1)), 1.0), "P@1 must be 1.0")
    require(math.isclose(float(metrics.get("R@1", -1)), 1.0), "R@1 must be 1.0")

    predictions = payload.get("predictions", [])
    require(len(predictions) == 2, "expected two prediction rows")
    expected = [
        ("football team wins match", "__label__sports"),
        ("python compiler builds software", "__label__tech"),
    ]
    for row, (text, label) in zip(predictions, expected):
        require(row.get("text") == text, f"unexpected prediction text: {row.get('text')}")
        labels = row.get("labels")
        probabilities = row.get("probabilities")
        require(isinstance(labels, list) and labels and labels[0] == label, f"unexpected top label for {text}")
        require(isinstance(probabilities, list) and probabilities, f"missing probabilities for {text}")
        require(float(probabilities[0]) >= 0.75, f"low top probability for {text}: {probabilities[0]}")
        require(all(math.isfinite(float(value)) for value in probabilities), f"non-finite probabilities for {text}")

    checks = payload.get("checks", {})
    require(isinstance(checks, dict) and checks, "missing checks")
    require(all(value is True for value in checks.values()), f"failing checks: {checks}")

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": SUCCESS_LEVEL,
        "artifact_dir": str(artifact_dir.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "checks": checks,
        "observed": {
            "repo_commit": payload.get("repo", {}).get("commit"),
            "test_metrics": metrics,
            "top_labels": [row["labels"][0] for row in predictions],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=TASK_ROOT / "artifacts")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = validate_artifact(args.artifact_dir)
    except Exception as exc:
        print(json.dumps({"task_id": TASK_ID, "status": "fail", "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
