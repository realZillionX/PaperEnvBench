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
TASK_ID = "sentence_transformers_similarity_minimal"
EXPECTED_COMMIT = "5b27be706546f5e094e0f506d8593250e9a37109"
SUCCESS_LEVEL = "L4_cpu_deterministic_fallback"
REQUIRED_ROUTE_TERMS = ["SentenceTransformer", "model.encode", "cos_sim"]
REQUIRED_SEMANTIC_TERMS = ["musical", "stock-market", "similarity"]


def canonical_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_path = artifact_dir / "expected_artifact.json"
    if not artifact_path.exists():
        raise AssertionError(f"missing artifact: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if payload.get("task_id") != TASK_ID:
        raise AssertionError(f"wrong task_id: {payload.get('task_id')}")
    if payload.get("success_level") != SUCCESS_LEVEL:
        raise AssertionError(f"unexpected success_level: {payload.get('success_level')}")
    if payload.get("repo", {}).get("commit") != EXPECTED_COMMIT:
        raise AssertionError("repo commit mismatch")

    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        raise AssertionError({"checks": checks})

    route = payload.get("route", {})
    route_text = json.dumps(route, sort_keys=True)
    missing_route = [term for term in REQUIRED_ROUTE_TERMS if term not in route_text]
    if missing_route:
        raise AssertionError(f"route evidence is missing terms: {missing_route}")
    if "paraphrase-MiniLM-L3-v2" not in route_text:
        raise AssertionError("public checkpoint route is missing")

    semantic = payload.get("semantic", {})
    semantic_text = " ".join(str(value) for value in semantic.values())
    missing_semantic = [term for term in REQUIRED_SEMANTIC_TERMS if term not in semantic_text]
    if missing_semantic:
        raise AssertionError(f"semantic evidence is missing terms: {missing_semantic}")

    numeric = payload.get("numeric", {})
    positive = float(numeric.get("positive_similarity"))
    negative = float(numeric.get("negative_similarity"))
    margin = float(numeric.get("margin"))
    minimum_margin = float(semantic.get("minimum_margin"))
    if not all(math.isfinite(value) for value in [positive, negative, margin, minimum_margin]):
        raise AssertionError("similarity values must be finite")
    if positive <= negative:
        raise AssertionError({"positive_similarity": positive, "negative_similarity": negative})
    if margin < minimum_margin:
        raise AssertionError({"margin": margin, "minimum_margin": minimum_margin})

    matrix = numeric.get("similarity_matrix")
    if not isinstance(matrix, list) or len(matrix) != 3:
        raise AssertionError("similarity_matrix must contain three rows")
    for row in matrix:
        if not isinstance(row, list) or len(row) != 3:
            raise AssertionError("similarity_matrix must be 3x3")
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in row):
            raise AssertionError("similarity_matrix contains non-finite values")

    sha = payload.get("sha256", {})
    if sha.get("route") != canonical_sha256(route):
        raise AssertionError("route sha256 mismatch")
    if sha.get("semantic") != canonical_sha256(semantic):
        raise AssertionError("semantic sha256 mismatch")

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": SUCCESS_LEVEL,
        "artifact_dir": str(artifact_dir.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "checks": checks,
        "observed": {
            "repo_commit": payload.get("repo", {}).get("commit"),
            "positive_similarity": positive,
            "negative_similarity": negative,
            "margin": margin,
            "checkpoint": route.get("checkpoint"),
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
