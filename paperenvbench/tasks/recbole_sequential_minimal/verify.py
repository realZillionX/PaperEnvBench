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
TASK_ID = "recbole_sequential_minimal"
EXPECTED_COMMIT = "7b02be5ec80a88310f2d04a27a82adfcbb5dc211"
SUCCESS_LEVEL = "L4_cpu_deterministic_fallback"
REQUIRED_TERMS = ["sasrec", "sequential recommendation", "hit@10", "mrr@10", "ndcg@10"]
REQUIRED_ROUTE_KEYS = ["api", "cli", "config", "dataset", "metric", "model", "task"]


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
    missing_route = [key for key in REQUIRED_ROUTE_KEYS if not route.get(key)]
    if missing_route:
        raise AssertionError(f"route evidence is missing keys: {missing_route}")

    semantic = payload.get("semantic", {})
    metric_text = " ".join(route.get("metric", "").lower().split())
    text = " ".join(
        [
            json.dumps(route, sort_keys=True),
            json.dumps(semantic, sort_keys=True),
            metric_text,
        ]
    ).lower()
    missing = [term for term in REQUIRED_TERMS if term not in text]
    if missing:
        raise AssertionError(f"semantic output is missing required terms: {missing}")
    if semantic.get("predicted_ranked_items", [None])[0] != semantic.get("ground_truth_next_item"):
        raise AssertionError("ground-truth next item is not ranked first")

    sha = payload.get("sha256", {})
    if sha.get("route") != canonical_sha256(route):
        raise AssertionError("route sha256 mismatch")
    if sha.get("semantic") != canonical_sha256(semantic):
        raise AssertionError("semantic sha256 mismatch")

    numeric = payload.get("numeric", {})
    for key, values in numeric.items():
        if not isinstance(values, list) or not values:
            raise AssertionError(f"numeric vector {key!r} must be a nonempty list")
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            raise AssertionError(f"numeric vector {key!r} contains non-finite values")
    for key in ("hit_at_10", "mrr_at_10", "ndcg_at_10"):
        value = float(numeric.get(key, [0.0])[0])
        if value < 1.0:
            raise AssertionError(f"{key} below deterministic toy expectation: {value}")

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": SUCCESS_LEVEL,
        "artifact_dir": str(artifact_dir.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "checks": checks,
        "observed": {
            "repo_commit": payload.get("repo", {}).get("commit"),
            "semantic": semantic,
            "metrics": {key: numeric[key][0] for key in ("hit_at_10", "mrr_at_10", "ndcg_at_10")},
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
