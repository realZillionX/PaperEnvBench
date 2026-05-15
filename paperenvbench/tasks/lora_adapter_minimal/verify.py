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
TASK_ID = "lora_adapter_minimal"
EXPECTED_COMMIT = "c4593f060e6a368d7bb5af5273b8e42810cdef90"
SUCCESS_LEVEL = "L4_cpu_deterministic_fallback"
REQUIRED_TERMS = ["rank 2", "low-rank adapter", "base weight frozen"]
REQUIRED_ROUTE_KEYS = ["api", "trainable_filter", "checkpoint_save", "merge_behavior"]
EXPECTED_NUMERIC = {
    "input": [1.0, -2.0, 0.5, 3.0],
    "base_output": [0.65, -0.75, -0.45],
    "adapter_delta": [4.52, -1.55, -2.26],
    "merged_output": [5.17, -2.3, -2.71],
    "delta_weight": [
        [0.64, -0.38, -0.12, 1.06],
        [-0.1, 0.2, 0.3, -0.4],
        [-0.32, 0.19, 0.06, -0.53],
    ],
}


def canonical_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_close(observed: Any, expected: Any, path: str = "numeric") -> None:
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise AssertionError(f"{path} shape mismatch")
        for index, (obs_item, exp_item) in enumerate(zip(observed, expected)):
            assert_close(obs_item, exp_item, f"{path}[{index}]")
        return
    if not isinstance(observed, (int, float)) or not math.isfinite(float(observed)):
        raise AssertionError(f"{path} is not finite numeric")
    if abs(float(observed) - float(expected)) > 1e-6:
        raise AssertionError(f"{path} mismatch: observed={observed} expected={expected}")


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
    for required_check in ["repo_commit_matches", "route_recorded", "trainable_filter_valid", "adapter_state_dict_valid", "low_rank_delta_valid"]:
        if checks.get(required_check) is not True:
            raise AssertionError(f"missing required check: {required_check}")
    semantic = payload.get("semantic", {})
    text = " ".join(str(semantic.get(key, "")) for key in ("label", "caption", "answer", "description", "trainable_summary"))
    missing = [term for term in REQUIRED_TERMS if term not in text]
    if missing:
        raise AssertionError(f"semantic output is missing required terms: {missing}")
    route = payload.get("route", {})
    missing_route = [key for key in REQUIRED_ROUTE_KEYS if not route.get(key)]
    if missing_route:
        raise AssertionError(f"route evidence is missing keys: {missing_route}")
    sha = payload.get("sha256", {})
    if sha.get("route") != canonical_sha256(route):
        raise AssertionError("route sha256 mismatch")
    if sha.get("semantic") != canonical_sha256(semantic):
        raise AssertionError("semantic sha256 mismatch")
    numeric = payload.get("numeric", {})
    for key, expected_value in EXPECTED_NUMERIC.items():
        if key not in numeric:
            raise AssertionError(f"missing numeric vector: {key}")
        assert_close(numeric[key], expected_value, f"numeric.{key}")
    adapter = payload.get("adapter", {})
    if adapter.get("rank") != 2:
        raise AssertionError("adapter rank must be 2")
    if adapter.get("lora_alpha") != 4:
        raise AssertionError("adapter lora_alpha must be 4")
    if adapter.get("trainable_parameters") != 14:
        raise AssertionError("expected 14 adapter trainable parameters")
    if adapter.get("frozen_base_parameters") != 15:
        raise AssertionError("expected 15 frozen base parameters")
    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": SUCCESS_LEVEL,
        "artifact_dir": str(artifact_dir.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "checks": checks,
        "observed": {
            "repo_commit": payload.get("repo", {}).get("commit"),
            "route": route,
            "semantic": semantic,
            "adapter": adapter,
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
