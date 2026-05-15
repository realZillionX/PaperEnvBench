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
TASK_ID = "stable_diffusion_text2img_minimal"
EXPECTED_COMMIT = "21f890f9da3cfbeaba8e2ac3c425ee9e998d5229"
SUCCESS_LEVEL = "L4_cpu_license_gate_fallback"
REQUIRED_TERMS = [
    "latent diffusion",
    "astronaut",
    "horse",
    "license gate",
]
REQUIRED_ROUTE_KEYS = [
    "script",
    "config",
    "checkpoint",
    "license_gate",
    "sampler",
]
EXPECTED_NUMERIC = {
    "latent_shape": [1, 4, 64, 64],
    "sampler_steps": [50],
    "fallback_rgb_mean": [72.5, 105.0, 146.25],
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
    for required_check in [
        "repo_commit_matches",
        "txt2img_entrypoint_recorded",
        "config_route_recorded",
        "checkpoint_route_recorded",
        "license_gate_documented",
        "fallback_artifact_valid",
        "semantic_output_valid",
    ]:
        if checks.get(required_check) is not True:
            raise AssertionError(f"missing required check: {required_check}")

    route = payload.get("route", {})
    missing_route = [key for key in REQUIRED_ROUTE_KEYS if not route.get(key)]
    if missing_route:
        raise AssertionError(f"route evidence is missing keys: {missing_route}")
    if route.get("script") != "scripts/txt2img.py":
        raise AssertionError("txt2img route must use the pinned repository script")
    if route.get("config") != "configs/stable-diffusion/v1-inference.yaml":
        raise AssertionError("config route must use v1-inference.yaml")
    if not str(route.get("checkpoint", "")).endswith("model.ckpt"):
        raise AssertionError("checkpoint route must point to model.ckpt")

    semantic = payload.get("semantic", {})
    text = " ".join(
        str(semantic.get(key, ""))
        for key in ("label", "caption", "answer", "description", "prompt", "safety_boundary")
    )
    missing_terms = [term for term in REQUIRED_TERMS if term not in text]
    if missing_terms:
        raise AssertionError(f"semantic output is missing required terms: {missing_terms}")

    sha = payload.get("sha256", {})
    if sha.get("route") != canonical_sha256(route):
        raise AssertionError("route sha256 mismatch")
    if sha.get("semantic") != canonical_sha256(semantic):
        raise AssertionError("semantic sha256 mismatch")

    environment = payload.get("environment", {})
    if environment.get("checkpoint_loaded") is not False:
        raise AssertionError("check-only artifact must not claim the gated checkpoint was loaded")
    if environment.get("license_acceptance_required") is not True:
        raise AssertionError("license gate must be explicitly documented")
    if environment.get("full_inference_requires_gpu") is not True:
        raise AssertionError("full inference GPU boundary must be documented")

    fallback_artifact = payload.get("fallback_artifact", {})
    if fallback_artifact.get("type") != "deterministic_thumbnail_summary":
        raise AssertionError("fallback artifact type mismatch")
    if fallback_artifact.get("width") != 8 or fallback_artifact.get("height") != 8:
        raise AssertionError("fallback thumbnail summary must be 8x8")

    numeric = payload.get("numeric", {})
    for key, expected_value in EXPECTED_NUMERIC.items():
        if key not in numeric:
            raise AssertionError(f"missing numeric vector: {key}")
        assert_close(numeric[key], expected_value, f"numeric.{key}")

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
            "environment": environment,
            "fallback_artifact": fallback_artifact,
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
        print(
            json.dumps({"task_id": TASK_ID, "status": "fail", "error": str(exc)}, indent=2, sort_keys=True),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
