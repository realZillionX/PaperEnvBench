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
TASK_ID = "latent_diffusion_sample_minimal"
EXPECTED_COMMIT = "a506df5756472e2ebaf9078affdde2c4f1502cd4"
SUCCESS_LEVEL = "L4_fallback"
REQUIRED_TERMS = ["latent diffusion", "DDIM sampler", "AutoencoderKL", "text-to-image"]
REQUIRED_ROUTE_KEYS = ["config", "sampler", "autoencoder", "script", "checkpoint", "decode", "model_target"]


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
    if route.get("config") != "configs/latent-diffusion/txt2img-1p4B-eval.yaml":
        raise AssertionError("latent diffusion text-to-image config route mismatch")
    if route.get("sampler") != "ldm.models.diffusion.ddim.DDIMSampler":
        raise AssertionError("DDIM sampler route mismatch")
    if route.get("autoencoder") != "ldm.models.autoencoder.AutoencoderKL":
        raise AssertionError("AutoencoderKL route mismatch")

    semantic = payload.get("semantic", {})
    text = " ".join(str(semantic.get(key, "")) for key in ("task", "prompt", "description"))
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
        raise AssertionError("check-only artifact must not claim full checkpoint loading")
    if environment.get("checkpoint_required_for_full_inference") is not True:
        raise AssertionError("checkpoint boundary is not documented")

    numeric = payload.get("numeric", {})
    for key, values in numeric.items():
        if not isinstance(values, list) or not values:
            raise AssertionError(f"numeric vector {key!r} must be a nonempty list")
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            raise AssertionError(f"numeric vector {key!r} contains non-finite values")

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
