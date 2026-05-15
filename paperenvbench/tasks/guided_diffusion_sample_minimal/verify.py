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
TASK_ID = "guided_diffusion_sample_minimal"
EXPECTED_COMMIT = "22e0df8183507e13a7813f8d38d51b072ca1e67c"
SUCCESS_LEVEL = "L4_cpu_deterministic_fallback"
REQUIRED_ROUTE_KEYS = [
    "sample_script",
    "unguided_script",
    "super_resolution_script",
    "diffusion_checkpoint",
    "classifier_checkpoint",
    "model_flags",
    "sample_flags",
]
REQUIRED_PINNED_FILES = {
    "scripts/classifier_sample.py",
    "scripts/image_sample.py",
    "scripts/super_res_sample.py",
    "guided_diffusion/script_util.py",
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


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssertionError(f"JSON artifact is not parseable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AssertionError(f"JSON artifact must be an object: {path}")
    return payload


def parse_ppm(path: Path) -> dict[str, Any]:
    tokens = path.read_text(encoding="ascii").split()
    if len(tokens) < 4 or tokens[0] != "P3":
        raise AssertionError("expected_sample_grid.ppm must be ASCII P3 PPM")
    width = int(tokens[1])
    height = int(tokens[2])
    max_value = int(tokens[3])
    values = [int(token) for token in tokens[4:]]
    if width != 16 or height != 16 or max_value != 255:
        raise AssertionError({"width": width, "height": height, "max_value": max_value})
    if len(values) != width * height * 3:
        raise AssertionError(f"unexpected PPM value count: {len(values)}")
    if any(value < 0 or value > max_value for value in values):
        raise AssertionError("PPM pixel values out of range")
    means = []
    for channel in range(3):
        channel_values = values[channel::3]
        means.append(round(sum(channel_values) / (len(channel_values) * max_value), 4))
    return {
        "width": width,
        "height": height,
        "max_value": max_value,
        "mean_rgb": means,
        "sha256": sha256_file(path),
    }


def validate_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_path = artifact_dir / "expected_artifact.json"
    ppm_path = artifact_dir / "expected_sample_grid.ppm"
    if not artifact_path.exists():
        raise AssertionError(f"missing artifact: {artifact_path}")
    if not ppm_path.exists():
        raise AssertionError(f"missing sample grid: {ppm_path}")

    payload = load_json(artifact_path)
    ppm = parse_ppm(ppm_path)
    if payload.get("task_id") != TASK_ID:
        raise AssertionError(f"wrong task_id: {payload.get('task_id')}")
    if payload.get("success_level") != SUCCESS_LEVEL:
        raise AssertionError(f"unexpected success_level: {payload.get('success_level')}")
    if payload.get("repo", {}).get("commit") != EXPECTED_COMMIT:
        raise AssertionError("repo commit mismatch")

    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        raise AssertionError({"checks": checks})

    evidence = payload.get("evidence", {})
    pinned_files = set(evidence.get("pinned_files", []))
    missing_files = sorted(REQUIRED_PINNED_FILES - pinned_files)
    if missing_files:
        raise AssertionError(f"pinned route evidence is missing files: {missing_files}")

    route = payload.get("route", {})
    missing_route = [key for key in REQUIRED_ROUTE_KEYS if not route.get(key)]
    if missing_route:
        raise AssertionError(f"route evidence is missing keys: {missing_route}")
    if route.get("sample_script") != "scripts/classifier_sample.py":
        raise AssertionError("classifier-guided script route is not recorded")
    if "64x64_diffusion.pt" not in str(route.get("diffusion_checkpoint")):
        raise AssertionError("64x64 diffusion checkpoint route is missing")
    if "64x64_classifier.pt" not in str(route.get("classifier_checkpoint")):
        raise AssertionError("64x64 classifier checkpoint route is missing")
    sample_flags = route.get("sample_flags", {})
    if float(sample_flags.get("classifier_scale", -1)) <= 0:
        raise AssertionError("classifier_scale must be positive for classifier guidance")
    if str(sample_flags.get("timestep_respacing")) != "250":
        raise AssertionError("expected timestep_respacing route is missing")

    semantic = payload.get("semantic", {})
    semantic_text = " ".join(str(value) for value in semantic.values())
    required_terms = ["classifier_guided_fallback", "guided-diffusion", "classifier_sample", "ImageNet"]
    missing_terms = [term for term in required_terms if term not in semantic_text]
    if missing_terms:
        raise AssertionError(f"semantic artifact is missing required terms: {missing_terms}")

    sha = payload.get("sha256", {})
    if sha.get("route") != canonical_sha256(route):
        raise AssertionError("route sha256 mismatch")
    if sha.get("semantic") != canonical_sha256(semantic):
        raise AssertionError("semantic sha256 mismatch")
    if payload.get("artifacts", {}).get("sample_grid_sha256") != ppm["sha256"]:
        raise AssertionError("sample grid sha256 mismatch")

    numeric = payload.get("numeric", {})
    for key, values in numeric.items():
        if not isinstance(values, list) or not values:
            raise AssertionError(f"numeric vector {key!r} must be a nonempty list")
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            raise AssertionError(f"numeric vector {key!r} contains non-finite values")

    red_mean, green_mean, blue_mean = ppm["mean_rgb"]
    if not (0.30 <= red_mean <= 0.70 and 0.30 <= green_mean <= 0.70 and 0.30 <= blue_mean <= 0.70):
        raise AssertionError(f"sample grid mean RGB is outside expected range: {ppm['mean_rgb']}")

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": SUCCESS_LEVEL,
        "artifact_dir": str(artifact_dir.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "checks": checks,
        "observed": {
            "repo_commit": payload.get("repo", {}).get("commit"),
            "route_script": route.get("sample_script"),
            "sample_grid": ppm,
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
