#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssertionError(f"JSON output is not parseable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AssertionError(f"JSON output must be an object: {path}")
    return payload


def first_existing(root: Path, candidates: list[str], label: str) -> Path:
    for rel in candidates:
        path = root / rel
        if path.exists():
            return path
    raise AssertionError(f"Missing {label}; checked: {', '.join(candidates)}")


def close_enough(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isfinite(actual):
        raise AssertionError(f"{label} is not finite: {actual}")
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label} differs: got {actual}, expected {expected} ± {tolerance}")


def verify(attempt_root: Path) -> dict[str, Any]:
    attempt_root = attempt_root.resolve()
    expected = load_json(TASK_ROOT / "expected_output.json")
    summary_path = first_existing(
        attempt_root,
        [
            "outputs/dino_vit_feature_minimal/feature_summary.json",
            "outputs/feature_summary.json",
            "artifacts/expected_artifact.json",
            "feature_summary.json",
        ],
        "DINO feature summary JSON",
    )
    image_path = first_existing(
        attempt_root,
        [
            "outputs/dino_vit_feature_minimal/dino_synthetic_input.png",
            "outputs/dino_synthetic_input.png",
            "artifacts/expected_artifact.png",
            "dino_synthetic_input.png",
        ],
        "synthetic input image artifact",
    )

    payload = load_json(summary_path)
    if payload.get("task_id") != "dino_vit_feature_minimal":
        raise AssertionError("feature summary has wrong task_id")
    if payload.get("status") != "pass":
        raise AssertionError("feature summary status must be 'pass'")
    if payload.get("repo_commit") != expected["source"]["repo_commit"]:
        raise AssertionError("repo_commit does not match the pinned gold commit")
    if payload.get("model") != "dino_vits16":
        raise AssertionError("model must be dino_vits16")

    load_state = payload.get("load_state_dict")
    if not isinstance(load_state, dict):
        raise AssertionError("load_state_dict must be an object")
    if load_state.get("strict") is not True:
        raise AssertionError("checkpoint load must be strict")
    if load_state.get("missing_keys") != [] or load_state.get("unexpected_keys") != []:
        raise AssertionError("strict checkpoint load must have no missing or unexpected keys")

    checkpoint_sha = payload.get("checkpoint_sha256")
    if checkpoint_sha != expected["checkpoint"]["sha256"]:
        raise AssertionError("checkpoint sha256 does not match expected DINO ViT-S/16 backbone")
    checkpoint_size = payload.get("checkpoint_size_bytes")
    if not isinstance(checkpoint_size, int) or checkpoint_size < 80_000_000:
        raise AssertionError("checkpoint_size_bytes is too small for the DINO ViT-S/16 backbone")

    input_payload = payload.get("input")
    if not isinstance(input_payload, dict):
        raise AssertionError("input must be an object")
    if input_payload.get("shape") != [1, 3, 224, 224]:
        raise AssertionError("input shape must be [1, 3, 224, 224]")
    if input_payload.get("sha256") != expected["input"]["sha256"]:
        raise AssertionError("synthetic input sha256 does not match expected artifact")
    if image_path.stat().st_size <= 0:
        raise AssertionError("synthetic input image is empty")

    feature = payload.get("feature")
    if not isinstance(feature, dict):
        raise AssertionError("feature must be an object")
    if feature.get("shape") != [1, 384]:
        raise AssertionError("feature shape must be [1, 384]")
    if feature.get("sha256_rounded_6") != expected["feature"]["sha256_rounded_6"]:
        raise AssertionError("rounded feature hash differs from gold output")

    close_enough(float(feature.get("mean")), expected["feature"]["mean"], 1e-5, "feature.mean")
    close_enough(float(feature.get("std")), expected["feature"]["std"], 1e-5, "feature.std")
    close_enough(float(feature.get("l2_norm")), expected["feature"]["l2_norm"], 1e-4, "feature.l2_norm")
    if float(feature.get("std")) <= 1.0 or float(feature.get("l2_norm")) <= 50.0:
        raise AssertionError("feature statistics look degenerate")

    return {
        "task_id": "dino_vit_feature_minimal",
        "status": "pass",
        "attempt_root": str(attempt_root),
        "checks": {
            "summary_json_exists": str(summary_path.relative_to(attempt_root)),
            "synthetic_image_exists": str(image_path.relative_to(attempt_root)),
            "checkpoint_loaded_strictly": True,
            "feature_shape": feature["shape"],
            "feature_hash_matches": True,
            "nondegenerate_feature_stats": True,
        },
        "observed": {
            "repo_commit": payload["repo_commit"],
            "checkpoint_sha256": checkpoint_sha,
            "feature": feature,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the PaperEnvBench DINO ViT feature minimal task output.")
    parser.add_argument(
        "attempt_root",
        nargs="?",
        default=".",
        help="Attempt root containing outputs/dino_vit_feature_minimal, or the task root containing artifacts/.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    args = parser.parse_args()

    try:
        result = verify(Path(args.attempt_root))
    except AssertionError as exc:
        failure = {"task_id": "dino_vit_feature_minimal", "status": "fail", "error": str(exc)}
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("PASS dino_vit_feature_minimal")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
