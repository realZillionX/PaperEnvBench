#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parent
TASK_ID = "detectron2_maskrcnn_minimal"
REPO_COMMIT = "e0ec4e189d438848521aee7926f9900e114229f5"
EXPECTED_SHA256 = {
    "expected_artifact.json": "e86e2e69775fda5bc15dfa69a4e1fc246af59409e0ade2e89681f52dc0c5b68c",
    "expected_input.png": "1ff61eef520e6c6966f5cffc6cf990ac4a76865768790e478fa31cddc49e5db9",
    "expected_mask.png": "a745d5a0231f1cf14d5cfc3589a8dc73f1cbe0982edc17a717ebf44d29c64b6b",
    "expected_overlay.png": "b98701205b899244d1ac3a9eb4430e2d892210e55418955d1a8c0c017ba9b07f",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise AssertionError(f"not a PNG file: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def verify(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    summary_path = artifact_dir / "expected_artifact.json"
    input_path = artifact_dir / "expected_input.png"
    mask_path = artifact_dir / "expected_mask.png"
    overlay_path = artifact_dir / "expected_overlay.png"

    for path in (summary_path, input_path, mask_path, overlay_path):
        if not path.exists() or path.stat().st_size <= 0:
            raise AssertionError(f"missing nonempty artifact: {path}")

    observed_sha = {
        "expected_artifact.json": sha256(summary_path),
        "expected_input.png": sha256(input_path),
        "expected_mask.png": sha256(mask_path),
        "expected_overlay.png": sha256(overlay_path),
    }
    if observed_sha != EXPECTED_SHA256:
        raise AssertionError({"checksum_mismatch": observed_sha})

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssertionError(f"summary JSON is not parseable: {summary_path}: {exc}") from exc

    if payload.get("task_id") != TASK_ID:
        raise AssertionError(f"wrong task_id: {payload.get('task_id')}")
    if payload.get("repo_commit") != REPO_COMMIT:
        raise AssertionError(f"wrong repo_commit: {payload.get('repo_commit')}")
    if payload.get("success_level") != "L4":
        raise AssertionError(f"wrong success_level: {payload.get('success_level')}")

    input_meta = payload.get("input", {})
    width, height = png_size(input_path)
    if (width, height) != (input_meta.get("width"), input_meta.get("height")):
        raise AssertionError({"png_size": (width, height), "input_meta": input_meta})
    if png_size(mask_path) != (width, height) or png_size(overlay_path) != (width, height):
        raise AssertionError("input, mask, and overlay PNG dimensions must match")

    checks = payload.get("checks", {})
    required_checks = {
        "native_extension_imported",
        "checkpoint_url_resolved",
        "prediction_count_positive",
        "scores_finite",
        "boxes_finite",
        "mask_pixels_positive",
        "all_checks_passed",
    }
    if set(checks) != required_checks or not all(checks.values()):
        raise AssertionError({"checks": checks})

    prediction = payload.get("prediction", {})
    scores = prediction.get("top_scores", [])
    boxes = prediction.get("top_boxes_xyxy", [])
    mask_pixels = prediction.get("top_mask_pixels", [])
    num_instances = prediction.get("num_instances")
    if not isinstance(num_instances, int) or not (1 <= num_instances <= 5):
        raise AssertionError({"num_instances": num_instances})
    if len(scores) != num_instances or len(boxes) != num_instances or len(mask_pixels) != num_instances:
        raise AssertionError({"prediction": prediction})
    if not all(finite_number(score) and 0.0 <= float(score) <= 1.0 for score in scores):
        raise AssertionError({"top_scores": scores})
    if not all(isinstance(pixels, int) and pixels > 0 for pixels in mask_pixels):
        raise AssertionError({"top_mask_pixels": mask_pixels})
    for box in boxes:
        if len(box) != 4 or not all(finite_number(value) for value in box):
            raise AssertionError({"bad_box": box})

    expected_mask = payload.get("expected_shape_mask", {})
    if expected_mask.get("mask_pixels") != 7056:
        raise AssertionError({"expected_shape_mask": expected_mask})
    if payload.get("checkpoint_url") != (
        "https://dl.fbaipublicfiles.com/detectron2/"
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl"
    ):
        raise AssertionError({"checkpoint_url": payload.get("checkpoint_url")})

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": "L4",
        "artifact_dir": str(artifact_dir),
        "checks": {
            "summary_json_valid": True,
            "checksums_match": True,
            "png_dimensions_match": True,
            "checkpoint_loaded": True,
            "mask_rcnn_predictions_valid": True,
        },
        "observed": {
            "num_instances": num_instances,
            "top_score": float(scores[0]),
            "top_mask_pixels": int(mask_pixels[0]),
            "input_size": [width, height],
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
