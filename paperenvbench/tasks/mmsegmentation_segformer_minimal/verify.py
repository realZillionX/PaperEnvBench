#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parent
TASK_ID = "mmsegmentation_segformer_minimal"
EXPECTED_COMMIT = "b040e147adfa027bbc071b624bedf0ae84dfc922"


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssertionError(f"JSON output is not parseable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AssertionError(f"JSON output must be an object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise AssertionError(f"not a valid PNG file: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify(attempt_root: Path) -> dict[str, Any]:
    attempt_root = attempt_root.resolve()
    artifact_dir = attempt_root if (attempt_root / "expected_artifact.json").exists() else attempt_root / "artifacts"
    summary_path = artifact_dir / "expected_artifact.json"
    input_path = artifact_dir / "expected_input.png"
    mask_path = artifact_dir / "expected_mask.png"
    expected = load_json(TASK_ROOT / "expected_output.json")

    for path, label in [
        (summary_path, "summary JSON"),
        (input_path, "input PNG"),
        (mask_path, "mask PNG"),
    ]:
        require(path.exists() and path.stat().st_size > 0, f"missing nonempty {label}: {path}")

    payload = load_json(summary_path)
    require(payload.get("task_id") == TASK_ID, "summary has wrong task_id")
    require(payload.get("success_level") == "L4_fallback", "success_level must be L4_fallback")
    require(payload.get("repo_commit") == EXPECTED_COMMIT, "repo_commit does not match pinned commit")

    checks = payload.get("checks")
    require(isinstance(checks, dict), "checks must be an object")
    required_checks = [
        "repo_commit_matches",
        "mmseg_imported",
        "segformer_config_loaded",
        "model_built_from_mmseg_registry",
        "logits_shape_matches",
        "logits_are_finite",
        "logits_have_variance",
        "probabilities_are_normalized",
        "mask_png_exists",
        "input_png_exists",
        "mask_pixels_match_input",
        "mmcv_ops_stub_not_called",
    ]
    for check in required_checks:
        require(checks.get(check) is True, f"required semantic check failed or missing: {check}")

    fallback = payload.get("fallback")
    require(isinstance(fallback, dict), "fallback must be an object")
    require(fallback.get("used") is True, "fallback.used must be true for this CPU gold task")
    require(fallback.get("mmcv_lite_warning_observed") is True, "mmcv-lite warning must be recorded")
    stub = fallback.get("mmcv_ops_stub")
    require(isinstance(stub, dict), "fallback.mmcv_ops_stub must be an object")
    require(stub.get("stub_called") is False, "mmcv.ops stub must not be called during SegFormer forward")

    model = payload.get("model")
    require(isinstance(model, dict), "model must be an object")
    require(model.get("parameter_count", 0) >= 3_000_000, "SegFormer model parameter count is too small")
    model_types = model.get("types")
    require(isinstance(model_types, dict), "model.types must be an object")
    require(model_types.get("backbone", "").endswith("mit.MixVisionTransformer"), "backbone must be MixVisionTransformer")
    require(model_types.get("decode_head", "").endswith("segformer_head.SegformerHead"), "decode head must be SegformerHead")

    output = payload.get("output")
    require(isinstance(output, dict), "output must be an object")
    thresholds = expected["semantic_thresholds"]
    require(output.get("logits_shape") == thresholds["expected_logits_shape"], "logits shape mismatch")
    require(float(output.get("logits_std", 0.0)) > float(thresholds["min_logits_std"]), "logits look degenerate")
    require(
        float(output.get("probability_sum_max_abs_error", 1.0)) <= float(thresholds["max_probability_sum_error"]),
        "probabilities are not normalized",
    )

    input_meta = payload.get("input_image")
    require(isinstance(input_meta, dict), "input_image must be an object")
    require(sha256_file(input_path) == input_meta.get("sha256"), "input PNG sha256 mismatch")
    require(png_size(input_path) == (64, 64), "input PNG must be 64x64")

    require(sha256_file(mask_path) == output.get("mask_sha256"), "mask PNG sha256 mismatch")
    require(png_size(mask_path) == tuple(thresholds["expected_mask_shape"][::-1]), "mask PNG must be 64x64")
    histogram = output.get("mask_class_histogram")
    require(isinstance(histogram, dict), "mask_class_histogram must be an object")
    mask_pixels = sum(int(value) for value in histogram.values())
    require(mask_pixels >= int(thresholds["min_mask_pixels"]), "mask pixel count is too small")
    require(set(histogram).issubset({"0", "1", "2"}), "mask contains class outside toy class range")

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": payload["success_level"],
        "attempt_root": str(attempt_root),
        "checks": {name: checks[name] for name in required_checks},
        "observed": {
            "repo_commit": payload["repo_commit"],
            "package_versions": payload.get("package_versions"),
            "logits_shape": output["logits_shape"],
            "logits_std": output["logits_std"],
            "mask_class_histogram": histogram,
            "fallback": fallback,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the PaperEnvBench MMSegmentation SegFormer minimal task output.")
    parser.add_argument(
        "attempt_root",
        nargs="?",
        default=".",
        help="Attempt root containing artifacts/expected_artifact.json and PNG artifacts.",
    )
    parser.add_argument("--artifact-dir", type=Path, help="Artifact directory to validate directly.")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    args = parser.parse_args()

    try:
        result = verify(args.artifact_dir if args.artifact_dir else Path(args.attempt_root))
    except AssertionError as exc:
        failure = {"task_id": TASK_ID, "status": "fail", "error": str(exc)}
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"PASS {TASK_ID}")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
