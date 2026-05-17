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
TASK_ID = "clip_zeroshot_minimal"
REPO_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
MODEL_NAME = "ViT-B/32"
CHECKPOINT_SHA = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
INPUT_IMAGE_SHA = "308a3ca4503f1c7a07803916c369d78c4ef501e5ab7fc727da9b5e1d2f9ec85b"
README_LABELS = ["a diagram", "a dog", "a cat"]
EXPECTED_TOP_LABEL = "a diagram"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssertionError(f"summary JSON is not parseable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AssertionError("summary JSON must be an object")
    return payload


def require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise AssertionError(f"{label} mismatch: expected {expected!r}, got {observed!r}")


def require_cuda(payload: dict[str, Any]) -> None:
    require_equal(payload.get("device"), "cuda", "device")
    torch_info = payload.get("torch")
    if not isinstance(torch_info, dict):
        raise AssertionError("torch evidence must be an object")
    if torch_info.get("cuda_available") is not True:
        raise AssertionError("torch.cuda_available must be true")
    torch_cuda = torch_info.get("torch_cuda")
    if not isinstance(torch_cuda, str) or not torch_cuda.startswith("12."):
        raise AssertionError(f"torch_cuda must record CUDA 12.x, got {torch_cuda!r}")
    if "4090" not in str(torch_info.get("gpu_name") or ""):
        raise AssertionError(f"gpu_name must identify the 4090 runtime, got {torch_info.get('gpu_name')!r}")


def softmax(values: list[float]) -> list[float]:
    pivot = max(values)
    exps = [math.exp(value - pivot) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def verify(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    summary_path = artifact_dir / "expected_artifact.json"
    image_path = artifact_dir / "expected_artifact.png"
    if not summary_path.exists():
        raise AssertionError(f"missing CLIP summary artifact: {summary_path}")
    if not image_path.exists() or image_path.stat().st_size <= 0:
        raise AssertionError(f"missing nonempty image artifact: {image_path}")

    payload = load_json(summary_path)
    require_equal(payload.get("task_id"), TASK_ID, "task_id")
    require_equal(payload.get("repo_commit"), REPO_COMMIT, "repo_commit")
    require_equal(payload.get("entrypoint"), "README zero-shot image classification example", "entrypoint")
    require_equal(payload.get("model_name"), MODEL_NAME, "model_name")
    require_equal(payload.get("checkpoint_sha256"), CHECKPOINT_SHA, "checkpoint_sha256")
    require_equal(payload.get("input_image_sha256"), INPUT_IMAGE_SHA, "input_image_sha256")
    require_equal(sha256(image_path), INPUT_IMAGE_SHA, "image artifact sha256")
    require_cuda(payload)

    checkpoint_size = payload.get("checkpoint_size_bytes")
    if not isinstance(checkpoint_size, int) or checkpoint_size < 350_000_000:
        raise AssertionError("checkpoint_size_bytes must prove the ViT-B/32 checkpoint was cached")

    labels = payload.get("labels")
    if labels != README_LABELS:
        raise AssertionError(f"labels must match the upstream README example: {README_LABELS!r}")
    forward_evidence = payload.get("forward_evidence")
    if not isinstance(forward_evidence, dict):
        raise AssertionError("forward_evidence must be an object")
    for key in ["used_clip_load", "used_clip_tokenize", "used_preprocess", "used_model_forward"]:
        if forward_evidence.get(key) is not True:
            raise AssertionError(f"forward_evidence.{key} must be true")

    logits = payload.get("logits_per_image")
    if not isinstance(logits, list) or len(logits) != len(README_LABELS):
        raise AssertionError("logits_per_image must contain one value per README label")
    logits_f = []
    for value in logits:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise AssertionError(f"logit must be finite, got {value!r}")
        logits_f.append(float(value))

    probs = payload.get("probabilities")
    if not isinstance(probs, dict):
        raise AssertionError("probabilities must be an object keyed by README label")
    missing = [label for label in README_LABELS if label not in probs]
    if missing:
        raise AssertionError(f"probabilities missing README labels: {missing}")
    ordered_probs = []
    for label in README_LABELS:
        value = probs[label]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise AssertionError(f"probability for {label!r} must be finite")
        prob = float(value)
        if not 0.0 <= prob <= 1.0:
            raise AssertionError(f"probability for {label!r} outside [0, 1]: {prob}")
        ordered_probs.append(prob)
    prob_sum = sum(ordered_probs)
    if not 0.999 <= prob_sum <= 1.001:
        raise AssertionError(f"probabilities must sum to approximately 1, got {prob_sum}")
    for label, observed, expected in zip(README_LABELS, ordered_probs, softmax(logits_f)):
        if abs(observed - expected) > 1e-3:
            raise AssertionError(f"probability for {label!r} is inconsistent with logits: {observed} vs {expected}")

    top_label = max(README_LABELS, key=lambda label: probs[label])
    require_equal(top_label, EXPECTED_TOP_LABEL, "top_label")
    require_equal(payload.get("top_label"), EXPECTED_TOP_LABEL, "payload.top_label")
    top_probability = float(probs[EXPECTED_TOP_LABEL])
    if top_probability < 0.9:
        raise AssertionError(f"expected top probability >= 0.9, got {top_probability}")

    image_shape = payload.get("image_tensor_shape")
    text_shape = payload.get("text_tensor_shape")
    if image_shape != [1, 3, 224, 224]:
        raise AssertionError(f"image_tensor_shape must prove CLIP preprocessing, got {image_shape!r}")
    if text_shape != [3, 77]:
        raise AssertionError(f"text_tensor_shape must prove CLIP tokenization, got {text_shape!r}")

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": "L4",
        "artifact_dir": str(artifact_dir),
        "checks": {
            "readme_clip_png_sha256_matches": True,
            "vit_b_32_checkpoint_sha256_matches": True,
            "readme_labels_exact": True,
            "clip_load_tokenize_preprocess_forward_evidence": True,
            "cuda_device_evidence_present": True,
            "logits_probability_consistency": True,
            "expected_label_rank_present": True,
        },
        "observed": {
            "top_label": top_label,
            "top_probability": top_probability,
            "probability_sum": prob_sum,
            "labels": probs,
            "checkpoint_sha256": payload.get("checkpoint_sha256"),
            "input_image_sha256": payload.get("input_image_sha256"),
            "device": payload.get("device"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default=str(TASK_ROOT / "artifacts"))
    parser.add_argument("--check-only", action="store_true")
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
