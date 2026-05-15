#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import time
from typing import Any


TASK_ID = "groundingdino_phrase_grounding_minimal"
EXPECTED_COMMIT = "856dde20aee659246248e20734ef9ba5214f5e44"
EXPECTED_CHECKPOINT_SHA256 = "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799"
EXPECTED_CHECKPOINT_SIZE = 693997677
EXPECTED_INPUT_SHA256 = "b4fab73ad0198aac5b926378e7e07955e4204fd55eca27be1f0c65c6968f9aed"
EXPECTED_PHRASES = {"red square", "blue circle", "green strip"}
EXPECTED_LOGITS_SHAPE = [1, 900, 256]
EXPECTED_BOXES_SHAPE = [1, 900, 4]
TASK_ROOT = pathlib.Path(__file__).resolve().parent


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_dir: pathlib.Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return ""


def write_input_image(path: pathlib.Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (320, 240), (238, 241, 234))
    draw = ImageDraw.Draw(image)
    draw.rectangle([44, 62, 164, 182], fill=(214, 45, 45), outline=(90, 15, 15), width=4)
    draw.ellipse([198, 70, 292, 164], fill=(48, 88, 205), outline=(14, 34, 95), width=4)
    draw.rectangle([32, 194, 300, 216], fill=(98, 158, 84))
    image.save(path)


def run_model(repo_dir: pathlib.Path, checkpoint_path: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from torchvision.ops import box_convert

    repo_dir = repo_dir.resolve()
    checkpoint_path = checkpoint_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = repo_dir / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    if not config_path.exists():
        raise FileNotFoundError(f"missing model config: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")

    sys.path.insert(0, str(repo_dir))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    import groundingdino.datasets.transforms as transforms
    from groundingdino.models import build_model
    from groundingdino.util.inference import preprocess_caption
    from groundingdino.util.misc import clean_state_dict
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import get_phrases_from_posmap

    input_path = output_dir / "expected_input.png"
    annotated_path = output_dir / "expected_annotated.png"
    write_input_image(input_path)
    image = Image.open(input_path).convert("RGB")
    image_source = np.asarray(image)

    args = SLConfig.fromfile(str(config_path))
    args.device = "cpu"
    build_started = time.time()
    model = build_model(args)
    build_seconds = time.time() - build_started

    load_started = time.time()
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    load_result = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    load_seconds = time.time() - load_started

    model.eval()
    torch.set_num_threads(4)
    torch.manual_seed(20260516)

    transform = transforms.Compose(
        [
            transforms.RandomResize([256], max_size=256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor, _ = transform(image, None)
    caption = preprocess_caption("red square. blue circle. green strip")

    forward_started = time.time()
    with torch.no_grad():
        outputs = model(image_tensor[None], captions=[caption])
    forward_seconds = time.time() - forward_started

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]
    prediction_boxes = outputs["pred_boxes"].cpu()[0]
    box_threshold = 0.05
    text_threshold = 0.05
    mask = prediction_logits.max(dim=1)[0] > box_threshold
    if int(mask.sum().item()) == 0:
        top_idx = int(prediction_logits.max(dim=1)[0].argmax().item())
        mask[top_idx] = True

    logits = prediction_logits[mask]
    boxes = prediction_boxes[mask]
    max_scores = logits.max(dim=1)[0]
    order = torch.argsort(max_scores, descending=True)[:5]
    logits = logits[order]
    boxes = boxes[order]
    max_scores = max_scores[order]

    tokenized = model.tokenizer(caption)
    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, model.tokenizer)
        .replace(".", "")
        .strip()
        for logit in logits
    ]

    height, width, _ = image_source.shape
    scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
    boxes_xyxy = box_convert(boxes=boxes * scale, in_fmt="cxcywh", out_fmt="xyxy")
    boxes_xyxy[:, 0::2].clamp_(0, width)
    boxes_xyxy[:, 1::2].clamp_(0, height)
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=0) * (
        boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    ).clamp(min=0)

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    colors = [(230, 20, 20), (30, 70, 210), (35, 140, 50), (180, 100, 20), (120, 20, 170)]
    for idx, (box, score, phrase) in enumerate(zip(boxes_xyxy.tolist(), max_scores.tolist(), phrases)):
        color = colors[idx % len(colors)]
        draw.rectangle([float(value) for value in box], outline=color, width=3)
        draw.text(
            (float(box[0]) + 2, max(0.0, float(box[1]) - 12)),
            f"{phrase or '<empty>'} {score:.3f}",
            fill=color,
        )
    annotated.save(annotated_path)

    artifact = {
        "task_id": TASK_ID,
        "repo_commit": git_commit(repo_dir),
        "checkpoint": {
            "filename": checkpoint_path.name,
            "size_bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": "cpu",
            "build_seconds": round(build_seconds, 4),
            "load_seconds": round(load_seconds, 4),
            "forward_seconds": round(forward_seconds, 4),
        },
        "input": {
            "path": "artifacts/expected_input.png",
            "width": width,
            "height": height,
            "sha256": sha256_file(input_path),
            "transform": "groundingdino.datasets.transforms.Compose(RandomResize([256]), ToTensor, Normalize)",
            "tensor_shape": list(image_tensor.shape),
        },
        "caption": caption,
        "thresholds": {"box_threshold": box_threshold, "text_threshold": text_threshold},
        "detections": [
            {
                "rank": int(idx),
                "phrase": phrase,
                "score": round(float(score), 10),
                "box_cxcywh": [round(float(value), 10) for value in box.tolist()],
                "box_xyxy": [round(float(value), 4) for value in xyxy.tolist()],
                "area_pixels": round(float(area), 4),
            }
            for idx, (phrase, score, box, xyxy, area) in enumerate(
                zip(phrases, max_scores, boxes, boxes_xyxy, areas)
            )
        ],
        "output_tensors": {
            "pred_logits_shape": list(outputs["pred_logits"].shape),
            "pred_boxes_shape": list(outputs["pred_boxes"].shape),
            "selected_count": int(len(phrases)),
        },
        "load_state_dict": {
            "missing_keys": len(load_result.missing_keys),
            "unexpected_keys": len(load_result.unexpected_keys),
        },
    }
    artifact["checks"] = build_checks(artifact, output_dir)
    artifact["success_level"] = "L4" if all(artifact["checks"].values()) else "below_L4"

    artifact_path = output_dir / "expected_artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact_path


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def build_checks(payload: dict[str, Any], artifact_dir: pathlib.Path) -> dict[str, bool]:
    detections = payload.get("detections", [])
    phrases = {str(item.get("phrase", "")).strip() for item in detections if isinstance(item, dict)}
    scores = [item.get("score") for item in detections if isinstance(item, dict)]
    boxes = [item.get("box_xyxy") for item in detections if isinstance(item, dict)]
    areas = [item.get("area_pixels") for item in detections if isinstance(item, dict)]
    width = payload.get("input", {}).get("width")
    height = payload.get("input", {}).get("height")
    input_path = artifact_dir / "expected_input.png"
    annotated_path = artifact_dir / "expected_annotated.png"

    boxes_are_finite = True
    boxes_in_bounds = True
    if not isinstance(width, int) or not isinstance(height, int):
        boxes_are_finite = False
        boxes_in_bounds = False
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4 or not all(finite_number(value) for value in box):
            boxes_are_finite = False
            boxes_in_bounds = False
            continue
        x0, y0, x1, y1 = [float(value) for value in box]
        if not (0 <= x0 <= width and 0 <= x1 <= width and 0 <= y0 <= height and 0 <= y1 <= height):
            boxes_in_bounds = False

    checks = {
        "repo_commit_matches": payload.get("repo_commit") == EXPECTED_COMMIT,
        "checkpoint_loaded": payload.get("checkpoint", {}).get("sha256") == EXPECTED_CHECKPOINT_SHA256
        and payload.get("checkpoint", {}).get("size_bytes") == EXPECTED_CHECKPOINT_SIZE,
        "uses_groundingdino_model_forward": payload.get("output_tensors", {}).get("pred_logits_shape")
        == EXPECTED_LOGITS_SHAPE
        and payload.get("output_tensors", {}).get("pred_boxes_shape") == EXPECTED_BOXES_SHAPE,
        "uses_text_phrase_extraction": len(phrases) > 0 and all(isinstance(phrase, str) for phrase in phrases),
        "at_least_one_box": len(detections) >= 1,
        "expected_phrases_present": EXPECTED_PHRASES.issubset(phrases),
        "boxes_are_finite": boxes_are_finite,
        "boxes_in_image_bounds": boxes_in_bounds,
        "nonzero_box_area": any(finite_number(area) and float(area) > 1.0 for area in areas),
        "scores_are_finite": all(finite_number(score) and 0.0 <= float(score) <= 1.0 for score in scores),
        "input_png_matches": input_path.exists() and sha256_file(input_path) == EXPECTED_INPUT_SHA256,
        "annotated_png_written": annotated_path.exists() and annotated_path.stat().st_size > 0,
    }
    return checks


def validate_artifact(artifact_dir: pathlib.Path, artifact_name: str = "expected_artifact.json") -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    artifact_path = artifact_dir / artifact_name
    if not artifact_path.exists():
        raise AssertionError(f"missing artifact: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if payload.get("task_id") != TASK_ID:
        raise AssertionError(f"wrong task_id: {payload.get('task_id')}")
    if payload.get("success_level") != "L4":
        raise AssertionError(f"unexpected success_level: {payload.get('success_level')}")

    checks = build_checks(payload, artifact_dir)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError({"failed_checks": failed, "payload_checks": payload.get("checks", {})})

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": "L4",
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "checks": checks,
        "observed": {
            "repo_commit": payload.get("repo_commit"),
            "selected_count": payload.get("output_tensors", {}).get("selected_count"),
            "phrases": [item.get("phrase") for item in payload.get("detections", [])],
            "top_score": payload.get("detections", [{}])[0].get("score"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify GroundingDINO phrase-grounding gold artifact.")
    parser.add_argument("--repo-dir", default=os.environ.get("PAPERENVBENCH_REPO_DIR", "repo"))
    parser.add_argument(
        "--checkpoint-path",
        default=os.environ.get(
            "GROUNDINGDINO_CHECKPOINT_PATH",
            os.environ.get("GROUNDINGDINO_CHECKPOINT", "models/groundingdino_swint_ogc.pth"),
        ),
    )
    parser.add_argument("--output-dir", default=os.environ.get("PAPERENVBENCH_OUTPUT_DIR", "artifacts"))
    parser.add_argument("--artifact-name", default="expected_artifact.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    repo_dir = pathlib.Path(args.repo_dir)
    checkpoint_path = pathlib.Path(args.checkpoint_path)
    output_dir = pathlib.Path(args.output_dir)
    model_ready = (repo_dir / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py").exists() and checkpoint_path.exists()

    try:
        if args.check_only or not model_ready:
            artifact_dir = output_dir if (output_dir / args.artifact_name).exists() else TASK_ROOT / "artifacts"
            result = validate_artifact(artifact_dir, args.artifact_name)
            result["mode"] = "check_only"
        else:
            artifact_path = run_model(repo_dir, checkpoint_path, output_dir)
            result = validate_artifact(artifact_path.parent, args.artifact_name)
            result["mode"] = "model_run"
    except Exception as exc:
        failure = {"task_id": TASK_ID, "status": "fail", "error": str(exc)}
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
