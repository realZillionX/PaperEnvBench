#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any


TASK_ID = "detr_object_detection_minimal"
REPO_URL = "https://github.com/facebookresearch/detr"
EXPECTED_COMMIT = "29901c51d7fe8712168b8d0d64351170bc0f83e0"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
CHECKPOINT_SHA256 = "e632da11ec76ae67bac2f8579fbed3724e08dead7d200ca13e019b197784eadc"
COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
COCO_IMAGE_SHA256 = "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e"
MIN_HIGH_CONFIDENCE_SCORE = 0.7
MIN_CAT_SCORE = 0.95

LABEL_NAMES = {
    17: "cat",
    63: "couch",
    65: "bed",
    75: "remote",
}


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


def git_commit(repo_dir: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        stderr=subprocess.STDOUT,
        text=True,
    ).strip()


def ensure_input_image(input_path: Path) -> str:
    if input_path.exists() and input_path.stat().st_size > 0:
        if sha256_file(input_path) == COCO_IMAGE_SHA256:
            return COCO_IMAGE_URL
        return "existing_local_input"

    try:
        urllib.request.urlretrieve(COCO_IMAGE_URL, input_path)
        return COCO_IMAGE_URL
    except Exception as exc:
        import numpy as np
        from PIL import Image

        arr = np.zeros((320, 480, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(30, 230, arr.shape[1], dtype=np.uint8)[None, :]
        arr[:, :, 1] = np.linspace(230, 30, arr.shape[0], dtype=np.uint8)[:, None]
        arr[80:240, 120:360, :] = np.array([210, 210, 180], dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(input_path)
        return f"synthetic_fallback_due_to_download_error: {exc}"


def rounded_box(row: Any) -> list[float]:
    return [round(float(value), 4) for value in row.tolist()]


def run_inference(repo_dir: Path, output_dir: Path, checkpoint_dir: Path) -> dict[str, Any]:
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    from torchvision import transforms as transforms

    repo_dir = repo_dir.resolve()
    output_dir = output_dir.resolve()
    checkpoint_dir = checkpoint_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(checkpoint_dir))

    if not (repo_dir / "hubconf.py").exists():
        raise FileNotFoundError(f"missing DETR hubconf.py under repo dir: {repo_dir}")

    input_path = output_dir / "expected_input.jpg"
    image_source = ensure_input_image(input_path)

    sys.path.insert(0, str(repo_dir))
    import models.backbone as detr_backbone

    detr_backbone.is_main_process = lambda: False

    import hubconf

    torch.set_num_threads(2)
    torch.manual_seed(20260516)

    model, postprocessor = hubconf.detr_resnet50(pretrained=True, return_postprocessor=True)
    model.eval()

    image = Image.open(input_path).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize(320),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        processed = postprocessor(outputs, torch.tensor([image.size[::-1]], dtype=torch.float32))[0]

    scores = processed["scores"].detach().cpu()
    labels = processed["labels"].detach().cpu()
    boxes = processed["boxes"].detach().cpu()
    order = torch.argsort(scores, descending=True)

    detections: list[dict[str, Any]] = []
    for idx in order[:10].tolist():
        label_id = int(labels[idx].item())
        detections.append(
            {
                "rank": len(detections) + 1,
                "score": round(float(scores[idx].item()), 6),
                "label_id": label_id,
                "label_name": LABEL_NAMES.get(label_id, f"id_{label_id}"),
                "box_xyxy": rounded_box(boxes[idx]),
            }
        )

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for detection in detections[:5]:
        if detection["score"] < 0.3:
            continue
        x0, y0, x1, y1 = detection["box_xyxy"]
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
        draw.text((x0 + 3, max(0, y0 + 3)), f"{detection['label_name']} {detection['score']:.2f}", fill=(255, 0, 0))
    overlay_path = output_dir / "expected_detection_overlay.jpg"
    overlay.save(overlay_path, quality=90)

    checkpoint_path = checkpoint_dir / "hub" / "checkpoints" / "detr-r50-e632da11.pth"
    high_confidence = [detection for detection in detections if detection["score"] >= MIN_HIGH_CONFIDENCE_SCORE]
    high_confidence_boxes_in_bounds = True
    for detection in high_confidence:
        x0, y0, x1, y1 = detection["box_xyxy"]
        if not (-1.0 <= x0 <= image.width + 1.0 and -1.0 <= x1 <= image.width + 1.0):
            high_confidence_boxes_in_bounds = False
        if not (-1.0 <= y0 <= image.height + 1.0 and -1.0 <= y1 <= image.height + 1.0):
            high_confidence_boxes_in_bounds = False

    repo_commit = git_commit(repo_dir)
    checks = {
        "repo_commit_matches": repo_commit == EXPECTED_COMMIT,
        "checkpoint_file_exists": checkpoint_path.exists() and checkpoint_path.stat().st_size > 100_000_000,
        "checkpoint_sha256_matches": checkpoint_path.exists() and sha256_file(checkpoint_path) == CHECKPOINT_SHA256,
        "coco_input_image_available": sha256_file(input_path) == COCO_IMAGE_SHA256,
        "repository_model_loaded": model.__class__.__module__ == "models.detr",
        "pred_logits_shape_matches": list(outputs["pred_logits"].shape) == [1, 100, 92],
        "pred_boxes_shape_matches": list(outputs["pred_boxes"].shape) == [1, 100, 4],
        "scores_finite": bool(torch.isfinite(scores).all().item()),
        "boxes_finite": bool(torch.isfinite(boxes).all().item()),
        "high_confidence_boxes_within_tolerance": high_confidence_boxes_in_bounds,
        "high_confidence_detection_count": len(high_confidence) >= 3,
        "cat_detected_for_coco_image": any(
            detection["label_id"] == 17 and detection["score"] >= MIN_CAT_SCORE for detection in detections[:5]
        ),
        "overlay_written": overlay_path.exists() and overlay_path.stat().st_size > 0,
    }

    summary = {
        "task_id": TASK_ID,
        "success_level": "L4" if all(checks.values()) else "below_L4",
        "repo_url": REPO_URL,
        "repo_commit": repo_commit,
        "python": sys.version.split()[0],
        "package_versions": {
            "torch": torch.__version__,
            "torchvision": metadata.version("torchvision"),
            "numpy": np.__version__,
            "pillow": Image.__version__,
            "scipy": metadata.version("scipy"),
        },
        "model": {
            "entrypoint": "hubconf.detr_resnet50(pretrained=True, return_postprocessor=True)",
            "class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "num_queries": 100,
            "parameter_count": int(sum(param.numel() for param in model.parameters())),
            "training": bool(model.training),
        },
        "checkpoint": {
            "url": CHECKPOINT_URL,
            "path": str(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
            "sha256": sha256_file(checkpoint_path) if checkpoint_path.exists() else None,
        },
        "input_image": {
            "source": image_source,
            "path": str(input_path),
            "width": image.width,
            "height": image.height,
            "sha256": sha256_file(input_path),
            "size_bytes": input_path.stat().st_size,
        },
        "raw_output": {
            "pred_logits_shape": list(outputs["pred_logits"].shape),
            "pred_boxes_shape": list(outputs["pred_boxes"].shape),
            "pred_logits_sha256": hashlib.sha256(
                outputs["pred_logits"].detach().cpu().numpy().astype("<f4").tobytes()
            ).hexdigest(),
            "pred_boxes_sha256": hashlib.sha256(
                outputs["pred_boxes"].detach().cpu().numpy().astype("<f4").tobytes()
            ).hexdigest(),
        },
        "detections_top10": detections,
        "artifact_files": {
            "summary": "expected_artifact.json",
            "input_image": input_path.name,
            "overlay_image": overlay_path.name,
        },
        "checks": checks,
    }

    artifact_path = output_dir / "expected_artifact.json"
    artifact_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def verify_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    artifact_path = artifact_dir / "expected_artifact.json"
    input_path = artifact_dir / "expected_input.jpg"
    overlay_path = artifact_dir / "expected_detection_overlay.jpg"
    payload = load_json(artifact_path)

    required_top_level = {
        "task_id",
        "repo_commit",
        "checkpoint",
        "input_image",
        "raw_output",
        "detections_top10",
        "checks",
        "success_level",
    }
    missing = sorted(required_top_level - set(payload))
    if missing:
        raise AssertionError(f"artifact missing fields: {missing}")
    if payload["task_id"] != TASK_ID:
        raise AssertionError("artifact has wrong task_id")
    if payload["repo_commit"] != EXPECTED_COMMIT:
        raise AssertionError("repo_commit does not match DETR pinned commit")
    if payload["checkpoint"].get("sha256") != CHECKPOINT_SHA256:
        raise AssertionError("checkpoint sha256 does not match detr-r50-e632da11.pth")
    if payload["checkpoint"].get("size_bytes", 0) < 100_000_000:
        raise AssertionError("checkpoint size is too small for DETR R50")
    if payload["input_image"].get("sha256") != COCO_IMAGE_SHA256:
        raise AssertionError("input image must be the pinned COCO validation image")
    if not input_path.exists() or sha256_file(input_path) != COCO_IMAGE_SHA256:
        raise AssertionError("expected_input.jpg is missing or has the wrong sha256")
    if not overlay_path.exists() or overlay_path.stat().st_size <= 0:
        raise AssertionError("expected_detection_overlay.jpg is missing or empty")
    if payload["raw_output"].get("pred_logits_shape") != [1, 100, 92]:
        raise AssertionError("pred_logits shape must be [1, 100, 92]")
    if payload["raw_output"].get("pred_boxes_shape") != [1, 100, 4]:
        raise AssertionError("pred_boxes shape must be [1, 100, 4]")

    detections = payload["detections_top10"]
    if not isinstance(detections, list) or len(detections) < 5:
        raise AssertionError("detections_top10 must contain at least five detections")
    top = detections[0]
    if top.get("label_id") != 17 or float(top.get("score", 0.0)) < MIN_CAT_SCORE:
        raise AssertionError("top detection must be a high-confidence cat")
    high_confidence_cats = [
        detection
        for detection in detections[:5]
        if detection.get("label_id") == 17 and float(detection.get("score", 0.0)) >= MIN_CAT_SCORE
    ]
    if len(high_confidence_cats) < 2:
        raise AssertionError("expected at least two high-confidence cat detections in top five")

    checks = payload["checks"]
    failed = sorted(key for key, ok in checks.items() if ok is not True)
    if failed:
        raise AssertionError(f"artifact semantic checks failed: {failed}")

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": payload["success_level"],
        "artifact_dir": str(artifact_dir),
        "observed": {
            "repo_commit": payload["repo_commit"],
            "checkpoint_sha256": payload["checkpoint"]["sha256"],
            "input_sha256": payload["input_image"]["sha256"],
            "top_detection": top,
            "num_high_confidence_cats": len(high_confidence_cats),
        },
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and verify the PaperEnvBench DETR minimal detection artifact.")
    parser.add_argument("--repo-dir", default=os.environ.get("PAPERENVBENCH_REPO_DIR", "repo"))
    parser.add_argument("--output-dir", default=os.environ.get("PAPERENVBENCH_OUTPUT_DIR", "artifacts"))
    parser.add_argument("--checkpoint-dir", default=os.environ.get("PAPERENVBENCH_CHECKPOINT_DIR", "checkpoints"))
    parser.add_argument("--check-only", action="store_true", help="Only validate an existing artifact directory.")
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()

    try:
        if not args.check_only:
            run_inference(Path(args.repo_dir), Path(args.output_dir), Path(args.checkpoint_dir))
        result = verify_artifact(Path(args.output_dir))
    except Exception as exc:
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
