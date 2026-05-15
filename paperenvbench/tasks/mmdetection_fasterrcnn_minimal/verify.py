#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys
import traceback
from typing import Any


TASK_ID = "mmdetection_fasterrcnn_minimal"
EXPECTED_COMMIT = "cfd5d3a985b0249de009b67d04f37263e11cdf3d"
EXPECTED_IMAGE_SIZE = [128, 96]
MIN_MATCH_IOU = 0.72
MIN_VIS_SIZE_BYTES = 1024


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def round_float(value: float) -> float:
    return round(float(value), 10)


def git_commit(repo_dir: pathlib.Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return ""


def load_repo_module(repo_dir: pathlib.Path, relative_path: str, module_name: str):
    module_path = repo_dir / relative_path
    if not module_path.exists():
        raise FileNotFoundError(f"missing repository source file: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attempt_faster_rcnn_config(repo_dir: pathlib.Path) -> dict[str, Any]:
    config_path = repo_dir / "configs" / "_base_" / "models" / "faster-rcnn_r50_fpn.py"
    try:
        from mmengine.config import Config

        cfg = Config.fromfile(config_path)
        model_type = cfg.model.get("type") if hasattr(cfg, "model") else None
        return {
            "ok": True,
            "config": str(config_path),
            "model_type": model_type,
            "error_type": None,
            "error": None,
        }
    except Exception as exc:  # MMDetection native config needs mmcv ops on this CPU image.
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return {
            "ok": False,
            "config": str(config_path),
            "model_type": "FasterRCNN",
            "error_type": type(exc).__name__,
            "error": error,
        }


def attempt_mmcv_ops() -> dict[str, Any]:
    try:
        from mmcv.ops import RoIAlign, nms

        return {
            "ok": True,
            "ops": {
                "RoIAlign": repr(RoIAlign),
                "nms": repr(nms),
            },
            "error_type": None,
            "error": None,
            "expected_blocker": False,
        }
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return {
            "ok": False,
            "ops": ["RoIAlign", "nms"],
            "error_type": type(exc).__name__,
            "error": error,
            "expected_blocker": ("mmcv._ext" in error or "mmcv.ops" in error),
        }


def write_visualization(path: pathlib.Path, detections: list[dict[str, Any]]) -> dict[str, Any]:
    width, height = EXPECTED_IMAGE_SIZE
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            pixels.extend(((23 + x * 2) % 256, (37 + y * 3) % 256, (91 + x + y) % 256))

    colors = {
        "target": (250, 250, 250),
        "det_0": (255, 32, 48),
        "det_1": (32, 224, 128),
        "det_2": (48, 128, 255),
    }

    def set_pixel(px: int, py: int, color: tuple[int, int, int]) -> None:
        if 0 <= px < width and 0 <= py < height:
            offset = (py * width + px) * 3
            pixels[offset : offset + 3] = bytes(color)

    def draw_box(box: list[float], color: tuple[int, int, int], thickness: int = 2) -> None:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        for t in range(thickness):
            for px in range(x1, x2 + 1):
                set_pixel(px, y1 + t, color)
                set_pixel(px, y2 - t, color)
            for py in range(y1, y2 + 1):
                set_pixel(x1 + t, py, color)
                set_pixel(x2 - t, py, color)

    targets = [[12.0, 10.0, 60.0, 58.0], [70.0, 20.0, 110.0, 76.0]]
    for box in targets:
        draw_box(box, colors["target"], thickness=1)
    for idx, det in enumerate(detections):
        draw_box(det["bbox"], colors.get(f"det_{idx}", (255, 255, 0)), thickness=2)

    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + bytes(pixels))
    return {
        "format": "ppm_p6",
        "width": width,
        "height": height,
        "channels": 3,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def check_existing(output_dir: pathlib.Path) -> dict[str, Any]:
    expected_path = pathlib.Path(__file__).resolve().parent / "expected_output.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    expected_observed = expected["gold_observed"]
    summary_path = output_dir / expected["required_artifact"]
    visual_path = output_dir / expected["required_side_artifact"]

    if not summary_path.exists() or summary_path.stat().st_size <= 0:
        raise AssertionError(f"missing nonempty summary artifact: {summary_path}")
    if not visual_path.exists() or visual_path.stat().st_size <= 0:
        raise AssertionError(f"missing nonempty visualization artifact: {visual_path}")
    if sha256_file(summary_path) != expected_observed["artifact_sha256"]:
        raise AssertionError("summary artifact checksum mismatch")
    if sha256_file(visual_path) != expected_observed["visualization_sha256"]:
        raise AssertionError("visualization checksum mismatch")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("task_id") != TASK_ID:
        raise AssertionError(f"wrong task_id: {payload.get('task_id')}")
    if payload.get("repo_commit") != EXPECTED_COMMIT:
        raise AssertionError(f"wrong repo_commit: {payload.get('repo_commit')}")
    if payload.get("success_level") != expected["expected_success_level"]:
        raise AssertionError(f"wrong success_level: {payload.get('success_level')}")
    checks = payload.get("checks", {})
    if not isinstance(checks, dict) or not all(checks.values()):
        raise AssertionError(f"semantic checks failed: {checks}")

    native_ops = payload.get("native_mmcv_ops", {})
    if native_ops.get("ok") is not False or native_ops.get("expected_blocker") is not True:
        raise AssertionError(f"native mmcv blocker was not recorded: {native_ops}")

    output = payload.get("output", {})
    kept = output.get("kept_detections", [])
    if [item.get("source_index") for item in kept] != expected["semantic_thresholds"]["expected_kept_source_indices"]:
        raise AssertionError(f"kept detections mismatch: {kept}")
    if visual_path.stat().st_size < expected["semantic_thresholds"]["min_visualization_size_bytes"]:
        raise AssertionError("visualization artifact is too small")

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": payload["success_level"],
        "mode": "check_only",
        "artifact_path": str(summary_path.resolve()),
        "artifact_sha256": expected_observed["artifact_sha256"],
        "observed": {
            "repo_commit": payload["repo_commit"],
            "kept_source_indices": [item.get("source_index") for item in kept],
            "native_mmcv_ops": native_ops,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the PaperEnvBench MMDetection Faster R-CNN minimal task output.")
    parser.add_argument("--repo-dir", default=os.environ.get("PAPERENVBENCH_REPO_DIR", "repo"))
    parser.add_argument("--output-dir", default=os.environ.get("PAPERENVBENCH_OUTPUT_DIR", "artifacts"))
    parser.add_argument("--artifact-name", default="expected_artifact.json")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_dir = pathlib.Path(args.repo_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.check_only:
        try:
            result = check_existing(output_dir)
        except AssertionError as exc:
            print(json.dumps({"task_id": TASK_ID, "status": "fail", "error": str(exc)}, indent=2), file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if not (repo_dir / "mmdet" / "__init__.py").exists():
        raise FileNotFoundError(f"missing MMDetection source tree: {repo_dir}")

    sys.path.insert(0, str(repo_dir))

    import mmcv
    import mmengine
    import mmdet
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(20260516)

    repo_commit = git_commit(repo_dir)
    native_config = attempt_faster_rcnn_config(repo_dir)
    native_mmcv_ops = attempt_mmcv_ops()

    bbox_module = load_repo_module(
        repo_dir,
        "mmdet/evaluation/functional/bbox_overlaps.py",
        "paperenvbench_mmdet_bbox_overlaps",
    )
    matrix_nms_module = load_repo_module(
        repo_dir,
        "mmdet/models/layers/matrix_nms.py",
        "paperenvbench_mmdet_matrix_nms",
    )

    gt_boxes = np.array(
        [[12.0, 10.0, 60.0, 58.0], [70.0, 20.0, 110.0, 76.0]],
        dtype=np.float32,
    )
    pred_boxes = np.array(
        [
            [14.0, 12.0, 59.0, 55.0],
            [68.0, 22.0, 112.0, 78.0],
            [6.0, 5.0, 32.0, 30.0],
        ],
        dtype=np.float32,
    )
    pred_scores = torch.tensor([0.93, 0.84, 0.18], dtype=torch.float32)
    pred_labels = torch.tensor([0, 1, 0], dtype=torch.long)

    iou_matrix = bbox_module.bbox_overlaps(pred_boxes, gt_boxes)
    best_ious = iou_matrix.max(axis=1)
    best_targets = iou_matrix.argmax(axis=1)

    masks = torch.zeros((3, EXPECTED_IMAGE_SIZE[1], EXPECTED_IMAGE_SIZE[0]), dtype=torch.bool)
    for idx, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        masks[idx, y1:y2, x1:x2] = True
    kept_scores, kept_labels, kept_masks, keep_inds = matrix_nms_module.mask_matrix_nms(
        masks,
        pred_labels,
        pred_scores,
        filter_thr=0.2,
        nms_pre=-1,
        max_num=2,
        kernel="gaussian",
        sigma=2.0,
    )

    detections = []
    for rank, keep_idx in enumerate(keep_inds.tolist()):
        detections.append(
            {
                "rank": rank,
                "source_index": int(keep_idx),
                "bbox": [round_float(v) for v in pred_boxes[keep_idx].tolist()],
                "label": int(pred_labels[keep_idx].item()),
                "score_after_matrix_nms": round_float(kept_scores[rank].item()),
                "best_target": int(best_targets[keep_idx]),
                "best_iou": round_float(float(best_ious[keep_idx])),
                "mask_pixels": int(kept_masks[rank].sum().item()),
            }
        )

    visual_path = output_dir / "expected_detection.ppm"
    visual_info = write_visualization(visual_path, detections)

    iou_values = [[round_float(v) for v in row] for row in iou_matrix.tolist()]
    checks = {
        "repo_commit_matches": repo_commit == EXPECTED_COMMIT,
        "mmdet_version_imported": bool(getattr(mmdet, "__version__", "")),
        "native_faster_rcnn_blocker_recorded": (not native_mmcv_ops["ok"]) and bool(native_mmcv_ops.get("expected_blocker")),
        "bbox_overlap_function_executed": list(iou_matrix.shape) == [3, 2],
        "matrix_nms_function_executed": len(detections) == 2 and [d["source_index"] for d in detections] == [0, 1],
        "detections_match_synthetic_targets": all(det["best_iou"] >= MIN_MATCH_IOU for det in detections),
        "visualization_artifact_written": visual_path.exists() and visual_path.stat().st_size >= MIN_VIS_SIZE_BYTES,
    }

    summary = {
        "task_id": TASK_ID,
        "repo_commit": repo_commit,
        "python": sys.version.split()[0],
        "package_versions": {
            "torch": torch.__version__,
            "torchvision": metadata.version("torchvision"),
            "mmcv": mmcv.__version__,
            "mmengine": mmengine.__version__,
            "mmdet": mmdet.__version__,
            "numpy": np.__version__,
        },
        "native_faster_rcnn_config": native_config,
        "native_mmcv_ops": native_mmcv_ops,
        "official_repo_code_paths": [
            "mmdet/__init__.py",
            "configs/_base_/models/faster-rcnn_r50_fpn.py",
            "mmdet/evaluation/functional/bbox_overlaps.py",
            "mmdet/models/layers/matrix_nms.py",
        ],
        "input": {
            "image_size": EXPECTED_IMAGE_SIZE,
            "gt_boxes_xyxy": [[round_float(v) for v in row] for row in gt_boxes.tolist()],
            "pred_boxes_xyxy": [[round_float(v) for v in row] for row in pred_boxes.tolist()],
            "pred_scores": [round_float(v) for v in pred_scores.tolist()],
            "pred_labels": pred_labels.tolist(),
        },
        "output": {
            "iou_matrix": iou_values,
            "iou_matrix_sha256": sha256_bytes(iou_matrix.astype(np.float32).tobytes()),
            "kept_detections": detections,
            "kept_labels": [int(v) for v in kept_labels.tolist()],
            "visualization": visual_info,
        },
        "checks": checks,
        "success_level": "L4_fallback" if all(checks.values()) else "below_L4",
    }

    artifact_path = output_dir / args.artifact_name
    artifact_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not all(checks.values()):
        failed = ", ".join(k for k, ok in checks.items() if not ok)
        raise SystemExit(f"semantic checks failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
