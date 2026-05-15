#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import pathlib
import subprocess
import sys
import traceback
from typing import Any


TASK_ROOT = pathlib.Path(__file__).resolve().parent
TASK_ID = "mmaction2_recognition_minimal"
EXPECTED_COMMIT = "a5a167dff2399e2d182a60332325f9c0d4663517"
CONFIG_RELATIVE = "configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py"
EXPECTED_VIDEO_FRAMES = 16
EXPECTED_INPUT_SHAPE = [1, 3, 3, 224, 224]
MIN_PARAMETER_COUNT = 20_000_000
MIN_FEATURE_STD = 1e-4


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_float(value: float) -> float:
    return round(float(value), 10)


def load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssertionError(f"JSON output is not parseable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AssertionError(f"JSON output must be an object: {path}")
    return payload


def git_commit(repo_dir: pathlib.Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return ""


def package_version(dist_name: str) -> str:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def make_synthetic_video(path: pathlib.Path) -> dict[str, Any]:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 128, 96
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {path}")
    for frame_idx in range(EXPECTED_VIDEO_FRAMES):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = np.linspace(10, 90, width, dtype=np.uint8)
        frame[:, :, 1] = np.linspace(20, 120, height, dtype=np.uint8)[:, None]
        frame[:, :, 2] = (frame_idx * 9) % 255
        x0 = 8 + frame_idx * 6
        cv2.rectangle(frame, (x0, 30), (x0 + 20, 60), (30, 180, 250), -1)
        cv2.circle(frame, (96, 48), 8 + frame_idx % 4, (200, 60, 30), -1)
        writer.write(frame)
    writer.release()
    return {
        "path": "artifacts/synthetic_action.mp4",
        "width": width,
        "height": height,
        "frames": EXPECTED_VIDEO_FRAMES,
        "fps": 8.0,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def probe_native_mmcv(run_root: pathlib.Path) -> dict[str, Any]:
    probe_log = run_root / "logs" / "mmcv_native_binary_probe.log"
    text = probe_log.read_text(encoding="utf-8", errors="replace") if probe_log.exists() else ""
    try:
        from mmcv.ops import RoIAlign, nms  # type: ignore

        return {
            "native_mmcv_ops_available": True,
            "expected_blocker": False,
            "ops": {"RoIAlign": repr(RoIAlign), "nms": repr(nms)},
            "binary_probe_log": "logs/mmcv_native_binary_probe.log",
            "binary_probe_mentions_no_wheel": "No matching distribution found" in text,
        }
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return {
            "native_mmcv_ops_available": False,
            "expected_blocker": "mmcv._ext" in error or "mmcv.ops" in error,
            "error_type": type(exc).__name__,
            "error": error,
            "binary_probe_log": "logs/mmcv_native_binary_probe.log",
            "binary_probe_mentions_no_wheel": "No matching distribution found" in text,
        }


def run_gold(repo_dir: pathlib.Path, output_dir: pathlib.Path) -> dict[str, Any]:
    if not (repo_dir / "mmaction" / "__init__.py").exists():
        raise FileNotFoundError(f"missing MMACTION2 source tree: {repo_dir}")
    sys.path.insert(0, str(repo_dir))

    import torch
    from mmengine.config import Config
    from mmengine.dataset import Compose
    from mmaction.registry import MODELS
    from mmaction.utils import register_all_modules

    torch.set_num_threads(1)
    torch.manual_seed(20260516)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_root = output_dir.parent

    video_path = output_dir / "synthetic_action.mp4"
    video_info = make_synthetic_video(video_path)

    register_all_modules(init_default_scope=True)
    cfg_path = repo_dir / CONFIG_RELATIVE
    cfg = Config.fromfile(cfg_path)
    pipeline_cfg = [dict(item) for item in cfg.test_pipeline]
    pipeline_types = [item["type"] for item in pipeline_cfg]
    for item in pipeline_cfg:
        if item["type"] == "SampleFrames":
            item.update(clip_len=1, frame_interval=1, num_clips=3, test_mode=True)
        if item["type"] == "TenCrop":
            item.update(type="CenterCrop", crop_size=224)
    pipeline = Compose(pipeline_cfg)
    data = pipeline({"filename": str(video_path), "label": 0, "start_index": 0, "modality": "RGB"})
    inputs = (data["inputs"].float() / 255.0).unsqueeze(0)

    model_cfg = cfg.model
    model_cfg.backbone.pretrained = None
    model_cfg.cls_head.num_classes = 5
    model = MODELS.build(model_cfg)
    model.eval()

    with torch.no_grad():
        predictions = model(inputs, data_samples=[data["data_samples"]], mode="predict")
        features = model.extract_feat(inputs)[0]

    scores = predictions[0].pred_score.detach().cpu()
    top_values, top_indices = torch.topk(scores, k=5)
    feature_stats = {
        "shape": list(features.shape),
        "mean": round_float(features.mean().item()),
        "std": round_float(features.std().item()),
        "min": round_float(features.min().item()),
        "max": round_float(features.max().item()),
    }
    score_list = [round_float(value) for value in scores.tolist()]
    checks = {
        "repo_commit_matches": git_commit(repo_dir) == EXPECTED_COMMIT,
        "config_loaded": cfg_path.exists() and cfg.model.type == "Recognizer2D",
        "pipeline_decoded_video": list(inputs.shape) == EXPECTED_INPUT_SHAPE,
        "recognizer_built_from_registry": type(model).__module__.startswith("mmaction.models.recognizers"),
        "backbone_is_resnet": model_cfg.backbone.type == "ResNet",
        "head_is_tsn_head": model_cfg.cls_head.type == "TSNHead",
        "parameter_count_large_enough": sum(p.numel() for p in model.parameters()) >= MIN_PARAMETER_COUNT,
        "prediction_shape_matches": list(scores.shape) == [5],
        "scores_are_finite": all(math.isfinite(float(value)) for value in score_list),
        "scores_are_normalized": abs(float(scores.sum().item()) - 1.0) <= 1e-6,
        "features_have_variance": feature_stats["std"] > MIN_FEATURE_STD,
        "synthetic_video_written": video_path.exists() and video_path.stat().st_size > 1024,
    }
    native_mmcv = probe_native_mmcv(run_root)

    summary = {
        "task_id": TASK_ID,
        "repo_url": "https://github.com/open-mmlab/mmaction2",
        "repo_commit": git_commit(repo_dir),
        "config": CONFIG_RELATIVE,
        "python": sys.version.split()[0],
        "package_versions": {
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
            "mmengine": package_version("mmengine"),
            "mmcv-lite": package_version("mmcv-lite"),
            "mmaction2": package_version("mmaction2"),
            "decord": package_version("decord"),
            "opencv-python-headless": package_version("opencv-python-headless"),
            "importlib_metadata": package_version("importlib_metadata"),
        },
        "fallback": {
            "used": True,
            "success_level_reason": "CPU Python 3.12 uses mmcv-lite and random-initialized TSN weights; no native mmcv wheel or pretrained Kinetics checkpoint is required for the gold artifact.",
            "native_mmcv": native_mmcv,
        },
        "official_repo_code_paths": [
            CONFIG_RELATIVE,
            "mmaction/datasets/transforms/loading.py",
            "mmaction/datasets/transforms/formatting.py",
            "mmaction/models/recognizers/recognizer2d.py",
            "mmaction/models/backbones/resnet.py",
            "mmaction/models/heads/tsn_head.py",
        ],
        "model": {
            "type": cfg.model.type,
            "backbone": model_cfg.backbone.type,
            "head": model_cfg.cls_head.type,
            "num_classes": model_cfg.cls_head.num_classes,
            "parameter_count": sum(p.numel() for p in model.parameters()),
        },
        "pipeline": {
            "original_types": pipeline_types,
            "gold_types": [item["type"] for item in pipeline_cfg],
            "input_tensor_shape": list(inputs.shape),
            "input_tensor_mean": round_float(inputs.mean().item()),
            "input_tensor_std": round_float(inputs.std().item()),
        },
        "video": video_info,
        "output": {
            "pred_label": int(predictions[0].pred_label.item()),
            "scores": score_list,
            "top5_indices": [int(value) for value in top_indices.tolist()],
            "top5_scores": [round_float(value) for value in top_values.tolist()],
            "score_sum": round_float(scores.sum().item()),
            "feature_stats": feature_stats,
        },
        "checks": checks,
        "success_level": "L4_fallback" if all(checks.values()) else "below_L4",
    }
    artifact_path = output_dir / "expected_artifact.json"
    artifact_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not all(checks.values()):
        failed = ", ".join(name for name, ok in checks.items() if not ok)
        raise SystemExit(f"semantic checks failed: {failed}")
    return summary


def check_existing(attempt_root: pathlib.Path) -> dict[str, Any]:
    expected = load_json(TASK_ROOT / "expected_output.json")
    artifact_dir = attempt_root / "artifacts"
    summary_path = artifact_dir / expected["required_artifact"]
    video_path = artifact_dir / "synthetic_action.mp4"
    if not summary_path.exists() or summary_path.stat().st_size <= 0:
        raise AssertionError(f"missing nonempty summary artifact: {summary_path}")
    if not video_path.exists() or video_path.stat().st_size <= 1024:
        raise AssertionError(f"missing nonempty synthetic video artifact: {video_path}")
    payload = load_json(summary_path)
    if payload.get("task_id") != TASK_ID:
        raise AssertionError(f"wrong task_id: {payload.get('task_id')}")
    if payload.get("repo_commit") != EXPECTED_COMMIT:
        raise AssertionError(f"wrong repo_commit: {payload.get('repo_commit')}")
    if payload.get("success_level") != expected["expected_success_level"]:
        raise AssertionError(f"wrong success_level: {payload.get('success_level')}")
    checks = payload.get("checks")
    if not isinstance(checks, dict) or not all(checks.values()):
        raise AssertionError(f"semantic checks failed: {checks}")
    output = payload.get("output", {})
    if output.get("top5_indices") != expected["semantic_thresholds"]["expected_top5_indices"]:
        raise AssertionError(f"top5 indices changed: {output.get('top5_indices')}")
    if abs(float(output.get("score_sum", 0.0)) - 1.0) > expected["semantic_thresholds"]["max_score_sum_error"]:
        raise AssertionError("prediction scores are not normalized")
    if payload.get("pipeline", {}).get("input_tensor_shape") != expected["semantic_thresholds"]["expected_input_tensor_shape"]:
        raise AssertionError("pipeline input tensor shape mismatch")
    fallback = payload.get("fallback", {})
    native = fallback.get("native_mmcv", {})
    if native.get("native_mmcv_ops_available") is not False or native.get("expected_blocker") is not True:
        raise AssertionError(f"native mmcv boundary not recorded: {native}")
    if sha256_file(summary_path) != expected["gold_observed"]["artifact_sha256"]:
        raise AssertionError("summary artifact checksum mismatch")
    if sha256_file(video_path) != expected["gold_observed"]["video_sha256"]:
        raise AssertionError("synthetic video checksum mismatch")
    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": payload["success_level"],
        "mode": "check_only",
        "artifact": str(summary_path.resolve()),
        "observed": {
            "repo_commit": payload["repo_commit"],
            "pred_label": output.get("pred_label"),
            "top5_indices": output.get("top5_indices"),
            "score_sum": output.get("score_sum"),
            "native_mmcv": native,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the PaperEnvBench MMACTION2 recognition minimal task output.")
    parser.add_argument("attempt_root", nargs="?", default=".")
    parser.add_argument("--repo-dir", default=os.environ.get("PAPERENVBENCH_REPO_DIR", "repo"))
    parser.add_argument("--output-dir", default=os.environ.get("PAPERENVBENCH_OUTPUT_DIR", "artifacts"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        attempt_root = pathlib.Path(args.attempt_root).resolve()
        repo_dir = pathlib.Path(args.repo_dir)
        if not repo_dir.is_absolute():
            repo_dir = attempt_root / repo_dir
        output_dir = pathlib.Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = attempt_root / output_dir
        if args.check_only or not (repo_dir / "mmaction" / "__init__.py").exists():
            result = check_existing(attempt_root)
        else:
            result = {
                "task_id": TASK_ID,
                "status": "pass",
                "success_level": run_gold(repo_dir.resolve(), output_dir.resolve())["success_level"],
                "mode": "generate",
                "artifact": str((output_dir / "expected_artifact.json").resolve()),
            }
    except Exception as exc:
        failure = {"task_id": TASK_ID, "status": "fail", "error": str(exc)}
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
