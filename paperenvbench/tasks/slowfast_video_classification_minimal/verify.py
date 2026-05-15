#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import pathlib
import subprocess
import sys
import types
from typing import Any


TASK_ID = "slowfast_video_classification_minimal"
EXPECTED_COMMIT = "287ec0076846560f44a9327e931a5a2360240533"
EXPECTED_SUCCESS_LEVEL = "L4_fallback"
EXPECTED_SHA256 = {
    "expected_artifact.json": "ad1bae52354024b44d3082b83ac8a8bc4118dc51176a306162aecb0512d1de91",
    "expected_clip_preview.ppm": "2471bfd2b17b4f6b34bdc720558dabebb8c5840f5218c44824321ce63cccc6c8",
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def git_commit(repo_dir: pathlib.Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def install_runtime_shims() -> dict[str, Any]:
    import torch.nn as nn
    import pytorchvideo.layers.distributed as pvd

    shimmed: dict[str, Any] = {
        "pytorchvideo_distributed_symbols": [],
        "detectron2_roi_align_stub": True,
    }
    fallbacks = {
        "cat_all_gather": lambda tensors: tensors,
        "get_local_process_group": lambda: None,
        "get_local_rank": lambda: 0,
        "get_local_size": lambda: 1,
        "get_world_size": lambda: 1,
        "init_distributed_training": lambda num_gpus, shard_id: None,
    }
    for name, fn in fallbacks.items():
        if not hasattr(pvd, name):
            setattr(pvd, name, fn)
            shimmed["pytorchvideo_distributed_symbols"].append(name)

    detectron2 = types.ModuleType("detectron2")
    layers = types.ModuleType("detectron2.layers")

    class ROIAlign(nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()

        def forward(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("ROIAlign shim is unused by the classification verifier")

    layers.ROIAlign = ROIAlign
    detectron2.layers = layers
    sys.modules.setdefault("detectron2", detectron2)
    sys.modules.setdefault("detectron2.layers", layers)

    loss_mod = types.ModuleType("pytorchvideo.losses.soft_target_cross_entropy")

    class SoftTargetCrossEntropyLoss(nn.CrossEntropyLoss):
        pass

    loss_mod.SoftTargetCrossEntropyLoss = SoftTargetCrossEntropyLoss
    sys.modules.setdefault("pytorchvideo.losses", types.ModuleType("pytorchvideo.losses"))
    sys.modules.setdefault("pytorchvideo.losses.soft_target_cross_entropy", loss_mod)
    shimmed["pytorchvideo_loss_stub"] = True
    return shimmed


def write_preview(path: pathlib.Path, clip: Any) -> dict[str, Any]:
    frame = clip[0].detach().cpu()
    frame = frame.permute(1, 2, 0).contiguous()
    frame = (frame * 255.0).clamp(0, 255).to(dtype=__import__("torch").uint8)
    height, width, channels = frame.shape
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + frame.numpy().tobytes())
    return {
        "format": "ppm_p6",
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def generate(repo_dir: pathlib.Path, artifact_dir: pathlib.Path) -> dict[str, Any]:
    import torch

    artifact_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = repo_dir.resolve()
    sys.path.insert(0, str(repo_dir))
    shimmed = install_runtime_shims()

    from slowfast.config.defaults import get_cfg
    from slowfast.datasets import transform
    from slowfast.models import build_model

    commit = git_commit(repo_dir)
    cfg = get_cfg()
    config_path = repo_dir / "configs" / "Kinetics" / "SLOWFAST_8x8_R50.yaml"
    cfg.merge_from_file(str(config_path))
    cfg.defrost()
    cfg.NUM_GPUS = 0
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    cfg.MODEL.NUM_CLASSES = 5
    cfg.MODEL.DROPOUT_RATE = 0.0
    cfg.RESNET.DEPTH = 18
    cfg.RESNET.WIDTH_PER_GROUP = 8
    cfg.RESNET.NUM_GROUPS = 1
    cfg.RESNET.NUM_BLOCK_TEMP_KERNEL = [[2, 2], [2, 2], [2, 2], [2, 2]]
    cfg.SLOWFAST.ALPHA = 4
    cfg.SLOWFAST.BETA_INV = 8
    cfg.DATA.NUM_FRAMES = 8
    cfg.DATA.TRAIN_CROP_SIZE = 32
    cfg.DATA.TEST_CROP_SIZE = 32
    cfg.DATA.INPUT_CHANNEL_NUM = [3, 3]
    cfg.freeze()

    torch.manual_seed(20260516)
    clip = torch.linspace(0, 1, steps=8 * 3 * 40 * 40, dtype=torch.float32).reshape(8, 3, 40, 40)
    cropped, boxes = transform.uniform_crop(clip, 32, 1)
    fast = cropped.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    slow_indices = torch.linspace(0, fast.shape[2] - 1, fast.shape[2] // cfg.SLOWFAST.ALPHA).long()
    slow = torch.index_select(fast, 2, slow_indices)

    model = build_model(cfg)
    model.eval()
    with torch.no_grad():
        logits = model([slow, fast])
        probs = torch.softmax(logits, dim=1)

    logits_list = [round(float(value), 10) for value in logits[0].tolist()]
    probs_list = [round(float(value), 10) for value in probs[0].tolist()]
    preview = write_preview(artifact_dir / "expected_clip_preview.ppm", cropped)
    payload = {
        "task_id": TASK_ID,
        "success_level": EXPECTED_SUCCESS_LEVEL,
        "repo": "facebookresearch/SlowFast",
        "repo_commit": commit,
        "paper_model": "SlowFast video classification",
        "config": "configs/Kinetics/SLOWFAST_8x8_R50.yaml",
        "official_repo_code_paths": [
            "slowfast/config/defaults.py",
            "slowfast/datasets/transform.py",
            "slowfast/models/build.py",
            "slowfast/models/video_model_builder.py",
            "slowfast/models/stem_helper.py",
            "slowfast/models/resnet_helper.py",
            "slowfast/models/head_helper.py",
        ],
        "fallback_reason": {
            "native_import_requires_missing_pytorchvideo_symbols": True,
            "classification_path_has_unused_detectron2_roi_align_import": True,
            "full_checkpoint_inference_not_claimed": True,
            "runtime_shims": shimmed,
        },
        "input": {
            "synthetic_clip_shape_tchw": list(clip.shape),
            "cropped_clip_shape_tchw": list(cropped.shape),
            "slow_pathway_shape": list(slow.shape),
            "fast_pathway_shape": list(fast.shape),
            "slow_indices": [int(value) for value in slow_indices.tolist()],
            "uniform_crop_boxes_is_none": boxes is None,
        },
        "output": {
            "logits_shape": list(logits.shape),
            "logits": logits_list,
            "probabilities": probs_list,
            "top_class": int(probs.argmax(dim=1).item()),
            "top_probability": round(float(probs.max().item()), 10),
            "probability_sum": round(float(probs.sum().item()), 10),
            "logits_sha256": sha256_json(logits_list),
            "preview": preview,
        },
        "package_versions": {
            "python": sys.version.split()[0],
            "torch": __import__("torch").__version__,
            "torchvision": metadata.version("torchvision"),
            "pytorchvideo": metadata.version("pytorchvideo"),
            "fvcore": metadata.version("fvcore"),
            "yacs": metadata.version("yacs"),
            "numpy": metadata.version("numpy"),
        },
        "checks": {
            "repo_commit_matches": commit == EXPECTED_COMMIT,
            "official_config_loaded": cfg.MODEL.MODEL_NAME == "SlowFast" and cfg.MODEL.ARCH == "slowfast",
            "official_uniform_crop_executed": list(cropped.shape) == [8, 3, 32, 32],
            "slowfast_two_pathway_tensor_built": list(slow.shape) == [1, 3, 2, 32, 32]
            and list(fast.shape) == [1, 3, 8, 32, 32],
            "slowfast_model_forward_executed": list(logits.shape) == [1, 5],
            "probabilities_finite_and_normalized": all(math.isfinite(v) for v in probs_list)
            and abs(sum(probs_list) - 1.0) < 1e-6,
            "preview_artifact_written": preview["size_bytes"] > 1024,
        },
    }
    payload["checks"]["all_checks_passed"] = all(payload["checks"].values())
    if not payload["checks"]["all_checks_passed"]:
        raise AssertionError(payload["checks"])
    out_path = artifact_dir / "expected_artifact.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def check_existing(artifact_dir: pathlib.Path) -> dict[str, Any]:
    summary_path = artifact_dir / "expected_artifact.json"
    preview_path = artifact_dir / "expected_clip_preview.ppm"
    for path in (summary_path, preview_path):
        if not path.exists() or path.stat().st_size <= 0:
            raise AssertionError(f"missing nonempty artifact: {path}")

    observed = {
        "expected_artifact.json": sha256_file(summary_path),
        "expected_clip_preview.ppm": sha256_file(preview_path),
    }
    if observed != EXPECTED_SHA256:
        raise AssertionError({"checksum_mismatch": observed})

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("task_id") != TASK_ID:
        raise AssertionError(f"wrong task_id: {payload.get('task_id')}")
    if payload.get("repo_commit") != EXPECTED_COMMIT:
        raise AssertionError(f"wrong repo_commit: {payload.get('repo_commit')}")
    if payload.get("success_level") != EXPECTED_SUCCESS_LEVEL:
        raise AssertionError(f"wrong success_level: {payload.get('success_level')}")
    checks = payload.get("checks", {})
    required = {
        "repo_commit_matches",
        "official_config_loaded",
        "official_uniform_crop_executed",
        "slowfast_two_pathway_tensor_built",
        "slowfast_model_forward_executed",
        "probabilities_finite_and_normalized",
        "preview_artifact_written",
        "all_checks_passed",
    }
    if set(checks) != required or not all(checks.values()):
        raise AssertionError({"checks": checks})
    output = payload.get("output", {})
    if output.get("logits_shape") != [1, 5]:
        raise AssertionError({"logits_shape": output.get("logits_shape")})
    if output.get("top_class") != 4:
        raise AssertionError({"top_class": output.get("top_class")})
    if abs(float(output.get("probability_sum", 0.0)) - 1.0) > 1e-6:
        raise AssertionError({"probability_sum": output.get("probability_sum")})
    preview = output.get("preview", {})
    if preview.get("sha256") != observed["expected_clip_preview.ppm"]:
        raise AssertionError({"preview_sha256": preview.get("sha256"), "observed": observed})
    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": EXPECTED_SUCCESS_LEVEL,
        "artifact_dir": str(artifact_dir.resolve()),
        "observed": observed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default=str(pathlib.Path(__file__).resolve().parent / "artifacts"))
    parser.add_argument("--repo-dir", default=str(pathlib.Path(__file__).resolve().parent / "repo"))
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    try:
        if args.generate:
            result = generate(pathlib.Path(args.repo_dir), pathlib.Path(args.artifact_dir))
            result = {
                "task_id": TASK_ID,
                "status": "generated",
                "success_level": result["success_level"],
                "artifact": str(pathlib.Path(args.artifact_dir) / "expected_artifact.json"),
            }
        else:
            result = check_existing(pathlib.Path(args.artifact_dir))
    except Exception as exc:
        print(json.dumps({"task_id": TASK_ID, "status": "fail", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
