#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections.abc
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any


TASK_ID = "timesformer_video_transformer_minimal"
REPO_URL = "https://github.com/facebookresearch/TimeSformer"
EXPECTED_COMMIT = "a5ef29a7b7264baff199a30b3306ac27de901133"
EXPECTED_SHAPE = [1, 7]
EXPECTED_PARAM_COUNT = 159239
EXPECTED_LOGITS_SUM = -0.22940867
EXPECTED_LOGITS_MEAN = -0.03277267
FLOAT_TOLERANCE = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: Any) -> str:
    import numpy as np

    array = tensor.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


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


def install_compat_shims() -> list[str]:
    import torch
    import torch.nn.modules.linear as linear_mod

    applied: list[str] = []
    if "torch._six" not in sys.modules:
        shim = types.ModuleType("torch._six")
        shim.container_abcs = collections.abc
        sys.modules["torch._six"] = shim
        applied.append("torch._six.container_abcs")
    if not hasattr(linear_mod, "_LinearWithBias"):
        linear_mod._LinearWithBias = torch.nn.Linear
        applied.append("torch.nn.modules.linear._LinearWithBias")
    return applied


def run_inference(repo_dir: Path, output_dir: Path) -> dict[str, Any]:
    import numpy as np
    import torch

    repo_dir = repo_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (repo_dir / "timesformer" / "models" / "vit.py").exists():
        raise FileNotFoundError(f"missing TimeSformer vit.py under repo dir: {repo_dir}")

    shims = install_compat_shims()
    sys.path.insert(0, str(repo_dir))
    from timesformer.models.vit import Attention, Block, VisionTransformer

    torch.set_num_threads(2)
    torch.manual_seed(20260516)
    model = VisionTransformer(
        img_size=32,
        patch_size=16,
        num_classes=7,
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        qkv_bias=True,
        num_frames=4,
        attention_type="divided_space_time",
        drop_path_rate=0.0,
    )
    model.eval()

    clip = torch.linspace(-1, 1, steps=1 * 3 * 4 * 32 * 32, dtype=torch.float32).reshape(1, 3, 4, 32, 32)
    with torch.no_grad():
        features = model.forward_features(clip)
        logits = model(clip)

    repo_commit = git_commit(repo_dir)
    logits_list = [round(float(value), 8) for value in logits.detach().cpu().flatten().tolist()]
    parameter_count = int(sum(param.numel() for param in model.parameters()))
    first_block = model.blocks[0]
    temporal_attention_class = f"{first_block.temporal_attn.__class__.__module__}.{first_block.temporal_attn.__class__.__name__}"
    spatial_attention_class = f"{first_block.attn.__class__.__module__}.{first_block.attn.__class__.__name__}"
    logits_sum = round(float(logits.sum().item()), 8)
    logits_mean = round(float(logits.mean().item()), 8)

    checks = {
        "repo_commit_matches": repo_commit == EXPECTED_COMMIT,
        "repository_model_loaded": model.__class__.__module__ == "timesformer.models.vit",
        "uses_pinned_attention_class": Attention.__module__ == "timesformer.models.vit",
        "uses_pinned_block_class": Block.__module__ == "timesformer.models.vit",
        "divided_space_time_attention": model.attention_type == "divided_space_time"
        and first_block.attention_type == "divided_space_time"
        and hasattr(first_block, "temporal_attn")
        and hasattr(first_block, "temporal_fc"),
        "synthetic_clip_shape_matches": list(clip.shape) == [1, 3, 4, 32, 32],
        "logits_shape_matches": list(logits.shape) == EXPECTED_SHAPE,
        "features_shape_matches": list(features.shape) == [1, 64],
        "logits_finite": bool(torch.isfinite(logits).all().item()),
        "features_finite": bool(torch.isfinite(features).all().item()),
        "parameter_count_matches": parameter_count == EXPECTED_PARAM_COUNT,
        "logits_sum_within_tolerance": abs(logits_sum - EXPECTED_LOGITS_SUM) <= FLOAT_TOLERANCE,
        "logits_mean_within_tolerance": abs(logits_mean - EXPECTED_LOGITS_MEAN) <= FLOAT_TOLERANCE,
        "compatibility_shims_documented": set(shims) <= {
            "torch._six.container_abcs",
            "torch.nn.modules.linear._LinearWithBias",
        },
    }

    summary = {
        "task_id": TASK_ID,
        "success_level": "L4_fallback" if all(checks.values()) else "below_L4",
        "repo_url": REPO_URL,
        "repo_commit": repo_commit,
        "python": sys.version.split()[0],
        "package_versions": {
            "torch": torch.__version__,
            "einops": metadata.version("einops"),
            "fvcore": metadata.version("fvcore"),
            "numpy": np.__version__,
            "pyyaml": metadata.version("pyyaml"),
            "yacs": metadata.version("yacs"),
            "simplejson": metadata.version("simplejson"),
        },
        "model": {
            "entrypoint": "timesformer.models.vit.VisionTransformer",
            "class": f"{model.__class__.__module__}.{model.__class__.__name__}",
            "attention_type": model.attention_type,
            "img_size": 32,
            "patch_size": 16,
            "num_frames": 4,
            "num_classes": 7,
            "embed_dim": 64,
            "depth": 2,
            "num_heads": 4,
            "parameter_count": parameter_count,
            "training": bool(model.training),
        },
        "attention_path": {
            "block_class": f"{first_block.__class__.__module__}.{first_block.__class__.__name__}",
            "temporal_attention_class": temporal_attention_class,
            "spatial_attention_class": spatial_attention_class,
            "temporal_fc_present": hasattr(first_block, "temporal_fc"),
        },
        "input_clip": {
            "kind": "deterministic_synthetic_clip",
            "shape": list(clip.shape),
            "dtype": str(clip.dtype),
            "min": round(float(clip.min().item()), 8),
            "max": round(float(clip.max().item()), 8),
            "mean": round(float(clip.mean().item()), 8),
            "sha256": sha256_tensor(clip),
        },
        "raw_output": {
            "features_shape": list(features.shape),
            "features_sha256": sha256_tensor(features),
            "logits_shape": list(logits.shape),
            "logits_sha256": sha256_tensor(logits),
            "logits": logits_list,
            "logits_sum": logits_sum,
            "logits_mean": logits_mean,
            "top_class": int(torch.argmax(logits, dim=1).item()),
        },
        "compatibility": {
            "level": "L4_fallback",
            "reason": "The upstream README targets Python 3.7 era dependencies; Python 3.12 with torch 2.8.0 needs private API shims for torch._six and _LinearWithBias.",
            "applied_shims": shims,
            "source_files_exercised": [
                "timesformer/models/vit.py",
                "timesformer/models/vit_utils.py",
                "timesformer/models/build.py",
            ],
        },
        "artifact_files": {
            "summary": "expected_artifact.json",
        },
        "checks": checks,
    }

    artifact_path = output_dir / "expected_artifact.json"
    artifact_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["artifact_digest_without_self_reference"] = sha256_file(artifact_path)
    artifact_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def check_artifact(output_dir: Path) -> dict[str, Any]:
    artifact_path = output_dir / "expected_artifact.json"
    payload = load_json(artifact_path)
    checks = payload.get("checks", {})
    if not isinstance(checks, dict):
        raise AssertionError("artifact checks must be an object")
    required = [
        "repo_commit_matches",
        "repository_model_loaded",
        "uses_pinned_attention_class",
        "uses_pinned_block_class",
        "divided_space_time_attention",
        "synthetic_clip_shape_matches",
        "logits_shape_matches",
        "features_shape_matches",
        "logits_finite",
        "features_finite",
        "parameter_count_matches",
        "logits_sum_within_tolerance",
        "logits_mean_within_tolerance",
        "compatibility_shims_documented",
    ]
    missing = [name for name in required if name not in checks]
    failed = [name for name in required if checks.get(name) is not True]
    if missing or failed:
        raise AssertionError(f"artifact checks failed; missing={missing}, failed={failed}")
    if payload.get("repo_commit") != EXPECTED_COMMIT:
        raise AssertionError(f"repo commit mismatch in artifact: {payload.get('repo_commit')}")
    if payload.get("success_level") != "L4_fallback":
        raise AssertionError(f"unexpected success level: {payload.get('success_level')}")
    if payload.get("raw_output", {}).get("logits_shape") != EXPECTED_SHAPE:
        raise AssertionError("logits shape mismatch in artifact")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, default=Path("repo"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.check_only:
        summary = check_artifact(args.output_dir)
    else:
        summary = run_inference(args.repo_dir, args.output_dir)
        if summary["success_level"] != "L4_fallback":
            raise AssertionError(f"semantic verification failed: {summary['checks']}")

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"{TASK_ID}: {summary['success_level']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
