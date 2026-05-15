#!/usr/bin/env python3
from __future__ import annotations


# PaperEnvBench artifact-only validation path. This exits before runtime imports
# when --check-only is requested, so the standalone benchmark repo can verify
# gold task packages without vendoring full upstream checkouts or weights.
import argparse as _peb_argparse
import hashlib as _peb_hashlib
import json as _peb_json
import pathlib as _peb_pathlib
import sys as _peb_sys

_PEB_TASK_ID = "videomae_classification_minimal"
_PEB_EXPECTED_ARTIFACT_SHA256 = "c96cf8182532a1b7d766a204a92de522b0916ba16b2596415bb511137db1cefc"
_PEB_REQUIRED_SIDE_ARTIFACTS = {'synthetic_video_strip.png': 200, 'verification_result.json': 100}


def _peb_sha256(path: _peb_pathlib.Path) -> str:
    digest = _peb_hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peb_check_only() -> None:
    if "--check-only" not in _peb_sys.argv:
        return
    parser = _peb_argparse.ArgumentParser(description=f"Check packaged gold artifact for {_PEB_TASK_ID}.")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--artifact-dir", "--output-dir", dest="artifact_dir", default="artifacts")
    parser.add_argument("--artifact-name", default="expected_artifact.json")
    parser.add_argument("--repo-dir", default=None)
    args, _unknown = parser.parse_known_args()
    task_root = _peb_pathlib.Path(__file__).resolve().parent
    artifact_dir = _peb_pathlib.Path(args.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = task_root / artifact_dir
    artifact_path = artifact_dir / args.artifact_name
    if not artifact_path.exists() and args.artifact_name != "expected_artifact.json":
        artifact_path = artifact_dir / "expected_artifact.json"
    payload = _peb_json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact_sha256 = _peb_sha256(artifact_path)
    payload_checks = payload.get("checks", {})
    payload_checks_true = all(bool(value) for value in payload_checks.values()) if isinstance(payload_checks, dict) else True
    side_checks = {}
    for name, min_size in _PEB_REQUIRED_SIDE_ARTIFACTS.items():
        path = artifact_dir / name
        side_checks[name] = path.exists() and path.stat().st_size >= int(min_size)
    checks = {
        "task_id_matches": payload.get("task_id") == _PEB_TASK_ID,
        "artifact_sha256_matches": artifact_sha256 == _PEB_EXPECTED_ARTIFACT_SHA256,
        "payload_checks_true": payload_checks_true,
        "payload_success_not_false": payload.get("success", True) is not False,
        "side_artifacts_present": all(side_checks.values()),
    }
    ok = all(checks.values())
    result = {
        "task_id": _PEB_TASK_ID,
        "status": "pass" if ok else "fail",
        "mode": "check_only",
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "success_level": payload.get("success_level") or payload.get("expected_success_level"),
        "checks": checks,
        "side_artifacts": side_checks,
    }
    print(_peb_json.dumps(result, indent=2, sort_keys=True) if args.json else result["status"])
    if not ok:
        raise SystemExit(1)
    raise SystemExit(0)


_peb_check_only()


import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


TASK_ID = "videomae_classification_minimal"
COMMIT = "14ef8d856287c94ef1f985fe30f958eb4ec2c55d"
IMG_SIZE = 32
NUM_FRAMES = 4
TUBELET_SIZE = 2
NUM_CLASSES = 5
SEED = 20260516


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_commit(repo_dir: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()


def prepare_import_path(repo_dir: Path, run_root: Path) -> None:
    repo_str = str(repo_dir.resolve())
    run_root_str = str(run_root.resolve())
    cleaned = []
    for item in sys.path:
        if item in ("", run_root_str):
            continue
        cleaned.append(item)
    sys.path[:] = [repo_str] + cleaned


def synthetic_video(torch: Any) -> Any:
    axis = torch.linspace(0.0, 1.0, IMG_SIZE)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    frames = []
    for frame_idx in range(NUM_FRAMES):
        phase = frame_idx / max(1, NUM_FRAMES - 1)
        red = (xx + 0.15 * frame_idx).fmod(1.0)
        green = (yy * (0.7 + 0.1 * frame_idx)).clamp(0, 1)
        blue = torch.full_like(xx, phase)
        frames.append(torch.stack([red, green, blue], dim=0))
    return torch.stack(frames, dim=1).unsqueeze(0).contiguous()


def save_video_strip(path: Path, video: Any) -> None:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    strip = Image.new("RGB", (IMG_SIZE * NUM_FRAMES, IMG_SIZE + 16), "white")
    draw = ImageDraw.Draw(strip)
    for frame_idx in range(NUM_FRAMES):
        array = (video[0, :, frame_idx].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().numpy()
        strip.paste(Image.fromarray(array, mode="RGB"), (frame_idx * IMG_SIZE, 16))
        draw.text((frame_idx * IMG_SIZE + 2, 2), f"t{frame_idx}", fill=(0, 0, 0))
    strip.save(path)


def run_videomae(repo_dir: Path, artifact_dir: Path) -> dict[str, Any]:
    import numpy as np
    import timm
    import torch

    import modeling_finetune

    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    video = synthetic_video(torch)
    model = modeling_finetune.vit_small_patch16_224(
        pretrained=False,
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        all_frames=NUM_FRAMES,
        tubelet_size=TUBELET_SIZE,
        drop_path_rate=0.0,
        fc_drop_rate=0.0,
        use_checkpoint=False,
        use_mean_pooling=True,
        init_scale=1.0,
    )
    model.eval()

    with torch.no_grad():
        features = model.forward_features(video)
        logits = model(video)
        probabilities = torch.softmax(logits, dim=-1)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "expected_artifact.json"
    result_path = artifact_dir / "verification_result.json"
    strip_path = artifact_dir / "synthetic_video_strip.png"
    save_video_strip(strip_path, video)

    checks = {
        "repo_commit_matches": repo_commit(repo_dir) == COMMIT,
        "video_shape_matches": list(video.shape) == [1, 3, NUM_FRAMES, IMG_SIZE, IMG_SIZE],
        "patch_embed_uses_conv3d": model.patch_embed.proj.__class__.__name__ == "Conv3d",
        "tubelet_kernel_matches": list(model.patch_embed.proj.kernel_size) == [TUBELET_SIZE, 16, 16],
        "num_temporal_spatial_tokens_matches": int(model.patch_embed.num_patches) == 8,
        "features_shape_matches": list(features.shape) == [1, 384],
        "logits_shape_matches": list(logits.shape) == [1, NUM_CLASSES],
        "logits_are_finite": bool(torch.isfinite(logits).all().item()),
        "probabilities_sum_to_one": abs(float(probabilities.sum().item()) - 1.0) < 1e-6,
        "artifact_png_nonempty": strip_path.exists() and strip_path.stat().st_size > 200,
    }
    if not all(checks.values()):
        raise AssertionError({"checks": checks})

    payload = {
        "task_id": TASK_ID,
        "success": True,
        "success_level": "L4_fallback_cpu_synthetic_video_forward",
        "repo": {
            "url": "https://github.com/MCG-NJU/VideoMAE",
            "commit": repo_commit(repo_dir),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "timm": timm.__version__,
            "numpy": np.__version__,
            "device": "cpu",
        },
        "model": {
            "name": "vit_small_patch16_224",
            "img_size": IMG_SIZE,
            "num_frames": NUM_FRAMES,
            "tubelet_size": TUBELET_SIZE,
            "num_classes": NUM_CLASSES,
            "seed": SEED,
            "checkpoint_loaded": False,
        },
        "checks": checks,
        "observed": {
            "video_mean": round(float(video.mean().item()), 8),
            "video_std": round(float(video.std().item()), 8),
            "features_mean": round(float(features.mean().item()), 8),
            "features_std": round(float(features.std().item()), 8),
            "logits": [round(float(value), 8) for value in logits[0].tolist()],
            "probabilities": [round(float(value), 8) for value in probabilities[0].tolist()],
            "top_class": int(torch.argmax(probabilities, dim=-1).item()),
        },
        "artifacts": {
            "summary": str(summary_path),
            "synthetic_video_strip": str(strip_path),
            "synthetic_video_strip_sha256": sha256(strip_path),
        },
        "notes": [
            "This is an L4 fallback because upstream finetuned checkpoints are Google Drive assets and not required for the CPU minimal task.",
            "Import-only checks are insufficient; the verifier runs the pinned repo Conv3d tubelet patch embedding, transformer, pooling, and classification head.",
        ],
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    run_root = Path(os.environ.get("PAPERENVBENCH_RUN_ROOT", os.getcwd())).resolve()
    repo_dir = Path(os.environ.get("VIDEOMAE_REPO", run_root / "VideoMAE")).resolve()
    artifact_dir = Path(os.environ.get("PAPERENVBENCH_ARTIFACT_DIR", run_root / "artifacts")).resolve()

    if not (repo_dir / "modeling_finetune.py").exists():
        raise FileNotFoundError(f"VideoMAE repo not found or incomplete: {repo_dir}")
    if repo_commit(repo_dir) != COMMIT:
        raise RuntimeError(f"Unexpected VideoMAE commit: {repo_commit(repo_dir)}; expected {COMMIT}")

    prepare_import_path(repo_dir, run_root)
    payload = run_videomae(repo_dir, artifact_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
