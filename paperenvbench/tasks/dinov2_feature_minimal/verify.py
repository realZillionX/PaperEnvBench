#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


TASK_ID = "dinov2_feature_minimal"
EXPECTED_COMMIT = "7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c"
EXPECTED_CLS_SHAPE = [1, 384]
EXPECTED_PATCH_SHAPE = [1, 256, 384]
IMAGE_SIZE = 224


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return ""


def make_synthetic_image(image_path: Path) -> Any:
    import torch
    from PIL import Image

    ys = torch.linspace(0, 1, IMAGE_SIZE).view(IMAGE_SIZE, 1).expand(IMAGE_SIZE, IMAGE_SIZE)
    xs = torch.linspace(0, 1, IMAGE_SIZE).view(1, IMAGE_SIZE).expand(IMAGE_SIZE, IMAGE_SIZE)
    checker = (
        (torch.arange(IMAGE_SIZE).view(IMAGE_SIZE, 1) // 16 + torch.arange(IMAGE_SIZE).view(1, IMAGE_SIZE) // 16)
        % 2
    ).float()
    tensor = torch.stack([xs, ys, checker], dim=0).unsqueeze(0).contiguous()

    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = (tensor.squeeze(0).permute(1, 2, 0).mul(255).round().byte().numpy())
    Image.fromarray(image_uint8, mode="RGB").save(image_path)
    return tensor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default=os.environ.get("DINOV2_REPO_DIR", "repo"))
    parser.add_argument("--output-dir", default=os.environ.get("PAPERENVBENCH_ARTIFACT_DIR", "artifacts"))
    parser.add_argument("--artifact-name", default="expected_artifact.json")
    args = parser.parse_args()

    os.environ.setdefault("XFORMERS_DISABLED", "1")

    repo_dir = Path(args.repo_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "synthetic_input.png"
    artifact_path = output_dir / args.artifact_name

    if not (repo_dir / ".git").exists():
        raise FileNotFoundError(f"missing DINOv2 git checkout: {repo_dir}")

    import torch
    from dinov2.hub.backbones import dinov2_vits14

    torch.manual_seed(0)
    torch.set_num_threads(1)
    image_tensor = make_synthetic_image(image_path)

    model = dinov2_vits14(pretrained=False, img_size=IMAGE_SIZE)
    model.eval()
    with torch.no_grad():
        features = model.forward_features(image_tensor)

    cls_embedding = features["x_norm_clstoken"]
    patch_embeddings = features["x_norm_patchtokens"]
    cls_norm = float(cls_embedding.norm().item())
    patch_mean_abs = float(patch_embeddings.abs().mean().item())
    cls_finite = bool(torch.isfinite(cls_embedding).all().item())
    patch_finite = bool(torch.isfinite(patch_embeddings).all().item())
    repo_commit = git_commit(repo_dir)

    checks = {
        "repo_commit_matches": repo_commit == EXPECTED_COMMIT,
        "model_entrypoint_resolves": callable(dinov2_vits14),
        "xformers_disabled": os.environ.get("XFORMERS_DISABLED") == "1",
        "cls_embedding_shape_matches": list(cls_embedding.shape) == EXPECTED_CLS_SHAPE,
        "patch_embedding_shape_matches": list(patch_embeddings.shape) == EXPECTED_PATCH_SHAPE,
        "embeddings_are_finite": cls_finite and patch_finite,
        "cls_norm_positive": cls_norm > 0,
        "patch_mean_abs_positive": patch_mean_abs > 0,
    }

    payload = {
        "task_id": TASK_ID,
        "success": all(checks.values()),
        "success_level": "L4_cpu_random_init_fallback" if all(checks.values()) else "below_L4",
        "repo": {
            "url": "https://github.com/facebookresearch/dinov2",
            "commit": repo_commit,
        },
        "python": sys.version.split()[0],
        "package_versions": {
            "dinov2": importlib.metadata.version("dinov2"),
            "torch": torch.__version__,
            "torchvision": importlib.metadata.version("torchvision"),
            "pillow": importlib.metadata.version("pillow"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "environment": {
            "device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "xformers_disabled": os.environ.get("XFORMERS_DISABLED"),
            "pretrained": False,
            "checkpoint_required": False,
            "checkpoint_fallback_reason": "CPU minimal verifier uses random initialization to avoid torch.hub checkpoint download.",
        },
        "input_image": {
            "path": str(image_path),
            "shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
            "sha256": sha256_file(image_path),
        },
        "model": {
            "entrypoint": "dinov2.hub.backbones.dinov2_vits14",
            "variant": "ViT-S/14",
            "img_size": IMAGE_SIZE,
            "patch_size": 14,
            "embed_dim": int(model.embed_dim),
        },
        "features": {
            "cls_shape": list(cls_embedding.shape),
            "patch_shape": list(patch_embeddings.shape),
            "cls_norm": round(cls_norm, 8),
            "patch_mean_abs": round(patch_mean_abs, 8),
            "cls_first8": [round(float(v), 8) for v in cls_embedding.reshape(-1)[:8].tolist()],
            "cls_sha256": sha256_tensor(cls_embedding),
            "patch_sha256": sha256_tensor(patch_embeddings),
            "cls_finite": cls_finite,
            "patch_finite": patch_finite,
        },
        "checks": checks,
    }

    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if not all(checks.values()):
        failed = ", ".join(key for key, ok in checks.items() if not ok)
        raise SystemExit(f"semantic checks failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
