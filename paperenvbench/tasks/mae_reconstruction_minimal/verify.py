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

_PEB_TASK_ID = "mae_reconstruction_minimal"
_PEB_EXPECTED_ARTIFACT_SHA256 = "9bccdbeba1c4eff075469d1e5e6cb86d676a303db41328e538e73d5358f7f2d6"
_PEB_REQUIRED_SIDE_ARTIFACTS = {'expected_artifact.png': 1000, 'synthetic_input.png': 100, 'verification_result.json': 100}


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
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


TASK_ID = "mae_reconstruction_minimal"
COMMIT = "efb2a8062c206524e35e47d04501ed4f544c0ae8"
IMG_SIZE = 32
PATCH_SIZE = 16
MASK_RATIO = 0.5
SEED = 123


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


def install_compatibility_shims() -> list[str]:
    import numpy as np
    import timm.models.vision_transformer as vision_transformer

    shims: list[str] = []
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
        shims.append("numpy.float_alias_for_mae_pos_embed")

    original_block = vision_transformer.Block
    if "qk_scale" not in inspect.signature(original_block.__init__).parameters:
        class CompatBlock(original_block):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, qk_scale: object = None, **kwargs: Any) -> None:
                if qk_scale is not None:
                    raise ValueError("MAE compatibility shim only supports qk_scale=None")
                super().__init__(*args, **kwargs)

        vision_transformer.Block = CompatBlock
        shims.append("timm_block_qk_scale_none_adapter")

    return shims


def tensor_to_image(tensor: Any) -> Any:
    from PIL import Image
    import torch

    image = tensor.detach().cpu().clamp(0, 1)
    if image.ndim == 4:
        image = image[0]
    image = image.permute(1, 2, 0)
    array = (image * 255).round().to(torch.uint8).numpy()
    return Image.fromarray(array, mode="RGB")


def save_grid(path: Path, panels: list[tuple[str, Any]]) -> None:
    from PIL import Image, ImageDraw

    images = [tensor_to_image(tensor).resize((128, 128), resample=Image.Resampling.NEAREST) for _, tensor in panels]
    label_h = 18
    grid = Image.new("RGB", (128 * len(images), 128 + label_h), "white")
    draw = ImageDraw.Draw(grid)
    for idx, ((label, _), image) in enumerate(zip(panels, images)):
        x = idx * 128
        grid.paste(image, (x, label_h))
        draw.text((x + 4, 3), label, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def run_mae(repo_dir: Path, artifact_dir: Path) -> dict[str, Any]:
    import numpy as np
    import timm
    import torch

    shims = install_compatibility_shims()
    import models_mae

    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    axis = torch.linspace(0.0, 1.0, steps=IMG_SIZE)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    checker = ((torch.arange(IMG_SIZE).view(-1, 1) + torch.arange(IMG_SIZE).view(1, -1)) % 2).float()
    image = torch.stack([xx, yy, 0.25 + 0.5 * checker], dim=0).unsqueeze(0)

    model = models_mae.mae_vit_base_patch16_dec512d8b(img_size=IMG_SIZE, norm_pix_loss=True)
    model.eval()
    with torch.no_grad():
        latent, mask, ids_restore = model.forward_encoder(image, MASK_RATIO)
        pred = model.forward_decoder(latent, ids_restore)
        loss = model.forward_loss(image, pred, mask)
        reconstruction = model.unpatchify(pred).clamp(0, 1)

    patch_mask = mask.reshape(1, IMG_SIZE // PATCH_SIZE, IMG_SIZE // PATCH_SIZE)
    patch_mask = patch_mask.repeat_interleave(PATCH_SIZE, dim=1).repeat_interleave(PATCH_SIZE, dim=2)
    patch_mask = patch_mask.unsqueeze(1).repeat(1, 3, 1, 1)
    masked_input = image * (1 - patch_mask)
    blended = image * (1 - patch_mask) + reconstruction * patch_mask

    png_path = artifact_dir / "expected_artifact.png"
    summary_path = artifact_dir / "expected_artifact.json"
    result_path = artifact_dir / "verification_result.json"
    input_path = artifact_dir / "synthetic_input.png"
    tensor_to_image(image).save(input_path)
    save_grid(
        png_path,
        [
            ("input", image),
            ("masked", masked_input),
            ("pred", reconstruction),
            ("blend", blended),
        ],
    )

    num_patches = int(mask.numel())
    masked_patches = int(mask.sum().item())
    checks = {
        "repo_commit_matches": repo_commit(repo_dir) == COMMIT,
        "loss_is_finite": bool(torch.isfinite(loss).item()),
        "latent_shape_matches": list(latent.shape) == [1, 3, 768],
        "prediction_shape_matches": list(pred.shape) == [1, 4, 768],
        "mask_ratio_matches": masked_patches == 2 and num_patches == 4,
        "reconstruction_shape_matches": list(reconstruction.shape) == [1, 3, IMG_SIZE, IMG_SIZE],
        "artifact_png_nonempty": png_path.exists() and png_path.stat().st_size > 1000,
    }
    if not all(checks.values()):
        raise AssertionError({"checks": checks})

    payload = {
        "task_id": TASK_ID,
        "success": True,
        "success_level": "L4_cpu_forward_reconstruction",
        "repo": {
            "url": "https://github.com/facebookresearch/mae",
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
            "name": "mae_vit_base_patch16_dec512d8b",
            "img_size": IMG_SIZE,
            "patch_size": PATCH_SIZE,
            "mask_ratio": MASK_RATIO,
            "norm_pix_loss": True,
            "seed": SEED,
        },
        "checks": checks,
        "observed": {
            "loss": round(float(loss.item()), 8),
            "latent_shape": list(latent.shape),
            "latent_mean": round(float(latent.mean().item()), 8),
            "latent_std": round(float(latent.std().item()), 8),
            "prediction_shape": list(pred.shape),
            "prediction_mean": round(float(pred.mean().item()), 8),
            "prediction_std": round(float(pred.std().item()), 8),
            "mask": mask.detach().cpu().int().tolist(),
            "masked_patches": masked_patches,
            "num_patches": num_patches,
            "reconstruction_mean": round(float(reconstruction.mean().item()), 8),
            "reconstruction_std": round(float(reconstruction.std().item()), 8),
            "blended_mean": round(float(blended.mean().item()), 8),
        },
        "compatibility_shims": shims,
        "artifacts": {
            "summary": str(summary_path),
            "image_grid": str(png_path),
            "synthetic_input": str(input_path),
            "image_grid_sha256": sha256(png_path),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    run_root = Path(os.environ.get("PAPERENVBENCH_RUN_ROOT", os.getcwd())).resolve()
    repo_dir = Path(os.environ.get("MAE_REPO", run_root / "mae")).resolve()
    artifact_dir = Path(os.environ.get("PAPERENVBENCH_ARTIFACT_DIR", run_root / "artifacts")).resolve()

    if not (repo_dir / "models_mae.py").exists():
        raise FileNotFoundError(f"MAE repo not found or incomplete: {repo_dir}")
    if repo_commit(repo_dir) != COMMIT:
        raise RuntimeError(f"Unexpected MAE commit: {repo_commit(repo_dir)}; expected {COMMIT}")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    prepare_import_path(repo_dir, run_root)
    payload = run_mae(repo_dir, artifact_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
