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

_PEB_TASK_ID = "moco_feature_minimal"
_PEB_EXPECTED_ARTIFACT_SHA256 = "4c4f377c98e861d490582717af979805070659e28c7f14f816a0ddcda699bef8"
_PEB_REQUIRED_SIDE_ARTIFACTS = {'verification_result.json': 100}


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


import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


TASK_ID = "moco_feature_minimal"
EXPECTED_COMMIT = "7397dfe146c7ca6bbb58e9c382498069178ba764"
FEATURE_DIM = 128
IMAGE_SIZE = 224
SEED = 20260515


def git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return ""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def to_builtin(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    return value


def make_synthetic_image(torch_module: Any) -> Any:
    coords = torch_module.linspace(-1.0, 1.0, IMAGE_SIZE, dtype=torch_module.float32)
    yy, xx = torch_module.meshgrid(coords, coords, indexing="ij")
    image = torch_module.stack(
        [
            (xx + 1.0) * 0.5,
            (yy + 1.0) * 0.5,
            (torch_module.sin(xx * 3.141592653589793) * torch_module.cos(yy * 3.141592653589793) + 1.0) * 0.5,
        ],
        dim=0,
    ).unsqueeze(0)
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default=os.environ.get("PAPERENVBENCH_REPO_DIR", "repo"))
    parser.add_argument("--output-dir", default=os.environ.get("PAPERENVBENCH_OUTPUT_DIR", "artifacts"))
    parser.add_argument("--artifact-name", default="expected_artifact.json")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / args.artifact_name
    result_path = output_dir / "verification_result.json"

    if not repo_dir.exists():
        raise FileNotFoundError(f"MoCo repo not found: {repo_dir}")

    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    import torch
    import torchvision
    from torch.nn import functional as F

    builder = importlib.import_module("moco.builder")

    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    model = builder.MoCo(
        torchvision.models.resnet50,
        dim=FEATURE_DIM,
        K=16,
        m=0.999,
        T=0.2,
        mlp=True,
    )
    model.eval()

    image = make_synthetic_image(torch)
    with torch.no_grad():
        raw_feature = model.encoder_q(image)
        normalized_feature = F.normalize(raw_feature, dim=1)

    raw_cpu = raw_feature.detach().cpu().contiguous()
    norm_cpu = normalized_feature.detach().cpu().contiguous()
    repo_commit = git_commit(repo_dir)
    raw_norm = float(raw_cpu.norm(dim=1).item())
    normalized_norm = float(norm_cpu.norm(dim=1).item())
    feature_values = norm_cpu.reshape(-1)
    feature_sha256 = sha256_bytes(feature_values.numpy().astype("<f4", copy=False).tobytes())

    checks = {
        "repo_commit_matches": repo_commit == EXPECTED_COMMIT,
        "moco_builder_imports": hasattr(builder, "MoCo"),
        "encoder_q_runs_on_cpu": str(raw_cpu.device) == "cpu",
        "input_shape_is_1x3x224x224": list(image.shape) == [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        "feature_shape_is_1x128": list(raw_cpu.shape) == [1, FEATURE_DIM],
        "feature_values_are_finite": bool(torch.isfinite(raw_cpu).all().item() and torch.isfinite(norm_cpu).all().item()),
        "normalized_feature_norm_is_one": abs(normalized_norm - 1.0) <= 1e-6,
    }

    payload = {
        "task_id": TASK_ID,
        "success": all(checks.values()),
        "success_level": "L4_cpu_encoder_forward" if all(checks.values()) else "below_L4",
        "repo_commit": repo_commit,
        "python": sys.version.split()[0],
        "package_versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "device": "cpu",
        "model": {
            "class": "moco.builder.MoCo",
            "base_encoder": "torchvision.models.resnet50",
            "feature_dim": FEATURE_DIM,
            "queue_size": 16,
            "mlp": True,
            "checkpoint_loaded": False,
        },
        "input": {
            "kind": "synthetic_rgb_tensor",
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "min": round(float(image.min().item()), 8),
            "max": round(float(image.max().item()), 8),
            "mean": round(float(image.mean().item()), 8),
            "sha256": sha256_bytes(image.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()),
        },
        "feature": {
            "raw_shape": list(raw_cpu.shape),
            "raw_norm": round(raw_norm, 8),
            "normalized_shape": list(norm_cpu.shape),
            "normalized_norm": round(normalized_norm, 8),
            "normalized_mean": round(float(norm_cpu.mean().item()), 8),
            "normalized_std": round(float(norm_cpu.std(unbiased=False).item()), 8),
            "normalized_min": round(float(norm_cpu.min().item()), 8),
            "normalized_max": round(float(norm_cpu.max().item()), 8),
            "normalized_sha256": feature_sha256,
            "normalized_first_8": [round(float(x), 8) for x in feature_values[:8].tolist()],
        },
        "checks": checks,
    }
    payload = to_builtin(payload)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    artifact_path.write_text(text)
    result_path.write_text(text)
    print(text, end="")

    if not all(checks.values()):
        failed = ", ".join(k for k, ok in checks.items() if not ok)
        raise SystemExit(f"semantic checks failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
