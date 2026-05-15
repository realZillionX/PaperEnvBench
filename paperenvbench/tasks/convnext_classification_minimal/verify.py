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

_PEB_TASK_ID = "convnext_classification_minimal"
_PEB_EXPECTED_ARTIFACT_SHA256 = "8b76df6ff6b05f55e00b0f0a5a77d074dcf8beaf6c7a79d615a402dce63d7420"
_PEB_REQUIRED_SIDE_ARTIFACTS = {'synthetic_convnext_input.ppm': 1000}


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
import importlib.metadata as metadata
import json
import math
import os
import pathlib
import subprocess
import sys
from typing import Any


TASK_ID = "convnext_classification_minimal"
EXPECTED_COMMIT = "048efcea897d999aed302f2639b6270aedf8d4c8"
EXPECTED_SHAPE = [1, 10]
MIN_LOGITS_STD = 1e-8


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_commit(repo_dir: pathlib.Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return ""


def write_ppm(path: pathlib.Path, size: int = 64) -> dict[str, Any]:
    pixels = bytearray()
    for y in range(size):
        for x in range(size):
            r = (x * 4 + y * 2) % 256
            g = (x * 3 + y * 5 + 17) % 256
            b = (x * 7 + y * 11 + 29) % 256
            pixels.extend((r, g, b))
    header = f"P6\n{size} {size}\n255\n".encode("ascii")
    path.write_bytes(header + bytes(pixels))
    return {
        "format": "ppm_p6",
        "height": size,
        "width": size,
        "channels": 3,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_ppm_as_tensor(path: pathlib.Path):
    import torch

    data = path.read_bytes()
    marker = b"\n255\n"
    header_end = data.index(marker) + len(marker)
    payload = data[header_end:]
    values = torch.tensor(list(payload), dtype=torch.float32).view(64, 64, 3)
    values = values.permute(2, 0, 1).unsqueeze(0) / 255.0
    return values


def round_float(value: float) -> float:
    return round(float(value), 10)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the PaperEnvBench ConvNeXt minimal classification task output.")
    parser.add_argument("--repo-dir", default=os.environ.get("PAPERENVBENCH_REPO_DIR", "repo"))
    parser.add_argument("--output-dir", default=os.environ.get("PAPERENVBENCH_OUTPUT_DIR", "artifacts"))
    parser.add_argument("--artifact-name", default="expected_artifact.json")
    args = parser.parse_args()

    repo_dir = pathlib.Path(args.repo_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (repo_dir / "models" / "convnext.py").exists():
        raise FileNotFoundError(f"missing ConvNeXt source file: {repo_dir / 'models' / 'convnext.py'}")

    sys.path.insert(0, str(repo_dir))

    import torch
    from models.convnext import ConvNeXt

    torch.set_num_threads(1)
    torch.manual_seed(20260516)

    image_path = output_dir / "synthetic_convnext_input.ppm"
    image_info = write_ppm(image_path)
    image_tensor = load_ppm_as_tensor(image_path)

    model = ConvNeXt(
        in_chans=3,
        num_classes=10,
        depths=[1, 1, 1, 1],
        dims=[8, 16, 32, 64],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
    )
    model.eval()

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)

    logits_cpu = logits.detach().cpu()
    probs_cpu = probabilities.detach().cpu()
    logits_shape = list(logits_cpu.shape)
    logits_flat = logits_cpu.reshape(-1)
    probs_flat = probs_cpu.reshape(-1)
    top_prob, top_class = probs_flat.max(dim=0)
    repo_commit = git_commit(repo_dir)
    logits_std = float(logits_flat.std(unbiased=False).item())
    logits_list = [round_float(v) for v in logits_flat.tolist()]
    probability_list = [round_float(v) for v in probs_flat.tolist()]

    checks = {
        "repo_commit_matches": repo_commit == EXPECTED_COMMIT,
        "repository_convnext_class_imported": getattr(ConvNeXt, "__module__", "") == "models.convnext",
        "synthetic_image_written": image_path.exists() and image_path.stat().st_size > 0,
        "logits_shape_matches": logits_shape == EXPECTED_SHAPE,
        "logits_are_finite": all(math.isfinite(float(v)) for v in logits_flat.tolist()),
        "logits_have_variance": logits_std > MIN_LOGITS_STD,
        "top_class_in_range": 0 <= int(top_class.item()) < EXPECTED_SHAPE[1],
    }

    summary = {
        "task_id": TASK_ID,
        "repo_commit": repo_commit,
        "python": sys.version.split()[0],
        "package_versions": {
            "torch": torch.__version__,
            "torchvision": metadata.version("torchvision"),
            "timm": metadata.version("timm"),
            "tensorboardX": metadata.version("tensorboardX"),
            "six": metadata.version("six"),
        },
        "input_image": image_info,
        "model": {
            "class": "models.convnext.ConvNeXt",
            "config": {
                "in_chans": 3,
                "num_classes": 10,
                "depths": [1, 1, 1, 1],
                "dims": [8, 16, 32, 64],
                "drop_path_rate": 0.0,
                "layer_scale_init_value": 1e-6,
            },
            "parameter_count": int(sum(param.numel() for param in model.parameters())),
            "training": bool(model.training),
        },
        "output": {
            "logits_shape": logits_shape,
            "logits": logits_list,
            "probabilities": probability_list,
            "top_class": int(top_class.item()),
            "top_probability": round_float(float(top_prob.item())),
            "logits_mean": round_float(float(logits_flat.mean().item())),
            "logits_std": round_float(logits_std),
            "logits_sha256": sha256_bytes(logits_cpu.numpy().tobytes()),
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
