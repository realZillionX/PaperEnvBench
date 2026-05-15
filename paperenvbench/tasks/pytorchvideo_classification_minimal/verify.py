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

_PEB_TASK_ID = "pytorchvideo_classification_minimal"
_PEB_EXPECTED_ARTIFACT_SHA256 = "8aa30f0408d690e3dbfaac8ca39c7f0867424f384c2946f6d0904146966d6ba2"
_PEB_REQUIRED_SIDE_ARTIFACTS = {'expected_clip.pt': 1000, 'expected_frame.ppm': 1000, 'expected_logits.pt': 100}


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
import platform
import subprocess
import sys
import time
from typing import Any


TASK_ID = "pytorchvideo_classification_minimal"
EXPECTED_COMMIT = "f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0"
EXPECTED_LOGITS_SHAPE = [1, 400]
MIN_LOGITS_STD = 1e-6


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


def round_float(value: float) -> float:
    return round(float(value), 10)


def write_ppm_frame(path: pathlib.Path, frame_tensor: Any) -> dict[str, Any]:
    import torch

    frame = frame_tensor.detach().cpu().clamp(0, 1)
    frame_u8 = (frame * 255.0).round().to(torch.uint8)
    height = int(frame_u8.shape[1])
    width = int(frame_u8.shape[2])
    payload = frame_u8.permute(1, 2, 0).contiguous().numpy().tobytes()
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + payload)
    return {
        "format": "ppm_p6",
        "height": height,
        "width": width,
        "channels": 3,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not_installed"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the PaperEnvBench PyTorchVideo minimal classification task.")
    parser.add_argument("--repo-dir", default=os.environ.get("PAPERENVBENCH_REPO_DIR", "repo"))
    parser.add_argument("--output-dir", default=os.environ.get("PAPERENVBENCH_OUTPUT_DIR", "artifacts"))
    parser.add_argument("--artifact-name", default="expected_artifact.json")
    args = parser.parse_args()

    repo_dir = pathlib.Path(args.repo_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hubconf_path = repo_dir / "hubconf.py"
    if not hubconf_path.exists():
        raise FileNotFoundError(f"missing PyTorchVideo hubconf: {hubconf_path}")

    sys.path.insert(0, str(repo_dir))

    import torch

    torch.set_num_threads(4)
    torch.manual_seed(20260516)

    model = torch.hub.load(str(repo_dir), "slow_r50", source="local", pretrained=False)
    model.eval()

    clip = torch.linspace(0, 1, steps=3 * 8 * 224 * 224, dtype=torch.float32).reshape(1, 3, 8, 224, 224)
    clip_path = output_dir / "expected_clip.pt"
    frame_path = output_dir / "expected_frame.ppm"
    logits_path = output_dir / "expected_logits.pt"

    frame_info = write_ppm_frame(frame_path, clip[0, :, 0])
    torch.save({"clip": clip, "shape": list(clip.shape), "description": "deterministic synthetic RGB video clip"}, clip_path)

    start = time.time()
    with torch.no_grad():
        logits = model(clip)
    elapsed_s = time.time() - start

    logits_cpu = logits.detach().cpu()
    torch.save({"logits": logits_cpu, "shape": list(logits_cpu.shape)}, logits_path)
    logits_flat = logits_cpu.reshape(-1)
    probabilities = torch.softmax(logits_cpu, dim=1)
    probs_flat = probabilities.reshape(-1)
    top_prob, top_class = probs_flat.max(dim=0)
    logits_shape = list(logits_cpu.shape)
    logits_std = float(logits_flat.std(unbiased=False).item())
    repo_commit = git_commit(repo_dir)

    checks = {
        "repo_commit_matches": repo_commit == EXPECTED_COMMIT,
        "hubconf_exists": hubconf_path.exists(),
        "model_loaded_from_local_hub": type(model).__module__.startswith("pytorchvideo."),
        "model_is_eval": not bool(model.training),
        "synthetic_clip_written": clip_path.exists() and clip_path.stat().st_size > 0,
        "representative_frame_written": frame_path.exists() and frame_path.stat().st_size > 0,
        "logits_artifact_written": logits_path.exists() and logits_path.stat().st_size > 0,
        "logits_shape_matches": logits_shape == EXPECTED_LOGITS_SHAPE,
        "logits_are_finite": all(math.isfinite(float(v)) for v in logits_flat.tolist()),
        "logits_have_variance": logits_std > MIN_LOGITS_STD,
        "probabilities_are_normalized": abs(float(probabilities.sum(dim=1).item()) - 1.0) < 1e-5,
        "top_class_in_range": 0 <= int(top_class.item()) < EXPECTED_LOGITS_SHAPE[1],
    }

    summary = {
        "task_id": TASK_ID,
        "repo_url": "https://github.com/facebookresearch/pytorchvideo",
        "repo_commit": repo_commit,
        "repo_commit_short": repo_commit[:7],
        "paper_title": "PyTorchVideo: A deep learning library for video understanding",
        "success_level": "L4_fallback" if all(checks.values()) else "below_L4",
        "fallback": {
            "used": True,
            "reason": "The CPU gold path does not download Kinetics checkpoints or external videos. It uses the pinned repository's local torch hub slow_r50 builder with pretrained=False and a deterministic synthetic video clip.",
            "checkpoint_download_attempted": False,
            "external_video_decode_required": False,
            "av_installed": package_version("av") != "not_installed",
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": {
            "torch": torch.__version__,
            "torchvision": package_version("torchvision"),
            "pytorchvideo": package_version("pytorchvideo"),
            "av": package_version("av"),
            "fvcore": package_version("fvcore"),
            "iopath": package_version("iopath"),
        },
        "model": {
            "loader": "torch.hub.load(local repo, slow_r50, pretrained=False)",
            "type": f"{type(model).__module__}.{type(model).__name__}",
            "parameter_count": int(sum(param.numel() for param in model.parameters())),
            "training": bool(model.training),
        },
        "input_video": {
            "path": "artifacts/expected_clip.pt",
            "shape": list(clip.shape),
            "dtype": str(clip.dtype),
            "value_min": round_float(float(clip.min().item())),
            "value_max": round_float(float(clip.max().item())),
            "sha256": sha256_file(clip_path),
            "size_bytes": clip_path.stat().st_size,
        },
        "representative_frame": {
            "path": "artifacts/expected_frame.ppm",
            **frame_info,
        },
        "output": {
            "logits_path": "artifacts/expected_logits.pt",
            "logits_shape": logits_shape,
            "logits_mean": round_float(float(logits_flat.mean().item())),
            "logits_std": round_float(logits_std),
            "logits_min": round_float(float(logits_flat.min().item())),
            "logits_max": round_float(float(logits_flat.max().item())),
            "logits_sha256": sha256_bytes(logits_cpu.numpy().tobytes()),
            "logits_file_sha256": sha256_file(logits_path),
            "top_class": int(top_class.item()),
            "top_probability": round_float(float(top_prob.item())),
            "probability_sum_max_abs_error": round_float(abs(float(probabilities.sum(dim=1).item()) - 1.0)),
            "forward_elapsed_s": round_float(elapsed_s),
        },
        "checks": checks,
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
