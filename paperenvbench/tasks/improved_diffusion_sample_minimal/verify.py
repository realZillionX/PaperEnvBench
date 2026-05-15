#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


TASK_ID = "improved_diffusion_sample_minimal"
TASK_ROOT = Path(__file__).resolve().parent
REPO_URL = "https://github.com/openai/improved-diffusion"
COMMIT = "1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5"
SEED = 20260516
IMAGE_SIZE = 32
SAMPLE_SHAPE = [1, 3, 32, 32]
MODEL_KWARGS = {
    "image_size": 32,
    "num_channels": 8,
    "num_res_blocks": 1,
    "num_heads": 1,
    "num_heads_upsample": -1,
    "attention_resolutions": "16",
    "dropout": 0.0,
    "learn_sigma": False,
    "sigma_small": False,
    "class_cond": False,
    "diffusion_steps": 4,
    "noise_schedule": "linear",
    "timestep_respacing": "4",
    "use_kl": False,
    "predict_xstart": False,
    "rescale_timesteps": True,
    "rescale_learned_sigmas": True,
    "use_checkpoint": False,
    "use_scale_shift_norm": True,
}
EXPECTED_CHECKS = [
    "repo_commit_matches",
    "script_route_matches",
    "model_route_matches",
    "diffusion_route_matches",
    "checkpoint_loaded",
    "sample_shape_matches",
    "sample_uint8_range_valid",
    "sample_statistics_valid",
    "artifact_records_cpu_fallback",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_commit(repo_dir: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_payload(payload: dict[str, Any]) -> dict[str, bool]:
    checks = payload.get("checks", {})
    observed = payload.get("observed", {})
    route = payload.get("route", {})
    repo = payload.get("repo", {})
    model = route.get("model", {})
    diffusion = route.get("diffusion", {})
    sample = route.get("sample", {})
    return {
        "repo_commit_matches": repo.get("commit") == COMMIT,
        "script_route_matches": sample.get("script") == "scripts/image_sample.py"
        and sample.get("loop_function") == "GaussianDiffusion.p_sample_loop",
        "model_route_matches": model.get("factory") == "improved_diffusion.script_util.create_model_and_diffusion"
        and model.get("unet_class") == "improved_diffusion.unet.UNetModel"
        and model.get("kwargs") == MODEL_KWARGS,
        "diffusion_route_matches": diffusion.get("class") == "improved_diffusion.respace.SpacedDiffusion"
        and diffusion.get("num_timesteps") == MODEL_KWARGS["diffusion_steps"]
        and diffusion.get("timestep_respacing") == MODEL_KWARGS["timestep_respacing"],
        "checkpoint_loaded": bool(model.get("checkpoint_loaded")),
        "sample_shape_matches": observed.get("sample_shape") == SAMPLE_SHAPE,
        "sample_uint8_range_valid": observed.get("sample_uint8_min") >= 0
        and observed.get("sample_uint8_max") <= 255,
        "sample_statistics_valid": 0.0 <= observed.get("sample_uint8_mean", -1.0) <= 255.0
        and observed.get("sample_uint8_std", -1.0) > 0.0,
        "artifact_records_cpu_fallback": payload.get("success_level") == "L4_fallback_cpu_tiny_unet_sampling"
        and payload.get("environment", {}).get("device") == "cpu"
        and payload.get("fallback_boundary", {}).get("public_large_checkpoints_required") is False,
        "embedded_checks_true": all(bool(checks.get(name)) for name in EXPECTED_CHECKS),
    }


def verify_check_only(artifact_dir: Path) -> dict[str, Any]:
    artifact_path = artifact_dir / "expected_artifact.json"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing expected artifact: {artifact_path}")
    payload = load_json(artifact_path)
    checks = check_payload(payload)
    success = all(checks.values())
    result = {
        "task_id": TASK_ID,
        "success": success,
        "mode": "check-only",
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "checks": checks,
    }
    if not success:
        raise AssertionError(json.dumps(result, indent=2, sort_keys=True))
    return result


def prepare_import_path(repo_dir: Path, run_root: Path) -> None:
    repo_str = str(repo_dir.resolve())
    run_root_str = str(run_root.resolve())
    cleaned = [item for item in sys.path if item not in ("", run_root_str)]
    sys.path[:] = [repo_str] + cleaned


def generate_artifact(repo_dir: Path, artifact_dir: Path) -> dict[str, Any]:
    import numpy as np
    import torch

    from improved_diffusion.script_util import create_model_and_diffusion

    if repo_commit(repo_dir) != COMMIT:
        raise RuntimeError(f"Unexpected commit: {repo_commit(repo_dir)}; expected {COMMIT}")

    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    model, diffusion = create_model_and_diffusion(**MODEL_KWARGS)
    checkpoint_path = artifact_dir / "tiny_initialized_checkpoint.pt"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    torch.manual_seed(SEED + 1)
    with torch.no_grad():
        sample = diffusion.p_sample_loop(
            model,
            tuple(SAMPLE_SHAPE),
            clip_denoised=True,
            model_kwargs={},
            device=torch.device("cpu"),
            progress=False,
        )
    sample_uint8 = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample_np = sample_uint8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    npz_path = artifact_dir / "expected_sample.npz"
    np.savez(npz_path, arr_0=sample_np)

    payload = {
        "task_id": TASK_ID,
        "success": True,
        "success_level": "L4_fallback_cpu_tiny_unet_sampling",
        "repo": {"url": REPO_URL, "commit": repo_commit(repo_dir)},
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
        },
        "route": {
            "sample": {
                "script": "scripts/image_sample.py",
                "loop_function": "GaussianDiffusion.p_sample_loop",
                "output_format": "npz arr_0 with NHWC uint8 samples",
                "shape": SAMPLE_SHAPE,
                "seed": SEED,
            },
            "model": {
                "factory": "improved_diffusion.script_util.create_model_and_diffusion",
                "unet_class": "improved_diffusion.unet.UNetModel",
                "kwargs": MODEL_KWARGS,
                "checkpoint_loaded": True,
                "checkpoint_kind": "deterministic tiny initialized state_dict",
            },
            "diffusion": {
                "class": "improved_diffusion.respace.SpacedDiffusion",
                "num_timesteps": int(diffusion.num_timesteps),
                "timestep_respacing": MODEL_KWARGS["timestep_respacing"],
                "noise_schedule": MODEL_KWARGS["noise_schedule"],
            },
        },
        "observed": {
            "sample_shape": list(sample.shape),
            "sample_uint8_shape": list(sample_np.shape),
            "sample_uint8_min": int(sample_np.min()),
            "sample_uint8_max": int(sample_np.max()),
            "sample_uint8_mean": round(float(sample_np.mean()), 8),
            "sample_uint8_std": round(float(sample_np.std()), 8),
            "sample_first_12_values": [int(value) for value in sample_np.reshape(-1)[:12].tolist()],
        },
        "artifacts": {
            "summary": "artifacts/expected_artifact.json",
            "sample_npz": "artifacts/expected_sample.npz",
            "sample_npz_sha256": sha256(npz_path),
        },
        "fallback_boundary": {
            "public_large_checkpoints_required": False,
            "reason": "The hidden CPU verifier checks the official sampling route with a tiny deterministic checkpoint instead of requiring 100M-270M public checkpoints.",
            "not_accepted": "Import-only success, README-only flag extraction, or an artifact that omits model/diffusion/sample routing.",
        },
    }
    payload["checks"] = check_payload(payload)
    summary_path = artifact_dir / "expected_artifact.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=TASK_ROOT / "artifacts")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    if args.check_only:
        result = verify_check_only(artifact_dir)
    else:
        run_root = Path(os.environ.get("PAPERENVBENCH_RUN_ROOT", os.getcwd())).resolve()
        repo_dir = (args.repo_dir or Path(os.environ.get("IMPROVED_DIFFUSION_REPO", run_root / "improved-diffusion"))).resolve()
        if not (repo_dir / "improved_diffusion" / "script_util.py").exists():
            raise FileNotFoundError(f"improved-diffusion repo not found or incomplete: {repo_dir}")
        prepare_import_path(repo_dir, run_root)
        result = generate_artifact(repo_dir, artifact_dir)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"{TASK_ID}: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
