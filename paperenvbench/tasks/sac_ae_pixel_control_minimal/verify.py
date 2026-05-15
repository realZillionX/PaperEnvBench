#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import types
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parent
TASK_ID = "sac_ae_pixel_control_minimal"
REPO_URL = "https://github.com/denisyarats/pytorch_sac_ae"
EXPECTED_COMMIT = "7fa560e21c026c04bb8dcd72959ecf4e3424476c"
SUCCESS_LEVEL = "L4_cpu_deterministic_fallback"
OBS_SHAPE = [9, 84, 84]
ACTION_SHAPE = [2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return ""


def install_gym_stub() -> None:
    if "gym" in sys.modules:
        return
    gym = types.ModuleType("gym")

    class Wrapper:
        def __init__(self, env: Any) -> None:
            self.env = env

    class Box:
        def __init__(self, low: Any, high: Any, shape: Any, dtype: Any) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    gym.Wrapper = Wrapper
    gym.spaces = types.SimpleNamespace(Box=Box)
    sys.modules["gym"] = gym


def round_float(value: float) -> float:
    return round(float(value), 10)


def generate_artifact(repo_dir: Path, output_dir: Path) -> dict[str, Any]:
    install_gym_stub()
    sys.path.insert(0, str(repo_dir.resolve()))

    import torch
    from sac_ae import SacAeAgent

    torch.set_num_threads(2)
    torch.manual_seed(20260516)
    device = torch.device("cpu")
    agent = SacAeAgent(
        obs_shape=tuple(OBS_SHAPE),
        action_shape=tuple(ACTION_SHAPE),
        device=device,
        hidden_dim=32,
        encoder_feature_dim=16,
        num_layers=4,
        num_filters=8,
        init_temperature=0.1,
    )
    agent.train(False)

    obs = torch.arange(1 * OBS_SHAPE[0] * OBS_SHAPE[1] * OBS_SHAPE[2], dtype=torch.float32).reshape(1, *OBS_SHAPE)
    obs = torch.remainder(obs, 256.0)
    with torch.no_grad():
        latent = agent.critic.encoder(obs)
        mu, pi, log_pi, log_std = agent.actor(obs)
        q1, q2 = agent.critic(obs, mu)
        rec = agent.decoder(latent)

    route = {
        "entrypoint": "sac_ae.SacAeAgent",
        "encoder": "encoder.PixelEncoder",
        "actor": "sac_ae.Actor",
        "critic": "sac_ae.Critic with twin QFunction heads",
        "decoder": "decoder.PixelDecoder",
        "train_entrypoint": "train.py",
        "full_training_env": "dmc2gym.make(..., from_pixels=True) plus FrameStack",
    }
    numeric = {
        "latent_shape": list(latent.shape),
        "mu_shape": list(mu.shape),
        "pi_shape": list(pi.shape),
        "log_pi_shape": list(log_pi.shape),
        "log_std_shape": list(log_std.shape),
        "q1_shape": list(q1.shape),
        "q2_shape": list(q2.shape),
        "reconstruction_shape": list(rec.shape),
        "mu": [round_float(v) for v in mu.reshape(-1).tolist()],
        "q_values": [round_float(q1.item()), round_float(q2.item())],
        "latent_mean_std": [round_float(latent.mean().item()), round_float(latent.std(unbiased=False).item())],
        "reconstruction_mean_std": [round_float(rec.mean().item()), round_float(rec.std(unbiased=False).item())],
    }
    checks = {
        "repo_commit_matches": git_commit(repo_dir) == EXPECTED_COMMIT,
        "pixel_observation_route": OBS_SHAPE == [9, 84, 84],
        "actor_route_present": numeric["mu_shape"] == [1, 2] and numeric["pi_shape"] == [1, 2],
        "critic_route_present": numeric["q1_shape"] == [1, 1] and numeric["q2_shape"] == [1, 1],
        "decoder_route_present": numeric["reconstruction_shape"] == [1, 9, 84, 84],
        "action_bounds_valid": all(-1.0 <= value <= 1.0 for value in numeric["mu"]),
        "q_outputs_valid": all(math.isfinite(value) for value in numeric["q_values"]),
        "reconstruction_shape_valid": numeric["reconstruction_shape"] == OBS_SHAPE[:0] + [1, *OBS_SHAPE],
        "cpu_fallback_declared": True,
    }
    artifact = {
        "task_id": TASK_ID,
        "repo": {
            "url": REPO_URL,
            "commit": git_commit(repo_dir),
            "commit_short": git_commit(repo_dir)[:7],
            "paper_title": "Improving Sample Efficiency in Model-Free Reinforcement Learning from Images",
        },
        "success_level": SUCCESS_LEVEL if all(checks.values()) else "below_L4",
        "fallback": {
            "used": True,
            "reason": "Full DMControl pixel training requires MuJoCo rendering and long replay-buffer training; the CPU gold route validates the pinned SAC-AE pixel network path deterministically.",
            "full_env_requires": ["dm_control", "dmc2gym", "MuJoCo renderer", "Python 3.6-era CUDA stack"],
        },
        "input": {
            "observation_shape": OBS_SHAPE,
            "action_shape": ACTION_SHAPE,
            "frame_stack": 3,
            "image_size": 84,
            "pixel_dtype": "float32 in [0, 255] before PixelEncoder normalization",
        },
        "route": route,
        "numeric": numeric,
        "checks": checks,
        "sha256": {
            "route": canonical_sha256(route),
            "numeric": canonical_sha256(numeric),
            "checks": canonical_sha256(checks),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "expected_artifact.json").write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_path = artifact_dir / "expected_artifact.json"
    if not artifact_path.exists():
        raise AssertionError(f"missing artifact: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if payload.get("task_id") != TASK_ID:
        raise AssertionError(f"wrong task_id: {payload.get('task_id')}")
    if payload.get("success_level") != SUCCESS_LEVEL:
        raise AssertionError(f"unexpected success_level: {payload.get('success_level')}")
    if payload.get("repo", {}).get("url") != REPO_URL:
        raise AssertionError("repo url mismatch")
    if payload.get("repo", {}).get("commit") != EXPECTED_COMMIT:
        raise AssertionError("repo commit mismatch")
    if payload.get("input", {}).get("observation_shape") != OBS_SHAPE:
        raise AssertionError("pixel observation shape mismatch")
    if payload.get("input", {}).get("action_shape") != ACTION_SHAPE:
        raise AssertionError("action shape mismatch")

    route = payload.get("route", {})
    for key, expected in {
        "entrypoint": "sac_ae.SacAeAgent",
        "encoder": "encoder.PixelEncoder",
        "actor": "sac_ae.Actor",
        "critic": "sac_ae.Critic with twin QFunction heads",
        "decoder": "decoder.PixelDecoder",
    }.items():
        if route.get(key) != expected:
            raise AssertionError(f"route mismatch for {key}: {route.get(key)}")

    numeric = payload.get("numeric", {})
    expected_shapes = {
        "latent_shape": [1, 16],
        "mu_shape": [1, 2],
        "pi_shape": [1, 2],
        "log_pi_shape": [1, 1],
        "log_std_shape": [1, 2],
        "q1_shape": [1, 1],
        "q2_shape": [1, 1],
        "reconstruction_shape": [1, 9, 84, 84],
    }
    for key, expected in expected_shapes.items():
        if numeric.get(key) != expected:
            raise AssertionError(f"numeric shape mismatch for {key}: {numeric.get(key)}")
    for key in ["mu", "q_values", "latent_mean_std", "reconstruction_mean_std"]:
        values = numeric.get(key)
        if not isinstance(values, list) or not values:
            raise AssertionError(f"missing numeric vector: {key}")
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            raise AssertionError(f"non-finite numeric vector: {key}")
    if not all(-1.0 <= float(value) <= 1.0 for value in numeric["mu"]):
        raise AssertionError("mu action is outside tanh bounds")

    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        raise AssertionError({"checks": checks})
    sha = payload.get("sha256", {})
    if sha.get("route") != canonical_sha256(route):
        raise AssertionError("route sha256 mismatch")
    if sha.get("numeric") != canonical_sha256(numeric):
        raise AssertionError("numeric sha256 mismatch")
    if sha.get("checks") != canonical_sha256(checks):
        raise AssertionError("checks sha256 mismatch")

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": SUCCESS_LEVEL,
        "artifact_dir": str(artifact_dir.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "checks": checks,
        "observed": {
            "repo_commit": payload.get("repo", {}).get("commit"),
            "route": route,
            "numeric": numeric,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the PaperEnvBench SAC-AE pixel-control minimal task.")
    parser.add_argument("--repo-dir", type=Path, default=Path("repo"))
    parser.add_argument("--output-dir", type=Path, default=TASK_ROOT / "artifacts")
    parser.add_argument("--artifact-dir", type=Path, default=TASK_ROOT / "artifacts")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        if args.check_only:
            result = validate_artifact(args.artifact_dir)
        else:
            result = generate_artifact(args.repo_dir, args.output_dir)
    except Exception as exc:
        print(json.dumps({"task_id": TASK_ID, "status": "fail", "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
