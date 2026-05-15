#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

TASK_ROOT = Path(__file__).resolve().parent
TASK_ID = "stable_baselines3_cartpole_minimal"
EXPECTED_COMMIT = "8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458"
SUCCESS_LEVEL = "L4_cpu_deterministic_fallback"
EXPECTED_ENV_ID = "CartPole-v1"
EXPECTED_ALGORITHM = "PPO"
EXPECTED_REWARD = 8.0
REQUIRED_ROUTE_TERMS = [
    "stable_baselines3",
    "PPO",
    "MlpPolicy",
    "gymnasium",
    "CartPole-v1",
    "model.learn",
    "model.predict",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_float_list(values: Any, expected_len: int, name: str) -> list[float]:
    require(isinstance(values, list) and len(values) == expected_len, f"{name} length mismatch")
    result = []
    for value in values:
        require(isinstance(value, (int, float)) and math.isfinite(float(value)), f"{name} contains non-finite value")
        result.append(float(value))
    return result


def validate_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_path = artifact_dir / "expected_artifact.json"
    require(artifact_path.exists(), f"missing artifact: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    require(payload.get("task_id") == TASK_ID, f"wrong task_id: {payload.get('task_id')}")
    require(payload.get("success") is True, "success flag must be true")
    require(payload.get("success_level") == SUCCESS_LEVEL, f"wrong success_level: {payload.get('success_level')}")
    require(payload.get("repo", {}).get("commit") == EXPECTED_COMMIT, "repo commit mismatch")

    route = payload.get("route", {})
    route_text = json.dumps(route, sort_keys=True)
    missing = [term for term in REQUIRED_ROUTE_TERMS if term not in route_text]
    require(not missing, f"route evidence is missing terms: {missing}")
    require("stable_baselines3/ppo/ppo.py" in route_text, "PPO source path is not recorded")
    require("stable_baselines3/common/env_util.py" in route_text, "environment factory source path is not recorded")

    env = payload.get("environment", {})
    require(env.get("device") == "cpu", "device must be cpu")
    require(env.get("env_id") == EXPECTED_ENV_ID, "environment id mismatch")
    require(env.get("gym_api") == "gymnasium", "Gymnasium route must be recorded")
    require(env.get("seed") == 123, "seed mismatch")
    require(env.get("external_dataset_required") is False, "CartPole check-only must not require an external dataset")

    config = payload.get("ppo_config", {})
    require(config.get("policy") == "MlpPolicy", "PPO policy mismatch")
    require(config.get("device") == "cpu", "PPO device mismatch")
    require(config.get("total_timesteps") == 64, "training smoke timestep mismatch")
    require(config.get("n_steps") == 16 and config.get("batch_size") == 16, "PPO batch shape mismatch")

    rollout = payload.get("rollout", {})
    observation = require_float_list(rollout.get("initial_observation"), 4, "initial_observation")
    require(all(-0.1 <= value <= 0.1 for value in observation), "CartPole reset observation is outside expected seeded range")
    actions = rollout.get("actions")
    require(isinstance(actions, list) and len(actions) == 8, "actions length mismatch")
    require(all(action in (0, 1) for action in actions), "CartPole actions must be discrete 0/1 values")
    rewards = require_float_list(rollout.get("rewards"), 8, "rewards")
    require(all(math.isclose(reward, 1.0, abs_tol=1e-12) for reward in rewards), "CartPole rewards must be one per live step")
    require(rollout.get("terminated") == [False] * 8, "termination flags mismatch")
    require(rollout.get("truncated") == [False] * 8, "truncation flags mismatch")
    episode_reward = sum(rewards)
    require(math.isclose(float(rollout.get("episode_reward")), episode_reward, abs_tol=1e-12), "episode_reward mismatch")
    require(math.isclose(episode_reward, EXPECTED_REWARD, abs_tol=1e-12), f"unexpected reward: {episode_reward}")
    require(episode_reward >= float(rollout.get("minimum_reward")), "reward below minimum")

    semantic = payload.get("semantic", {})
    semantic_text = json.dumps(semantic, sort_keys=True)
    require("reinforcement learning simulation" in semantic_text, "semantic task not recorded")
    require("Stable-Baselines3 PPO" in semantic_text, "PPO semantic route not recorded")
    require(semantic.get("import_success_alone_is_sufficient") is False, "import-only success boundary missing")

    checks = payload.get("checks", {})
    require(isinstance(checks, dict) and checks, "missing checks")
    require(all(value is True for value in checks.values()), f"failing checks: {checks}")

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": SUCCESS_LEVEL,
        "artifact_dir": str(artifact_dir.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "checks": checks,
        "observed": {
            "repo_commit": payload.get("repo", {}).get("commit"),
            "algorithm": EXPECTED_ALGORITHM,
            "environment_id": env.get("env_id"),
            "seed": env.get("seed"),
            "episode_reward": episode_reward,
            "steps": len(actions),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=TASK_ROOT / "artifacts")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = validate_artifact(args.artifact_dir)
    except Exception as exc:
        print(json.dumps({"task_id": TASK_ID, "status": "fail", "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
