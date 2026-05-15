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
TASK_ID = "gym_cartpole_minimal"
EXPECTED_COMMIT = "bc212954b6713d5db303b3ead124de6cba66063e"
SUCCESS_LEVEL = "L4_cpu_deterministic_fallback"
EXPECTED_ACTIONS = [1, 0, 1, 1, 0]
EXPECTED_RESET = [0.01823519, -0.0446179, -0.02796401, -0.03156282]
EXPECTED_FINAL_OBSERVATION = [0.02946884, 0.15292421, -0.05643985, -0.37814762]
REQUIRED_ROUTE_TERMS = [
    "gym.make('CartPole-v1')",
    "gym.envs.classic_control.cartpole.CartPoleEnv",
    "CartPoleEnv.reset",
    "CartPoleEnv.step",
    "CPU deterministic fallback",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_close_list(observed: list[Any], expected: list[float], path: str, tolerance: float = 1e-7) -> None:
    require(len(observed) == len(expected), f"{path} length mismatch")
    for index, (obs_value, exp_value) in enumerate(zip(observed, expected)):
        require(isinstance(obs_value, (int, float)), f"{path}[{index}] is not numeric")
        require(math.isfinite(float(obs_value)), f"{path}[{index}] is not finite")
        require(abs(float(obs_value) - exp_value) <= tolerance, f"{path}[{index}] mismatch: {obs_value} != {exp_value}")


def validate_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_path = artifact_dir / "expected_artifact.json"
    require(artifact_path.exists(), f"missing artifact: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    require(payload.get("task_id") == TASK_ID, f"wrong task_id: {payload.get('task_id')}")
    require(payload.get("success_level") == SUCCESS_LEVEL, f"wrong success_level: {payload.get('success_level')}")
    require(payload.get("repo", {}).get("commit") == EXPECTED_COMMIT, "repo commit mismatch")

    route = payload.get("route", {})
    route_text = " ".join(str(value) for value in route.values())
    missing_terms = [term for term in REQUIRED_ROUTE_TERMS if term not in route_text]
    require(not missing_terms, f"route evidence is missing terms: {missing_terms}")
    require(route.get("device") == "cpu", "route must record CPU execution")
    require(route.get("render_mode") == "None", "route must avoid rendering dependencies")

    reset = payload.get("reset", {})
    require(reset.get("seed") == 123, "reset seed mismatch")
    assert_close_list(reset.get("observation", []), EXPECTED_RESET, "reset.observation")
    require(reset.get("info") == {}, "reset info should be empty")

    rollout = payload.get("rollout", {})
    actions = rollout.get("actions")
    steps = rollout.get("steps")
    require(actions == EXPECTED_ACTIONS, f"unexpected actions: {actions}")
    require(isinstance(steps, list) and len(steps) == len(EXPECTED_ACTIONS), "unexpected step count")
    require(float(rollout.get("total_reward", -1.0)) == 5.0, "unexpected total reward")
    require(rollout.get("all_not_terminated") is True, "rollout should stay unterminated")
    require(rollout.get("all_not_truncated") is True, "rollout should stay untruncated")
    assert_close_list(rollout.get("final_observation", []), EXPECTED_FINAL_OBSERVATION, "rollout.final_observation")

    for index, row in enumerate(steps):
        require(row.get("step") == index + 1, f"step index mismatch at {index}")
        require(row.get("action") == EXPECTED_ACTIONS[index], f"action mismatch at {index}")
        require(float(row.get("reward", -1.0)) == 1.0, f"reward mismatch at {index}")
        require(row.get("terminated") is False, f"terminated at {index}")
        require(row.get("truncated") is False, f"truncated at {index}")
        assert_close_list(row.get("observation", []), row.get("observation", []), f"steps[{index}].observation")

    physics = payload.get("physics", {})
    require(float(physics.get("gravity")) == 9.8, "gravity mismatch")
    require(float(physics.get("force_mag")) == 10.0, "force magnitude mismatch")
    require(float(physics.get("tau")) == 0.02, "tau mismatch")
    require(abs(float(physics.get("theta_threshold_radians")) - 0.20943951023931953) <= 1e-12, "theta threshold mismatch")
    require(float(physics.get("x_threshold")) == 2.4, "x threshold mismatch")

    semantic = payload.get("semantic", {})
    semantic_text = " ".join(str(value) for value in semantic.values())
    require("CartPole" in semantic_text, "semantic summary must mention CartPole")
    require("reset" in semantic_text and "step" in semantic_text, "semantic summary must mention reset and step")
    require("no rendering" in semantic_text, "semantic summary must mention no rendering")

    checks = payload.get("checks", {})
    require(isinstance(checks, dict) and checks, "missing checks")
    require(all(value is True for value in checks.values()), f"failing checks: {checks}")

    return {
        "artifact_dir": str(artifact_dir.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "checks": checks,
        "observed": {
            "actions": actions,
            "final_observation": rollout.get("final_observation"),
            "repo_commit": payload.get("repo", {}).get("commit"),
            "rollout_sha256": canonical_sha256(rollout),
            "total_reward": rollout.get("total_reward"),
        },
        "status": "pass",
        "success_level": SUCCESS_LEVEL,
        "task_id": TASK_ID,
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
        print(json.dumps({"error": str(exc), "status": "fail", "task_id": TASK_ID}, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
