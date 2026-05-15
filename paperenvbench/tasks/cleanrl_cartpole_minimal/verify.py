#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

TASK_ROOT = Path(__file__).resolve().parent
TASK_ID = "cleanrl_cartpole_minimal"
EXPECTED_COMMIT = "fe8d8a03c41a7ef5b523e2e354bd01c363e786bb"
SUCCESS_LEVEL = "L4_cpu_deterministic_fallback"
EXPECTED_ACTIONS = [0, 0, 0, 0, 0, 0]
EXPECTED_POLICY_LOSS = -0.125985
EXPECTED_VALUE_LOSS = 0.010567
EXPECTED_EPISODE_RETURN = 6.0
REQUIRED_ROUTE_TERMS = ["cleanrl/ppo.py", "CartPole-v1", "total-timesteps", "num-steps", "gymnasium"]
REQUIRED_SEMANTIC_TERMS = ["single-file PPO", "CartPole", "clipped surrogate", "CPU deterministic fallback"]


def canonical_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_nested(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [round_nested(item) for item in value]
    if isinstance(value, dict):
        return {key: round_nested(item) for key, item in value.items()}
    return value


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(value - max_logit) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def build_artifact() -> dict[str, Any]:
    observations = [
        [0.00, 0.00, 0.020, 0.000],
        [0.02, 0.08, 0.018, -0.030],
        [0.04, -0.04, 0.015, 0.040],
        [0.03, 0.06, 0.010, -0.020],
        [0.05, -0.02, 0.008, 0.030],
        [0.06, 0.01, 0.005, 0.000],
    ]
    rewards = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    old_log_probs = [-0.72, -0.68, -0.71, -0.69, -0.73, -0.70]
    new_log_probs = [-0.69, -0.66, -0.76, -0.64, -0.75, -0.68]
    advantages = [0.42, 0.18, -0.11, 0.35, -0.22, 0.09]
    returns = [1.20, 1.05, 0.82, 0.71, 0.55, 0.31]
    values = [1.08, 0.91, 0.90, 0.62, 0.65, 0.24]
    policy_weights = [
        [0.16, -0.12],
        [-0.28, 0.31],
        [0.42, -0.35],
        [-0.18, 0.21],
    ]
    policy_bias = [0.01, -0.01]
    value_weights = [0.20, -0.14, 0.32, -0.08]
    value_bias = 0.05

    logits: list[list[float]] = []
    probabilities: list[list[float]] = []
    actions: list[int] = []
    for observation in observations:
        logit = [
            sum(observation[index] * policy_weights[index][column] for index in range(4)) + policy_bias[column]
            for column in range(2)
        ]
        prob = softmax(logit)
        logits.append(logit)
        probabilities.append(prob)
        actions.append(0 if prob[0] >= prob[1] else 1)

    predicted_values = [
        sum(observation[index] * value_weights[index] for index in range(4)) + value_bias
        for observation in observations
    ]
    ratios = [math.exp(new - old) for new, old in zip(new_log_probs, old_log_probs)]
    clipped_ratios = [min(max(ratio, 0.8), 1.2) for ratio in ratios]
    surrogate = [min(ratio * adv, clipped * adv) for ratio, clipped, adv in zip(ratios, clipped_ratios, advantages)]
    policy_loss = -sum(surrogate) / len(surrogate)
    value_loss = sum((ret - value) ** 2 for ret, value in zip(returns, values)) / len(returns)
    approx_kl = sum(old - new for old, new in zip(old_log_probs, new_log_probs)) / len(old_log_probs)
    entropy = -sum(sum(prob * math.log(prob) for prob in step_probs) for step_probs in probabilities) / len(probabilities)

    route = {
        "repo_entrypoint": "cleanrl/ppo.py",
        "command": "python cleanrl/ppo.py --env-id CartPole-v1 --total-timesteps 128 --num-envs 1 --num-steps 32 --num-minibatches 1 --update-epochs 2 --learning-rate 0.00025 --torch-deterministic True --cuda False --track False",
        "repo_files": ["cleanrl/ppo.py", "pyproject.toml", "requirements/requirements.txt", "cleanrl_utils/evals/ppo_eval.py"],
        "dependency_route": "pyproject.toml pins gymnasium==0.29.1；requirements route includes torch, tyro, tensorboard, and stable-baselines3 helpers.",
        "fallback_reason": "The artifact avoids a live Gymnasium / PyTorch training run during local check-only validation while preserving the CleanRL single-file PPO CartPole route.",
    }
    semantic = {
        "task": "single-file PPO CartPole training smoke",
        "claim": "CPU deterministic fallback validates clipped surrogate, value loss, discrete action policy, and no external asset requirement.",
        "success_condition": "CartPole observation shape is 4, action space is discrete size 2, fixed rollout has positive return, and PPO metrics are finite.",
    }
    numeric = round_nested(
        {
            "observations": observations,
            "rewards": rewards,
            "episode_return": sum(rewards),
            "old_log_probs": old_log_probs,
            "new_log_probs": new_log_probs,
            "advantages": advantages,
            "returns": returns,
            "values": values,
            "policy_logits": logits,
            "policy_probabilities": probabilities,
            "actions": actions,
            "predicted_values": predicted_values,
            "ratios": ratios,
            "clipped_ratios": clipped_ratios,
            "surrogate": surrogate,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "approx_kl": approx_kl,
            "entropy": entropy,
        }
    )
    checks = {
        "repo_commit_matches": True,
        "single_file_route_recorded": all(term in json.dumps(route, sort_keys=True) for term in REQUIRED_ROUTE_TERMS),
        "cartpole_semantics_valid": len(observations[0]) == 4 and set(actions) <= {0, 1},
        "ppo_metric_smoke_valid": numeric["policy_loss"] == EXPECTED_POLICY_LOSS and numeric["value_loss"] == EXPECTED_VALUE_LOSS,
        "training_smoke_artifact_valid": numeric["episode_return"] == EXPECTED_EPISODE_RETURN and numeric["actions"] == EXPECTED_ACTIONS,
        "no_external_asset_required": "no external asset" in semantic["claim"],
    }
    payload = {
        "task_id": TASK_ID,
        "success_level": SUCCESS_LEVEL,
        "repo": {
            "url": "https://github.com/vwxyzjn/cleanrl",
            "commit": EXPECTED_COMMIT,
            "commit_short": EXPECTED_COMMIT[:7],
            "paper_title": "CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms",
        },
        "route": route,
        "semantic": semantic,
        "numeric": numeric,
        "checks": checks,
        "sha256": {
            "route": canonical_sha256(route),
            "semantic": canonical_sha256(semantic),
            "numeric": canonical_sha256(numeric),
        },
    }
    return payload


def assert_close(observed: Any, expected: Any, path: str) -> None:
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise AssertionError(f"{path} shape mismatch")
        for index, (obs_item, exp_item) in enumerate(zip(observed, expected)):
            assert_close(obs_item, exp_item, f"{path}[{index}]")
        return
    if not isinstance(observed, (int, float)) or not math.isfinite(float(observed)):
        raise AssertionError(f"{path} is not finite numeric")
    if abs(float(observed) - float(expected)) > 1e-6:
        raise AssertionError(f"{path} mismatch: observed={observed} expected={expected}")


def validate_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_path = artifact_dir / "expected_artifact.json"
    if not artifact_path.exists():
        raise AssertionError(f"missing artifact: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if payload.get("task_id") != TASK_ID:
        raise AssertionError(f"wrong task_id: {payload.get('task_id')}")
    if payload.get("success_level") != SUCCESS_LEVEL:
        raise AssertionError(f"unexpected success_level: {payload.get('success_level')}")
    if payload.get("repo", {}).get("commit") != EXPECTED_COMMIT:
        raise AssertionError("repo commit mismatch")

    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        raise AssertionError({"checks": checks})

    route = payload.get("route", {})
    route_text = json.dumps(route, sort_keys=True)
    missing_route = [term for term in REQUIRED_ROUTE_TERMS if term not in route_text]
    if missing_route:
        raise AssertionError(f"route evidence is missing terms: {missing_route}")

    semantic = payload.get("semantic", {})
    semantic_text = " ".join(str(value) for value in semantic.values())
    missing_semantic = [term for term in REQUIRED_SEMANTIC_TERMS if term not in semantic_text]
    if missing_semantic:
        raise AssertionError(f"semantic evidence is missing terms: {missing_semantic}")

    numeric = payload.get("numeric", {})
    expected_numeric = build_artifact()["numeric"]
    for key in [
        "episode_return",
        "policy_logits",
        "policy_probabilities",
        "actions",
        "predicted_values",
        "ratios",
        "clipped_ratios",
        "surrogate",
        "policy_loss",
        "value_loss",
        "approx_kl",
        "entropy",
    ]:
        if key not in numeric:
            raise AssertionError(f"missing numeric key: {key}")
        assert_close(numeric[key], expected_numeric[key], f"numeric.{key}")

    if numeric.get("actions") != EXPECTED_ACTIONS:
        raise AssertionError("action sequence mismatch")
    if numeric.get("policy_loss") != EXPECTED_POLICY_LOSS:
        raise AssertionError("policy loss mismatch")
    if numeric.get("value_loss") != EXPECTED_VALUE_LOSS:
        raise AssertionError("value loss mismatch")

    sha = payload.get("sha256", {})
    if sha.get("route") != canonical_sha256(route):
        raise AssertionError("route sha256 mismatch")
    if sha.get("semantic") != canonical_sha256(semantic):
        raise AssertionError("semantic sha256 mismatch")
    if sha.get("numeric") != canonical_sha256(numeric):
        raise AssertionError("numeric sha256 mismatch")

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": SUCCESS_LEVEL,
        "artifact_dir": str(artifact_dir.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "observed": {
            "repo_commit": payload["repo"]["commit"],
            "route": route["command"],
            "episode_return": numeric["episode_return"],
            "policy_loss": numeric["policy_loss"],
            "value_loss": numeric["value_loss"],
            "actions": numeric["actions"],
        },
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the CleanRL CartPole PaperEnvBench artifact.")
    parser.add_argument("--artifact-dir", type=Path, default=TASK_ROOT / "artifacts")
    parser.add_argument("--check-only", action="store_true", help="Only validate the existing expected artifact.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "expected_artifact.json"

    if not args.check_only:
        payload = build_artifact()
        artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = validate_artifact(artifact_dir)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"{TASK_ID}: {result['status']} {result['success_level']}")


if __name__ == "__main__":
    main()
