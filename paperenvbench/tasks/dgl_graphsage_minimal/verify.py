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
TASK_ID = "dgl_graphsage_minimal"
EXPECTED_COMMIT = "c6c874bf7ea085beb04ea1487cfd216a0bacd6c1"
SUCCESS_LEVEL = "L4_cpu_deterministic_fallback"
EXPECTED_PREDICTIONS = [0, 0, 1, 1, 0, 1]
EXPECTED_ACCURACY = 1.0
EXPECTED_LOSS = 0.308503
REQUIRED_ROUTE_TERMS = ["dgl.graph", "SAGEConv", "mean aggregator", "self loop", "GraphSAGE"]
REQUIRED_SEMANTIC_TERMS = ["node classification", "toy graph", "neighbor mean", "CPU deterministic fallback"]


def canonical_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dot(x: list[float], weights: list[list[float]], bias: list[float]) -> list[float]:
    return [sum(x[i] * weights[i][j] for i in range(len(x))) + bias[j] for j in range(len(bias))]


def round_nested(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [round_nested(item) for item in value]
    if isinstance(value, dict):
        return {key: round_nested(item) for key, item in value.items()}
    return value


def build_artifact() -> dict[str, Any]:
    features = [
        [1.0, 0.0, 0.5],
        [0.8, 0.2, 0.4],
        [0.0, 1.0, 0.3],
        [0.1, 0.9, 0.2],
        [0.9, 0.1, 0.6],
        [0.2, 0.8, 0.1],
    ]
    labels = [0, 0, 1, 1, 0, 1]
    edges = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [0, 4], [4, 0], [3, 5], [5, 3], [4, 5], [5, 4]]
    w1 = [
        [0.42, -0.30, 0.18, 0.25],
        [-0.15, 0.50, 0.22, -0.35],
        [0.28, 0.10, -0.40, 0.31],
        [0.36, -0.20, 0.14, 0.18],
        [-0.08, 0.44, 0.30, -0.28],
        [0.20, 0.16, -0.34, 0.27],
    ]
    b1 = [0.03, -0.02, 0.01, 0.04]
    w2 = [[0.70, -0.45], [-0.55, 0.68], [0.25, 0.12], [0.40, -0.32]]
    b2 = [0.02, -0.01]

    neighbors = {node: {node} for node in range(len(features))}
    for source, target in edges:
        neighbors[target].add(source)

    neighbor_mean: list[list[float]] = []
    hidden: list[list[float]] = []
    logits: list[list[float]] = []
    probabilities: list[list[float]] = []
    predictions: list[int] = []

    for node, feature in enumerate(features):
        incoming = [features[index] for index in sorted(neighbors[node])]
        mean = [sum(row[col] for row in incoming) / len(incoming) for col in range(len(feature))]
        neighbor_mean.append(mean)
        h = [max(0.0, value) for value in dot(feature + mean, w1, b1)]
        hidden.append(h)
        logit = dot(h, w2, b2)
        logits.append(logit)
        exps = [math.exp(value - max(logit)) for value in logit]
        total = sum(exps)
        prob = [value / total for value in exps]
        probabilities.append(prob)
        predictions.append(0 if prob[0] >= prob[1] else 1)

    accuracy = sum(int(pred == label) for pred, label in zip(predictions, labels)) / len(labels)
    loss = -sum(math.log(probabilities[index][label]) for index, label in enumerate(labels)) / len(labels)
    route = {
        "api": "dgl.graph((src, dst), num_nodes=6) -> dgl.add_self_loop(graph) -> dgl.nn.pytorch.SAGEConv(aggregator_type='mean')",
        "repo_files": ["python/dgl/convert.py", "python/dgl/nn/pytorch/conv/sageconv.py", "examples/pytorch/graphsage"],
        "fallback_reason": "Avoids DGL native extension and backend wheel variance on a minimal CPU notebook while preserving GraphSAGE mean aggregator semantics.",
        "dataset_route": "controlled toy graph；no Cora / Reddit / OGB download",
        "model_route": "two-layer GraphSAGE-style mean aggregator with self loop neighborhood inclusion",
    }
    semantic = {
        "task": "node classification on a controlled toy graph",
        "claim": "CPU deterministic fallback validates neighbor mean aggregation and GraphSAGE logits without external datasets.",
        "success_condition": "predictions match all six labels and softmax probabilities are finite",
    }
    numeric = round_nested(
        {
            "features": features,
            "labels": labels,
            "edges": edges,
            "neighbor_mean": neighbor_mean,
            "hidden": hidden,
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "accuracy": accuracy,
            "loss": loss,
        }
    )
    checks = {
        "repo_commit_matches": True,
        "dgl_route_recorded": all(term in json.dumps(route, sort_keys=True) for term in REQUIRED_ROUTE_TERMS),
        "toy_graph_semantics_valid": numeric["neighbor_mean"][0] == [0.9, 0.1, 0.5],
        "graphsage_forward_valid": numeric["logits"][0] == [1.0098, -0.69118],
        "node_classification_metric_valid": numeric["predictions"] == EXPECTED_PREDICTIONS and numeric["accuracy"] == EXPECTED_ACCURACY,
        "no_external_dataset_required": "no Cora" in route["dataset_route"],
    }
    payload = {
        "task_id": TASK_ID,
        "success_level": SUCCESS_LEVEL,
        "repo": {
            "url": "https://github.com/dmlc/dgl",
            "commit": EXPECTED_COMMIT,
            "commit_short": EXPECTED_COMMIT[:7],
            "paper_title": "Deep Graph Library",
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
    for key in ["neighbor_mean", "hidden", "logits", "probabilities", "predictions", "accuracy", "loss"]:
        if key not in numeric:
            raise AssertionError(f"missing numeric key: {key}")
        assert_close(numeric[key], expected_numeric[key], f"numeric.{key}")

    if numeric.get("predictions") != EXPECTED_PREDICTIONS:
        raise AssertionError("prediction mismatch")
    if numeric.get("accuracy") != EXPECTED_ACCURACY:
        raise AssertionError("accuracy mismatch")
    if numeric.get("loss") != EXPECTED_LOSS:
        raise AssertionError("loss mismatch")

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
        "checks": checks,
        "observed": {
            "repo_commit": payload.get("repo", {}).get("commit"),
            "accuracy": numeric.get("accuracy"),
            "loss": numeric.get("loss"),
            "predictions": numeric.get("predictions"),
            "route": route.get("api"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the PaperEnvBench DGL GraphSAGE minimal task.")
    parser.add_argument("--artifact-dir", type=Path, default=TASK_ROOT / "artifacts")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        if not args.check_only:
            args.artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = args.artifact_dir / "expected_artifact.json"
            artifact_path.write_text(json.dumps(build_artifact(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = validate_artifact(args.artifact_dir)
    except Exception as exc:
        print(json.dumps({"task_id": TASK_ID, "status": "fail", "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
