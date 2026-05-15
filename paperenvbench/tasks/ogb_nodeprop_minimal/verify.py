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
TASK_ID = "ogb_nodeprop_minimal"
EXPECTED_COMMIT = "61e9784ca76edeaa6e259ba0f836099608ff0586"
SUCCESS_LEVEL = "L4_cpu_deterministic_fallback"
EXPECTED_DATASET = "ogbn-arxiv"
EXPECTED_ACCURACY = 0.8
MINIMUM_ACCURACY = 0.75
REQUIRED_ROUTE_TERMS = ["ogb.nodeproppred", "Evaluator", "ogbn-arxiv", "y_true", "y_pred"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_shape(matrix: Any, rows: int, cols: int, name: str) -> None:
    require(isinstance(matrix, list) and len(matrix) == rows, f"{name} row count mismatch")
    for row in matrix:
        require(isinstance(row, list) and len(row) == cols, f"{name} column count mismatch")


def compute_accuracy(labels: list[list[int]], predictions: list[list[int]]) -> float:
    require(len(labels) == len(predictions), "label / prediction length mismatch")
    total = len(labels)
    require(total > 0, "empty labels")
    correct = 0
    for truth, pred in zip(labels, predictions):
        require(len(truth) == 1 and len(pred) == 1, "expected one node-label task")
        if truth[0] == pred[0]:
            correct += 1
    return correct / total


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
    require("ogb/nodeproppred/evaluate.py" in route_text, "Evaluator source path is not recorded")
    require("PygNodePropPredDataset" in route_text, "optional public dataset route is not recorded")

    graph = payload.get("toy_graph", {})
    num_nodes = int(graph.get("num_nodes", -1))
    feature_dim = int(graph.get("feature_dim", -1))
    edges = graph.get("edges", [])
    features = graph.get("node_features", [])
    labels = graph.get("labels", [])
    predictions = graph.get("predictions", [])
    require(num_nodes == 10, "unexpected toy graph node count")
    require(len(edges) == int(graph.get("num_edges", -1)) == 12, "unexpected toy graph edge count")
    validate_shape(features, num_nodes, feature_dim, "node_features")
    validate_shape(labels, num_nodes, 1, "labels")
    validate_shape(predictions, num_nodes, 1, "predictions")
    split = graph.get("split", {})
    require(split.get("train") == [0, 1, 2, 3, 4, 5], "train split mismatch")
    require(split.get("valid") == [6, 7], "valid split mismatch")
    require(split.get("test") == [8, 9], "test split mismatch")

    metric = payload.get("metric", {})
    require(metric.get("dataset_name") == EXPECTED_DATASET, "dataset name mismatch")
    require(metric.get("evaluator") == "ogb.nodeproppred.Evaluator", "wrong evaluator")
    require(metric.get("metric_name") == "acc", "wrong metric")
    require(metric.get("input_format", {}).get("y_true_shape") == [10, 1], "wrong y_true shape")
    require(metric.get("input_format", {}).get("y_pred_shape") == [10, 1], "wrong y_pred shape")
    accuracy = compute_accuracy(labels, predictions)
    require(math.isclose(accuracy, EXPECTED_ACCURACY, abs_tol=1e-12), f"computed accuracy mismatch: {accuracy}")
    require(math.isclose(float(metric.get("accuracy")), EXPECTED_ACCURACY, abs_tol=1e-12), "recorded accuracy mismatch")
    require(float(metric.get("minimum_accuracy")) == MINIMUM_ACCURACY, "minimum accuracy mismatch")
    require(accuracy >= MINIMUM_ACCURACY, "accuracy below minimum")
    require(metric.get("correct_count") == 8 and metric.get("total_count") == 10, "correct / total mismatch")

    semantic = payload.get("semantic", {})
    require("node property prediction" in json.dumps(semantic, sort_keys=True), "semantic task not recorded")
    require(semantic.get("dataset_download_required") is False, "check-only must not require dataset download")

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
            "dataset_name": metric.get("dataset_name"),
            "metric": metric.get("metric_name"),
            "accuracy": accuracy,
            "num_nodes": num_nodes,
            "num_edges": len(edges),
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
