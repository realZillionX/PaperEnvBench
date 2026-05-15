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
TASK_ID = "pyg_gcn_minimal"
EXPECTED_COMMIT = "a5b69c37a05561ebb92931b3d586d664a7269585"
SUCCESS_LEVEL = "L4_cpu_deterministic_fallback"
EXPECTED_PREDICTIONS = [0, 0, 0, 1, 1, 1]
REQUIRED_ROUTE_TERMS = [
    "torch_geometric.data.Data",
    "torch_geometric.nn.GCNConv",
    "gcn_norm",
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
    require("GCNConv.forward" in str(route.get("entrypoint")), "GCNConv forward entrypoint is not recorded")

    toy_graph = payload.get("toy_graph", {})
    features = toy_graph.get("features")
    edge_index = toy_graph.get("edge_index")
    labels = toy_graph.get("labels")
    train_mask = toy_graph.get("train_mask")
    require(isinstance(features, list) and len(features) == 6, "expected six feature rows")
    require(all(isinstance(row, list) and len(row) == 3 for row in features), "expected 3D node features")
    require(isinstance(edge_index, list) and len(edge_index) == 2, "edge_index must have two rows")
    require(len(edge_index[0]) == len(edge_index[1]) == 10, "expected ten directed edges")
    require(labels == EXPECTED_PREDICTIONS, f"unexpected labels: {labels}")
    require(train_mask == [True, True, False, True, True, False], f"unexpected train mask: {train_mask}")

    metrics = payload.get("metrics", {})
    require(int(metrics.get("epochs", 0)) >= 1, "missing epoch count")
    require(float(metrics.get("final_train_accuracy", -1.0)) >= 1.0, "train accuracy below threshold")
    require(float(metrics.get("final_full_accuracy", -1.0)) >= 5.0 / 6.0, "full accuracy below threshold")
    losses = metrics.get("loss_trace")
    require(isinstance(losses, list) and len(losses) >= 2, "missing loss trace")
    require(all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in losses), "loss trace must be finite")
    require(float(losses[-1]) < float(losses[0]), "loss trace must decrease")

    numeric = payload.get("numeric", {})
    predictions = numeric.get("predictions")
    require(predictions == EXPECTED_PREDICTIONS, f"unexpected predictions: {predictions}")
    logits = numeric.get("final_logits")
    require(isinstance(logits, list) and len(logits) == 6, "expected six logits rows")
    for row in logits:
        require(isinstance(row, list) and len(row) == 2, "each logits row must have two classes")
        require(all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in row), "logits must be finite")

    semantic = payload.get("semantic", {})
    semantic_text = " ".join(str(value) for value in semantic.values())
    require("node classification" in semantic_text, "semantic summary must mention node classification")
    require("toy graph" in semantic_text, "semantic summary must mention toy graph")

    checks = payload.get("checks", {})
    require(isinstance(checks, dict) and checks, "missing checks")
    require(all(value is True for value in checks.values()), f"failing checks: {checks}")

    return {
        "artifact_dir": str(artifact_dir.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "checks": checks,
        "observed": {
            "full_accuracy": metrics.get("final_full_accuracy"),
            "repo_commit": payload.get("repo", {}).get("commit"),
            "route_sha256": canonical_sha256(route),
            "toy_graph_sha256": canonical_sha256(toy_graph),
            "train_accuracy": metrics.get("final_train_accuracy"),
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
