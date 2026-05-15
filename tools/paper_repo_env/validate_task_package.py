from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


REQUIRED_TASK_FILES = [
    "meta.yaml",
    "taxonomy.yaml",
    "assets_manifest.yaml",
    "failure_tags.yaml",
    "scoring.yaml",
    "README_eval.md",
    "repo_snapshot.json",
    "gold_install.sh",
    "verify.py",
    "expected_output.json",
    "requirements_lock.txt",
    "logs/gold_install.log",
    "logs/gold_verify.log",
    "artifacts/expected_artifact.json",
]

EXPECTED_SPLITS = Counter({"dev": 20, "val": 10, "test": 20})
EXPECTED_MODALITIES = Counter(
    {
        "audio_speech": 5,
        "image_classification_representation": 5,
        "object_detection_segmentation": 6,
        "video_understanding_generation": 5,
        "vision_language_multimodal": 6,
        "llm_nlp": 6,
        "diffusion_generation": 5,
        "graph_learning_recommender": 4,
        "reinforcement_learning_simulation": 4,
        "scientific_geometry_3d": 4,
    }
)


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_command(command: list[str]) -> None:
    subprocess.check_call(command, stdout=subprocess.DEVNULL)


def validate_registry(root: Path) -> dict[str, Any]:
    taxonomy = load_yaml(root / "paperenvbench/taxonomy.yaml")
    registry = load_yaml(root / "paperenvbench/registries/task_registry.yaml")
    tasks = registry["tasks"]

    errors: list[str] = []
    if len(tasks) != 50:
        errors.append(f"expected 50 tasks, found {len(tasks)}")

    split_counts = Counter(task["split"] for task in tasks)
    if split_counts != EXPECTED_SPLITS:
        errors.append(f"split counts mismatch: {dict(split_counts)}")

    modality_counts = Counter(task["taxonomy"]["modality_primary"] for task in tasks)
    if modality_counts != EXPECTED_MODALITIES:
        errors.append(f"modality counts mismatch: {dict(modality_counts)}")

    allowed_modalities = set(taxonomy["modality_axis"])
    allowed_failures = set(taxonomy["failure_mechanism_axis"])
    allowed_verifiers = set(taxonomy["verification_entry_axis"])

    for task in tasks:
        task_id = task["task_id"]
        modality = task["taxonomy"]["modality_primary"]
        failures = set(task["taxonomy"]["environment_challenges"])
        verifiers = set(task["taxonomy"]["verification_type"])
        if modality not in allowed_modalities:
            errors.append(f"{task_id}: unknown modality {modality}")
        if failures - allowed_failures:
            errors.append(f"{task_id}: unknown failure tags {sorted(failures - allowed_failures)}")
        if verifiers - allowed_verifiers:
            errors.append(f"{task_id}: unknown verifier tags {sorted(verifiers - allowed_verifiers)}")

    if errors:
        raise SystemExit("\n".join(errors))

    return registry


def validate_task(root: Path, task_id: str) -> None:
    task_root = root / "paperenvbench/tasks" / task_id
    if not task_root.exists():
        raise SystemExit(f"{task_id}: task directory does not exist")

    missing = [rel for rel in REQUIRED_TASK_FILES if not (task_root / rel).exists()]
    if missing:
        raise SystemExit(f"{task_id}: missing files: {missing}")

    for rel in ["meta.yaml", "taxonomy.yaml", "assets_manifest.yaml", "failure_tags.yaml", "scoring.yaml"]:
        load_yaml(task_root / rel)
    for rel in ["repo_snapshot.json", "expected_output.json", "artifacts/expected_artifact.json"]:
        load_json(task_root / rel)

    artifacts = [path for path in (task_root / "artifacts").iterdir() if path.is_file()]
    if not artifacts:
        raise SystemExit(f"{task_id}: artifacts directory is empty")

    check_command(["bash", "-n", str(task_root / "gold_install.sh")])
    check_command([sys.executable, "-m", "py_compile", str(task_root / "verify.py")])


def resolve_task_ids(root: Path, registry: dict[str, Any], requested: list[str]) -> list[str]:
    if requested:
        return requested
    return [
        task["task_id"]
        for task in registry["tasks"]
        if task.get("gold_ready") and task.get("verifier_ready") and task.get("asset_ready")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PaperEnvBench registry and gold task package structure.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--task", action="append", default=[], help="Task id to validate. Defaults to all ready tasks.")
    args = parser.parse_args()

    root = args.root.resolve()
    registry = validate_registry(root)
    task_ids = resolve_task_ids(root, registry, args.task)
    for task_id in task_ids:
        validate_task(root, task_id)

    print(f"ok registry=50 tasks={len(task_ids)}")


if __name__ == "__main__":
    main()
