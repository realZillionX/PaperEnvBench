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

ENVIRONMENT_DEPENDENCY_TRIGGER_TAGS = {"torch_cuda_matrix", "hardware_pressure", "native_extension_build"}
REQUIRED_RUNTIME_TARGET = "internet_accelerator_4090_cuda128"
REQUIRED_L4_ENVIRONMENT_GATE = {
    "requires_profile_closure_pass",
    "requires_gpu_cuda_evidence",
    "requires_minimal_reproduction_experiment",
    "disallow_blocked_or_deferred_dependencies",
}


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_command(command: list[str]) -> None:
    subprocess.check_call(command, stdout=subprocess.DEVNULL)


def verify_cli_contract(task_root: Path, task_id: str) -> None:
    text = (task_root / "verify.py").read_text(encoding="utf-8", errors="replace")
    missing = [flag for flag in ["--check-only", "--json"] if flag not in text]
    if missing:
        raise SystemExit(f"{task_id}: verify.py missing CLI flags: {missing}")


def run_check_only(task_root: Path, task_id: str) -> None:
    subprocess.check_call(
        [sys.executable, "verify.py", "--check-only", "--json"],
        cwd=task_root,
        stdout=subprocess.DEVNULL,
    )


def validate_registry(root: Path) -> dict[str, Any]:
    taxonomy = load_yaml(root / "paperenvbench/taxonomy.yaml")
    registry = load_yaml(root / "paperenvbench/registries/task_registry.yaml")
    environment_registry = load_yaml(root / "paperenvbench/registries/environment_dependency_registry.yaml")
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

    errors.extend(validate_environment_dependency_registry(tasks, environment_registry))

    if errors:
        raise SystemExit("\n".join(errors))

    return registry


def validate_environment_dependency_registry(tasks: list[dict[str, Any]], environment_registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(environment_registry, dict):
        return ["environment_dependency_registry.yaml: expected mapping"]

    runtime_targets = environment_registry.get("runtime_targets", {})
    probe_profiles = environment_registry.get("probe_profiles", {})
    task_bindings = environment_registry.get("task_bindings", [])
    policy = environment_registry.get("policy", {})
    if not runtime_targets:
        errors.append("environment_dependency_registry.yaml: missing runtime_targets")
    if not probe_profiles:
        errors.append("environment_dependency_registry.yaml: missing probe_profiles")
    if not isinstance(policy, dict) or not policy.get("l4_environment_contract"):
        errors.append("environment_dependency_registry.yaml: missing policy.l4_environment_contract")
    if int(policy.get("gpu_utilization_floor_percent", 0) or 0) < 15:
        errors.append("environment_dependency_registry.yaml: gpu_utilization_floor_percent must be at least 15")
    required_report_sections = set(str(item) for item in policy.get("environment_report_required_sections", []) or [])
    expected_report_sections = {
        "runtime",
        "python_packages",
        "dependency_profiles",
        "route_boundary",
        "dependency_inventory",
        "heavyweight_dependency_decisions",
        "verification",
        "validation_experiments",
    }
    missing_report_sections = sorted(expected_report_sections - required_report_sections)
    if missing_report_sections:
        errors.append(f"environment_dependency_registry.yaml: missing environment_report_required_sections {missing_report_sections}")
    dependency_axes = set(str(item) for item in policy.get("dependency_axes", []) or [])
    expected_axes = {
        "python_packages",
        "system_packages",
        "accelerator_runtime",
        "native_extensions",
        "checkpoints_and_assets",
        "full_dependency_inventory",
        "heavyweight_dependency_decisions",
        "blocked_dependency_evidence",
        "key_experiment_smokes",
    }
    missing_axes = sorted(expected_axes - dependency_axes)
    if missing_axes:
        errors.append(f"environment_dependency_registry.yaml: missing dependency_axes {missing_axes}")
    if "gpu_occupancy_guard" not in probe_profiles:
        errors.append("environment_dependency_registry.yaml: missing gpu_occupancy_guard profile")
    else:
        command_text = " ".join(str(item) for item in probe_profiles["gpu_occupancy_guard"].get("command", []))
        if "gpu_occupancy_guard.py" not in command_text or "--min-utilization" not in command_text:
            errors.append("environment_dependency_registry.yaml: gpu_occupancy_guard command must run gpu_occupancy_guard.py with --min-utilization")
    torch_profile = probe_profiles.get("torch_vision_audio_cuda_matrix", {})
    if "gpu_occupancy_guard" not in [str(item) for item in torch_profile.get("depends_on", []) or []]:
        errors.append("environment_dependency_registry.yaml: torch_vision_audio_cuda_matrix must depend on gpu_occupancy_guard")
    if not isinstance(task_bindings, list) or not task_bindings:
        errors.append("environment_dependency_registry.yaml: missing task_bindings")
        return errors

    task_ids = {str(task["task_id"]) for task in tasks}
    required_bound_tasks = {
        str(task["task_id"])
        for task in tasks
        if set(task["taxonomy"]["environment_challenges"]) & ENVIRONMENT_DEPENDENCY_TRIGGER_TAGS
    }
    covered: set[str] = set()
    duplicate_cover: list[str] = []
    profile_ids = set(probe_profiles)
    profile_closures: dict[str, list[str]] = {}

    def visit_profile(profile_id: str, resolved: list[str], visiting: set[str]) -> None:
        if profile_id in resolved or profile_id not in probe_profiles:
            return
        if profile_id in visiting:
            errors.append(f"environment_dependency_registry.yaml:{profile_id}: dependency cycle")
            return
        visiting.add(profile_id)
        for dep in probe_profiles[profile_id].get("depends_on", []) or []:
            visit_profile(str(dep), resolved, visiting)
        visiting.remove(profile_id)
        resolved.append(profile_id)

    def closure(refs: list[str]) -> list[str]:
        key = ",".join(sorted(str(ref) for ref in refs))
        if key not in profile_closures:
            resolved: list[str] = []
            for ref in sorted(str(item) for item in refs):
                visit_profile(ref, resolved, set())
            profile_closures[key] = resolved
        return profile_closures[key]

    for binding in task_bindings:
        group = binding.get("group", "<missing group>")
        ids = binding.get("task_ids", [])
        refs = binding.get("profile_refs", [])
        if not isinstance(ids, list) or not ids:
            errors.append(f"environment_dependency_registry.yaml:{group}: task_ids must be a nonempty list")
            continue
        if not isinstance(refs, list) or not refs:
            errors.append(f"environment_dependency_registry.yaml:{group}: profile_refs must be a nonempty list")
        unknown_refs = sorted(set(str(ref) for ref in refs) - profile_ids)
        if unknown_refs:
            errors.append(f"environment_dependency_registry.yaml:{group}: unknown profile_refs {unknown_refs}")
        expanded_refs = set(closure([str(ref) for ref in refs]))
        for required_profile in ("accelerator_runtime_base", "gpu_occupancy_guard"):
            if required_profile not in expanded_refs:
                errors.append(f"environment_dependency_registry.yaml:{group}: profile closure must include {required_profile}")
        for task_id in ids:
            task_id = str(task_id)
            if task_id not in task_ids:
                errors.append(f"environment_dependency_registry.yaml:{group}: unknown task_id {task_id}")
                continue
            if task_id in covered:
                duplicate_cover.append(task_id)
            covered.add(task_id)

    for profile_id, profile in probe_profiles.items():
        target = profile.get("runtime_target")
        if target and target not in runtime_targets:
            errors.append(f"environment_dependency_registry.yaml:{profile_id}: unknown runtime_target {target}")
        evidence_axes = profile.get("evidence_axes")
        if not isinstance(evidence_axes, list) or not evidence_axes:
            errors.append(f"environment_dependency_registry.yaml:{profile_id}: evidence_axes must be a nonempty list")
        experiments = profile.get("key_validation_experiments")
        if not isinstance(experiments, list) or not experiments:
            errors.append(f"environment_dependency_registry.yaml:{profile_id}: key_validation_experiments must be a nonempty list")
        command = profile.get("command")
        if not isinstance(command, list) or not command:
            errors.append(f"environment_dependency_registry.yaml:{profile_id}: command must be a nonempty list")
        depends_on = profile.get("depends_on", [])
        if depends_on:
            unknown_deps = sorted(set(str(dep) for dep in depends_on) - profile_ids)
            if unknown_deps:
                errors.append(f"environment_dependency_registry.yaml:{profile_id}: unknown depends_on {unknown_deps}")

    missing_all = sorted(task_ids - covered)
    if missing_all:
        errors.append(f"environment_dependency_registry.yaml: missing dependency bindings for all-task coverage {missing_all}")
    missing = sorted(required_bound_tasks - covered)
    if missing:
        errors.append(f"environment_dependency_registry.yaml: missing dependency bindings for {missing}")
    if duplicate_cover:
        errors.append(f"environment_dependency_registry.yaml: duplicate task bindings for {sorted(set(duplicate_cover))}")

    return errors


def task_bound_profiles(environment_registry: dict[str, Any], task_id: str) -> list[str]:
    profiles = environment_registry.get("probe_profiles", {})
    refs: set[str] = set()
    for binding in environment_registry.get("task_bindings", []) or []:
        if task_id in {str(item) for item in binding.get("task_ids", []) or []}:
            refs.update(str(item) for item in binding.get("profile_refs", []) or [])
    resolved: list[str] = []
    visiting: set[str] = set()

    def visit(profile_id: str) -> None:
        if profile_id in resolved or profile_id not in profiles:
            return
        if profile_id in visiting:
            return
        visiting.add(profile_id)
        for dep in profiles[profile_id].get("depends_on", []) or []:
            visit(str(dep))
        visiting.remove(profile_id)
        resolved.append(profile_id)

    for ref in sorted(refs):
        visit(ref)
    return resolved


def validate_task(root: Path, task_id: str, run_verifier: bool = False) -> None:
    task_root = root / "paperenvbench/tasks" / task_id
    if not task_root.exists():
        raise SystemExit(f"{task_id}: task directory does not exist")

    missing = [rel for rel in REQUIRED_TASK_FILES if not (task_root / rel).exists()]
    if missing:
        raise SystemExit(f"{task_id}: missing files: {missing}")

    meta = load_yaml(task_root / "meta.yaml") or {}
    scoring = load_yaml(task_root / "scoring.yaml") or {}
    for rel in ["taxonomy.yaml", "assets_manifest.yaml", "failure_tags.yaml"]:
        load_yaml(task_root / rel)
    for rel in ["repo_snapshot.json", "expected_output.json", "artifacts/expected_artifact.json"]:
        load_json(task_root / rel)

    artifacts = [path for path in (task_root / "artifacts").iterdir() if path.is_file()]
    if not artifacts:
        raise SystemExit(f"{task_id}: artifacts directory is empty")

    check_command(["bash", "-n", str(task_root / "gold_install.sh")])
    check_command([sys.executable, "-m", "py_compile", str(task_root / "verify.py")])
    verify_cli_contract(task_root, task_id)
    environment_registry = load_yaml(root / "paperenvbench/registries/environment_dependency_registry.yaml")
    expected_profiles = task_bound_profiles(environment_registry, task_id)
    if meta.get("runtime_target") != REQUIRED_RUNTIME_TARGET:
        raise SystemExit(f"{task_id}: meta.yaml must set runtime_target: {REQUIRED_RUNTIME_TARGET}")
    contract = meta.get("environment_contract")
    if not isinstance(contract, dict):
        raise SystemExit(f"{task_id}: meta.yaml missing environment_contract")
    if contract.get("runtime_target") != REQUIRED_RUNTIME_TARGET:
        raise SystemExit(f"{task_id}: environment_contract.runtime_target mismatch")
    if set(contract.get("dependency_profiles", []) or []) != set(expected_profiles):
        raise SystemExit(f"{task_id}: environment_contract.dependency_profiles must match environment registry closure")
    gpu_contract = contract.get("gpu")
    if not isinstance(gpu_contract, dict) or gpu_contract.get("required_for_standard_reproduction") is not True:
        raise SystemExit(f"{task_id}: environment_contract.gpu.required_for_standard_reproduction must be true")
    if int(gpu_contract.get("gpu_occupancy_floor_percent", 0) or 0) < 15:
        raise SystemExit(f"{task_id}: environment_contract.gpu.gpu_occupancy_floor_percent must be at least 15")
    l4_requires = set(contract.get("l4_requires", []) or [])
    missing_l4 = REQUIRED_L4_ENVIRONMENT_GATE - l4_requires
    if missing_l4:
        raise SystemExit(f"{task_id}: environment_contract.l4_requires missing {sorted(missing_l4)}")
    l4_gate = scoring.get("l4_environment_gate") if isinstance(scoring, dict) else None
    if not isinstance(l4_gate, dict):
        raise SystemExit(f"{task_id}: scoring.yaml missing l4_environment_gate")
    for key in REQUIRED_L4_ENVIRONMENT_GATE:
        if l4_gate.get(key) is not True:
            raise SystemExit(f"{task_id}: scoring.yaml l4_environment_gate.{key} must be true")
    if l4_gate.get("runtime_target") != REQUIRED_RUNTIME_TARGET:
        raise SystemExit(f"{task_id}: scoring.yaml l4_environment_gate.runtime_target mismatch")
    if run_verifier:
        run_check_only(task_root, task_id)


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
    parser.add_argument("--run-verifiers", action="store_true", help="Run every selected verify.py --check-only --json.")
    args = parser.parse_args()

    root = args.root.resolve()
    registry = validate_registry(root)
    task_ids = resolve_task_ids(root, registry, args.task)
    for task_id in task_ids:
        validate_task(root, task_id, run_verifier=args.run_verifiers)

    print(f"ok registry=50 tasks={len(task_ids)}")


if __name__ == "__main__":
    main()
