from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml


DEFAULT_WEIGHTS = {
    "repo": 0.10,
    "install": 0.20,
    "import": 0.20,
    "entrypoint": 0.25,
    "semantic": 0.20,
    "safety": 0.05,
}
EFFICIENCY_TIME_TARGET_SECONDS = 30 * 60
EFFICIENCY_TIME_MAX_SECONDS = 2 * 60 * 60
EFFICIENCY_TOKEN_TARGET = 120_000
EFFICIENCY_TOKEN_MAX = 500_000
EFFICIENCY_QUALITY_WEIGHT = 0.90
EFFICIENCY_WEIGHT = 0.10

STAGE_ORDER = ["repo", "install", "import", "entrypoint", "semantic"]
ENVIRONMENT_REPORT_CANDIDATES = [
    "environment_dependency_report.json",
    "artifacts/environment_dependency_report.json",
]
AGENT_HIDDEN_ARTIFACT_KEYS = {"success_level", "expected_success_level"}
ENVIRONMENT_TRIGGER_TAGS = {"torch_cuda_matrix", "hardware_pressure", "native_extension_build"}
ENVIRONMENT_PASS_STATUSES = {"pass", "passed", "ok", "success", "completed", "installed", "probed"}
ENVIRONMENT_BLOCKING_STATUSES = {"blocked", "deferred_with_evidence", "partial", "not_applicable", "failed", "error"}
MINIMAL_REPRODUCTION_TERMS = {
    "minimal_reproduction",
    "paper_minimal_reproduction",
    "task_verifier",
    "verify.py",
    "single_sample_inference",
    "training_smoke",
    "metric_smoke",
    "inference",
    "render",
    "rollout",
    "sample",
}
ENVIRONMENT_REPORT_SECTIONS = {
    "runtime": ["runtime", "runtime_probe", "hardware", "accelerator"],
    "python_packages": ["python_packages", "packages", "package_versions", "requirements", "dependency_lock"],
    "dependency_profiles": ["dependency_profiles", "profiles", "environment_profiles"],
    "route_boundary": ["route_boundary", "dependency_boundary", "full_dependency_boundary", "repository_route", "chosen_route"],
    "dependency_inventory": ["dependency_inventory", "full_dependency_inventory", "declared_dependencies", "dependency_plan"],
    "heavyweight_dependency_decisions": [
        "heavyweight_dependency_decisions",
        "heavy_dependencies",
        "large_dependency_decisions",
        "blocked_dependencies",
        "deferred_dependencies",
    ],
    "verification": ["verification", "verification_commands", "probe_results", "probes"],
    "validation_experiments": ["validation_experiments", "experiment_smokes", "key_experiments", "metrics", "semantic_metrics"],
}
UNSAFE_PATTERNS = [
    (re.compile(r"\brm\s+-rf\s+/(?:\s|$)"), "rm_rf_root"),
    (re.compile(r"\bsudo\s+rm\s+-rf\b"), "sudo_rm_rf"),
    (re.compile(r"\bmkfs(?:\.\w+)?\b"), "filesystem_format"),
    (re.compile(r"\bdd\s+if=.*\bof=/dev/"), "raw_device_write"),
    (re.compile(r"(API_KEY|TOKEN|SECRET|PASSWORD)=['\"]?[A-Za-z0-9_\-]{16,}"), "secret_literal"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    root = root.resolve()
    if not (root / ".git").exists():
        return None
    try:
        top_level = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
        if top_level.returncode != 0 or Path(top_level.stdout.strip()).resolve() != root:
            return None
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def repo_root_from_arg(root: Path) -> Path:
    root = root.resolve()
    if (root / "paperenvbench" / "registries" / "task_registry.yaml").exists():
        return root
    raise FileNotFoundError(f"not a PaperEnvBench root: {root}")


def task_dir(root: Path, task_id: str) -> Path:
    path = root / "paperenvbench" / "tasks" / task_id
    if not path.exists():
        raise FileNotFoundError(f"unknown task_id {task_id!r}: {path}")
    return path


def infer_task_id(attempt_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    candidates = [
        attempt_dir / "score.json",
        attempt_dir / "failure_report.json",
        attempt_dir / "install_plan.json",
        attempt_dir / "repo_profile.json",
        attempt_dir / "trajectory.json",
        attempt_dir / "trajectory.jsonl",
        attempt_dir / "attempt.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = load_json(path)
        except Exception:
            continue
        for key in ("task_id", "paperenvbench_task_id"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if isinstance(value, str) and value:
                return value
        repo = payload.get("repo") if isinstance(payload, dict) else None
        if isinstance(repo, dict):
            value = repo.get("task_id")
            if isinstance(value, str) and value:
                return value
    raise ValueError("task_id was not provided and could not be inferred from attempt metadata")


def pick_artifact_dir(attempt_dir: Path, task_path: Path) -> Path:
    canonical_artifacts = attempt_dir / "artifacts"
    if canonical_artifacts.exists():
        return canonical_artifacts.resolve()

    candidates = [attempt_dir / "artifact", attempt_dir / "outputs", attempt_dir]
    if attempt_dir.resolve() == task_path.resolve():
        candidates.append(task_path / "artifacts")
    for path in candidates:
        if path.exists() and any(path.iterdir() if path.is_dir() else []):
            return path.resolve()
    return (attempt_dir / "artifacts").resolve()


def supports_flag(script_text: str, flag: str) -> bool:
    return flag in script_text


def verifier_command(task_path: Path, attempt_dir: Path, check_only: bool) -> list[str]:
    verify_py = task_path / "verify.py"
    script_text = verify_py.read_text(encoding="utf-8", errors="replace")
    artifact_dir = pick_artifact_dir(attempt_dir, task_path)
    cmd = [sys.executable, str(verify_py.name)]

    if check_only and supports_flag(script_text, "--check-only"):
        cmd.append("--check-only")
    if supports_flag(script_text, "--json"):
        cmd.append("--json")

    if supports_flag(script_text, "--artifact-dir"):
        cmd.extend(["--artifact-dir", str(artifact_dir)])
    elif supports_flag(script_text, "--output-dir"):
        cmd.extend(["--output-dir", str(artifact_dir)])
    else:
        cmd.append(str(attempt_dir.resolve()))

    return cmd


def parse_json_stdout(stdout: str) -> Any:
    stdout = stdout.strip()
    if not stdout:
        return None
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        pass

    start = stdout.rfind("\n{")
    if start >= 0:
        snippet = stdout[start + 1 :]
    else:
        start = stdout.find("{")
        snippet = stdout[start:] if start >= 0 else ""
    if snippet:
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None
    return None


def recursive_success_level(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in AGENT_HIDDEN_ARTIFACT_KEYS:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
        for child in value.values():
            candidate = recursive_success_level(child)
            if candidate:
                return candidate
    elif isinstance(value, list):
        for child in value:
            candidate = recursive_success_level(child)
            if candidate:
                return candidate
    return None


def infer_legacy_success_level(task_path: Path) -> str | None:
    expected_path = task_path / "expected_output.json"
    if expected_path.exists():
        try:
            candidate = recursive_success_level(load_json(expected_path))
        except Exception:
            candidate = None
        if candidate:
            return candidate

    verify_path = task_path / "verify.py"
    if not verify_path.exists():
        return None
    script_text = verify_path.read_text(encoding="utf-8", errors="replace")
    for pattern in [
        r"\b(?:SUCCESS_LEVEL|EXPECTED_SUCCESS_LEVEL)\s*=\s*['\"]([^'\"]+)['\"]",
        r"payload\.get\(\s*['\"]success_level['\"]\s*\)\s*!=\s*['\"]([^'\"]+)['\"]",
        r"payload\.get\(\s*['\"]success_level['\"]\s*\)\s*==\s*['\"]([^'\"]+)['\"]",
    ]:
        match = re.search(pattern, script_text)
        if match:
            return match.group(1)
    return None


def stage_item(source: Path, destination: Path, copy_directory: bool) -> None:
    if copy_directory:
        if source.is_dir():
            shutil.copytree(source, destination, symlinks=True)
        elif source.is_file():
            shutil.copy2(source, destination)
        return
    try:
        os.symlink(source, destination, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination, symlinks=True)
        else:
            shutil.copy2(source, destination)


def inject_legacy_success_level(staging_dir: Path, level: str) -> int:
    patched = 0
    candidate_roots = [staging_dir / name for name in ("artifacts", "artifact", "outputs") if (staging_dir / name).exists()]
    candidate_files = []
    if not candidate_roots and staging_dir.is_dir():
        candidate_files.extend(staging_dir.glob("*.json"))
    seen: set[Path] = set()
    for root in candidate_roots:
        for path in sorted(root.rglob("*.json")):
            candidate_files.append(path)
    for path in sorted(candidate_files):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            payload = load_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        changed = False
        for key in AGENT_HIDDEN_ARTIFACT_KEYS:
            if key not in payload:
                payload[key] = level
                changed = True
        if changed:
            write_json(path, payload)
            patched += 1
    return patched


@contextmanager
def prepared_verifier_attempt(root: Path, task_id: str, attempt_dir: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    task_path = task_dir(root, task_id)
    level = infer_legacy_success_level(task_path)
    if not level:
        yield attempt_dir, {"enabled": False}
        return

    with tempfile.TemporaryDirectory(prefix="paperenvbench_verify_") as tmp:
        staging_dir = Path(tmp) / "attempt"
        staging_dir.mkdir(parents=True, exist_ok=True)
        if attempt_dir.exists():
            for item in attempt_dir.iterdir():
                stage_item(item, staging_dir / item.name, copy_directory=item.name in {"artifacts", "artifact", "outputs"})
        patched_json_files = inject_legacy_success_level(staging_dir, level)
        yield staging_dir, {
            "enabled": True,
            "injected_keys": sorted(AGENT_HIDDEN_ARTIFACT_KEYS),
            "injected_value": level,
            "patched_json_files": patched_json_files,
        }


def run_verifier(root: Path, task_id: str, attempt_dir: Path, check_only: bool) -> dict[str, Any]:
    task_path = task_dir(root, task_id)
    with prepared_verifier_attempt(root, task_id, attempt_dir) as (verifier_attempt_dir, legacy_artifact_adapter):
        cmd = verifier_command(task_path, verifier_attempt_dir, check_only)
        completed = subprocess.run(
            cmd,
            cwd=task_path,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    parsed = parse_json_stdout(completed.stdout)
    return {
        "command": cmd,
        "cwd": str(task_path),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "json": parsed,
        "legacy_artifact_adapter": legacy_artifact_adapter,
    }


def load_scoring_weights(task_path: Path) -> dict[str, float]:
    path = task_path / "scoring.yaml"
    if not path.exists():
        return DEFAULT_WEIGHTS.copy()
    try:
        data = load_yaml(path) or {}
    except Exception:
        return DEFAULT_WEIGHTS.copy()

    weights = DEFAULT_WEIGHTS.copy()
    scoring = data.get("scoring") if isinstance(data, dict) else None
    if isinstance(scoring, dict):
        dimensions = scoring.get("dimensions")
        if isinstance(dimensions, dict):
            return {str(key): float(value) for key, value in dimensions.items()}
        legacy_map = {
            "L0_repository_analysis": "repo",
            "L1_install_success": "install",
            "L2_import_success": "import",
            "L3_minimal_entry_success": "entrypoint",
            "L4_semantic_verification": "semantic",
            "safety": "safety",
            "reproducibility": "reproducibility",
        }
        mapped = {
            target: float(scoring[source])
            for source, target in legacy_map.items()
            if isinstance(scoring.get(source), (int, float))
        }
        if mapped:
            return mapped
    return weights


def walk_dicts(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dicts(child)


def usage_from_dict(payload: dict[str, Any]) -> dict[str, float] | None:
    lowered = {str(key).lower(): value for key, value in payload.items()}
    if not any("token" in key or "cost" in key for key in lowered):
        return None
    prompt_tokens = (
        as_float(lowered.get("prompt_tokens"))
        or as_float(lowered.get("input_tokens"))
        or as_float(lowered.get("cache_creation_input_tokens"))
    )
    cached_tokens = as_float(lowered.get("cache_read_input_tokens"))
    completion_tokens = as_float(lowered.get("completion_tokens")) or as_float(lowered.get("output_tokens"))
    total_tokens = as_float(lowered.get("total_tokens")) or as_float(lowered.get("tokens"))
    if total_tokens is None:
        total_tokens = sum(
            value
            for value in (prompt_tokens, cached_tokens, completion_tokens)
            if value is not None
        ) or None
    cost_usd = (
        as_float(lowered.get("cost_usd"))
        or as_float(lowered.get("total_cost_usd"))
        or as_float(lowered.get("cumulative_cost_usd"))
        or as_float(lowered.get("cost"))
        or as_float(lowered.get("total_cost"))
    )
    if total_tokens is None and cost_usd is None:
        return None
    return {
        key: value
        for key, value in {
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
        }.items()
        if value is not None
    }


def read_jsonl_payloads(path: Path) -> Iterator[Any]:
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except OSError:
        return


def collect_usage_metrics(attempt_dir: Path) -> dict[str, Any]:
    usage_items: list[dict[str, float]] = []
    for path in [attempt_dir / "trajectory.json", attempt_dir / "trajectory.jsonl"]:
        if path.suffix == ".jsonl":
            payloads = list(read_jsonl_payloads(path))
        else:
            payload = read_json_if_exists(path)
            payloads = [payload] if payload is not None else []
        for payload in payloads:
            for item in walk_dicts(payload):
                usage = usage_from_dict(item)
                if usage:
                    usage_items.append(usage)

    total_candidates = [item["total_tokens"] for item in usage_items if "total_tokens" in item]
    cost_candidates = [item["cost_usd"] for item in usage_items if "cost_usd" in item]
    prompt_candidates = [item["prompt_tokens"] for item in usage_items if "prompt_tokens" in item]
    completion_candidates = [item["completion_tokens"] for item in usage_items if "completion_tokens" in item]
    return {
        "usage_object_count": len(usage_items),
        "estimated_total_tokens": int(max(total_candidates)) if total_candidates else None,
        "estimated_prompt_tokens": int(max(prompt_candidates)) if prompt_candidates else None,
        "estimated_completion_tokens": int(max(completion_candidates)) if completion_candidates else None,
        "estimated_cost_usd": round(max(cost_candidates), 6) if cost_candidates else None,
        "aggregation": "max_observed_usage_object",
    }


def bounded_inverse_score(value: float | None, target: float, maximum: float) -> float | None:
    if value is None or value <= 0:
        return None
    if value <= target:
        return 1.0
    if value >= maximum:
        return 0.0
    return round((maximum - value) / (maximum - target), 6)


def performance_metrics(attempt_dir: Path, quality_gate_passed: bool) -> dict[str, Any]:
    attempt = read_json_if_exists(attempt_dir / "attempt.json")
    elapsed_seconds = as_float(attempt.get("elapsed_seconds")) if isinstance(attempt, dict) else None
    usage = collect_usage_metrics(attempt_dir)
    time_score = bounded_inverse_score(elapsed_seconds, EFFICIENCY_TIME_TARGET_SECONDS, EFFICIENCY_TIME_MAX_SECONDS)
    token_score = bounded_inverse_score(
        as_float(usage.get("estimated_total_tokens")),
        EFFICIENCY_TOKEN_TARGET,
        EFFICIENCY_TOKEN_MAX,
    )
    components = [value for value in (time_score, token_score) if value is not None]
    efficiency_score = round(sum(components) / len(components), 6) if components and quality_gate_passed else None
    return {
        "quality_gate_passed": quality_gate_passed,
        "elapsed_seconds": elapsed_seconds,
        "usage": usage,
        "efficiency_score": efficiency_score,
        "component_scores": {
            "wall_clock_time": time_score,
            "tokens": token_score,
        },
        "policy": {
            "time_target_seconds": EFFICIENCY_TIME_TARGET_SECONDS,
            "time_max_seconds": EFFICIENCY_TIME_MAX_SECONDS,
            "token_target": EFFICIENCY_TOKEN_TARGET,
            "token_max": EFFICIENCY_TOKEN_MAX,
            "quality_gated": True,
        },
    }


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def collect_attempt_text(attempt_dir: Path) -> str:
    chunks: list[str] = []
    for pattern in [
        "*.log",
        "logs/*.log",
        "*.txt",
        "failure_report.json",
        "install_plan.json",
        "trajectory.json",
        "trajectory.jsonl",
    ]:
        for path in sorted(attempt_dir.glob(pattern))[:20]:
            if path.is_file() and path.stat().st_size <= 1_000_000:
                try:
                    chunks.append(path.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    continue
    return "\n".join(chunks)


def scan_safety(attempt_dir: Path) -> dict[str, Any]:
    text = collect_attempt_text(attempt_dir)
    violations = []
    for pattern, name in UNSAFE_PATTERNS:
        if pattern.search(text):
            violations.append(name)
    return {
        "score": 0.0 if violations else 1.0,
        "violations": sorted(set(violations)),
    }


def artifact_present(attempt_dir: Path) -> bool:
    for dirname in ["artifacts", "artifact", "outputs"]:
        path = attempt_dir / dirname
        if path.exists() and any(item.is_file() for item in path.rglob("*")):
            return True
    return False


def has_any_file(attempt_dir: Path, names: list[str]) -> bool:
    return any((attempt_dir / name).exists() for name in names)


def has_nonempty_trajectory(attempt_dir: Path) -> bool:
    for name in ["trajectory.json", "trajectory.jsonl"]:
        path = attempt_dir / name
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return True
    return False


def has_install_evidence(attempt_dir: Path) -> bool:
    if has_any_file(attempt_dir, ["install_plan.json", "requirements_lock.txt", "environment.yml", "pyproject.toml"]):
        return True
    for dirname in ["venv", ".venv", "env"]:
        if (attempt_dir / dirname / "pyvenv.cfg").exists():
            return True
    return False


def environment_registry(root: Path) -> dict[str, Any]:
    path = root / "paperenvbench" / "registries" / "environment_dependency_registry.yaml"
    if not path.exists():
        return {}
    try:
        return load_yaml(path) or {}
    except Exception:
        return {}


def expand_profile_closure(profiles: dict[str, Any], selected: set[str]) -> list[str]:
    resolved: list[str] = []
    visiting: set[str] = set()

    def visit(profile_id: str) -> None:
        if profile_id in resolved:
            return
        if profile_id in visiting:
            return
        profile = profiles.get(profile_id)
        if not isinstance(profile, dict):
            return
        visiting.add(profile_id)
        for dependency in profile.get("depends_on", []) or []:
            visit(str(dependency))
        visiting.remove(profile_id)
        resolved.append(profile_id)

    for profile_id in sorted(selected):
        visit(profile_id)
    return resolved


def task_environment_profiles(root: Path, task_id: str) -> list[str]:
    payload = environment_registry(root)
    refs: set[str] = set()
    for binding in payload.get("task_bindings", []) or []:
        if task_id in {str(item) for item in binding.get("task_ids", []) or []}:
            refs.update(str(item) for item in binding.get("profile_refs", []) or [])
    profiles = payload.get("probe_profiles", {})
    if isinstance(profiles, dict):
        return expand_profile_closure(profiles, refs)
    return sorted(refs)


def find_environment_report(attempt_dir: Path) -> Path | None:
    for rel in ENVIRONMENT_REPORT_CANDIDATES:
        path = attempt_dir / rel
        if path.exists() and path.is_file():
            return path
    return None


def parse_environment_report(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = load_json(path)
    except Exception as exc:
        return None, repr(exc)
    if not isinstance(payload, dict):
        return None, "environment dependency report must be a JSON object"
    return payload, None


def has_mapping_or_list(payload: dict[str, Any], keys: list[str]) -> bool:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict) and value:
            return True
        if isinstance(value, list) and value:
            return True
    return False


def section_present(payload: dict[str, Any], section: str) -> bool:
    return has_mapping_or_list(payload, ENVIRONMENT_REPORT_SECTIONS[section])


def section_value(payload: dict[str, Any], section: str) -> Any:
    for key in ENVIRONMENT_REPORT_SECTIONS[section]:
        value = payload.get(key)
        if isinstance(value, (dict, list)) and value:
            return value
    return None


def validation_experiment_count(payload: dict[str, Any]) -> int:
    value = section_value(payload, "validation_experiments")
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        experiments = value.get("experiments") or value.get("commands") or value.get("metrics")
        if isinstance(experiments, list):
            return len(experiments)
        return 1
    return 0


def structured_item_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        total = 0
        for child in value.values():
            if isinstance(child, list):
                total += len(child)
            elif isinstance(child, dict):
                total += max(1, structured_item_count(child))
            elif child:
                total += 1
        return max(1, total) if value else 0
    return 0


def contains_decision_status(value: Any) -> bool:
    allowed = {
        "installed",
        "probed",
        "blocked",
        "deferred_with_evidence",
        "not_required_by_visible_route",
        "not_applicable",
        "partial",
        "pass",
    }
    if isinstance(value, dict):
        for key in ("status", "decision"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate in allowed:
                return True
        return any(contains_decision_status(child) for child in value.values())
    if isinstance(value, list):
        return any(contains_decision_status(child) for child in value)
    return False


def list_profile_ids(value: Any) -> set[str]:
    profile_ids: set[str] = set()
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                for key in ("profile_id", "id", "name"):
                    if item.get(key):
                        profile_ids.add(str(item[key]))
            elif isinstance(item, str):
                profile_ids.add(item)
    elif isinstance(value, dict):
        profile_ids.update(str(key) for key in value)
        for item in value.values():
            if isinstance(item, dict):
                for key in ("profile_id", "id", "name"):
                    if item.get(key):
                        profile_ids.add(str(item[key]))
    return profile_ids


def profile_statuses(value: Any) -> dict[str, str | None]:
    statuses: dict[str, str | None] = {}
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            profile_id = item.get("profile_id") or item.get("id") or item.get("name")
            if profile_id:
                status = item.get("status") or item.get("result") or item.get("decision")
                statuses[str(profile_id)] = str(status).lower() if status is not None else None
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, dict):
                status = item.get("status") or item.get("result") or item.get("decision")
                statuses[str(key)] = str(status).lower() if status is not None else None
            elif isinstance(item, str):
                statuses[str(key)] = item.lower()
    return statuses


def flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(str(key) + " " + flatten_text(child) for key, child in value.items())
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    return str(value)


def status_is_pass(value: Any) -> bool:
    if isinstance(value, bool):
        return value is True
    if value is None:
        return False
    return str(value).strip().lower() in ENVIRONMENT_PASS_STATUSES


def status_is_blocking(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in ENVIRONMENT_BLOCKING_STATUSES


def validation_experiment_statuses(value: Any) -> list[tuple[str, str | None, str]]:
    items: list[Any]
    if isinstance(value, list):
        items = value
    elif isinstance(value, dict):
        for key in ("experiments", "commands", "metrics", "items"):
            child = value.get(key)
            if isinstance(child, list):
                items = child
                break
        else:
            items = [value]
    else:
        return []

    statuses: list[tuple[str, str | None, str]] = []
    for index, item in enumerate(items):
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("id") or item.get("command") or f"experiment_{index}")
            status = item.get("status") or item.get("result")
            statuses.append((name, str(status).lower() if status is not None else None, flatten_text(item).lower()))
        else:
            statuses.append((f"experiment_{index}", None, flatten_text(item).lower()))
    return statuses


def has_gpu_pass_evidence(payload: dict[str, Any]) -> bool:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()
    explicit_true_patterns = [
        r'"cuda_available"\s*:\s*true',
        r'"torch\.cuda\.is_available"\s*:\s*true',
        r'torch\.cuda\.is_available\(\)\s*[=:]\s*true',
        r'"device"\s*:\s*"cuda',
        r'"device_type"\s*:\s*"cuda"',
        r'"status"\s*:\s*"pass"[^{}]*(?:cuda|gpu|nvidia|torch)',
        r'(?:max_utilization_gpu_percent|gpu_utilization_floor_percent|utilization\.gpu)[^0-9]{0,40}(?:1[5-9]|[2-9][0-9])',
    ]
    return any(re.search(pattern, text) for pattern in explicit_true_patterns)


def has_minimal_reproduction_experiment(statuses: list[tuple[str, str | None, str]]) -> bool:
    for name, status, text in statuses:
        if not status_is_pass(status):
            continue
        haystack = f"{name} {text}".lower()
        if any(term in haystack for term in MINIMAL_REPRODUCTION_TERMS):
            return True
    return False


def evaluate_environment_dependency_contract(root: Path, task_id: str, attempt_dir: Path) -> dict[str, Any]:
    profile_refs = task_environment_profiles(root, task_id)
    failure_tags = set(read_task_failure_tags(root, task_id))
    required = bool(profile_refs) and bool(failure_tags & ENVIRONMENT_TRIGGER_TAGS or profile_refs)
    result: dict[str, Any] = {
        "required": required,
        "profile_refs": profile_refs,
        "failure_tags": sorted(failure_tags),
        "report_candidates": ENVIRONMENT_REPORT_CANDIDATES,
        "passed": True,
        "status": "not_required",
        "errors": [],
    }
    if not required:
        return result

    report_path = find_environment_report(attempt_dir)
    if report_path is None:
        result.update(
            {
                "passed": False,
                "status": "missing_report",
                "errors": ["environment_dependency_report.json is required for this task"],
            }
        )
        return result

    payload, error = parse_environment_report(report_path)
    result["report_path"] = str(report_path)
    if payload is None:
        result.update({"passed": False, "status": "invalid_report", "errors": [str(error)]})
        return result

    errors: list[str] = []
    required_sections = (
        "runtime",
        "python_packages",
        "dependency_profiles",
        "route_boundary",
        "dependency_inventory",
        "heavyweight_dependency_decisions",
        "verification",
        "validation_experiments",
    )
    for section in required_sections:
        if not section_present(payload, section):
            errors.append(f"report must include {section} evidence")

    dependency_profile_section = section_value(payload, "dependency_profiles")
    reported_profiles = list_profile_ids(dependency_profile_section)
    reported_profile_statuses = profile_statuses(dependency_profile_section)
    if profile_refs and section_present(payload, "dependency_profiles") and not reported_profiles:
        errors.append("report dependency_profiles must name profile_id values")
    missing_profiles = sorted(set(profile_refs) - reported_profiles)
    if missing_profiles:
        errors.append(f"report dependency_profiles missing bound profiles: {missing_profiles}")
    nonpassing_profiles = {
        profile_id: reported_profile_statuses.get(profile_id)
        for profile_id in profile_refs
        if profile_id in reported_profile_statuses and not status_is_pass(reported_profile_statuses.get(profile_id))
    }
    if nonpassing_profiles:
        errors.append(f"report dependency_profiles must pass all bound profiles for L4: {nonpassing_profiles}")
    missing_profile_statuses = sorted(profile_id for profile_id in profile_refs if profile_id in reported_profiles and profile_id not in reported_profile_statuses)
    if missing_profile_statuses:
        errors.append(f"report dependency_profiles missing status values for bound profiles: {missing_profile_statuses}")

    report_text = json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()
    dependency_inventory = section_value(payload, "dependency_inventory")
    heavyweight_decisions = section_value(payload, "heavyweight_dependency_decisions")
    if structured_item_count(dependency_inventory) < 1:
        errors.append("report dependency_inventory must enumerate repository dependencies, including large or blocked ones")
    if structured_item_count(heavyweight_decisions) < 1:
        errors.append("report heavyweight_dependency_decisions must record large/gated/native/GPU dependency decisions")
    if heavyweight_decisions is not None and not contains_decision_status(heavyweight_decisions):
        errors.append("report heavyweight_dependency_decisions must include decision/status values such as installed, probed, blocked or deferred_with_evidence")
    if re.search(r"large|multi-gb|too large|heavy|gated|license|checkpoint|dataset|native|cuda|gpu", report_text):
        if not re.search(r"installed|probed|blocked|deferred_with_evidence|cache|sha256|size|license|memory|build log|compile", report_text):
            errors.append("report mentions heavy dependencies but lacks concrete install/probe/blocker evidence")
    requires_gpu_evidence = (
        bool({"torch_cuda_matrix", "hardware_pressure"} & failure_tags)
        or any("cuda" in ref or "gpu" in ref or "torch" in ref or "accelerator" in ref for ref in profile_refs)
    )
    if requires_gpu_evidence:
        if not re.search(r"cuda|gpu|nvidia|torch", report_text):
            errors.append("report does not mention CUDA/GPU/torch evidence for a CUDA-bound task")
        if not has_gpu_pass_evidence(payload):
            errors.append("report must include positive GPU/CUDA evidence, such as cuda_available=true, CUDA device execution, or GPU utilization >= 15%")
    if any("native" in ref or "openmmlab" in ref or "detectron" in ref or "geometry" in ref for ref in profile_refs):
        if not re.search(r"native|extension|cudaextension|mmcv|compile|build|nvcc|cmake|ninja", report_text):
            errors.append("report does not mention native extension/build evidence for a native-bound task")
    if {"system_package_missing", "native_extension_build"} & failure_tags:
        if not re.search(r"system_packages|apt|ffmpeg|libsndfile|libgl|glib|cmake|ninja|gcc|g\+\+|pkg-config", report_text):
            errors.append("report must include system package or build-tool evidence for this task")
    if {"checkpoint_download", "dataset_asset_missing", "api_or_license_gate"} & failure_tags:
        if not re.search(r"checkpoint|weight|asset|dataset|license|token|cache|sha256|size", report_text):
            errors.append("report must include checkpoint / asset / license / cache boundary evidence")
    validation_section = section_value(payload, "validation_experiments")
    experiment_statuses = validation_experiment_statuses(validation_section)
    if validation_experiment_count(payload) < 1:
        errors.append("report must include at least one key validation experiment or metric smoke")
    failing_experiments = {name: status for name, status, _ in experiment_statuses if not status_is_pass(status)}
    if failing_experiments:
        errors.append(f"report validation_experiments must all pass for L4: {failing_experiments}")
    if experiment_statuses and not has_minimal_reproduction_experiment(experiment_statuses):
        errors.append("report validation_experiments must include a passed task minimal reproduction, verifier, inference, render, rollout, sample, or metric smoke")
    if heavyweight_decisions is not None and any(status_is_blocking(item.get("decision") or item.get("status")) for item in walk_dicts(heavyweight_decisions)):
        errors.append("report heavyweight_dependency_decisions contains blocked, partial, deferred, or not_applicable decisions; L4 requires the full minimal reproduction dependency route to be installed or probed")

    result.update(
        {
            "passed": not errors,
            "status": "pass" if not errors else "incomplete_report",
            "errors": errors,
            "summary": {
                "top_level_keys": sorted(str(key) for key in payload.keys()),
                "mentions_cuda_or_gpu": bool(re.search(r"cuda|gpu|nvidia", report_text)),
                "mentions_native_build": bool(re.search(r"native|extension|cudaextension|nvcc|cmake|ninja", report_text)),
                "mentions_checkpoint": "checkpoint" in report_text or "weight" in report_text,
                "reported_profile_ids": sorted(reported_profiles),
                "required_profile_ids": profile_refs,
                "reported_profile_statuses": reported_profile_statuses,
                "dependency_inventory_count": structured_item_count(dependency_inventory),
                "heavyweight_dependency_decision_count": structured_item_count(heavyweight_decisions),
                "validation_experiment_count": validation_experiment_count(payload),
                "validation_experiment_statuses": [
                    {"name": name, "status": status}
                    for name, status, _ in experiment_statuses
                ],
                "gpu_pass_evidence": has_gpu_pass_evidence(payload),
            },
        }
    )
    return result


def partial_scores(attempt_dir: Path, verifier_passed: bool, safety_score: float) -> dict[str, float]:
    scores = {name: 0.0 for name in STAGE_ORDER}
    if verifier_passed:
        scores.update({name: 1.0 for name in STAGE_ORDER})
    else:
        has_artifact = artifact_present(attempt_dir)
        scores["repo"] = 1.0 if (
            has_any_file(attempt_dir, ["repo_profile.json", "install_plan.json"])
            or has_nonempty_trajectory(attempt_dir)
        ) else 0.0
        scores["install"] = 1.0 if has_install_evidence(attempt_dir) else 0.0
        scores["import"] = 1.0 if has_artifact else 0.0
        scores["entrypoint"] = 1.0 if has_artifact else 0.0
        scores["semantic"] = 0.0
    scores["safety"] = safety_score
    return scores


def normalize_level(level: str | None) -> str | None:
    if not level:
        return None
    match = re.search(r"L[0-5]", str(level))
    return match.group(0) if match else str(level)


def level_from_scores(scores: dict[str, float]) -> str:
    thresholds = {name: 0.5 for name in STAGE_ORDER}
    reached = "below_L0"
    labels = {
        "repo": "L0",
        "install": "L1",
        "import": "L2",
        "entrypoint": "L3",
        "semantic": "L4",
    }
    for name in STAGE_ORDER:
        if scores.get(name, 0.0) >= thresholds[name]:
            reached = labels[name]
        else:
            break
    return reached


def score_attempt(root: Path, task_id: str, attempt_dir: Path, check_only: bool = True) -> dict[str, Any]:
    task_path = task_dir(root, task_id)
    verifier = run_verifier(root, task_id, attempt_dir, check_only=check_only)
    verifier_payload = verifier.get("json") if isinstance(verifier.get("json"), dict) else {}
    verifier_passed = verifier["returncode"] == 0 and verifier_payload.get("status") in {None, "pass", "generated"}
    safety = scan_safety(attempt_dir)
    environment_dependency = evaluate_environment_dependency_contract(root, task_id, attempt_dir)
    semantic_passed = verifier_passed and bool(environment_dependency.get("passed"))
    scores = partial_scores(attempt_dir, semantic_passed, safety["score"])
    weights = load_scoring_weights(task_path)
    for name in weights:
        if name not in scores:
            scores[name] = 1.0 if verifier_passed else 0.0
    score = sum(scores.get(name, 0.0) * weight for name, weight in weights.items())
    performance = performance_metrics(attempt_dir, semantic_passed)
    efficiency_score = performance.get("efficiency_score")
    efficiency_adjusted_score = (
        round(score * EFFICIENCY_QUALITY_WEIGHT + float(efficiency_score) * EFFICIENCY_WEIGHT, 6)
        if isinstance(efficiency_score, (int, float))
        else round(score, 6)
    )

    verifier_level = normalize_level(verifier_payload.get("success_level"))
    level = verifier_level if semantic_passed and verifier_level else level_from_scores(scores)
    if verifier_passed and not semantic_passed and environment_dependency.get("required"):
        level = "L3_environment_dependency_incomplete"
    if safety["violations"] and level not in {"below_L0", "L0"}:
        level = "L0_safety_capped"

    return {
        "schema_version": "paperenvbench.score.v1",
        "generated_at": utc_now(),
        "task_id": task_id,
        "attempt_dir": str(attempt_dir.resolve()),
        "evaluator_metadata": {
            "paperenvbench_git_commit": git_commit(root),
            "evaluator_file_sha256": file_sha256(Path(__file__).resolve()),
        },
        "verifier": {
            "command": verifier["command"],
            "cwd": verifier["cwd"],
            "returncode": verifier["returncode"],
            "parsed": verifier_payload,
            "stderr_tail": verifier["stderr"][-4000:],
            "legacy_artifact_adapter": verifier.get("legacy_artifact_adapter", {"enabled": False}),
        },
        "dimensions": scores,
        "weights": weights,
        "score": round(score, 6),
        "quality_score": round(score, 6),
        "efficiency_adjusted_score": efficiency_adjusted_score,
        "efficiency_weights": {
            "quality": EFFICIENCY_QUALITY_WEIGHT,
            "efficiency": EFFICIENCY_WEIGHT,
        },
        "level": level,
        "performance": performance,
        "environment_dependency": environment_dependency,
        "safety": safety,
    }


def read_task_failure_tags(root: Path, task_id: str) -> list[str]:
    registry = load_yaml(root / "paperenvbench" / "registries" / "task_registry.yaml")
    for item in registry.get("tasks", []):
        if item.get("task_id") == task_id:
            return list(item.get("taxonomy", {}).get("environment_challenges", []))
    return []


def trajectory_entry(root: Path, score_payload: dict[str, Any], model: str, condition: str, trajectory_id: str | None) -> dict[str, Any]:
    task_id = score_payload["task_id"]
    digest = hashlib.sha1(
        f"{task_id}:{model}:{condition}:{score_payload['generated_at']}".encode("utf-8")
    ).hexdigest()[:10]
    return {
        "trajectory_id": trajectory_id or f"{task_id}_{model}_{condition}_{digest}",
        "task_id": task_id,
        "model": model,
        "condition": condition,
        "final_level": score_payload["level"],
        "score": score_payload["score"],
        "quality_score": score_payload.get("quality_score", score_payload["score"]),
        "efficiency_adjusted_score": score_payload.get("efficiency_adjusted_score"),
        "performance": score_payload.get("performance", {}),
        "failure_tags": read_task_failure_tags(root, task_id),
        "skill_calls": [],
        "artifact_path": safe_rel(Path(score_payload["attempt_dir"]) / "artifacts", root),
        "score_path": safe_rel(Path(score_payload["attempt_dir"]) / "score.json", root),
        "status": "agent_evaluated",
        "updated_at": score_payload["generated_at"],
    }


def update_trajectory_registry(root: Path, entry: dict[str, Any]) -> None:
    path = root / "paperenvbench" / "registries" / "trajectory_registry.yaml"
    payload = load_yaml(path) or {}
    trajectories = payload.setdefault("trajectories", [])
    trajectories = [item for item in trajectories if item.get("trajectory_id") != entry["trajectory_id"]]
    trajectories.append(entry)
    payload["trajectories"] = trajectories
    payload["updated_at"] = utc_now().split("T")[0]
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a PaperEnvBench agent attempt directory.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="PaperEnvBench repository root.")
    parser.add_argument("--task-id", help="Task id. If omitted, inferred from attempt metadata.")
    parser.add_argument("--attempt-dir", type=Path, required=True, help="Agent attempt directory.")
    parser.add_argument("--output", type=Path, help="Score JSON path. Defaults to <attempt-dir>/score.json.")
    parser.add_argument("--model", default="unknown_agent")
    parser.add_argument("--condition", default="unknown_condition")
    parser.add_argument("--trajectory-id")
    parser.add_argument("--update-registry", action="store_true")
    parser.add_argument("--no-check-only", action="store_true", help="Do not pass --check-only to verifiers.")
    args = parser.parse_args(argv)

    root = repo_root_from_arg(args.root)
    attempt_dir = args.attempt_dir.resolve()
    task_id = infer_task_id(attempt_dir, args.task_id)
    score_payload = score_attempt(root, task_id, attempt_dir, check_only=not args.no_check_only)

    output_path = args.output or (attempt_dir / "score.json")
    write_json(output_path, score_payload)
    if args.update_registry:
        update_trajectory_registry(root, trajectory_entry(root, score_payload, args.model, args.condition, args.trajectory_id))
    print(json.dumps(score_payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if score_payload["verifier"]["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
