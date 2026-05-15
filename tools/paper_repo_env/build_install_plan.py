#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dependency_commands(profile: dict[str, Any]) -> list[dict[str, Any]]:
    deps = profile.get("dependencies", {})
    python_deps = deps.get("python_dependencies", [])
    commands: list[dict[str, Any]] = [
        {
            "phase": "environment",
            "command": "python -m venv .venv && . .venv/bin/activate && python -m pip install --upgrade pip setuptools wheel",
            "rationale": "Create an isolated Python environment before installing research-repo dependencies.",
        }
    ]
    system = deps.get("system_dependency_keywords", [])
    if system:
        commands.append(
            {
                "phase": "system",
                "command": "apt-get update && apt-get install -y " + " ".join(sorted(set(system))),
                "requires_privilege": True,
                "rationale": "Install system libraries inferred from README or dependency files.",
            }
        )
    for req_file in deps.get("metadata_files", {}).get("requirements", []):
        commands.append(
            {
                "phase": "python_dependencies",
                "command": f"python -m pip install -r {req_file}",
                "rationale": "Install pinned or declared requirements before editable install.",
            }
        )
    if python_deps and not deps.get("metadata_files", {}).get("requirements"):
        commands.append(
            {
                "phase": "python_dependencies",
                "command": "python -m pip install " + " ".join(str(dep) for dep in python_deps[:30]),
                "rationale": "Install direct dependencies discovered from project metadata.",
            }
        )
    commands.append(
        {
            "phase": "repo_install",
            "command": "python -m pip install -e .",
            "rationale": "Expose local package modules and console scripts from the pinned repository checkout.",
        }
    )
    return commands


def verification_plan(profile: dict[str, Any]) -> list[dict[str, Any]]:
    entrypoints = profile.get("entrypoints", [])
    routes = profile.get("taxonomy_hints", {}).get("verification_type", [])
    plan: list[dict[str, Any]] = []
    imports = [item for item in entrypoints if item.get("type") == "import"]
    if imports:
        module = imports[0]["name"]
        plan.append(
            {
                "level": "L2",
                "command": f"python - <<'PY'\nimport {module}\nprint({module}.__name__)\nPY",
                "expected": "Core module imports without import-time dependency errors.",
            }
        )
    scripts = [item for item in entrypoints if item.get("type") in {"console_script", "script"}]
    if scripts:
        target = scripts[0]["name"]
        plan.append(
            {
                "level": "L3",
                "command": f"# Run the smallest documented help or smoke path for {target}",
                "expected": "Entrypoint exits with code 0 and writes a minimal artifact.",
            }
        )
    if "output_artifact" in routes or "single_sample_inference" in routes:
        plan.append(
            {
                "level": "L4",
                "command": "python verify.py --artifact-dir artifacts --json",
                "expected": "Verifier accepts the generated artifact bundle and semantic checks pass.",
            }
        )
    return plan


def build_plan(profile: dict[str, Any], task_id: str | None) -> dict[str, Any]:
    taxonomy = profile.get("taxonomy_hints", {})
    return {
        "schema_version": "paperenvbench.install_plan.v1",
        "generated_at": utc_now(),
        "task_id": task_id or profile.get("task_id"),
        "repo": {
            "name": profile.get("repo_name"),
            "path": profile.get("repo_path"),
            "commit": profile.get("repo_commit"),
            "remote_url": profile.get("remote_url"),
            "package_manager": profile.get("package_manager"),
        },
        "taxonomy_hints": taxonomy,
        "constraints": {
            "docker_allowed": False,
            "final_success_requires": "L4 artifact-level semantic verification",
            "secrets_policy": "temporary environment variables only",
        },
        "commands": dependency_commands(profile),
        "verification": verification_plan(profile),
        "risk_flags": taxonomy.get("environment_challenges", []),
        "expected_outputs": [
            "attempt.log",
            "artifacts/expected_artifact.json",
            "failure_report.json",
            "score.json",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a PaperEnvBench install_plan.json from repo_profile.json.")
    parser.add_argument("repo_profile", type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--output", type=Path, default=Path("install_plan.json"))
    args = parser.parse_args()

    profile = load_json(args.repo_profile)
    plan = build_plan(profile, args.task_id)
    write_json(args.output, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
