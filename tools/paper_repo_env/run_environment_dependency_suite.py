from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def repo_root_from_arg(root: Path) -> Path:
    root = root.resolve()
    if (root / "paperenvbench/registries/environment_dependency_registry.yaml").exists():
        return root
    raise FileNotFoundError(f"not a PaperEnvBench root: {root}")


def registry(root: Path) -> dict[str, Any]:
    return load_yaml(root / "paperenvbench/registries/environment_dependency_registry.yaml")


def task_to_profiles(payload: dict[str, Any]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for binding in payload.get("task_bindings", []):
        refs = {str(item) for item in binding.get("profile_refs", [])}
        for task_id in binding.get("task_ids", []):
            mapping.setdefault(str(task_id), set()).update(refs)
    return mapping


def expand_profile_closure(profiles: dict[str, Any], selected: set[str]) -> list[str]:
    resolved: list[str] = []
    visiting: set[str] = set()

    def visit(profile_id: str) -> None:
        if profile_id in resolved:
            return
        if profile_id in visiting:
            raise ValueError(f"dependency cycle at profile {profile_id}")
        if profile_id not in profiles:
            raise KeyError(f"unknown profile {profile_id}")
        visiting.add(profile_id)
        for dep in profiles[profile_id].get("depends_on", []) or []:
            visit(str(dep))
        visiting.remove(profile_id)
        resolved.append(profile_id)

    for item in sorted(selected):
        visit(item)
    return resolved


def select_profiles(payload: dict[str, Any], profile_args: list[str], task_args: list[str], group_args: list[str]) -> list[str]:
    profiles = payload.get("probe_profiles", {})
    selected: set[str] = set()
    if profile_args:
        selected.update(profile_args)
    if task_args:
        mapping = task_to_profiles(payload)
        for task_id in task_args:
            if task_id not in mapping:
                raise KeyError(f"task has no environment dependency binding: {task_id}")
            selected.update(mapping[task_id])
    if group_args:
        for binding in payload.get("task_bindings", []):
            if binding.get("group") in group_args:
                selected.update(str(item) for item in binding.get("profile_refs", []))
    if not selected:
        selected = set(profiles)
    return expand_profile_closure(profiles, selected)


def render_command(command: list[Any], python_executable: str) -> list[str]:
    rendered = []
    for item in command:
        text = str(item)
        rendered.append(text.replace("{python}", python_executable))
    return rendered


def parse_json_stdout(stdout: str) -> Any:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start >= 0:
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            return None
    return None


def dotted_value(payload: Any, dotted_path: str) -> Any:
    current = payload
    parts = dotted_path.split(".")
    for index, part in enumerate(parts):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if index == 0 and isinstance(current, dict) and isinstance(current.get("modules"), dict) and part in current["modules"]:
            current = current["modules"][part]
            continue
        return None
    return current


def compare_values(actual: Any, operator: str, expected: str) -> bool:
    if operator == "=":
        return str(actual).lower() == expected.lower()
    try:
        left = float(actual)
        right = float(expected)
    except (TypeError, ValueError):
        return False
    if operator == ">=":
        return left >= right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    if operator == "<":
        return left < right
    return False


def evaluate_expected(parsed: Any, expected: list[Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not expected:
        return results
    for raw_expectation in expected:
        expectation = str(raw_expectation)
        operator = None
        for candidate in (">=", "<=", ">", "<", "="):
            if candidate in expectation:
                operator = candidate
                break
        if operator:
            path, expected_value = [part.strip() for part in expectation.split(operator, 1)]
            actual = dotted_value(parsed, path)
            passed = compare_values(actual, operator, expected_value)
        else:
            path = expectation.strip()
            expected_value = None
            actual = dotted_value(parsed, path)
            passed = actual not in (None, False, "", [], {})
        results.append(
            {
                "expectation": expectation,
                "path": path,
                "operator": operator or "exists",
                "expected": expected_value,
                "actual": actual,
                "passed": passed,
            }
        )
    return results


def infer_status(returncode: int, parsed: Any) -> str:
    if returncode != 0:
        return "error"
    if isinstance(parsed, dict):
        value = parsed.get("status")
        if value in {"pass", "blocked", "error"}:
            return str(value)
        if parsed.get("ok") is False:
            return "blocked"
        summary = parsed.get("summary")
        if isinstance(summary, dict) and summary.get("has_error_blocker") is True:
            return "blocked"
    return "pass"


def run_profile(root: Path, profile_id: str, profile: dict[str, Any], python_executable: str, output_dir: Path, timeout: int) -> dict[str, Any]:
    command = render_command(profile.get("command", []), python_executable)
    output_path = output_dir / f"{profile_id}.json"
    command_with_output = list(command)
    if "--output" not in command_with_output:
        command_with_output.extend(["--output", str(output_path)])
    completed = subprocess.run(
        command_with_output,
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )
    parsed = None
    if output_path.exists():
        try:
            parsed = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:
            parsed = None
    if parsed is None:
        parsed = parse_json_stdout(completed.stdout)
    status = infer_status(completed.returncode, parsed)
    expectation_results = evaluate_expected(parsed, profile.get("expected", []) or [])
    expectation_failures = [item for item in expectation_results if not item["passed"]]
    if status == "pass" and expectation_failures:
        status = "blocked"
    return {
        "profile_id": profile_id,
        "description": profile.get("description"),
        "runtime_target": profile.get("runtime_target"),
        "command": command_with_output,
        "returncode": completed.returncode,
        "status": status,
        "output_path": str(output_path),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "parsed_summary": summarize_parsed(parsed),
        "expectations": expectation_results,
    }


def summarize_parsed(parsed: Any) -> Any:
    if not isinstance(parsed, dict):
        return None
    summary: dict[str, Any] = {}
    for key in ("status", "ok", "summary"):
        if key in parsed:
            summary[key] = parsed[key]
    if "blockers" in parsed and isinstance(parsed["blockers"], list):
        summary["blocker_count"] = len(parsed["blockers"])
    if "tasks" in parsed and isinstance(parsed["tasks"], dict):
        summary["task_status_counts"] = {}
        for value in parsed["tasks"].values():
            if isinstance(value, dict):
                status = str(value.get("status") or value.get("ok"))
                summary["task_status_counts"][status] = summary["task_status_counts"].get(status, 0) + 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PaperEnvBench environment dependency probe profiles.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", action="append", default=[], help="Probe profile id to run. Repeatable.")
    parser.add_argument("--task", action="append", default=[], help="Task id whose bound profiles should run. Repeatable.")
    parser.add_argument("--group", action="append", default=[], help="Task binding group whose profiles should run. Repeatable.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run probe commands.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/environment_dependency_suite/latest"))
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any selected profile is blocked or errors.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = repo_root_from_arg(args.root)
    payload = registry(root)
    profiles = payload.get("probe_profiles", {})
    selected = select_profiles(payload, args.profile, args.task, args.group)
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir

    if args.dry_run:
        result = {
            "generated_at": utc_now(),
            "selected_profiles": selected,
            "commands": {
                profile_id: render_command(profiles[profile_id].get("command", []), args.python)
                for profile_id in selected
            },
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_profile(root, profile_id, profiles[profile_id], args.python, output_dir, args.timeout_seconds)
        for profile_id in selected
    ]
    status_counts: dict[str, int] = {}
    for item in results:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    suite = {
        "schema_version": "paperenvbench.environment_dependency_suite.v1",
        "generated_at": utc_now(),
        "root": str(root),
        "python": args.python,
        "selected_profiles": selected,
        "status_counts": status_counts,
        "results": results,
    }
    write_json(output_dir / "suite_summary.json", suite)
    if args.json:
        print(json.dumps(suite, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"profiles={len(selected)} status_counts={status_counts}")
        print(f"summary={output_dir / 'suite_summary.json'}")
    has_bad = any(item["status"] != "pass" for item in results)
    return 1 if args.strict and has_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
