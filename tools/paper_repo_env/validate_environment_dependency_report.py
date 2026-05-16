#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paperenvbench.evaluator import evaluate_environment_dependency_contract, repo_root_from_arg


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an attempt-level environment_dependency_report.json.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="PaperEnvBench repository root.")
    parser.add_argument("--task-id", required=True, help="Task id whose environment dependency contract should be checked.")
    parser.add_argument("--attempt-dir", type=Path, required=True, help="Agent attempt directory.")
    parser.add_argument("--json", action="store_true", help="Print structured JSON.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when the report is required and incomplete.")
    args = parser.parse_args()

    root = repo_root_from_arg(args.root)
    payload = evaluate_environment_dependency_contract(root, args.task_id, args.attempt_dir.resolve())
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"required={payload['required']} status={payload['status']} passed={payload['passed']}")
        if payload.get("report_path"):
            print(f"report={payload['report_path']}")
        for error in payload.get("errors", []):
            print(f"- {error}")
    return 1 if args.strict and not payload.get("passed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
