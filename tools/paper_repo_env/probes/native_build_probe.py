from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import sysconfig
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_text(command: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=timeout)
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"returncode": 127, "stdout": "", "stderr": repr(exc)}


def executable_probe(name: str, version_args: list[str] | None = None) -> dict[str, Any]:
    path = shutil.which(name)
    payload: dict[str, Any] = {"available": path is not None, "path": path}
    if path and version_args is not None:
        payload["version"] = run_text([path, *version_args])
    return payload


def python_header_probe() -> dict[str, Any]:
    include_dir = Path(sysconfig.get_paths().get("include", ""))
    pyconfig = include_dir / "Python.h"
    return {
        "include_dir": str(include_dir),
        "python_h_exists": pyconfig.exists(),
        "python_h": str(pyconfig),
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "cc": sysconfig.get_config_var("CC"),
        "cxx": sysconfig.get_config_var("CXX"),
    }


def build_probe() -> dict[str, Any]:
    return {
        "python_headers": python_header_probe(),
        "executables": {
            "gcc": executable_probe("gcc", ["--version"]),
            "g++": executable_probe("g++", ["--version"]),
            "clang": executable_probe("clang", ["--version"]),
            "cmake": executable_probe("cmake", ["--version"]),
            "ninja": executable_probe("ninja", ["--version"]),
            "make": executable_probe("make", ["--version"]),
            "pkg-config": executable_probe("pkg-config", ["--version"]),
        },
    }


def collect_blockers(payload: dict[str, Any]) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    if not payload["python_headers"]["python_h_exists"]:
        blockers.append(
            {
                "severity": "error",
                "code": "python_headers_missing",
                "message": f"Python.h not found at {payload['python_headers']['python_h']}",
            }
        )
    if not (payload["executables"]["gcc"]["available"] or payload["executables"]["clang"]["available"]):
        blockers.append(
            {
                "severity": "error",
                "code": "c_compiler_missing",
                "message": "Neither gcc nor clang is available.",
            }
        )
    if not (payload["executables"]["g++"]["available"] or payload["executables"]["clang"]["available"]):
        blockers.append(
            {
                "severity": "warning",
                "code": "cxx_compiler_missing",
                "message": "Neither g++ nor clang is available for C++ extension builds.",
            }
        )
    if not payload["executables"]["cmake"]["available"]:
        blockers.append(
            {
                "severity": "warning",
                "code": "cmake_missing",
                "message": "cmake is unavailable; CMake-based native extensions may fail.",
            }
        )
    if not payload["executables"]["ninja"]["available"]:
        blockers.append(
            {
                "severity": "warning",
                "code": "ninja_missing",
                "message": "ninja is unavailable; PyTorch native extension builds may fall back or fail.",
            }
        )
    return blockers


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe native Python extension build dependencies.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    probes = build_probe()
    blockers = collect_blockers(probes)
    errors = [item for item in blockers if item["severity"] == "error"]
    payload = {
        "generated_at": utc_now(),
        "probe": "native_build_probe",
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "probes": probes,
        "blockers": blockers,
        "ok": not errors,
        "summary": {"error_blocker_count": len(errors), "warning_blocker_count": len(blockers) - len(errors)},
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.json:
        print(text)
    else:
        print(f"ok={payload['ok']} errors={len(errors)} warnings={len(blockers) - len(errors)}")
        for blocker in blockers:
            print(f"{blocker['severity']} {blocker['code']}: {blocker['message']}")
    return 1 if args.strict and not payload["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
