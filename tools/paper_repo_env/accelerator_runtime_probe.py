from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_text(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=30)
        return completed.returncode, (completed.stdout + completed.stderr).strip()
    except Exception as exc:
        return 127, repr(exc)


def nvidia_smi() -> dict[str, Any]:
    rc, text = run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    header_rc, header_text = run_text(["nvidia-smi"])
    rows = []
    if rc == 0:
        for line in text.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                rows.append(
                    {
                        "name": parts[0],
                        "memory_total": parts[1],
                        "driver_version": parts[2],
                    }
                )
    return {"returncode": rc, "raw": text, "header_returncode": header_rc, "header": header_text, "gpus": rows}


def torch_probe() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": repr(exc)}

    payload: dict[str, Any] = {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_compiled": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        payload["device_name"] = torch.cuda.get_device_name(0)
        payload["capability"] = list(torch.cuda.get_device_capability(0))
        x = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
        y = x @ x.T
        torch.cuda.synchronize()
        payload["matmul_sum"] = float(y.sum().detach().cpu().item())
        payload["memory_allocated"] = int(torch.cuda.memory_allocated(0))
    return payload


def module_probe(names: list[str]) -> dict[str, Any]:
    result = {}
    for name in names:
        try:
            module = importlib.import_module(name)
            result[name] = {
                "available": True,
                "version": getattr(module, "__version__", None),
                "path": getattr(module, "__file__", None),
            }
        except Exception as exc:
            result[name] = {"available": False, "error": repr(exc)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe accelerator runtime and CUDA-aware Python dependencies.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when GPU or torch CUDA is unavailable.")
    parser.add_argument("--module", action="append", default=[], help="Additional module import probe.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    payload = {
        "generated_at": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "nvidia_smi": nvidia_smi(),
        "torch": torch_probe(),
        "modules": module_probe(args.module),
    }
    ok = bool(payload["nvidia_smi"]["gpus"]) and payload["torch"].get("cuda_available") is True

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"gpu_count={len(payload['nvidia_smi']['gpus'])}")
        print(f"torch_available={payload['torch'].get('available')} cuda_available={payload['torch'].get('cuda_available')}")
        if payload["torch"].get("device_name"):
            print(f"device={payload['torch']['device_name']} torch_cuda={payload['torch'].get('cuda_compiled')}")

    return 0 if ok or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
