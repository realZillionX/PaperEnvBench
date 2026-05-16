#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_text(command: list[str], timeout: int = 15) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=timeout)
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"returncode": 127, "stdout": "", "stderr": repr(exc)}


def parse_gpu_sample(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            rows.append(
                {
                    "timestamp": parts[0],
                    "index": int(parts[1]),
                    "name": parts[2],
                    "utilization_gpu_percent": int(parts[3]),
                    "memory_used_mib": int(parts[4]),
                    "power_draw_w": float(parts[5]),
                }
            )
        except ValueError:
            continue
    return rows


def nvidia_smi_sample(device_index: int) -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "returncode": 127, "rows": [], "error": "nvidia-smi not found"}
    result = run_text(
        [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-gpu=timestamp,index,name,utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "available": result["returncode"] == 0,
        "returncode": result["returncode"],
        "rows": parse_gpu_sample(result["stdout"]),
        "stderr": result["stderr"],
    }


def torch_preflight(device_index: int) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}

    payload: dict[str, Any] = {
        "ok": bool(torch.cuda.is_available()) and torch.cuda.device_count() > device_index,
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_compiled": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if payload["ok"]:
        torch.cuda.set_device(device_index)
        props = torch.cuda.get_device_properties(device_index)
        payload.update(
            {
                "device_index": device_index,
                "device_name": torch.cuda.get_device_name(device_index),
                "capability": list(torch.cuda.get_device_capability(device_index)),
                "total_memory_gb": round(props.total_memory / (1024**3), 3),
            }
        )
    return payload


def worker_main(
    stop: mp.Event,
    ready: mp.Event,
    device_index: int,
    matrix_size: int,
    dtype_name: str,
    synchronize_every: int,
) -> None:
    import torch

    torch.cuda.set_device(device_index)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[dtype_name]
    device = torch.device(f"cuda:{device_index}")
    a = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
    b = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
    c = torch.empty((matrix_size, matrix_size), device=device, dtype=dtype)
    torch.cuda.synchronize(device)
    ready.set()

    iteration = 0
    while not stop.is_set():
        c = torch.mm(a, b, out=c)
        a, c = c, a
        iteration += 1
        if synchronize_every > 0 and iteration % synchronize_every == 0:
            torch.cuda.synchronize(device)
    torch.cuda.synchronize(device)


def summarize_samples(samples: list[dict[str, Any]], min_utilization: int) -> dict[str, Any]:
    values = [
        int(row["utilization_gpu_percent"])
        for sample in samples
        for row in sample.get("rows", [])
        if isinstance(row.get("utilization_gpu_percent"), int)
    ]
    if not values:
        return {
            "sample_count": 0,
            "max_utilization_gpu_percent": None,
            "avg_utilization_gpu_percent": None,
            "active_sample_count": 0,
            "threshold_sample_count": 0,
        }
    active = [value for value in values if value > 0]
    above = [value for value in values if value >= min_utilization]
    return {
        "sample_count": len(values),
        "max_utilization_gpu_percent": max(values),
        "avg_utilization_gpu_percent": round(sum(values) / len(values), 3),
        "active_sample_count": len(active),
        "threshold_sample_count": len(above),
        "first_utilization_gpu_percent": values[0],
        "last_utilization_gpu_percent": values[-1],
    }


def run_guard(args: argparse.Namespace) -> dict[str, Any]:
    preflight = torch_preflight(args.device)
    samples: list[dict[str, Any]] = []
    started_at = time.time()

    payload: dict[str, Any] = {
        "generated_at": utc_now(),
        "probe": "gpu_occupancy_guard",
        "target": {
            "device": args.device,
            "min_utilization_gpu_percent": args.min_utilization,
            "duration_seconds": args.duration_seconds,
            "warmup_seconds": args.warmup_seconds,
            "matrix_size": args.matrix_size,
            "dtype": args.dtype,
        },
        "preflight": preflight,
        "samples": samples,
    }
    if not preflight.get("ok"):
        payload["ok"] = False
        payload["status"] = "error"
        payload["summary"] = {
            "reason": "torch_cuda_preflight_failed",
            "elapsed_seconds": round(time.time() - started_at, 3),
        }
        return payload

    stop = mp.Event()
    ready = mp.Event()
    process = mp.Process(
        target=worker_main,
        args=(stop, ready, args.device, args.matrix_size, args.dtype, args.synchronize_every),
        daemon=True,
    )
    process.start()
    try:
        if not ready.wait(timeout=args.startup_timeout_seconds):
            payload["ok"] = False
            payload["status"] = "error"
            payload["summary"] = {
                "reason": "workload_startup_timeout",
                "worker_exitcode": process.exitcode,
                "elapsed_seconds": round(time.time() - started_at, 3),
            }
            return payload

        time.sleep(max(args.warmup_seconds, 0.0))
        deadline = time.time() + max(args.duration_seconds, 0.0)
        while time.time() <= deadline:
            sample = nvidia_smi_sample(args.device)
            sample["sampled_at"] = utc_now()
            samples.append(sample)
            time.sleep(max(args.sample_interval_seconds, 0.1))
    finally:
        stop.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    summary = summarize_samples(samples, args.min_utilization)
    max_util = summary.get("max_utilization_gpu_percent")
    passed = isinstance(max_util, int) and max_util >= args.min_utilization
    payload["ok"] = passed
    payload["status"] = "pass" if passed else "blocked"
    payload["summary"] = {
        **summary,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "worker_exitcode": process.exitcode,
        "meets_min_utilization": passed,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bounded CUDA matmul workload and verify visible GPU utilization.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--min-utilization", type=int, default=15, help="Required nvidia-smi utilization percentage.")
    parser.add_argument("--duration-seconds", type=float, default=60.0, help="Sampling duration after warmup.")
    parser.add_argument("--warmup-seconds", type=float, default=5.0, help="Warmup before utilization sampling.")
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0, help="Seconds between nvidia-smi samples.")
    parser.add_argument("--startup-timeout-seconds", type=float, default=30.0, help="Worker startup timeout.")
    parser.add_argument("--matrix-size", type=int, default=4096, help="Square matrix size used by the CUDA workload.")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--synchronize-every", type=int, default=16, help="CUDA synchronize cadence inside the worker.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if the utilization target is not reached.")
    args = parser.parse_args()

    if args.matrix_size < 256:
        raise SystemExit("--matrix-size must be at least 256")

    mp.set_start_method("spawn", force=True)
    payload = run_guard(args)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.json:
        print(text)
    else:
        summary = payload.get("summary", {})
        print(
            "gpu_occupancy_guard "
            f"status={payload.get('status')} "
            f"max_util={summary.get('max_utilization_gpu_percent')} "
            f"target={args.min_utilization}"
        )
    return 1 if args.strict and not payload.get("ok") else 0


if __name__ == "__main__":
    raise SystemExit(main())
