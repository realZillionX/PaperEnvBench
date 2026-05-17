#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import signal
import subprocess
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


def parse_gpu_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_gpu_percent": int(parts[2]),
                    "memory_used_mib": int(parts[3]),
                    "power_draw_w": float(parts[4]),
                }
            )
        except ValueError:
            continue
    return rows


def nvidia_smi_rows() -> list[dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    result = run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    if result["returncode"] != 0:
        return []
    return parse_gpu_rows(result["stdout"])


def average(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def worker_main(stop: mp.Event, ready: mp.Event, device_index: int, matrix_size: int, dtype_name: str) -> None:
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
    while not stop.is_set():
        c = torch.mm(a, b, out=c)
        a, c = c, a
    torch.cuda.synchronize(device)


class GuardProcess:
    def __init__(self, matrix_size: int, dtype: str) -> None:
        self.matrix_size = matrix_size
        self.dtype = dtype
        self.device: int | None = None
        self.stop: mp.Event | None = None
        self.process: mp.Process | None = None

    def start(self, device: int, timeout_seconds: float) -> None:
        self.stop_running()
        stop = mp.Event()
        ready = mp.Event()
        process = mp.Process(target=worker_main, args=(stop, ready, device, self.matrix_size, self.dtype), daemon=True)
        process.start()
        if not ready.wait(timeout=timeout_seconds):
            stop.set()
            process.terminate()
            process.join(timeout=5)
            raise RuntimeError(f"guard worker failed to start on cuda:{device}")
        self.device = device
        self.stop = stop
        self.process = process

    def stop_running(self) -> None:
        if self.stop is not None:
            self.stop.set()
        if self.process is not None:
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
        self.device = None
        self.stop = None
        self.process = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.is_alive() and self.device is not None


def write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def append_jsonl(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def choose_guard_device(rows: list[dict[str, Any]]) -> int:
    return min(rows, key=lambda row: (row["utilization_gpu_percent"], row["memory_used_mib"]))["index"]


def run_controller(args: argparse.Namespace) -> int:
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    guard = GuardProcess(matrix_size=args.matrix_size, dtype=args.dtype)
    started_at = time.time()
    sample_index = 0
    try:
        while not stop_requested:
            rows = nvidia_smi_rows()
            values = [int(row["utilization_gpu_percent"]) for row in rows]
            total_avg = average(values)
            guard_device = guard.device if guard.is_running() else None
            other_values = [
                int(row["utilization_gpu_percent"])
                for row in rows
                if guard_device is None or row["index"] != guard_device
            ]
            other_avg = average(other_values)
            action = "hold"

            if not rows:
                action = "missing_nvidia_smi_sample"
            elif guard.is_running():
                if other_avg >= args.min_utilization or total_avg >= args.release_utilization:
                    guard.stop_running()
                    action = "stop_guard_real_workload_sufficient"
                else:
                    action = "keep_guard_floor"
            elif total_avg < args.min_utilization:
                guard.start(choose_guard_device(rows), args.startup_timeout_seconds)
                action = "start_guard_floor"
            else:
                action = "guard_not_needed"

            payload = {
                "generated_at": utc_now(),
                "probe": "adaptive_gpu_floor_guard",
                "ok": True,
                "target": {
                    "min_utilization_gpu_percent": args.min_utilization,
                    "release_utilization_gpu_percent": args.release_utilization,
                    "matrix_size": args.matrix_size,
                    "dtype": args.dtype,
                },
                "summary": {
                    "elapsed_seconds": round(time.time() - started_at, 3),
                    "sample_index": sample_index,
                    "action": action,
                    "guard_running": guard.is_running(),
                    "guard_device": guard.device,
                    "total_avg_utilization_gpu_percent": round(total_avg, 3),
                    "other_avg_utilization_gpu_percent": round(other_avg, 3),
                },
                "devices": rows,
            }
            write_json(args.state_output, payload)
            append_jsonl(args.jsonl_log, payload)
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)

            sample_index += 1
            time.sleep(args.sample_interval_seconds)
    finally:
        guard.stop_running()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep a GPU notebook above a utilization floor with one adaptive guard GPU.")
    parser.add_argument("--min-utilization", type=int, default=17)
    parser.add_argument("--release-utilization", type=int, default=35)
    parser.add_argument("--sample-interval-seconds", type=float, default=10.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--state-output", type=Path)
    parser.add_argument("--jsonl-log", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.release_utilization <= args.min_utilization:
        raise SystemExit("--release-utilization must be greater than --min-utilization")
    mp.set_start_method("spawn", force=True)
    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
