from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run(command: list[str], timeout: int = 60) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=timeout)
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
    except Exception as exc:
        return {"command": command, "returncode": 127, "error": repr(exc)}


def module_available(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {"available": True, "version": getattr(module, "__version__", None), "path": getattr(module, "__file__", None)}
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def torch_cuda_smoke() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "cuda_available": False, "error": repr(exc)}
    payload: dict[str, Any] = {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_compiled": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        x = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
        y = x @ x.T
        torch.cuda.synchronize()
        payload["device"] = "cuda"
        payload["matmul_sum"] = float(y.detach().cpu().sum().item())
    return payload


def make_tiny_video(path: Path) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {"ok": False, "error": "ffmpeg binary is unavailable"}
    return run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=4",
            "-frames:v",
            "4",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
    )


def ffmpeg_decode(path: Path) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {"ok": False, "error": "ffmpeg binary is unavailable"}
    with tempfile.TemporaryDirectory(prefix="paperenvbench_ffmpeg_decode_") as tmp:
        frame_path = Path(tmp) / "frame_001.png"
        result = run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-frames:v",
                "1",
                str(frame_path),
            ]
        )
        return {
            **result,
            "frame_exists": frame_path.exists(),
            "frame_bytes": frame_path.stat().st_size if frame_path.exists() else 0,
            "ok": result.get("returncode") == 0 and frame_path.exists() and frame_path.stat().st_size > 0,
        }


def pyav_decode(path: Path) -> dict[str, Any]:
    available = module_available("av")
    if not available["available"]:
        return available
    try:
        import av

        frame_count = 0
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                frame.to_ndarray(format="rgb24")
                frame_count += 1
        return {**available, "decoded_frames": frame_count, "ok": frame_count >= 1}
    except Exception as exc:
        return {**available, "ok": False, "error": repr(exc)}


def decord_decode(path: Path) -> dict[str, Any]:
    available = module_available("decord")
    if not available["available"]:
        return available
    try:
        import decord

        reader = decord.VideoReader(str(path))
        frame = reader[0].asnumpy()
        return {**available, "decoded_frames": len(reader), "first_frame_shape": list(frame.shape), "ok": len(reader) >= 1}
    except Exception as exc:
        return {**available, "ok": False, "error": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe video decode, ffmpeg, and CUDA tensor surfaces.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when required video surfaces are blocked.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="paperenvbench_video_probe_") as tmp:
        video_path = Path(tmp) / "tiny_probe.mp4"
        ffmpeg_create = make_tiny_video(video_path)
        ffmpeg_native_decode = (
            ffmpeg_decode(video_path) if video_path.exists() else {"ok": False, "error": "tiny video was not created"}
        )
        pyav = pyav_decode(video_path) if video_path.exists() else {"available": False, "error": "tiny video was not created"}
        decord = decord_decode(video_path) if video_path.exists() else {"available": False, "error": "tiny video was not created"}

    blockers = []
    if ffmpeg_create.get("returncode") != 0:
        blockers.append({"code": "ffmpeg_video_generation_failed", "evidence": ffmpeg_create})
    if not (ffmpeg_native_decode.get("ok") or pyav.get("ok") or decord.get("ok")):
        blockers.append(
            {
                "code": "video_decode_backend_unavailable",
                "ffmpeg_native_decode": ffmpeg_native_decode,
                "pyav": pyav,
                "decord": decord,
            }
        )
    torch = torch_cuda_smoke()
    if torch.get("cuda_available") is not True:
        blockers.append({"code": "torch_cuda_unavailable", "torch": torch})

    payload = {
        "generated_at": utc_now(),
        "status": "pass" if not blockers else "blocked",
        "ffmpeg": {"path": shutil.which("ffmpeg"), "create_tiny_video": ffmpeg_create, "decode_tiny_video": ffmpeg_native_decode},
        "pyav": pyav,
        "decord": decord,
        "torch": torch,
        "blockers": blockers,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"video_decode_probe status={payload['status']} blockers={len(blockers)}")
    return 1 if args.strict and blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
