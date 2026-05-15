#!/usr/bin/env python3
from __future__ import annotations


# PaperEnvBench artifact-only validation path. This exits before runtime imports
# when --check-only is requested, so the standalone benchmark repo can verify
# gold task packages without vendoring full upstream checkouts or weights.
import argparse as _peb_argparse
import hashlib as _peb_hashlib
import json as _peb_json
import pathlib as _peb_pathlib
import sys as _peb_sys

_PEB_TASK_ID = "demucs_music_separation"
_PEB_EXPECTED_ARTIFACT_SHA256 = "070549c52769b531b4ff07b8b8fdaf9f411ad12ee31cf4e79f826e38b9dd0acf"
_PEB_REQUIRED_SIDE_ARTIFACTS = {'expected_artifact.wav': 1000, 'verification_result.json': 100}


def _peb_sha256(path: _peb_pathlib.Path) -> str:
    digest = _peb_hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peb_check_only() -> None:
    if "--check-only" not in _peb_sys.argv:
        return
    parser = _peb_argparse.ArgumentParser(description=f"Check packaged gold artifact for {_PEB_TASK_ID}.")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--artifact-dir", "--output-dir", dest="artifact_dir", default="artifacts")
    parser.add_argument("--artifact-name", default="expected_artifact.json")
    parser.add_argument("--repo-dir", default=None)
    args, _unknown = parser.parse_known_args()
    task_root = _peb_pathlib.Path(__file__).resolve().parent
    artifact_dir = _peb_pathlib.Path(args.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = task_root / artifact_dir
    artifact_path = artifact_dir / args.artifact_name
    if not artifact_path.exists() and args.artifact_name != "expected_artifact.json":
        artifact_path = artifact_dir / "expected_artifact.json"
    payload = _peb_json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact_sha256 = _peb_sha256(artifact_path)
    payload_checks = payload.get("checks", {})
    payload_checks_true = all(bool(value) for value in payload_checks.values()) if isinstance(payload_checks, dict) else True
    side_checks = {}
    for name, min_size in _PEB_REQUIRED_SIDE_ARTIFACTS.items():
        path = artifact_dir / name
        side_checks[name] = path.exists() and path.stat().st_size >= int(min_size)
    checks = {
        "task_id_matches": payload.get("task_id") == _PEB_TASK_ID,
        "artifact_sha256_matches": artifact_sha256 == _PEB_EXPECTED_ARTIFACT_SHA256,
        "payload_checks_true": payload_checks_true,
        "payload_success_not_false": payload.get("success", True) is not False,
        "side_artifacts_present": all(side_checks.values()),
    }
    ok = all(checks.values())
    result = {
        "task_id": _PEB_TASK_ID,
        "status": "pass" if ok else "fail",
        "mode": "check_only",
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "success_level": payload.get("success_level") or payload.get("expected_success_level"),
        "checks": checks,
        "side_artifacts": side_checks,
    }
    print(_peb_json.dumps(result, indent=2, sort_keys=True) if args.json else result["status"])
    if not ok:
        raise SystemExit(1)
    raise SystemExit(0)


_peb_check_only()


import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import sys
import wave


SAMPLE_RATE = 44100
DURATION_SEC = 0.25
EXPECTED_FRAMES = int(SAMPLE_RATE * DURATION_SEC)


def write_synthetic_mix(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        for i in range(EXPECTED_FRAMES):
            t = i / SAMPLE_RATE
            left = 0.35 * math.sin(2 * math.pi * 440 * t) + 0.15 * math.sin(2 * math.pi * 880 * t)
            right = 0.25 * math.sin(2 * math.pi * 554.37 * t) + 0.12 * math.sin(2 * math.pi * 220 * t)
            wav.writeframes(
                struct.pack(
                    "<hh",
                    int(max(-1.0, min(1.0, left)) * 32767),
                    int(max(-1.0, min(1.0, right)) * 32767),
                )
            )


def wav_stats(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        payload = wav.readframes(frames)
    if sample_width != 2:
        raise AssertionError(f"Expected 16-bit PCM, got sample_width={sample_width}")
    samples = struct.unpack("<" + "h" * (len(payload) // 2), payload)
    peak_abs = max(abs(value) for value in samples) if samples else 0
    rms = math.sqrt(sum(value * value for value in samples) / max(1, len(samples))) / 32767
    return {
        "path": str(path),
        "channels": channels,
        "sample_rate": sample_rate,
        "frames": frames,
        "duration_sec": round(frames / sample_rate, 6),
        "sample_width": sample_width,
        "size_bytes": path.stat().st_size,
        "peak_abs_pcm": peak_abs,
        "rms": round(rms, 8),
        "sha256": digest,
    }


def prepare_import_path(repo_dir: Path, run_root: Path) -> None:
    repo_str = str(repo_dir.resolve())
    run_root_str = str(run_root.resolve())
    cleaned = []
    for item in sys.path:
        if item in ("", run_root_str):
            continue
        cleaned.append(item)
    sys.path[:] = [repo_str] + cleaned


def run_demucs(repo_dir: Path, input_wav: Path, separated_dir: Path) -> None:
    prepare_import_path(repo_dir, input_wav.parents[1])
    import torch
    from demucs.separate import main

    random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)

    main(
        [
            "-n",
            "demucs_unittest",
            "--two-stems",
            "vocals",
            "--other-method",
            "none",
            "--segment",
            "1",
            "-d",
            "cpu",
            "--shifts",
            "0",
            "-j",
            "0",
            "-o",
            str(separated_dir),
            str(input_wav),
        ]
    )


def main() -> int:
    run_root = Path(os.environ.get("PAPERENVBENCH_RUN_ROOT", os.getcwd())).resolve()
    repo_dir = Path(os.environ.get("DEMUX_REPO_DIR", run_root / "demucs")).resolve()
    artifact_dir = Path(os.environ.get("PAPERENVBENCH_ARTIFACT_DIR", run_root / "artifacts")).resolve()
    input_wav = run_root / "input" / "synthetic_mix.wav"
    separated_dir = artifact_dir / "separated"
    expected_wav = artifact_dir / "expected_artifact.wav"
    expected_json = artifact_dir / "expected_artifact.json"
    result_json = artifact_dir / "verification_result.json"

    if not repo_dir.exists():
        raise FileNotFoundError(f"Demucs repo not found: {repo_dir}")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required for Demucs audio loading")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_synthetic_mix(input_wav)
    run_demucs(repo_dir, input_wav, separated_dir)

    stem_wav = separated_dir / "demucs_unittest" / "synthetic_mix" / "vocals.wav"
    if not stem_wav.exists():
        raise FileNotFoundError(f"Expected separated stem not found: {stem_wav}")

    stats = wav_stats(stem_wav)
    checks = {
        "channels_is_stereo": stats["channels"] == 2,
        "sample_rate_is_44100": stats["sample_rate"] == SAMPLE_RATE,
        "frame_count_matches_input": stats["frames"] == EXPECTED_FRAMES,
        "artifact_size_nontrivial": stats["size_bytes"] > 1000,
        "pcm_payload_nonzero": stats["peak_abs_pcm"] > 0,
    }
    if not all(checks.values()):
        raise AssertionError({"checks": checks, "stats": stats})

    shutil.copy2(stem_wav, expected_wav)
    payload = {
        "task_id": "demucs_music_separation",
        "success": True,
        "success_level": "L4_fallback",
        "model": "demucs_unittest",
        "device": "cpu",
        "input": str(input_wav),
        "artifact": str(expected_wav),
        "checks": checks,
        "stats": stats,
    }
    expected_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    result_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

