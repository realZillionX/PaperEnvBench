#!/usr/bin/env python3
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import pathlib
import subprocess
import sys
from typing import Any


TASK_ID = "silero_vad_speech_activity"
EXPECTED_COMMIT = "980b17e9d56463e51393a8d92ded473f1b17896a"
MIN_SEGMENTS = 1
MIN_TOTAL_SPEECH_SEC = 0.5
MIN_VAD_MAX_PROB = 0.5


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_dir: pathlib.Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return ""


def to_builtin(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default=os.environ.get("PAPERENVBENCH_REPO_DIR", "repo"))
    parser.add_argument("--output-dir", default=os.environ.get("PAPERENVBENCH_OUTPUT_DIR", "artifacts"))
    parser.add_argument("--artifact-name", default="speech_activity_summary.json")
    args = parser.parse_args()

    repo_dir = pathlib.Path(args.repo_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = repo_dir / "tests" / "data" / "test.wav"
    jit_path = repo_dir / "src" / "silero_vad" / "data" / "silero_vad.jit"
    if not audio_path.exists():
        raise FileNotFoundError(f"missing audio fixture: {audio_path}")
    if not jit_path.exists():
        raise FileNotFoundError(f"missing packaged JIT model: {jit_path}")

    import torch
    import torchaudio
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

    torch.set_num_threads(1)
    model = load_silero_vad(onnx=False)
    wav = read_audio(str(audio_path), sampling_rate=16000)
    if wav.ndim != 1:
        wav = wav.reshape(-1)

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=16000,
        return_seconds=True,
        visualize_probs=False,
    )
    vad_probs = model.audio_forward(wav, sr=16000).detach().cpu().reshape(-1)
    repo_commit = git_commit(repo_dir)

    total_speech_duration_sec = round(
        sum(float(item["end"]) - float(item["start"]) for item in speech_timestamps),
        6,
    )
    duration_sec = round(float(wav.numel()) / 16000.0, 6)
    max_prob = float(vad_probs.max().item()) if vad_probs.numel() else 0.0
    mean_prob = float(vad_probs.mean().item()) if vad_probs.numel() else 0.0

    checks = {
        "repo_commit_matches": repo_commit == EXPECTED_COMMIT,
        "packaged_jit_model_loaded": jit_path.exists() and sha256_file(jit_path) == "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720",
        "speech_timestamps_nonempty": len(speech_timestamps) >= MIN_SEGMENTS,
        "total_speech_duration_positive": total_speech_duration_sec >= MIN_TOTAL_SPEECH_SEC,
        "vad_probability_above_threshold": max_prob >= MIN_VAD_MAX_PROB,
    }

    summary = {
        "task_id": TASK_ID,
        "repo_commit": repo_commit,
        "python": sys.version.split()[0],
        "package_versions": {
            "silero-vad": metadata.version("silero-vad"),
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
        },
        "input_audio": {
            "path": str(audio_path),
            "sample_rate": 16000,
            "num_samples": int(wav.numel()),
            "duration_sec": duration_sec,
            "sha256": sha256_file(audio_path),
        },
        "model_assets": {
            "jit_path": str(jit_path),
            "jit_sha256": sha256_file(jit_path),
            "torch_hub_used": False,
        },
        "speech_timestamps": to_builtin(speech_timestamps),
        "speech_segment_count": len(speech_timestamps),
        "total_speech_duration_sec": total_speech_duration_sec,
        "vad_probability": {
            "num_frames": int(vad_probs.numel()),
            "max": round(max_prob, 8),
            "mean": round(mean_prob, 8),
        },
        "checks": checks,
        "success_level": "L4" if all(checks.values()) else "below_L4",
    }

    artifact_path = output_dir / args.artifact_name
    artifact_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not all(checks.values()):
        failed = ", ".join(k for k, ok in checks.items() if not ok)
        raise SystemExit(f"semantic checks failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
