from __future__ import annotations


# PaperEnvBench artifact-only validation path. This exits before runtime imports
# when --check-only is requested, so the standalone benchmark repo can verify
# gold task packages without vendoring full upstream checkouts or weights.
import argparse as _peb_argparse
import hashlib as _peb_hashlib
import json as _peb_json
import pathlib as _peb_pathlib
import sys as _peb_sys

_PEB_TASK_ID = "encodec_audio_codec_minimal"
_PEB_EXPECTED_ARTIFACT_SHA256 = "a5f6bcafc704bd307ae724dd883303c021bfaba29511f02b54874f8bd352343e"
_PEB_REQUIRED_SIDE_ARTIFACTS = {'expected_artifact.wav': 1000}


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


import json
import math
import os
import sys
import wave
from pathlib import Path


TASK_ID = "encodec_audio_codec_minimal"
REPO_COMMIT = "0e2d0aed29362c8e8f52494baf3e6f99056b214f"


def maybe_reexec_in_task_venv(root: Path) -> None:
    venv_python = root / ".venv" / "bin" / "python"
    if sys.prefix != sys.base_prefix or not venv_python.exists():
        return
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


def write_pcm16_wav(path: Path, audio, sample_rate: int) -> None:
    import torch

    audio = audio.detach().cpu()
    if audio.dim() == 3:
        audio = audio.squeeze(0)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    audio = audio.clamp(-1.0, 1.0)
    interleaved = audio.transpose(0, 1).contiguous().view(-1)
    pcm = (interleaved * 32767.0).round().to(torch.int16).numpy().tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(int(audio.shape[0]))
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm)


def main() -> None:
    root = Path(__file__).resolve().parent
    maybe_reexec_in_task_venv(root)
    os.environ.setdefault("TORCH_HOME", str(root / "torch_home"))

    import torch
    import torchaudio
    from encodec import EncodecModel

    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)

    torch.manual_seed(0)
    torch.set_num_threads(2)

    sample_rate = 24_000
    duration_s = 1.0
    samples = int(sample_rate * duration_s)
    time = torch.arange(samples, dtype=torch.float32) / sample_rate
    wav = (0.20 * torch.sin(2 * math.pi * 440.0 * time)).unsqueeze(0).unsqueeze(0)

    input_wav = artifacts / "synthetic_input.wav"
    write_pcm16_wav(input_wav, wav, sample_rate)

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(1.5)
    model.eval()

    with torch.no_grad():
        encoded_frames = model.encode(wav)
        decoded = model.decode(encoded_frames)

    decoded = decoded.detach().cpu().clamp(-1.0, 1.0)
    reconstructed_wav = artifacts / "expected_artifact.wav"
    write_pcm16_wav(reconstructed_wav, decoded, model.sample_rate)

    codes = torch.cat([frame[0].detach().cpu() for frame in encoded_frames], dim=-1)
    common_len = min(wav.shape[-1], decoded.shape[-1])
    mse = torch.mean((wav[..., :common_len].detach().cpu() - decoded[..., :common_len]) ** 2).item()

    summary = {
        "channels": int(model.channels),
        "codes_shape": list(codes.shape),
        "decoded_shape": list(decoded.shape),
        "encoded_frame_count": len(encoded_frames),
        "input_duration_s": duration_s,
        "input_wav": "artifacts/synthetic_input.wav",
        "model": "encodec_model_24khz",
        "reconstructed_wav": "artifacts/expected_artifact.wav",
        "reconstruction_mse": mse,
        "repo_commit": REPO_COMMIT,
        "sample_rate": int(model.sample_rate),
        "success": bool(
            len(encoded_frames) >= 1
            and codes.numel() > 0
            and decoded.numel() > 0
            and math.isfinite(mse)
            and mse < 0.5
        ),
        "target_bandwidth_kbps": 1.5,
        "task_id": TASK_ID,
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__,
    }

    (artifacts / "expected_artifact.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "expected_output.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["success"]:
        raise SystemExit("semantic verification failed")


if __name__ == "__main__":
    main()
