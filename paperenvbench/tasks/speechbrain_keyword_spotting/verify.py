#!/usr/bin/env python3


# PaperEnvBench artifact-only validation path. This exits before runtime imports
# when --check-only is requested, so the standalone benchmark repo can verify
# gold task packages without vendoring full upstream checkouts or weights.
import argparse as _peb_argparse
import hashlib as _peb_hashlib
import json as _peb_json
import pathlib as _peb_pathlib
import sys as _peb_sys

_PEB_TASK_ID = "speechbrain_keyword_spotting"
_PEB_EXPECTED_ARTIFACT_SHA256 = "f57edf0139e1b90a6eec3ea94159eb36a7c514437044691abca4b673debfab32"
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
import pathlib
import random
import sys

import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import FetchConfig

TASK_ID = "speechbrain_keyword_spotting"
REPO_URL = "https://github.com/speechbrain/speechbrain"
COMMIT = "8a89ebad72af734b75bbd37565ae96a6819e146b"
MODEL_SOURCE = "speechbrain/google_speech_command_xvector"
MODEL_REVISION = "b0cec0fb42423936ca0da2724ce52d82eb807e20"
SEED = 20260515

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

run_dir = pathlib.Path(__file__).resolve().parent
artifacts_dir = run_dir / "artifacts"
artifacts_dir.mkdir(parents=True, exist_ok=True)
expected_artifact = artifacts_dir / "expected_artifact.json"
audio_path = artifacts_dir / "expected_artifact.wav"

sample_rate = 16000
duration_sec = 1.0
t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
wave = (0.15 * np.sin(2 * math.pi * 440.0 * t)).astype(np.float32)
sf.write(audio_path, wave, sample_rate)

model_dir = artifacts_dir / "pretrained_google_speech_command_xvector"
classifier = EncoderClassifier.from_hparams(
    source=MODEL_SOURCE,
    savedir=str(model_dir),
    run_opts={"device": "cpu"},
    fetch_config=FetchConfig(revision=MODEL_REVISION, allow_updates=True),
)

with torch.no_grad():
    out_prob, score, index, text_lab = classifier.classify_file(str(audio_path))

labels = classifier.hparams.label_encoder.ind2lab
num_classes = len(labels)
predicted_index = int(index.item() if hasattr(index, "item") else index)
predicted_label = text_lab[0] if isinstance(text_lab, (list, tuple)) else str(text_lab)
score_value = float(score.item() if hasattr(score, "item") else score)
prob_shape = list(out_prob.shape) if hasattr(out_prob, "shape") else []

result = {
    "task_id": TASK_ID,
    "repo_url": REPO_URL,
    "commit": COMMIT,
    "model_source": MODEL_SOURCE,
    "model_revision": MODEL_REVISION,
    "device": "cpu",
    "seed": SEED,
    "audio": {
        "path": "artifacts/expected_artifact.wav",
        "sample_rate": sample_rate,
        "duration_sec": duration_sec,
        "kind": "synthetic_440hz_sine",
    },
    "prediction": {
        "label": predicted_label,
        "index": predicted_index,
        "score": score_value,
        "prob_shape": prob_shape,
        "num_classes": num_classes,
    },
    "versions": {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    },
    "checks": {
        "model_loaded": model_dir.exists(),
        "audio_exists": audio_path.exists(),
        "num_classes_positive": num_classes > 0,
        "prediction_label_nonempty": bool(predicted_label),
        "score_finite": math.isfinite(score_value),
    },
}

if not all(result["checks"].values()):
    raise SystemExit(json.dumps(result, indent=2, ensure_ascii=False))

expected_artifact.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
