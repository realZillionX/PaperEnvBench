#!/usr/bin/env python3
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
