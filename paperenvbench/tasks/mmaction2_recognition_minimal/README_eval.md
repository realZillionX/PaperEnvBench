# MMACTION2 Recognition Minimal

This task verifies a CPU-only MMACTION2 action-recognition smoke on the Inspire CPU notebook `paper-repro-ci-prep`. The gold path pins MMACTION2 at `a5a167dff2399e2d182a60332325f9c0d4663517`, generates a synthetic moving-shape video, executes the official TSN config through `mmengine.Config`, decodes the video with the MMACTION2 test pipeline, builds `Recognizer2D(ResNet + TSNHead)` from the registry, and writes `artifacts/expected_artifact.json`.

Success level is `L4_fallback`. Python 3.12 on the CPU notebook has no installable native `mmcv==2.1.0` wheel from PyPI, so the reproducible path uses `mmcv-lite==2.1.0`. The verified model path does not execute native `mmcv.ops`; the artifact records the native boundary and the pipeline / recognizer paths that did run.

Verified remote commands:

```bash
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/video/mmaction2_recognition_minimal "bash gold_install.sh > logs/gold_install.log 2>&1"
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/video/mmaction2_recognition_minimal "./.venv/bin/python verify.py --repo-dir repo --output-dir artifacts > logs/gold_verify.log 2>&1"
```

Local check-only validation:

```bash
python verify.py --check-only
```

Required artifacts:

- `artifacts/expected_artifact.json`: structured summary of package versions, config, pipeline, model, prediction scores, feature statistics, and semantic checks.
- `artifacts/synthetic_action.mp4`: deterministic synthetic video decoded by the official MMACTION2 pipeline.
