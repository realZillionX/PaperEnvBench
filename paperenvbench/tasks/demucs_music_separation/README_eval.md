# demucs_music_separation

This task checks whether an agent can reproduce a minimal CPU Demucs music source separation workflow from the pinned `facebookresearch/demucs` repository.

The gold path uses the repository at commit `e976d93ecc3865e5757426930257e200846a520a`, installs `ffmpeg`, creates an isolated Python venv, installs a Python 3.12-compatible `torch` / `torchaudio` pair, and installs Demucs with `--no-deps` to bypass the stale upstream `torchaudio<2.1` constraint.

L4 does not require downloading the default public `htdemucs` checkpoint. The verifier uses the repository's built-in `demucs_unittest` model, generates a 0.25-second synthetic stereo WAV, runs real separation on CPU, and validates the resulting `vocals.wav` artifact. CLI help or import success alone is not sufficient.

Validated remote run:

```bash
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/audio/demucs_music_separation "./gold_install.sh"
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/audio/demucs_music_separation "venv/bin/python verify.py"
```

