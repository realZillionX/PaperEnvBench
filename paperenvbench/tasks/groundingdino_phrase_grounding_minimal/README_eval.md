# groundingdino_phrase_grounding_minimal

This task checks whether an agent can reproduce a minimal CPU phrase-grounding path from the pinned `IDEA-Research/GroundingDINO` repository.

The gold path fixes the repository to commit `856dde20aee659246248e20734ef9ba5214f5e44`, creates an isolated Python venv, installs a CPU PyTorch stack, installs the upstream package in editable mode, downloads the public `groundingdino_swint_ogc.pth` checkpoint, and validates a deterministic synthetic image with the caption `red square. blue circle. green strip.`.

Two installation details are part of the task. First, non-torch packages are installed from PyPI because the observed SII internal PyPI mirror returned a stale `requests` wheel URL. Second, editable install uses `--no-build-isolation` because upstream `setup.py` imports `torch` during metadata generation. On the CPU notebook the custom C++/CUDA op is not built; the repository emits the CPU-only warning and then uses its PyTorch multi-scale deformable attention fallback.

L4 requires more than import success. The verifier loads the official SwinT checkpoint, builds the repository `GroundingDINO` model, runs a CPU forward pass, extracts phrases with `get_phrases_from_posmap`, converts predicted boxes into image coordinates, writes an annotated image, and checks the summary artifact.

Validated remote run:

```bash
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/detection/groundingdino_phrase_grounding_minimal "bash gold_install.sh > logs/gold_install.log 2>&1"
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/detection/groundingdino_phrase_grounding_minimal "venv/bin/python verify.py --repo-dir repo --checkpoint-path models/groundingdino_swint_ogc.pth --output-dir artifacts > logs/gold_verify.log 2>&1"
```
