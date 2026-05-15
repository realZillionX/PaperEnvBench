# convnext_classification_minimal

This task checks whether an agent can reproduce a minimal CPU image-classification path from the pinned `facebookresearch/ConvNeXt` repository.

The gold path fixes the repository to commit `048efcea897d999aed302f2639b6270aedf8d4c8`, creates an isolated Python venv, installs a CPU PyTorch stack, and places the repository on `PYTHONPATH` because the upstream project has no `setup.py`, `pyproject.toml`, or `requirements.txt`. The upstream install notes recommend `timm==0.3.2`; the gold installer first tries that pin and falls back to a newer compatible `timm` only if the old pin cannot import against the available CPU PyTorch.

L4 does not require downloading the public ImageNet checkpoints. The verifier generates a deterministic synthetic RGB image, imports `models.convnext.ConvNeXt` from the pinned repository, builds a small ConvNeXt-shaped classifier, runs a real CPU forward pass, and validates the logits summary artifact. Import success or use of a library-provided ConvNeXt model is not sufficient.

Validated remote run:

```bash
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/image/convnext_classification_minimal "./gold_install.sh > logs/gold_install.log 2>&1"
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/image/convnext_classification_minimal "venv/bin/python verify.py > logs/gold_verify.log 2>&1"
```
