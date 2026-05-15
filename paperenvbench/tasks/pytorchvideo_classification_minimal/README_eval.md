# pytorchvideo_classification_minimal

This task validates a minimal CPU video-classification path through the pinned `facebookresearch/pytorchvideo` repository. The gold run fixes PyTorchVideo at `f3142bb05cdb56af0704ab6f0adfb0c7bbafe4a0`, creates an isolated Python venv, installs CPU PyTorch, installs the repository in editable mode, and executes the local torch hub `slow_r50` model builder.

The success level is `L4_fallback`. The CPU gold path does not download Kinetics checkpoints and does not require an external video file. Instead, `verify.py` creates a deterministic synthetic RGB video clip with shape `[1, 3, 8, 224, 224]`, loads `slow_r50(pretrained=False)` from the pinned local repository via `torch.hub.load(..., source="local")`, runs a real CPU forward pass, and validates the logits artifact. Import-only success or using a separately installed model implementation is not sufficient.

Gold run directory:

```bash
paper-repro:runs/paperenvbench/video/pytorchvideo_classification_minimal
```

Remote commands used:

```bash
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/video/pytorchvideo_classification_minimal "bash gold_install.sh > logs/gold_install.log 2>&1"
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/video/pytorchvideo_classification_minimal "venv/bin/python verify.py > logs/gold_verify.log 2>&1"
```

Expected artifacts:

- `artifacts/expected_artifact.json`: structured summary of package versions, fallback boundary, model evidence, video tensor metadata, logits statistics, and semantic checks.
- `artifacts/expected_clip.pt`: deterministic synthetic input video tensor.
- `artifacts/expected_frame.ppm`: representative first frame from the synthetic clip.
- `artifacts/expected_logits.pt`: raw classification logits tensor.

Run the verifier with:

```bash
venv/bin/python verify.py
```
