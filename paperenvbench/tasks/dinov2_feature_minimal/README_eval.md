# dinov2_feature_minimal

This task checks whether an agent can build a minimal CPU DINOv2 feature extraction environment from the pinned `facebookresearch/dinov2` repository.

The upstream requirements are CUDA-oriented and include `xformers` and `cuml-cu11`, so the gold route does not install `requirements.txt` verbatim. It creates an isolated venv, installs a Python 3.12-compatible CPU `torch` / `torchvision` pair, installs the pinned repository with `--no-deps`, and sets `XFORMERS_DISABLED=1`.

L4 does not require downloading pretrained DINOv2 checkpoints. The verifier uses `dinov2.hub.backbones.dinov2_vits14(pretrained=False)`, generates one deterministic synthetic RGB image, runs `forward_features` on CPU, and validates embedding shape, finite values, positive norm, and repository commit. A pretrained checkpoint path may be used by agents only if the cache path and any network failure are recorded.

Validated remote run:

```bash
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/image/dinov2_feature_minimal "./gold_install.sh"
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/image/dinov2_feature_minimal "venv/bin/python verify.py"
```
