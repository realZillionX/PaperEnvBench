# mae_reconstruction_minimal

This task checks whether an agent can reproduce a minimal CPU MAE reconstruction workflow from the pinned `facebookresearch/mae` repository.

The gold path uses commit `efb2a8062c206524e35e47d04501ed4f544c0ae8`, creates an isolated Python venv, installs a Python 3.12-compatible PyTorch stack, and executes the repository's `models_mae` encoder, decoder, loss, and `unpatchify` reconstruction path on a deterministic synthetic image.

L4 does not require ImageNet, distributed pretraining, fine-tuning, or downloading the public MAE checkpoints. The verifier constructs `mae_vit_base_patch16_dec512d8b(img_size=32)` with initialized weights, applies compatibility shims for current Python dependencies, masks 2 of 4 image patches, and writes a reconstruction grid PNG plus JSON summary. Import success alone is not sufficient.

Validated remote run:

```bash
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/image/mae_reconstruction_minimal "./gold_install.sh"
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/image/mae_reconstruction_minimal "venv/bin/python verify.py"
```
