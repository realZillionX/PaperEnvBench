# videomae_classification_minimal

This task checks whether an agent can reproduce a minimal CPU VideoMAE classification workflow from the pinned `MCG-NJU/VideoMAE` repository.

The gold path uses commit `14ef8d856287c94ef1f985fe30f958eb4ec2c55d`, creates an isolated Python venv, installs a Python 3.12-compatible PyTorch stack, and executes the repository's `modeling_finetune.vit_small_patch16_224` classifier on a deterministic synthetic video tensor.

L4 is marked as fallback because the upstream model zoo publishes finetuned classification checkpoints through Google Drive links and the full training/evaluation path depends on large video datasets. The verifier still exercises the real pinned repository model path: `PatchEmbed` `Conv3d` tubelet embedding, transformer blocks, mean pooling, and classification head. Import success alone is not sufficient.

Validated remote run:

```bash
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/video/videomae_classification_minimal "./gold_install.sh"
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/video/videomae_classification_minimal "venv/bin/python verify.py"
```
