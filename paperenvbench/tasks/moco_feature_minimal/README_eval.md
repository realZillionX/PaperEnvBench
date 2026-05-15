# moco_feature_minimal

This task checks whether an agent can reproduce a minimal CPU MoCo representation-learning workflow from the pinned `facebookresearch/moco` repository.

The upstream `main` branch currently points to commit `8976944da9c6b94cbd9158d7ebe50912aef807ef`, whose tree is empty. The gold task therefore pins `7397dfe146c7ca6bbb58e9c382498069178ba764`, the MIT relicense commit that still contains the original MoCo files.

The gold route creates an isolated Python venv, installs the matched CPU wheel pair `torch==2.8.0` and `torchvision==0.23.0`, checks out the pinned repository, and runs a deterministic synthetic RGB tensor through `moco.builder.MoCo(...).encoder_q` on CPU. This intentionally avoids the upstream training `forward` path, which assumes DistributedDataParallel and hard-codes CUDA labels.

L4 success does not require ImageNet, Docker, GPU, or pretrained checkpoint download. Import-only checks are insufficient: the verifier must produce `artifacts/expected_artifact.json` and `artifacts/verification_result.json` with finite feature values, shape `[1, 128]`, normalized feature norm near `1.0`, and a deterministic feature checksum.

Validated remote run:

```bash
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/image/moco_feature_minimal "./gold_install.sh"
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/image/moco_feature_minimal "venv/bin/python verify.py"
```
