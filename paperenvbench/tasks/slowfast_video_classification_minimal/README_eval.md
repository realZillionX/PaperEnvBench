# slowfast_video_classification_minimal

This task checks whether an agent can reproduce a minimal CPU SlowFast video-classification path from the pinned `facebookresearch/SlowFast` repository.

The pinned upstream commit is `287ec0076846560f44a9327e931a5a2360240533`. The gold route is `L4_fallback`: it loads the official `configs/Kinetics/SLOWFAST_8x8_R50.yaml`, executes the repository `uniform_crop` transform, builds a reduced CPU SlowFast model through `slowfast.models.build_model`, creates the slow and fast pathway tensors, and runs one deterministic CPU forward pass on a synthetic clip.

The fallback is explicit because native import of the current pinned repo on Python 3.12 with public `pytorchvideo==0.1.5` fails before classification inference: SlowFast expects distributed helper symbols that are not present in the packaged PyTorchVideo release, and `slowfast.models.head_helper` imports `detectron2.layers.ROIAlign` even when `DETECTION.ENABLE=False`. The verifier isolates compatibility shims for those unused import surfaces and does not modify the pinned repository source.

Validated remote run:

```bash
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/video/slowfast_video_classification_minimal "bash gold_install.sh > logs/gold_install.log 2>&1"
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/video/slowfast_video_classification_minimal "venv/bin/python verify.py --generate > logs/gold_verify.log 2>&1"
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/video/slowfast_video_classification_minimal "venv/bin/python verify.py > logs/gold_verify_check.log 2>&1"
```

Expected success level: `L4_fallback`.

Key artifact contract:

- `artifacts/expected_artifact.json` records repo commit, dependency shims, official code paths, input tensor shapes, logits, probabilities, and semantic checks.
- `artifacts/expected_clip_preview.ppm` is a deterministic preview of the cropped synthetic clip used for inference.
- `logs/native_import_probe.log` records the native import failure before compatibility shims are applied.
