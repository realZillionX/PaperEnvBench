# mmdetection_fasterrcnn_minimal

This task checks whether an agent can reproduce a minimal CPU MMDetection object-detection path from the pinned `open-mmlab/mmdetection` repository.

The pinned upstream commit is `cfd5d3a985b0249de009b67d04f37263e11cdf3d`. The gold route first verifies that the Faster R-CNN base config parses, then records why the native CPU path cannot run as a real Faster R-CNN inference path on `paper-repro-ci-prep`: full `mmcv` falls back to a source build and fails during wheel metadata preparation on Python 3.12, while the installed `mmcv-lite==2.1.0` package does not provide `mmcv._ext`, so `RoIAlign` and `nms` are unavailable.

The verified `L4_fallback` route still executes official MMDetection repository code. It imports the pinned `mmdet` package via `PYTHONPATH`, parses `configs/_base_/models/faster-rcnn_r50_fpn.py`, executes `mmdet/evaluation/functional/bbox_overlaps.py`, executes `mmdet/models/layers/matrix_nms.py`, and produces a deterministic synthetic detection artifact. Import success alone is not sufficient.

Validated remote run:

```bash
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/detection/mmdetection_fasterrcnn_minimal "bash gold_install.sh > logs/gold_install.log 2>&1"
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/detection/mmdetection_fasterrcnn_minimal "venv/bin/python verify.py > logs/gold_verify.log 2>&1"
```

Expected success level: `L4_fallback`.

Key artifact contract:

- `artifacts/expected_artifact.json` records package versions, commit, native blocker, IoU matrix, kept detections, and semantic checks.
- `artifacts/expected_detection.ppm` is the deterministic visualization side artifact.
- `logs/native_mmcv_dryrun_probe.log` records the native `mmcv` dry-run build failure.
- `logs/native_mmcv_ops_probe.log` records the missing `mmcv._ext` blocker for `RoIAlign` and `nms`.
