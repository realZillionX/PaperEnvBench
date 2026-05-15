# detectron2_maskrcnn_minimal

该任务验证 agent 是否能在 CPU Notebook 上复现 `facebookresearch/detectron2` 的最小 Mask R-CNN 路线。成功路径需要固定源码 commit、编译 Detectron2 native extension、加载官方 model zoo checkpoint，并在 CPU 上对合成图像执行一次实例分割推理，生成可检查的 JSON 与图像 artifacts。

固定版本为 `e0ec4e189d438848521aee7926f9900e114229f5`。远端 gold run 位于 `paper-repro:runs/paperenvbench/detection/detectron2_maskrcnn_minimal`，Notebook 为 `paper-repro-ci-prep`。

实际安装边界：

- 基础镜像缺少 `gcc` / `g++`，需要 `apt-get install build-essential`。
- Detectron2 `setup.py` 会导入 `torch`，因此 editable install 必须在已安装 CPU `torch` 后使用 `--no-build-isolation`。
- C++ extension 编译需要 `Python.h`，因此需要 `python3.12-dev`。
- SII 内部 PyPI 对 `fonttools 4.63.0` 返回 404，gold path 改用公共 PyPI 安装 `matplotlib` 与 `iopath==0.1.9`，随后用 `--no-deps` 编译 Detectron2。

验证使用官方 `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml` 配置和 checkpoint `model_final_a54504.pkl`。为了让 CPU synthetic image 稳定产生 COCO-style 输出，gold verifier 将 test score threshold 设为 `0.0`，只检查 output contract、数值有限性、mask artifact 和 checkpoint load，不把低分预测解释为真实语义识别。

Validated remote run summary（完整 stdout / stderr 见 `logs/`）：

```bash
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/detection/detectron2_maskrcnn_minimal "set -euxo pipefail; ...; FORCE_CUDA=0 MAX_JOBS=2 CC=gcc CXX=g++ .venv/bin/python -m pip install -e repo --no-build-isolation --no-deps"
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/detection/detectron2_maskrcnn_minimal "set -euxo pipefail; .venv/bin/python - <<'PY' ... official model_zoo Mask R-CNN checkpoint inference ... PY"
```

本地 `verify.py` 只检查已经生成并迁回的 artifacts，包含 `expected_artifact.json`、`expected_input.png`、`expected_mask.png` 和 `expected_overlay.png`。单纯 import `detectron2` 不构成本任务通过。
