# OpenMMLab、Detectron2 与 GroundingDINO 原生环境依赖切片

本文档记录 5 个检测、分割和视觉 grounding 任务的原生环境依赖切片。这里的 probe 不负责安装大依赖、下载 checkpoint 或跑完整模型；它们负责给出当前环境是否具备任务真实路线所需依赖的结构化证据，并在依赖未安装时返回可复查 blocker。

## Probe 命令

OpenMMLab 任务：

```bash
python3 tools/paper_repo_env/probes/openmmlab_native_probe.py --task all --json
python3 tools/paper_repo_env/probes/openmmlab_native_probe.py --task mmdetection_fasterrcnn_minimal --repo-dir /path/to/mmdetection --require-repo --output openmmlab_probe.json --json
```

Detectron2 / GroundingDINO 任务：

```bash
python3 tools/paper_repo_env/probes/detectron_grounding_probe.py --task all --json
python3 tools/paper_repo_env/probes/detectron_grounding_probe.py --task groundingdino_phrase_grounding_minimal --repo-dir /path/to/GroundingDINO --checkpoint-path /path/to/groundingdino_swint_ogc.pth --json
```

默认退出码为 `0`，即使环境被判定为 `blocked` 也会输出 JSON。需要让 CI 在 blocker 出现时失败时，加 `--strict`。

## 任务依赖矩阵

### `detectron2_maskrcnn_minimal`

- 真实原生依赖：CUDA 版 `torch` / `torchvision`、可用 NVIDIA driver、`gcc` / `g++`、`ninja`、Python headers、从源码编译出的 `detectron2._C`，以及 official model zoo checkpoint。
- 推荐 probe：`detectron_grounding_probe.py --task detectron2_maskrcnn_minimal`。若有源码树，加 `--repo-dir` 和 `--require-repo`。
- 成功证据：`torch.cuda.is_available()` 为 true；`detectron2._C` 可导入；`detectron2.layers.nms` 能在 CUDA tensor 上执行；源码树存在 `detectron2/__init__.py`；完整任务还需要加载 `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml` 和 `model_final_a54504.pkl` 并生成 mask / overlay artifacts。
- 常见 blocker：非 CUDA torch wheel、`CUDA_HOME` / `nvcc` 缺失、Python headers 缺失、编译器版本不匹配、Detectron2 编译时用了错误的 torch ABI、只完成 import 但没有加载 checkpoint。

### `mmdetection_fasterrcnn_minimal`

- 真实原生依赖：CUDA 版 `torch` / `torchvision`、`mmengine`、完整 `mmcv` 原生扩展而不是 `mmcv-lite`、`mmcv._ext`、`mmcv.ops.RoIAlign`、`mmcv.ops.nms`、固定 commit 的 MMDetection 源码和 Faster R-CNN config。
- 推荐 probe：`openmmlab_native_probe.py --task mmdetection_fasterrcnn_minimal`。有源码树时使用 `--repo-dir /path/to/mmdetection --require-repo`。
- 成功证据：`mmcv._ext` 可导入；`RoIAlign` 与 `nms` 可导入；`nms` 能在 CUDA tensor 上执行；`configs/_base_/models/faster-rcnn_r50_fpn.py` 能通过 `mmengine.Config` 加载；完整任务应进一步跑 Faster R-CNN inference 或记录可解释的 model / output evidence。
- 常见 blocker：Python 3.12 无匹配 `mmcv` wheel、误装 `mmcv-lite`、`mmcv.ops` 报 `No module named 'mmcv._ext'`、torch CUDA 与 `mmcv` wheel 的 `cuXXX` / torch 版本矩阵不匹配、源码路径只在 `PYTHONPATH` 中但没有固定 commit。

### `mmsegmentation_segformer_minimal`

- 真实原生依赖：CUDA 版 `torch`、完整 `mmcv` 原生扩展、`mmengine`、`mmsegmentation`、固定 commit 的 MMSegmentation 源码和 SegFormer config。
- 推荐 probe：`openmmlab_native_probe.py --task mmsegmentation_segformer_minimal`。有源码树时使用 `--repo-dir /path/to/mmsegmentation --require-repo`。
- 成功证据：不需要注入 `mmcv.ops` stub；`mmcv._ext` 可导入；`configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py` 能加载；完整任务应在 CUDA 上构建 `MixVisionTransformer` + `SegformerHead` 并生成有限、非退化 logits 与 mask artifact。
- 常见 blocker：`mmcv-lite` 导致注册阶段请求的 `sigmoid_focal_loss`、`CrissCrossAttention`、`PSAMask` 缺失；`SyncBN` 单卡 / CPU 配置处理错误；预训练权重下载意外触发；OpenMMLab 版本矩阵不一致。

### `mmaction2_recognition_minimal`

- 真实原生依赖：CUDA 版 `torch`、完整 `mmcv` 原生扩展、`mmengine`、`mmaction2`、视频解码依赖 `decord` 或等价后端、固定 commit 的 MMACTION2 源码和 TSN config。
- 推荐 probe：`openmmlab_native_probe.py --task mmaction2_recognition_minimal`。有源码树时使用 `--repo-dir /path/to/mmaction2 --require-repo`。
- 成功证据：`decord` 可导入；`mmcv._ext` 和基础 native op 可用；`configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py` 能加载；完整任务应解码视频、通过官方 pipeline 构建 `Recognizer2D(ResNet + TSNHead)`，并输出有限归一化 scores。
- 常见 blocker：视频 codec / `decord` wheel 缺失、误装 `mmcv-lite`、torch / `mmcv` CUDA 矩阵不匹配、Kinetics checkpoint 下载或类别数修改处理不清、`opencv-python` 与 headless wheel 冲突。

### `groundingdino_phrase_grounding_minimal`

- 真实原生依赖：CUDA 版 `torch` / `torchvision`、`transformers`、`timm`、`supervision`、`pycocotools`、GroundingDINO editable install、`groundingdino._C` 自定义 C++ / CUDA op、官方 `groundingdino_swint_ogc.pth` checkpoint。
- 推荐 probe：`detectron_grounding_probe.py --task groundingdino_phrase_grounding_minimal`。需要验证 checkpoint 时，加 `--checkpoint-path /path/to/groundingdino_swint_ogc.pth`。
- 成功证据：`groundingdino._C` 可导入；`preprocess_caption` 可用；checkpoint 大小为 `693997677` bytes 且 SHA256 为 `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799`；完整任务应构建 GroundingDINO SwinT、加载 checkpoint、对合成图像做 forward，并输出 phrase、box 和 annotated PNG。
- 常见 blocker：`CUDA_HOME` / `nvcc` 缺失导致自定义 op 未编译；`setup.py` 在 build isolation 中找不到 torch；checkpoint 下载不完整；`transformers` 版本漂移；`torchvision.ops.box_convert` 与 torch 版本不匹配。

## JSON 判读

两个 probe 都输出顶层 `status` 和每个任务的 `status`。`pass` 表示当前环境具备该切片的原生依赖证据；`blocked` 表示至少一个 blocker 存在。`blockers[].code` 用于聚合统计，`blockers[].message` 保留原始错误摘要，便于把失败归因到 CUDA runtime、torch wheel、native extension、源码树或 checkpoint。

这些 probe 是环境依赖 suite 证据，不替代任务 verifier。最终 L4 仍应由各任务 `verify.py` 对 attempt artifacts 做检查。
