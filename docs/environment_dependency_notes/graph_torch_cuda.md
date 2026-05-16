# Graph / Torch CUDA / Audio-Video Dependency Notes

本文档覆盖 `pyg_gcn_minimal`、`dgl_graphsage_minimal`、`ogb_nodeprop_minimal`，以及音频、图像、视频任务共用的 torch CUDA 基础 smoke。它只描述环境依赖切片，不替代任务 verifier。

## Runtime Slice

推荐把本切片拆成 4 层探测：

- `torch_cuda_base`：`torch`、`torchvision`、`torchaudio` 同一 CUDA wheel 行，外加 `nvidia-smi`、CUDA tensor、`torchvision.ops`、`torchaudio.functional` smoke。
- `pyg_cuda_wheels`：PyG 核心包和可选扩展 wheel，重点检查 PyG wheel 页面是否匹配当前 `torch` 和 CUDA tag。
- `dgl_cuda_wheels`：DGL PyTorch backend wheel，重点检查 DGL wheel 页面是否匹配当前 `torch` major.minor 与 CUDA tag。
- `ogb_nodeprop`：OGB 评价器与数值依赖，通常不需要 CUDA wheel，但会被上游 PyG / DGL / torch 环境间接影响。

通用命令：

```bash
python3 tools/paper_repo_env/probes/graph_torch_cuda_probe.py --json --output /tmp/graph_torch_cuda_probe.json
```

在 GPU 节点上要求真实 CUDA 路径：

```bash
python3 tools/paper_repo_env/probes/graph_torch_cuda_probe.py --json --require-cuda --strict
```

只检查某一层：

```bash
python3 tools/paper_repo_env/probes/graph_torch_cuda_probe.py --group pyg --json
python3 tools/paper_repo_env/probes/graph_torch_cuda_probe.py --group dgl --json --require-cuda
```

## Wheel Matrix

当前推荐 CUDA wheel 证据锚点：

- PyTorch 官方 wheel 行：`torch==2.8.0`、`torchvision==0.23.0`、`torchaudio==2.8.0` 支持 `--index-url https://download.pytorch.org/whl/cu128`。参考：[PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)。
- PyG 官方 wheel 行：`torch-2.8.0+cu128` 对应 `https://data.pyg.org/whl/torch-2.8.0+cu128.html`，并覆盖 `pyg_lib`、`torch_scatter`、`torch_sparse`、`torch_cluster`、`torch_spline_conv` 等扩展。参考：[PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)。
- DGL CUDA wheel 行需要同时匹配 DGL 支持的 `torch` 行与 CUDA tag。DGL 社区当前常见稳定组合是 `torch-2.3/cu121`，即 `pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html`；这与 `torch==2.8.0+cu128` 不是同一行，不能混装后宣称 DGL CUDA 已验证。参考：[DGL start page](https://www.dgl.ai/pages/start.html) 和 DGL 官方讨论中的 per-torch wheel path 说明。

结论：本切片不能只问“有没有 CUDA”。要分别证明：

1. `torch.version.cuda` 与 `torch.__version__` 是 CUDA wheel，而不是 CPU wheel。
2. `torchvision`、`torchaudio` 与 `torch` 来自同一 PyTorch 版本行。
3. PyG 扩展来自与 `torch` / CUDA tag 匹配的 `data.pyg.org` 页面。
4. DGL 来自与 `torch` major.minor / CUDA tag 匹配的 DGL wheel 页面；若当前 DGL 仅覆盖旧 torch 行，则必须单独建 DGL 环境或记录该切片未通过。

## Task Groups

### PyG：`pyg_gcn_minimal`

真实依赖：

- 基础：`torch`。
- PyG：`torch_geometric`。
- 加速器 / 性能扩展：`pyg_lib`、`torch_scatter`、`torch_sparse`、`torch_cluster`、`torch_spline_conv`。图任务切片应把这些扩展作为 wheel matrix 风险记录。

推荐 probe：

```bash
python3 tools/paper_repo_env/probes/graph_torch_cuda_probe.py --group pyg --json --output /tmp/pyg_probe.json
```

成功证据：

- `probes.torch_cuda_base.cuda_available == true`（GPU 节点 strict 路径）。
- `probes.pyg_cuda_wheels.torch_geometric_version` 存在。
- `pyg_gcn_forward` 成功；GPU 路径下 `result.device == "cuda"`。
- 扩展缺失最多只能作为 warning，不能作为 PyG CUDA wheel 已验证的证据。

常见 blocker：

- `pyg_import_failed`：`torch_geometric` 未安装或被不兼容 `torch` 版本阻断。
- `pyg_optional_extensions_missing`：扩展 wheel 未装；基础 smoke 可能还能过，但 CUDA / 大图路径未被证明。
- `pyg_gcn_forward_failed`：常见于 `torch_sparse` / `pyg_lib` ABI 不匹配、不同 wheel 行混装、Python ABI 不匹配。

### DGL：`dgl_graphsage_minimal`

真实依赖：

- 基础：`torch`。
- DGL PyTorch backend：`dgl`，必要时还需要 `torchdata`。
- CUDA backend：必须安装 CUDA-enabled DGL wheel；仅 `torch.cuda.is_available()` 为 true 不代表 DGL 支持 CUDA graph device。

推荐 probe：

```bash
DGLBACKEND=pytorch python3 tools/paper_repo_env/probes/graph_torch_cuda_probe.py --group dgl --json --output /tmp/dgl_probe.json
```

成功证据：

- `probes.dgl_cuda_wheels.dgl_version` 存在。
- `probes.dgl_cuda_wheels.backend` 为 PyTorch backend。
- `dgl_graphsage_forward` 成功；GPU strict 路径下 `result.device == "cuda"`。

常见 blocker：

- `dgl_import_failed`：DGL 版本缺少 `torchdata`、GraphBolt native library 加载失败，或 wheel 与 `torch` ABI 不匹配。
- `dgl_cuda_backend_disabled`：DGL wheel 未启用 CUDA backend，调用 `.to("cuda")` 时出现 “Device API cuda is not enabled” 类错误。
- DGL wheel 行落后于 PyTorch wheel 行：例如 DGL CUDA 组合固定在 `torch-2.3/cu121`，而通用 torch CUDA 环境是 `torch-2.8/cu128`。这种情况应拆分环境，不要强行复用同一个 venv。

### OGB：`ogb_nodeprop_minimal`

真实依赖：

- `ogb`、`numpy`、`pandas`、`scikit-learn`、`scipy`、`torch`。
- OGB 评价器本身不要求 CUDA wheel；CUDA 风险主要来自上游 GNN 框架和 torch 版本。

推荐 probe：

```bash
python3 tools/paper_repo_env/probes/graph_torch_cuda_probe.py --group ogb --json --output /tmp/ogb_probe.json
```

成功证据：

- `probes.ogb_nodeprop.packages.ogb` 存在。
- `ogb_nodeprop_evaluator` 成功，并返回 `acc` 类 metric。
- probe 不下载数据集；它只证明 evaluator / numeric stack 可用。

常见 blocker：

- `ogb_import_failed`：OGB 或数值依赖缺失。
- `ogb_evaluator_failed`：OGB 版本与 `numpy` / `scikit-learn` 组合不兼容，或任务错误地把 dataset download 放进基础 probe。

### Torch CUDA Base：音频、图像、视频共用层

覆盖任务：

- 音频：`whisper_asr_minimal`、`encodec_audio_codec_minimal`。
- 图像 / 视觉语言：`clip_zeroshot_minimal`、`open_clip_zeroshot_minimal`、`sam_mask_minimal`。
- 视频：`pytorchvideo_classification_minimal`、`slowfast_video_classification_minimal`、`timesformer_video_transformer_minimal`、`videomae_classification_minimal`。

真实依赖：

- `torch`：CUDA tensor 与模型 checkpoint 加载的基础。
- `torchvision`：图像 / 视频 transforms、compiled ops，如 `torchvision.ops.nms`。
- `torchaudio`：音频 tensor、spectrogram / codec 辅助路径。
- 系统库：视频与音频任务常额外依赖 `ffmpeg`、`libsndfile`、OpenCV / decord / av 类解码库；这些不是 CUDA wheel，但会阻断 smoke。

推荐 probe：

```bash
python3 tools/paper_repo_env/probes/graph_torch_cuda_probe.py --group torch --json --require-cuda --output /tmp/torch_cuda_base_probe.json
```

成功证据：

- `nvidia_smi.available == true`。
- `torch_cuda_base.torch_cuda_compiled` 非空，且 GPU 节点上 `torch_cuda_base.cuda_available == true`。
- `torch_cuda_matmul` 成功。
- `torchvision_nms` 成功；GPU 节点上 `result.device == "cuda"`。
- `torchaudio_spectrogram` 成功；GPU 节点上 `result.device == "cuda"`。

常见 blocker：

- `torch_cuda_unavailable`：安装了非 CUDA torch wheel，或 NVIDIA driver / container device 不可见。
- `torchvision_ops_failed`：`torchvision` 版本行与 `torch` 不一致，compiled ops 未加载。
- `torchaudio_ops_failed`：`torchaudio` 版本行与 `torch` 不一致，或 native audio library linkage 出错。
- 系统解码器缺失：probe 通过但视频 / 音频任务仍失败，此时应补充 `ffmpeg -version`、`python -c "import av, cv2, decord"` 等任务级检查。

## Output Contract

`graph_torch_cuda_probe.py` 的 JSON 输出包含：

- `ok`：无 error blocker 时为 true。
- `summary`：被检查的 group、是否要求 CUDA、error / warning blocker 数量。
- `nvidia_smi`：GPU、driver、显存和 compute capability 探测。
- `probes`：按 dependency slice 分组的版本、import 和 smoke 结果。
- `blockers[]`：结构化 blocker，字段为 `code`、`scope`、`severity`、`message`、`evidence`、`recommendation`。

评测记录建议保存完整 JSON，而不是只复制终端摘要。CUDA 环境证据至少需要 `--require-cuda --strict` 的退出码和 `blockers[]` 为空；存在 CUDA blocker 时，不能把它写成 CUDA wheel matrix 已验证。
