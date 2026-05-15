# timesformer_video_transformer_minimal

该任务验证 agent 是否能在启智 CPU Notebook 中安装并使用 facebookresearch/TimeSformer，固定上游 `a5ef29a7b7264baff199a30b3306ac27de901133`，对 deterministic synthetic video clip 执行 TimeSformer divided space-time attention smoke。

L4 fallback 成功要求：使用 pinned repository 的 `timesformer.models.vit.VisionTransformer` 路径；输入 clip 形状为 `[1, 3, 4, 32, 32]`；模型必须包含并执行 `divided_space_time` 分支、`Block.temporal_attn` 和 `Attention`；输出 logits 形状为 `[1, 7]`；生成 `expected_artifact.json`，并通过 artifact 中的 semantic checks。

该任务不要求下载 README 中的 Dropbox Kinetics checkpoint。原始 README 面向 Python 3.7 时代环境，当前启智 CPU Notebook 为 Python 3.12；gold 路径以 L4 fallback 记录两个兼容 shim：`torch._six.container_abcs` 和 `torch.nn.modules.linear._LinearWithBias`。shim 只在 verifier 进程内注入，不修改上游源码。

远端 gold 运行命令：

```bash
cd "/inspire/ssd/project/embodied-multimodality/tongjingqi-CZXS25110029/Paper Reproduction/runs/paperenvbench/video/timesformer_video_transformer_minimal"
./gold_install.sh > logs/gold_install.log 2>&1
venv/bin/python verify.py --repo-dir repo --output-dir artifacts --json > logs/gold_verify.log 2>&1
```

本地只读 artifact 检查命令：

```bash
python verify.py --output-dir artifacts --check-only --json
```
