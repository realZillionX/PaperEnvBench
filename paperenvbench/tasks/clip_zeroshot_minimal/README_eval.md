# clip_zeroshot_minimal

该任务验证 agent 是否能安装 CLIP 的 Python 包和 PyTorch 依赖矩阵、下载 `ViT-B/32` 权重，并对单张本地图片执行零样本图文匹配。L4 成功要求输出 `label_probs.json`，并满足概率向量合法、标签排序可解释的语义检查。
