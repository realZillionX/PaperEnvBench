# clip_zeroshot_minimal

该任务验证 agent 是否能安装 CLIP 的 Python 包和 PyTorch / CUDA 依赖矩阵、下载 `ViT-B/32` 权重，并跑通上游 README 中的最小图文分类示例。该任务按原始实训参考文档的 CLIP 成功标准定义：必须执行 README zero-shot image classification route，并输出类别概率或相似度。

L4 成功要求使用固定 commit 的 `openai/CLIP`、`clip.load("ViT-B/32")`、仓库内 `CLIP.png` 和 README 标签 `["a diagram", "a dog", "a cat"]` 完成一次真实 CUDA forward。输出 artifact 必须记录 checkpoint SHA、输入图片 SHA、labels、logits、probabilities、top label、CUDA 设备和 `clip.load` / `clip.tokenize` / preprocess / model forward 证据；只输出任意概率字典或合成图片不算完整复现。
