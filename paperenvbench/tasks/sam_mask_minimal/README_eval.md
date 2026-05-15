# sam_mask_minimal

该任务验证 agent 是否能安装 Segment Anything 的基础依赖、显式缓存 `vit_b` checkpoint，并在 CPU 上使用缩小图片生成 mask artifact。L4 成功要求 checkpoint load、mask 文件落盘、`summary.json` 可解析，并通过 mask 面积范围等轻量语义检查。
