# sam_mask_minimal

该任务验证 agent 是否能安装 Segment Anything 的基础依赖、显式缓存官方 `vit_b` checkpoint，并跑通最小预测脚本或 `scripts/amg.py` 的可验证入口。该任务按原始实训参考文档的 Segment Anything 成功标准定义：必须下载 checkpoint，并完成一次可审计 mask 预测。

L4 成功要求在 4090 / CUDA runtime 中加载 `sam_vit_b_01ec64.pth`，通过 `sam_model_registry["vit_b"]` 和 `SamPredictor.predict` 对固定输入图片完成真实 forward，生成非空 mask PNG 与 summary JSON。summary 必须记录 repo commit、checkpoint SHA、输入图片 SHA、mask SHA、prompt、score、mask pixels、mask shape、CUDA 设备和 torch CUDA 证据；只生成图片文件或只检查 JSON 字段不算完整复现。
