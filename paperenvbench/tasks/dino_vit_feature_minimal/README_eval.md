# dino_vit_feature_minimal

该任务验证 agent 是否能在 CPU Notebook 中为 `facebookresearch/dino` 建立最小可复现环境，固定到 `7c446df5b9f45747937fb0d72314eb9f7b66930a`，下载公开 `dino_deitsmall16_pretrain.pth` backbone checkpoint，并对确定性合成 RGB 图像运行 ViT-S/16 feature extraction。

L4 成功要求 strict checkpoint load 无 missing / unexpected keys，输出 `1 x 384` feature summary JSON，feature hash 与统计量匹配 gold，并保留非空 synthetic image artifact。单纯 `import vision_transformer`、查看 README 或调用 help 不构成 L4。
