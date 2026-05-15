# detr_object_detection_minimal

该任务验证 agent 是否能在启智 CPU Notebook 中安装并使用 facebookresearch/detr，固定上游 `29901c51d7fe8712168b8d0d64351170bc0f83e0`，加载官方 `detr-r50-e632da11.pth` checkpoint，并对 COCO `000000039769.jpg` 执行单图 object detection smoke。

L4 成功要求：使用 pinned repository 的 `hubconf.detr_resnet50(pretrained=True, return_postprocessor=True)` 路径；checkpoint SHA-256 匹配；生成 `expected_artifact.json`、`expected_input.jpg` 和 `expected_detection_overlay.jpg`；artifact 中的 raw output shape 为 `[1, 100, 92]` 和 `[1, 100, 4]`；top-5 中至少有两个高置信 `cat` 检测。

远端 gold 运行命令：

```bash
cd "/inspire/ssd/project/embodied-multimodality/tongjingqi-CZXS25110029/Paper Reproduction/runs/paperenvbench/detection/detr_object_detection_minimal"
./gold_install.sh > logs/gold_install.log 2>&1
venv/bin/python verify.py --repo-dir repo --output-dir artifacts --checkpoint-dir checkpoints --json > logs/gold_verify.log 2>&1
```

本地只读 artifact 检查命令：

```bash
python verify.py --output-dir artifacts --check-only --json
```
