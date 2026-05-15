# gaussian_splatting_scene_minimal

该任务验证 agent 是否能定位 3D Gaussian Splatting 官方仓库的真实训练 / 渲染路径、处理 CUDA rasterizer submodule 和 PyTorch / CUDA 矩阵边界，并产出可验证的 scene artifact。

完整路线是：使用 `train.py -s <scene> -m <model>` 从 COLMAP 或 NeRF Synthetic 输入优化 `GaussianModel`，再用 `render.py -m <model>` 调用 `gaussian_renderer.render` 生成 novel-view rendering。由于官方路线依赖 `diff-gaussian-rasterization`、`simple-knn` 和 CUDA device，check-only gold 使用确定性 CPU fallback：记录真实 route，并验证一个最小 3D Gaussian scene 的 JSON / PLY artifact。

验收命令：

```bash
python paperenvbench/tasks/gaussian_splatting_scene_minimal/verify.py --check-only --json
python tools/paper_repo_env/validate_task_package.py --task gaussian_splatting_scene_minimal
```
