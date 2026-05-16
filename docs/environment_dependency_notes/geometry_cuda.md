# Geometry Native Dependency Notes

本文档记录 PaperEnvBench 的 3D / geometry 切片中，真实 upstream 路线涉及的 accelerator / native dependency 边界。这里的 probe 负责把“真实环境能否支撑原生路线”转成结构化证据和 blocker。

推荐统一入口：

```bash
python3 tools/paper_repo_env/probes/geometry_cuda_probe.py --json
python3 tools/paper_repo_env/probes/geometry_cuda_probe.py --json --output /tmp/geometry_cuda_probe.json
python3 tools/paper_repo_env/probes/geometry_cuda_probe.py --task openpcdet_pointcloud_minimal --repo-dir openpcdet_pointcloud_minimal=/path/to/OpenPCDet --json
```

加 `--strict` 时，只要任一选中任务处于 `blocked`，命令就返回非零状态码。默认不加 `--strict`，方便在任意开发环境中产出可读 blocker。

## gaussian_splatting_scene_minimal

真实依赖边界：

- CUDA-enabled PyTorch runtime。
- `submodules/diff-gaussian-rasterization` 编译出的 `diff_gaussian_rasterization`，提供 `GaussianRasterizationSettings` 和 `GaussianRasterizer`。
- `submodules/simple-knn` 编译出的 `simple_knn._C`，提供 `distCUDA2`。
- upstream full route 通过 `train.py`、`render.py` 和 `gaussian_renderer.render` 使用这些 CUDA kernel。

推荐 probe：

```bash
python3 tools/paper_repo_env/probes/geometry_cuda_probe.py --task gaussian_splatting_scene_minimal --json
```

成功证据：

- `runtime.torch.cuda_available == true` 且 `torch_cuda_smoke == true`。
- `diff_gaussian_rasterization` 可 import，且 rasterizer symbols 完整。
- `simple_knn._C` 可 import，且 `distCUDA2` symbol 存在。

常见 blocker：

- `gpu_runtime_missing`：`nvidia-smi` 看不到 GPU。
- `torch_cuda_unavailable`：装了非 CUDA PyTorch，或 driver / CUDA ABI 不匹配。
- `python_module_missing`：没有从 submodule 编译安装 `diff-gaussian-rasterization` 或 `simple-knn`。
- `python_module_symbol_missing`：包名可 import，但编译产物与当前 PyTorch / CUDA ABI 不匹配。

## openpcdet_pointcloud_minimal

真实依赖边界：

- CUDA-enabled PyTorch runtime。
- 通常需要 `nvcc`，因为 upstream `setup.py` 通过 `CUDAExtension` 编译 detector ops。
- OpenPCDet CUDA ops 包括 `iou3d_nms_cuda`、`roiaware_pool3d_cuda`、`roipoint_pool3d_cuda`、`pointnet2_stack_cuda`、`pointnet2_batch_cuda`、`bev_pool_ext` 和 `ingroup_inds_cuda`。
- 任务 artifact 若没有这些 detector ops 证据，不能被解释为 OpenPCDet CUDA ops 已成功编译。

推荐 probe：

```bash
python3 tools/paper_repo_env/probes/geometry_cuda_probe.py --task openpcdet_pointcloud_minimal --repo-dir openpcdet_pointcloud_minimal=/path/to/OpenPCDet --json
```

成功证据：

- `runtime.torch.cuda_available == true` 且 `torch_cuda_smoke == true`。
- `runtime.nvcc.available == true`。
- `setup_py.uses_cuda_extension == true`，并列出多个 declared CUDA extension。
- OpenPCDet ops 模块可 import，`success_evidence.imported_extension_count` 覆盖主要 CUDA ops。

常见 blocker：

- `cuda_toolkit_missing`：只有 CUDA wheel runtime，没有 `nvcc`，源码 extension 无法构建。
- `python_module_missing`：未执行 editable install，或 extension 编译失败。
- `torch_cuda_unavailable`：PyTorch 与 driver / CUDA wheel 不匹配。
- `openpcdet_setup_cudaextension_missing`：`--repo-dir` 指向错误 checkout，或 upstream 文件不是预期 OpenPCDet。

## nerfstudio_nerfacto_minimal

真实依赖边界：

- CUDA-enabled PyTorch runtime。
- `nerfacc`：occupancy grid、ray marching 和 volume rendering 相关 CUDA / native 路线。
- `gsplat`：Gaussian splatting CUDA kernel，Nerfstudio 新路线和 splat 系列方法会触发。
- `tiny-cuda-nn` 的 Python binding `tinycudann`：hash encoding / MLP 加速路线。
- `open3d`、`xatlas`、`pymeshlab`：export、mesh 和 geometry 辅助路线的 native 包；对 `nerfacto` 最小训练不一定是硬阻断，但会影响完整工具面。

推荐 probe：

```bash
python3 tools/paper_repo_env/probes/geometry_cuda_probe.py --task nerfstudio_nerfacto_minimal --json
```

成功证据：

- `runtime.torch.cuda_available == true` 且 `torch_cuda_smoke == true`。
- `nerfstudio`、`nerfacc`、`gsplat` 和 `tinycudann` 均可 import。
- `optional_missing` 为空时，说明 Open3D / xatlas / pymeshlab 这类 native geometry 辅助包也齐全。

常见 blocker：

- `python_module_missing`：`nerfacc`、`gsplat` 或 `tinycudann` 缺失，通常来自 Python / CUDA wheel matrix 不匹配。
- `torch_cuda_unavailable`：Nerfstudio 可 import，但 GPU route 无法执行。
- `gpu_runtime_missing`：Notebook / worker 未拿到 GPU。
- Open3D / xatlas / pymeshlab 缺失：最小 `nerfacto` 训练 route 可先继续，但 export / mesh route 应记录为能力缺口。

## open3d_pointcloud_minimal

真实依赖边界：

- `open3d` 编译 wheel。
- PointCloud geometry kernels：`open3d.geometry.PointCloud`、`KDTreeSearchParamKNN`、`estimate_normals`、`voxel_down_sample`。
- 本切片不要求 visualization window，也不要求 headless OpenGL / EGL。

推荐 probe：

```bash
python3 tools/paper_repo_env/probes/geometry_cuda_probe.py --task open3d_pointcloud_minimal --json
```

成功证据：

- `open3d` 可 import。
- probe 能构造 $5 \times 5$ tilted-plane point cloud。
- `estimate_normals` 产出 $25$ 个 finite normals。
- `voxel_down_sample(voxel_size=0.55)` 产出 $25$ 个点。
- `open3d_core_cuda_available` 只是附加信息，不作为 baseline 成功条件。

常见 blocker：

- `open3d_native_geometry_failed`：wheel 缺失、二进制 ABI 不匹配，或 native geometry op 执行失败。
- OpenGL / EGL / X11 缺失：不应阻断本 baseline；避免调用 visualization window 或 rendering route。
- 源码构建成本过高：优先使用 pinned wheel route，而不是在任务 worker 中临时完整 CMake build。
