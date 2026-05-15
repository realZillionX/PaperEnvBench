# OpenPCDet Point Cloud Minimal

This task verifies a pinned OpenPCDet checkout through a deterministic PointPillar CPU fallback. The gold route reads the official `tools/cfgs/kitti_models/pointpillar.yaml`, applies the same point range and voxel geometry, executes OpenPCDet utility code for range filtering / voxel centers, and runs the official `PointPillarScatter` module on synthetic point cloud pillars.

Native OpenPCDet detector inference is not claimed in this CPU package: `setup.py` declares CUDA extensions for 3D IoU, ROI-aware pooling, pointnet2, BEV pool, and related kernels. The expected artifact records that boundary and requires the fallback artifact to preserve the pinned repo commit, config path, voxel route, BEV tensor shape, and preview artifact.

Gold verification:

```bash
python verify.py --check-only --json
python ../../../tools/paper_repo_env/validate_task_package.py --task openpcdet_pointcloud_minimal
```
