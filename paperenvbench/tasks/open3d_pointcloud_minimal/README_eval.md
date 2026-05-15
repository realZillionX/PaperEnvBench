# open3d_pointcloud_minimal

This task checks whether an agent can build a minimal CPU Open3D point-cloud environment from the pinned `isl-org/Open3D` repository.

The full upstream project has a large native CMake build surface and rendering dependencies. The gold route keeps a pinned source checkout for provenance, installs the matching `open3d==0.19.0` Python wheel, and avoids visualization or headless rendering.

L4 requires a deterministic geometry artifact. The verifier creates a 5x5 tilted-plane point cloud, runs `open3d.geometry.PointCloud.estimate_normals(...)`, orients and normalizes the normals, applies `voxel_down_sample(voxel_size=0.55)`, and validates point counts, unit normal lengths, finite geometry values, route evidence, and repository commit metadata.

Validated remote run:

```bash
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/geometry/open3d_pointcloud_minimal "./gold_install.sh"
inspire notebook exec paper-repro-ci-prep --cwd paper-repro:runs/paperenvbench/geometry/open3d_pointcloud_minimal "venv/bin/python verify.py --check-only --json"
```
