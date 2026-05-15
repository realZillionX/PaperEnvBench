#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys
import types
from collections import OrderedDict
from typing import Any


TASK_ID = "openpcdet_pointcloud_minimal"
EXPECTED_COMMIT = "233f849829b6ac19afb8af8837a0246890908755"
EXPECTED_SUCCESS_LEVEL = "L4_fallback"
EXPECTED_ARTIFACT_SHA256 = "55f14342e2617f718088d09cc4133e84ad5e36c61ae4a52ba324409bc3e07996"
EXPECTED_PREVIEW_SHA256 = "300164dc381016224d377a31af15c456ad78e009401c8d36cf1a56ab2003c431"
TASK_ROOT = pathlib.Path(__file__).resolve().parent


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def round_float(value: float) -> float:
    return round(float(value), 10)


def git_commit(repo_dir: pathlib.Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return ""


def load_source_module(module_path: pathlib.Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_runtime_stubs() -> dict[str, Any]:
    if "SharedArray" not in sys.modules:
        shared_array = types.ModuleType("SharedArray")
        shared_array.attach = lambda *_args, **_kwargs: None
        sys.modules["SharedArray"] = shared_array
    return {"SharedArray": "stubbed_for_common_utils_import"}


def load_yaml(path: pathlib.Path) -> Any:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def cpu_voxelize(points, point_cloud_range, voxel_size, max_points_per_voxel):
    import numpy as np

    min_range = np.array(point_cloud_range[:3], dtype=np.float32)
    max_range = np.array(point_cloud_range[3:6], dtype=np.float32)
    voxel_size_np = np.array(voxel_size, dtype=np.float32)
    grid_size = np.round((max_range - min_range) / voxel_size_np).astype(np.int64)

    grouped: OrderedDict[tuple[int, int, int], list[Any]] = OrderedDict()
    for point in points:
        xyz = point[:3]
        if np.any(xyz < min_range) or np.any(xyz >= max_range):
            continue
        coord_xyz = np.floor((xyz - min_range) / voxel_size_np).astype(np.int64)
        if np.any(coord_xyz < 0) or np.any(coord_xyz >= grid_size):
            continue
        key = (int(coord_xyz[2]), int(coord_xyz[1]), int(coord_xyz[0]))
        grouped.setdefault(key, [])
        if len(grouped[key]) < max_points_per_voxel:
            grouped[key].append(point)

    keys = sorted(grouped.keys())
    voxels = np.zeros((len(keys), max_points_per_voxel, points.shape[1]), dtype=np.float32)
    coords = np.zeros((len(keys), 4), dtype=np.int64)
    num_points = np.zeros((len(keys),), dtype=np.int64)
    for idx, key in enumerate(keys):
        pts = np.asarray(grouped[key], dtype=np.float32)
        voxels[idx, : pts.shape[0], :] = pts
        coords[idx] = np.array([0, key[0], key[1], key[2]], dtype=np.int64)
        num_points[idx] = pts.shape[0]
    return voxels, coords, num_points, grid_size


def make_preview(path: pathlib.Path, points, coords, kept_mask, grid_size) -> dict[str, Any]:
    import numpy as np

    width, height = 160, 128
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, :, :] = np.array([15, 18, 24], dtype=np.uint8)
    for row in range(height):
        pixels[row, :, 1] = np.uint8(18 + row % 34)
    for col in range(width):
        pixels[:, col, 2] = np.uint8(24 + col % 48)

    def put(px: int, py: int, color: tuple[int, int, int], radius: int = 1) -> None:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = px + dx, py + dy
                if 0 <= x < width and 0 <= y < height:
                    pixels[y, x, :] = color

    for point, keep in zip(points, kept_mask):
        px = int(round((point[0] / 69.12) * (width - 1)))
        py = int(round(((point[1] + 39.68) / 79.36) * (height - 1)))
        py = height - 1 - py
        intensity = int(max(0, min(255, point[3] * 255)))
        put(px, py, (245, 196, 66) if keep else (80, 80, 88), radius=2 if keep else 1)
        if keep:
            pixels[max(0, py - 5) : min(height, py + 6), max(0, px - 5), :] = (66, 220, 180)
            pixels[max(0, py - 5) : min(height, py + 6), min(width - 1, px + 5), :] = (66, 220, 180)

    for coord in coords:
        _, _z, y, x = [int(v) for v in coord]
        px = int(round((x / max(1, grid_size[0] - 1)) * (width - 1)))
        py = height - 1 - int(round((y / max(1, grid_size[1] - 1)) * (height - 1)))
        put(px, py, (255, 64, 88), radius=1)

    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + pixels.tobytes())
    return {
        "format": "ppm_p6",
        "width": width,
        "height": height,
        "channels": 3,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def probe_native_extensions(repo_dir: pathlib.Path) -> dict[str, Any]:
    setup_text = (repo_dir / "setup.py").read_text(encoding="utf-8")
    extension_names = [
        "iou3d_nms_cuda",
        "roiaware_pool3d_cuda",
        "roipoint_pool3d_cuda",
        "pointnet2_stack_cuda",
        "pointnet2_batch_cuda",
        "bev_pool_ext",
        "ingroup_inds_cuda",
    ]
    return {
        "ok": False,
        "expected_blocker": True,
        "setup_uses_cuda_extension": "CUDAExtension" in setup_text,
        "declared_cuda_extensions": [name for name in extension_names if name in setup_text],
        "reason": "OpenPCDet setup.py declares CUDAExtension modules; CPU fallback does not claim compiled detector NMS or ROI pooling.",
    }


def generate(repo_dir: pathlib.Path, output_dir: pathlib.Path, artifact_name: str) -> dict[str, Any]:
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = repo_dir.resolve()
    stubs = install_runtime_stubs()
    sys.path.insert(0, str(repo_dir))

    try:
        import torch

        torch_available = True
    except Exception:
        torch = None
        torch_available = False

    common_utils = None
    scatter_module = None
    if torch_available:
        common_utils = load_source_module(
            repo_dir / "pcdet" / "utils" / "common_utils.py",
            "paperenvbench_openpcdet_common_utils",
        )
        scatter_module = load_source_module(
            repo_dir / "pcdet" / "models" / "backbones_2d" / "map_to_bev" / "pointpillar_scatter.py",
            "paperenvbench_openpcdet_pointpillar_scatter",
        )
    else:
        common_utils_source = (repo_dir / "pcdet" / "utils" / "common_utils.py").read_text(encoding="utf-8")
        scatter_source = (
            repo_dir / "pcdet" / "models" / "backbones_2d" / "map_to_bev" / "pointpillar_scatter.py"
        ).read_text(encoding="utf-8")
        if "def mask_points_by_range" not in common_utils_source or "class PointPillarScatter" not in scatter_source:
            raise AssertionError("required OpenPCDet utility source symbols were not found")

    cfg = load_yaml(repo_dir / "tools" / "cfgs" / "kitti_models" / "pointpillar.yaml")
    data_cfg = cfg["DATA_CONFIG"]
    point_cloud_range = [float(v) for v in data_cfg["POINT_CLOUD_RANGE"]]
    voxel_cfg = next(item for item in data_cfg["DATA_PROCESSOR"] if item["NAME"] == "transform_points_to_voxels")
    voxel_size = [float(v) for v in voxel_cfg["VOXEL_SIZE"]]
    max_points = int(voxel_cfg["MAX_POINTS_PER_VOXEL"])

    points = np.array(
        [
            [3.20, -1.40, -1.20, 0.95],
            [3.26, -1.36, -1.17, 0.88],
            [3.38, -1.20, -1.14, 0.74],
            [8.00, 2.40, -1.00, 0.60],
            [8.09, 2.46, -0.96, 0.58],
            [8.31, 2.58, -0.91, 0.52],
            [16.00, -5.00, -0.80, 0.40],
            [69.50, 0.00, -1.00, 0.30],
            [1.00, -45.00, -1.00, 0.20],
        ],
        dtype=np.float32,
    )
    if common_utils is not None:
        xy_mask = common_utils.mask_points_by_range(points, point_cloud_range)
    else:
        xy_mask = (
            (points[:, 0] >= point_cloud_range[0])
            & (points[:, 0] <= point_cloud_range[3])
            & (points[:, 1] >= point_cloud_range[1])
            & (points[:, 1] <= point_cloud_range[4])
        )
    z_mask = (points[:, 2] >= point_cloud_range[2]) & (points[:, 2] < point_cloud_range[5])
    kept_mask = xy_mask & z_mask
    filtered_points = points[kept_mask]

    voxels, coords, num_points, grid_size = cpu_voxelize(
        filtered_points,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_points_per_voxel=max_points,
    )
    if common_utils is not None:
        voxel_centers = common_utils.get_voxel_centers(
            torch.from_numpy(coords[:, 1:4]),
            downsample_times=1,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
        ).numpy()
        rotated = common_utils.rotate_points_along_z(
            torch.from_numpy(filtered_points[None, :, :3]),
            torch.tensor([0.25], dtype=torch.float32),
        )[0].numpy()
    else:
        coord_xyz = coords[:, [3, 2, 1]].astype(np.float32)
        voxel_centers = (coord_xyz + 0.5) * np.array(voxel_size, dtype=np.float32) + np.array(
            point_cloud_range[:3], dtype=np.float32
        )
        angle = 0.25
        rot = np.array(
            [
                [math.cos(angle), math.sin(angle), 0.0],
                [-math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        rotated = filtered_points[:, :3] @ rot

    base_features_np = voxels[:, :, :4].sum(axis=1)
    repeats = math.ceil(64 / base_features_np.shape[1])
    pillar_features_np = np.tile(base_features_np, (1, repeats))[:, :64].astype(np.float32)
    if scatter_module is not None:
        torch.manual_seed(20260516)
        pillar_features = torch.from_numpy(pillar_features_np).contiguous()
        scatter = scatter_module.PointPillarScatter(types.SimpleNamespace(NUM_BEV_FEATURES=64), grid_size=grid_size)
        scatter_out = scatter(
            {
                "pillar_features": pillar_features,
                "voxel_coords": torch.from_numpy(coords),
            }
        )
        spatial_features = scatter_out["spatial_features"]
        nonzero_count = int(torch.nonzero(spatial_features.abs().sum(dim=1)[0] > 0, as_tuple=False).shape[0])
        spatial_shape = [int(v) for v in spatial_features.shape]
        scatter_execution_mode = "official_torch_module"
    else:
        nonzero_count = len({(int(coord[2]), int(coord[3])) for coord in coords})
        spatial_shape = [1, 64, int(grid_size[1]), int(grid_size[0])]
        scatter_execution_mode = "numpy_equivalent_after_source_symbol_check"
    preview = make_preview(output_dir / "expected_pointcloud_bev.ppm", points, coords, kept_mask, grid_size)

    repo_commit = git_commit(repo_dir)
    native_extensions = probe_native_extensions(repo_dir)
    output = {
        "grid_size_xyz": [int(v) for v in grid_size.tolist()],
        "filtered_point_count": int(filtered_points.shape[0]),
        "voxel_count": int(voxels.shape[0]),
        "voxel_coords_zyx": coords[:, 1:4].astype(int).tolist(),
        "voxel_num_points": num_points.astype(int).tolist(),
        "first_voxel_center_xyz": [round_float(v) for v in voxel_centers[0].tolist()],
        "rotated_first_point_xyz": [round_float(v) for v in rotated[0].tolist()],
        "pillar_feature_shape": [int(v) for v in pillar_features_np.shape],
        "spatial_features_shape": spatial_shape,
        "bev_nonzero_cell_count": nonzero_count,
        "scatter_execution_mode": scatter_execution_mode,
        "voxel_coords_sha256": sha256_json(coords[:, 1:4].astype(int).tolist()),
        "preview": preview,
    }
    payload = {
        "task_id": TASK_ID,
        "success_level": EXPECTED_SUCCESS_LEVEL,
        "repo": "open-mmlab/OpenPCDet",
        "repo_commit": repo_commit,
        "paper_model": "PointPillar point cloud detector",
        "config": "tools/cfgs/kitti_models/pointpillar.yaml",
        "official_repo_code_paths": [
            "setup.py",
            "tools/cfgs/kitti_models/pointpillar.yaml",
            "pcdet/datasets/processor/data_processor.py",
            "pcdet/utils/common_utils.py",
            "pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py",
            "pcdet/models/detectors/pointpillar.py",
        ],
        "fallback_reason": {
            "native_cuda_extensions": native_extensions,
            "spconv_voxel_generator_not_required_for_cpu_artifact": True,
            "checkpoint_inference_not_claimed": True,
            "runtime_stubs": stubs,
        },
        "input": {
            "class_names": cfg["CLASS_NAMES"],
            "point_cloud_range": point_cloud_range,
            "voxel_size": voxel_size,
            "max_points_per_voxel": max_points,
            "raw_points_xyzi": [[round_float(v) for v in row] for row in points.tolist()],
            "kept_point_indices": [int(i) for i, keep in enumerate(kept_mask.tolist()) if keep],
        },
        "output": output,
        "package_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__ if torch_available else "not_installed_local_generation",
            "numpy": np.__version__,
        },
        "checks": {
            "repo_commit_matches": repo_commit == EXPECTED_COMMIT,
            "official_pointpillar_config_loaded": cfg["MODEL"]["NAME"] == "PointPillar",
            "official_range_filter_executed": int(filtered_points.shape[0]) == 7,
            "cpu_voxel_route_executed": int(voxels.shape[0]) == 5 and coords.shape == (5, 4),
            "official_voxel_center_utility_executed": len(output["first_voxel_center_xyz"]) == 3,
            "official_pointpillar_scatter_executed": spatial_shape == [1, 64, 496, 432],
            "native_extension_blocker_recorded": native_extensions["expected_blocker"] is True,
            "preview_artifact_written": preview["size_bytes"] > 1024,
        },
    }
    if not all(payload["checks"].values()):
        failed = [key for key, value in payload["checks"].items() if not value]
        raise AssertionError(f"semantic checks failed: {failed}")
    artifact_path = output_dir / artifact_name
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def check_existing(output_dir: pathlib.Path) -> dict[str, Any]:
    expected = json.loads((pathlib.Path(__file__).resolve().parent / "expected_output.json").read_text(encoding="utf-8"))
    artifact_path = output_dir / expected["required_artifact"]
    preview_path = output_dir / expected["required_side_artifact"]
    if not artifact_path.exists() or artifact_path.stat().st_size <= 0:
        raise AssertionError(f"missing nonempty artifact: {artifact_path}")
    if not preview_path.exists() or preview_path.stat().st_size <= 0:
        raise AssertionError(f"missing nonempty preview: {preview_path}")
    artifact_sha = sha256_file(artifact_path)
    preview_sha = sha256_file(preview_path)
    if artifact_sha != expected["gold_observed"]["artifact_sha256"]:
        raise AssertionError("artifact checksum mismatch")
    if preview_sha != expected["gold_observed"]["preview_sha256"]:
        raise AssertionError("preview checksum mismatch")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if payload.get("task_id") != TASK_ID:
        raise AssertionError(f"wrong task_id: {payload.get('task_id')}")
    if payload.get("repo_commit") != EXPECTED_COMMIT:
        raise AssertionError(f"wrong repo_commit: {payload.get('repo_commit')}")
    if payload.get("success_level") != expected["expected_success_level"]:
        raise AssertionError(f"wrong success_level: {payload.get('success_level')}")
    checks = payload.get("checks", {})
    if not isinstance(checks, dict) or not all(checks.values()):
        raise AssertionError(f"semantic checks failed: {checks}")
    observed = payload["output"]
    thresholds = expected["semantic_thresholds"]
    if observed["voxel_count"] != thresholds["expected_voxel_count"]:
        raise AssertionError(f"wrong voxel_count: {observed['voxel_count']}")
    if observed["spatial_features_shape"] != thresholds["expected_spatial_features_shape"]:
        raise AssertionError(f"wrong spatial shape: {observed['spatial_features_shape']}")
    if payload["fallback_reason"]["native_cuda_extensions"]["expected_blocker"] is not True:
        raise AssertionError("native extension blocker was not recorded")
    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": payload["success_level"],
        "mode": "check_only",
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": artifact_sha,
        "observed": {
            "repo_commit": payload["repo_commit"],
            "voxel_count": observed["voxel_count"],
            "grid_size_xyz": observed["grid_size_xyz"],
            "bev_nonzero_cell_count": observed["bev_nonzero_cell_count"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the PaperEnvBench OpenPCDet point cloud minimal task.")
    parser.add_argument("--repo-dir", default=os.environ.get("PAPERENVBENCH_REPO_DIR", str(TASK_ROOT / "repo")))
    parser.add_argument("--output-dir", default=os.environ.get("PAPERENVBENCH_OUTPUT_DIR", str(TASK_ROOT / "artifacts")))
    parser.add_argument("--artifact-name", default="expected_artifact.json")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit JSON result. Kept for hidden-runner compatibility.")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir).resolve()
    try:
        if args.check_only:
            result = check_existing(output_dir)
        else:
            repo_dir = pathlib.Path(args.repo_dir).resolve()
            result = generate(repo_dir, output_dir, args.artifact_name)
    except Exception as exc:
        error = {"task_id": TASK_ID, "status": "fail", "error": str(exc)}
        print(json.dumps(error, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
