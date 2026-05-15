#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

TASK_ROOT = Path(__file__).resolve().parent
TASK_ID = "open3d_pointcloud_minimal"
EXPECTED_COMMIT = "1e7b17438687a0b0c1e5a7187321ac7044afe275"
SUCCESS_LEVEL = "L4_cpu_deterministic_fallback"
EXPECTED_POINT_COUNT = 25
EXPECTED_NORMAL_COUNT = 25
EXPECTED_VOXEL_COUNT = 25
REQUIRED_ROUTE_TERMS = [
    "open3d.geometry.PointCloud",
    "KDTreeSearchParamKNN",
    "estimate_normals",
    "voxel_down_sample",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


def git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return ""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def make_points() -> tuple[Any, Any]:
    import numpy as np

    xs = np.linspace(-1.0, 1.0, 5)
    ys = np.linspace(-1.0, 1.0, 5)
    points: list[list[float]] = []
    colors: list[list[float]] = []
    for y in ys:
        for x in xs:
            z = 0.2 * x + 0.1 * y
            points.append([float(x), float(y), float(z)])
            colors.append([float((x + 1.0) / 2.0), float((y + 1.0) / 2.0), float((z + 0.3) / 0.6)])
    return np.asarray(points, dtype=np.float64), np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0)


def run_open3d(repo_dir: Path, output_dir: Path) -> dict[str, Any]:
    import numpy as np
    import open3d as o3d

    output_dir.mkdir(parents=True, exist_ok=True)
    points, colors = make_points()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=8))
    pcd.orient_normals_to_align_with_direction(np.asarray([-0.2, -0.1, 1.0], dtype=np.float64))
    pcd.normalize_normals()
    normals = np.asarray(pcd.normals)
    voxel_cloud = pcd.voxel_down_sample(voxel_size=0.55)
    voxel_points = np.asarray(voxel_cloud.points)

    normal_lengths = np.linalg.norm(normals, axis=1)
    finite_geometry = bool(np.isfinite(points).all() and np.isfinite(normals).all() and np.isfinite(voxel_points).all())
    repo_commit = git_commit(repo_dir)
    route = {
        "package": "open3d",
        "pointcloud_class": "open3d.geometry.PointCloud",
        "point_vector": "open3d.utility.Vector3dVector",
        "normal_estimation": "PointCloud.estimate_normals(open3d.geometry.KDTreeSearchParamKNN(knn=8))",
        "normal_orientation": "PointCloud.orient_normals_to_align_with_direction([-0.2, -0.1, 1.0])",
        "voxel_route": "PointCloud.voxel_down_sample(voxel_size=0.55)",
        "rendering_required": False,
    }
    checks = {
        "repo_commit_matches": repo_commit == EXPECTED_COMMIT,
        "open3d_import_resolves": hasattr(o3d, "geometry") and hasattr(o3d.geometry, "PointCloud"),
        "pointcloud_route_recorded": all(term in json.dumps(route, sort_keys=True) for term in REQUIRED_ROUTE_TERMS),
        "point_count_matches": int(len(points)) == EXPECTED_POINT_COUNT,
        "normal_count_matches": int(len(normals)) == EXPECTED_NORMAL_COUNT,
        "voxel_count_matches": int(len(voxel_points)) == EXPECTED_VOXEL_COUNT,
        "normal_lengths_unit": float(normal_lengths.min()) >= 0.999999 and float(normal_lengths.max()) <= 1.000001,
        "finite_geometry_values": finite_geometry,
    }

    payload = {
        "task_id": TASK_ID,
        "success": all(checks.values()),
        "success_level": SUCCESS_LEVEL if all(checks.values()) else "below_L4",
        "repo": {
            "url": "https://github.com/isl-org/Open3D",
            "commit": repo_commit,
            "tag": "v0.19.0",
        },
        "python": sys.version.split()[0],
        "package_versions": {
            "open3d": importlib.metadata.version("open3d"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "route": route,
        "point_cloud": {
            "point_count": int(len(points)),
            "color_count": int(len(colors)),
            "centroid": [round(float(v), 10) for v in points.mean(axis=0).tolist()],
            "bbox_min": [round(float(v), 10) for v in points.min(axis=0).tolist()],
            "bbox_max": [round(float(v), 10) for v in points.max(axis=0).tolist()],
            "points_sha256": sha256_array(points),
            "colors_sha256": sha256_array(colors),
        },
        "normals": {
            "count": int(len(normals)),
            "mean_normal": [round(float(v), 10) for v in normals.mean(axis=0).tolist()],
            "normal_norm_min": round(float(normal_lengths.min()), 10),
            "normal_norm_max": round(float(normal_lengths.max()), 10),
            "normals_sha256": sha256_array(normals),
        },
        "voxel_downsample": {
            "voxel_size": 0.55,
            "point_count": int(len(voxel_points)),
            "centroid": [round(float(v), 10) for v in voxel_points.mean(axis=0).tolist()],
            "points_sha256": sha256_array(voxel_points),
        },
        "checks": checks,
    }
    return payload


def validate_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_path = artifact_dir / "expected_artifact.json"
    require(artifact_path.exists(), f"missing artifact: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    require(payload.get("task_id") == TASK_ID, f"wrong task_id: {payload.get('task_id')}")
    require(payload.get("success") is True, "success flag must be true")
    require(payload.get("success_level") == SUCCESS_LEVEL, f"wrong success_level: {payload.get('success_level')}")
    require(payload.get("repo", {}).get("commit") == EXPECTED_COMMIT, "repo commit mismatch")
    require(payload.get("package_versions", {}).get("open3d") == "0.19.0", "open3d version mismatch")

    route_text = json.dumps(payload.get("route", {}), sort_keys=True)
    missing = [term for term in REQUIRED_ROUTE_TERMS if term not in route_text]
    require(not missing, f"route evidence is missing terms: {missing}")
    require(payload.get("route", {}).get("rendering_required") is False, "minimal route must not require rendering")

    point_cloud = payload.get("point_cloud", {})
    normals = payload.get("normals", {})
    voxel = payload.get("voxel_downsample", {})
    require(point_cloud.get("point_count") == EXPECTED_POINT_COUNT, "point count mismatch")
    require(normals.get("count") == EXPECTED_NORMAL_COUNT, "normal count mismatch")
    require(voxel.get("point_count") == EXPECTED_VOXEL_COUNT, "voxel point count mismatch")
    require(point_cloud.get("centroid") == [0.0, 0.0, 0.0], "centroid mismatch")
    require(point_cloud.get("bbox_min") == [-1.0, -1.0, -0.3], "bbox_min mismatch")
    require(point_cloud.get("bbox_max") == [1.0, 1.0, 0.3], "bbox_max mismatch")
    require(float(normals.get("normal_norm_min")) >= 0.999999, "normal min norm too small")
    require(float(normals.get("normal_norm_max")) <= 1.000001, "normal max norm too large")
    require(math.isclose(float(voxel.get("voxel_size")), 0.55, abs_tol=1e-12), "voxel size mismatch")

    checks = payload.get("checks", {})
    require(isinstance(checks, dict) and checks, "missing checks")
    require(all(value is True for value in checks.values()), f"failing checks: {checks}")

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": SUCCESS_LEVEL,
        "artifact_dir": str(artifact_dir.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "checks": checks,
        "observed": {
            "repo_commit": payload.get("repo", {}).get("commit"),
            "open3d_version": payload.get("package_versions", {}).get("open3d"),
            "point_count": point_cloud.get("point_count"),
            "normal_count": normals.get("count"),
            "voxel_point_count": voxel.get("point_count"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, default=Path("repo"))
    parser.add_argument("--output-dir", type=Path, default=TASK_ROOT / "artifacts")
    parser.add_argument("--artifact-dir", type=Path, default=TASK_ROOT / "artifacts")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        if args.check_only:
            result = validate_artifact(args.artifact_dir)
        else:
            payload = run_open3d(args.repo_dir.resolve(), args.output_dir.resolve())
            artifact_path = args.output_dir / "expected_artifact.json"
            artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result = payload
            if not payload["success"]:
                failed = ", ".join(key for key, ok in payload["checks"].items() if not ok)
                raise AssertionError(f"semantic checks failed: {failed}")
    except Exception as exc:
        print(json.dumps({"task_id": TASK_ID, "status": "fail", "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
