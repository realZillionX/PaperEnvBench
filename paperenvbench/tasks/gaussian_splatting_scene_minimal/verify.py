#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parent
TASK_ID = "gaussian_splatting_scene_minimal"
EXPECTED_COMMIT = "54c035f7834b564019656c3e3fcc3646292f727d"
SUCCESS_LEVEL = "L4_cpu_deterministic_fallback"
REQUIRED_ROUTE_KEYS = [
    "train_script",
    "render_script",
    "metrics_script",
    "dataset_route",
    "model_output",
    "render_output",
    "rasterizer_submodule",
    "knn_submodule",
    "fused_ssim_submodule",
    "python_entrypoints",
    "core_code_paths",
]
REQUIRED_PINNED_FILES = {
    "train.py",
    "render.py",
    "metrics.py",
    "gaussian_renderer/__init__.py",
    "scene/gaussian_model.py",
    "scene/dataset_readers.py",
    "arguments/__init__.py",
    "environment.yml",
}


def canonical_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssertionError(f"JSON artifact is not parseable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AssertionError(f"JSON artifact must be an object: {path}")
    return payload


def parse_scene_ply(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="ascii").splitlines()
    if not lines or lines[0] != "ply":
        raise AssertionError("expected_gaussian_scene.ply must be an ASCII PLY file")
    try:
        header_end = lines.index("end_header")
    except ValueError as exc:
        raise AssertionError("PLY header is missing end_header") from exc

    vertex_count = None
    for line in lines[:header_end]:
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
    if vertex_count != 5:
        raise AssertionError(f"expected 5 Gaussian vertices, found {vertex_count}")

    rows = []
    for line in lines[header_end + 1 :]:
        if line.strip():
            rows.append(line.split())
    if len(rows) != vertex_count:
        raise AssertionError(f"expected {vertex_count} PLY rows, found {len(rows)}")

    xyz = []
    scales = []
    opacities = []
    colors = []
    for row in rows:
        if len(row) != 10:
            raise AssertionError(f"unexpected PLY row width: {len(row)}")
        values = [float(value) for value in row[:7]]
        rgb = [int(value) for value in row[7:]]
        if any(not math.isfinite(value) for value in values):
            raise AssertionError("PLY contains non-finite floating point values")
        if any(value <= 0 for value in values[3:6]):
            raise AssertionError("Gaussian scales must be positive")
        if not 0.0 < values[6] < 1.0:
            raise AssertionError("Gaussian opacity must be in (0, 1)")
        if any(value < 0 or value > 255 for value in rgb):
            raise AssertionError("PLY RGB values must be uint8")
        xyz.append(values[:3])
        scales.append(values[3:6])
        opacities.append(values[6])
        colors.append(rgb)

    mean_xyz = [round(sum(point[i] for point in xyz) / len(xyz), 6) for i in range(3)]
    mean_scale = [round(sum(scale[i] for scale in scales) / len(scales), 6) for i in range(3)]
    mean_opacity = round(sum(opacities) / len(opacities), 6)
    mean_rgb = [round(sum(rgb[i] for rgb in colors) / (len(colors) * 255), 4) for i in range(3)]
    max_radius = round(max(math.sqrt(sum(coord * coord for coord in point)) for point in xyz), 6)
    return {
        "vertex_count": vertex_count,
        "mean_xyz": mean_xyz,
        "mean_scale": mean_scale,
        "mean_opacity": mean_opacity,
        "mean_rgb": mean_rgb,
        "max_radius": max_radius,
        "sha256": sha256_file(path),
    }


def assert_vector(name: str, values: Any, length: int, low: float | None = None, high: float | None = None) -> None:
    if not isinstance(values, list) or len(values) != length:
        raise AssertionError(f"{name} must be a list of length {length}")
    for value in values:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise AssertionError(f"{name} contains non-finite numeric values")
        if low is not None and float(value) < low:
            raise AssertionError(f"{name} contains value below {low}: {value}")
        if high is not None and float(value) > high:
            raise AssertionError(f"{name} contains value above {high}: {value}")


def validate_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_path = artifact_dir / "expected_artifact.json"
    scene_path = artifact_dir / "expected_gaussian_scene.ply"
    if not artifact_path.exists():
        raise AssertionError(f"missing artifact: {artifact_path}")
    if not scene_path.exists():
        raise AssertionError(f"missing scene artifact: {scene_path}")

    payload = load_json(artifact_path)
    scene_ply = parse_scene_ply(scene_path)
    if payload.get("task_id") != TASK_ID:
        raise AssertionError(f"wrong task_id: {payload.get('task_id')}")
    if payload.get("success_level") != SUCCESS_LEVEL:
        raise AssertionError(f"unexpected success_level: {payload.get('success_level')}")
    if payload.get("repo", {}).get("commit") != EXPECTED_COMMIT:
        raise AssertionError("repo commit mismatch")

    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        raise AssertionError({"checks": checks})

    evidence = payload.get("evidence", {})
    pinned_files = set(evidence.get("pinned_files", []))
    missing_files = sorted(REQUIRED_PINNED_FILES - pinned_files)
    if missing_files:
        raise AssertionError(f"pinned route evidence is missing files: {missing_files}")

    route = payload.get("route", {})
    missing_route = [key for key in REQUIRED_ROUTE_KEYS if not route.get(key)]
    if missing_route:
        raise AssertionError(f"route evidence is missing keys: {missing_route}")
    if route.get("train_script") != "train.py" or route.get("render_script") != "render.py":
        raise AssertionError("train/render script route is not recorded")
    if route.get("rasterizer_submodule") != "submodules/diff-gaussian-rasterization":
        raise AssertionError("diff-gaussian-rasterization route is missing")
    if route.get("knn_submodule") != "submodules/simple-knn":
        raise AssertionError("simple-knn route is missing")
    if "COLMAP" not in str(route.get("dataset_route")) and "NeRF" not in str(route.get("dataset_route")):
        raise AssertionError("dataset route must mention COLMAP or NeRF Synthetic inputs")

    semantic = payload.get("semantic", {})
    semantic_text = " ".join(str(value) for value in semantic.values())
    required_terms = ["3D Gaussian Splatting", "GaussianModel", "gaussian_renderer.render", "cpu_deterministic_gaussian_scene_fallback"]
    missing_terms = [term for term in required_terms if term not in semantic_text]
    if missing_terms:
        raise AssertionError(f"semantic artifact is missing required terms: {missing_terms}")

    scene = payload.get("scene", {})
    if scene.get("gaussian_count") != 5:
        raise AssertionError("scene must record five Gaussians")
    if scene.get("camera_count") != 3:
        raise AssertionError("scene must record three camera poses")
    if scene.get("image_width") != 32 or scene.get("image_height") != 24:
        raise AssertionError("scene image dimensions are not the expected deterministic fallback dimensions")
    assert_vector("scene.mean_xyz", scene.get("mean_xyz"), 3)
    assert_vector("scene.scale_mean", scene.get("scale_mean"), 3, low=0.01, high=1.0)
    assert_vector("scene.render_rgb_mean", scene.get("render_rgb_mean"), 3, low=0.2, high=0.7)
    assert_vector("scene.render_rgb_std", scene.get("render_rgb_std"), 3, low=0.01, high=0.5)
    assert_vector("scene.depth_range", scene.get("depth_range"), 2, low=0.0)
    if not 0.4 <= float(scene.get("opacity_mean", -1)) <= 0.9:
        raise AssertionError("opacity_mean is outside expected range")
    if float(scene.get("extent_radius", 0)) < 1.0:
        raise AssertionError("extent_radius is too small for a non-degenerate scene")

    sha = payload.get("sha256", {})
    if sha.get("route") != canonical_sha256(route):
        raise AssertionError("route sha256 mismatch")
    if sha.get("scene") != canonical_sha256(scene):
        raise AssertionError("scene sha256 mismatch")
    if sha.get("semantic") != canonical_sha256(semantic):
        raise AssertionError("semantic sha256 mismatch")
    if scene_ply["vertex_count"] != scene["gaussian_count"]:
        raise AssertionError("PLY vertex count does not match scene gaussian_count")
    if not all(abs(a - b) <= 1e-6 for a, b in zip(scene_ply["mean_xyz"], scene["mean_xyz"])):
        raise AssertionError("PLY mean_xyz does not match JSON scene")
    if not all(abs(a - b) <= 1e-6 for a, b in zip(scene_ply["mean_scale"], scene["scale_mean"])):
        raise AssertionError("PLY scale_mean does not match JSON scene")
    if abs(scene_ply["mean_opacity"] - scene["opacity_mean"]) > 1e-6:
        raise AssertionError("PLY mean opacity does not match JSON scene")

    if payload.get("environment", {}).get("cuda_rasterizer_executed") is not False:
        raise AssertionError("check-only fallback must not claim that the CUDA rasterizer was executed")

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": SUCCESS_LEVEL,
        "artifact_dir": str(artifact_dir.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "checks": checks,
        "observed": {
            "repo_commit": payload.get("repo", {}).get("commit"),
            "route": {
                "train_script": route.get("train_script"),
                "render_script": route.get("render_script"),
                "rasterizer_submodule": route.get("rasterizer_submodule"),
            },
            "scene": scene,
            "scene_ply": scene_ply,
            "semantic": semantic,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=TASK_ROOT / "artifacts")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = validate_artifact(args.artifact_dir)
    except Exception as exc:
        print(json.dumps({"task_id": TASK_ID, "status": "fail", "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
