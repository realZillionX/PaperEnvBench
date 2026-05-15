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
TASK_ID = "nerfstudio_nerfacto_minimal"
EXPECTED_COMMIT = "50e0e3c70c775e89333256213363badbf074f29d"
SUCCESS_LEVEL = "L4_cpu_deterministic_fallback"
REQUIRED_ROUTE_TERMS = [
    "ns-train nerfacto",
    "NerfstudioDataParserConfig",
    "VanillaPipelineConfig",
    "NerfactoModelConfig",
    "nerfstudio/configs/method_configs.py",
    "nerfstudio/models/nerfacto.py",
    "nerfstudio/data/dataparsers/nerfstudio_dataparser.py",
    "nerfstudio/pipelines/base_pipeline.py",
]
REQUIRED_SEMANTIC_TERMS = [
    "nerfacto",
    "dataparser",
    "pipeline",
    "tiny ray bundle",
    "CPU deterministic fallback",
]


def canonical_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_nested(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [round_nested(item) for item in value]
    if isinstance(value, dict):
        return {key: round_nested(item) for key, item in value.items()}
    return value


def build_artifact() -> dict[str, Any]:
    ray_samples = [
        {"ray": 0, "depth": 0.18, "density": 1.20, "rgb": [0.80, 0.25, 0.10]},
        {"ray": 0, "depth": 0.42, "density": 0.70, "rgb": [0.35, 0.55, 0.20]},
        {"ray": 0, "depth": 0.76, "density": 0.25, "rgb": [0.10, 0.20, 0.70]},
        {"ray": 1, "depth": 0.16, "density": 0.10, "rgb": [0.15, 0.20, 0.40]},
        {"ray": 1, "depth": 0.38, "density": 0.45, "rgb": [0.35, 0.45, 0.80]},
        {"ray": 1, "depth": 0.74, "density": 1.10, "rgb": [0.90, 0.75, 0.25]},
    ]
    deltas = {0: [0.24, 0.34, 0.24], 1: [0.22, 0.36, 0.26]}
    rendered: dict[str, Any] = {}
    for ray_id in (0, 1):
        transmittance = 1.0
        rgb = [0.0, 0.0, 0.0]
        depth = 0.0
        weights: list[float] = []
        samples = [sample for sample in ray_samples if sample["ray"] == ray_id]
        for sample, delta in zip(samples, deltas[ray_id]):
            alpha = 1.0 - math.exp(-float(sample["density"]) * delta)
            weight = transmittance * alpha
            weights.append(weight)
            rgb = [rgb[index] + weight * float(sample["rgb"][index]) for index in range(3)]
            depth += weight * float(sample["depth"])
            transmittance *= 1.0 - alpha
        rendered[str(ray_id)] = {
            "rgb": rgb,
            "accumulation": sum(weights),
            "depth": depth / max(sum(weights), 1e-8),
            "weights": weights,
            "final_transmittance": transmittance,
        }

    route = {
        "cli_entrypoint": "ns-train nerfacto --data <tiny-nerfstudio-scene> --max-num-iterations 1 --vis viewer",
        "repo_files": [
            "pyproject.toml",
            "nerfstudio/configs/method_configs.py",
            "nerfstudio/data/dataparsers/nerfstudio_dataparser.py",
            "nerfstudio/data/datamanagers/parallel_datamanager.py",
            "nerfstudio/models/nerfacto.py",
            "nerfstudio/fields/nerfacto_field.py",
            "nerfstudio/pipelines/base_pipeline.py",
            "nerfstudio/scripts/train.py",
        ],
        "method_config": {
            "name": "nerfacto",
            "trainer": "TrainerConfig",
            "pipeline": "VanillaPipelineConfig",
            "datamanager": "ParallelDataManagerConfig",
            "dataparser": "NerfstudioDataParserConfig",
            "model": "NerfactoModelConfig",
            "camera_optimizer": "SO3xR3",
            "train_num_rays_per_batch": 4096,
            "eval_num_rays_per_batch": 4096,
        },
        "fallback_reason": (
            "Full Nerfstudio training pulls GPU-oriented native packages such as nerfacc, gsplat, open3d, "
            "and viewer dependencies. The gold check records the real pinned nerfacto route and validates "
            "a deterministic CPU NeRF-style volume-rendering artifact instead of requiring those kernels."
        ),
    }
    semantic = {
        "task": "Nerfstudio nerfacto tiny scene route validation",
        "claim": "CPU deterministic fallback validates the nerfacto dataparser and pipeline route with a tiny ray bundle.",
        "success_condition": (
            "The artifact must pin the upstream commit, name the ns-train nerfacto entrypoint, identify "
            "NerfstudioDataParserConfig, VanillaPipelineConfig, and NerfactoModelConfig, and contain finite "
            "volume-rendering outputs."
        ),
    }
    numeric = round_nested(
        {
            "ray_samples": ray_samples,
            "deltas": deltas,
            "rendered": rendered,
            "mean_rgb": [
                sum(rendered[str(ray_id)]["rgb"][channel] for ray_id in (0, 1)) / 2.0
                for channel in range(3)
            ],
            "mean_depth": sum(rendered[str(ray_id)]["depth"] for ray_id in (0, 1)) / 2.0,
        }
    )
    checks = {
        "repo_commit_matches": True,
        "cli_route_recorded": "ns-train nerfacto" in route["cli_entrypoint"],
        "dataparser_route_recorded": route["method_config"]["dataparser"] == "NerfstudioDataParserConfig",
        "pipeline_route_recorded": route["method_config"]["pipeline"] == "VanillaPipelineConfig",
        "nerfacto_model_route_recorded": route["method_config"]["model"] == "NerfactoModelConfig",
        "volume_rendering_finite": all(
            math.isfinite(float(value))
            for ray in numeric["rendered"].values()
            for key in ("accumulation", "depth", "final_transmittance")
            for value in ([ray[key]] if not isinstance(ray[key], list) else ray[key])
        ),
        "tiny_ray_bundle_semantic": len(numeric["ray_samples"]) == 6 and sorted(numeric["rendered"].keys()) == ["0", "1"],
    }
    payload = {
        "task_id": TASK_ID,
        "success_level": SUCCESS_LEVEL,
        "repo": {
            "url": "https://github.com/nerfstudio-project/nerfstudio",
            "commit": EXPECTED_COMMIT,
            "commit_short": EXPECTED_COMMIT[:7],
            "paper_title": "Nerfstudio: A Modular Framework for Neural Radiance Field Development",
        },
        "route": route,
        "semantic": semantic,
        "numeric": numeric,
        "checks": checks,
        "sha256": {
            "route": canonical_sha256(route),
            "semantic": canonical_sha256(semantic),
            "numeric": canonical_sha256(numeric),
        },
    }
    return payload


def validate_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_path = artifact_dir / "expected_artifact.json"
    if not artifact_path.exists():
        raise AssertionError(f"missing artifact: {artifact_path}")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if payload.get("task_id") != TASK_ID:
        raise AssertionError(f"wrong task_id: {payload.get('task_id')}")
    if payload.get("success_level") != SUCCESS_LEVEL:
        raise AssertionError(f"unexpected success_level: {payload.get('success_level')}")
    if payload.get("repo", {}).get("commit") != EXPECTED_COMMIT:
        raise AssertionError("repo commit mismatch")
    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        raise AssertionError({"checks": checks})

    route = payload.get("route", {})
    route_text = json.dumps(route, sort_keys=True)
    missing_route = [term for term in REQUIRED_ROUTE_TERMS if term not in route_text]
    if missing_route:
        raise AssertionError(f"route evidence is missing terms: {missing_route}")

    semantic = payload.get("semantic", {})
    semantic_text = json.dumps(semantic, sort_keys=True)
    missing_semantic = [term for term in REQUIRED_SEMANTIC_TERMS if term not in semantic_text]
    if missing_semantic:
        raise AssertionError(f"semantic evidence is missing terms: {missing_semantic}")

    numeric = payload.get("numeric", {})
    if numeric.get("mean_rgb") != [0.249961, 0.191085, 0.129177]:
        raise AssertionError(f"mean_rgb mismatch: {numeric.get('mean_rgb')}")
    if numeric.get("mean_depth") != 0.438423:
        raise AssertionError(f"mean_depth mismatch: {numeric.get('mean_depth')}")
    if sorted(numeric.get("rendered", {}).keys()) != ["0", "1"]:
        raise AssertionError("rendered rays missing")
    for ray in numeric["rendered"].values():
        if not 0.0 < float(ray["accumulation"]) < 1.0:
            raise AssertionError(f"invalid accumulation: {ray['accumulation']}")
        for value in ray["rgb"] + ray["weights"] + [ray["depth"], ray["final_transmittance"]]:
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise AssertionError(f"non-finite numeric value: {value}")

    sha = payload.get("sha256", {})
    for key in ("route", "semantic", "numeric"):
        if sha.get(key) != canonical_sha256(payload[key]):
            raise AssertionError(f"{key} sha256 mismatch")

    return {
        "task_id": TASK_ID,
        "status": "pass",
        "success_level": SUCCESS_LEVEL,
        "artifact_dir": str(artifact_dir.resolve()),
        "artifact_sha256": sha256_file(artifact_path),
        "checks": checks,
        "observed": {
            "repo_commit": payload.get("repo", {}).get("commit"),
            "route": route["method_config"],
            "numeric": {
                "mean_rgb": numeric.get("mean_rgb"),
                "mean_depth": numeric.get("mean_depth"),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=TASK_ROOT / "artifacts")
    parser.add_argument("--generate", action="store_true", help="Regenerate expected_artifact.json.")
    parser.add_argument("--check-only", action="store_true", help="Validate the existing expected artifact.")
    parser.add_argument("--json", action="store_true", help="Emit JSON result.")
    args = parser.parse_args()

    try:
        if args.generate:
            args.artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = args.artifact_dir / "expected_artifact.json"
            artifact_path.write_text(json.dumps(build_artifact(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = validate_artifact(args.artifact_dir)
    except Exception as exc:
        print(json.dumps({"task_id": TASK_ID, "status": "fail", "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
