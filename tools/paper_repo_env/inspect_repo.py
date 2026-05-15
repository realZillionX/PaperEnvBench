from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 on local machines.
    tomllib = None  # type: ignore[assignment]


MODALITY_RULES: list[tuple[str, list[str]]] = [
    ("audio_speech", ["wav", "mp3", "asr", "tts", "mel", "spectrogram", "torchaudio", "librosa", "whisper"]),
    (
        "image_classification_representation",
        ["torchvision", "pillow", "pil", "classification", "embedding", "imagenet", "resnet"],
    ),
    ("object_detection_segmentation", ["mask", "bbox", "coco", "mmdet", "detectron2", "segment", "sam"]),
    ("video_understanding_generation", ["mp4", "frames", "decord", "pyav", "temporal", "video"]),
    (
        "vision_language_multimodal",
        ["image-text", "image text", "processor", "tokenizer", "clip", "blip", "llava", "multimodal"],
    ),
    ("llm_nlp", ["tokenizer", "generate", "finetune", "datasets", "language model", "transformers"]),
    ("diffusion_generation", ["diffusion", "unet", "vae", "scheduler", "safetensors", "diffusers"]),
    ("graph_learning_recommender", ["graph", "node", "edge", "torch-geometric", "pyg", "dgl"]),
    ("reinforcement_learning_simulation", ["gym", "gymnasium", "mujoco", "atari", "simulator"]),
    ("scientific_geometry_3d", ["mesh", "point cloud", "nerf", "3dgs", "open3d", "trimesh", "cuda kernel"]),
]

FAILURE_RULES: dict[str, list[str]] = {
    "python_version_conflict": ["python 3.7", "python3.7", "python 3.8", "python3.8", "python_requires"],
    "torch_cuda_matrix": ["torch", "torchvision", "torchaudio", "cuda", "cu11", "cu12"],
    "native_extension_build": ["setup.py build", "cudaextension", "cpp_extension", "cmake", "ninja", "nvcc"],
    "system_package_missing": ["ffmpeg", "libsndfile", "libgl", "glib", "libjpeg", "libpng", "apt-get"],
    "hidden_dependency": ["importerror", "modulenotfounderror", "requirements"],
    "stale_readme": ["deprecated", "legacy", "not maintained", "old version"],
    "checkpoint_download": ["checkpoint", ".pth", ".pt", "weights", "pretrained", "download"],
    "dataset_asset_missing": ["dataset", "download data", "kaggle", "coco", "imagenet"],
    "entrypoint_ambiguity": ["train.py", "eval.py", "demo.py", "example", "script"],
    "verification_ambiguity": ["demo", "example", "quickstart", "inference"],
    "hardware_pressure": ["gpu", "vram", "cuda", "memory", "a100", "h100"],
    "docker_only_instruction": ["docker build", "docker run", "docker compose", "dockerfile"],
    "package_name_collision": ["pip install -e", "local package", "module name"],
    "api_or_license_gate": ["token", "license", "gated", "api key", "huggingface login"],
}

SKILL_ROUTE_BY_FAILURE: dict[str, str] = {
    "torch_cuda_matrix": "torch_cuda_matrix_solver",
    "native_extension_build": "native_extension_repair",
    "system_package_missing": "system_package_resolver",
    "checkpoint_download": "checkpoint_validation",
    "hidden_dependency": "hidden_dependency_inferer",
    "entrypoint_ambiguity": "minimal_entrypoint_discovery",
    "verification_ambiguity": "minimal_entrypoint_discovery",
}

SKILL_ROUTE_BY_MODALITY: dict[str, str] = {
    "audio_speech": "audio_env_setup",
    "object_detection_segmentation": "vision_segmentation_env_setup",
    "video_understanding_generation": "video_env_setup",
    "vision_language_multimodal": "vision_language_env_setup",
    "diffusion_generation": "diffusion_env_setup",
    "graph_learning_recommender": "graph_learning_env_setup",
    "reinforcement_learning_simulation": "rl_simulation_env_setup",
    "scientific_geometry_3d": "geometry_3d_env_setup",
}


def run_git(repo: Path, args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def read_text(path: Path, limit: int = 200_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except FileNotFoundError:
        return ""


def load_pyproject(repo: Path) -> dict[str, Any]:
    pyproject = repo / "pyproject.toml"
    if not pyproject.exists():
        return {}
    if tomllib is None:
        return {"_parse_error": "tomllib unavailable; run with Python 3.11+ for pyproject parsing"}
    try:
        with pyproject.open("rb") as f:
            return tomllib.load(f)
    except Exception as exc:
        return {"_parse_error": str(exc)}


def find_readme(repo: Path) -> Path | None:
    candidates = sorted(repo.glob("README*"))
    return candidates[0] if candidates else None


def detect_entrypoints(repo: Path, pyproject: dict[str, Any]) -> list[dict[str, str]]:
    entrypoints: list[dict[str, str]] = []
    scripts = pyproject.get("project", {}).get("scripts", {})
    for name, target in sorted(scripts.items()):
        entrypoints.append({"type": "console_script", "name": name, "target": str(target)})

    setup_py = read_text(repo / "setup.py")
    if "entry_points" in setup_py:
        entrypoints.append({"type": "setup_py_entry_points", "name": "setup.py", "target": "entry_points"})

    for pattern in ["scripts/*.py", "examples/*.py", "demo/*.py"]:
        for path in sorted(repo.glob(pattern))[:20]:
            entrypoints.append({"type": "script", "name": str(path.relative_to(repo)), "target": str(path)})

    packages = [p for p in repo.iterdir() if p.is_dir() and (p / "__init__.py").exists()]
    for pkg in sorted(packages)[:20]:
        entrypoints.append({"type": "import", "name": pkg.name, "target": pkg.name})

    return entrypoints


def detect_dependencies(repo: Path, pyproject: dict[str, Any], readme_text: str) -> dict[str, Any]:
    files = {
        "pyproject": (repo / "pyproject.toml").exists(),
        "setup_py": (repo / "setup.py").exists(),
        "requirements": sorted(str(p.relative_to(repo)) for p in repo.glob("requirements*.txt")),
        "environment": sorted(str(p.relative_to(repo)) for p in repo.glob("environment*.yml")),
    }

    py_deps = list(pyproject.get("project", {}).get("dependencies", []) or [])
    for req in repo.glob("requirements*.txt"):
        for line in read_text(req).splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                py_deps.append(line)

    lower = readme_text.lower()
    system = []
    for keyword in ["ffmpeg", "libgl", "gcc", "cmake", "cuda", "git-lfs"]:
        if keyword in lower:
            system.append(keyword)

    assets = []
    for keyword in ["checkpoint", ".pth", "download", "model weights", "weights"]:
        if keyword in lower:
            assets.append(keyword)

    return {
        "metadata_files": files,
        "python_dependencies": sorted(set(py_deps)),
        "system_dependency_keywords": sorted(set(system)),
        "asset_keywords": sorted(set(assets)),
    }


def keyword_score(corpus: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if keyword.lower() in corpus)


def detect_taxonomy_hints(
    repo: Path,
    readme_text: str,
    dependencies: dict[str, Any],
    entrypoints: list[dict[str, str]],
) -> dict[str, Any]:
    dep_text = " ".join(dependencies.get("python_dependencies", []))
    entry_text = " ".join(f"{item.get('name', '')} {item.get('target', '')}" for item in entrypoints)
    file_text = " ".join(str(path.relative_to(repo)) for path in repo.glob("*") if path.is_file())
    corpus = f"{repo.name} {readme_text} {dep_text} {entry_text} {file_text}".lower()

    modality_scores = [
        {"name": name, "score": keyword_score(corpus, keywords)}
        for name, keywords in MODALITY_RULES
    ]
    modality_scores = sorted(
        [item for item in modality_scores if item["score"] > 0],
        key=lambda item: (-item["score"], item["name"]),
    )
    primary_modality = modality_scores[0]["name"] if modality_scores else "unknown"

    challenges = [
        name
        for name, keywords in FAILURE_RULES.items()
        if keyword_score(corpus, keywords) > 0
    ]
    if dependencies.get("asset_keywords") and "checkpoint_download" not in challenges:
        challenges.append("checkpoint_download")
    if len(entrypoints) > 5 and "entrypoint_ambiguity" not in challenges:
        challenges.append("entrypoint_ambiguity")
    if not entrypoints and "verification_ambiguity" not in challenges:
        challenges.append("verification_ambiguity")

    verification_types: set[str] = set()
    if any(item["type"] in {"console_script", "setup_py_entry_points"} for item in entrypoints):
        verification_types.add("cli_help")
    if any(item["type"] == "import" for item in entrypoints):
        verification_types.add("import_graph")
    if any(word in corpus for word in ["demo", "example", "inference", "predict", "generate", "transcribe"]):
        verification_types.add("single_sample_inference")
    if "checkpoint_download" in challenges:
        verification_types.add("checkpoint_loading")
    if any(word in corpus for word in ["output", "mask", "json", "image", "artifact"]):
        verification_types.add("output_artifact")
    if not verification_types:
        verification_types.add("verification_ambiguity")

    routes = ["global_reproducibility_safety"]
    if primary_modality in SKILL_ROUTE_BY_MODALITY:
        routes.append(SKILL_ROUTE_BY_MODALITY[primary_modality])
    for challenge in challenges:
        route = SKILL_ROUTE_BY_FAILURE.get(challenge)
        if route:
            routes.append(route)
    routes.append("minimal_entrypoint_discovery")

    return {
        "modality": {
            "primary": primary_modality,
            "candidates": modality_scores,
        },
        "environment_challenges": sorted(set(challenges)),
        "verification_type": sorted(verification_types),
        "expected_skill_routes": sorted(set(routes), key=routes.index),
    }


def inspect_repo(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    pyproject = load_pyproject(repo)
    readme = find_readme(repo)
    readme_text = read_text(readme) if readme else ""
    entrypoints = detect_entrypoints(repo, pyproject)
    dependencies = detect_dependencies(repo, pyproject, readme_text)
    return {
        "repo_name": repo.name,
        "repo_path": str(repo),
        "repo_commit": run_git(repo, ["rev-parse", "--short", "HEAD"]),
        "remote_url": run_git(repo, ["remote", "get-url", "origin"]),
        "readme": str(readme.relative_to(repo)) if readme else None,
        "package_manager": "pip" if any((repo / name).exists() for name in ["pyproject.toml", "setup.py"]) else "unknown",
        "entrypoints": entrypoints,
        "dependencies": dependencies,
        "taxonomy_hints": detect_taxonomy_hints(repo, readme_text, dependencies, entrypoints),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = inspect_repo(args.repo)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
