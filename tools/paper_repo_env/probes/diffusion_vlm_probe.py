from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_ORDER = [
    "stable_diffusion_text2img_minimal",
    "latent_diffusion_sample_minimal",
    "guided_diffusion_sample_minimal",
    "improved_diffusion_sample_minimal",
    "diffusers_tiny_pipeline_minimal",
    "blip_caption_minimal",
    "lavis_blip2_minimal",
    "llava_single_image_minimal",
    "imagebind_embedding_minimal",
]


@dataclass(frozen=True)
class TaskSpec:
    group: str
    required_modules: tuple[str, ...]
    optional_modules: tuple[str, ...]
    checkpoint_boundary: str
    recommended_probe: str
    min_vram_gb: float
    needs_hf_auth: bool = False
    large_checkpoint: bool = False


TASK_SPECS: dict[str, TaskSpec] = {
    "stable_diffusion_text2img_minimal": TaskSpec(
        group="diffusion",
        required_modules=("torch", "torchvision", "omegaconf", "einops", "transformers", "diffusers", "PIL"),
        optional_modules=("pytorch_lightning", "safetensors", "accelerate", "xformers", "cv2"),
        checkpoint_boundary="Stable Diffusion v1 weights are license gated; full inference needs accepted weights and a CUDA-capable route.",
        recommended_probe="import stack plus CUDA memory probe; do not download gated weights inside the probe.",
        min_vram_gb=8.0,
        needs_hf_auth=True,
        large_checkpoint=True,
    ),
    "latent_diffusion_sample_minimal": TaskSpec(
        group="diffusion",
        required_modules=("torch", "torchvision", "omegaconf", "einops", "PIL"),
        optional_modules=("pytorch_lightning", "safetensors", "accelerate", "xformers"),
        checkpoint_boundary="The text2img-large checkpoint is a large external model artifact; check-only can record route without loading it.",
        recommended_probe="import ldm dependencies and verify CUDA availability before full DDIM or PLMS sampling.",
        min_vram_gb=8.0,
        large_checkpoint=True,
    ),
    "guided_diffusion_sample_minimal": TaskSpec(
        group="diffusion",
        required_modules=("torch", "torchvision", "blobfile", "tqdm"),
        optional_modules=("numpy",),
        checkpoint_boundary="OpenAI ImageNet diffusion and classifier checkpoints are public but large; full route uses classifier_sample.py.",
        recommended_probe="import torch/blobfile and confirm enough GPU memory for 64x64 classifier-guided sampling.",
        min_vram_gb=8.0,
        large_checkpoint=True,
    ),
    "improved_diffusion_sample_minimal": TaskSpec(
        group="diffusion",
        required_modules=("torch", "blobfile", "numpy", "tqdm"),
        optional_modules=("requests",),
        checkpoint_boundary="Gold route uses a tiny deterministic checkpoint; public full-size checkpoints remain a large-checkpoint boundary.",
        recommended_probe="import improved-diffusion dependencies and run a tiny tensor allocation before sampling.",
        min_vram_gb=4.0,
    ),
    "diffusers_tiny_pipeline_minimal": TaskSpec(
        group="diffusion",
        required_modules=("torch", "diffusers", "transformers", "huggingface_hub", "safetensors", "accelerate", "PIL"),
        optional_modules=("xformers",),
        checkpoint_boundary="hf-internal-testing/tiny-stable-diffusion-pipe is small and public; it still exercises HF cache and safetensors paths.",
        recommended_probe="import diffusers stack, inspect HF cache, then run the task verifier or tiny pipeline smoke.",
        min_vram_gb=0.0,
    ),
    "blip_caption_minimal": TaskSpec(
        group="vision_language",
        required_modules=("torch", "torchvision", "timm", "transformers", "fairscale"),
        optional_modules=("PIL", "numpy"),
        checkpoint_boundary="BLIP caption weights are large external checkpoints; check-only may use deterministic semantic fallback.",
        recommended_probe="import BLIP dependencies and confirm checkpoint path or documented fallback before single-image captioning.",
        min_vram_gb=6.0,
        large_checkpoint=True,
    ),
    "lavis_blip2_minimal": TaskSpec(
        group="vision_language",
        required_modules=("torch", "torchvision", "timm", "transformers", "omegaconf"),
        optional_modules=("PIL", "accelerate", "sentencepiece"),
        checkpoint_boundary="BLIP-2 routes pair a vision encoder with a large language model; full inference can exceed CPU and low-VRAM limits.",
        recommended_probe="import LAVIS/transformers config stack and verify GPU memory before loading FLAN-T5 variants.",
        min_vram_gb=16.0,
        large_checkpoint=True,
    ),
    "llava_single_image_minimal": TaskSpec(
        group="vision_language",
        required_modules=("torch", "torchvision", "transformers", "sentencepiece", "PIL"),
        optional_modules=("accelerate", "safetensors"),
        checkpoint_boundary="LLaVA full inference needs a base LLM plus vision adapter; some checkpoints are gated or license constrained.",
        recommended_probe="import transformers/sentencepiece and check HF auth/cache before any model download.",
        min_vram_gb=16.0,
        needs_hf_auth=True,
        large_checkpoint=True,
    ),
    "imagebind_embedding_minimal": TaskSpec(
        group="vision_language",
        required_modules=("torch", "torchvision", "torchaudio", "pytorchvideo", "timm"),
        optional_modules=("PIL", "numpy"),
        checkpoint_boundary="ImageBind huge checkpoint is non-commercial licensed and large; random-init evidence must be recorded as non-checkpoint evidence.",
        recommended_probe="import multimodal torch stack, confirm audio/video libraries, and record license/checkpoint boundary.",
        min_vram_gb=8.0,
        large_checkpoint=True,
    ),
}


MODULE_TO_DISTRIBUTION = {
    "PIL": "Pillow",
    "cv2": "opencv-python-headless",
    "huggingface_hub": "huggingface-hub",
    "pytorch_lightning": "pytorch-lightning",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_text(command: list[str], timeout: int = 30) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=timeout)
        return completed.returncode, (completed.stdout + completed.stderr).strip()
    except Exception as exc:
        return 127, repr(exc)


def module_probe(module_names: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in module_names:
        distribution = MODULE_TO_DISTRIBUTION.get(name, name)
        try:
            spec = importlib.util.find_spec(name)
            if spec is None:
                raise ModuleNotFoundError(f"No module named {name!r}")
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                version = None
            result[name] = {
                "available": True,
                "version": version,
                "path": spec.origin,
            }
        except Exception as exc:
            result[name] = {"available": False, "error": repr(exc)}
    return result


def parse_memory_mib(text: str) -> list[dict[str, Any]]:
    gpus = []
    for idx, line in enumerate(line.strip() for line in text.splitlines() if line.strip()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            memory_total_mib = int(parts[1])
            memory_free_mib = int(parts[2])
        except ValueError:
            memory_total_mib = None
            memory_free_mib = None
        gpus.append(
            {
                "index": idx,
                "name": parts[0],
                "memory_total_mib": memory_total_mib,
                "memory_free_mib": memory_free_mib,
                "driver_version": parts[3],
            }
        )
    return gpus


def nvidia_smi_probe() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "returncode": 127, "gpus": [], "raw": "nvidia-smi not found"}
    rc, text = run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    return {"available": rc == 0, "returncode": rc, "gpus": parse_memory_mib(text) if rc == 0 else [], "raw": text}


def torch_probe() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": repr(exc)}

    payload: dict[str, Any] = {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_compiled": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        payload["devices"] = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free_bytes = None
            total_bytes = None
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            except Exception:
                total_bytes = getattr(props, "total_memory", None)
            payload["devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_gb": round(float(total_bytes or 0) / (1024**3), 3),
                    "free_memory_gb": round(float(free_bytes or 0) / (1024**3), 3) if free_bytes is not None else None,
                }
            )
        try:
            x = torch.arange(16, dtype=torch.float32, device="cuda:0").reshape(4, 4)
            y = x @ x.T
            torch.cuda.synchronize()
            payload["cuda_tensor_smoke"] = {"ok": True, "matmul_sum": float(y.sum().detach().cpu().item())}
        except Exception as exc:
            payload["cuda_tensor_smoke"] = {"ok": False, "error": repr(exc)}
    return payload


def hf_boundary_probe() -> dict[str, Any]:
    env_keys = ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"]
    cache_candidates = []
    for key in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(key)
        if value:
            cache_candidates.append(Path(value))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cache_candidates.append(Path(hf_home) / "hub")
    else:
        cache_candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    return {
        "token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")),
        "env": {key: bool(os.environ.get(key)) for key in env_keys},
        "cache_candidates": [
            {
                "path": str(path),
                "exists": path.exists(),
                "is_dir": path.is_dir(),
            }
            for path in cache_candidates
        ],
    }


def model_root_probe(paths: list[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        entry: dict[str, Any] = {"path": str(path), "exists": path.exists(), "is_dir": path.is_dir()}
        if path.is_file():
            entry["size_bytes"] = path.stat().st_size
        elif path.is_dir():
            files = [child for child in path.rglob("*") if child.is_file()]
            entry["file_count"] = len(files)
            entry["total_size_bytes"] = sum(child.stat().st_size for child in files[:5000])
            entry["truncated_size_scan"] = len(files) > 5000
        result.append(entry)
    return result


def package_names_for_tasks(task_ids: list[str]) -> list[str]:
    names = set()
    for task_id in task_ids:
        spec = TASK_SPECS[task_id]
        names.update(spec.required_modules)
        names.update(spec.optional_modules)
    return sorted(names)


def best_free_vram_gb(torch_payload: dict[str, Any], smi_payload: dict[str, Any]) -> float | None:
    torch_devices = torch_payload.get("devices") or []
    free_values = [device.get("free_memory_gb") for device in torch_devices if device.get("free_memory_gb") is not None]
    if free_values:
        return float(max(free_values))
    smi_values = [
        gpu.get("memory_free_mib") / 1024
        for gpu in smi_payload.get("gpus", [])
        if isinstance(gpu.get("memory_free_mib"), int)
    ]
    if smi_values:
        return float(max(smi_values))
    return None


def make_blocker(
    *,
    task_id: str,
    code: str,
    severity: str,
    message: str,
    remediation: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "code": code,
        "severity": severity,
        "message": message,
        "remediation": remediation,
        "evidence": evidence or {},
    }


def task_report(
    task_id: str,
    modules: dict[str, dict[str, Any]],
    torch_payload: dict[str, Any],
    smi_payload: dict[str, Any],
    hf_payload: dict[str, Any],
) -> dict[str, Any]:
    spec = TASK_SPECS[task_id]
    blockers = []
    missing_required = [name for name in spec.required_modules if not modules.get(name, {}).get("available")]
    missing_optional = [name for name in spec.optional_modules if not modules.get(name, {}).get("available")]

    if missing_required:
        blockers.append(
            make_blocker(
                task_id=task_id,
                code="missing_required_python_modules",
                severity="error",
                message="Required Python modules are not importable.",
                remediation="Install the task requirements in the active venv, then rerun this probe.",
                evidence={"modules": missing_required},
            )
        )
    if missing_optional:
        blockers.append(
            make_blocker(
                task_id=task_id,
                code="missing_optional_acceleration_modules",
                severity="warning",
                message="Optional acceleration or serialization modules are not importable.",
                remediation="Install these modules when running the full GPU route; check-only fallbacks may not require them.",
                evidence={"modules": missing_optional},
            )
        )
    if spec.min_vram_gb > 0:
        free_vram = best_free_vram_gb(torch_payload, smi_payload)
        if torch_payload.get("cuda_available") is not True:
            blockers.append(
                make_blocker(
                    task_id=task_id,
                    code="cuda_not_available",
                    severity="warning",
                    message="Torch CUDA is not available, so the CUDA-visible inference route is not proven.",
                    remediation="Use a CUDA-enabled PyTorch wheel on a GPU node before claiming full GPU inference.",
                    evidence={"torch_cuda_available": torch_payload.get("cuda_available")},
                )
            )
        elif free_vram is not None and free_vram < spec.min_vram_gb:
            blockers.append(
                make_blocker(
                    task_id=task_id,
                    code="insufficient_free_vram",
                    severity="error",
                    message="Detected free GPU memory is below the recommended boundary for this task.",
                    remediation="Use a larger GPU, reduce batch size, or run only the CPU/checkpoint-boundary contract.",
                    evidence={"free_vram_gb": round(free_vram, 3), "recommended_min_vram_gb": spec.min_vram_gb},
                )
            )
    if spec.needs_hf_auth and not hf_payload["token_present"]:
        blockers.append(
            make_blocker(
                task_id=task_id,
                code="hf_auth_or_license_gate_unconfirmed",
                severity="warning",
                message="No Hugging Face token environment variable is visible for gated or license-constrained assets.",
                remediation="Pass HF_TOKEN or HUGGINGFACE_HUB_TOKEN at runtime after accepting the required model license.",
                evidence={"token_present": False},
            )
        )
    if spec.large_checkpoint:
        blockers.append(
            make_blocker(
                task_id=task_id,
                code="large_checkpoint_boundary",
                severity="info",
                message=spec.checkpoint_boundary,
                remediation="Record checkpoint path, license/source, checksum when available, and whether full weights were loaded.",
            )
        )

    return {
        "task_id": task_id,
        "group": spec.group,
        "recommended_probe": spec.recommended_probe,
        "checkpoint_boundary": spec.checkpoint_boundary,
        "min_vram_gb": spec.min_vram_gb,
        "required_modules": {name: modules.get(name, {"available": False}) for name in spec.required_modules},
        "optional_modules": {name: modules.get(name, {"available": False}) for name in spec.optional_modules},
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe diffusion and vision-language dependency boundaries for selected PaperEnvBench tasks."
    )
    parser.add_argument("--json", action="store_true", help="Print structured JSON output.")
    parser.add_argument("--output", type=Path, help="Optional path for JSON output.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when any selected task has an error blocker.")
    parser.add_argument(
        "--task",
        action="append",
        choices=TASK_ORDER,
        help="Task id to probe. May be passed multiple times. Defaults to all diffusion/VLM tasks in this slice.",
    )
    parser.add_argument(
        "--model-root",
        action="append",
        type=Path,
        default=[],
        help="Optional local checkpoint/cache path to summarize without reading model contents.",
    )
    args = parser.parse_args()

    task_ids = args.task or TASK_ORDER
    smi_payload = nvidia_smi_probe()
    torch_payload = torch_probe()
    hf_payload = hf_boundary_probe()
    modules = module_probe(package_names_for_tasks(task_ids))
    tasks = [task_report(task_id, modules, torch_payload, smi_payload, hf_payload) for task_id in task_ids]
    blockers = [blocker for task in tasks for blocker in task["blockers"]]
    blocker_counts: dict[str, int] = {"error": 0, "warning": 0, "info": 0}
    for blocker in blockers:
        blocker_counts[blocker["severity"]] = blocker_counts.get(blocker["severity"], 0) + 1

    payload = {
        "generated_at": utc_now(),
        "probe": "diffusion_vlm_probe",
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "selected_tasks": task_ids,
        "runtime": {
            "nvidia_smi": smi_payload,
            "torch": torch_payload,
            "hf_boundary": hf_payload,
            "model_roots": model_root_probe(args.model_root),
        },
        "modules": modules,
        "tasks": tasks,
        "blockers": blockers,
        "summary": {
            "task_count": len(tasks),
            "blocker_counts": blocker_counts,
            "has_error_blocker": blocker_counts.get("error", 0) > 0,
        },
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"probe=diffusion_vlm_probe tasks={len(tasks)}")
        print(
            "torch_available="
            f"{torch_payload.get('available')} cuda_available={torch_payload.get('cuda_available')} "
            f"gpu_count={len(smi_payload.get('gpus', []))}"
        )
        print(
            "blockers="
            f"errors:{blocker_counts.get('error', 0)} "
            f"warnings:{blocker_counts.get('warning', 0)} "
            f"info:{blocker_counts.get('info', 0)}"
        )
        for task in tasks:
            error_count = sum(1 for blocker in task["blockers"] if blocker["severity"] == "error")
            warning_count = sum(1 for blocker in task["blockers"] if blocker["severity"] == "warning")
            print(f"{task['task_id']}: errors={error_count} warnings={warning_count}")

    return 1 if args.strict and payload["summary"]["has_error_blocker"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
