from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OPENMMLAB_TASKS = {
    "mmdetection_fasterrcnn_minimal": {
        "framework": "mmdet",
        "repo_marker": "mmdet/__init__.py",
        "config": "configs/_base_/models/faster-rcnn_r50_fpn.py",
        "required_distributions": ["torch", "torchvision", "mmengine", "mmcv", "mmdet"],
        "required_imports": ["torch", "torchvision", "mmengine", "mmcv", "mmdet"],
        "native_ops": ["nms", "RoIAlign"],
        "gpu_evidence": "CUDA torch plus native mmcv._ext executing mmcv.ops.nms on a CUDA tensor.",
    },
    "mmsegmentation_segformer_minimal": {
        "framework": "mmseg",
        "repo_marker": "mmseg/__init__.py",
        "config": "configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py",
        "required_distributions": ["torch", "torchvision", "mmengine", "mmcv", "mmsegmentation"],
        "required_imports": ["torch", "torchvision", "mmengine", "mmcv", "mmseg"],
        "native_ops": ["nms"],
        "gpu_evidence": "CUDA torch, full mmcv native extension, and SegFormer config loading without mmcv.ops stubs.",
    },
    "mmaction2_recognition_minimal": {
        "framework": "mmaction",
        "repo_marker": "mmaction/__init__.py",
        "config": "configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py",
        "required_distributions": ["torch", "torchvision", "mmengine", "mmcv", "mmaction2", "decord"],
        "required_imports": ["torch", "torchvision", "mmengine", "mmcv", "mmaction", "decord"],
        "native_ops": ["nms"],
        "gpu_evidence": "CUDA torch, full mmcv native extension, video decode dependency, and TSN config loading.",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_text(command: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=timeout)
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"returncode": 127, "stdout": "", "stderr": repr(exc)}


def git_commit(repo_dir: Path) -> str | None:
    if not (repo_dir / ".git").exists():
        return None
    result = run_text(["git", "-C", str(repo_dir), "rev-parse", "HEAD"])
    if result["returncode"] == 0:
        return str(result["stdout"]).strip()
    return None


def distribution_version(name: str) -> dict[str, Any]:
    try:
        return {"installed": True, "version": metadata.version(name)}
    except metadata.PackageNotFoundError:
        return {"installed": False, "version": None}
    except Exception as exc:
        return {"installed": False, "version": None, "error": repr(exc)}


def import_probe(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {
            "available": True,
            "version": getattr(module, "__version__", None),
            "path": getattr(module, "__file__", None),
        }
    except Exception as exc:
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
        }


def torch_probe() -> dict[str, Any]:
    result = import_probe("torch")
    if not result.get("available"):
        return result
    import torch

    payload: dict[str, Any] = {
        **result,
        "cuda_compiled": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        payload["device_name"] = torch.cuda.get_device_name(0)
        payload["device_capability"] = list(torch.cuda.get_device_capability(0))
        try:
            tensor = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
            value = tensor.mm(tensor.T).sum()
            torch.cuda.synchronize()
            payload["cuda_tensor_smoke"] = {"ok": True, "matmul_sum": float(value.detach().cpu().item())}
        except Exception as exc:
            payload["cuda_tensor_smoke"] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            }
    return payload


def nvidia_smi_probe() -> dict[str, Any]:
    query = run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    gpus = []
    if query["returncode"] == 0:
        for line in str(query["stdout"]).splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                gpus.append({"name": parts[0], "memory_total": parts[1], "driver_version": parts[2]})
    return {"returncode": query["returncode"], "stderr": query["stderr"], "gpus": gpus}


def mmcv_native_probe(run_cuda_op: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mmcv": import_probe("mmcv"),
        "mmcv_distribution": distribution_version("mmcv"),
        "mmcv_lite_distribution": distribution_version("mmcv-lite"),
        "extension_import": import_probe("mmcv._ext"),
    }
    try:
        from mmcv.ops import RoIAlign, nms

        payload["ops_import"] = {
            "ok": True,
            "symbols": {"RoIAlign": repr(RoIAlign), "nms": repr(nms)},
        }
    except Exception as exc:
        payload["ops_import"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
        }
        return payload

    try:
        import torch
        from mmcv.ops import nms

        boxes = torch.tensor([[0.0, 0.0, 12.0, 12.0], [1.0, 1.0, 11.0, 11.0]], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
        kept = nms(boxes, scores, 0.5)
        payload["cpu_nms_smoke"] = {"ok": True, "repr": repr(kept)}
    except Exception as exc:
        payload["cpu_nms_smoke"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
        }

    if run_cuda_op:
        try:
            import torch
            from mmcv.ops import nms

            boxes = torch.tensor(
                [[0.0, 0.0, 12.0, 12.0], [1.0, 1.0, 11.0, 11.0]],
                dtype=torch.float32,
                device="cuda",
            )
            scores = torch.tensor([0.9, 0.8], dtype=torch.float32, device="cuda")
            kept = nms(boxes, scores, 0.5)
            torch.cuda.synchronize()
            payload["cuda_nms_smoke"] = {"ok": True, "kept": kept.detach().cpu().tolist()}
        except Exception as exc:
            payload["cuda_nms_smoke"] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            }
    return payload


def repo_probe(repo_dir: Path | None, task: str) -> dict[str, Any]:
    if repo_dir is None:
        return {"provided": False}
    spec = OPENMMLAB_TASKS[task]
    marker = repo_dir / str(spec["repo_marker"])
    config = repo_dir / str(spec["config"])
    payload: dict[str, Any] = {
        "provided": True,
        "path": str(repo_dir),
        "commit": git_commit(repo_dir),
        "marker_exists": marker.exists(),
        "marker": str(marker),
        "config_exists": config.exists(),
        "config": str(config),
    }
    if config.exists():
        try:
            from mmengine.config import Config

            cfg = Config.fromfile(config)
            payload["config_load"] = {
                "ok": True,
                "model_type": cfg.model.get("type") if hasattr(cfg, "model") else None,
            }
        except Exception as exc:
            payload["config_load"] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            }
    return payload


def collect_blockers(
    task: str,
    torch_info: dict[str, Any],
    imports: dict[str, Any],
    mmcv_native: dict[str, Any],
    repo: dict[str, Any],
    require_repo: bool,
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    if not nvidia_smi_probe()["gpus"]:
        blockers.append({"code": "nvidia_smi_no_gpu", "message": "nvidia-smi did not report a GPU."})
    if not torch_info.get("available"):
        blockers.append({"code": "torch_import_failed", "message": str(torch_info.get("error"))})
    else:
        if not torch_info.get("cuda_compiled"):
            blockers.append({"code": "torch_cpu_wheel", "message": "Installed torch does not report a CUDA build."})
        if not torch_info.get("cuda_available"):
            blockers.append({"code": "torch_cuda_unavailable", "message": "torch.cuda.is_available() is false."})

    for name, result in imports.items():
        if not result.get("available"):
            blockers.append({"code": f"import_failed:{name}", "message": str(result.get("error"))})

    if mmcv_native.get("mmcv_lite_distribution", {}).get("installed"):
        blockers.append({"code": "mmcv_lite_installed", "message": "mmcv-lite is installed; native mmcv._ext is not guaranteed."})
    if not mmcv_native.get("extension_import", {}).get("available"):
        blockers.append({"code": "mmcv_ext_missing", "message": str(mmcv_native.get("extension_import", {}).get("error"))})
    if not mmcv_native.get("ops_import", {}).get("ok"):
        blockers.append({"code": "mmcv_ops_import_failed", "message": str(mmcv_native.get("ops_import", {}).get("error"))})
    if torch_info.get("cuda_available") and not mmcv_native.get("cuda_nms_smoke", {}).get("ok"):
        blockers.append({"code": "mmcv_cuda_op_failed", "message": str(mmcv_native.get("cuda_nms_smoke", {}).get("error"))})

    if require_repo:
        if not repo.get("provided"):
            blockers.append({"code": "repo_dir_missing", "message": "Pass --repo-dir for config and source-tree evidence."})
        elif not repo.get("marker_exists"):
            blockers.append({"code": "repo_marker_missing", "message": str(repo.get("marker"))})
        elif not repo.get("config_exists"):
            blockers.append({"code": "repo_config_missing", "message": str(repo.get("config"))})
        elif repo.get("config_load", {}).get("ok") is False:
            blockers.append({"code": "repo_config_load_failed", "message": str(repo.get("config_load", {}).get("error"))})

    expected_ops = ", ".join(OPENMMLAB_TASKS[task]["native_ops"])
    if not expected_ops:
        blockers.append({"code": "internal_task_spec_missing", "message": "Task native op list is empty."})
    return blockers


def probe_task(task: str, repo_dir: Path | None, require_repo: bool) -> dict[str, Any]:
    spec = OPENMMLAB_TASKS[task]
    torch_info = torch_probe()
    imports = {name: import_probe(name) for name in spec["required_imports"] if name != "torch"}
    mmcv_native = mmcv_native_probe(run_cuda_op=bool(torch_info.get("cuda_available")))
    repo = repo_probe(repo_dir, task)
    blockers = collect_blockers(task, torch_info, imports, mmcv_native, repo, require_repo)
    return {
        "task_id": task,
        "framework": spec["framework"],
        "status": "pass" if not blockers else "blocked",
        "native_gpu_requirement": spec["gpu_evidence"],
        "required_distributions": {name: distribution_version(name) for name in spec["required_distributions"]},
        "torch": torch_info,
        "imports": imports,
        "mmcv_native": mmcv_native,
        "repo": repo,
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe native GPU dependencies for OpenMMLab PaperEnvBench tasks.")
    parser.add_argument(
        "--task",
        choices=["all", *OPENMMLAB_TASKS.keys()],
        default="all",
        help="Task slice to probe.",
    )
    parser.add_argument("--repo-dir", type=Path, help="Optional checked-out upstream repository for the selected task.")
    parser.add_argument("--require-repo", action="store_true", help="Treat missing repo/config evidence as a blocker.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when any task is blocked.")
    args = parser.parse_args()

    tasks = list(OPENMMLAB_TASKS) if args.task == "all" else [args.task]
    payload = {
        "probe": "openmmlab_native_probe",
        "generated_at": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "executables": {
            "nvidia-smi": shutil.which("nvidia-smi"),
            "nvcc": shutil.which("nvcc"),
            "gcc": shutil.which("gcc"),
            "g++": shutil.which("g++"),
            "ninja": shutil.which("ninja"),
        },
        "nvidia_smi": nvidia_smi_probe(),
        "tasks": {task: probe_task(task, args.repo_dir.resolve() if args.repo_dir else None, args.require_repo) for task in tasks},
    }
    payload["status"] = "pass" if all(item["status"] == "pass" for item in payload["tasks"].values()) else "blocked"

    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.json:
        print(text)
    else:
        print(f"status={payload['status']}")
        for task, result in payload["tasks"].items():
            print(f"{task}: {result['status']} blockers={len(result['blockers'])}")
            for blocker in result["blockers"]:
                print(f"  - {blocker['code']}: {blocker['message']}")

    return 1 if args.strict and payload["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
