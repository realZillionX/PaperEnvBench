from __future__ import annotations

import argparse
import hashlib
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


TASKS = {
    "detectron2_maskrcnn_minimal": {
        "framework": "detectron2",
        "repo_marker": "detectron2/__init__.py",
        "required_distributions": ["torch", "torchvision", "detectron2", "fvcore", "iopath", "pycocotools"],
        "required_imports": ["torch", "torchvision", "detectron2", "fvcore", "iopath", "pycocotools"],
        "native_evidence": "detectron2._C imports and detectron2.layers.nms executes on a CUDA tensor.",
    },
    "groundingdino_phrase_grounding_minimal": {
        "framework": "groundingdino",
        "repo_marker": "groundingdino/__init__.py",
        "required_distributions": [
            "torch",
            "torchvision",
            "groundingdino",
            "transformers",
            "timm",
            "supervision",
            "pycocotools",
        ],
        "required_imports": [
            "torch",
            "torchvision",
            "groundingdino",
            "transformers",
            "timm",
            "supervision",
            "pycocotools",
        ],
        "native_evidence": "groundingdino._C imports, proving the custom GroundingDINO native ops are available.",
    },
}

GROUNDINGDINO_CHECKPOINT_SHA256 = "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799"
GROUNDINGDINO_CHECKPOINT_SIZE = 693997677


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            x = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
            y = x @ x.T
            torch.cuda.synchronize()
            payload["cuda_tensor_smoke"] = {"ok": True, "matmul_sum": float(y.sum().detach().cpu().item())}
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


def repo_probe(repo_dir: Path | None, task: str) -> dict[str, Any]:
    if repo_dir is None:
        return {"provided": False}
    marker = repo_dir / str(TASKS[task]["repo_marker"])
    return {
        "provided": True,
        "path": str(repo_dir),
        "commit": git_commit(repo_dir),
        "marker": str(marker),
        "marker_exists": marker.exists(),
    }


def detectron2_native_probe(run_cuda_op: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "package": import_probe("detectron2"),
        "extension_import": import_probe("detectron2._C"),
    }
    try:
        from detectron2.layers import ROIAlign, nms

        payload["layers_import"] = {"ok": True, "symbols": {"ROIAlign": repr(ROIAlign), "nms": repr(nms)}}
    except Exception as exc:
        payload["layers_import"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
        }
        return payload

    try:
        import torch
        from detectron2.layers import nms

        boxes = torch.tensor([[0.0, 0.0, 12.0, 12.0], [1.0, 1.0, 11.0, 11.0]], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8], dtype=torch.float32)
        keep = nms(boxes, scores, 0.5)
        payload["cpu_nms_smoke"] = {"ok": True, "keep": keep.detach().cpu().tolist()}
    except Exception as exc:
        payload["cpu_nms_smoke"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
        }

    if run_cuda_op:
        try:
            import torch
            from detectron2.layers import nms

            boxes = torch.tensor(
                [[0.0, 0.0, 12.0, 12.0], [1.0, 1.0, 11.0, 11.0]],
                dtype=torch.float32,
                device="cuda",
            )
            scores = torch.tensor([0.9, 0.8], dtype=torch.float32, device="cuda")
            keep = nms(boxes, scores, 0.5)
            torch.cuda.synchronize()
            payload["cuda_nms_smoke"] = {"ok": True, "keep": keep.detach().cpu().tolist()}
        except Exception as exc:
            payload["cuda_nms_smoke"] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            }
    return payload


def groundingdino_native_probe() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "package": import_probe("groundingdino"),
        "extension_import": import_probe("groundingdino._C"),
        "preprocess_caption": None,
    }
    try:
        from groundingdino.util.inference import preprocess_caption

        payload["preprocess_caption"] = {"ok": True, "value": preprocess_caption("red square")}
    except Exception as exc:
        payload["preprocess_caption"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
        }
    return payload


def checkpoint_probe(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"provided": False}
    payload: dict[str, Any] = {"provided": True, "path": str(path), "exists": path.exists()}
    if path.exists():
        payload["size_bytes"] = path.stat().st_size
        payload["size_matches"] = path.stat().st_size == GROUNDINGDINO_CHECKPOINT_SIZE
        payload["sha256"] = sha256_file(path)
        payload["sha256_matches"] = payload["sha256"] == GROUNDINGDINO_CHECKPOINT_SHA256
    return payload


def collect_common_blockers(
    torch_info: dict[str, Any],
    imports: dict[str, Any],
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

    if require_repo:
        if not repo.get("provided"):
            blockers.append({"code": "repo_dir_missing", "message": "Pass --repo-dir for source-tree evidence."})
        elif not repo.get("marker_exists"):
            blockers.append({"code": "repo_marker_missing", "message": str(repo.get("marker"))})
    return blockers


def probe_detectron2(torch_info: dict[str, Any], repo: dict[str, Any], require_repo: bool) -> dict[str, Any]:
    spec = TASKS["detectron2_maskrcnn_minimal"]
    imports = {name: import_probe(name) for name in spec["required_imports"] if name != "torch"}
    native = detectron2_native_probe(run_cuda_op=bool(torch_info.get("cuda_available")))
    blockers = collect_common_blockers(torch_info, imports, repo, require_repo)
    if not native.get("extension_import", {}).get("available"):
        blockers.append({"code": "detectron2_ext_missing", "message": str(native.get("extension_import", {}).get("error"))})
    if not native.get("layers_import", {}).get("ok"):
        blockers.append({"code": "detectron2_layers_import_failed", "message": str(native.get("layers_import", {}).get("error"))})
    if torch_info.get("cuda_available") and not native.get("cuda_nms_smoke", {}).get("ok"):
        blockers.append({"code": "detectron2_cuda_op_failed", "message": str(native.get("cuda_nms_smoke", {}).get("error"))})
    return {
        "task_id": "detectron2_maskrcnn_minimal",
        "framework": spec["framework"],
        "status": "pass" if not blockers else "blocked",
        "native_gpu_requirement": spec["native_evidence"],
        "required_distributions": {name: distribution_version(name) for name in spec["required_distributions"]},
        "imports": imports,
        "native": native,
        "repo": repo,
        "blockers": blockers,
    }


def probe_groundingdino(
    torch_info: dict[str, Any],
    repo: dict[str, Any],
    require_repo: bool,
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    spec = TASKS["groundingdino_phrase_grounding_minimal"]
    imports = {name: import_probe(name) for name in spec["required_imports"] if name != "torch"}
    native = groundingdino_native_probe()
    checkpoint = checkpoint_probe(checkpoint_path)
    blockers = collect_common_blockers(torch_info, imports, repo, require_repo)
    if not native.get("extension_import", {}).get("available"):
        blockers.append({"code": "groundingdino_ext_missing", "message": str(native.get("extension_import", {}).get("error"))})
    if checkpoint_path is not None:
        if not checkpoint.get("exists"):
            blockers.append({"code": "groundingdino_checkpoint_missing", "message": str(checkpoint_path)})
        elif not checkpoint.get("sha256_matches"):
            blockers.append({"code": "groundingdino_checkpoint_hash_mismatch", "message": str(checkpoint.get("sha256"))})
    return {
        "task_id": "groundingdino_phrase_grounding_minimal",
        "framework": spec["framework"],
        "status": "pass" if not blockers else "blocked",
        "native_gpu_requirement": spec["native_evidence"],
        "required_distributions": {name: distribution_version(name) for name in spec["required_distributions"]},
        "imports": imports,
        "native": native,
        "checkpoint": checkpoint,
        "repo": repo,
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Detectron2 and GroundingDINO native GPU dependencies.")
    parser.add_argument("--task", choices=["all", *TASKS.keys()], default="all", help="Task slice to probe.")
    parser.add_argument("--repo-dir", type=Path, help="Optional checked-out upstream repository for the selected task.")
    parser.add_argument("--require-repo", action="store_true", help="Treat missing source-tree evidence as a blocker.")
    parser.add_argument("--checkpoint-path", type=Path, help="Optional GroundingDINO checkpoint path to hash-check.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when any task is blocked.")
    args = parser.parse_args()

    torch_info = torch_probe()
    requested = list(TASKS) if args.task == "all" else [args.task]
    repo_dir = args.repo_dir.resolve() if args.repo_dir else None
    task_payloads: dict[str, Any] = {}
    for task in requested:
        repo = repo_probe(repo_dir, task)
        if task == "detectron2_maskrcnn_minimal":
            task_payloads[task] = probe_detectron2(torch_info, repo, args.require_repo)
        else:
            task_payloads[task] = probe_groundingdino(torch_info, repo, args.require_repo, args.checkpoint_path)

    payload = {
        "probe": "detectron_grounding_probe",
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
        "torch": torch_info,
        "tasks": task_payloads,
    }
    payload["status"] = "pass" if all(item["status"] == "pass" for item in task_payloads.values()) else "blocked"

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
