#!/usr/bin/env python3
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
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_IDS = (
    "gaussian_splatting_scene_minimal",
    "openpcdet_pointcloud_minimal",
    "nerfstudio_nerfacto_minimal",
    "open3d_pointcloud_minimal",
)


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


def blocker(
    code: str,
    message: str,
    *,
    severity: str = "blocker",
    evidence: Any | None = None,
    remediation: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "severity": severity, "message": message}
    if evidence is not None:
        payload["evidence"] = evidence
    if remediation:
        payload["remediation"] = remediation
    return payload


def version_for(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def import_probe(module_name: str, *, distribution: str | None = None, symbols: list[str] | None = None) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "module": module_name,
            "available": False,
            "error": repr(exc),
            "traceback_tail": traceback.format_exc(limit=2).strip().splitlines()[-3:],
            "distribution_version": version_for(distribution or module_name.split(".")[0]),
        }

    missing_symbols = [symbol for symbol in symbols or [] if not hasattr(module, symbol)]
    return {
        "module": module_name,
        "available": True,
        "path": getattr(module, "__file__", None),
        "distribution_version": version_for(distribution or module_name.split(".")[0]),
        "missing_symbols": missing_symbols,
        "symbols_ok": not missing_symbols,
    }


def nvidia_smi_probe() -> dict[str, Any]:
    query = run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader",
        ]
    )
    gpus: list[dict[str, Any]] = []
    if query["returncode"] == 0:
        for line in query["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 4:
                gpus.append(
                    {
                        "name": parts[0],
                        "memory_total": parts[1],
                        "driver_version": parts[2],
                        "compute_capability": parts[3],
                    }
                )
    return {
        "available": bool(gpus),
        "gpus": gpus,
        "query": query,
        "header": run_text(["nvidia-smi"], timeout=15),
    }


def nvcc_probe() -> dict[str, Any]:
    nvcc_path = shutil.which("nvcc")
    if not nvcc_path:
        return {"available": False, "path": None, "version": None}
    version = run_text([nvcc_path, "--version"], timeout=15)
    return {"available": version["returncode"] == 0, "path": nvcc_path, "version": version}


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
        try:
            x = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
            y = x @ x.T
            torch.cuda.synchronize()
            payload.update(
                {
                    "device_name": torch.cuda.get_device_name(0),
                    "capability": list(torch.cuda.get_device_capability(0)),
                    "matmul_sum": float(y.sum().detach().cpu().item()),
                    "smoke_ok": True,
                }
            )
        except Exception as exc:
            payload.update({"smoke_ok": False, "smoke_error": repr(exc)})
    return payload


def runtime_probe() -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "nvidia_smi": nvidia_smi_probe(),
        "nvcc": nvcc_probe(),
        "torch": torch_probe(),
    }


def runtime_blockers(runtime: dict[str, Any], *, require_nvcc: bool = False) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if not runtime["nvidia_smi"]["available"]:
        blockers.append(
            blocker(
                "gpu_runtime_missing",
                "No NVIDIA GPU is visible through nvidia-smi.",
                evidence=runtime["nvidia_smi"]["query"],
                remediation="Run on a CUDA-capable worker with a visible NVIDIA driver.",
            )
        )
    torch_info = runtime["torch"]
    if not torch_info.get("available"):
        blockers.append(
            blocker(
                "torch_missing",
                "PyTorch is not importable.",
                evidence=torch_info.get("error"),
                remediation="Install a CUDA-enabled PyTorch wheel matching the target driver and Python version.",
            )
        )
    elif not torch_info.get("cuda_available"):
        blockers.append(
            blocker(
                "torch_cuda_unavailable",
                "PyTorch imports, but torch.cuda.is_available() is false.",
                evidence={"torch_version": torch_info.get("version"), "torch_cuda": torch_info.get("cuda_compiled")},
                remediation="Install a CUDA PyTorch wheel and verify that the driver can initialize CUDA.",
            )
        )
    if require_nvcc and not runtime["nvcc"]["available"]:
        blockers.append(
            blocker(
                "cuda_toolkit_missing",
                "nvcc is not available, so source CUDA extensions may fail to build.",
                evidence=runtime["nvcc"],
                remediation="Install a CUDA toolkit compatible with the PyTorch CUDA ABI, or use prebuilt extension wheels.",
            )
        )
    return blockers


def module_blockers(probes: list[dict[str, Any]], remediation: str) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for probe in probes:
        if not probe.get("available"):
            blockers.append(
                blocker(
                    "python_module_missing",
                    f"{probe['module']} is not importable.",
                    evidence=probe,
                    remediation=remediation,
                )
            )
        elif probe.get("missing_symbols"):
            blockers.append(
                blocker(
                    "python_module_symbol_missing",
                    f"{probe['module']} imports but lacks required symbols.",
                    evidence=probe,
                    remediation=remediation,
                )
            )
    return blockers


def inspect_openpcdet_setup(repo_dir: Path | None) -> dict[str, Any]:
    if repo_dir is None:
        return {"available": False, "reason": "repo_dir_not_provided"}
    setup_py = repo_dir / "setup.py"
    if not setup_py.exists():
        return {"available": False, "repo_dir": str(repo_dir), "reason": "setup_py_missing"}
    text = setup_py.read_text(encoding="utf-8", errors="replace")
    names = [
        "iou3d_nms_cuda",
        "roiaware_pool3d_cuda",
        "roipoint_pool3d_cuda",
        "pointnet2_stack_cuda",
        "pointnet2_batch_cuda",
        "bev_pool_ext",
        "ingroup_inds_cuda",
    ]
    return {
        "available": True,
        "repo_dir": str(repo_dir),
        "uses_cuda_extension": "CUDAExtension" in text,
        "declared_extensions": [name for name in names if name in text],
    }


def gaussian_probe(runtime: dict[str, Any]) -> dict[str, Any]:
    imports = [
        import_probe(
            "diff_gaussian_rasterization",
            distribution="diff-gaussian-rasterization",
            symbols=["GaussianRasterizationSettings", "GaussianRasterizer"],
        ),
        import_probe("simple_knn._C", distribution="simple-knn", symbols=["distCUDA2"]),
    ]
    blockers = runtime_blockers(runtime, require_nvcc=False)
    blockers.extend(
        module_blockers(
            imports,
            "Install the Gaussian Splatting submodules with the same CUDA ABI as PyTorch: "
            "submodules/diff-gaussian-rasterization and submodules/simple-knn.",
        )
    )
    return {
        "task_id": "gaussian_splatting_scene_minimal",
        "dependency_boundary": [
            "diff_gaussian_rasterization CUDA rasterizer",
            "simple_knn._C distCUDA2 extension",
            "CUDA-enabled PyTorch runtime",
        ],
        "ok": not blockers,
        "status": "pass" if not blockers else "blocked",
        "imports": imports,
        "blockers": blockers,
        "success_evidence": {
            "torch_cuda_smoke": runtime["torch"].get("smoke_ok") is True,
            "rasterizer_symbols": imports[0].get("symbols_ok") is True,
            "simple_knn_distCUDA2": imports[1].get("symbols_ok") is True,
        },
    }


def openpcdet_probe(runtime: dict[str, Any], repo_dir: Path | None) -> dict[str, Any]:
    if repo_dir is not None and repo_dir.exists():
        sys.path.insert(0, str(repo_dir))
    imports = [
        import_probe("pcdet.ops.iou3d_nms.iou3d_nms_cuda"),
        import_probe("pcdet.ops.roiaware_pool3d.roiaware_pool3d_cuda"),
        import_probe("pcdet.ops.roipoint_pool3d.roipoint_pool3d_cuda"),
        import_probe("pcdet.ops.pointnet2.pointnet2_stack.pointnet2_stack_cuda"),
        import_probe("pcdet.ops.pointnet2.pointnet2_batch.pointnet2_batch_cuda"),
        import_probe("pcdet.ops.bev_pool.bev_pool_ext"),
        import_probe("pcdet.ops.ingroup_inds.ingroup_inds_cuda"),
    ]
    setup = inspect_openpcdet_setup(repo_dir)
    blockers = runtime_blockers(runtime, require_nvcc=True)
    if setup.get("available") and not setup.get("uses_cuda_extension"):
        blockers.append(
            blocker(
                "openpcdet_setup_cudaextension_missing",
                "OpenPCDet setup.py did not declare CUDAExtension.",
                evidence=setup,
                remediation="Check that repo_dir points at the pinned OpenPCDet checkout.",
            )
        )
    blockers.extend(
        module_blockers(
            imports,
            "Build OpenPCDet with `python setup.py develop` or `pip install -e .` after CUDA-enabled PyTorch and nvcc are available.",
        )
    )
    imported = [probe["module"] for probe in imports if probe.get("available")]
    return {
        "task_id": "openpcdet_pointcloud_minimal",
        "dependency_boundary": [
            "OpenPCDet setup.py CUDAExtension build surface",
            "iou3d / roiaware / roipoint / pointnet2 / bev_pool / ingroup CUDA ops",
            "CUDA-enabled PyTorch plus nvcc for source extension builds",
        ],
        "ok": not blockers,
        "status": "pass" if not blockers else "blocked",
        "setup_py": setup,
        "imports": imports,
        "blockers": blockers,
        "success_evidence": {
            "torch_cuda_smoke": runtime["torch"].get("smoke_ok") is True,
            "nvcc_available": runtime["nvcc"].get("available") is True,
            "setup_uses_cuda_extension": setup.get("uses_cuda_extension") is True,
            "imported_extension_count": len(imported),
            "imported_extensions": imported,
        },
    }


def nerfstudio_probe(runtime: dict[str, Any]) -> dict[str, Any]:
    imports = [
        import_probe("nerfstudio", distribution="nerfstudio"),
        import_probe("nerfacc", distribution="nerfacc"),
        import_probe("gsplat", distribution="gsplat"),
        import_probe("tinycudann", distribution="tiny-cuda-nn"),
        import_probe("open3d", distribution="open3d"),
        import_probe("xatlas", distribution="xatlas"),
        import_probe("pymeshlab", distribution="pymeshlab"),
    ]
    blockers = runtime_blockers(runtime, require_nvcc=False)
    required = [probe for probe in imports if probe["module"] in {"nerfstudio", "nerfacc", "gsplat", "tinycudann"}]
    blockers.extend(
        module_blockers(
            required,
            "Install Nerfstudio with CUDA-compatible nerfacc, gsplat, and tiny-cuda-nn wheels or build them against the active PyTorch CUDA ABI.",
        )
    )
    optional_missing = [
        probe["module"]
        for probe in imports
        if probe["module"] in {"open3d", "xatlas", "pymeshlab"} and not probe.get("available")
    ]
    return {
        "task_id": "nerfstudio_nerfacto_minimal",
        "dependency_boundary": [
            "nerfstudio route and CLI package",
            "nerfacc occupancy / ray marching kernels",
            "gsplat CUDA kernels",
            "tiny-cuda-nn Python binding",
            "Open3D / xatlas / pymeshlab native geometry helpers for export and mesh routes",
        ],
        "ok": not blockers,
        "status": "pass" if not blockers else "blocked",
        "imports": imports,
        "optional_missing": optional_missing,
        "blockers": blockers,
        "success_evidence": {
            "torch_cuda_smoke": runtime["torch"].get("smoke_ok") is True,
            "nerfstudio_import": imports[0].get("available") is True,
            "nerfacc_import": imports[1].get("available") is True,
            "gsplat_import": imports[2].get("available") is True,
            "tinycudann_import": imports[3].get("available") is True,
        },
    }


def open3d_functional_probe() -> dict[str, Any]:
    imported = import_probe("open3d", distribution="open3d")
    if not imported.get("available"):
        return {"ok": False, "import": imported, "error": imported.get("error")}
    try:
        import numpy as np
        import open3d as o3d

        points = np.asarray(
            [[x, y, 0.2 * x + 0.1 * y] for y in np.linspace(-1.0, 1.0, 5) for x in np.linspace(-1.0, 1.0, 5)],
            dtype=np.float64,
        )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=8))
        normals = np.asarray(pcd.normals)
        voxel = pcd.voxel_down_sample(voxel_size=0.55)
        cuda_available = None
        if hasattr(o3d, "core") and hasattr(o3d.core, "cuda"):
            cuda_available = bool(o3d.core.cuda.is_available())
        return {
            "ok": bool(np.isfinite(normals).all() and len(normals) == 25 and len(voxel.points) == 25),
            "import": imported,
            "point_count": int(len(points)),
            "normal_count": int(len(normals)),
            "voxel_point_count": int(len(voxel.points)),
            "open3d_core_cuda_available": cuda_available,
        }
    except Exception as exc:
        return {"ok": False, "import": imported, "error": repr(exc), "traceback": traceback.format_exc(limit=4)}


def open3d_probe(_runtime: dict[str, Any]) -> dict[str, Any]:
    functional = open3d_functional_probe()
    blockers: list[dict[str, Any]] = []
    if not functional.get("ok"):
        blockers.append(
            blocker(
                "open3d_native_geometry_failed",
                "Open3D did not complete the minimal PointCloud normal-estimation and voxel-downsample probe.",
                evidence=functional,
                remediation="Install a compatible open3d wheel and avoid visualization / OpenGL routes for the minimal baseline.",
            )
        )
    return {
        "task_id": "open3d_pointcloud_minimal",
        "dependency_boundary": [
            "Open3D compiled Python wheel",
            "CPU PointCloud geometry kernels",
            "no required CUDA or visualization route for the baseline task",
        ],
        "ok": not blockers,
        "status": "pass" if not blockers else "blocked",
        "functional": functional,
        "blockers": blockers,
        "success_evidence": {
            "open3d_import": functional.get("import", {}).get("available") is True,
            "pointcloud_operation": functional.get("ok") is True,
            "open3d_core_cuda_available": functional.get("open3d_core_cuda_available"),
        },
    }


def parse_repo_dir(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--repo-dir must use task_id=/path form, got: {value}")
        task_id, path = value.split("=", 1)
        if task_id not in TASK_IDS:
            raise SystemExit(f"unknown task_id for --repo-dir: {task_id}")
        result[task_id] = Path(path).expanduser().resolve()
    return result


def selected_tasks(raw: list[str]) -> list[str]:
    if not raw or "all" in raw:
        return list(TASK_IDS)
    tasks: list[str] = []
    for task_id in raw:
        if task_id not in TASK_IDS:
            raise SystemExit(f"unknown task_id: {task_id}")
        if task_id not in tasks:
            tasks.append(task_id)
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe native GPU / geometry dependency boundaries for PaperEnvBench tasks.")
    parser.add_argument("--task", action="append", default=[], help="Task id to probe, or all. Can be repeated.")
    parser.add_argument(
        "--repo-dir",
        action="append",
        default=[],
        help="Optional upstream repo checkout in task_id=/path form. Currently used for OpenPCDet setup.py evidence.",
    )
    parser.add_argument("--json", action="store_true", help="Print structured JSON.")
    parser.add_argument("--output", type=Path, help="Optional path to write JSON report.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when any selected task is blocked.")
    args = parser.parse_args()

    tasks = selected_tasks(args.task)
    repo_dirs = parse_repo_dir(args.repo_dir)
    runtime = runtime_probe()

    task_reports: dict[str, Any] = {}
    for task_id in tasks:
        if task_id == "gaussian_splatting_scene_minimal":
            task_reports[task_id] = gaussian_probe(runtime)
        elif task_id == "openpcdet_pointcloud_minimal":
            task_reports[task_id] = openpcdet_probe(runtime, repo_dirs.get(task_id))
        elif task_id == "nerfstudio_nerfacto_minimal":
            task_reports[task_id] = nerfstudio_probe(runtime)
        elif task_id == "open3d_pointcloud_minimal":
            task_reports[task_id] = open3d_probe(runtime)

    blocked = [task_id for task_id, report in task_reports.items() if not report.get("ok")]
    payload = {
        "generated_at": utc_now(),
        "probe": "geometry_cuda_probe",
        "selected_tasks": tasks,
        "ok": not blocked,
        "blocked_tasks": blocked,
        "runtime": runtime,
        "tasks": task_reports,
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.json:
        print(text)
    else:
        print(f"geometry_cuda_probe ok={payload['ok']} blocked_tasks={len(blocked)}")
        for task_id, report in task_reports.items():
            print(f"{task_id}: {report['status']}")
            for item in report.get("blockers", []):
                print(f"  - {item['code']}: {item['message']}")
    return 1 if args.strict and blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
