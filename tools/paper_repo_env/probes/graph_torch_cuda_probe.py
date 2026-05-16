from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


Blocker = dict[str, Any]


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
    except FileNotFoundError as exc:
        return {"returncode": 127, "stdout": "", "stderr": str(exc)}
    except Exception as exc:
        return {"returncode": 126, "stdout": "", "stderr": repr(exc)}


def blocker(
    code: str,
    scope: str,
    message: str,
    evidence: Any,
    recommendation: str,
    severity: str = "error",
) -> Blocker:
    return {
        "code": code,
        "scope": scope,
        "severity": severity,
        "message": message,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def module_version(module: Any) -> str | None:
    return getattr(module, "__version__", None)


def short_exception(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def nvidia_smi_probe() -> dict[str, Any]:
    query = run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader",
        ]
    )
    rows = []
    if query["returncode"] == 0:
        for line in query["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5:
                rows.append(
                    {
                        "index": parts[0],
                        "name": parts[1],
                        "memory_total": parts[2],
                        "driver_version": parts[3],
                        "compute_capability": parts[4],
                    }
                )
    return {
        "available": query["returncode"] == 0 and bool(rows),
        "gpus": rows,
        "query": query,
    }


def import_probe(module_name: str, package_name: str | None = None) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        return {
            "available": True,
            "module": module_name,
            "version": module_version(module) or package_version(package_name or module_name),
            "path": getattr(module, "__file__", None),
        }
    except Exception as exc:
        return {
            "available": False,
            "module": module_name,
            "version": package_version(package_name or module_name),
            "error": short_exception(exc),
        }


def run_step(name: str, func: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return {"name": name, "ok": True, "result": func()}
    except Exception as exc:
        return {
            "name": name,
            "ok": False,
            "error": short_exception(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-8:],
        }


def torch_probe(require_cuda: bool) -> tuple[dict[str, Any], list[Blocker]]:
    blockers: list[Blocker] = []
    payload: dict[str, Any] = {
        "packages": {
            name: package_version(name)
            for name in ["torch", "torchvision", "torchaudio", "triton"]
        },
        "steps": [],
    }
    try:
        import torch
    except Exception as exc:
        blockers.append(
            blocker(
                "torch_import_failed",
                "torch_cuda_base",
                "Cannot import torch.",
                short_exception(exc),
                "Install a torch wheel matching the Python ABI and desired CUDA runtime.",
            )
        )
        payload["import_error"] = short_exception(exc)
        return payload, blockers

    payload.update(
        {
            "torch_version": getattr(torch, "__version__", None),
            "torch_cuda_compiled": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
        }
    )

    payload["steps"].append(
        run_step(
            "torch_cpu_matmul",
            lambda: {
                "sum": float((torch.arange(16, dtype=torch.float32).reshape(4, 4) @ torch.eye(4)).sum().item())
            },
        )
    )

    if torch.cuda.is_available():
        def cuda_matmul() -> dict[str, Any]:
            device = torch.device("cuda:0")
            x = torch.arange(64, dtype=torch.float32, device=device).reshape(8, 8)
            y = x @ x.T
            torch.cuda.synchronize()
            return {
                "device_name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "sum": float(y.sum().detach().cpu().item()),
                "memory_allocated": int(torch.cuda.memory_allocated(0)),
            }

        payload["steps"].append(run_step("torch_cuda_matmul", cuda_matmul))
    elif require_cuda:
        blockers.append(
            blocker(
                "torch_cuda_unavailable",
                "torch_cuda_base",
                "torch imports, but CUDA execution is unavailable.",
                {
                    "torch_version": getattr(torch, "__version__", None),
                    "torch_cuda_compiled": getattr(torch.version, "cuda", None),
                    "cuda_available": False,
                },
                "Install CUDA-enabled torch / torchvision / torchaudio wheels from the same PyTorch CUDA index and verify the NVIDIA driver is visible.",
            )
        )

    payload["torchvision"] = import_probe("torchvision")
    if payload["torchvision"]["available"]:
        def torchvision_ops() -> dict[str, Any]:
            import torchvision
            from torchvision.ops import nms

            device = "cuda" if torch.cuda.is_available() else "cpu"
            boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0], [1.0, 1.0, 5.0, 5.0]], device=device)
            scores = torch.tensor([0.9, 0.8], device=device)
            keep = nms(boxes, scores, 0.5)
            if device == "cuda":
                torch.cuda.synchronize()
            return {
                "version": getattr(torchvision, "__version__", None),
                "device": device,
                "kept_indices": keep.detach().cpu().tolist(),
            }

        step = run_step("torchvision_nms", torchvision_ops)
        payload["steps"].append(step)
        if not step["ok"]:
            blockers.append(
                blocker(
                    "torchvision_ops_failed",
                    "torch_cuda_base",
                    "torchvision imports, but compiled torchvision ops failed.",
                    step.get("error"),
                    "Reinstall torchvision from the same CUDA wheel index and version row as torch.",
                )
            )
    else:
        blockers.append(
            blocker(
                "torchvision_import_failed",
                "torch_cuda_base",
                "Cannot import torchvision.",
                payload["torchvision"].get("error"),
                "Install torchvision from the same PyTorch CUDA wheel row as torch.",
            )
        )

    payload["torchaudio"] = import_probe("torchaudio")
    if payload["torchaudio"]["available"]:
        def torchaudio_spectrogram() -> dict[str, Any]:
            import torchaudio.functional as F

            device = "cuda" if torch.cuda.is_available() else "cpu"
            waveform = torch.sin(torch.linspace(0, 40, 2048, device=device)).unsqueeze(0)
            window = torch.hann_window(128, device=device)
            spec = F.spectrogram(
                waveform,
                pad=0,
                window=window,
                n_fft=128,
                hop_length=64,
                win_length=128,
                power=2.0,
                normalized=False,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            return {"device": device, "shape": list(spec.shape), "mean": float(spec.detach().cpu().mean().item())}

        step = run_step("torchaudio_spectrogram", torchaudio_spectrogram)
        payload["steps"].append(step)
        if not step["ok"]:
            blockers.append(
                blocker(
                    "torchaudio_ops_failed",
                    "torch_cuda_base",
                    "torchaudio imports, but a tensor audio smoke failed.",
                    step.get("error"),
                    "Reinstall torchaudio from the same CUDA wheel row as torch and check native library linkage.",
                )
            )
    else:
        blockers.append(
            blocker(
                "torchaudio_import_failed",
                "torch_cuda_base",
                "Cannot import torchaudio.",
                payload["torchaudio"].get("error"),
                "Install torchaudio from the same PyTorch CUDA wheel row as torch.",
                severity="warning",
            )
        )

    return payload, blockers


def pyg_probe(require_cuda: bool) -> tuple[dict[str, Any], list[Blocker]]:
    blockers: list[Blocker] = []
    extension_names = ["pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv"]
    payload: dict[str, Any] = {
        "packages": {
            "torch-geometric": package_version("torch-geometric"),
            **{name.replace("_", "-"): package_version(name.replace("_", "-")) for name in extension_names},
        },
        "extensions": {name: import_probe(name) for name in extension_names},
        "steps": [],
    }
    try:
        import torch
        import torch_geometric
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv
    except Exception as exc:
        blockers.append(
            blocker(
                "pyg_import_failed",
                "pyg_cuda_wheels",
                "Cannot import PyG core modules.",
                short_exception(exc),
                "Install torch_geometric plus optional extension wheels from the PyG wheel index matching torch and CUDA.",
            )
        )
        payload["import_error"] = short_exception(exc)
        return payload, blockers

    payload.update(
        {
            "torch_geometric_version": getattr(torch_geometric, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    )

    def gcn_forward() -> dict[str, Any]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data = Data(
            x=torch.eye(4, dtype=torch.float32),
            edge_index=torch.tensor([[0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 2, 0]], dtype=torch.long),
        ).to(device)
        conv = GCNConv(4, 2).to(device)
        out = conv(data.x, data.edge_index)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return {"device": device.type, "shape": list(out.shape), "sum": float(out.detach().cpu().sum().item())}

    step = run_step("pyg_gcn_forward", gcn_forward)
    payload["steps"].append(step)
    if not step["ok"]:
        blockers.append(
            blocker(
                "pyg_gcn_forward_failed",
                "pyg_cuda_wheels",
                "PyG imports, but a minimal GCN forward pass failed.",
                step.get("error"),
                "Check that PyG extension wheels, torch, and CUDA tags all come from a compatible matrix row.",
            )
        )
    elif require_cuda and step["result"].get("device") != "cuda":
        blockers.append(
            blocker(
                "pyg_cuda_not_exercised",
                "pyg_cuda_wheels",
                "PyG smoke ran only on CPU.",
                step["result"],
                "Run this probe in a CUDA-enabled environment after installing matching PyG CUDA wheels.",
            )
        )

    missing_extensions = [name for name, result in payload["extensions"].items() if not result["available"]]
    if missing_extensions:
        blockers.append(
            blocker(
                "pyg_optional_extensions_missing",
                "pyg_cuda_wheels",
                "One or more PyG optional extension packages are missing.",
                {"missing": missing_extensions},
                "For GPU graph workloads, install pyg_lib, torch_scatter, torch_sparse, torch_cluster, and torch_spline_conv from the matching https://data.pyg.org wheel page.",
                severity="warning",
            )
        )

    return payload, blockers


def dgl_probe(require_cuda: bool) -> tuple[dict[str, Any], list[Blocker]]:
    blockers: list[Blocker] = []
    payload: dict[str, Any] = {
        "packages": {"dgl": package_version("dgl"), "torchdata": package_version("torchdata")},
        "steps": [],
    }
    if "DGLBACKEND" not in os.environ:
        os.environ["DGLBACKEND"] = "pytorch"
        payload["dglbackend_defaulted"] = "pytorch"

    try:
        import torch
        import dgl
        from dgl.nn import SAGEConv
    except Exception as exc:
        blockers.append(
            blocker(
                "dgl_import_failed",
                "dgl_cuda_wheels",
                "Cannot import DGL with the PyTorch backend.",
                short_exception(exc),
                "Install DGL from the wheel page that matches the torch major.minor line and CUDA runtime; also install torchdata when required by the DGL release.",
            )
        )
        payload["import_error"] = short_exception(exc)
        return payload, blockers

    payload.update(
        {
            "dgl_version": getattr(dgl, "__version__", None),
            "backend": getattr(getattr(dgl, "backend", None), "backend_name", None),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    )

    def graphsage_forward() -> dict[str, Any]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        graph = dgl.graph(([0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 2, 0])).to(device)
        features = torch.eye(4, dtype=torch.float32, device=device)
        conv = SAGEConv(4, 2, "mean").to(device)
        out = conv(graph, features)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return {"device": device.type, "shape": list(out.shape), "sum": float(out.detach().cpu().sum().item())}

    step = run_step("dgl_graphsage_forward", graphsage_forward)
    payload["steps"].append(step)
    if not step["ok"]:
        message = str(step.get("error", {}).get("message", ""))
        code = "dgl_cuda_backend_disabled" if "Device API cuda is not enabled" in message else "dgl_graphsage_forward_failed"
        blockers.append(
            blocker(
                code,
                "dgl_cuda_wheels",
                "DGL imports, but a minimal GraphSAGE forward pass failed.",
                step.get("error"),
                "Install a CUDA-enabled DGL wheel for the active torch version, or record this dependency slice as blocked.",
            )
        )
    elif require_cuda and step["result"].get("device") != "cuda":
        blockers.append(
            blocker(
                "dgl_cuda_not_exercised",
                "dgl_cuda_wheels",
                "DGL smoke ran only on CPU.",
                step["result"],
                "Run this probe in a CUDA-enabled environment with a CUDA-enabled DGL wheel.",
            )
        )

    return payload, blockers


def ogb_probe() -> tuple[dict[str, Any], list[Blocker]]:
    blockers: list[Blocker] = []
    payload: dict[str, Any] = {
        "packages": {
            name: package_version(name)
            for name in ["ogb", "numpy", "pandas", "scikit-learn", "torch"]
        },
        "steps": [],
    }
    try:
        from ogb.nodeproppred import Evaluator
        import numpy as np
    except Exception as exc:
        blockers.append(
            blocker(
                "ogb_import_failed",
                "ogb_nodeprop",
                "Cannot import OGB node property evaluator.",
                short_exception(exc),
                "Install ogb and its numeric Python dependencies; OGB itself does not require a CUDA wheel.",
            )
        )
        payload["import_error"] = short_exception(exc)
        return payload, blockers

    def evaluator_smoke() -> dict[str, Any]:
        evaluator = Evaluator(name="ogbn-arxiv")
        result = evaluator.eval(
            {
                "y_true": np.array([[0], [1], [1], [0]]),
                "y_pred": np.array([[0], [1], [0], [0]]),
            }
        )
        return {"metric": result}

    step = run_step("ogb_nodeprop_evaluator", evaluator_smoke)
    payload["steps"].append(step)
    if not step["ok"]:
        blockers.append(
            blocker(
                "ogb_evaluator_failed",
                "ogb_nodeprop",
                "OGB imports, but the node property evaluator smoke failed.",
                step.get("error"),
                "Check the OGB version and numeric dependency pins; avoid requiring dataset download for this probe.",
            )
        )

    return payload, blockers


def build_payload(groups: set[str], require_cuda: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "environment": {
            "DGLBACKEND": os.environ.get("DGLBACKEND"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "nvidia_smi": nvidia_smi_probe(),
        "probes": {},
        "blockers": [],
    }

    if require_cuda and not payload["nvidia_smi"]["available"]:
        payload["blockers"].append(
            blocker(
                "nvidia_smi_unavailable",
                "host_cuda_runtime",
                "nvidia-smi did not report a visible NVIDIA GPU.",
                payload["nvidia_smi"]["query"],
                "Run on a GPU node with the NVIDIA driver mounted and visible to the process.",
            )
        )

    if "torch" in groups:
        result, blockers = torch_probe(require_cuda=require_cuda)
        payload["probes"]["torch_cuda_base"] = result
        payload["blockers"].extend(blockers)
    if "pyg" in groups:
        result, blockers = pyg_probe(require_cuda=require_cuda)
        payload["probes"]["pyg_cuda_wheels"] = result
        payload["blockers"].extend(blockers)
    if "dgl" in groups:
        result, blockers = dgl_probe(require_cuda=require_cuda)
        payload["probes"]["dgl_cuda_wheels"] = result
        payload["blockers"].extend(blockers)
    if "ogb" in groups:
        result, blockers = ogb_probe()
        payload["probes"]["ogb_nodeprop"] = result
        payload["blockers"].extend(blockers)

    error_blockers = [item for item in payload["blockers"] if item.get("severity") == "error"]
    payload["ok"] = not error_blockers
    payload["summary"] = {
        "groups": sorted(groups),
        "require_cuda": require_cuda,
        "error_blocker_count": len(error_blockers),
        "warning_blocker_count": len(payload["blockers"]) - len(error_blockers),
    }
    return payload


def parse_groups(values: list[str]) -> set[str]:
    if not values or "all" in values:
        return {"torch", "pyg", "dgl", "ogb"}
    groups = set(values)
    if {"pyg", "dgl"} & groups:
        groups.add("torch")
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe graph, torch CUDA, PyG, DGL, OGB, torchvision, and torchaudio dependency readiness."
    )
    parser.add_argument("--json", action="store_true", help="Print structured JSON.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument(
        "--group",
        action="append",
        choices=["all", "torch", "pyg", "dgl", "ogb"],
        default=[],
        help="Probe group to run. Repeatable. Default: all.",
    )
    parser.add_argument("--require-cuda", action="store_true", help="Treat missing CUDA execution as a blocker.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when error blockers are present.")
    args = parser.parse_args()

    payload = build_payload(groups=parse_groups(args.group), require_cuda=args.require_cuda)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"ok={payload['ok']} groups={','.join(payload['summary']['groups'])}")
        print(
            "blockers="
            f"{payload['summary']['error_blocker_count']} errors/"
            f"{payload['summary']['warning_blocker_count']} warnings"
        )
        for item in payload["blockers"]:
            print(f"{item['severity']} {item['scope']} {item['code']}: {item['message']}")

    return 1 if args.strict and not payload["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
