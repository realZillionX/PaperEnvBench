from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def module_available(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {"available": True, "version": getattr(module, "__version__", None), "path": getattr(module, "__file__", None)}
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def torch_cuda_smoke() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "cuda_available": False, "error": repr(exc)}
    payload: dict[str, Any] = {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_compiled": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        x = torch.arange(8, dtype=torch.float32, device="cuda")
        y = x * x
        torch.cuda.synchronize()
        payload["device"] = "cuda"
        payload["sum"] = float(y.detach().cpu().sum().item())
    return payload


def cartpole_rollout() -> dict[str, Any]:
    backend = None
    try:
        import gymnasium as gym

        backend = "gymnasium"
    except Exception:
        try:
            import gym

            backend = "gym"
        except Exception as exc:
            return {"ok": False, "error": repr(exc), "backend": None}

    try:
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        reset_result = env.reset(seed=0)
        observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        total_reward = 0.0
        frame_shape = None
        for _ in range(4):
            step_result = env.step(env.action_space.sample())
            if len(step_result) == 5:
                observation, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                observation, reward, done, _ = step_result
            total_reward += float(reward)
            frame = env.render()
            if frame is not None and hasattr(frame, "shape"):
                frame_shape = list(frame.shape)
            if done:
                break
        env.close()
        return {
            "ok": True,
            "backend": backend,
            "observation_shape": list(getattr(observation, "shape", [])),
            "total_reward": total_reward,
            "rgb_render_shape": frame_shape,
        }
    except Exception as exc:
        return {"ok": False, "backend": backend, "error": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe RL simulator, rendering, and CUDA policy runtime surfaces.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when required RL surfaces are blocked.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    modules = {
        "gymnasium": module_available("gymnasium"),
        "gym": module_available("gym"),
        "stable_baselines3": module_available("stable_baselines3"),
        "mujoco": module_available("mujoco"),
        "dm_control": module_available("dm_control"),
    }
    rollout = cartpole_rollout()
    torch = torch_cuda_smoke()
    blockers = []
    if not (modules["gymnasium"]["available"] or modules["gym"]["available"]):
        blockers.append({"code": "gym_runtime_unavailable", "modules": {key: modules[key] for key in ("gymnasium", "gym")}})
    if not rollout.get("ok"):
        blockers.append({"code": "cartpole_rollout_or_render_failed", "rollout": rollout})
    if torch.get("cuda_available") is not True:
        blockers.append({"code": "torch_cuda_unavailable", "torch": torch})

    payload = {
        "generated_at": utc_now(),
        "status": "pass" if not blockers else "blocked",
        "modules": modules,
        "cartpole_rollout": rollout,
        "torch": torch,
        "blockers": blockers,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"rl_simulation_probe status={payload['status']} blockers={len(blockers)}")
    return 1 if args.strict and blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
