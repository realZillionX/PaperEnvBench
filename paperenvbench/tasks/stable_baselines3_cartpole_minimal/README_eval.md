# stable_baselines3_cartpole_minimal

This task verifies a pinned Stable-Baselines3 PPO / CartPole CPU reproduction route for `https://github.com/DLR-RM/stable-baselines3`.

The gold route pins commit `8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458`, installs Stable-Baselines3 from that checkout with CPU PyTorch, creates `CartPole-v1` through Gymnasium, and exercises `stable_baselines3.PPO` with a short deterministic smoke configuration. The check-only package avoids long training and validates the recorded route plus a deterministic rollout artifact.

Import success alone is not sufficient. An accepted result must preserve the pinned repo evidence, PPO entrypoint, Gymnasium CartPole environment route, CPU-only boundary, seed, rollout actions, and reward summary.

Validated local equivalent:

```bash
bash gold_install.sh
python verify.py --check-only --json
```

Validated PaperEnvBench package check:

```bash
python3 tools/paper_repo_env/validate_task_package.py --task stable_baselines3_cartpole_minimal
```
