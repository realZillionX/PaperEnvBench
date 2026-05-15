# cleanrl_cartpole_minimal

This task verifies a minimal, pinned CPU-safe reproduction route for `https://github.com/vwxyzjn/cleanrl` at commit `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`.

The intended public route is CleanRL's single-file PPO script:

```bash
python cleanrl/ppo.py --env-id CartPole-v1 --total-timesteps 128 --num-envs 1 --num-steps 32 --num-minibatches 1 --update-epochs 2 --learning-rate 0.00025 --torch-deterministic True --cuda False --track False
```

The check-only artifact uses a deterministic CPU fallback instead of requiring a live Gymnasium / PyTorch training run during local validation. It preserves the relevant semantics: CartPole observation shape, discrete actions, a short fixed rollout, clipped PPO surrogate, value loss, KL, entropy, and positive episode return.

Expected artifacts:

- `artifacts/expected_artifact.json`

Validation:

```bash
python verify.py --check-only --json
```
