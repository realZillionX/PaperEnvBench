# gym_cartpole_minimal

This task verifies a pinned OpenAI Gym route for a minimal CartPole environment smoke test. The gold route clones `openai/gym` at commit `bc212954b6713d5db303b3ead124de6cba66063e`, identifies `gym.envs.classic_control.cartpole.CartPoleEnv`, and validates a deterministic `reset(seed=123)` plus five `step` transitions on CPU.

Expected verifier:

```bash
python verify.py --check-only --json
```

The check-only verifier validates the recorded artifact and route evidence. It does not require GPU hardware, dataset downloads, pygame rendering, or a display server. A valid attempt must identify the pinned repository, the `gym.make("CartPole-v1")` / `CartPoleEnv` entrypoint, the Gym reset and step return contract, and the CPU deterministic fallback boundary for Python 3.12 environments where the archived Gym package may need compatibility handling.
