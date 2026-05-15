#!/usr/bin/env bash
set -euo pipefail

TASK_ID="stable_baselines3_cartpole_minimal"
REPO_URL="https://github.com/DLR-RM/stable-baselines3.git"
REPO_COMMIT="8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458"

ROOT_DIR="${PAPERENV_WORKDIR:-$PWD}"
RUN_DIR="${ROOT_DIR}/runs/${TASK_ID}"
REPO_DIR="${RUN_DIR}/repo"
VENV_DIR="${RUN_DIR}/venv"

mkdir -p "${RUN_DIR}"
if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi
git -C "${REPO_DIR}" fetch --depth 1 origin "${REPO_COMMIT}"
git -C "${REPO_DIR}" checkout --detach "${REPO_COMMIT}"

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu
"${VENV_DIR}/bin/python" -m pip install -e "${REPO_DIR}"

"${VENV_DIR}/bin/python" - <<'PY'
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("CartPole-v1", n_envs=1, seed=123)
model = PPO(
    "MlpPolicy",
    env,
    seed=123,
    n_steps=16,
    batch_size=16,
    n_epochs=1,
    gamma=0.98,
    learning_rate=3e-4,
    device="cpu",
    verbose=0,
)
model.learn(total_timesteps=64)
obs = env.reset()
episode_reward = 0.0
actions = []
for _ in range(8):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    actions.append(int(action[0]))
    episode_reward += float(reward[0])
assert episode_reward >= 8.0, episode_reward
print({"algorithm": "PPO", "env_id": "CartPole-v1", "episode_reward": episode_reward, "actions": actions})
env.close()
PY
