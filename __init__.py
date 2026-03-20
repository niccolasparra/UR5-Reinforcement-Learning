import gymnasium as gym
from env import UR5P2PEnv

gym.register(
    id="UR5-v0",
    entry_point="env:UR5P2PEnv",
    kwargs={"render_mode": None, "target_mode": "random"},
    max_episode_steps=None,   # or your preferred horizon
)