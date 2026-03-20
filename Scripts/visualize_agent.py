#!/usr/bin/env python3
"""Visualize trained agent in MuJoCo viewer.

Usage:
    python visualize_agent.py --model-path runs/YOUR_RUN_NAME/ppo_continuous_action.cleanrl_model
"""

import argparse
import sys
import os
import numpy as np
import torch
import gymnasium as gym

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(__file__))

import UR5  # Register environment
from env import UR5P2PEnv
from cleanrl.cleanrl.ppo_continuous_action import Agent, Args, make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to .cleanrl_model file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target-mode", type=str, default="random", help="Goal sampling mode")
    args_viz = parser.parse_args()

    device = torch.device("cpu")

    # Create environment with rendering
    env = UR5P2PEnv(
        render_mode="human",
        seed=args_viz.seed,
        target_mode=args_viz.target_mode,
        target_pos=None,
    )

    # Setup agent args (must match training config)
    args = Args()
    args.env_id = "UR5-v0"
    args.target_mode = args_viz.target_mode
    args.target_pos = None

    # Create dummy env for agent initialization (to get observation space)
    dummy_envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, 0, False, "eval", 0.99, args=args)]
    )
    agent = Agent(dummy_envs).to(device)

    # Load trained weights
    print(f"Loading model from: {args_viz.model_path}")
    agent.load_state_dict(torch.load(args_viz.model_path, map_location=device))
    agent.eval()

    # Run episodes
    for ep in range(args_viz.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"\n=== Episode {ep + 1}/{args_viz.episodes} ===")
        print(f"Goal position: {env.goal_pos}")
        print(f"Start position: {env.start_pos}")

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t)
            
            obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            total_reward += reward
            steps += 1
            done = terminated or truncated

            # Print progress every 100 steps
            if steps % 100 == 0:
                ee_pos = env._ee_pos()
                dist = np.linalg.norm(ee_pos - env.goal_pos) if env.goal_pos is not None else 0
                print(f"  Step {steps}: dist={dist:.4f}, reward={reward:.3f}")

        print(f"Episode finished: success={info.get('success', False)}, "
              f"total_reward={total_reward:.2f}, steps={steps}")

    env.close()
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
