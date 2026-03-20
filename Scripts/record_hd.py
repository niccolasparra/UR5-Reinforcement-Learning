#!/usr/bin/env python3
"""Record HIGH QUALITY videos of trained agent.

Usage:
    # Default HD (1280x720)
    python record_hd.py --model-path runs/YOUR_RUN/model.cleanrl_model --episodes 5
    
    # Full HD (1920x1080) - Best quality
    python record_hd.py --model-path runs/YOUR_RUN/model.cleanrl_model --episodes 5 --fullhd
"""

import argparse
import sys
import os
import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, os.path.dirname(__file__))

import UR5
from env import UR5P2PEnv


class HighQualityVideoWrapper(gym.Wrapper):
    """Custom video wrapper with configurable quality."""
    
    def __init__(self, env, video_folder, episode_trigger, name_prefix, width=1280, height=720):
        super().__init__(env)
        self.video_folder = video_folder
        self.episode_trigger = episode_trigger
        self.name_prefix = name_prefix
        self.width = width
        self.height = height
        self.episode_id = 0
        self.video_recorder = None
        
        os.makedirs(video_folder, exist_ok=True)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Start new video if needed
        if self.episode_trigger(self.episode_id):
            if self.video_recorder is not None:
                self.video_recorder.close()
            
            video_path = os.path.join(
                self.video_folder,
                f"{self.name_prefix}-episode-{self.episode_id}.mp4"
            )
            self._start_video_recorder(video_path)
            
        self.episode_id += 1
        return obs, info
    
    def _start_video_recorder(self, video_path):
        """Start video recording with custom resolution."""
        try:
            import moviepy.editor as mpy
            from PIL import Image
            
            self.frames = []
            self.video_path = video_path
            self.recording = True
            
            # Capture first frame
            frame = self.env.render()
            if frame is not None:
                # Resize to target resolution
                from PIL import Image
                img = Image.fromarray(frame)
                img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
                self.frames.append(np.array(img))
                
        except Exception as e:
            print(f"Warning: Could not start video recorder: {e}")
            self.recording = False
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Record frame
        if hasattr(self, 'recording') and self.recording:
            frame = self.env.render()
            if frame is not None:
                from PIL import Image
                img = Image.fromarray(frame)
                img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
                self.frames.append(np.array(img))
        
        # Save video on episode end
        if terminated or truncated:
            if hasattr(self, 'recording') and self.recording and len(self.frames) > 0:
                self._save_video()
        
        return obs, reward, terminated, truncated, info
    
    def _save_video(self):
        """Save recorded frames as video."""
        try:
            import moviepy.editor as mpy
            
            # Create video from frames
            clip = mpy.ImageSequenceClip(self.frames, fps=30)
            clip.write_videofile(
                self.video_path,
                codec='libx264',
                audio=False,
                verbose=False,
                logger=None,
                bitrate='5000k'  # High bitrate for quality
            )
            clip.close()
            
            print(f"  Video saved: {self.video_path}")
            
        except Exception as e:
            print(f"  Warning: Could not save video: {e}")
        finally:
            self.frames = []
            self.recording = False
    
    def close(self):
        if hasattr(self, 'frames') and len(self.frames) > 0:
            self._save_video()
        super().close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="videos/hd")
    parser.add_argument("--fullhd", action="store_true", help="Use Full HD 1920x1080 (default: HD 1280x720)")
    args_viz = parser.parse_args()

    # Set resolution
    if args_viz.fullhd:
        width, height = 1920, 1080
        quality = "Full HD (1920x1080)"
    else:
        width, height = 1280, 720
        quality = "HD (1280x720)"

    device = torch.device("cpu")
    os.makedirs(args_viz.output_dir, exist_ok=True)

    print(f"Creating environment with {quality} resolution...")
    
    # Import after setting up path
    from cleanrl.cleanrl.ppo_continuous_action import Agent, Args, make_env

    # Create base environment with rgb_array
    base_env = gym.make("UR5-v0", render_mode="rgb_array", target_mode="random")
    
    # Wrap with high-quality video recorder
    env = HighQualityVideoWrapper(
        base_env,
        video_folder=args_viz.output_dir,
        episode_trigger=lambda ep: True,
        name_prefix="ppo_hd",
        width=width,
        height=height
    )

    # Load agent
    args = Args()
    args.env_id = "UR5-v0"
    args.target_mode = "random"
    args.target_pos = None

    dummy_envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, 0, False, "eval", 0.99, args=args)]
    )
    agent = Agent(dummy_envs).to(device)
    
    print(f"Loading model: {args_viz.model_path}")
    agent.load_state_dict(torch.load(args_viz.model_path, map_location=device, weights_only=False))
    agent.eval()

    # Run episodes
    successes = 0
    rewards_list = []
    steps_list = []

    print(f"\nRecording {args_viz.episodes} episodes at {quality}...")
    print(f"Output directory: {args_viz.output_dir}/\n")

    for ep in range(args_viz.episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        steps = 0

        print(f"Episode {ep + 1}/{args_viz.episodes}... ", end="", flush=True)

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t)
            
            obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        success = info.get('success', False)
        successes += int(success)
        rewards_list.append(ep_reward)
        steps_list.append(steps)

        print(f"{'SUCCESS ✓' if success else 'FAILED ✗'} | "
              f"Reward: {ep_reward:.1f} | Steps: {steps}")

    env.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"RECORDING COMPLETE ({quality})")
    print(f"{'='*60}")
    print(f"Success Rate: {successes}/{args_viz.episodes} ({successes/args_viz.episodes*100:.1f}%)")
    print(f"Avg Reward:   {np.mean(rewards_list):.2f} ± {np.std(rewards_list):.2f}")
    print(f"Avg Steps:    {np.mean(steps_list):.1f} ± {np.std(steps_list):.1f}")
    print(f"\nVideos saved to: {args_viz.output_dir}/")
    
    # List generated videos
    import glob
    videos = sorted(glob.glob(f"{args_viz.output_dir}/*.mp4"))
    if videos:
        print("Files:")
        for vid in videos:
            size_mb = os.path.getsize(vid) / (1024*1024)
            print(f"  - {os.path.basename(vid)} ({size_mb:.1f} MB)")
    else:
        print("Note: Videos may still be processing...")


if __name__ == "__main__":
    main()
