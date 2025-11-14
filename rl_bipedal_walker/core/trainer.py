"""
Reinforcement Learning Trainer for BipedalWalker

Uses PPO (Proximal Policy Optimization) to train an agent to walk.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from PIL import Image
import io


class BipedalWalkerTrainer:
    """
    Trainer for BipedalWalker environment using PPO

    Features:
    - Configurable hyperparameters
    - Training progress tracking
    - Model evaluation
    - Model saving/loading
    """

    def __init__(
        self,
        n_envs=16,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01
    ):
        """
        Initialize trainer

        Args:
            n_envs: Number of parallel environments
            n_steps: Steps per environment per update
            batch_size: Batch size for training
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            ent_coef: Entropy coefficient
        """
        self.n_envs = n_envs
        self.env = None
        self.model = None
        self.training_history = []

        self.config = {
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'ent_coef': ent_coef
        }

    def create_environment(self):
        """Create vectorized training environment"""
        self.env = make_vec_env('BipedalWalker-v3', n_envs=self.n_envs)
        print(f"✓ Created {self.n_envs} parallel environments")

    def create_model(self):
        """Create PPO model"""
        if self.env is None:
            self.create_environment()

        self.model = PPO(
            policy='MlpPolicy',
            env=self.env,
            n_steps=self.config['n_steps'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['n_epochs'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            ent_coef=self.config['ent_coef'],
            verbose=1
        )

        print("✓ PPO model created")
        return self.model

    def train(self, total_timesteps=1_000_000, progress_callback=None):
        """
        Train the model

        Args:
            total_timesteps: Total training timesteps
            progress_callback: Optional callback for progress updates
        """
        if self.model is None:
            self.create_model()

        print(f"\nStarting training for {total_timesteps:,} timesteps...")

        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True
        )

        print("\n✓ Training complete!")

    def evaluate(self, n_eval_episodes=10):
        """
        Evaluate the trained model

        Args:
            n_eval_episodes: Number of episodes for evaluation

        Returns:
            mean_reward, std_reward
        """
        if self.model is None:
            raise RuntimeError("No model to evaluate. Train a model first.")

        eval_env = Monitor(gym.make("BipedalWalker-v3"))

        mean_reward, std_reward = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True
        )

        eval_env.close()

        print(f"Evaluation: {mean_reward:.2f} +/- {std_reward:.2f}")

        return mean_reward, std_reward

    def save_model(self, path):
        """Save the trained model"""
        if self.model is None:
            raise RuntimeError("No model to save")

        self.model.save(path)
        print(f"✓ Model saved to {path}")

    def load_model(self, path):
        """Load a trained model"""
        if self.env is None:
            self.create_environment()

        self.model = PPO.load(path, env=self.env)
        print(f"✓ Model loaded from {path}")

    def get_env_info(self):
        """Get information about the environment"""
        env = gym.make("BipedalWalker-v3")

        info = {
            "observation_space_shape": env.observation_space.shape,
            "action_space_shape": env.action_space.shape,
            "observation_space_sample": env.observation_space.sample(),
            "action_space_sample": env.action_space.sample()
        }

        env.close()
        return info

    def visualize_episode(self, save_path=None, max_steps=1000):
        """
        Run and visualize a single episode with frame recording

        Args:
            save_path: Optional path to save video/GIF
            max_steps: Maximum steps per episode

        Returns:
            frames, total_reward, steps
        """
        if self.model is None:
            raise RuntimeError("No model to visualize. Train or load a model first.")

        env = gym.make("BipedalWalker-v3", render_mode="rgb_array")

        obs, info = env.reset()
        total_reward = 0
        done = False
        frames = []
        steps = 0

        while not done and steps < max_steps:
            # Render frame
            frame = env.render()
            frames.append(frame)

            # Take action
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        env.close()

        return frames, total_reward, steps

    def create_gif(self, frames, output_path='episode.gif', duration=50):
        """
        Create a GIF from episode frames

        Args:
            frames: List of RGB arrays
            output_path: Path to save GIF
            duration: Duration per frame in milliseconds

        Returns:
            Path to saved GIF
        """
        images = [Image.fromarray(frame) for frame in frames]

        # Save as GIF
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )

        return output_path

    def get_sample_frames(self, frames, num_samples=6):
        """
        Get evenly spaced sample frames from episode

        Args:
            frames: List of all frames
            num_samples: Number of frames to sample

        Returns:
            List of sampled frames
        """
        if len(frames) <= num_samples:
            return frames

        indices = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
        return [frames[i] for i in indices]
