"""
Reinforcement Learning controller for battery fast charging
"""

import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
import os


class BatteryRLController:
    """
    RL agent that learns optimal charging currents.
    Can be trained from scratch or loaded from saved model.
    """
    
    def __init__(self, env, algorithm="SAC", model_path=None):
        """
        Initialize RL controller.
        
        Args:
            env: Gym environment (BatteryPlatingEnv)
            algorithm: "SAC" or "PPO"
            model_path: Path to saved model (optional)
        """
        self.env = env
        self.algorithm = algorithm
        
        if model_path and os.path.exists(model_path):
            # Load existing model
            if algorithm == "SAC":
                self.model = SAC.load(model_path, env=env)
            else:
                self.model = PPO.load(model_path, env=env)
            print(f"Loaded model from {model_path}")
        else:
            # Create new model
            if algorithm == "SAC":
                self.model = SAC(
                    "MlpPolicy",
                    env,
                    learning_rate=3e-4,
                    buffer_size=50000,
                    batch_size=128,
                    gamma=0.99,
                    tau=0.005,
                    verbose=0,
                    tensorboard_log="./logs/"
                )
            else:
                self.model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    verbose=0
                )
            print(f"Created new {algorithm} model")
    
    def train(self, total_timesteps=100000, save_path="rl_model"):
        """
        Train the RL agent.
        
        Args:
            total_timesteps: Number of training steps
            save_path: Path to save the trained model
        """
        print(f"Training RL agent for {total_timesteps} timesteps...")
        
        # Progress callback
        class ProgressCallback(BaseCallback):
            def __init__(self, verbose=0):
                super().__init__(verbose)
                self.episodes = 0
                self.total_reward = 0
            
            def _on_step(self):
                if self.locals.get('done', False):
                    self.episodes += 1
                    if self.episodes % 10 == 0:
                        print(f"Episode {self.episodes}")
                return True
        
        callback = ProgressCallback()
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        
        # Save the model
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
        
        return self.model
    
    def get_action(self, observation, deterministic=True):
        """
        Get action from trained policy.
        
        Args:
            observation: Current state observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action (charging current in C-rate)
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return float(action[0])
    
    def evaluate(self, n_episodes=10):
        """
        Evaluate the trained agent.
        
        Args:
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"Evaluating RL agent for {n_episodes} episodes...")
        
        results = {
            'plating_events': 0,
            'charging_times': [],
            'final_soc': [],
            'max_temperatures': []
        }
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            step = 0
            
            while not done:
                action = self.get_action(obs)
                obs, reward, terminated, truncated, info = self.env.step([action])
                done = terminated or truncated
                step += 1
            
            results['plating_events'] += 1 if info.get('plating_detected', False) else 0
            results['charging_times'].append(info.get('time', step * self.env.dt))
            results['final_soc'].append(info.get('soc', 0))
            results['max_temperatures'].append(info.get('temperature', 298.15) - 273.15)
            
            print(f"  Episode {episode+1}: Plating={info.get('plating_detected', False)}, "
                  f"SoC={info.get('soc', 0):.2f}, Time={info.get('time', 0):.0f}s")
        
        return results
