"""
Reinforcement Learning Controller for Battery Fast Charging
Research-grade implementation compatible with PyBaMM surrogate environment.
Uses SAC (Soft Actor-Critic) for continuous action space optimization.
"""

import numpy as np
import torch  # FIXED ISSUE 1: Added torch import at top
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime


class BatteryRLController:
    """
    RL agent that learns optimal charging currents using SAC algorithm.
    Designed for the BatteryPlatingEnv with surrogate PyBaMM model.
    """
    
    def __init__(self, env, algorithm="SAC", model_path=None, tensorboard_log="./logs/"):
        """
        Initialize RL controller.
        
        Args:
            env: Gym environment (BatteryPlatingEnv with surrogate model)
            algorithm: Only "SAC" is supported for continuous actions
            model_path: Path to saved model (optional)
            tensorboard_log: Directory for TensorBoard logs
        """
        # Store original environment parameters before wrapping
        self.max_current = env.max_current
        self.dt = env.dt
        self.target_soc = env.target_soc
        self.original_env = env  # Keep reference to unwrapped env
        
        self.env = env
        self.algorithm = algorithm
        
        # Wrap environment with Monitor for metrics recording
        self.monitor_dir = f"./monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.env = Monitor(env, self.monitor_dir)
        
        # Hyperparameters optimized for battery charging control
        self.hyperparams = {
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "policy_kwargs": dict(
                net_arch=[256, 256],
                activation_fn=torch.nn.ReLU
            ),
        }
        
        if model_path and os.path.exists(model_path):
            self.model = SAC.load(model_path, env=self.env)
            print(f"Loaded existing model from {model_path}")
        else:
            self.model = SAC(
                "MlpPolicy",
                self.env,
                **self.hyperparams,
                verbose=1,
                tensorboard_log=tensorboard_log
            )
            print(f"Created new SAC model")
        
        # For tracking training progress
        self.training_history = {
            'timesteps': [],
            'mean_rewards': [],
            'plating_rates': []
        }
    
    def train(self, total_timesteps=200000, save_path="rl_model", eval_freq=5000):
        """
        Train the RL agent with evaluation callbacks.
        """
        print(f"\n{'='*60}")
        print(f"Training RL Agent")
        print(f"Total timesteps: {total_timesteps}")
        print(f"{'='*60}\n")
        
        class TrainingCallback(BaseCallback):
            def __init__(self, controller, verbose=0):
                super().__init__(verbose)
                self.controller = controller
                self.episodes = 0
                self.total_reward = 0
                self.episode_rewards = []
                self.plating_episodes = 0
            
            def _on_step(self) -> bool:
                if 'rewards' in self.locals:
                    self.total_reward += self.locals['rewards'][0]
                
                if self.locals.get('done', False):
                    self.episodes += 1
                    self.episode_rewards.append(self.total_reward)
                    
                    # Check if plating occurred
                    infos = self.locals.get('infos', [{}])
                    if infos and infos[0].get('plating_detected', False):
                        self.plating_episodes += 1
                    
                    if self.episodes % 20 == 0:
                        avg_reward = np.mean(self.episode_rewards[-20:])
                        plating_rate = (self.plating_episodes / self.episodes) * 100
                        print(f"Episode {self.episodes}: Avg Reward={avg_reward:.2f}, "
                              f"Plating Rate={plating_rate:.1f}%")
                    
                    self.total_reward = 0
                
                return True
        
        # Create evaluation environment
        eval_env = self._create_eval_env()
        
        # FIXED ISSUE 6: Use directory path for best model
        best_model_dir = f"./best_{save_path}"
        os.makedirs(best_model_dir, exist_ok=True)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_dir,
            log_path=f"./eval_logs/",
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            verbose=1
        )
        
        training_callback = TrainingCallback(self)
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[training_callback, eval_callback],
            progress_bar=True
        )
        
        self.model.save(save_path)
        print(f"\nModel saved to {save_path}")
        
        self.training_history['timesteps'].append(total_timesteps)
        
        return self.model
    
    def _create_eval_env(self):
        """Create a separate environment for evaluation"""
        from battery_environment import BatteryPlatingEnv
        # FIXED ISSUE 2: Use stored parameters instead of env attributes
        return BatteryPlatingEnv(
            max_current_C=self.max_current,
            dt=self.dt,
            target_soc=self.target_soc
        )
    
    def get_action(self, observation, deterministic=True):
        """
        Get action from trained policy.
        """
        if isinstance(observation, np.ndarray) and observation.ndim == 1:
            observation = observation.reshape(1, -1)
        
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return float(action[0])
    
    def get_observation(self):
        """
        Safely get current observation from the original environment.
        FIXED ISSUE 3: Public method instead of private _get_observation
        """
        return self.original_env._get_observation()
    
    def evaluate(self, n_episodes=20, render=False):
        """
        Evaluate the trained agent comprehensively.
        """
        print(f"\n{'='*60}")
        print(f"Evaluating RL Agent")
        print(f"Number of episodes: {n_episodes}")
        print(f"{'='*60}\n")
        
        results = {
            'plating_events': 0,
            'charging_times': [],
            'final_soc': [],
            'max_temperatures': [],
            'min_anode_potential': [],
            'total_rewards': [],
            'steps_per_episode': []
        }
        
        # Use original environment for evaluation (not wrapped)
        eval_env = self.original_env
        
        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            step = 0
            episode_reward = 0
            min_anode = float('inf')
            
            while not done:
                action = self.get_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step([action])
                
                done = terminated or truncated
                episode_reward += reward
                step += 1
                
                # FIXED ISSUE 4: Handle missing 'time' key safely
                current_time = info.get('time', step * self.dt)
                current_anode = info.get('anode_potential', 0)
                if current_anode < min_anode:
                    min_anode = current_anode
            
            results['plating_events'] += 1 if info.get('plating_detected', False) else 0
            results['charging_times'].append(info.get('time', step * self.dt))
            results['final_soc'].append(info.get('soc', 0))
            results['max_temperatures'].append(info.get('temperature', 298.15) - 273.15)
            results['min_anode_potential'].append(min_anode)
            results['total_rewards'].append(episode_reward)
            results['steps_per_episode'].append(step)
            
            status = "⚠️ PLATING" if info.get('plating_detected', False) else "✓ SAFE"
            print(f"Episode {episode+1:3d}: {status} | "
                  f"SoC={info.get('soc', 0):.2f} | "
                  f"Time={info.get('time', 0):.0f}s | "
                  f"Temp={info.get('temperature', 298.15)-273.15:.1f}°C | "
                  f"Reward={episode_reward:.2f}")
        
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Plating events: {results['plating_events']}/{n_episodes} ({results['plating_events']/n_episodes*100:.1f}%)")
        print(f"Avg charging time: {np.mean(results['charging_times']):.1f}s")
        print(f"Avg final SoC: {np.mean(results['final_soc']):.3f}")
        print(f"{'='*60}\n")
        
        return results
    
    def predict_charging_profile(self, initial_soc=0.0, max_steps=200):
        """
        Generate a complete charging profile using the trained policy.
        """
        # FIXED ISSUE 5: Use original environment, not wrapped
        env = self.original_env
        obs, _ = env.reset()
        
        profile = {
            'time': [0],
            'current': [0],
            'soc': [env.soc],
            'temperature': [env.temperature],
            'anode_potential': [env.anode_potential],
            'voltage': [env.voltage]
        }
        
        done = False
        step = 0
        
        while not done and step < max_steps:
            action = self.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step([action])
            done = terminated or truncated
            
            profile['time'].append(info.get('time', step * self.dt))
            profile['current'].append(info.get('current', action))
            profile['soc'].append(info.get('soc', env.soc))
            profile['temperature'].append(info.get('temperature', env.temperature))
            profile['anode_potential'].append(info.get('anode_potential', env.anode_potential))
            profile['voltage'].append(info.get('voltage', env.voltage))
            
            step += 1
        
        return profile
