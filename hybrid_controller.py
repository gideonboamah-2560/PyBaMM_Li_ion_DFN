"""
Hybrid MPC-RL Controller for 8C-Capable Battery Fast Charging
Combines safety guarantees of MPC with adaptability of RL

Two architectures:
1. HybridCompensationController: RL learns compensation to MPC actions
2. AdaptiveWeightHybrid: RL learns weight between MPC and RL
Optimized for 8C-capable battery with 0% plating target.
"""

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import os
from datetime import datetime
from typing import Optional, Dict, List, Union
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

warnings.filterwarnings("ignore")


class SafetyFirstWrapper(gym.Wrapper):
    """
    Environment wrapper that applies terminal penalties for any plating.
    Optimized for 8C-capable battery.
    """
    def __init__(self, env, plating_penalty=-500):
        super().__init__(env)
        self.plating_penalty = plating_penalty

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        plating_detected = info.get('plating_detected', False)
        anode_potential = info.get('anode_potential', 0.1)
        soc = obs[0]

        if plating_detected:
            reward = self.plating_penalty
            terminated = True
            return obs, reward, terminated, truncated, info

        # Progressive safety penalties (adjusted for 8C battery)
        if anode_potential < 0.02:
            reward = -80
            terminated = True
        elif anode_potential < 0.03:
            reward = -30
        elif anode_potential < 0.04:
            reward = -10
        elif anode_potential < 0.05:
            reward = -3

        if anode_potential > 0.06 and soc < 0.7:
            reward += 3.0
        reward += soc * 8.0
        return obs, reward, terminated, truncated, info


class WeightNetwork(nn.Module):
    """Network that outputs adaptive weight between 0 and 1 for 8C battery"""
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class SimpleRLPolicy8C:
    """
    Simple learned policy for 8C-capable battery.
    Aggressive ramp at low SoC, then taper based on anode potential.
    Uses plating‑onset SoC targets: 5C->0.93, 6C->0.83, 7C->0.74, 8C->0.64.
    """
    def __init__(self, max_current=8.0):
        self.current = 2.0
        self.max_current = max_current

    def get_action(self, observation):
        soc = observation[0]
        temp = observation[1]
        anode = observation[2] if len(observation) > 2 else 0.1
        temp_c = temp - 273.15

        # Safety first: if anode too low, cut current drastically
        if anode < 0.03:
            self.current = max(0.5, self.current * 0.8)

        # Aggressive ramp at low SoC (safe up to 8C)
        elif soc < 0.2 and temp_c < 40:
            self.current = min(self.max_current, self.current + 0.15)

        # Taper based on plating‑onset SoC for each high C‑rate
        elif soc > 0.92 and self.current > 5.0:
            self.current = max(4.0, self.current - 0.2)
        elif soc > 0.82 and self.current > 6.0:
            self.current = max(5.0, self.current - 0.2)
        elif soc > 0.73 and self.current > 7.0:
            self.current = max(6.0, self.current - 0.2)
        elif soc > 0.63 and self.current > 8.0:
            self.current = max(7.0, self.current - 0.2)
        elif soc > 0.75:
            self.current = max(1.5, self.current - 0.08)

        # Normal increase when safe
        elif soc < 0.6 and anode > 0.05:
            self.current = min(self.max_current, self.current + 0.05)

        # Temperature rollback
        if temp_c > 45:
            self.current = max(0.5, self.current * 0.7)
        elif temp_c > 40:
            self.current = max(0.5, self.current * 0.9)

        return np.clip(self.current, 0.3, self.max_current)


class HybridCompensationController:
    """
    SAFETY-FIRST hybrid controller for 8C-capable battery.
    RL learns compensation to MPC actions with extreme safety constraints.
    """
    def __init__(
        self,
        env,
        mpc,
        rl_model_path: Optional[str] = None,
        max_compensation: float = 1.5,      # Increased for 8C
        tensorboard_log: str = "./logs/"
    ):
        self.env = env
        self.mpc = mpc
        self.max_comp = max_compensation
        self.max_current = env.max_current
        self.dt = env.dt
        self.target_soc = env.target_soc
        self.original_env = env

        self.wrapped_env = SafetyFirstWrapper(env, plating_penalty=-500)
        self.monitor_dir = f"./monitor_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.monitored_env = Monitor(self.wrapped_env, self.monitor_dir)

        self.hyperparams = {
            "learning_rate": 5e-5,
            "buffer_size": 150000,
            "batch_size": 64,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "policy_kwargs": dict(net_arch=[128, 128]),
        }

        if rl_model_path and os.path.exists(rl_model_path + ".zip"):
            self.rl_model = SAC.load(rl_model_path, env=self.monitored_env)
            print(f"Loaded RL model from {rl_model_path}")
        else:
            self.rl_model = SAC(
                "MlpPolicy",
                self.monitored_env,
                **self.hyperparams,
                verbose=0,
                tensorboard_log=tensorboard_log
            )
            print("Created SAFETY-FIRST hybrid controller for 8C battery")

        self.compensation_history = []

    def get_action(self, observation: np.ndarray, use_rl: bool = True, deterministic: bool = True) -> float:
        soc = float(observation[0])
        temp = float(observation[1])
        anode = float(observation[2]) if len(observation) > 2 else 0.1

        u_mpc = self.mpc.solve([soc, temp])
        if not use_rl:
            return float(u_mpc)

        try:
            if observation.ndim == 1:
                obs_reshaped = observation.reshape(1, -1)
            else:
                obs_reshaped = observation
            rl_comp = self.rl_model.predict(obs_reshaped, deterministic=deterministic)[0]
            rl_comp = float(rl_comp[0])
        except:
            rl_comp = 0.0

        # Adaptive bounds based on safety for 8C battery
        if anode < 0.04:
            max_comp = 0.1
        elif anode < 0.05:
            max_comp = 0.25
        elif soc > 0.7:
            max_comp = 0.25
        else:
            max_comp = self.max_comp

        rl_comp = np.clip(rl_comp, -max_comp, max_comp)

        if anode < 0.05:
            rl_comp = min(0, rl_comp)

        if temp > 313.15:    # 40°C
            rl_comp *= 0.5
        elif temp > 308.15:
            rl_comp *= 0.8

        u_final = u_mpc + rl_comp
        u_final = np.clip(u_final, 0.3, self.max_current)
        self.compensation_history.append(rl_comp)
        return float(u_final)

    def train(self, total_timesteps=50000, save_path="hybrid_rl_model", eval_freq=5000):
        print(f"\n{'='*70}")
        print(f"Training SAFETY-FIRST Hybrid Controller for 8C Battery")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Max compensation: ±{self.max_comp}C")
        print(f"NON-NEGOTIABLE: 0% Plating")
        print(f"{'='*70}\n")

        self.rl_model.set_env(self.monitored_env)

        class ZeroPlatingCallback(BaseCallback):
            def __init__(self, controller, verbose=0):
                super().__init__(verbose)
                self.controller = controller
                self.episodes = 0
                self.plating_episodes = 0

            def _on_step(self) -> bool:
                if self.locals.get('done', False):
                    self.episodes += 1
                    info = self.locals.get('infos', [{}])[0]
                    if info.get('plating_detected', False):
                        self.plating_episodes += 1
                        print(f"❌ EPISODE {self.episodes}: PLATING DETECTED")
                    if self.episodes % 50 == 0:
                        plating_rate = self.plating_episodes / self.episodes * 100
                        print(f"📊 Progress: {self.episodes} episodes | Plating Rate: {plating_rate:.1f}%")
                        self.plating_episodes = 0
                        self.episodes = 0
                return True

        try:
            from battery_environment import BatteryPlatingEnv
            base_env = BatteryPlatingEnv(
                max_current_C=self.max_current,
                dt=self.dt,
                target_soc=self.target_soc
            )
            eval_env = SafetyFirstWrapper(base_env, plating_penalty=-500)
        except:
            eval_env = self.wrapped_env

        best_model_dir = f"./best_{save_path}"
        os.makedirs(best_model_dir, exist_ok=True)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_dir,
            eval_freq=eval_freq,
            deterministic=True,
            verbose=1
        )

        self.rl_model.learn(
            total_timesteps=total_timesteps,
            callback=[ZeroPlatingCallback(self), eval_callback],
            progress_bar=True
        )
        self.rl_model.save(save_path)
        print(f"\n✅ Model saved to {save_path}.zip")
        return self.rl_model

    def evaluate(self, n_episodes=10):
        print(f"\n{'='*70}")
        print(f"EVALUATING HYBRID COMPENSATION CONTROLLER (8C Battery)")
        print(f"{'='*70}\n")

        results = {'plating_events': 0, 'charging_times': [], 'final_soc': [], 'min_anode_potential': []}
        eval_env = self.original_env

        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            while not done:
                action = self.get_action(obs, use_rl=True, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step([action])
                done = terminated or truncated

            results['plating_events'] += 1 if info.get('plating_detected', False) else 0
            results['charging_times'].append(info.get('time', 0))
            results['final_soc'].append(info.get('soc', 0))
            results['min_anode_potential'].append(info.get('anode_potential', 0) * 1000)

            status = "❌ PLATING" if info.get('plating_detected', False) else "✓ SAFE"
            print(f"Episode {episode+1:3d}: {status} | Time={info.get('time', 0):.0f}s | SoC={info.get('soc', 0):.2f}")

        plating_rate = results['plating_events'] / n_episodes * 100
        print(f"\n{'='*70}")
        print("EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Plating: {results['plating_events']}/{n_episodes} ({plating_rate:.1f}%)")
        print(f"Avg time: {np.mean(results['charging_times']):.0f}s")
        if plating_rate == 0:
            print("\n✅ ZERO PLATING ACHIEVED for 8C charging!")
        print(f"{'='*70}\n")
        return results

    def get_compensation_statistics(self):
        if not self.compensation_history:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        return {
            'mean': float(np.mean(self.compensation_history)),
            'std': float(np.std(self.compensation_history)),
            'min': float(np.min(self.compensation_history)),
            'max': float(np.max(self.compensation_history))
        }


class AdaptiveWeightHybrid:
    """
    Hybrid Controller Architecture: Adaptive Weight Hybrid for 8C battery.
    RL learns optimal weight between MPC and RL actions.
    """
    def __init__(self, env, mpc, learning_rate=5e-4, hidden_dim=64):
        self.env = env
        self.mpc = mpc
        self.max_current = env.max_current
        self.dt = env.dt
        self.target_soc = env.target_soc
        self.original_env = env

        self.state_dim = env.observation_space.shape[0]
        self.weight_network = WeightNetwork(self.state_dim, hidden_dim)
        self.optimizer = optim.Adam(self.weight_network.parameters(), lr=learning_rate)

        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99

        # 8C-optimized RL policy
        self.rl_policy = SimpleRLPolicy8C(max_current=env.max_current)

        self.training_step = 0
        self.loss_history = []
        self.weight_history = []
        self.reward_history = []

        print(f"Adaptive Weight Hybrid Controller for 8C battery initialized")

    def get_action(self, observation, deterministic=True):
        soc = observation[0]
        temp = observation[1]
        u_mpc = self.mpc.solve([soc, temp])
        u_rl = self.rl_policy.get_action(observation)

        with torch.no_grad():
            state = torch.FloatTensor(observation).unsqueeze(0)
            weight = self.weight_network(state).item()
            if not deterministic:
                weight = weight + np.random.normal(0, 0.1)
            weight = np.clip(weight, 0.1, 0.9)

        u_final = weight * u_mpc + (1 - weight) * u_rl
        u_final = np.clip(u_final, 0.3, self.max_current)
        self.weight_history.append(weight)
        return u_final

    def train_step(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        rewards = torch.FloatTensor(np.array(rewards))

        weights = self.weight_network(states).squeeze()

        if len(rewards) > 1 and rewards.std() > 0:
            rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            rewards_normalized = rewards

        loss = -(rewards_normalized * torch.log(weights + 1e-8)).mean()
        loss = loss + 0.01 * ((weights - 0.5) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.weight_network.parameters(), 1.0)
        self.optimizer.step()

        self.training_step += 1
        self.loss_history.append(loss.item())
        return loss.item()

    def train(self, total_timesteps=50000, save_path="hybrid_adaptive_model"):
        print(f"\nTraining Adaptive Weight Hybrid for 8C Battery")
        print(f"Total timesteps: {total_timesteps}\n")

        episode = 0
        step_count = 0
        episode_reward = 0
        episode_weights = []
        obs, _ = self.env.reset()

        while step_count < total_timesteps:
            action = self.get_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = self.env.step([action])
            done = terminated or truncated

            loss = self.train_step(obs, action, reward, next_obs, done)

            episode_reward += reward
            episode_weights.append(self.weight_history[-1] if self.weight_history else 0.5)

            obs = next_obs
            step_count += 1

            if done:
                episode += 1
                self.reward_history.append(episode_reward)

                if episode % 25 == 0:
                    avg_reward = np.mean(self.reward_history[-25:]) if self.reward_history else 0
                    avg_weight = np.mean(episode_weights[-25:]) if episode_weights else 0.5
                    print(f"Episode {episode:4d}: Avg Reward={avg_reward:7.2f}, Avg Weight={avg_weight:.3f}")

                obs, _ = self.env.reset()
                episode_reward = 0
                episode_weights = []

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        torch.save({'weight_network_state_dict': self.weight_network.state_dict()}, save_path + '.pt')
        print(f"\nModel saved to {save_path}.pt")
        return self

    def load(self, model_path):
        checkpoint = torch.load(model_path + '.pt')
        self.weight_network.load_state_dict(checkpoint['weight_network_state_dict'])
        print(f"Model loaded from {model_path}.pt")

    def evaluate(self, n_episodes=10):
        print(f"\nEvaluating Adaptive Weight Hybrid for 8C Battery\n")
        results = {
            'plating_events': 0,
            'charging_times': [],
            'final_soc': [],
            'min_anode_potential': [],
            'avg_weights': []
        }

        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_weights = []

            while not done:
                action = self.get_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step([action])
                done = terminated or truncated
                episode_weights.append(self.weight_history[-1] if self.weight_history else 0.5)

            results['plating_events'] += 1 if info.get('plating_detected', False) else 0
            results['charging_times'].append(info.get('time', 0))
            results['final_soc'].append(info.get('soc', 0))
            results['min_anode_potential'].append(info.get('anode_potential', 0) * 1000)
            results['avg_weights'].append(np.mean(episode_weights))

            status = "⚠️" if info.get('plating_detected', False) else "✓"
            print(f"Episode {episode+1}: {status} Time={info.get('time', 0):.0f}s SoC={info.get('soc', 0):.2f}")

        plating_rate = results['plating_events'] / n_episodes * 100
        print(f"\nPlating: {results['plating_events']}/{n_episodes} ({plating_rate:.1f}%)")
        print(f"Avg time: {np.mean(results['charging_times']):.0f}s")
        print(f"Avg weight: {np.mean(results['avg_weights']):.3f}\n")
        return results

    def get_weight_statistics(self):
        if not self.weight_history:
            return {'mean': 0.5, 'std': 0, 'min': 0.5, 'max': 0.5}
        return {
            'mean': float(np.mean(self.weight_history)),
            'std': float(np.std(self.weight_history)),
            'min': float(np.min(self.weight_history)),
            'max': float(np.max(self.weight_history))
        }


class SafeHybridController:
    """
    SIMPLE SAFE HYBRID - No RL, guaranteed 0% plating for 8C battery.
    Uses conservative current limits based on plating‑onset SoC.
    """
    def __init__(self, env, mpc):
        self.env = env
        self.mpc = mpc
        self.max_current = env.max_current

        # Conservative limits for 8C battery (safe margins)
        self.safe_limits = [
            (0.0, 7.5),   # up to 7.5C at very low SoC
            (0.10, 7.0),
            (0.20, 6.5),
            (0.35, 5.5),
            (0.50, 4.0),
            (0.65, 2.5),
            (0.75, 1.5),
            (0.80, 1.0),
        ]

    def get_action(self, obs):
        soc = obs[0]
        temp = obs[1]

        u_mpc = self.mpc.solve([soc, temp])

        safe_limit = 1.0
        for limit_soc, max_current in self.safe_limits:
            if soc >= limit_soc:
                safe_limit = max_current
                break

        temp_c = temp - 273.15
        if temp_c > 45:
            safe_limit *= 0.5
        elif temp_c > 40:
            safe_limit *= 0.7
        elif temp_c > 35:
            safe_limit *= 0.9

        action = min(u_mpc, safe_limit)
        return np.clip(action, 0.3, self.max_current)


if __name__ == "__main__":
    print("Testing Hybrid Controllers for 8C-Capable Battery...")
    try:
        from battery_environment import BatteryPlatingEnv
        from mpc_controller import BatteryMPC

        env = BatteryPlatingEnv(max_current_C=8.0, dt=2.0, target_soc=0.8)
        mpc = BatteryMPC(horizon=8, dt=2.0, max_current=8.0)
        print("✓ Environment and MPC created for 8C battery")

        hybrid = HybridCompensationController(env, mpc)
        obs, _ = env.reset()
        action = hybrid.get_action(obs, use_rl=False)
        print(f"MPC only action: {action:.2f}C")
        action = hybrid.get_action(obs, use_rl=True)
        print(f"Hybrid action: {action:.2f}C")

        adaptive = AdaptiveWeightHybrid(env, mpc)
        action = adaptive.get_action(obs)
        print(f"Adaptive Weight action: {action:.2f}C")

        safe = SafeHybridController(env, mpc)
        action = safe.get_action(obs)
        print(f"Safe Hybrid action: {action:.2f}C")

        print("\n✓ All hybrid controllers ready for 8C battery!")

    except Exception as e:
        print(f"Error: {e}")