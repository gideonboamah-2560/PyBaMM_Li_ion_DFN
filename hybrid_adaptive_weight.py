"""
Hybrid Controller Architecture 1: Adaptive Weight Hybrid
RL learns adaptive weight between MPC and RL actions.

u_final = w * u_mpc + (1-w) * u_rl
where w is learned by RL based on current state.

This allows the controller to trust MPC when it's safe and switch to
RL when MPC is too conservative.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
from typing import Optional, Dict, List, Tuple
import warnings


class WeightNetwork(nn.Module):
    """Network that outputs adaptive weight between 0 and 1"""

    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, x):
        return self.net(x)


class SimpleRLPolicy:
    """Simple learned policy for RL component"""

    def __init__(self, max_current=3.0):
        self.current = 1.5
        self.max_current = max_current

    def get_action(self, observation):
        soc = observation[0]
        temp = observation[1]
        anode = observation[2] if len(observation) > 2 else 0.1

        temp_c = temp - 273.15

        # Simple adaptive policy
        if anode < 0.05:
            self.current = max(0.3, self.current * 0.9)
        elif soc < 0.3 and temp_c < 35:
            self.current = min(self.max_current, self.current + 0.05)
        elif soc > 0.7:
            self.current = max(0.5, self.current - 0.05)
        elif temp_c > 40:
            self.current = max(0.3, self.current - 0.1)

        return np.clip(self.current, 0.2, self.max_current)


class AdaptiveWeightHybrid:
    """
    Hybrid Controller Architecture 1: Adaptive Weight Hybrid

    u_final = w * u_mpc + (1-w) * u_rl
    where w is learned by RL based on current state.
    """

    def __init__(self, env, mpc, learning_rate=1e-3, hidden_dim=64):
        """
        Initialize Adaptive Weight Hybrid Controller.

        Args:
            env: Battery environment
            mpc: MPC controller instance
            learning_rate: Learning rate for weight network
            hidden_dim: Hidden layer dimension for neural network
        """
        self.env = env
        self.mpc = mpc
        self.max_current = env.max_current
        self.dt = env.dt
        self.target_soc = env.target_soc

        # Store original env for evaluation
        self.original_env = env

        # State dimension (SoC, Temperature, Anode_Potential, Voltage)
        self.state_dim = env.observation_space.shape[0]

        # Weight network: learns optimal mixing weight based on state
        self.weight_network = WeightNetwork(self.state_dim, hidden_dim)
        self.optimizer = optim.Adam(self.weight_network.parameters(), lr=learning_rate)

        # Experience replay for training
        self.replay_buffer = deque(maxlen=5000)
        self.batch_size = 64
        self.gamma = 0.99

        # RL policy (can be upgraded to trained DQN later)
        self.rl_policy = SimpleRLPolicy(max_current=env.max_current)

        # Training tracking
        self.training_step = 0
        self.loss_history = []
        self.weight_history = []
        self.reward_history = []

        print(f"Adaptive Weight Hybrid Controller initialized:")
        print(f"  State dimension: {self.state_dim}")
        print(f"  Hidden dimension: {hidden_dim}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {self.batch_size}")

    def get_action(self, observation, deterministic=True):
        """
        Get hybrid action with adaptive weighting.

        Args:
            observation: Current state [SoC, Temperature, Anode, Voltage]
            deterministic: If True, use current weight directly (no exploration)

        Returns:
            Charging current in C-rate
        """
        # Get MPC action
        soc = observation[0]
        temp = observation[1]
        u_mpc = self.mpc.solve([soc, temp])

        # Get RL action
        u_rl = self.rl_policy.get_action(observation)

        # Get adaptive weight from network
        with torch.no_grad():
            state = torch.FloatTensor(observation).unsqueeze(0)
            weight = self.weight_network(state).item()

            # Add exploration noise during training
            if not deterministic:
                weight = weight + np.random.normal(0, 0.1)

            # Keep weight in reasonable range (both controllers contribute)
            weight = np.clip(weight, 0.1, 0.9)

        # Combine actions
        u_final = weight * u_mpc + (1 - weight) * u_rl
        u_final = np.clip(u_final, 0.2, self.max_current)

        # Store for analysis
        self.weight_history.append(weight)

        return u_final

    def train_step(self, state, action, reward, next_state, done):
        """
        Train the weight network using experience replay.

        The network learns to produce weights that lead to higher rewards.
        """
        # Store experience
        self.replay_buffer.append((state, action, reward, next_state, done))

        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        rewards = torch.FloatTensor(np.array(rewards))

        # Get current weights
        weights = self.weight_network(states).squeeze()

        # Normalize rewards for stability
        if len(rewards) > 1 and rewards.std() > 0:
            rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            rewards_normalized = rewards

        # Policy gradient loss: encourage weights that lead to higher rewards
        # Higher weight = more trust in MPC, lower weight = more trust in RL
        # The loss pushes weights in the direction that increases rewards
        loss = -(rewards_normalized * torch.log(weights + 1e-8)).mean()

        # Optional: Add regularization to keep weights from extremes
        loss = loss + 0.01 * ((weights - 0.5) ** 2).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.weight_network.parameters(), 1.0)
        self.optimizer.step()

        self.training_step += 1
        self.loss_history.append(loss.item())

        return loss.item()

    def train(self, total_timesteps=30000, save_path="hybrid_adaptive_model"):
        """
        Train the adaptive weight hybrid controller.

        Args:
            total_timesteps: Number of training steps
            save_path: Path to save the trained model
        """
        print(f"\n{'=' * 60}")
        print(f"Training Adaptive Weight Hybrid Controller")
        print(f"Total timesteps: {total_timesteps}")
        print(f"{'=' * 60}\n")

        episode = 0
        step_count = 0
        episode_reward = 0
        episode_weights = []

        obs, _ = self.env.reset()

        while step_count < total_timesteps:
            # Get action with exploration
            action = self.get_action(obs, deterministic=False)

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step([action])
            done = terminated or truncated

            # Train network on this transition
            loss = self.train_step(obs, action, reward, next_obs, done)

            episode_reward += reward
            episode_weights.append(self.weight_history[-1] if self.weight_history else 0.5)

            obs = next_obs
            step_count += 1

            if done:
                episode += 1
                self.reward_history.append(episode_reward)

                if episode % 20 == 0:
                    avg_reward = np.mean(self.reward_history[-20:]) if self.reward_history else 0
                    avg_weight = np.mean(episode_weights[-20:]) if episode_weights else 0.5
                    print(f"Episode {episode:4d}: Avg Reward={avg_reward:7.2f}, "
                          f"Avg Weight={avg_weight:.3f}, Loss={loss:.4f}")

                # Reset
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_weights = []

        # Save model
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        torch.save({
            'weight_network_state_dict': self.weight_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'reward_history': self.reward_history
        }, save_path + '.pt')

        print(f"\nModel saved to {save_path}.pt")
        print(f"Training completed: {episode} episodes, {step_count} steps")

        return self

    def load(self, model_path):
        """Load trained model"""
        checkpoint = torch.load(model_path + '.pt')
        self.weight_network.load_state_dict(checkpoint['weight_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {model_path}.pt")

    def evaluate(self, n_episodes=10):
        """
        Evaluate the trained hybrid controller.

        Args:
            n_episodes: Number of episodes to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n{'=' * 60}")
        print(f"Evaluating Adaptive Weight Hybrid Controller")
        print(f"Number of episodes: {n_episodes}")
        print(f"{'=' * 60}\n")

        results = {
            'plating_events': 0,
            'charging_times': [],
            'final_soc': [],
            'max_temperatures': [],
            'min_anode_potential': [],
            'total_rewards': [],
            'avg_weights': [],
            'mpc_actions': [],
            'rl_actions': []
        }

        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            step = 0
            episode_reward = 0
            min_anode = float('inf')
            episode_weights = []
            episode_mpc_actions = []
            episode_rl_actions = []

            while not done:
                # Get MPC and RL actions for logging
                soc = obs[0]
                temp = obs[1]
                u_mpc = self.mpc.solve([soc, temp])
                u_rl = self.rl_policy.get_action(obs)

                # Get hybrid action (deterministic for evaluation)
                with torch.no_grad():
                    state = torch.FloatTensor(obs).unsqueeze(0)
                    weight = self.weight_network(state).item()
                    weight = np.clip(weight, 0.1, 0.9)

                u_final = weight * u_mpc + (1 - weight) * u_rl
                u_final = np.clip(u_final, 0.2, self.max_current)

                # Step environment
                obs, reward, terminated, truncated, info = self.env.step([u_final])
                done = terminated or truncated
                episode_reward += reward
                step += 1

                # Track metrics
                episode_weights.append(weight)
                episode_mpc_actions.append(u_mpc)
                episode_rl_actions.append(u_rl)

                anode = info.get('anode_potential', 0)
                if anode < min_anode:
                    min_anode = anode

            # Record results
            results['plating_events'] += 1 if info.get('plating_detected', False) else 0
            results['charging_times'].append(info.get('time', step * self.dt))
            results['final_soc'].append(info.get('soc', 0))
            results['max_temperatures'].append(info.get('temperature', 298.15) - 273.15)
            results['min_anode_potential'].append(min_anode)
            results['total_rewards'].append(episode_reward)
            results['avg_weights'].append(np.mean(episode_weights))
            results['mpc_actions'].append(np.mean(episode_mpc_actions))
            results['rl_actions'].append(np.mean(episode_rl_actions))

            status = "⚠️ PLATING" if info.get('plating_detected', False) else "✓ SAFE"
            print(f"Episode {episode + 1:3d}: {status} | "
                  f"SoC={info.get('soc', 0):.2f} | "
                  f"Time={info.get('time', 0):.0f}s | "
                  f"Weight={np.mean(episode_weights):.3f}")

        # Summary
        print(f"\n{'=' * 60}")
        print("EVALUATION SUMMARY")
        print(f"{'=' * 60}")
        plating_rate = results['plating_events'] / n_episodes * 100
        avg_time = np.mean(results['charging_times'])
        avg_weight = np.mean(results['avg_weights'])
        print(f"Plating events: {results['plating_events']}/{n_episodes} ({plating_rate:.1f}%)")
        print(f"Avg charging time: {avg_time:.1f}s")
        print(f"Avg final SoC: {np.mean(results['final_soc']):.3f}")
        print(f"Avg weight (MPC trust): {avg_weight:.3f}")
        print(f"{'=' * 60}\n")

        return results

    def get_weight_statistics(self):
        """Get statistics about learned weights"""
        if not self.weight_history:
            return {'mean': 0.5, 'std': 0, 'min': 0.5, 'max': 0.5}

        return {
            'mean': float(np.mean(self.weight_history)),
            'std': float(np.std(self.weight_history)),
            'min': float(np.min(self.weight_history)),
            'max': float(np.max(self.weight_history))
        }


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING ADAPTIVE WEIGHT HYBRID CONTROLLER")
    print("=" * 60)

    try:
        from battery_environment import BatteryPlatingEnv
        from mpc_controller import BatteryMPC

        # Create environment and MPC
        print("\n1. Creating environment and MPC...")
        env = BatteryPlatingEnv(max_current_C=3.0, dt=10.0, target_soc=0.8)
        mpc = BatteryMPC(horizon=5, dt=10.0, max_current=3.0)
        print("   ✓ Environment and MPC created")

        # Create hybrid controller
        print("\n2. Creating Adaptive Weight Hybrid Controller...")
        hybrid = AdaptiveWeightHybrid(env, mpc)
        print("   ✓ Hybrid controller created")

        # Test action
        print("\n3. Testing action generation...")
        obs, _ = env.reset()
        action = hybrid.get_action(obs, deterministic=True)
        print(f"   ✓ Action: {action:.3f}C")

        # Test short training
        print("\n4. Testing short training (500 steps)...")
        hybrid.train(total_timesteps=500, save_path="test_hybrid_model")
        print("   ✓ Training completed")

        # Test evaluation
        print("\n5. Testing evaluation...")
        results = hybrid.evaluate(n_episodes=3)
        print("   ✓ Evaluation completed")

        print("\n" + "=" * 60)
        print("✓ All tests passed for Adaptive Weight Hybrid!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()