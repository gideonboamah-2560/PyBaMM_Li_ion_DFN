"""
Hybrid Controller Architecture 3: Switching Hybrid
RL learns to choose between MPC and RL controllers based on state.

action = MPC if switch_probability > threshold else RL_action

This meta-controller learns when MPC is better (high SoC, safety-critical)
and when RL is better (low SoC, need speed).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import warnings

warnings.filterwarnings("ignore")


class SwitchNetwork(nn.Module):
    """Network that outputs probability of using MPC (0 to 1)"""

    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1] = probability of using MPC
        )

    def forward(self, x):
        return self.net(x)


class SimpleRLPolicy:
    """Simple RL policy used when switching chooses RL"""

    def __init__(self, max_current=3.0):
        self.current = 1.5
        self.max_current = max_current

    def get_action(self, observation):
        soc = observation[0]
        temp = observation[1]
        anode = observation[2] if len(observation) > 2 else 0.1

        temp_c = temp - 273.15

        if anode < 0.05:
            self.current = max(0.3, self.current * 0.9)
        elif soc < 0.3 and temp_c < 35:
            self.current = min(self.max_current, self.current + 0.05)
        elif soc > 0.7:
            self.current = max(0.5, self.current - 0.05)
        elif temp_c > 40:
            self.current = max(0.3, self.current - 0.1)

        return np.clip(self.current, 0.2, self.max_current)


class SwitchingHybrid:
    """
    Hybrid Controller Architecture 3: Switching Hybrid

    The RL agent learns to choose between MPC and RL based on state.
    This is a meta-controller that selects the best strategy for each situation.
    """

    def __init__(self, env, mpc, learning_rate=1e-3, hidden_dim=64, temperature=1.0):
        """
        Initialize Switching Hybrid Controller.

        Args:
            env: Battery environment
            mpc: MPC controller instance
            learning_rate: Learning rate for switch network
            hidden_dim: Hidden layer dimension
            temperature: Temperature for softmax (controls exploration vs exploitation)
        """
        self.env = env
        self.mpc = mpc
        self.max_current = env.max_current
        self.dt = env.dt
        self.target_soc = env.target_soc
        self.temperature = temperature

        self.state_dim = env.observation_space.shape[0]

        # Switch network: learns when to use MPC vs RL
        self.switch_network = SwitchNetwork(self.state_dim, hidden_dim)
        self.optimizer = optim.Adam(self.switch_network.parameters(), lr=learning_rate)

        # Experience replay
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99

        # RL policy (used when switching chooses RL)
        self.rl_policy = SimpleRLPolicy(max_current=env.max_current)

        # Tracking
        self.training_step = 0
        self.loss_history = []
        self.switch_history = []  # 1 = MPC chosen, 0 = RL chosen
        self.reward_history = []

        # Statistics
        self.mpc_usage_count = 0
        self.rl_usage_count = 0

        # Cache MPC results
        self.mpc_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        print(f"Switching Hybrid Controller initialized:")
        print(f"  State dimension: {self.state_dim}")
        print(f"  Hidden dimension: {hidden_dim}")
        print(f"  Temperature: {temperature}")
        print(f"  Batch size: {self.batch_size}")

    def _get_mpc_action_cached(self, soc, temp):
        """Get MPC action with caching"""
        key = (round(soc, 3), round(temp, 3))

        if key in self.mpc_cache:
            self.cache_hits += 1
            return self.mpc_cache[key]

        self.cache_misses += 1
        action = self.mpc.solve([soc, temp])
        self.mpc_cache[key] = action
        return action

    def get_action(self, observation, deterministic=False):
        """
        Get action by switching between MPC and RL.

        Args:
            observation: Current state [SoC, Temperature, Anode, Voltage]
            deterministic: If True, use greedy policy (always choose higher probability)

        Returns:
            Charging current in C-rate
        """
        # Get MPC action
        soc = observation[0]
        temp = observation[1]
        u_mpc = self._get_mpc_action_cached(soc, temp)

        # Get RL action
        u_rl = self.rl_policy.get_action(observation)

        # Get probability of using MPC
        with torch.no_grad():
            state = torch.FloatTensor(observation).unsqueeze(0)
            mpc_prob = self.switch_network(state).item()
            mpc_prob = np.clip(mpc_prob, 0.05, 0.95)  # Keep both options possible

        # Decide which controller to use
        if deterministic:
            # Greedy: choose the one with higher probability
            use_mpc = mpc_prob > 0.5
        else:
            # Explore: sample based on probability with temperature
            # Apply temperature for exploration
            prob = mpc_prob / self.temperature if self.temperature > 0 else mpc_prob
            prob = np.clip(prob, 0.1, 0.9)
            use_mpc = random.random() < prob

        # Select action
        if use_mpc:
            action = u_mpc
            self.mpc_usage_count += 1
            self.switch_history.append(1)
        else:
            action = u_rl
            self.rl_usage_count += 1
            self.switch_history.append(0)

        return np.clip(action, 0.2, self.max_current)

    def train_step(self, state, action, reward, next_state, done, used_mpc):
        """
        Train the switch network using policy gradient.

        The network learns to choose the controller that leads to higher rewards.
        """
        self.replay_buffer.append((state, used_mpc, reward, next_state, done))

        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, used_mpc, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        used_mpc = torch.FloatTensor(np.array(used_mpc))
        rewards = torch.FloatTensor(np.array(rewards))

        # Get probabilities (probability of using MPC)
        probs = self.switch_network(states).squeeze()
        probs = torch.clamp(probs, 0.01, 0.99)  # Numerical stability

        # Normalize rewards for stability
        if len(rewards) > 1 and rewards.std() > 0:
            rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            rewards_normalized = rewards

        # Policy gradient loss:
        # log(prob) when MPC was chosen, log(1-prob) when RL was chosen
        log_probs = used_mpc * torch.log(probs) + (1 - used_mpc) * torch.log(1 - probs)
        loss = -(log_probs * rewards_normalized).mean()

        # Add entropy bonus for exploration
        entropy = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8)).mean()
        loss = loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.switch_network.parameters(), 1.0)
        self.optimizer.step()

        self.training_step += 1
        self.loss_history.append(loss.item())

        return loss.item()

    def train(self, total_timesteps=5000, save_path="hybrid_switching_model"):
        """
        Train the switching hybrid controller.

        Args:
            total_timesteps: Number of training steps
            save_path: Path to save the trained model
        """
        print(f"\n{'=' * 60}")
        print(f"Training Switching Hybrid Controller")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Temperature: {self.temperature}")
        print(f"{'=' * 60}\n")

        episode = 0
        step_count = 0
        episode_reward = 0
        episode_switches = []

        obs, _ = self.env.reset()

        while step_count < total_timesteps:
            # Get action with exploration
            action = self.get_action(obs, deterministic=False)
            used_mpc = self.switch_history[-1] if self.switch_history else 1

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step([action])
            done = terminated or truncated

            # Train network
            loss = self.train_step(obs, action, reward, next_obs, done, used_mpc)

            episode_reward += reward
            episode_switches.append(used_mpc)

            obs = next_obs
            step_count += 1

            if done:
                episode += 1
                self.reward_history.append(episode_reward)

                if episode % 10 == 0:
                    avg_reward = np.mean(self.reward_history[-10:]) if self.reward_history else 0
                    mpc_rate = np.mean(episode_switches) * 100 if episode_switches else 0
                    print(f"Episode {episode:4d}: Avg Reward={avg_reward:7.2f}, "
                          f"MPC Usage={mpc_rate:.1f}%, Loss={loss:.4f}")

                obs, _ = self.env.reset()
                episode_reward = 0
                episode_switches = []

        # Print statistics
        total_choices = self.mpc_usage_count + self.rl_usage_count
        print(f"\nController Usage Statistics:")
        print(f"  MPC used: {self.mpc_usage_count} times ({self.mpc_usage_count / total_choices * 100:.1f}%)")
        print(f"  RL used: {self.rl_usage_count} times ({self.rl_usage_count / total_choices * 100:.1f}%)")
        print(f"  MPC Cache: {self.cache_hits} hits, {self.cache_misses} misses")

        # Save model
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        torch.save({
            'switch_network_state_dict': self.switch_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
        }, save_path + '.pt')

        print(f"\nModel saved to {save_path}.pt")
        return self

    def load(self, model_path):
        """Load trained model"""
        checkpoint = torch.load(model_path + '.pt')
        self.switch_network.load_state_dict(checkpoint['switch_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {model_path}.pt")

    def evaluate(self, n_episodes=5):
        """
        Evaluate the trained switching hybrid controller.

        Args:
            n_episodes: Number of episodes to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n{'=' * 60}")
        print(f"Evaluating Switching Hybrid Controller")
        print(f"Number of episodes: {n_episodes}")
        print(f"{'=' * 60}\n")

        results = {
            'plating_events': 0,
            'charging_times': [],
            'final_soc': [],
            'max_temperatures': [],
            'min_anode_potential': [],
            'total_rewards': [],
            'mpc_usage_rate': [],
        }

        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            step = 0
            episode_reward = 0
            min_anode = float('inf')
            episode_mpc_usage = []

            while not done:
                # Get action (deterministic for evaluation)
                action = self.get_action(obs, deterministic=True)

                # Determine which controller was used
                soc = obs[0]
                temp = obs[1]
                with torch.no_grad():
                    state = torch.FloatTensor(obs).unsqueeze(0)
                    mpc_prob = self.switch_network(state).item()
                    used_mpc = mpc_prob > 0.5
                episode_mpc_usage.append(1 if used_mpc else 0)

                # Step environment
                obs, reward, terminated, truncated, info = self.env.step([action])
                done = terminated or truncated
                episode_reward += reward
                step += 1

                anode = info.get('anode_potential', 0)
                if anode < min_anode:
                    min_anode = anode

            results['plating_events'] += 1 if info.get('plating_detected', False) else 0
            results['charging_times'].append(info.get('time', step * self.dt))
            results['final_soc'].append(info.get('soc', 0))
            results['max_temperatures'].append(info.get('temperature', 298.15) - 273.15)
            results['min_anode_potential'].append(min_anode)
            results['total_rewards'].append(episode_reward)
            results['mpc_usage_rate'].append(np.mean(episode_mpc_usage))

            status = "⚠️" if info.get('plating_detected', False) else "✓"
            mpc_rate = results['mpc_usage_rate'][-1] * 100
            print(f"Episode {episode + 1:3d}: {status} SoC={info.get('soc', 0):.2f} "
                  f"Time={info.get('time', 0):.0f}s MPC Usage={mpc_rate:.0f}%")

        print(f"\n{'=' * 60}")
        print("EVALUATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Plating events: {results['plating_events']}/{n_episodes}")
        print(f"Avg charging time: {np.mean(results['charging_times']):.1f}s")
        print(f"Avg MPC usage: {np.mean(results['mpc_usage_rate']) * 100:.1f}%")
        print(f"{'=' * 60}\n")

        return results

    def get_switch_statistics(self):
        """Get statistics about controller switching"""
        if not self.switch_history:
            return {'mpc_rate': 0.5, 'rl_rate': 0.5, 'total_switches': 0}

        mpc_rate = np.mean(self.switch_history)
        return {
            'mpc_rate': float(mpc_rate),
            'rl_rate': float(1 - mpc_rate),
            'total_switches': len(self.switch_history),
            'mpc_count': self.mpc_usage_count,
            'rl_count': self.rl_usage_count
        }


# Test code
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING SWITCHING HYBRID CONTROLLER (ARCHITECTURE 3)")
    print("=" * 60)

    try:
        from battery_environment import BatteryPlatingEnv
        from mpc_controller import BatteryMPC

        print("\n1. Creating environment...")
        env = BatteryPlatingEnv(max_current_C=3.0, dt=10.0, target_soc=0.8)
        mpc = BatteryMPC(horizon=5, dt=10.0, max_current=3.0)
        print("   ✓ Created")

        print("\n2. Creating Switching Hybrid Controller...")
        hybrid = SwitchingHybrid(env, mpc, temperature=1.0)
        print("   ✓ Created")

        print("\n3. Testing action generation...")
        obs, _ = env.reset()
        action = hybrid.get_action(obs, deterministic=True)
        print(f"   ✓ Action: {action:.3f}C")

        print("\n4. Quick training (500 steps)...")
        hybrid.train(total_timesteps=500, save_path="test_switching_model")
        print("   ✓ Training complete")

        stats = hybrid.get_switch_statistics()
        print(f"\n   Switch statistics:")
        print(f"     MPC usage rate: {stats['mpc_rate'] * 100:.1f}%")
        print(f"     Total switches: {stats['total_switches']}")

        print("\n5. Quick evaluation...")
        results = hybrid.evaluate(n_episodes=2)
        print("   ✓ Evaluation complete")

        print("\n" + "=" * 60)
        print("✓ All tests passed for Switching Hybrid!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()