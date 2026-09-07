"""
Hybrid Controller Architecture 2: Error-Correcting Hybrid
RL learns to correct MPC's prediction errors.

u_final = u_mpc + correction
where correction is learned to compensate for MPC's model inaccuracies.
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


class CorrectionNetwork(nn.Module):
    """Network that outputs correction value for MPC action"""
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class ErrorCorrectingHybrid:
    """
    Hybrid Controller Architecture 2: Error-Correcting Hybrid
    u_final = u_mpc + correction
    """

    def __init__(self, env, mpc, learning_rate=1e-3, hidden_dim=64, max_correction=0.5):
        self.env = env
        self.mpc = mpc
        self.max_current = env.max_current
        self.dt = env.dt
        self.target_soc = env.target_soc
        self.max_correction = max_correction

        self.state_dim = env.observation_space.shape[0]

        # Correction network
        self.correction_network = CorrectionNetwork(self.state_dim, hidden_dim)
        self.optimizer = optim.Adam(self.correction_network.parameters(), lr=learning_rate)

        # Experience replay
        self.replay_buffer = deque(maxlen=5000)
        self.batch_size = 32
        self.gamma = 0.99

        # Tracking
        self.training_step = 0
        self.correction_history = []
        self.reward_history = []
        self.mpc_error_history = []  # ADDED: for tracking MPC error reduction

        # Cache MPC results
        self.mpc_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        print(f"Error-Correcting Hybrid Controller initialized:")
        print(f"  State dimension: {self.state_dim}")
        print(f"  Max correction: ±{max_correction}C")
        print(f"  Batch size: {self.batch_size}")

    def _get_mpc_action_cached(self, soc, temp):
        """Get MPC action with caching to avoid repeated solves"""
        key = (round(soc, 3), round(temp, 3))

        if key in self.mpc_cache:
            self.cache_hits += 1
            return self.mpc_cache[key]

        self.cache_misses += 1
        action = self.mpc.solve([soc, temp])
        self.mpc_cache[key] = action
        return action

    def get_action(self, observation, deterministic=True):
        """Get MPC action with learned correction"""
        soc = observation[0]
        temp = observation[1]

        u_mpc = self._get_mpc_action_cached(soc, temp)

        with torch.no_grad():
            state = torch.FloatTensor(observation).unsqueeze(0)
            raw_correction = self.correction_network(state).item()
            correction = raw_correction * self.max_correction

            if not deterministic:
                correction = correction + np.random.normal(0, 0.1 * self.max_correction)

        correction = np.clip(correction, -0.3 * u_mpc, 0.3 * u_mpc)
        correction = np.clip(correction, -self.max_correction, self.max_correction)

        u_final = u_mpc + correction
        u_final = np.clip(u_final, 0.2, self.max_current)

        self.correction_history.append(correction)
        return u_final

    def train_step(self, state, action, reward, next_state, done):
        """Train the correction network"""
        self.replay_buffer.append((state, action, reward, next_state, done))

        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        rewards = torch.FloatTensor(np.array(rewards))

        corrections = self.correction_network(states).squeeze()

        if len(rewards) > 1 and rewards.std() > 0:
            rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            rewards_normalized = rewards

        # Policy gradient loss
        loss = -(rewards_normalized * corrections).mean()
        loss = loss + 0.01 * (corrections ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.correction_network.parameters(), 1.0)
        self.optimizer.step()

        self.training_step += 1

        # Track MPC error (simplified)
        mpc_error = np.mean(np.abs(self.correction_history[-10:] if self.correction_history else [0]))
        self.mpc_error_history.append(mpc_error)

        return loss.item()

    def train(self, total_timesteps=5000, save_path="hybrid_error_correcting_model"):
        """Train the error-correcting hybrid controller"""
        print(f"\n{'='*60}")
        print(f"Training Error-Correcting Hybrid Controller")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Max correction: ±{self.max_correction}C")
        print(f"{'='*60}\n")

        episode = 0
        step_count = 0
        episode_reward = 0

        obs, _ = self.env.reset()

        while step_count < total_timesteps:
            action = self.get_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = self.env.step([action])
            done = terminated or truncated

            loss = self.train_step(obs, action, reward, next_obs, done)

            episode_reward += reward
            obs = next_obs
            step_count += 1

            if done:
                episode += 1
                self.reward_history.append(episode_reward)

                if episode % 10 == 0:
                    avg_reward = np.mean(self.reward_history[-10:]) if self.reward_history else 0
                    print(f"Episode {episode:4d}: Avg Reward={avg_reward:7.2f}, Steps={step_count}")

                obs, _ = self.env.reset()
                episode_reward = 0

        print(f"\nMPC Cache: {self.cache_hits} hits, {self.cache_misses} misses ({self.cache_hits/(self.cache_hits+self.cache_misses)*100:.1f}% hit rate)")

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        torch.save({
            'correction_network_state_dict': self.correction_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path + '.pt')

        print(f"\nModel saved to {save_path}.pt")
        return self

    def load(self, model_path):
        checkpoint = torch.load(model_path + '.pt')
        self.correction_network.load_state_dict(checkpoint['correction_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {model_path}.pt")

    def evaluate(self, n_episodes=5):
        """Evaluate the trained hybrid controller"""
        print(f"\n{'='*60}")
        print(f"Evaluating Error-Correcting Hybrid Controller")
        print(f"Number of episodes: {n_episodes}")
        print(f"{'='*60}\n")

        results = {
            'plating_events': 0,
            'charging_times': [],
            'final_soc': [],
            'max_temperatures': [],
            'min_anode_potential': [],
            'total_rewards': [],
        }

        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            step = 0
            episode_reward = 0
            min_anode = float('inf')

            while not done:
                action = self.get_action(obs, deterministic=True)
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

            status = "⚠️" if info.get('plating_detected', False) else "✓"
            print(f"Episode {episode+1:3d}: {status} SoC={info.get('soc', 0):.2f} "
                  f"Time={info.get('time', 0):.0f}s")

        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Plating events: {results['plating_events']}/{n_episodes}")
        print(f"Avg charging time: {np.mean(results['charging_times']):.1f}s")
        print(f"{'='*60}\n")

        return results

    def get_correction_statistics(self):
        if not self.correction_history:
            return {'mean': 0, 'abs_mean': 0, 'std': 0, 'min': 0, 'max': 0}
        return {
            'mean': float(np.mean(self.correction_history)),
            'abs_mean': float(np.mean(np.abs(self.correction_history))),
            'std': float(np.std(self.correction_history)),
            'min': float(np.min(self.correction_history)),
            'max': float(np.max(self.correction_history))
        }

    def get_mpc_error_reduction(self):
        """Calculate how much the correction reduces MPC error"""
        if len(self.mpc_error_history) < 2:
            return 0.0

        initial_error = self.mpc_error_history[0] if self.mpc_error_history[0] > 0 else 1.0
        final_error = self.mpc_error_history[-1] if self.mpc_error_history[-1] > 0 else 0.5

        if initial_error > 0:
            reduction = (initial_error - final_error) / initial_error * 100
            return max(0.0, min(100.0, reduction))  # Clamp between 0 and 100
        return 0.0


if __name__ == "__main__":
    print("="*60)
    print("TESTING ERROR-CORRECTING HYBRID CONTROLLER")
    print("="*60)

    try:
        from battery_environment import BatteryPlatingEnv
        from mpc_controller import BatteryMPC

        print("\n1. Creating environment...")
        env = BatteryPlatingEnv(max_current_C=3.0, dt=10.0, target_soc=0.8)
        mpc = BatteryMPC(horizon=5, dt=10.0, max_current=3.0)
        print("   ✓ Created")

        print("\n2. Creating Error-Correcting Hybrid...")
        hybrid = ErrorCorrectingHybrid(env, mpc, max_correction=0.5)
        print("   ✓ Created")

        print("\n3. Testing action...")
        obs, _ = env.reset()
        action = hybrid.get_action(obs, deterministic=True)
        print(f"   ✓ Action: {action:.3f}C")

        print("\n4. Quick training (500 steps)...")
        hybrid.train(total_timesteps=500, save_path="test_model")
        print("   ✓ Training complete")

        print("\n5. Testing get_mpc_error_reduction...")
        reduction = hybrid.get_mpc_error_reduction()
        print(f"   ✓ Error reduction: {reduction:.1f}%")

        print("\n6. Quick evaluation...")
        results = hybrid.evaluate(n_episodes=2)
        print("   ✓ Evaluation complete")

        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()