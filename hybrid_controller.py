"""
Hybrid MPC-RL controller for battery fast charging
Combines safety guarantees of MPC with adaptability of RL
"""

import numpy as np
from stable_baselines3 import SAC


class HybridCompensationController:
    """
    Hybrid controller where RL learns a compensation to MPC actions.
    u_final = u_mpc + u_rl, with u_rl bounded for safety.
    """
    
    def __init__(self, env, mpc, rl_model_path=None, max_compensation=0.5):
        """
        Initialize hybrid compensation controller.
        
        Args:
            env: Gym environment
            mpc: MPC controller instance
            rl_model_path: Path to trained RL model (optional)
            max_compensation: Maximum RL compensation in C-rate
        """
        self.env = env
        self.mpc = mpc
        self.max_comp = max_compensation
        
        # Create or load RL model for compensation
        if rl_model_path and self._check_file_exists(rl_model_path):
            self.rl_model = SAC.load(rl_model_path)
            print(f"Loaded RL model from {rl_model_path}")
        else:
            # Create new RL model for compensation
            # Note: Action space is compensation amount (bounded)
            self.rl_model = SAC(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                buffer_size=50000,
                batch_size=128,
                verbose=0
            )
            print("Created new RL model for compensation learning")
        
        self.compensation_history = []
    
    def _check_file_exists(self, path):
        """Check if model file exists"""
        import os
        return os.path.exists(path)
    
    def get_action(self, observation, use_rl=True):
        """
        Get combined MPC+RL action.
        
        Args:
            observation: Current state [SoC, Temp, Anode, Voltage]
            use_rl: Whether to use RL compensation
            
        Returns:
            Final charging current in C-rate
        """
        # Extract SoC and temperature for MPC
        soc = observation[0]
        temp = observation[1]
        
        # Get MPC baseline action
        u_mpc = self.mpc.solve([soc, temp])
        
        if not use_rl:
            return u_mpc
        
        # Get RL compensation
        rl_comp = self.rl_model.predict(observation, deterministic=True)[0]
        rl_comp = float(rl_comp[0])
        
        # Bound compensation for safety
        rl_comp = np.clip(rl_comp, -self.max_comp, self.max_comp)
        
        # Combine
        u_final = u_mpc + rl_comp
        u_final = np.clip(u_final, 0, self.mpc.max_current)
        
        self.compensation_history.append(rl_comp)
        
        return u_final
    
    def train(self, total_timesteps=50000, save_path="hybrid_rl_model"):
        """
        Train the RL compensation agent.
        
        Args:
            total_timesteps: Number of training steps
            save_path: Path to save the model
        """
        print(f"Training hybrid compensation agent for {total_timesteps} timesteps...")
        
        # Note: Training happens in the environment with MPC baseline
        self.rl_model.learn(total_timesteps=total_timesteps)
        self.rl_model.save(save_path)
        print(f"Model saved to {save_path}")
    
    def evaluate(self, n_episodes=10):
        """
        Evaluate hybrid controller.
        
        Args:
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"Evaluating Hybrid Controller for {n_episodes} episodes...")
        
        results = {
            'plating_events': 0,
            'charging_times': [],
            'final_soc': [],
            'max_temperatures': [],
            'avg_compensation': []
        }
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            step = 0
            
            while not done:
                action = self.get_action(obs, use_rl=True)
                obs, reward, terminated, truncated, info = self.env.step([action])
                done = terminated or truncated
                step += 1
            
            results['plating_events'] += 1 if info.get('plating_detected', False) else 0
            results['charging_times'].append(info.get('time', step * self.env.dt))
            results['final_soc'].append(info.get('soc', 0))
            results['max_temperatures'].append(info.get('temperature', 298.15) - 273.15)
            results['avg_compensation'].append(np.mean(self.compensation_history) if self.compensation_history else 0)
            
            print(f"  Episode {episode+1}: Plating={info.get('plating_detected', False)}, "
                  f"SoC={info.get('soc', 0):.2f}, Time={info.get('time', 0):.0f}s")
        
        return results


class HybridWeightController:
    """
    Hybrid controller where RL adapts MPC's cost weights.
    RL outputs [w_current, w_soc] that reshape MPC's objective.
    """
    
    def __init__(self, env, mpc, rl_model_path=None):
        """
        Initialize hybrid weight adaptation controller.
        
        Args:
            env: Gym environment
            mpc: MPC controller instance (must support rl_weights parameter)
            rl_model_path: Path to trained RL model
        """
        self.env = env
        self.mpc = mpc
        
        # RL outputs weights in range [0.1, 10.0]
        if rl_model_path and self._check_file_exists(rl_model_path):
            self.rl_model = SAC.load(rl_model_path)
        else:
            self.rl_model = SAC(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                buffer_size=50000,
                batch_size=128,
                verbose=0
            )
    
    def _check_file_exists(self, path):
        import os
        return os.path.exists(path)
    
    def get_action(self, observation, use_rl=True):
        """
        Get action using RL-adapted MPC weights.
        
        Args:
            observation: Current state
            use_rl: Whether to use RL for weight adaptation
            
        Returns:
            Charging current in C-rate
        """
        soc = observation[0]
        temp = observation[1]
        
        if use_rl:
            # Get RL weights
            weights = self.rl_model.predict(observation, deterministic=True)[0]
            weights = np.clip(weights, 0.1, 10.0)
            u = self.mpc.solve([soc, temp], rl_weights=weights)
        else:
            u = self.mpc.solve([soc, temp])
        
        return u
    
    def train(self, total_timesteps=50000, save_path="hybrid_weight_model"):
        """Train the RL weight adaptation agent"""
        print(f"Training hybrid weight adaptation agent for {total_timesteps} timesteps...")
        self.rl_model.learn(total_timesteps=total_timesteps)
        self.rl_model.save(save_path)
