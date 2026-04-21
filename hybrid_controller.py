"""
Hybrid MPC-RL Controller for Battery Fast Charging
Combines safety guarantees of MPC with adaptability of RL

Two architectures:
1. HybridCompensationController: RL learns compensation to MPC actions
2. HybridWeightController: RL adapts MPC's cost weights

Research-grade implementation compatible with BatteryPlatingEnv and BatteryMPC.
"""

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime
from typing import Optional, Dict, List, Union
import warnings


class HybridCompensationController:
    """
    Hybrid controller where RL learns a compensation to MPC actions.
    
    u_final = u_mpc + u_rl, with u_rl bounded for safety.
    
    This architecture preserves MPC's safety guarantees while allowing RL
    to learn fine-tuned corrections that improve performance.
    
    Reference: Chen et al., IEEE TTE 2025 - "MPC-Guided Deep Reinforcement Learning"
    """
    
    def __init__(
        self, 
        env, 
        mpc, 
        rl_model_path: Optional[str] = None, 
        max_compensation: float = 0.5,
        tensorboard_log: str = "./logs/",
        force_torch_check: bool = False
    ):
        """
        Initialize hybrid compensation controller.
        
        Args:
            env: Gym environment (BatteryPlatingEnv)
            mpc: MPC controller instance (BatteryMPC)
            rl_model_path: Path to trained RL model (optional)
            max_compensation: Maximum RL compensation in C-rate (safety bound)
            tensorboard_log: Directory for TensorBoard logs
            force_torch_check: Force check for torch installation
        """
        self.env = env
        self.mpc = mpc
        self.max_comp = max_compensation
        
        # Store original environment parameters
        self.max_current = env.max_current
        self.dt = env.dt
        self.target_soc = env.target_soc
        
        # FIXED: Use a single environment reference
        self.working_env = env
        
        # Wrap environment with Monitor for metrics recording (only if training)
        self.monitor_dir = None
        self.monitored_env = None
        
        # Check torch availability
        self._torch_available = self._check_torch()
        if not self._torch_available and not force_torch_check:
            warnings.warn("PyTorch not installed. SAC may fail. Install with: pip install torch")
        
        # Hyperparameters for SAC (optimized for compensation learning)
        self.hyperparams = {
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
        }
        
        # Add policy kwargs only if torch is available
        if self._torch_available:
            self.hyperparams["policy_kwargs"] = dict(
                net_arch=[256, 256],
                activation_fn=self._get_activation_fn()
            )
        
        # Create or load RL model
        if rl_model_path and os.path.exists(rl_model_path):
            self.rl_model = SAC.load(rl_model_path, env=self.working_env)
            print(f"Loaded RL model from {rl_model_path}")
        else:
            self.rl_model = SAC(
                "MlpPolicy",
                self.working_env,
                **self.hyperparams,
                verbose=1,
                tensorboard_log=tensorboard_log
            )
            print("Created new SAC model for compensation learning")
        
        # Tracking variables with size limits
        self.compensation_history = []
        self.max_history_size = 10000
        self.training_history = {
            'timesteps': [],
            'mean_rewards': [],
            'plating_rates': []
        }
    
    def _check_torch(self) -> bool:
        """Check if torch is available"""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def _get_activation_fn(self):
        """Get activation function (requires torch)"""
        import torch
        return torch.nn.ReLU
    
    def _check_file_exists(self, path: str) -> bool:
        """Check if model file exists"""
        return os.path.exists(path)
    
    def _add_to_history(self, value: float, history_list: list):
        """Add value to history with size limit"""
        history_list.append(value)
        if len(history_list) > self.max_history_size:
            history_list.pop(0)
    
    def get_action(
        self, 
        observation: np.ndarray, 
        use_rl: bool = True,
        deterministic: bool = True
    ) -> float:
        """
        Get combined MPC+RL action.
        
        Args:
            observation: Current state [SoC, Temp, Anode, Voltage]
            use_rl: Whether to use RL compensation
            deterministic: If True, use deterministic policy
            
        Returns:
            Final charging current in C-rate
        """
        # Extract SoC and temperature for MPC
        soc = float(observation[0])
        temp = float(observation[1])
        
        # Get MPC baseline action
        u_mpc = self.mpc.solve([soc, temp])
        
        if not use_rl:
            return float(u_mpc)
        
        # Get RL compensation
        rl_comp = 0.0
        try:
            # Ensure observation has correct shape
            if observation.ndim == 1:
                obs_reshaped = observation.reshape(1, -1)
            else:
                obs_reshaped = observation
            
            rl_comp_result = self.rl_model.predict(obs_reshaped, deterministic=deterministic)
            rl_comp = float(rl_comp_result[0][0])
        except Exception as e:
            # Fallback if RL prediction fails
            warnings.warn(f"RL prediction failed: {e}. Using MPC only.")
        
        # Bound compensation for safety
        rl_comp = np.clip(rl_comp, -self.max_comp, self.max_comp)
        
        # Combine MPC and RL actions
        u_final = u_mpc + rl_comp
        u_final = np.clip(u_final, 0.0, self.max_current)
        
        # Record compensation for analysis (with size limit)
        self._add_to_history(rl_comp, self.compensation_history)
        
        return float(u_final)
    
    def train(
        self, 
        total_timesteps: int = 100000, 
        save_path: str = "hybrid_rl_model",
        eval_freq: int = 5000
    ) -> object:
        """
        Train the RL compensation agent.
        
        The RL agent learns to add compensation to MPC actions to improve
        charging speed while respecting safety bounds.
        
        Args:
            total_timesteps: Number of training steps
            save_path: Path to save the trained model
            eval_freq: Evaluate every N steps
            
        Returns:
            Trained model
        """
        print(f"\n{'='*60}")
        print(f"Training Hybrid Compensation Controller")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Max compensation: ±{self.max_comp}C")
        print(f"{'='*60}\n")
        
        # Create monitored environment for training
        self.monitor_dir = f"./monitor_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.monitored_env = Monitor(self.working_env, self.monitor_dir)
        
        # Update model's environment
        self.rl_model.set_env(self.monitored_env)
        
        # Custom callback for training progress
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
                        avg_reward = np.mean(self.episode_rewards[-20:]) if self.episode_rewards else 0
                        plating_rate = (self.plating_episodes / self.episodes) * 100 if self.episodes > 0 else 0
                        print(f"Episode {self.episodes}: Avg Reward={avg_reward:.2f}, "
                              f"Plating Rate={plating_rate:.1f}%")
                    
                    self.total_reward = 0
                
                return True
        
        # Create evaluation environment
        eval_env = self._create_eval_env()
        
        # Setup evaluation callback
        best_model_dir = f"./best_{save_path}"
        os.makedirs(best_model_dir, exist_ok=True)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_dir,
            log_path=f"./eval_logs_hybrid/",
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            verbose=1
        )
        
        # Training callback
        training_callback = TrainingCallback(self)
        
        # Train the model
        self.rl_model.learn(
            total_timesteps=total_timesteps,
            callback=[training_callback, eval_callback],
            progress_bar=True
        )
        
        # Save the final model
        self.rl_model.save(save_path)
        print(f"\nModel saved to {save_path}")
        
        # Store training history
        self.training_history['timesteps'].append(total_timesteps)
        
        return self.rl_model
    
    def _create_eval_env(self):
        """Create a separate environment for evaluation"""
        try:
            from battery_environment import BatteryPlatingEnv
            return BatteryPlatingEnv(
                max_current_C=self.max_current,
                dt=self.dt,
                target_soc=self.target_soc
            )
        except ImportError:
            warnings.warn("Could not import BatteryPlatingEnv. Using existing environment.")
            return self.working_env
    
    def evaluate(self, n_episodes: int = 20) -> Dict[str, Union[float, List[float]]]:
        """
        Evaluate the hybrid controller comprehensively.
        
        Args:
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating Hybrid Compensation Controller")
        print(f"Number of episodes: {n_episodes}")
        print(f"{'='*60}\n")
        
        results = {
            'plating_events': 0,
            'charging_times': [],
            'final_soc': [],
            'max_temperatures': [],
            'min_anode_potential': [],
            'total_rewards': [],
            'steps_per_episode': [],
            'avg_compensation': [],
            'mpc_only_baseline': []
        }
        
        # Use the working environment directly
        eval_env = self.working_env
        
        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            step = 0
            episode_reward = 0
            min_anode = float('inf')
            episode_compensations = []
            
            # Get MPC-only baseline for comparison
            soc = obs[0]
            temp = obs[1]
            mpc_only = self.mpc.solve([soc, temp])
            results['mpc_only_baseline'].append(mpc_only)
            
            # Reset compensation tracking for this episode
            episode_compensations = []
            
            while not done:
                # Get hybrid action
                action = self.get_action(obs, use_rl=True, deterministic=True)
                
                # Step environment
                obs, reward, terminated, truncated, info = eval_env.step([action])
                
                done = terminated or truncated
                episode_reward += reward
                step += 1
                
                # Track minimum anode potential
                anode_val = info.get('anode_potential', 0)
                if anode_val < min_anode:
                    min_anode = anode_val
            
            # Record results
            results['plating_events'] += 1 if info.get('plating_detected', False) else 0
            results['charging_times'].append(info.get('time', step * self.dt))
            results['final_soc'].append(info.get('soc', 0))
            results['max_temperatures'].append(info.get('temperature', 298.15) - 273.15)
            results['min_anode_potential'].append(min_anode)
            results['total_rewards'].append(episode_reward)
            results['steps_per_episode'].append(step)
            results['avg_compensation'].append(np.mean(self.compensation_history[-step:]) if self.compensation_history else 0)
            
            # Print progress
            status = "⚠️ PLATING" if info.get('plating_detected', False) else "✓ SAFE"
            print(f"Episode {episode+1:3d}: {status} | "
                  f"SoC={info.get('soc', 0):.2f} | "
                  f"Time={info.get('time', 0):.0f}s | "
                  f"Temp={info.get('temperature', 298.15)-273.15:.1f}°C")
        
        # Compute summary statistics
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total episodes:        {n_episodes}")
        print(f"Plating events:        {results['plating_events']} ({results['plating_events']/n_episodes*100:.1f}%)")
        print(f"Avg charging time:     {np.mean(results['charging_times']):.1f}s")
        print(f"Avg final SoC:         {np.mean(results['final_soc']):.3f}")
        print(f"Avg max temperature:   {np.mean(results['max_temperatures']):.1f}°C")
        print(f"{'='*60}\n")
        
        return results
    
    def get_compensation_statistics(self) -> Dict[str, float]:
        """Get statistics about the learned compensation"""
        if not self.compensation_history:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        return {
            'mean': float(np.mean(self.compensation_history)),
            'std': float(np.std(self.compensation_history)),
            'min': float(np.min(self.compensation_history)),
            'max': float(np.max(self.compensation_history))
        }


class HybridWeightController:
    """
    Hybrid controller where RL adapts MPC's cost weights.
    
    RL outputs [w_current, w_soc] that reshape MPC's objective function.
    This allows RL to adjust the trade-off between current smoothness
    and charging speed based on the current state.
    """
    
    def __init__(
        self, 
        env, 
        mpc, 
        rl_model_path: Optional[str] = None,
        tensorboard_log: str = "./logs/"
    ):
        """
        Initialize hybrid weight adaptation controller.
        
        Args:
            env: Gym environment
            mpc: MPC controller instance (must support rl_weights parameter)
            rl_model_path: Path to trained RL model
            tensorboard_log: Directory for TensorBoard logs
        """
        self.env = env
        self.mpc = mpc
        
        # Store original environment parameters
        self.max_current = env.max_current
        self.dt = env.dt
        self.target_soc = env.target_soc
        
        # Use a single environment reference
        self.working_env = env
        
        # RL outputs weights in range [0.1, 10.0]
        self.weight_bounds = (0.1, 10.0)
        
        # Hyperparameters
        self.hyperparams = {
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
        }
        
        # Create or load RL model
        if rl_model_path and os.path.exists(rl_model_path):
            self.rl_model = SAC.load(rl_model_path, env=self.working_env)
            print(f"Loaded RL model from {rl_model_path}")
        else:
            self.rl_model = SAC(
                "MlpPolicy",
                self.working_env,
                **self.hyperparams,
                verbose=1,
                tensorboard_log=tensorboard_log
            )
            print("Created new SAC model for weight adaptation learning")
        
        # Tracking with size limits
        self.weight_history = []
        self.max_history_size = 10000
    
    def _add_to_history(self, value, history_list):
        """Add value to history with size limit"""
        history_list.append(value)
        if len(history_list) > self.max_history_size:
            history_list.pop(0)
    
    def _check_file_exists(self, path: str) -> bool:
        """Check if model file exists"""
        return os.path.exists(path)
    
    def get_action(
        self, 
        observation: np.ndarray, 
        use_rl: bool = True,
        deterministic: bool = True
    ) -> float:
        """
        Get action using RL-adapted MPC weights.
        
        Args:
            observation: Current state [SoC, Temp, Anode, Voltage]
            use_rl: Whether to use RL for weight adaptation
            deterministic: If True, use deterministic policy
            
        Returns:
            Charging current in C-rate
        """
        soc = float(observation[0])
        temp = float(observation[1])
        
        if use_rl:
            try:
                # Get RL weights
                if observation.ndim == 1:
                    obs_reshaped = observation.reshape(1, -1)
                else:
                    obs_reshaped = observation
                
                weights = self.rl_model.predict(obs_reshaped, deterministic=deterministic)[0]
                weights = np.clip(weights, self.weight_bounds[0], self.weight_bounds[1])
                
                # Store for analysis (with size limit)
                self._add_to_history(weights.copy(), self.weight_history)
                
                # Solve MPC with adapted weights
                u = self.mpc.solve([soc, temp], rl_weights=weights)
            except Exception as e:
                warnings.warn(f"RL weight prediction failed: {e}. Using MPC only.")
                u = self.mpc.solve([soc, temp])
        else:
            u = self.mpc.solve([soc, temp])
        
        return float(u)
    
    def train(
        self, 
        total_timesteps: int = 100000, 
        save_path: str = "hybrid_weight_model",
        eval_freq: int = 5000
    ) -> object:
        """
        Train the RL weight adaptation agent.
        
        Args:
            total_timesteps: Number of training steps
            save_path: Path to save the model
            eval_freq: Evaluate every N steps
            
        Returns:
            Trained model
        """
        print(f"\n{'='*60}")
        print(f"Training Hybrid Weight Adaptation Controller")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Weight bounds: [{self.weight_bounds[0]}, {self.weight_bounds[1]}]")
        print(f"{'='*60}\n")
        
        # Create monitored environment for training
        monitor_dir = f"./monitor_weight_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        monitored_env = Monitor(self.working_env, monitor_dir)
        
        # Update model's environment
        self.rl_model.set_env(monitored_env)
        
        # Create evaluation environment
        try:
            from battery_environment import BatteryPlatingEnv
            eval_env = BatteryPlatingEnv(
                max_current_C=self.max_current,
                dt=self.dt,
                target_soc=self.target_soc
            )
        except ImportError:
            eval_env = self.working_env
        
        # Setup evaluation callback
        best_model_dir = f"./best_{save_path}"
        os.makedirs(best_model_dir, exist_ok=True)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_dir,
            log_path=f"./eval_logs_weight/",
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            verbose=1
        )
        
        # Train
        self.rl_model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        # Save model
        self.rl_model.save(save_path)
        print(f"\nModel saved to {save_path}")
        
        return self.rl_model
    
    def evaluate(self, n_episodes: int = 20) -> Dict[str, Union[float, List[float]]]:
        """
        Evaluate the weight adaptation controller.
        
        Args:
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating Hybrid Weight Adaptation Controller")
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
        
        eval_env = self.working_env
        
        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            step = 0
            episode_reward = 0
            min_anode = float('inf')
            
            while not done:
                action = self.get_action(obs, use_rl=True, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step([action])
                
                done = terminated or truncated
                episode_reward += reward
                step += 1
                
                anode_val = info.get('anode_potential', 0)
                if anode_val < min_anode:
                    min_anode = anode_val
            
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
                  f"Temp={info.get('temperature', 298.15)-273.15:.1f}°C")
        
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total episodes:        {n_episodes}")
        print(f"Plating events:        {results['plating_events']} ({results['plating_events']/n_episodes*100:.1f}%)")
        print(f"Avg charging time:     {np.mean(results['charging_times']):.1f}s")
        print(f"Avg final SoC:         {np.mean(results['final_soc']):.3f}")
        print(f"{'='*60}\n")
        
        return results
    
    def get_weight_statistics(self) -> Dict[str, float]:
        """Get statistics about the learned weights"""
        if not self.weight_history:
            return {'mean_current_weight': 0, 'mean_soc_weight': 0}
        
        weights_array = np.array(self.weight_history)
        return {
            'mean_current_weight': float(np.mean(weights_array[:, 0])),
            'mean_soc_weight': float(np.mean(weights_array[:, 1])),
            'std_current_weight': float(np.std(weights_array[:, 0])),
            'std_soc_weight': float(np.std(weights_array[:, 1]))
        }


# Test code
if __name__ == "__main__":
    print("Testing Hybrid Controllers...")
    print("Note: This requires battery_environment.py and mpc_controller.py")
    
    try:
        from battery_environment import BatteryPlatingEnv
        from mpc_controller import BatteryMPC
        
        # Create environment and MPC
        print("\n1. Creating environment and MPC...")
        env = BatteryPlatingEnv(max_current_C=3.0, dt=10.0, target_soc=0.8)
        mpc = BatteryMPC(horizon=5, dt=10.0, max_current=3.0)
        print("   ✓ Created successfully")
        
        # Test Hybrid Compensation Controller
        print("\n2. Testing HybridCompensationController...")
        hybrid_comp = HybridCompensationController(env, mpc)
        obs, _ = env.reset()
        action = hybrid_comp.get_action(obs, use_rl=False)
        print(f"   MPC only action: {action:.3f}C")
        action = hybrid_comp.get_action(obs, use_rl=True)
        print(f"   Hybrid action: {action:.3f}C")
        
        # Test Hybrid Weight Controller
        print("\n3. Testing HybridWeightController...")
        hybrid_weight = HybridWeightController(env, mpc)
        action = hybrid_weight.get_action(obs, use_rl=False)
        print(f"   MPC only action: {action:.3f}C")
        action = hybrid_weight.get_action(obs, use_rl=True)
        print(f"   Hybrid action: {action:.3f}C")
        
        print("\n" + "="*50)
        print("All hybrid controller tests passed!")
        print("="*50)
        
    except ImportError as e:
        print(f"\nError: Could not import required modules: {e}")
        print("Make sure battery_environment.py and mpc_controller.py are in the same directory.")
    except Exception as e:
        print(f"\nError during testing: {e}")
