"""
Complete Experiment Runner for Battery Fast Charging Controller Comparison
Master script to run and compare all controllers.

Research-grade implementation for journal publication.
"""

import numpy as np
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Import project modules with error handling
try:
    from battery_environment import BatteryPlatingEnv
    print("✓ Imported BatteryPlatingEnv")
except ImportError as e:
    print(f"✗ Failed to import BatteryPlatingEnv: {e}")
    raise

try:
    from mpc_controller import BatteryMPC, SimpleMPC
    print("✓ Imported BatteryMPC")
except ImportError as e:
    print(f"✗ Failed to import BatteryMPC: {e}")
    raise

try:
    from rl_controller import BatteryRLController
    print("✓ Imported BatteryRLController")
except ImportError as e:
    print(f"✗ Failed to import BatteryRLController: {e}")
    raise

try:
    from hybrid_controller import HybridCompensationController, HybridWeightController
    print("✓ Imported HybridCompensationController")
except ImportError as e:
    print(f"✗ Failed to import HybridController: {e}")
    raise


class ExperimentRunner:
    """
    Complete experiment runner for comparing all charging controllers.
    """
    
    def __init__(
        self,
        max_current: float = 3.0,
        dt: float = 10.0,
        target_soc: float = 0.8,
        seed: int = 42,
        results_dir: str = "./experiment_results"
    ):
        """
        Initialize the experiment runner.
        """
        self.max_current = max_current
        self.dt = dt
        self.target_soc = target_soc
        self.seed = seed
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "models"), exist_ok=True)
        
        # Set random seed
        np.random.seed(seed)
        
        # Results storage
        self.all_results = {}
        
        print("="*80)
        print("BATTERY FAST CHARGING CONTROLLER COMPARISON")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Max current: {max_current}C")
        print(f"  Time step: {dt}s")
        print(f"  Target SoC: {target_soc*100:.0f}%")
        print(f"  Random seed: {seed}")
        print(f"  Results dir: {results_dir}")
        print("="*80)
    
    def _create_env(self) -> BatteryPlacingEnv:
        """Create a new environment instance."""
        return BatteryPlatingEnv(
            max_current_C=self.max_current,
            dt=self.dt,
            target_soc=self.target_soc
        )
    
    def run_constant_current_baselines(
        self, 
        currents: List[float] = None,
        n_episodes: int = 3
    ) -> Dict[str, Dict]:
        """
        Run constant current charging baselines.
        """
        if currents is None:
            currents = [0.5, 1.0, 1.5, 2.0]
        
        print("\n" + "-"*60)
        print("CONSTANT CURRENT BASELINES")
        print("-"*60)
        
        results = {}
        
        for current in currents:
            print(f"\nTesting constant current: {current}C")
            start_time = time.time()
            
            env = self._create_env()
            current_results = {
                'plating_events': 0,
                'charging_times': [],
                'final_soc': [],
                'max_temperatures': [],
                'min_anode_potentials': []
            }
            
            for episode in range(n_episodes):
                obs, _ = env.reset()
                done = False
                step = 0
                episode_min_anode = float('inf')
                
                while not done:
                    obs, reward, terminated, truncated, info = env.step([current])
                    done = terminated or truncated
                    step += 1
                    
                    anode = info.get('anode_potential', 0)
                    if anode < episode_min_anode:
                        episode_min_anode = anode
                
                current_results['plating_events'] += 1 if info.get('plating_detected', False) else 0
                current_results['charging_times'].append(info.get('time', step * self.dt))
                current_results['final_soc'].append(info.get('soc', 0))
                current_results['max_temperatures'].append(info.get('temperature', 298.15) - 273.15)
                current_results['min_anode_potentials'].append(episode_min_anode)
                
                status = "⚠️" if info.get('plating_detected', False) else "✓"
                print(f"  Episode {episode+1}: {status} SoC={info.get('soc', 0):.2f} "
                      f"Time={info.get('time', 0):.0f}s")
            
            # Compute summary
            n = n_episodes
            results[f"Constant {current}C"] = {
                'plating_rate': current_results['plating_events'] / n,
                'avg_charging_time': float(np.mean(current_results['charging_times'])),
                'std_charging_time': float(np.std(current_results['charging_times'])),
                'avg_final_soc': float(np.mean(current_results['final_soc'])),
                'avg_max_temp': float(np.mean(current_results['max_temperatures'])),
                'avg_min_anode': float(np.mean(current_results['min_anode_potentials'])),
            }
            
            elapsed = time.time() - start_time
            print(f"  Completed in {elapsed:.1f}s | "
                  f"Plating: {current_results['plating_events']}/{n}")
        
        return results
    
    def run_mpc_baseline(
        self, 
        horizon: int = 10,
        n_episodes: int = 5
    ) -> Dict[str, Any]:
        """
        Run MPC baseline controller.
        """
        print("\n" + "-"*60)
        print("MPC BASELINE CONTROLLER")
        print("-"*60)
        
        start_time = time.time()
        
        env = self._create_env()
        mpc = BatteryMPC(
            horizon=horizon,
            dt=self.dt,
            max_current=self.max_current,
            capacity=3.0
        )
        
        results = {
            'plating_events': 0,
            'charging_times': [],
            'final_soc': [],
            'max_temperatures': [],
            'min_anode_potentials': [],
            'solve_times': []
        }
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            step = 0
            episode_min_anode = float('inf')
            
            while not done:
                solve_start = time.time()
                action = mpc.solve([obs[0], obs[1]])
                solve_time = time.time() - solve_start
                results['solve_times'].append(solve_time)
                
                obs, reward, terminated, truncated, info = env.step([action])
                done = terminated or truncated
                step += 1
                
                anode = info.get('anode_potential', 0)
                if anode < episode_min_anode:
                    episode_min_anode = anode
            
            results['plating_events'] += 1 if info.get('plating_detected', False) else 0
            results['charging_times'].append(info.get('time', step * self.dt))
            results['final_soc'].append(info.get('soc', 0))
            results['max_temperatures'].append(info.get('temperature', 298.15) - 273.15)
            results['min_anode_potentials'].append(episode_min_anode)
            
            status = "⚠️" if info.get('plating_detected', False) else "✓"
            print(f"Episode {episode+1:3d}: {status} SoC={info.get('soc', 0):.2f} "
                  f"Time={info.get('time', 0):.0f}s")
        
        elapsed = time.time() - start_time
        
        n = n_episodes
        summary = {
            'plating_rate': results['plating_events'] / n,
            'avg_charging_time': float(np.mean(results['charging_times'])),
            'std_charging_time': float(np.std(results['charging_times'])),
            'avg_final_soc': float(np.mean(results['final_soc'])),
            'avg_max_temp': float(np.mean(results['max_temperatures'])),
            'avg_min_anode': float(np.mean(results['min_anode_potentials'])),
            'avg_solve_time_ms': float(np.mean(results['solve_times']) * 1000),
            'total_time_sec': elapsed,
        }
        
        print(f"\nMPC Summary:")
        print(f"  Plating rate: {summary['plating_rate']*100:.1f}%")
        print(f"  Avg charging time: {summary['avg_charging_time']:.1f}s")
        
        return summary
    
    def run_rl_controller(
        self,
        total_timesteps: int = 30000,  # Reduced for faster testing
        n_eval_episodes: int = 5,
        load_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Train and evaluate RL controller.
        """
        print("\n" + "-"*60)
        print("RL CONTROLLER (SAC)")
        print("-"*60)
        
        env = self._create_env()
        model_path = os.path.join(self.results_dir, "models", "rl_model")
        
        # FIXED: Proper load_existing logic
        model_exists = os.path.exists(model_path + ".zip") or os.path.exists(model_path)
        
        if load_existing and model_exists:
            print(f"Loading existing model from {model_path}")
            rl = BatteryRLController(env, model_path=model_path)
        else:
            print(f"Training new RL agent for {total_timesteps} timesteps...")
            rl = BatteryRLController(env)
            
            train_start = time.time()
            rl.train(total_timesteps=total_timesteps, save_path=model_path)
            train_time = time.time() - train_start
            print(f"Training completed in {train_time/60:.1f} minutes")
        
        # Evaluate
        print("\nEvaluating RL agent...")
        eval_results = rl.evaluate(n_episodes=n_eval_episodes)
        
        # FIXED: Safe access to eval_results
        plating_events = eval_results.get('plating_events', 0)
        charging_times = eval_results.get('charging_times', [0] * n_eval_episodes)
        final_soc = eval_results.get('final_soc', [0] * n_eval_episodes)
        max_temps = eval_results.get('max_temperatures', [25] * n_eval_episodes)
        min_anodes = eval_results.get('min_anode_potential', [0.1] * n_eval_episodes)
        
        summary = {
            'plating_rate': plating_events / n_eval_episodes,
            'avg_charging_time': float(np.mean(charging_times)),
            'std_charging_time': float(np.std(charging_times)),
            'avg_final_soc': float(np.mean(final_soc)),
            'avg_max_temp': float(np.mean(max_temps)),
            'avg_min_anode': float(np.mean(min_anodes)),
        }
        
        print(f"\nRL Summary:")
        print(f"  Plating rate: {summary['plating_rate']*100:.1f}%")
        print(f"  Avg charging time: {summary['avg_charging_time']:.1f}s")
        
        return summary
    
    def run_hybrid_controller(
        self,
        total_timesteps: int = 20000,  # Reduced for faster testing
        n_eval_episodes: int = 5,
        max_compensation: float = 0.5,
        load_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Train and evaluate hybrid compensation controller.
        """
        print("\n" + "-"*60)
        print("HYBRID CONTROLLER (MPC + RL Compensation)")
        print("-"*60)
        
        env = self._create_env()
        mpc = BatteryMPC(horizon=10, dt=self.dt, max_current=self.max_current)
        model_path = os.path.join(self.results_dir, "models", "hybrid_model")
        
        # FIXED: Proper load_existing logic
        model_exists = os.path.exists(model_path + ".zip") or os.path.exists(model_path)
        
        if load_existing and model_exists:
            print(f"Loading existing hybrid model from {model_path}")
            hybrid = HybridCompensationController(
                env, mpc, rl_model_path=model_path, max_compensation=max_compensation
            )
        else:
            print(f"Training new hybrid agent for {total_timesteps} timesteps...")
            hybrid = HybridCompensationController(
                env, mpc, max_compensation=max_compensation
            )
            
            train_start = time.time()
            hybrid.train(total_timesteps=total_timesteps, save_path=model_path)
            train_time = time.time() - train_start
            print(f"Training completed in {train_time/60:.1f} minutes")
        
        # Evaluate
        print("\nEvaluating hybrid controller...")
        eval_results = hybrid.evaluate(n_episodes=n_eval_episodes)
        
        # FIXED: Safe access to eval_results
        plating_events = eval_results.get('plating_events', 0)
        charging_times = eval_results.get('charging_times', [0] * n_eval_episodes)
        final_soc = eval_results.get('final_soc', [0] * n_eval_episodes)
        max_temps = eval_results.get('max_temperatures', [25] * n_eval_episodes)
        min_anodes = eval_results.get('min_anode_potential', [0.1] * n_eval_episodes)
        
        # Get compensation statistics if available
        try:
            comp_stats = hybrid.get_compensation_statistics()
            avg_comp = comp_stats.get('mean', 0)
        except:
            avg_comp = 0
        
        summary = {
            'plating_rate': plating_events / n_eval_episodes,
            'avg_charging_time': float(np.mean(charging_times)),
            'std_charging_time': float(np.std(charging_times)),
            'avg_final_soc': float(np.mean(final_soc)),
            'avg_max_temp': float(np.mean(max_temps)),
            'avg_min_anode': float(np.mean(min_anodes)),
            'avg_compensation': avg_comp,
        }
        
        print(f"\nHybrid Summary:")
        print(f"  Plating rate: {summary['plating_rate']*100:.1f}%")
        print(f"  Avg charging time: {summary['avg_charging_time']:.1f}s")
        
        return summary
    
    def run_complete_experiment(
        self,
        skip_constant: bool = False,
        skip_mpc: bool = False,
        skip_rl: bool = False,
        skip_hybrid: bool = False,
        rl_timesteps: int = 30000,
        hybrid_timesteps: int = 20000
    ) -> Dict[str, Any]:
        """
        Run the complete experiment comparing all controllers.
        """
        print("\n" + "="*80)
        print("STARTING COMPLETE EXPERIMENT")
        print("="*80)
        
        all_results = {}
        
        # 1. Constant current baselines
        if not skip_constant:
            all_results['constant_current'] = self.run_constant_current_baselines()
        
        # 2. MPC baseline
        if not skip_mpc:
            all_results['mpc'] = self.run_mpc_baseline()
        
        # 3. RL controller
        if not skip_rl:
            all_results['rl'] = self.run_rl_controller(total_timesteps=rl_timesteps)
        
        # 4. Hybrid controller
        if not skip_hybrid:
            all_results['hybrid'] = self.run_hybrid_controller(total_timesteps=hybrid_timesteps)
        
        # Save results
        self._save_results(all_results)
        
        # Print comparison
        self._print_comparison(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"experiment_results_{timestamp}.json")
        
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        
        serializable = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable[key] = {}
                for subkey, subvalue in value.items():
                    if subkey != 'raw_data':
                        serializable[key][subkey] = convert(subvalue)
            else:
                serializable[key] = convert(value)
        
        with open(results_file, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
    
    def _print_comparison(self, results: Dict[str, Any]) -> None:
        """Print final comparison table."""
        print("\n" + "="*80)
        print("FINAL RESULTS COMPARISON")
        print("="*80)
        
        print(f"\n{'Controller':<30} {'Plating Rate':<15} {'Time (s)':<12} {'Temp (°C)':<10}")
        print("-"*70)
        
        # Constant current
        if 'constant_current' in results:
            for name, data in results['constant_current'].items():
                rate = data.get('plating_rate', 0) * 100
                time_val = data.get('avg_charging_time', 0)
                temp_val = data.get('avg_max_temp', 0)
                print(f"{name:<30} {rate:>6.1f}%{'':<8} {time_val:>8.1f}   {temp_val:>8.1f}")
        
        # MPC
        if 'mpc' in results:
            data = results['mpc']
            rate = data.get('plating_rate', 0) * 100
            time_val = data.get('avg_charging_time', 0)
            temp_val = data.get('avg_max_temp', 0)
            print(f"{'MPC':<30} {rate:>6.1f}%{'':<8} {time_val:>8.1f}   {temp_val:>8.1f}")
        
        # RL
        if 'rl' in results:
            data = results['rl']
            rate = data.get('plating_rate', 0) * 100
            time_val = data.get('avg_charging_time', 0)
            temp_val = data.get('avg_max_temp', 0)
            print(f"{'RL (SAC)':<30} {rate:>6.1f}%{'':<8} {time_val:>8.1f}   {temp_val:>8.1f}")
        
        # Hybrid
        if 'hybrid' in results:
            data = results['hybrid']
            rate = data.get('plating_rate', 0) * 100
            time_val = data.get('avg_charging_time', 0)
            temp_val = data.get('avg_max_temp', 0)
            print(f"{'Hybrid (MPC+RL)':<30} {rate:>6.1f}%{'':<8} {time_val:>8.1f}   {temp_val:>8.1f}")
        
        print("-"*70)
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE")
        print("="*80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Battery Fast Charging Controller Comparison')
    parser.add_argument('--skip-constant', action='store_true', help='Skip constant current')
    parser.add_argument('--skip-mpc', action='store_true', help='Skip MPC')
    parser.add_argument('--skip-rl', action='store_true', help='Skip RL')
    parser.add_argument('--skip-hybrid', action='store_true', help='Skip hybrid')
    parser.add_argument('--rl-timesteps', type=int, default=30000, help='RL training steps')
    parser.add_argument('--hybrid-timesteps', type=int, default=20000, help='Hybrid training steps')
    parser.add_argument('--load-models', action='store_true', help='Load existing models')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test only')
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test()
        return
    
    runner = ExperimentRunner(
        max_current=3.0,
        dt=10.0,
        target_soc=0.8,
        seed=42,
        results_dir="./experiment_results"
    )
    
    results = runner.run_complete_experiment(
        skip_constant=args.skip_constant,
        skip_mpc=args.skip_mpc,
        skip_rl=args.skip_rl,
        skip_hybrid=args.skip_hybrid,
        rl_timesteps=args.rl_timesteps,
        hybrid_timesteps=args.hybrid_timesteps
    )
    
    return results


def quick_test():
    """Quick test to verify imports and basic functionality."""
    print("\n" + "="*60)
    print("QUICK TEST")
    print("="*60)
    
    try:
        print("\n1. Creating environment...")
        env = BatteryPlatingEnv(max_current_C=3.0, dt=10.0, target_soc=0.8)
        print("   ✓ Environment created")
        
        print("\n2. Testing MPC...")
        mpc = BatteryMPC(horizon=5, dt=10.0, max_current=3.0)
        obs, _ = env.reset()
        action = mpc.solve([obs[0], obs[1]])
        print(f"   ✓ MPC action: {action:.2f}C")
        
        print("\n3. Testing environment step...")
        obs, reward, terminated, truncated, info = env.step([1.0])
        print(f"   ✓ Step complete: SoC={obs[0]:.2f}, Reward={reward:.2f}")
        
        print("\n" + "="*60)
        print("Quick test passed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Quick test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
