"""
Master script to run all controllers and compare results
Run this file to execute the complete experiment
"""

import numpy as np
from battery_environment import BatteryPlatingEnv
from mpc_controller import BatteryMPC, SimpleMPC
from rl_controller import BatteryRLController
from hybrid_controller import HybridCompensationController, HybridWeightController


def run_mpc_baseline(env, mpc, n_episodes=5):
    """Run MPC baseline controller"""
    print("\n" + "="*60)
    print("RUNNING MPC BASELINE")
    print("="*60)
    
    results = {'plating': 0, 'times': [], 'temps': []}
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done:
            action = mpc.solve([obs[0], obs[1]])
            obs, reward, terminated, truncated, info = env.step([action])
            done = terminated or truncated
            step += 1
        
        results['plating'] += 1 if info.get('plating_detected', False) else 0
        results['times'].append(info.get('time', step * env.dt))
        results['temps'].append(info.get('temperature', 298.15) - 273.15)
        
        print(f"Episode {episode+1}: Plating={info.get('plating_detected', False)}, "
              f"SoC={info.get('soc', 0):.2f}, Time={info.get('time', 0):.0f}s, "
              f"Temp={info.get('temperature', 298.15)-273.15:.1f}°C")
    
    return results


def run_rl_controller(env, n_timesteps=50000, n_eval_episodes=5):
    """Train and evaluate RL controller"""
    print("\n" + "="*60)
    print("RUNNING RL CONTROLLER")
    print("="*60)
    
    # Create RL controller
    rl = BatteryRLController(env, algorithm="SAC")
    
    # Train
    print(f"\nTraining RL for {n_timesteps} timesteps...")
    rl.train(total_timesteps=n_timesteps, save_path="models/rl_model")
    
    # Evaluate
    results = rl.evaluate(n_episodes=n_eval_episodes)
    
    return results


def run_hybrid_controller(env, n_timesteps=50000, n_eval_episodes=5):
    """Train and evaluate hybrid compensation controller"""
    print("\n" + "="*60)
    print("RUNNING HYBRID CONTROLLER (MPC + RL Compensation)")
    print("="*60)
    
    # Create MPC and hybrid controller
    mpc = BatteryMPC(horizon=10, dt=env.dt, max_current=env.max_current)
    hybrid = HybridCompensationController(env, mpc, rl_model_path=None)
    
    # Train RL compensation
    print(f"\nTraining hybrid for {n_timesteps} timesteps...")
    hybrid.train(total_timesteps=n_timesteps, save_path="models/hybrid_model")
    
    # Evaluate
    results = hybrid.evaluate(n_episodes=n_eval_episodes)
    
    return results


def run_constant_current(env, current=1.0, n_episodes=3):
    """Run constant current charging as baseline"""
    print(f"\n--- Constant Current {current}C ---")
    
    results = {'plating': 0, 'times': [], 'temps': []}
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done:
            obs, reward, terminated, truncated, info = env.step([current])
            done = terminated or truncated
            step += 1
        
        results['plating'] += 1 if info.get('plating_detected', False) else 0
        results['times'].append(info.get('time', step * env.dt))
        results['temps'].append(info.get('temperature', 298.15) - 273.15)
        
        print(f"  Episode {episode+1}: Plating={info.get('plating_detected', False)}, "
              f"SoC={info.get('soc', 0):.2f}, Time={info.get('time', 0):.0f}s")
    
    return results


def main():
    """Main function to run complete experiment"""
    print("\n" + "="*70)
    print("BATTERY FAST CHARGING CONTROLLER COMPARISON")
    print("Physics-Informed Hybrid MPC-RL for Lithium Plating Avoidance")
    print("="*70)
    
    # Create environment
    print("\nInitializing battery environment...")
    env = BatteryPlatingEnv(max_current_C=3.0, dt=10.0, target_soc=0.8)
    
    # Results storage
    all_results = {}
    
    # 1. Constant current baselines
    print("\n" + "-"*40)
    print("CONSTANT CURRENT BASELINES")
    print("-"*40)
    
    for current in [0.5, 1.0, 1.5, 2.0]:
        results = run_constant_current(env, current=current, n_episodes=2)
        all_results[f"Constant {current}C"] = results
    
    # 2. MPC baseline
    mpc = BatteryMPC(horizon=10, dt=env.dt, max_current=env.max_current)
    results = run_mpc_baseline(env, mpc, n_episodes=5)
    all_results["MPC"] = results
    
    # 3. RL controller (trains from scratch - may take time)
    print("\n" + "-"*40)
    print("RL CONTROLLER (Training from scratch)")
    print("-"*40)
    results = run_rl_controller(env, n_timesteps=30000, n_eval_episodes=5)
    all_results["RL"] = results
    
    # 4. Hybrid controller
    print("\n" + "-"*40)
    print("HYBRID CONTROLLER (MPC + RL Compensation)")
    print("-"*40)
    results = run_hybrid_controller(env, n_timesteps=30000, n_eval_episodes=5)
    all_results["Hybrid"] = results
    
    # Print final comparison
    print("\n" + "="*70)
    print("FINAL RESULTS COMPARISON")
    print("="*70)
    print(f"{'Controller':<20} {'Plating Events':<15} {'Avg Time (s)':<15} {'Max Temp (°C)':<15}")
    print("-"*70)
    
    for name, results in all_results.items():
        avg_time = np.mean(results['times']) if results['times'] else 0
        avg_temp = np.mean(results['temps']) if results['temps'] else 0
        print(f"{name:<20} {results['plating']:<15} {avg_time:<15.0f} {avg_temp:<15.1f}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    # Create models directory if it doesn't exist
    import os
    os.makedirs("models", exist_ok=True)
    
    # Run the experiment
    results = main()
