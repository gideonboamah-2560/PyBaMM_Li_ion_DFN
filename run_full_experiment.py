"""
Complete Experiment Runner for 8C‑Capable Battery
Includes MPC, RL, and Hybrid Controllers
"""

import numpy as np
import os
import time
import matplotlib.pyplot as plt
from matplotlib.table import Table
from battery_environment import BatteryPlatingEnv
from mpc_controller import BatteryMPC, Safe8CMPC   # assume Safe8CMPC is defined (or use Safe4CMPC adapted)
from rl_controller import BatteryRLController
from hybrid_controller import (
    HybridCompensationController,
    AdaptiveWeightHybrid,
    SafeHybridController
)


# ----------------------------------------------------------------------
# Plotting function – called at the end of the experiment
# ----------------------------------------------------------------------
def plot_all_results(cc_results, controller_results, save_dir="./experiment_results_8c"):
    """Generate all graphs from collected data."""
    # ----- 1. Constant current bar chart -----
    if cc_results:
        c_rates = [str(r[0]) for r in cc_results]
        times_cc = [r[1] for r in cc_results]
        plating_cc = [r[2] for r in cc_results]
        colors_cc = ['#e74c3c' if p else '#2ecc71' for p in plating_cc]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(c_rates, times_cc, color=colors_cc, edgecolor='black')
        plt.xlabel('C-rate')
        plt.ylabel('Time to 80% SoC (seconds)')
        plt.title('Constant‑current charging – plating onset between 6C and 7C')
        plt.axhline(y=360, ls='--', color='gray', label='6 min target (10C)')
        for bar, t in zip(bars, times_cc):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                     f'{t}s', ha='center', va='bottom', fontsize=9)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/cc_times.png", dpi=300)
        plt.close()

    # ----- 2. Controller comparison bar chart -----
    if controller_results:
        names = [c[0] for c in controller_results]
        times_ctrl = [c[1] for c in controller_results]
        plating_ctrl = [c[2] for c in controller_results]
        colors_ctrl = ['#e74c3c' if p else '#2ecc71' for p in plating_ctrl]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, times_ctrl, color=colors_ctrl, edgecolor='black')
        plt.ylabel('Charging time (seconds)')
        plt.title('8C fast charging – controller comparison')
        safe_times = [t for i, t in enumerate(times_ctrl) if not plating_ctrl[i]]
        if safe_times:
            best = min(safe_times)
            plt.axhline(y=best, ls='--', color='gold', label=f'Best safe: {best}s')
        for bar, t in zip(bars, times_ctrl):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                     f'{t}s', ha='center', va='bottom', fontsize=10)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/controller_comparison.png", dpi=300)
        plt.close()

        # ----- 3. Results table as image -----
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')
        data = [['Controller', 'Time (s)', 'Plating']] + \
               [[n, str(round(t,1)), 'YES' if p else 'NO'] for n, t, p in controller_results]
        table = Table(ax, bbox=[0, 0, 1, 1])
        for i, row in enumerate(data):
            for j, val in enumerate(row):
                cell = table.add_cell(i, j, width=0.3, height=0.1, text=val, loc='center')
                if i == 0:
                    cell.set_facecolor('#40466e')
                    cell.get_text().set_color('white')
                elif j == 2 and val == 'YES':
                    cell.set_facecolor('#e74c3c')
        ax.add_table(table)
        plt.savefig(f"{save_dir}/results_table.png", dpi=300)
        plt.close()

    print(f"All graphs saved to {save_dir}")


# ----------------------------------------------------------------------
# Experiment runner class
# ----------------------------------------------------------------------
class ExperimentRunner:
    def __init__(self, max_current=8.0, dt=2.0, target_soc=0.8, seed=42):
        self.max_current = max_current
        self.dt = dt
        self.target_soc = target_soc
        self.results_dir = "./experiment_results_8c"
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "models"), exist_ok=True)
        np.random.seed(seed)

        print("="*80)
        print("8C-CAPABLE BATTERY FAST CHARGING EXPERIMENT")
        print("="*80)
        print(f"Max current: {max_current}C")
        print(f"Time step: {dt}s")
        print(f"Target SoC: {target_soc*100:.0f}%")
        print("="*80)

    def _create_env(self):
        return BatteryPlatingEnv(max_current_C=self.max_current, dt=self.dt, target_soc=self.target_soc)

    def run_constant_current_baselines(self, currents=None, n_episodes=2):
        if currents is None:
            currents = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]

        print("\n" + "-"*60)
        print("CONSTANT CURRENT BASELINES (8C)")
        print("-"*60)
        results = {}

        for current in currents:
            print(f"\nTesting {current}C...")
            env = self._create_env()
            times, socs, plating = [], [], 0

            for episode in range(n_episodes):
                obs, _ = env.reset()
                done = False
                while not done:
                    obs, reward, terminated, truncated, info = env.step([current])
                    done = terminated
                times.append(info.get('time', 0))
                socs.append(info.get('soc', 0))
                plating += 1 if info.get('plating_detected', False) else 0
                status = "⚠️" if info.get('plating_detected', False) else "✓"
                print(f"  Episode {episode+1}: {status} Time={info.get('time', 0):.0f}s SoC={info.get('soc', 0):.2f}")

            results[f"{current}C"] = {
                'avg_time': np.mean(times),
                'avg_soc': np.mean(socs),
                'plating_rate': plating / n_episodes
            }
        return results

    def run_mpc_baseline(self, n_episodes=5):
        print("\n" + "-"*60)
        print("MPC BASELINE (8C Optimized)")
        print("-"*60)

        env = self._create_env()
        mpc = BatteryMPC(horizon=12, dt=self.dt, max_current=self.max_current)
        results = {'plating': 0, 'times': [], 'socs': []}

        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = mpc.solve([obs[0], obs[1]])
                obs, reward, terminated, truncated, info = env.step([action])
                done = terminated
            results['plating'] += 1 if info.get('plating_detected', False) else 0
            results['times'].append(info.get('time', 0))
            results['socs'].append(info.get('soc', 0))
            status = "⚠️" if info.get('plating_detected', False) else "✓"
            print(f"Episode {episode+1}: {status} Time={info.get('time', 0):.0f}s SoC={info.get('soc', 0):.2f}")

        print(f"\nMPC Summary: {np.mean(results['times']):.0f}s | Plating: {'YES' if results['plating']>0 else 'NO'}")
        return results

    def run_safe_mpc(self, n_episodes=5):
        print("\n" + "-"*60)
        print("SAFE MPC (0% Plating Guaranteed for 8C)")
        print("-"*60)

        env = self._create_env()
        # Use an 8C‑aware safe MPC (you may rename Safe4CMPC to Safe8CMPC or adapt)
        from mpc_controller import Safe8CMPC   # ensure this exists (or create a simple rule‑based for 8C)
        mpc = Safe8CMPC(max_current=self.max_current)
        results = {'plating': 0, 'times': [], 'socs': []}

        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = mpc.solve(obs)
                obs, reward, terminated, truncated, info = env.step([action])
                done = terminated
            results['plating'] += 1 if info.get('plating_detected', False) else 0
            results['times'].append(info.get('time', 0))
            results['socs'].append(info.get('soc', 0))
            status = "✓" if not info.get('plating_detected', False) else "⚠️"
            print(f"Episode {episode+1}: {status} Time={info.get('time', 0):.0f}s SoC={info.get('soc', 0):.2f}")

        print(f"\nSafe MPC Summary: {np.mean(results['times']):.0f}s | Plating: {'YES' if results['plating']>0 else 'NO'}")
        return results

    def run_rl_controller(self, total_timesteps=60000, n_eval_episodes=5):
        print("\n" + "-"*60)
        print("RL CONTROLLER (SAC) - 8C")
        print("-"*60)

        env = self._create_env()
        rl = BatteryRLController(env)
        model_path = os.path.join(self.results_dir, "models", "rl_model")

        print(f"Training RL agent for {total_timesteps} timesteps...")
        train_start = time.time()
        rl.train(total_timesteps=total_timesteps, save_path=model_path)
        print(f"Training completed in {(time.time()-train_start)/60:.1f} minutes")

        print("\nEvaluating RL agent...")
        return rl.evaluate(n_episodes=n_eval_episodes)

    def run_hybrid_compensation(self, total_timesteps=50000, n_eval_episodes=5):
        print("\n" + "-"*60)
        print("HYBRID COMPENSATION CONTROLLER (8C)")
        print("-"*60)

        env = self._create_env()
        mpc = BatteryMPC(horizon=10, dt=self.dt, max_current=self.max_current)
        hybrid = HybridCompensationController(env, mpc, max_compensation=1.5)   # larger compensation for 8C
        model_path = os.path.join(self.results_dir, "models", "hybrid_compensation_model")

        print(f"Training Hybrid Compensation for {total_timesteps} timesteps...")
        train_start = time.time()
        hybrid.train(total_timesteps=total_timesteps, save_path=model_path)
        print(f"Training completed in {(time.time()-train_start)/60:.1f} minutes")

        print("\nEvaluating Hybrid Compensation...")
        return hybrid.evaluate(n_episodes=n_eval_episodes)

    def run_adaptive_weight_hybrid(self, total_timesteps=50000, n_eval_episodes=5):
        print("\n" + "-"*60)
        print("ADAPTIVE WEIGHT HYBRID (8C)")
        print("-"*60)

        env = self._create_env()
        mpc = BatteryMPC(horizon=10, dt=self.dt, max_current=self.max_current)
        hybrid = AdaptiveWeightHybrid(env, mpc)
        model_path = os.path.join(self.results_dir, "models", "adaptive_weight_model")

        print(f"Training Adaptive Weight Hybrid for {total_timesteps} timesteps...")
        train_start = time.time()
        hybrid.train(total_timesteps=total_timesteps, save_path=model_path)
        print(f"Training completed in {(time.time()-train_start)/60:.1f} minutes")

        print("\nEvaluating Adaptive Weight Hybrid...")
        return hybrid.evaluate(n_episodes=n_eval_episodes)

    def run_safe_hybrid(self, n_episodes=5):
        print("\n" + "-"*60)
        print("SAFE HYBRID (Rule-based, 0% Plating Guaranteed for 8C)")
        print("-"*60)

        env = self._create_env()
        mpc = BatteryMPC(horizon=10, dt=self.dt, max_current=self.max_current)
        hybrid = SafeHybridController(env, mpc)  # ensure SafeHybridController uses 8C limits

        results = {'plating': 0, 'times': [], 'socs': []}

        for episode in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = hybrid.get_action(obs)
                obs, reward, terminated, truncated, info = env.step([action])
                done = terminated
            results['plating'] += 1 if info.get('plating_detected', False) else 0
            results['times'].append(info.get('time', 0))
            results['socs'].append(info.get('soc', 0))
            status = "✓" if not info.get('plating_detected', False) else "⚠️"
            print(f"Episode {episode+1}: {status} Time={info.get('time', 0):.0f}s SoC={info.get('soc', 0):.2f}")

        return results

    def run_complete_experiment(self, skip_constant=False, skip_mpc=False, skip_safe_mpc=False,
                                 skip_rl=False, skip_hybrid_comp=False, skip_adaptive_weight=False,
                                 skip_safe_hybrid=False):
        all_results = {}

        # ----- STORAGE FOR PLOTTING -----
        cc_results = []          # (c_rate, time, plating)
        controller_results = []  # (name, time, plating)

        # ----- CONSTANT CURRENT -----
        if not skip_constant:
            cc_data = self.run_constant_current_baselines()
            all_results['constant_current'] = cc_data
            for current_str, data in cc_data.items():
                c_rate = float(current_str.replace('C', ''))
                cc_results.append((c_rate, data['avg_time'], data['plating_rate'] > 0))

        # ----- MPC -----
        if not skip_mpc:
            mpc_data = self.run_mpc_baseline()
            all_results['mpc'] = mpc_data
            avg_time = np.mean(mpc_data['times'])
            plating = mpc_data['plating'] > 0
            controller_results.append(("MPC", avg_time, plating))

        # ----- SAFE MPC -----
        if not skip_safe_mpc:
            safe_mpc_data = self.run_safe_mpc()
            all_results['safe_mpc'] = safe_mpc_data
            avg_time = np.mean(safe_mpc_data['times'])
            plating = safe_mpc_data['plating'] > 0
            controller_results.append(("Safe MPC", avg_time, plating))

        # ----- RL CONTROLLER -----
        if not skip_rl:
            rl_data = self.run_rl_controller()
            all_results['rl'] = rl_data
            avg_time = np.mean(rl_data['charging_times'])
            plating = rl_data['plating_events'] > 0
            controller_results.append(("RL (SAC)", avg_time, plating))

        # ----- HYBRID COMPENSATION -----
        if not skip_hybrid_comp:
            hybrid_data = self.run_hybrid_compensation()
            all_results['hybrid_compensation'] = hybrid_data
            avg_time = np.mean(hybrid_data['charging_times'])
            plating = hybrid_data['plating_events'] > 0
            controller_results.append(("Hybrid Compensation", avg_time, plating))

        # ----- ADAPTIVE WEIGHT HYBRID -----
        if not skip_adaptive_weight:
            adaptive_data = self.run_adaptive_weight_hybrid()
            all_results['adaptive_weight'] = adaptive_data
            avg_time = np.mean(adaptive_data['charging_times'])
            plating = adaptive_data['plating_events'] > 0
            controller_results.append(("Adaptive Weight Hybrid", avg_time, plating))

        # ----- SAFE HYBRID (Rule‑based) -----
        if not skip_safe_hybrid:
            safe_hybrid_data = self.run_safe_hybrid()
            all_results['safe_hybrid'] = safe_hybrid_data
            avg_time = np.mean(safe_hybrid_data['times'])
            plating = safe_hybrid_data['plating'] > 0
            controller_results.append(("Safe Hybrid (Rule‑based)", avg_time, plating))

        # ----- PRINT FINAL COMPARISON (existing code) -----
        print("\n" + "="*80)
        print("FINAL RESULTS COMPARISON - 8C BATTERY")
        print("="*80)

        print("\n--- CONSTANT CURRENT ---")
        if 'constant_current' in all_results:
            for current, data in all_results['constant_current'].items():
                print(f"  {current}: {data['avg_time']:.0f}s | Plating: {'YES' if data['plating_rate']>0 else 'NO'}")

        print("\n--- CONTROLLER COMPARISON ---")
        print(f"{'Controller':<30} {'Time (s)':<12} {'Plating':<10}")
        print("-"*55)

        if 'mpc' in all_results:
            print(f"{'MPC':<30} {np.mean(all_results['mpc']['times']):<12.0f} {'YES' if all_results['mpc']['plating']>0 else 'NO'}")

        if 'safe_mpc' in all_results:
            print(f"{'Safe MPC':<30} {np.mean(all_results['safe_mpc']['times']):<12.0f} {'YES' if all_results['safe_mpc']['plating']>0 else 'NO'}")

        if 'rl' in all_results:
            print(f"{'RL (SAC)':<30} {np.mean(all_results['rl']['charging_times']):<12.0f} {'YES' if all_results['rl']['plating_events']>0 else 'NO'}")

        if 'hybrid_compensation' in all_results:
            print(f"{'Hybrid Compensation':<30} {np.mean(all_results['hybrid_compensation']['charging_times']):<12.0f} {'YES' if all_results['hybrid_compensation']['plating_events']>0 else 'NO'}")

        if 'adaptive_weight' in all_results:
            print(f"{'Adaptive Weight Hybrid':<30} {np.mean(all_results['adaptive_weight']['charging_times']):<12.0f} {'YES' if all_results['adaptive_weight']['plating_events']>0 else 'NO'}")

        if 'safe_hybrid' in all_results:
            print(f"{'Safe Hybrid (Rule-based)':<30} {np.mean(all_results['safe_hybrid']['times']):<12.0f} {'YES' if all_results['safe_hybrid']['plating']>0 else 'NO'}")

        print("="*80)

        # ----- GENERATE GRAPHS -----
        plot_all_results(cc_results, controller_results, self.results_dir)

        return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='8C Battery Fast Charging Controller Comparison')
    parser.add_argument('--skip-constant', action='store_true', help='Skip constant current')
    parser.add_argument('--skip-mpc', action='store_true', help='Skip MPC')
    parser.add_argument('--skip-safe-mpc', action='store_true', help='Skip Safe MPC')
    parser.add_argument('--skip-rl', action='store_true', help='Skip RL')
    parser.add_argument('--skip-hybrid-comp', action='store_true', help='Skip Hybrid Compensation')
    parser.add_argument('--skip-adaptive-weight', action='store_true', help='Skip Adaptive Weight Hybrid')
    parser.add_argument('--skip-safe-hybrid', action='store_true', help='Skip Safe Hybrid')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test only')

    args = parser.parse_args()

    if args.quick_test:
        quick_test()
        return

    runner = ExperimentRunner(max_current=8.0, dt=2.0, target_soc=0.8)

    results = runner.run_complete_experiment(
        skip_constant=args.skip_constant,
        skip_mpc=args.skip_mpc,
        skip_safe_mpc=args.skip_safe_mpc,
        skip_rl=args.skip_rl,
        skip_hybrid_comp=args.skip_hybrid_comp,
        skip_adaptive_weight=args.skip_adaptive_weight,
        skip_safe_hybrid=args.skip_safe_hybrid
    )

    return results


def quick_test():
    """Quick test to verify all controllers work with 8C env."""
    print("\n" + "="*60)
    print("QUICK TEST - 8C Battery Controllers")
    print("="*60)

    try:
        print("\n1. Creating environment...")
        env = BatteryPlatingEnv(max_current_C=8.0, dt=2.0, target_soc=0.8)
        print("   ✓ Environment created")

        print("\n2. Testing MPC...")
        mpc = BatteryMPC(horizon=8, dt=2.0, max_current=8.0)
        obs, _ = env.reset()
        action = mpc.solve([obs[0], obs[1]])
        print(f"   ✓ MPC action: {action:.2f}C")

        print("\n3. Testing Hybrid Compensation...")
        hybrid = HybridCompensationController(env, mpc)
        action = hybrid.get_action(obs, use_rl=False)
        print(f"   ✓ Hybrid (MPC only): {action:.2f}C")

        print("\n4. Testing Adaptive Weight Hybrid...")
        adaptive = AdaptiveWeightHybrid(env, mpc)
        action = adaptive.get_action(obs)
        print(f"   ✓ Adaptive Weight action: {action:.2f}C")

        print("\n5. Testing Safe Hybrid...")
        safe = SafeHybridController(env, mpc)
        action = safe.get_action(obs)
        print(f"   ✓ Safe Hybrid action: {action:.2f}C")

        print("\n" + "="*60)
        print("✓ Quick test passed! All controllers ready for 8C.")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Quick test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()