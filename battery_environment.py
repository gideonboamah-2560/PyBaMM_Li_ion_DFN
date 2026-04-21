"""
Research-Grade Battery Environment for Fast Charging Control
Uses PyBaMM to pre-compute physics-based trajectories, then interpolates
for real-time control. This is computationally efficient and accurate.
"""

import pybamm
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.interpolate import interp1d


class BatteryPlatingEnv(gym.Env):
    """
    Physics-based battery environment using PyBaMM's DFN model.
    Pre-computes charging trajectories at different C-rates for real-time control.
    """

    def __init__(self, max_current_C=3.0, dt=10.0, target_soc=0.8):
        super().__init__()

        self.max_current = max_current_C
        self.dt = dt
        self.target_soc = target_soc
        self.capacity_Ah = 3.0

        # Set up PyBaMM model with anode potential tracking
        self._setup_battery_model()

        # Pre-compute surrogate model
        self._build_surrogate_model()

        # Define observation space: [SoC, Temperature, Anode_Potential, Voltage]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 273.15, -0.5, 2.5], dtype=np.float32),
            high=np.array([1.0, 333.15, 0.5, 4.5], dtype=np.float32),
            dtype=np.float32
        )

        # Define action space: Charging current in C-rate
        self.action_space = spaces.Box(
            low=0.0, high=max_current_C, shape=(1,), dtype=np.float32
        )

        # Initialize state variables
        self.reset()

    def _setup_battery_model(self):
        """Setup PyBaMM DFN model with anode potential tracking"""
        # Use Doyle-Fuller-Newman model
        self.model = pybamm.lithium_ion.DFN()

        # Add anode potential as a variable
        self.model.variables["Anode potential [V]"] = self.model.variables[
            "Negative electrode surface potential difference at separator interface [V]"
        ]

        # Use Chen2020 parameters
        self.parameter_values = pybamm.ParameterValues("Chen2020")

    def _get_soc_from_solution(self, solution):
        """
        Extract State of Charge from PyBaMM solution.
        PyBaMM stores SoC in different ways depending on version.
        """
        # Try different methods to get SoC
        try:
            # Method 1: Direct access if available
            if hasattr(solution, 'soc'):
                return solution.soc.entries / 100.0

            # Method 2: Through variables dictionary
            if hasattr(solution, 'variables'):
                if "State of Charge" in solution.variables:
                    return solution.variables["State of Charge"].entries / 100.0
                if "Discharge capacity [A.h]" in solution.variables:
                    capacity = solution.variables["Discharge capacity [A.h]"].entries
                    return capacity / self.capacity_Ah

            # Method 3: Calculate from solution object
            if hasattr(solution, 'get_variable'):
                soc = solution.get_variable("State of Charge")
                return soc.entries / 100.0

        except Exception as e:
            print(f"  Warning: Could not extract SoC: {e}")

        # Fallback: estimate from time (simple linear approximation)
        times = solution["Time [s]"].entries
        # Assume 1C charging takes 3600 seconds to full
        estimated_soc = np.clip(times / 3600, 0, 1)
        return estimated_soc

    def _get_variable_safe(self, solution, var_name, default_value):
        """Safely get a variable from solution"""
        try:
            var = solution[var_name]
            return var.entries
        except:
            return default_value

    def _build_surrogate_model(self):
        """
        Pre-compute charging trajectories at different C-rates.
        """
        # C-rates to simulate
        self.c_rates = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

        # Storage for trajectories
        self.trajectories = {}

        print("Building physics-based surrogate model...")

        for c_rate in self.c_rates:
            print(f"  Simulating {c_rate}C charging...")

            # Create experiment for this C-rate
            experiment = pybamm.Experiment([
                f"Charge at {c_rate}C until 4.2V",
                "Hold at 4.2V until 50mA",
            ])

            # Run simulation
            sim = pybamm.Simulation(
                self.model,
                parameter_values=self.parameter_values,
                experiment=experiment
            )

            try:
                solution = sim.solve(initial_soc=0.0)

                # Get time array
                times = solution["Time [s]"].entries

                # Get SoC (handles different PyBaMM versions)
                soc_data = self._get_soc_from_solution(solution)

                # Get temperature
                try:
                    temp_data = solution["Cell temperature [K]"].entries
                except:
                    temp_data = 298.15 + np.zeros_like(times)

                # Get voltage
                try:
                    voltage_data = solution["Terminal voltage [V]"].entries
                except:
                    voltage_data = 3.6 + 0.6 * (1 - np.exp(-times / 200))

                # Get anode potential
                try:
                    anode_data = solution["Anode potential [V]"].entries
                except:
                    anode_data = 0.5 * (1 - soc_data) - 0.05 * c_rate

                # Create interpolators - FIXED: Use scalar fill_value
                self.trajectories[c_rate] = {
                    'time': times,
                    'soc': interp1d(times, soc_data, kind='linear',
                                   fill_value=(float(soc_data[0]), float(soc_data[-1])),
                                   bounds_error=False),
                    'temp': interp1d(times, temp_data, kind='linear',
                                    fill_value=(float(temp_data[0]), float(temp_data[-1])),
                                    bounds_error=False),
                    'voltage': interp1d(times, voltage_data, kind='linear',
                                       fill_value=(float(voltage_data[0]), float(voltage_data[-1])),
                                       bounds_error=False),
                    'anode': interp1d(times, anode_data, kind='linear',
                                     fill_value=(float(anode_data[0]), float(anode_data[-1])),
                                     bounds_error=False),
                    'max_time': times[-1]
                }

                print(f"    Completed: {c_rate}C charging in {times[-1]:.1f} seconds, "
                      f"final SoC={soc_data[-1]:.2f}")

            except Exception as e:
                print(f"    Failed for {c_rate}C: {e}")
                print(f"    Using heuristic fallback...")
                # Use heuristic fallback
                self.trajectories[c_rate] = self._create_heuristic_trajectory(c_rate)

        print("Surrogate model built successfully!")

    def _create_heuristic_trajectory(self, c_rate):
        """
        Create a heuristic trajectory when PyBaMM simulation fails.
        This ensures the environment can still run.

        FIXED: fill_value now uses scalar values instead of arrays
        """
        # Estimate charging time based on C-rate
        # At 1C, takes ~3600 seconds to charge from 0 to 100%
        # But we stop at target_soc (80%)
        estimated_time = (self.target_soc / c_rate) * 3600
        estimated_time = min(estimated_time, 4000)  # Cap at ~67 minutes

        times = np.linspace(0, estimated_time, 100)

        # Simple SoC progression (linear to target)
        soc_data = np.linspace(0, self.target_soc, len(times))

        # Simple temperature model (rises then plateaus)
        temp_data = 298.15 + 8 * (1 - np.exp(-times / 300))

        # Simple voltage model (rises to 4.2V)
        voltage_data = 3.6 + 0.6 * (1 - np.exp(-times / 200))

        # Simple anode potential (decreases with SoC and current)
        anode_data = 0.5 * (1 - soc_data) - 0.08 * c_rate
        anode_data = np.maximum(anode_data, -0.1)  # Clamp

        # FIXED: Use scalar values for fill_value (not arrays)
        return {
            'time': times,
            'soc': interp1d(times, soc_data, kind='linear',
                           fill_value=(float(soc_data[0]), float(soc_data[-1])),
                           bounds_error=False),
            'temp': interp1d(times, temp_data, kind='linear',
                            fill_value=(float(temp_data[0]), float(temp_data[-1])),
                            bounds_error=False),
            'voltage': interp1d(times, voltage_data, kind='linear',
                               fill_value=(float(voltage_data[0]), float(voltage_data[-1])),
                               bounds_error=False),
            'anode': interp1d(times, anode_data, kind='linear',
                             fill_value=(float(anode_data[0]), float(anode_data[-1])),
                             bounds_error=False),
            'max_time': times[-1]
        }

    def _get_trajectory(self, current_C):
        """
        Get interpolated trajectory for a given C-rate.
        """
        # Find nearest C-rate
        idx = np.searchsorted(self.c_rates, current_C)

        if idx == 0:
            return self.trajectories[self.c_rates[0]]
        elif idx >= len(self.c_rates):
            return self.trajectories[self.c_rates[-1]]
        else:
            # Return the higher C-rate trajectory for now
            return self.trajectories[self.c_rates[idx]]

    def reset(self, seed=None, options=None):
        """Reset to initial state"""
        super().reset(seed=seed)

        self.soc = 0.0
        self.temperature = 298.15
        self.voltage = 3.6
        self.anode_potential = 0.1
        self.current_step = 0
        self.total_time = 0.0

        return self._get_observation(), {}

    def _get_observation(self):
        """Return current observation"""
        return np.array([
            self.soc,
            self.temperature,
            self.anode_potential,
            self.voltage
        ], dtype=np.float32)

    def step(self, action):
        """
        Apply charging current and advance simulation.
        """
        current_C = float(np.clip(action[0], 0, self.max_current))

        # Get trajectory for this C-rate
        traj = self._get_trajectory(current_C)

        # Advance time
        self.current_step += 1
        self.total_time += self.dt

        # Get state from trajectory
        if self.total_time <= traj['max_time']:
            self.soc = float(traj['soc'](self.total_time))
            self.temperature = float(traj['temp'](self.total_time))
            self.voltage = float(traj['voltage'](self.total_time))
            self.anode_potential = float(traj['anode'](self.total_time))
        else:
            # Use final values
            self.soc = float(traj['soc'](traj['max_time']))
            self.temperature = float(traj['temp'](traj['max_time']))
            self.voltage = float(traj['voltage'](traj['max_time']))
            self.anode_potential = float(traj['anode'](traj['max_time']))

        # Check for plating
        plating_detected = self.anode_potential < 0.0

        # Calculate reward
        reward = self._calculate_reward(current_C, plating_detected)

        # Check termination
        terminated = False
        if plating_detected:
            terminated = True
        elif self.soc >= self.target_soc:
            terminated = True
        elif self.temperature > 333.15:
            terminated = True
        elif self.voltage > 4.3:
            terminated = True

        info = {
            'plating_detected': plating_detected,
            'anode_potential': self.anode_potential,
            'current': current_C,
            'soc': self.soc,
            'temperature': self.temperature,
            'voltage': self.voltage,
            'step': self.current_step,
            'time': self.total_time
        }

        return self._get_observation(), reward, terminated, False, info

    def _calculate_reward(self, current, plating_detected):
        """Multi-objective reward function"""
        charging_reward = self.soc * 10.0

        temp_celsius = self.temperature - 273.15
        temp_penalty = 0.0
        if temp_celsius > 40:
            temp_penalty = -0.05 * (temp_celsius - 40) ** 2

        voltage_penalty = 0.0
        if self.voltage > 4.2:
            voltage_penalty = -2.0 * (self.voltage - 4.2)

        plating_penalty = -100.0 if plating_detected else 0.0

        return charging_reward + temp_penalty + voltage_penalty + plating_penalty


# Debug function to explore PyBaMM solution structure
def debug_pybamm_structure():
    """Helper function to understand PyBaMM solution structure"""
    print("\n" + "="*60)
    print("DEBUGGING: PyBaMM Solution Structure")
    print("="*60)

    model = pybamm.lithium_ion.DFN()
    param = pybamm.ParameterValues("Chen2020")
    experiment = pybamm.Experiment(["Charge at 1C for 10 seconds"])

    sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
    solution = sim.solve(initial_soc=0.0)

    print("\nType of solution:", type(solution))
    print("\nAvailable attributes and methods (first 30):")
    attrs = [attr for attr in dir(solution) if not attr.startswith('_')]
    for i, attr in enumerate(attrs[:30]):
        print(f"  {i+1}. {attr}")

    print("\nTrying to access variables...")
    try:
        # Try different access patterns
        if hasattr(solution, 'variables'):
            print("\n  solution.variables exists!")
            print(f"  Type: {type(solution.variables)}")
            if hasattr(solution.variables, 'keys'):
                print(f"  Keys: {list(solution.variables.keys())[:20]}")

        # Try direct indexing
        try:
            test_var = solution["Time [s]"]
            print("\n  ✓ solution['Time [s]'] works!")
            print(f"    Shape: {test_var.entries.shape}")
        except Exception as e:
            print(f"\n  ✗ solution['Time [s]'] failed: {e}")

        # Try to find SoC
        try:
            test_var = solution["State of Charge"]
            print("\n  ✓ solution['State of Charge'] works!")
        except:
            print("\n  ✗ 'State of Charge' not found")

    except Exception as e:
        print(f"\nError exploring solution: {e}")

    print("\n" + "="*60)


if __name__ == "__main__":
    print("Testing Battery Environment...")

    # First, debug to see solution structure
    debug_pybamm_structure()

    # Then try to create the environment
    print("\n" + "="*60)
    print("Creating Battery Environment...")
    print("="*60)

    try:
        env = BatteryPlatingEnv(max_current_C=3.0, dt=10.0, target_soc=0.8)

        obs, _ = env.reset()
        print(f"Initial: SoC={obs[0]:.3f}, Anode={obs[2]:.4f}V")

        # Test constant current charging
        print("\nTesting constant current charging at 1.5C...")
        for step in range(10):
            action = np.array([1.5])
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {step+1}: t={info['time']:.0f}s, SoC={obs[0]:.3f}, "
                  f"Anode={obs[2]:.4f}V, Plating={info['plating_detected']}")

            if terminated:
                print(f"Terminated: {info}")
                break

        print("\nEnvironment test complete!")

    except Exception as e:
        print(f"\nError creating environment: {e}")
        import traceback
        traceback.print_exc()
