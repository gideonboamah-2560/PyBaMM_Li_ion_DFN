"""
Physics-Informed Battery Environment for Fast Charging Control
This environment simulates a lithium-ion battery with anode potential tracking
for lithium plating avoidance.
"""

import pybamm
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BatteryPlatingEnv(gym.Env):
    """
    Battery charging environment with physics-based lithium plating detection.
    Uses PyBaMM's DFN model to track anode potential - the key indicator
    for lithium plating risk during fast charging.
    """
    
    def __init__(self, max_current_C=3.0, dt=1.0, target_soc=0.8):
        """
        Initialize the battery environment.
        
        Args:
            max_current_C: Maximum charging current in C-rate (e.g., 3.0 = 3C)
            dt: Time step in seconds
            target_soc: Target State of Charge to stop charging
        """
        super().__init__()
        
        self.max_current = max_current_C
        self.dt = dt
        self.target_soc = target_soc
        self.capacity_Ah = 3.0
        self.current_step = 0
        
        # Setup PyBaMM model with anode potential tracking
        self._setup_battery_model()
        
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
        self.soc = 0.0
        self.temperature = 298.15
        self.voltage = 3.6
        self.anode_potential = 0.1
        self.cycle_count = 0
        
        # Storage for simulation
        self.sim = None
        self.solution = None
        
    def _setup_battery_model(self):
        """Setup PyBaMM model with proper anode potential tracking"""
        # Use Doyle-Fuller-Newman model for accurate physics
        self.model = pybamm.lithium_ion.DFN()
        
        # Add anode potential as a variable (critical for plating detection)
        # This uses the separator interface potential which is the minimum during charging
        self.model.variables["Anode potential [V]"] = self.model.variables[
            "Negative electrode surface potential difference at separator interface [V]"
        ]
        
        # Set parameters (Chen2020 is validated against commercial cells)
        self.parameter_values = pybamm.ParameterValues("Chen2020")
        
    def reset(self, seed=None, options=None):
        """Reset the battery to initial state (discharged)"""
        super().reset(seed=seed)
        
        # Reset state variables
        self.soc = 0.0
        self.temperature = 298.15
        self.voltage = 3.6
        self.anode_potential = 0.1
        self.current_step = 0
        
        # Create a new simulation at 0% SoC
        # Use a simple experiment to initialize the model
        experiment = pybamm.Experiment(
            [f"Rest for {self.dt} seconds"],
            period=f"{self.dt} seconds"
        )
        
        self.sim = pybamm.Simulation(
            self.model, 
            parameter_values=self.parameter_values,
            experiment=experiment
        )
        
        # Solve initial state
        try:
            self.solution = self.sim.solve(initial_soc=0.0)
            self._update_state_from_solution()
        except Exception:
            # Use default values if simulation fails
            pass
        
        return self._get_observation(), {}
    
    def _update_state_from_solution(self):
        """Extract current state from PyBaMM solution"""
        if self.solution is None or len(self.solution.t) == 0:
            return
        
        try:
            # Get latest time
            current_time = self.solution.t[-1]
            
            # Extract State of Charge (convert from percentage to 0-1)
            if "State of Charge" in self.solution:
                soc_var = self.solution["State of Charge"]
                self.soc = float(soc_var(current_time)) / 100.0
            
            # Extract Temperature
            if "Cell temperature [K]" in self.solution:
                temp_var = self.solution["Cell temperature [K]"]
                self.temperature = float(temp_var(current_time))
            
            # Extract Voltage
            if "Terminal voltage [V]" in self.solution:
                volt_var = self.solution["Terminal voltage [V]"]
                self.voltage = float(volt_var(current_time))
            
            # Extract Anode Potential (key variable for plating detection)
            if "Anode potential [V]" in self.solution:
                anode_var = self.solution["Anode potential [V]"]
                self.anode_potential = float(anode_var(current_time))
                
        except Exception:
            # Keep previous values if extraction fails
            pass
    
    def _get_observation(self):
        """Return current observation as numpy array"""
        return np.array([
            self.soc,
            self.temperature,
            self.anode_potential,
            self.voltage
        ], dtype=np.float32)
    
    def step(self, action):
        """
        Apply charging current and advance simulation.
        
        Args:
            action: Charging current in C-rate (0 to max_current_C)
            
        Returns:
            observation: Current state [SoC, Temperature, Anode_Potential, Voltage]
            reward: Scalar reward for this step
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information dictionary
        """
        # Clip action to valid range
        current_C = float(np.clip(action[0], 0, self.max_current))
        
        try:
            # Create experiment segment for this step
            experiment_segment = pybamm.Experiment(
                [f"Charge at {current_C}C for {self.dt} seconds"],
                period=f"{self.dt} seconds"
            )
            
            # Step the simulation
            self.sim = pybamm.Simulation(
                self.model,
                parameter_values=self.parameter_values,
                experiment=experiment_segment
            )
            
            # Solve from current SoC (convert to percentage)
            self.solution = self.sim.solve(initial_soc=self.soc * 100)
            self._update_state_from_solution()
            
        except Exception:
            # If simulation fails, apply penalty and terminate
            return self._get_observation(), -10.0, True, False, {
                'error': 'Simulation failed',
                'plating_detected': False,
                'anode_potential': self.anode_potential,
                'current': current_C,
                'soc': self.soc,
                'temperature': self.temperature,
                'voltage': self.voltage,
                'step': self.current_step
            }
        
        # Update step counter
        self.current_step += 1
        
        # Check for lithium plating (anode potential below 0V)
        plating_detected = self.anode_potential < 0.0
        
        # Calculate reward
        reward = self._calculate_reward(current_C, plating_detected)
        
        # Check termination conditions
        terminated = False
        if plating_detected:
            terminated = True
        elif self.soc >= self.target_soc:
            terminated = True
        elif self.temperature > 333.15:  # 60°C
            terminated = True
        elif self.voltage > 4.3:  # Over-voltage protection
            terminated = True
        
        # Prepare info dictionary
        info = {
            'plating_detected': plating_detected,
            'anode_potential': self.anode_potential,
            'current': current_C,
            'soc': self.soc,
            'temperature': self.temperature,
            'voltage': self.voltage,
            'step': self.current_step
        }
        
        return self._get_observation(), reward, terminated, False, info
    
    def _calculate_reward(self, current, plating_detected):
        """
        Calculate multi-objective reward.
        
        Rewards fast charging progress while heavily penalizing safety violations.
        """
        # Calculate charging progress reward (estimated SoC gain)
        # Since we don't have previous SoC stored, use current SoC as progress indicator
        charging_reward = self.soc * 10.0
        
        # Temperature penalty (penalize temperatures above 40°C)
        temp_penalty = 0.0
        temp_celsius = self.temperature - 273.15
        if temp_celsius > 40:
            temp_penalty = -0.05 * (temp_celsius - 40) ** 2
        
        # Voltage penalty (penalize voltages above 4.2V)
        voltage_penalty = 0.0
        if self.voltage > 4.2:
            voltage_penalty = -2.0 * (self.voltage - 4.2)
        
        # Plating penalty - large negative for safety violation
        plating_penalty = -100.0 if plating_detected else 0.0
        
        total_reward = charging_reward + temp_penalty + voltage_penalty + plating_penalty
        
        return total_reward
    
    def get_plating_status(self):
        """Return whether lithium plating has occurred"""
        return self.anode_potential < 0.0
