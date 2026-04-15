"""
Model Predictive Control for battery fast charging with plating constraints
"""

import casadi as ca
import numpy as np


class BatteryMPC:
    """
    MPC controller that enforces safety constraints including anode potential
    to prevent lithium plating during fast charging.
    """
    
    def __init__(self, horizon=10, dt=1.0, max_current=3.0, capacity=3.0):
        """
        Initialize MPC controller.
        
        Args:
            horizon: Prediction horizon (number of steps)
            dt: Time step in seconds
            max_current: Maximum charging current in C-rate
            capacity: Battery capacity in Ah
        """
        self.N = horizon
        self.dt = dt
        self.max_current = max_current
        self.capacity = capacity
        
        # Battery parameters (simplified model for MPC)
        self.internal_resistance = 0.05  # Ohms
        self.thermal_mass = 100.0  # J/K
        self.cooling_coeff = 0.01  # W/K
        
        # Setup optimization problem
        self._setup_optimizer()
        
    def _setup_optimizer(self):
        """Setup the MPC optimization problem using CasADi"""
        self.opti = ca.Opti()
        
        # State variables: [SoC, Temperature]
        self.x = self.opti.variable(2, self.N + 1)
        # Control input: Charging current
        self.u = self.opti.variable(1, self.N)
        
        # Initial state
        self.x0 = self.opti.parameter(2, 1)
        self.opti.subject_to(self.x[:, 0] == self.x0)
        
        # Dynamics constraints
        for k in range(self.N):
            SoC = self.x[0, k]
            Temp = self.x[1, k]
            current = self.u[0, k]
            
            # SoC dynamics: d(SoC)/dt = I / (capacity * 3600)
            SoC_next = SoC + (self.dt / 3600) * current / self.capacity
            
            # Thermal dynamics
            heat_gen = (current ** 2) * self.internal_resistance
            Temp_next = Temp + self.dt * (heat_gen - self.cooling_coeff * (Temp - 298.15)) / self.thermal_mass
            
            self.opti.subject_to(self.x[0, k+1] == SoC_next)
            self.opti.subject_to(self.x[1, k+1] == Temp_next)
        
        # Constraints
        for k in range(self.N):
            # Current limits
            self.opti.subject_to(self.u[0, k] <= self.max_current)
            self.opti.subject_to(self.u[0, k] >= 0)
            
            # Temperature constraint
            self.opti.subject_to(self.x[1, k+1] <= 318.15)  # 45°C
            
            # Voltage constraint (simplified)
            voltage = self._estimate_voltage(self.x[0, k], self.u[0, k])
            self.opti.subject_to(voltage <= 4.2)
            
            # Anode potential constraint (key for plating avoidance)
            anode_potential = self._estimate_anode_potential(self.x[0, k], self.u[0, k], self.x[1, k])
            self.opti.subject_to(anode_potential >= 0.02)  # 20mV safety margin
        
        # Cost function: minimize charging time and penalize high currents
        cost = 0
        for k in range(self.N):
            cost += self.u[0, k] ** 2  # Penalize high currents
        cost += -10 * self.x[0, self.N]  # Reward reaching high SoC
        
        self.opti.minimize(cost)
        
        # Solver options
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.opti.solver('ipopt', opts)
    
    def _estimate_voltage(self, soc, current):
        """Simplified voltage estimation"""
        # Open circuit voltage as function of SoC (simplified)
        ocv = 3.0 + 1.5 * soc
        return ocv + current * self.internal_resistance
    
    def _estimate_anode_potential(self, soc, current, temperature):
        """
        Simplified anode potential estimation for MPC.
        This is a surrogate model - in research you'd train this from PyBaMM data.
        """
        # Anode potential decreases with higher current and higher SoC
        # Cold temperature makes it worse
        base_potential = 0.5 * (1 - soc)
        current_effect = -0.1 * current
        temp_effect = -0.01 * max(0, 298.15 - temperature)  # Cold penalty
        
        return max(0, base_potential + current_effect + temp_effect)
    
    def solve(self, current_state, rl_weights=None):
        """
        Solve MPC optimization.
        
        Args:
            current_state: [SoC, Temperature] at current time
            rl_weights: Optional [w1, w2] for RL-adapted cost
            
        Returns:
            Optimal charging current in C-rate
        """
        self.opti.set_value(self.x0, current_state)
        
        # If RL provides weights, adapt the cost function
        if rl_weights is not None:
            # Rebuild cost with RL weights
            cost = 0
            for k in range(self.N):
                cost += rl_weights[0] * self.u[0, k] ** 2
            cost += -rl_weights[1] * self.x[0, self.N]
            self.opti.minimize(cost)
        
        try:
            sol = self.opti.solve()
            optimal_current = float(sol.value(self.u[0, 0]))
            return optimal_current
        except:
            return 0.5  # Safe fallback current


# Simple wrapper for compatibility
class SimpleMPC:
    """Simpler MPC for faster execution during RL training"""
    
    def __init__(self, max_current=3.0):
        self.max_current = max_current
        self.current = 1.0
    
    def solve(self, state):
        """Simple rule-based fallback"""
        soc = state[0]
        temp = state[1]
        anode = state[2] if len(state) > 2 else 0.1
        
        # Reduce current if anode potential is low
        if anode < 0.05:
            self.current = max(0.3, self.current * 0.9)
        # Increase current if safe and not near full
        elif soc < 0.7 and temp < 313.15:
            self.current = min(self.max_current, self.current * 1.02)
        
        return self.current
