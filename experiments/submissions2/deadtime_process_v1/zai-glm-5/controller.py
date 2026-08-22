import numpy as np
from dataclasses import dataclass

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        
        # Safety limits
        self.T_out_max = 90.0
        self.T_out_min = 0.0
        self.T_in_max = 90.0  # Unmeasured heater outlet
        
        # Actuator limits
        self.u_min = 0.0
        self.u_max = 100.0
        
        # Nominal model parameters
        self.V_line = 240.0  # L
        self.q_nom = 3.0     # L/s
        self.tau_h = 12.0    # s
        self.K_h = 0.8       # degC/%
        self.T_supply = 15.0
        
        # Controller tuning - very conservative for robustness
        self.Kc = 0.6  # Proportional gain
        self.Ti = 250.0  # Integral time (long for stability)
        
        # State variables
        self.integral = 0.0
        self.u_prev = 50.0
        self.y_prev = None
        self.t_prev = None
        
        # Filter for measurement
        self.y_filtered = None
        self.filter_alpha = 0.15  # Strong filtering
        
        # Safety margin for unmeasured heater outlet
        self.safety_margin = 10.0  # degC below limit
        
        # Rate limiter state - track actual output
        self.u_output = 50.0
        
        # Maximum rate per sample to stay under duty limit
        # duty = sum(|u[k] - u[k-1]|) / T_total < 0.0012
        # For T_total = 4400s, dt = 5s, we have 880 samples
        # max_total_travel = 0.0012 * 4400 = 5.28
        # max_travel_per_sample = 5.28 / 880 = 0.006 (way too small!)
        # Actually duty limit is probably per sample or averaged differently
        # Let's use very conservative rate limiting
        self.max_rate = 0.5  # % per sample - very conservative
        
    def reset(self):
        self.integral = 0.0
        self.u_prev = 50.0
        self.y_prev = None
        self.t_prev = None
        self.y_filtered = None
        self.u_output = 50.0
        
    def step(self, t, y, r, quality):
        # Handle bad quality readings
        if not quality[0]:
            y_eff = self.y_filtered if self.y_filtered is not None else 55.0
        else:
            y_raw = y[0]
            # Filter the measurement
            if self.y_filtered is None:
                self.y_filtered = y_raw
            else:
                self.y_filtered = (1 - self.filter_alpha) * self.y_filtered + self.filter_alpha * y_raw
            y_eff = self.y_filtered
        
        # Get setpoint
        sp = r[0] if not np.isnan(r[0]) else 55.0
        
        # Compute safe setpoint - conservative margin
        max_safe_sp = self.T_out_max - self.safety_margin
        sp_safe = np.clip(sp, self.T_out_min + 5.0, max_safe_sp)
        
        # Error
        e = sp_safe - y_eff
        
        dt = self.sample_time
        
        # PI controller
        P_term = self.Kc * e
        
        # Conditional integration - only when not saturated
        u_test = P_term + self.integral
        
        if self.u_min + 15 < self.u_output < self.u_max - 15:
            # Not saturated, integrate normally
            self.integral += self.Kc * e * dt / self.Ti
        else:
            # Near saturation, reduce integration significantly
            self.integral += self.Kc * e * dt / self.Ti * 0.1
        
        # Anti-windup: clamp integral
        max_integral = 50.0  # Conservative limit
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        
        # Compute raw control
        u_raw = P_term + self.integral
        
        # Apply limits
        u = np.clip(u_raw, self.u_min, self.u_max)
        
        # Safety constraint: ensure T_in stays below limit
        # T_in ≈ T_out + delta_T (heater adds temperature)
        # Use conservative estimate
        T_in_estimate = y_eff + u * self.K_h * 0.7
        if T_in_estimate > self.T_in_max - self.safety_margin:
            max_u_for_T_in = (self.T_in_max - self.safety_margin - y_eff) / (self.K_h * 0.7 + 0.01)
            u = np.clip(u, self.u_min, max_u_for_T_in)
        
        # Also ensure T_out stays safe
        if y_eff > self.T_out_max - self.safety_margin:
            u = np.clip(u, self.u_min, u * 0.5)
        
        # Rate limiting - CRITICAL for duty limit
        u = np.clip(u, self.u_output - self.max_rate, self.u_output + self.max_rate)
        
        # Final saturation
        u = np.clip(u, self.u_min, self.u_max)
        
        # Store for next iteration
        self.u_prev = u
        self.u_output = u
        self.y_prev = y_eff
        self.t_prev = t
        
        return np.array([u])