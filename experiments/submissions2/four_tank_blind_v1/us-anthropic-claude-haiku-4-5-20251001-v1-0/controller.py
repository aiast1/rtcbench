import numpy as np
from dataclasses import dataclass

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = brief.sample_time
        
        # PID parameters - further tuned for better tracking with minimal effort
        # Increase proportional for faster response, reduce derivative further
        self.kp = np.array([1.5, 1.5])
        self.ki = np.array([0.35, 0.35])
        self.kd = np.array([0.08, 0.08])
        
        # Integral state and anti-windup
        self.integral = np.array([0.0, 0.0])
        self.integral_max = np.array([10.0, 10.0])
        
        # Previous error for derivative
        self.prev_error = np.array([0.0, 0.0])
        
        # Actuator limits
        self.u_min = 0.0
        self.u_max = 10.0
        
        # Previous actuator output for duty cycle tracking
        self.prev_u = np.array([3.0, 3.0])
        self.actuator_travel = 0.0
        
        # Low-pass filter for derivative (reduce noise sensitivity)
        self.deriv_filter_alpha = 0.15
        self.filtered_deriv = np.array([0.0, 0.0])
        
        # Track measurement quality
        self.quality_history = np.array([True, True])
        
        # Saturation tracking for anti-windup
        self.saturated = np.array([False, False])
        
    def reset(self):
        self.integral = np.array([0.0, 0.0])
        self.prev_error = np.array([0.0, 0.0])
        self.prev_u = np.array([3.0, 3.0])
        self.actuator_travel = 0.0
        self.filtered_deriv = np.array([0.0, 0.0])
        self.quality_history = np.array([True, True])
        self.saturated = np.array([False, False])
    
    def step(self, t, y, r, quality):
        # Handle measurement quality - use previous error if measurement is bad
        valid = quality & np.isfinite(r)
        
        # Compute errors
        error = np.where(valid, r - y, self.prev_error)
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with conditional anti-windup
        # Only integrate when measurement is good and not saturated
        for i in range(2):
            if valid[i] and not self.saturated[i]:
                self.integral[i] += error[i] * self.dt
                # Anti-windup: clamp integral
                self.integral[i] = np.clip(self.integral[i], -self.integral_max[i], self.integral_max[i])
            elif valid[i] and self.saturated[i]:
                # Back off integral if saturated
                self.integral[i] *= 0.95
        
        i_term = self.ki * self.integral
        
        # Derivative term with aggressive low-pass filtering
        # Only compute derivative on good measurements to reduce noise
        d_error = np.zeros(2)
        for i in range(2):
            if valid[i] and self.quality_history[i]:
                d_error[i] = (error[i] - self.prev_error[i]) / self.dt
            else:
                d_error[i] = 0.0
        
        # Aggressive low-pass filter derivative to reduce noise impact and effort
        self.filtered_deriv = self.deriv_filter_alpha * d_error + (1.0 - self.deriv_filter_alpha) * self.filtered_deriv
        d_term = self.kd * self.filtered_deriv
        
        # Compute raw control output
        u_raw = p_term + i_term + d_term
        
        # Saturate
        u = np.clip(u_raw, self.u_min, self.u_max)
        
        # Track saturation for next iteration
        self.saturated = (u != u_raw)
        
        # Store state for next iteration
        self.prev_error = error.copy()
        self.prev_u = u.copy()
        self.quality_history = valid.copy()
        
        return u