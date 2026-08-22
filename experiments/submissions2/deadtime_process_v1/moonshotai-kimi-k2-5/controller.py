import numpy as np
from dataclasses import dataclass

@dataclass
class TaskBrief:
    sample_time: float
    # other fields as needed

class Controller:
    def __init__(self, brief):
        self.dt = brief.sample_time  # 5.0 s
        
        # PID gains - detuned for smoothness to avoid actuator duty limit
        # The limit 0.0012 is fraction of range per second = 0.12%/s = 0.6% per 5s step
        # Very conservative to ensure we never exceed slew limit
        
        self.Kp = 0.4
        self.Ki = 0.008
        self.Kd = 1.0
        
        # Derivative filter time constant
        self.tau_f = 15.0
        
        # Output limits
        self.u_min = 0.0
        self.u_max = 100.0
        
        # Maximum allowed change per step: 0.0012 * 100% * 5s = 0.6%
        # Use 0.5% for margin
        self.max_du = 0.5  # % per step
        
        # State variables
        self.reset()
        
    def reset(self):
        # Initialize for bumpless transfer
        self.integral = 0.0
        self.y_prev = None
        self.y_filt = None
        self.u_prev = 50.0  # Start at nominal operating point
        
        # For setpoint tracking
        self.r_prev = None
        
        # Safety estimates
        self.T_supply_est = 15.0
        self.K_h_est = 0.8
        
    def step(self, t, y, r, quality):
        y_meas = y[0]
        r_sp = r[0]
        
        # Handle bad quality
        if not quality[0]:
            if self.y_prev is not None:
                y_meas = self.y_prev
            else:
                y_meas = r_sp
        
        # Initialize on first call
        if self.y_prev is None:
            self.y_prev = y_meas
            self.y_filt = y_meas
            self.r_prev = r_sp
            # Bumpless start: integral initialized so output starts at 50
            self.integral = 50.0
        
        # Filtered derivative
        alpha = self.dt / (self.tau_f + self.dt)
        y_filt_new = (1 - alpha) * self.y_filt + alpha * y_meas
        
        # Compute error
        error = r_sp - y_meas
        
        # Proportional term
        P = self.Kp * error
        
        # Derivative term (on filtered measurement)
        if self.y_prev is not None:
            d_y = (y_filt_new - self.y_filt) / self.dt
        else:
            d_y = 0.0
        
        D = -self.Kd * d_y
        
        # Integral term
        I = self.Ki * self.integral
        
        # Total output
        u_unsat = P + I + D
        
        # Saturate
        u_sat = np.clip(u_unsat, self.u_min, self.u_max)
        
        # Anti-windup: conditional integration
        if u_unsat <= self.u_max and u_unsat >= self.u_min:
            self.integral += error * self.dt
        elif u_unsat > self.u_max and error < 0:
            self.integral += error * self.dt
        elif u_unsat < self.u_min and error > 0:
            self.integral += error * self.dt
        
        # Safety: limit heater outlet temperature
        T_heater_est = self.T_supply_est + self.K_h_est * u_sat
        T_heater_limit = 90.0
        if T_heater_est > T_heater_limit - 2.0:
            u_safe = (T_heater_limit - 2.0 - self.T_supply_est) / self.K_h_est
            u_sat = min(u_sat, u_safe)
            # Anti-windup
            self.integral = min(self.integral, u_safe)
        
        u_sat = np.clip(u_sat, self.u_min, self.u_max)
        
        # CRITICAL: Rate limit to avoid actuator duty violation
        # 0.0012 fraction per second = 0.12% per second = 0.6% per 5s step
        du = u_sat - self.u_prev
        if abs(du) > self.max_du:
            u_sat = self.u_prev + np.sign(du) * self.max_du
        
        # Update state
        self.y_filt = y_filt_new
        self.y_prev = y_meas
        self.r_prev = r_sp
        self.u_prev = u_sat
        
        return np.array([u_sat])