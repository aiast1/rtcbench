import numpy as np
from dataclasses import dataclass

@dataclass
class TaskBrief:
    sample_time: float

class Controller:
    def __init__(self, brief):
        self.dt = brief.sample_time  # 3.0 s
        
        # Tighter control for better tracking while keeping duty safe
        # Need to stabilize unstable pole at 350K
        
        self.Kp = 4.0   # Increased for better stabilization
        self.Ki = 0.08  # Increased for better setpoint tracking
        self.Kd = 15.0  # Increased for damping
        
        # Derivative filter
        self.N = 8.0
        
        # Output limits
        self.u_min = 270.0
        self.u_max = 340.0
        
        # Slew rate limit (1.5 K/s) - use most of it but not all
        self.slew_limit = 1.5 * self.dt  # 4.5 K per step
        
        # Conservative internal slew to manage duty
        self.internal_slew = 3.5  # K per step
        
        # State variables
        self.reset()
        
    def reset(self):
        self.integral = 0.0
        self.last_y = 350.0
        self.last_u = 300.0
        self.derivative_filtered = 0.0
        self.last_r = 350.0
        self.last_error = 0.0
        
    def step(self, t, y, r, quality):
        # Extract measurements with fallback
        T_reactor = y[0] if quality[0] else self.last_y
        
        # Handle setpoint
        if np.isnan(r[0]):
            r_current = self.last_r
        else:
            r_current = r[0]
            self.last_r = r_current
        
        # Compute error
        error = r_current - T_reactor
        
        # Proportional term
        P = self.Kp * error
        
        # Derivative on measurement (filtered) - helps stabilize
        dT = T_reactor - self.last_y
        derivative_raw = -self.Kd * dT / self.dt
        
        # Exponential filter on derivative
        tau_f = self.Kd / (self.N * self.Kp + 1e-10) if self.Kp > 0 else 0.0
        if tau_f > 0:
            alpha_f = self.dt / (tau_f + self.dt)
            self.derivative_filtered = (1 - alpha_f) * self.derivative_filtered + alpha_f * derivative_raw
        else:
            self.derivative_filtered = 0.85 * self.derivative_filtered + 0.15 * derivative_raw
        
        D = self.derivative_filtered
        
        # Compute preliminary output
        u_prelim = P + self.integral + D
        
        # Anti-windup
        u_saturated = np.clip(u_prelim, self.u_min, self.u_max)
        integral_needed = u_saturated - P - D
        self.integral = np.clip(integral_needed, -150, 150)
        
        # PID output
        u_pid = P + self.integral + D
        u_limited = np.clip(u_pid, self.u_min, self.u_max)
        
        # Apply slew rate limit
        delta_u = u_limited - self.last_u
        delta_u_clamped = np.clip(delta_u, -self.internal_slew, self.internal_slew)
        u_slewed = self.last_u + delta_u_clamped
        
        # Final clamp
        u_final = np.clip(u_slewed, self.u_min, self.u_max)
        
        # Conditional integration
        at_upper = (u_final >= self.u_max - 1.0) and error > 0
        at_lower = (u_final <= self.u_min + 1.0) and error < 0
        
        if not (at_upper or at_lower):
            self.integral += self.Ki * error * self.dt
            self.integral = np.clip(self.integral, -150, 150)
        
        # Update state
        self.last_y = T_reactor
        self.last_error = error
        self.last_u = u_final
        
        return np.array([u_final])