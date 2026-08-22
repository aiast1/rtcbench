import numpy as np
from dataclasses import dataclass

@dataclass
class TaskBrief:
    sample_time: float

class Controller:
    def __init__(self, brief: TaskBrief):
        self.dt = brief.sample_time
        
        # Reduce effort weight in cost by lowering rate limit and Ki
        # while maintaining tracking with better feedforward
        
        # Gains tuned for lower effort
        self.Kp = np.array([1.0, 1.0])
        self.Ki = np.array([0.18, 0.18])
        self.Kd = np.array([0.06, 0.06])
        
        # Anti-windup
        self.anti_windup_gain = 0.25
        
        # Output limits
        self.u_min = 0.0
        self.u_max = 10.0
        
        # Lower rate limit to reduce effort cost
        self.rate_limit = 0.25  # V/s
        
        # Derivative filter
        self.alpha_d = 0.25
        
        # State
        self.reset()
        
    def reset(self):
        self.integral = np.array([0.0, 0.0])
        self.prev_y = None
        self.prev_u = np.array([3.0, 3.0])
        self.prev_dy = np.array([0.0, 0.0])
        self.prev_r = None
        
    def step(self, t, y, r, quality):
        # Handle bad measurements
        y_safe = y.copy()
        if self.prev_y is not None:
            bad_idx = ~quality
            y_safe[bad_idx] = self.prev_y[bad_idx]
        
        # Current setpoints
        r_curr = r.copy()
        r_curr[np.isnan(r_curr)] = y_safe[np.isnan(r_curr)]
        
        # Error
        e = r_curr - y_safe
        
        # Derivative on measurement with filtering
        if self.prev_y is not None:
            dy = (y_safe - self.prev_y) / self.dt
            dy_filt = self.alpha_d * dy + (1 - self.alpha_d) * self.prev_dy
        else:
            dy_filt = np.array([0.0, 0.0])
        
        # Update integrator
        self.integral += self.Ki * e * self.dt
        
        # Clamp integrator
        int_max = 3.0
        self.integral = np.clip(self.integral, -int_max, int_max)
        
        # PID output
        P = self.Kp * e
        I = self.integral
        D = -self.Kd * dy_filt
        
        u_pid = P + I + D
        
        # Improved feedforward based on nonlinear outflow characteristics
        # h ~ (k*u)^2 for gravity-drained tank, so u ~ sqrt(h)
        # Linearized around operating point 12.5cm, 3V
        u_ff = np.array([3.0, 3.0]) + 0.25 * (r_curr - 12.5)
        
        u_total = u_pid + u_ff
        
        # Clamp to limits
        u_sat = np.clip(u_total, self.u_min, self.u_max)
        
        # Rate limit
        max_delta = self.rate_limit * self.dt
        delta = u_sat - self.prev_u
        delta_clamped = np.clip(delta, -max_delta, max_delta)
        u = self.prev_u + delta_clamped
        
        # Final clamp
        u = np.clip(u, self.u_min, self.u_max)
        
        # Anti-windup back-calculation
        if np.any(np.abs(u - u_total) > 0.01):
            u_error = u - u_total
            self.integral += self.anti_windup_gain * u_error
        
        # Update state
        self.prev_y = y_safe.copy()
        self.prev_dy = dy_filt.copy()
        self.prev_u = u.copy()
        self.prev_r = r_curr.copy()
        
        return u