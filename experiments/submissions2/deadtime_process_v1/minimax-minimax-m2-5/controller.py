import numpy as np
from dataclasses import dataclass

@dataclass
class TaskBrief:
    sample_time: float
    measurement_names: list
    actuator_names: list
    scored: list
    setpoint_schedule: list
    safety_limits: dict

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.Ts = brief.sample_time
        
        # Increased gains for better tracking with large deadtime
        # The plant has deadtime 40-200s, so we need more aggressive control
        self.Kp = 0.5
        self.Ki = 0.008
        
        # Anti-windup
        self.u_min = 0.0
        self.u_max = 100.0
        self.k_aw = 0.2
        
        # Rate limiting - keep well under duty limit (0.0012 = 0.12%/s at 5s sample = 0.6%/sample)
        # Use 0.5%/sample = 0.1%/s to be safe
        self.max_delta_u = 0.5
        
        # State
        self.integral = 0.0
        self.last_u = 50.0
        self.prev_sp = 55.0
        
    def reset(self):
        self.integral = 0.0
        self.last_u = 50.0
        self.prev_sp = 55.0
        
    def step(self, t, y, r, quality):
        # Extract measured temperature
        y_meas = y[0]
        
        # Handle bad quality measurement
        if not quality[0]:
            return np.array([self.last_u])
        
        # Setpoint
        sp = r[0] if not np.isnan(r[0]) else self.prev_sp
        self.prev_sp = sp
        
        # Error
        error = sp - y_meas
        
        # PI control
        u_p = self.Kp * error
        u_pre = u_p + self.integral
        
        # Anti-windup
        if u_pre > self.u_max:
            self.integral += self.Ki * error - self.k_aw * (u_pre - self.u_max)
            u_out = self.u_max
        elif u_pre < self.u_min:
            self.integral += self.Ki * error - self.k_aw * (u_pre - self.u_min)
            u_out = self.u_min
        else:
            self.integral += self.Ki * error
            u_out = u_pre
        
        # Rate limiting
        delta_u = u_out - self.last_u
        if abs(delta_u) > self.max_delta_u:
            u_out = self.last_u + np.sign(delta_u) * self.max_delta_u
        
        # Hard limits
        u_out = np.clip(u_out, self.u_min, self.u_max)
        
        # Ensure finite
        if not np.isfinite(u_out):
            u_out = self.last_u
        
        self.last_u = u_out
        
        return np.array([u_out])