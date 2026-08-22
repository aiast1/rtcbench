import numpy as np
from dataclasses import dataclass

@dataclass
class TaskBrief:
    sample_time: float

class Controller:
    def __init__(self, brief):
        self.Ts = brief.sample_time  # 2.0 seconds
        
        # More aggressive tuning for better tracking
        # Increased Kp and Ki for faster response, reduced Kd for less noise sensitivity
        self.Kp = np.array([1.2, 1.2])
        self.Ki = np.array([0.25, 0.25])
        self.Kd = np.array([0.3, 0.3])
        
        # Derivative filter - more smoothing
        self.gamma = 0.3  # heavier filtering
        
        # Output limits
        self.u_min = np.array([0.0, 0.0])
        self.u_max = np.array([10.0, 10.0])
        
        # Anti-windup
        self.anti_windup_gain = 1.0
        
        # Setpoint rate limit (cm/s)
        self.max_sp_rate = 0.3  # gentler setpoint changes
        
        # State
        self.reset()
        
    def reset(self):
        self.u_prev = np.array([3.0, 3.0])
        self.integral = np.array([0.0, 0.0])
        self.y_prev = np.array([0.0, 0.0])
        self.y_filt_prev = np.array([0.0, 0.0])
        self.r_prev = np.array([12.26, 12.78])
        self.first_step = True
        
    def step(self, t, y, r, quality):
        # Handle bad measurements
        y_use = y.copy()
        if not self.first_step:
            for i in range(len(y)):
                if not quality[i]:
                    y_use[i] = self.y_prev[i]
        
        # Initialize
        if self.first_step:
            self.y_prev = y_use.copy()
            self.y_filt_prev = y_use.copy()
            e_init = self.r_prev - y_use
            self.integral = (self.u_prev - self.Kp * e_init) / self.Ki
            self.integral = np.clip(self.integral, -20.0, 20.0)
            self.first_step = False
        
        # Smooth setpoint
        r_curr = np.where(np.isnan(r), self.r_prev, r)
        r_rate = (r_curr - self.r_prev) / self.Ts
        r_rate_lim = np.clip(r_rate, -self.max_sp_rate, self.max_sp_rate)
        r_smooth = self.r_prev + r_rate_lim * self.Ts
        
        # Error
        e = r_smooth - y_use
        
        # Filtered derivative
        y_filt = self.gamma * y_use + (1 - self.gamma) * self.y_filt_prev
        dy = (y_filt - self.y_filt_prev) / self.Ts
        
        # PID
        P = self.Kp * e
        I = self.Ki * self.integral
        D = -self.Kd * dy
        
        u_unsat = P + I + D
        
        # Output limiting with anti-windup
        u_sat = np.clip(u_unsat, self.u_min, self.u_max)
        
        # Back-calculation anti-windup
        for i in range(2):
            if u_unsat[i] != u_sat[i]:
                # Saturated - adjust integral
                self.integral[i] += self.anti_windup_gain * (u_sat[i] - u_unsat[i]) / self.Ki[i]
            else:
                self.integral[i] += e[i] * self.Ts
        
        # Clamp integral
        self.integral = np.clip(self.integral, -30.0, 30.0)
        
        # Update states
        self.y_prev = y_use.copy()
        self.y_filt_prev = y_filt.copy()
        self.r_prev = r_smooth.copy()
        self.u_prev = u_sat.copy()
        
        return u_sat