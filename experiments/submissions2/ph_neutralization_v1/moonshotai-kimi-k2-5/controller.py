import numpy as np
from dataclasses import dataclass

@dataclass
class TaskBrief:
    sample_time: float

class Controller:
    def __init__(self, brief):
        self.Ts = brief.sample_time  # 5.0 s
        
        # PID tuning - much gentler to avoid duty limit
        self.Kp = 2.0
        self.Ki = 0.03
        self.Kd = 5.0
        
        # Derivative filter
        self.Tf = 15.0
        
        # Output limits
        self.u_min = 0.0
        self.u_max = 30.0
        
        # Level safety margins
        self.h_low_safe = 8.0
        self.h_high_safe = 27.0
        self.h_crit_low = 6.0
        self.h_crit_high = 29.0
        
        # pH safety margins
        self.ph_low_safe = 4.5
        self.ph_high_safe = 10.0
        self.ph_crit_low = 4.2
        self.ph_crit_high = 10.2
        
        # Duty limit: 0.0041 - need to minimize total variation
        self.max_step = 0.5  # Very conservative step limit
        
        # State
        self.reset()
        
    def reset(self):
        self.integral = 14.22
        self.y_prev = None
        self.y_filt = None
        self.u_prev = 14.22
        self.t_prev = None
        self.r_filt = None
        self.y_prev_filt = None
        
    def step(self, t, y, r, quality):
        # Extract measurements
        ph_meas = y[0] if quality[0] else (self.y_prev[0] if self.y_prev is not None else 6.5)
        level = y[1] if quality[1] else (self.y_prev[1] if self.y_prev is not None else 17.5)
        
        # Initialize on first call
        if self.y_prev is None:
            self.y_prev = np.array([ph_meas, level])
            self.y_filt = ph_meas
            self.r_filt = r[0] if not np.isnan(r[0]) else 6.5
            self.y_prev_filt = ph_meas
            return np.array([14.22])
        
        # Use previous good values if current is bad
        if not quality[0]:
            ph_meas = self.y_prev[0]
        if not quality[1]:
            level = self.y_prev[1]
        
        # Setpoint handling
        r_curr = r[0] if not np.isnan(r[0]) else self.r_filt
        # Very gentle setpoint filter
        alpha_r = 0.1
        self.r_filt = alpha_r * r_curr + (1 - alpha_r) * self.r_filt
        
        # Measurement filtering
        alpha_y = self.Ts / (self.Tf + self.Ts)
        y_filt_new = (1 - alpha_y) * self.y_filt + alpha_y * ph_meas
        self.y_filt = y_filt_new
        
        # Compute error
        e = self.r_filt - self.y_filt
        
        # Compute derivative (filtered, on measurement)
        deriv = 0.0
        if self.t_prev is not None:
            dt = t - self.t_prev
            if dt > 0:
                deriv = -(self.y_filt - self.y_prev_filt) / dt
        
        self.y_prev_filt = self.y_filt
        self.t_prev = t
        self.y_prev = np.array([ph_meas, level])
        
        # PID calculation
        P = self.Kp * e
        D = self.Kd * deriv
        
        # Compute preliminary output
        u_pid = P + self.integral + D
        
        # Safety overrides based on level
        u_safe = u_pid
        
        # Level-based safety
        if level < self.h_crit_low:
            u_safe = max(u_safe, self.u_prev + 0.3)
        elif level < self.h_low_safe:
            level_factor = 1.0 + 0.2 * (self.h_low_safe - level) / (self.h_low_safe - 5.0)
            u_safe = max(u_safe, u_pid * level_factor)
        
        if level > self.h_crit_high:
            u_safe = min(u_safe, self.u_prev - 0.3)
        elif level > self.h_high_safe:
            level_factor = 1.0 - 0.2 * (level - self.h_high_safe) / (30.0 - self.h_high_safe)
            u_safe = min(u_safe, u_pid * level_factor)
        
        # pH safety
        if ph_meas < self.ph_crit_low:
            u_safe = max(u_safe, self.u_prev + 0.5)
        elif ph_meas < self.ph_low_safe:
            acid_factor = 1.0 + 0.5 * (self.ph_low_safe - ph_meas) / (self.ph_low_safe - 4.0)
            u_safe = max(u_safe, u_pid * acid_factor)
        
        if ph_meas > self.ph_crit_high:
            u_safe = min(u_safe, self.u_prev - 0.5)
        elif ph_meas > self.ph_high_safe:
            base_factor = 1.0 - 0.5 * (ph_meas - self.ph_high_safe) / (10.5 - self.ph_high_safe)
            u_safe = min(u_safe, u_pid * base_factor)
        
        # Hard limits
        u = np.clip(u_safe, self.u_min, self.u_max)
        
        # Anti-windup
        if self.u_min < u < self.u_max:
            self.integral += self.Ki * e * self.Ts
            self.integral = np.clip(self.integral, -30, 50)
        else:
            if u >= self.u_max and e < 0:
                self.integral += self.Ki * e * self.Ts
            elif u <= self.u_min and e > 0:
                self.integral += self.Ki * e * self.Ts
        
        # Very strict rate limiting to avoid duty limit
        u_rate_limited = np.clip(u, self.u_prev - self.max_step, self.u_prev + self.max_step)
        
        # Final output
        u_final = np.clip(u_rate_limited, self.u_min, self.u_max)
        
        self.u_prev = u_final
        
        return np.array([u_final])