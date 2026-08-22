import numpy as np
from dataclasses import dataclass

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        
        # PID gains - balanced for tracking and robustness
        self.Kp = np.array([1.5, 1.5])
        self.Ki = np.array([0.03, 0.03])
        self.Kd = np.array([0.5, 0.5])
        
        # Anti-windup limits
        self.integral = np.array([0.0, 0.0])
        self.integral_max = np.array([25.0, 25.0])
        self.integral_min = np.array([-25.0, -25.0])
        
        # Derivative filtering
        self.deriv_filter = np.array([0.0, 0.0])
        self.deriv_alpha = 0.2
        
        # Previous values
        self.prev_y = np.array([12.0, 12.0])
        self.prev_u = np.array([3.0, 3.0])
        
        # Actuator limits
        self.u_min = np.array([0.0, 0.0])
        self.u_max = np.array([10.0, 10.0])
        
        # Slew rate limiting
        self.max_slew = 1.0
        
        # Reference filter
        self.filtered_r = np.array([12.0, 12.0])
        self.r_filter_alpha = 0.25
        
        # Quality tracking
        self.last_good_y = np.array([12.0, 12.0])
        
        self.reset()
    
    def reset(self):
        self.integral = np.array([0.0, 0.0])
        self.deriv_filter = np.array([0.0, 0.0])
        self.prev_y = np.array([12.0, 12.0])
        self.prev_u = np.array([3.0, 3.0])
        self.filtered_r = np.array([12.0, 12.0])
        self.last_good_y = np.array([12.0, 12.0])
    
    def step(self, t, y, r, quality):
        y_valid = y.copy()
        for i in range(len(y)):
            if not quality[i] or np.isnan(y[i]):
                y_valid[i] = self.last_good_y[i]
            else:
                self.last_good_y[i] = y[i]
        
        r_valid = r.copy()
        for i in range(len(r)):
            if np.isnan(r[i]):
                r_valid[i] = self.filtered_r[i]
        
        # Reference pre-filtering
        self.filtered_r = self.r_filter_alpha * r_valid + (1 - self.r_filter_alpha) * self.filtered_r
        
        # Error
        error = self.filtered_r - y_valid
        
        # Proportional
        P = self.Kp * error
        
        # Integral with anti-windup
        integral_change = self.Ki * error * self.sample_time
        
        for i in range(2):
            if self.prev_u[i] >= self.u_max[i] - 0.05 and error[i] > 0:
                integral_change[i] *= 0.02
            elif self.prev_u[i] <= self.u_min[i] + 0.05 and error[i] < 0:
                integral_change[i] *= 0.02
        
        self.integral += integral_change
        self.integral = np.clip(self.integral, self.integral_min, self.integral_max)
        I = self.integral
        
        # Derivative on measurement
        deriv_raw = -(y_valid - self.prev_y) / self.sample_time
        self.deriv_filter = self.deriv_alpha * deriv_raw + (1 - self.deriv_alpha) * self.deriv_filter
        D = self.Kd * self.deriv_filter
        
        # Feedforward
        u_ff = np.array([3.0, 3.0])
        level_offset = (self.filtered_r - 12.0) * 0.1
        u_ff += level_offset
        
        u = P + I + D + u_ff
        
        # Slew limiting
        u_delta = u - self.prev_u
        u_delta = np.clip(u_delta, -self.max_slew, self.max_slew)
        u = self.prev_u + u_delta
        
        # Output limits
        u = np.clip(u, self.u_min, self.u_max)
        
        # Safety: reduce if level too high
        for i in range(2):
            if y_valid[i] > 18.0:
                reduction = (y_valid[i] - 18.0) / 2.0
                u[i] = max(0, u[i] * (1.0 - reduction * 0.5))
        
        self.prev_y = y_valid.copy()
        self.prev_u = u.copy()
        
        return u