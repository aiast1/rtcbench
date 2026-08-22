import numpy as np
from dataclasses import dataclass

@dataclass
class TaskBrief:
    sample_time: float

class Controller:
    def __init__(self, brief):
        self.Ts = brief.sample_time
        
        self.ny = 3
        self.nu = 3
        
        # Nominal model
        self.K_nom = np.array([
            [4.05, 1.77, 5.88],
            [5.39, 5.72, 6.90],
            [4.38, 4.42, 7.20]
        ])
        
        self.TAU_nom = np.array([
            [50.0, 60.0, 50.0],
            [50.0, 60.0, 40.0],
            [33.0, 44.0, 19.0]
        ])
        
        # Actuator limits
        self.u_min = np.array([-0.5, -0.5, -0.5])
        self.u_max = np.array([0.5, 0.5, 0.5])
        
        # Safety constraint
        self.y3_min = -0.5
        
        # Tighter PI tuning for better tracking
        self.Kp = np.array([0.22, 0.22, 0.15])
        self.Ki = np.array([0.015, 0.015, 0.010])
        self.Kd = np.array([1.5, 1.5, 0.9])
        
        # Derivative filter
        self.Tf = 6.0
        
        # Anti-windup
        self.Kaw = 1.5
        
        # Setpoint weighting
        self.b = 0.5
        
        # Feedforward gain for setpoint changes
        self.Kff = 0.3
        
        self.reset()
    
    def reset(self):
        self.integral = np.zeros(self.nu)
        self.y_filt = np.zeros(self.ny)
        self.y_filt2 = np.zeros(self.ny)
        self.dy_filt = np.zeros(self.ny)
        self.u_prev = np.zeros(self.nu)
        self.r_prev = np.zeros(self.ny)
        self.r_filtered = np.zeros(self.ny)
        self.initialized = False
        
        # For feedforward
        self.dr_prev = np.zeros(self.ny)
    
    def _filter_measurement(self, y, quality):
        """Two-stage filtering"""
        alpha1 = np.array([0.5, 0.5, 0.7])
        
        y_filt_new = np.zeros(self.ny)
        for i in range(self.ny):
            if quality[i]:
                y_filt_new[i] = alpha1[i] * y[i] + (1 - alpha1[i]) * self.y_filt[i]
            else:
                y_filt_new[i] = self.y_filt[i]
        
        alpha2 = 0.6
        y_filt2_new = alpha2 * y_filt_new + (1 - alpha2) * self.y_filt2
        
        return y_filt_new, y_filt2_new
    
    def _filter_setpoint(self, r):
        """Filter setpoint for smooth tracking"""
        alpha_r = 0.3
        r_filt = np.zeros(self.ny)
        for i in range(self.ny):
            if not np.isnan(r[i]):
                r_filt[i] = alpha_r * r[i] + (1 - alpha_r) * self.r_filtered[i]
            else:
                r_filt[i] = self.r_filtered[i]
        return r_filt
    
    def _compute_derivative(self, y_filt2_new):
        dy = (y_filt2_new - self.y_filt2) / self.Ts
        beta = self.Ts / (self.Tf + self.Ts)
        self.dy_filt = beta * dy + (1 - beta) * self.dy_filt
        return self.dy_filt
    
    def step(self, t, y, r, quality):
        # Filter measurements
        y_filt_new, y_filt2_new = self._filter_measurement(y, quality)
        
        # Initialize
        if not self.initialized:
            self.y_filt = y_filt_new.copy()
            self.y_filt2 = y_filt2_new.copy()
            for i in range(self.ny):
                if not np.isnan(r[i]):
                    self.r_prev[i] = r[i]
                    self.r_filtered[i] = r[i]
                else:
                    self.r_prev[i] = y_filt2_new[i]
                    self.r_filtered[i] = y_filt2_new[i]
            self.initialized = True
        
        # Update filtered values
        self.y_filt = y_filt_new
        self.y_filt2 = y_filt2_new
        
        # Filter setpoint
        r_filt = self._filter_setpoint(r)
        
        # Compute derivative
        dy_filt = self._compute_derivative(y_filt2_new)
        
        # Update r_filtered for next time
        for i in range(self.ny):
            if not np.isnan(r[i]):
                self.r_filtered[i] = r_filt[i]
        
        # Compute errors
        e = np.zeros(self.ny)
        e_p = np.zeros(self.ny)
        
        for i in range(self.ny):
            if not np.isnan(r[i]):
                e[i] = r_filt[i] - y_filt2_new[i]
                e_p[i] = self.b * r_filt[i] + (1 - self.b) * self.r_prev[i] - y_filt2_new[i]
            else:
                e[i] = 0.0
                e_p[i] = 0.0
        
        # Update previous setpoint
        for i in range(self.ny):
            if not np.isnan(r[i]):
                self.r_prev[i] = r[i]
        
        # PI-D control
        u_p = self.Kp * e_p
        u_i = self.Ki * self.integral
        u_d = -self.Kd * dy_filt
        
        # Feedforward on setpoint rate
        dr = np.zeros(self.ny)
        for i in range(self.ny):
            if not np.isnan(r[i]):
                dr[i] = (r_filt[i] - self.r_prev[i]) / self.Ts if not np.isnan(self.r_prev[i]) else 0
        
        u_ff = self.Kff * dr
        
        u_raw = u_p + u_i + u_d + u_ff
        
        # Decoupling
        decouple = 0.5
        
        # u0 affects y1 and y2
        u_raw[1] -= decouple * 0.35 * u_raw[0]
        u_raw[2] -= decouple * 0.25 * u_raw[0]
        
        # u1 affects y0 and y2
        u_raw[0] -= decouple * 0.18 * u_raw[1]
        u_raw[2] -= decouple * 0.30 * u_raw[1]
        
        # u2 affects y0 and y1 strongly
        u_raw[0] -= decouple * 0.50 * u_raw[2]
        u_raw[1] -= decouple * 0.55 * u_raw[2]
        
        # Constraint handling for y3
        y3_margin = y_filt2_new[2] - self.y3_min
        
        if y3_margin < 0.25:
            constraint_action = np.clip(0.4 * (0.25 - y3_margin) / 0.25, 0, 0.6)
            u_raw[2] += constraint_action
            
            if u_raw[0] > 0:
                u_raw[0] *= 0.7
            if u_raw[1] > 0:
                u_raw[1] *= 0.7
        
        if y3_margin < 0.08:
            u_raw[2] += 0.3
        
        # Apply limits
        u_sat = np.clip(u_raw, self.u_min, self.u_max)
        
        # Anti-windup
        u_diff = u_sat - u_raw
        
        for i in range(self.nu):
            if not np.isnan(r[i]) or i == 2:
                self.integral[i] += self.Ts * e[i] + self.Kaw * u_diff[i]
                self.integral[i] = np.clip(self.integral[i], -4.0, 4.0)
        
        # Rate limiting
        max_rate = 0.015
        u_rate = (u_sat - self.u_prev) / self.Ts
        u_rate_clipped = np.clip(u_rate, -max_rate, max_rate)
        u_smooth = self.u_prev + u_rate_clipped * self.Ts
        
        u_out = np.clip(u_smooth, self.u_min, self.u_max)
        
        self.u_prev = u_out
        
        return u_out