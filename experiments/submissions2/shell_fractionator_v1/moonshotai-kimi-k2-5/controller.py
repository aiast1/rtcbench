import numpy as np
from dataclasses import dataclass

@dataclass
class TaskBrief:
    sample_time: float

class Controller:
    def __init__(self, brief):
        self.dt = brief.sample_time
        
        self.n_y = 3
        self.n_u = 3
        
        # Nominal model
        self.K_nom = np.array([
            [4.05, 1.77, 5.88],
            [5.39, 5.72, 6.90],
            [4.38, 4.42, 7.20]
        ])
        
        self.u_min = np.array([-0.5, -0.5, -0.5])
        self.u_max = np.array([0.5, 0.5, 0.5])
        
        self.y3_min = -0.5
        
        # Tuning: more aggressive for better tracking
        self.Kp = np.array([0.22, 0.18, 0.25])
        self.Ti = np.array([35.0, 40.0, 28.0])
        self.Td = np.array([4.0, 4.0, 3.0])
        
        self.N = 6.0
        
        # Decoupling
        K_reg = self.K_nom + 0.3 * np.eye(3)
        self.K_inv = np.linalg.inv(K_reg)
        self.decoupling_gain = 0.85
        
        # Feedforward for setpoint changes
        self.Kff = 0.5 * self.K_inv
        
        self.reset()
        
    def reset(self):
        self.integral = np.zeros(self.n_u)
        self.y_prev = np.zeros(self.n_y)
        self.y_filt = np.zeros(self.n_y)
        self.u_prev = np.zeros(self.n_u)
        self.dy_filt = np.zeros(self.n_y)
        self.r_prev = np.zeros(self.n_y)
        self.r_filt = np.zeros(self.n_y)
        
    def step(self, t, y, r, quality):
        y_used = np.where(quality, y, self.y_filt)
        
        # Measurement filtering
        alpha_filt = 0.35
        self.y_filt = alpha_filt * y_used + (1 - alpha_filt) * self.y_filt
        
        # Setpoint filtering for smooth tracking
        alpha_r = 0.25
        r_clean = np.where(np.isnan(r), self.r_filt, r)
        self.r_filt = alpha_r * r_clean + (1 - alpha_r) * self.r_filt
        
        # Setpoint rate for feedforward
        dr = (self.r_filt - self.r_prev) / self.dt
        
        # Errors
        e = self.r_filt - self.y_filt
        e_scored = np.where(np.isnan(r), 0.0, e)
        
        # Derivative
        dy_raw = (self.y_filt - self.y_prev) / self.dt
        beta = self.N * self.dt
        self.dy_filt = (self.dy_filt + beta * dy_raw) / (1 + beta)
        
        # PID
        P = self.Kp * e_scored
        I = self.Kp / self.Ti * self.integral
        D = -self.Kp * self.Td * self.dy_filt
        
        v = P + I + D
        
        # Decoupled control with feedforward for setpoint changes
        u_fb = self.decoupling_gain * self.K_inv @ v
        u_ff = self.Kff @ dr
        u_raw = u_fb + u_ff
        
        # Safety for y[2]
        y3_margin = self.y_filt[2] - self.y3_min
        if y3_margin < 0.2:
            u_raw[2] += 0.5 * (0.25 - y3_margin)
            u_raw[0] -= 0.15 * (0.25 - y3_margin)
            u_raw[1] -= 0.15 * (0.25 - y3_margin)
        
        # Rate limit
        du_max = 0.009
        du = u_raw - self.u_prev
        du_sat = np.clip(du, -du_max, du_max)
        u_rl = self.u_prev + du_sat
        
        u = np.clip(u_rl, self.u_min, self.u_max)
        
        # Anti-windup
        for i in range(self.n_u):
            if self.u_min[i] + 0.001 < u[i] < self.u_max[i] - 0.001:
                self.integral[i] += e_scored[i % self.n_y] * self.dt
            elif u[i] >= self.u_max[i] - 0.001 and e_scored[i % self.n_y] < 0:
                self.integral[i] += e_scored[i % self.n_y] * self.dt
            elif u[i] <= self.u_min[i] + 0.001 and e_scored[i % self.n_y] > 0:
                self.integral[i] += e_scored[i % self.n_y] * self.dt
        
        self.y_prev = self.y_filt.copy()
        self.u_prev = u.copy()
        self.r_prev = self.r_filt.copy()
        
        return u