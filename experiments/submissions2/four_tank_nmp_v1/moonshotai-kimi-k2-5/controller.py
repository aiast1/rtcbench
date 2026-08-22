import numpy as np
from dataclasses import dataclass

@dataclass
class TaskBrief:
    sample_time: float

class Controller:
    def __init__(self, brief: TaskBrief):
        self.Ts = brief.sample_time  # 2.0 seconds
        
        # Much more conservative tuning - prioritize safety over performance
        # The plant has delays, noise, and unmeasured states that can overflow
        self.Kp = np.array([0.3, 0.3])
        self.Ki = np.array([0.05, 0.05])
        self.Kd = np.array([0.1, 0.1])
        
        # Derivative filter time constant
        self.Tf = 5.0
        
        # Actuator limits
        self.u_min = np.array([0.0, 0.0])
        self.u_max = np.array([10.0, 10.0])
        
        # Initial operating point
        self.u_op = np.array([3.0, 3.0])
        
        # State
        self.y_filt = None
        self.y_prev = None
        self.integral = np.zeros(2)
        self.u_prev = None
        
        # Nominal model for upper tank estimation
        self.k_nom = np.array([3.14, 3.29])
        self.gamma_nom = np.array([0.43, 0.34])
        self.a_upper = np.array([0.071, 0.057])
        
        # Conservative upper tank limit (well below 20)
        self.h_upper_limit = 15.0
        
        # Rate limiter for smooth control (prevent duty limit violations)
        self.max_du = 0.5  # max change per step
        
        # Setpoint rate limiter for smooth tracking
        self.r_filt = None
        
    def reset(self):
        self.y_filt = None
        self.y_prev = None
        self.integral = np.zeros(2)
        self.u_prev = None
        self.r_filt = None
        
    def _estimate_upper_tank_levels(self, u):
        """Estimate upper tank levels from pump voltages."""
        h3 = ((self.gamma_nom[0] * self.k_nom[0] * u[0]) / self.a_upper[0]) ** 2
        h4 = ((self.gamma_nom[1] * self.k_nom[1] * u[1]) / self.a_upper[1]) ** 2
        return np.array([h3, h4])
    
    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        quality = np.asarray(quality, dtype=bool)
        
        # Initialize
        if self.y_filt is None:
            self.y_filt = y.copy()
            self.y_prev = y.copy()
            self.r_filt = r.copy()
            self.u_prev = self.u_op.copy()
            self.integral = np.zeros(2)
            return self.u_op.copy()
        
        # Handle bad measurements
        y_use = y.copy()
        for i in range(len(y)):
            if not quality[i]:
                y_use[i] = self.y_filt[i]
        
        # Smooth setpoint changes
        alpha_r = 0.1
        for i in range(len(r)):
            if not np.isnan(r[i]):
                self.r_filt[i] = (1 - alpha_r) * self.r_filt[i] + alpha_r * r[i]
        
        # Filter measurements
        alpha_y = self.Ts / (self.Tf + self.Ts)
        self.y_filt = (1 - alpha_y) * self.y_filt + alpha_y * y_use
        
        # Compute error
        e = np.zeros(2)
        for i in range(2):
            if not np.isnan(self.r_filt[i]):
                e[i] = self.r_filt[i] - self.y_filt[i]
        
        # Derivative
        dy = (self.y_filt - self.y_prev) / self.Ts
        
        # PID output
        P = self.Kp * e
        D = -self.Kd * dy  # derivative on measurement
        u_dev = P + self.Ki * self.integral + D
        
        # Base control
        u_raw = self.u_op + u_dev
        
        # Safety: estimate upper tanks and limit if needed
        h_upper = self._estimate_upper_tank_levels(u_raw)
        
        u_safe = u_raw.copy()
        for i in range(2):
            if h_upper[i] > self.h_upper_limit:
                # Reduce pump to keep upper tank safe
                # h = (gamma*k*u/a)^2 -> u = a*sqrt(h)/(gamma*k)
                u_safe[i] = self.a_upper[i] * np.sqrt(self.h_upper_limit * 0.9) / (self.gamma_nom[i] * self.k_nom[i])
        
        # Hard limits
        u_lim = np.clip(u_safe, self.u_min, self.u_max)
        
        # Rate limit to prevent duty violation
        du = u_lim - self.u_prev
        for i in range(2):
            if abs(du[i]) > self.max_du:
                du[i] = np.sign(du[i]) * self.max_du
        
        u_rate = self.u_prev + du
        
        # Final saturation
        u_out = np.clip(u_rate, self.u_min, self.u_max)
        
        # Anti-windup: only integrate if not saturated
        for i in range(2):
            if self.u_min[i] + 0.01 < u_out[i] < self.u_max[i] - 0.01:
                self.integral[i] += e[i] * self.Ts
        
        # Clamp integral
        self.integral = np.clip(self.integral, -20.0, 20.0)
        
        # Update state
        self.y_prev = self.y_filt.copy()
        self.u_prev = u_out.copy()
        
        return u_out