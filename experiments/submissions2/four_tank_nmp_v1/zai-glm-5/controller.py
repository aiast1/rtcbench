import numpy as np
from dataclasses import dataclass

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        
        # Nominal model parameters
        self.A1, self.A2, self.A3, self.A4 = 28.0, 32.0, 28.0, 32.0
        self.a1, self.a2, self.a3, self.a4 = 0.071, 0.057, 0.071, 0.057
        self.k1, self.k2 = 3.14, 3.29
        self.gamma1, self.gamma2 = 0.43, 0.34
        
        # Steady-state operating point
        self.h1_ss = 11.28
        self.h2_ss = 11.94
        self.h3_ss = 6.0
        self.h4_ss = 6.0
        self.u1_ss = 3.0
        self.u2_ss = 3.0
        
        # Controller gains - conservative for robustness
        self.Kp = np.array([0.8, 0.8])
        self.Ki = np.array([0.04, 0.04])
        self.Kd = np.array([1.5, 1.5])
        
        # Anti-windup time constant
        self.Tt = 20.0
        
        # State variables
        self.integral = np.zeros(2)
        self.prev_error = None
        self.prev_t = None
        self.prev_y = None
        self.prev_u = np.array([3.0, 3.0])
        
        # Filter coefficients
        self.Tf = 8.0  # Derivative filter time constant
        self.y_filter = np.zeros(2)
        
        # Safety margins for unmeasured tanks
        self.safety_margin = 0.5
        
        # Maximum duty cycle
        self.max_duty = 0.0037
        
    def reset(self):
        self.integral = np.zeros(2)
        self.prev_error = None
        self.prev_t = None
        self.prev_y = None
        self.prev_u = np.array([3.0, 3.0])
        self.y_filter = np.zeros(2)
        
    def _estimate_unmeasured(self, y, u):
        """Estimate unmeasured tank levels using mass balance"""
        h1, h2 = y[0], y[1]
        u1, u2 = u[0], u[1]
        
        # Simplified steady-state estimation
        # Upper tanks receive gamma fraction of pump flow
        # h3 ~ (gamma1 * k1 * u1 / a3)^2
        # h4 ~ (gamma2 * k2 * u2 / a4)^2
        
        h3_est = max(0.5, min(19.5, (self.gamma1 * self.k1 * u1 / self.a3)**2 * 0.8 + 3.0))
        h4_est = max(0.5, min(19.5, (self.gamma2 * self.k2 * u2 / self.a4)**2 * 0.8 + 3.0))
        
        return h3_est, h4_est
    
    def _safety_constraint(self, u, y):
        """Apply safety constraints for all tanks including unmeasured"""
        u_safe = u.copy()
        
        # Estimate unmeasured tank levels
        h3_est, h4_est = self._estimate_unmeasured(y, u)
        
        # Upper limit safety - reduce flow if tanks approaching overflow
        upper_limit = 19.0
        
        if y[0] > upper_limit:
            u_safe[0] = min(u_safe[0], self.u1_ss * 0.5)
        if y[1] > upper_limit:
            u_safe[1] = min(u_safe[1], self.u2_ss * 0.5)
        if h3_est > upper_limit:
            u_safe[0] = min(u_safe[0], self.u1_ss * 0.6)
        if h4_est > upper_limit:
            u_safe[1] = min(u_safe[1], self.u2_ss * 0.6)
        
        # Lower limit safety - increase flow if tanks too low
        lower_limit = 1.0
        
        if y[0] < lower_limit:
            u_safe[0] = max(u_safe[0], self.u1_ss * 1.5)
        if y[1] < lower_limit:
            u_safe[1] = max(u_safe[1], self.u2_ss * 1.5)
        
        return u_safe
    
    def step(self, t, y, r, quality):
        dt = self.sample_time
        
        # Handle bad quality readings
        y_valid = y.copy()
        if quality is not None:
            for i in range(len(y)):
                if not quality[i]:
                    y_valid[i] = self.y_filter[i] if self.y_filter[i] != 0 else y[i]
        
        # Filter measurements
        alpha = dt / (self.Tf + dt)
        if self.y_filter[0] == 0:
            self.y_filter = y_valid.copy()
        else:
            self.y_filter = (1 - alpha) * self.y_filter + alpha * y_valid
        
        y_filtered = self.y_filter
        
        # Calculate error
        error = np.zeros(2)
        for i in range(2):
            if np.isnan(r[i]):
                error[i] = 0.0
            else:
                error[i] = r[i] - y_filtered[i]
        
        # Initialize on first step
        if self.prev_error is None:
            self.prev_error = error.copy()
            self.prev_y = y_filtered.copy()
            self.prev_u = np.array([self.u1_ss, self.u2_ss])
            # Initialize integral for bumpless transfer
            self.integral = np.array([self.u1_ss, self.u2_ss]) - self.Kp * error
            return np.array([self.u1_ss, self.u2_ss])
        
        # Derivative on measurement (not error) to avoid derivative kick
        if self.prev_y is not None:
            D = -self.Kd * (y_filtered - self.prev_y) / dt
        else:
            D = np.zeros(2)
        
        # Calculate raw control output
        u_raw = self.Kp * error + self.integral + D
        
        # Apply actuator limits
        u_sat = np.clip(u_raw, 0.0, 10.0)
        
        # Anti-windup: back-calculate integral
        if dt > 0:
            windup_error = (u_sat - u_raw)
            self.integral += self.Ki * error * dt + windup_error * dt / self.Tt
        
        # Apply safety constraints
        u_safe = self._safety_constraint(u_sat, y_filtered)
        u_safe = np.clip(u_safe, 0.0, 10.0)
        
        # Rate limiting for duty cycle compliance
        max_change = self.max_duty * dt * 100  # Conservative limit
        u_rate_limited = np.zeros(2)
        for i in range(2):
            delta = u_safe[i] - self.prev_u[i]
            if abs(delta) > max_change:
                u_rate_limited[i] = self.prev_u[i] + np.sign(delta) * max_change
            else:
                u_rate_limited[i] = u_safe[i]
        
        # Store state for next iteration
        self.prev_error = error.copy()
        self.prev_y = y_filtered.copy()
        self.prev_u = u_rate_limited.copy()
        
        return u_rate_limited