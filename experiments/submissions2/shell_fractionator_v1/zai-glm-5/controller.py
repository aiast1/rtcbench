import numpy as np
from dataclasses import dataclass

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.n_u = 3
        self.n_y = 3
        
        # Nominal model parameters
        self.K = np.array([
            [4.05, 1.77, 5.88],
            [5.39, 5.72, 6.90],
            [4.38, 4.42, 7.20]
        ])
        self.TAU = np.array([
            [50.0, 60.0, 50.0],
            [50.0, 60.0, 40.0],
            [33.0, 44.0, 19.0]
        ])
        self.L = np.array([
            [27.0, 28.0, 27.0],
            [18.0, 14.0, 15.0],
            [20.0, 22.0, 0.0]
        ])
        
        # Disturbance model
        self.KD = np.array([
            [1.2, 1.44],
            [1.52, 1.83],
            [1.14, 1.26]
        ])
        self.TAUD = np.array([
            [45.0, 40.0],
            [25.0, 20.0],
            [27.0, 32.0]
        ])
        self.LD = np.array([
            [27.0, 27.0],
            [15.0, 15.0],
            [27.0, 32.0]
        ])
        
        # Actuator limits
        self.u_min = np.array([-0.5, -0.5, -0.5])
        self.u_max = np.array([0.5, 0.5, 0.5])
        
        # Safety constraint: y[2] >= -0.5
        self.y_min = np.array([-np.inf, -np.inf, -0.5])
        
        # Tuning parameters - conservative for robustness
        # Using IMC tuning rules with lambda factor for robustness
        self.lambda_factor = 2.5
        
        # Initialize gains based on diagonal elements (primary pairing)
        self.Kp = np.zeros(self.n_y)
        self.Ki = np.zeros(self.n_y)
        self.Kd = np.zeros(self.n_y)
        
        for i in range(self.n_y):
            # Primary pairing: u[i] -> y[i]
            k = self.K[i, i]
            tau = self.TAU[i, i]
            L = max(self.L[i, i], 1.0)
            
            # IMC-based PID tuning
            self.Kp[i] = tau / (k * (L + self.lambda_factor * tau))
            self.Ki[i] = self.Kp[i] / max(tau, 10.0)
            self.Kd[i] = self.Kp[i] * min(tau * 0.3, 15.0)
        
        # Reduce gains for interaction
        self.Kp *= 0.6
        self.Ki *= 0.5
        self.Kd *= 0.3
        
        # Further reduce derivative for noisy channels
        self.Kd[0] *= 0.5  # AI-101 likely noisy
        self.Kd[1] *= 0.5  # AI-102 likely noisy
        
        # State variables
        self.integral = np.zeros(self.n_y)
        self.prev_error = np.zeros(self.n_y)
        self.prev_y = np.zeros(self.n_y)
        self.prev_u = np.zeros(self.n_u)
        self.prev_r = np.zeros(self.n_y)
        
        # Filter coefficients
        self.Tf = 5.0  # Filter time constant for derivative
        self.alpha = self.sample_time / (self.Tf + self.sample_time)
        
        # Anti-windup back-calculation coefficient
        self.Kb = 0.5
        
        # Rate limit for actuator movement (duty limit consideration)
        self.max_rate = 0.01  # Conservative rate limit
        
        # Reference filter for smooth setpoint tracking
        self.ref_filter_tc = 20.0
        self.ref_filter_alpha = self.sample_time / (self.ref_filter_tc + self.sample_time)
        self.filtered_r = np.zeros(self.n_y)
        
        # Safety margin for y[2] constraint
        self.safety_margin = 0.05
        
        # Initialize
        self.initialized = False
        self.t_prev = 0.0
        
    def reset(self):
        self.integral = np.zeros(self.n_y)
        self.prev_error = np.zeros(self.n_y)
        self.prev_y = np.zeros(self.n_y)
        self.prev_u = np.zeros(self.n_u)
        self.prev_r = np.zeros(self.n_y)
        self.filtered_r = np.zeros(self.n_y)
        self.initialized = False
        self.t_prev = 0.0
        
    def step(self, t, y, r, quality):
        dt = self.sample_time
        
        # Handle bad quality readings
        y_valid = y.copy()
        for i in range(self.n_y):
            if not quality[i]:
                y_valid[i] = self.prev_y[i] if self.initialized else 0.0
        
        # Initialize on first step
        if not self.initialized:
            self.prev_y = y_valid.copy()
            self.prev_r = r.copy()
            self.filtered_r = r.copy()
            self.prev_u = np.zeros(self.n_u)
            self.initialized = True
            return np.zeros(self.n_u)
        
        # Filter setpoints for smooth tracking
        for i in range(self.n_y):
            if np.isfinite(r[i]):
                self.filtered_r[i] = self.filtered_r[i] + self.ref_filter_alpha * (r[i] - self.filtered_r[i])
        
        # Calculate error
        error = self.filtered_r - y_valid
        
        # Safety constraint handling for y[2]
        # Predict future y[2] based on current trend
        y2_trend = (y_valid[2] - self.prev_y[2]) / dt if dt > 0 else 0.0
        y2_predicted = y_valid[2] + y2_trend * 10.0  # 10 second prediction
        
        # If approaching safety limit, modify setpoint and add corrective action
        safety_limit = self.y_min[2] + self.safety_margin
        if y2_predicted < safety_limit or y_valid[2] < safety_limit:
            # Override error for y[2] to push away from limit
            error[2] = max(error[2], safety_limit - y_valid[2] + 0.1)
        
        # PID calculation
        u = np.zeros(self.n_u)
        
        for i in range(self.n_y):
            # Proportional term
            P = self.Kp[i] * error[i]
            
            # Integral term with anti-windup
            # Back-calculation anti-windup
            windup_correction = 0.0
            for j in range(self.n_u):
                if self.prev_u[j] >= self.u_max[j] or self.prev_u[j] <= self.u_min[j]:
                    windup_correction += self.Kb * (self.prev_u[j] - np.clip(self.prev_u[j], self.u_min[j], self.u_max[j]))
            
            self.integral[i] += self.Ki[i] * error[i] * dt - windup_correction * dt
            I = self.integral[i]
            
            # Derivative term on measurement (not error) to avoid derivative kick
            dy = y_valid[i] - self.prev_y[i]
            D = -self.Kd[i] * dy / dt if dt > 0 else 0.0
            
            # Filter derivative
            D = self.alpha * D + (1 - self.alpha) * self.prev_error[i] * self.Kd[i] if self.initialized else D
            
            u[i] = P + I + D
        
        # Decoupling: use relative gain array insight
        # Simple decoupling by reducing cross-coupled gains
        u_decoupled = np.zeros(self.n_u)
        for i in range(self.n_u):
            u_decoupled[i] = u[i]
            # Add small correction from other loops
            for j in range(self.n_y):
                if i != j:
                    # Reduce cross-coupling effect
                    u_decoupled[i] -= 0.1 * self.K[i, j] / self.K[i, i] * u[j]
        
        u = u_decoupled
        
        # Apply safety override for y[2]
        if y_valid[2] < self.y_min[2] + self.safety_margin:
            # Increase u[2] to raise temperature
            u[2] = max(u[2], 0.1)
        
        # Rate limiting for actuator duty constraint
        for i in range(self.n_u):
            rate = (u[i] - self.prev_u[i]) / dt if dt > 0 else 0.0
            if abs(rate) > self.max_rate:
                u[i] = self.prev_u[i] + np.sign(rate) * self.max_rate * dt
        
        # Saturate actuators
        u = np.clip(u, self.u_min, self.u_max)
        
        # Update state
        self.prev_y = y_valid.copy()
        self.prev_u = u.copy()
        self.prev_r = r.copy()
        self.prev_error = error.copy()
        self.t_prev = t
        
        return u