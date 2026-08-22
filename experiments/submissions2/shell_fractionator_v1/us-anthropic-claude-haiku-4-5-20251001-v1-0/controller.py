import numpy as np
from dataclasses import dataclass

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = brief.sample_time
        
        # Nominal model parameters
        self.K = np.array([
            [4.05, 1.77, 5.88],
            [5.39, 5.72, 6.9],
            [4.38, 4.42, 7.2]
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
        
        # Control parameters
        self.ny = 3
        self.nu = 3
        
        # PID tuning - balanced for tracking and effort
        self.Kp = np.array([0.30, 0.25, 0.18])
        self.Ki = np.array([0.020, 0.016, 0.010])
        self.Kd = np.array([1.0, 0.8, 0.4])
        
        # Actuator limits
        self.u_min = np.array([-0.5, -0.5, -0.5])
        self.u_max = np.array([0.5, 0.5, 0.5])
        
        # State tracking
        self.integral_error = np.zeros(self.ny)
        self.last_error = np.zeros(self.ny)
        self.last_y = np.zeros(self.ny)
        self.u_prev = np.zeros(self.nu)
        
        # Derivative filter
        self.deriv_filter_alpha = 0.5
        self.filtered_deriv = np.zeros(self.ny)
        
        # Duty cycle tracking
        self.duty_accumulator = 0.0
        self.last_duty_reset = 0.0
        
    def reset(self):
        self.integral_error = np.zeros(self.ny)
        self.last_error = np.zeros(self.ny)
        self.last_y = np.zeros(self.ny)
        self.u_prev = np.zeros(self.nu)
        self.filtered_deriv = np.zeros(self.ny)
        self.duty_accumulator = 0.0
        self.last_duty_reset = 0.0
    
    def step(self, t, y, r, quality):
        # Handle bad measurements - use last known good value
        y_use = np.where(quality, y, self.last_y)
        self.last_y = y_use.copy()
        
        # Compute error only for scored channels (where r is not nan)
        error = np.zeros(self.ny)
        for i in range(self.ny):
            if not np.isnan(r[i]):
                error[i] = r[i] - y_use[i]
        
        # Proportional term
        P = self.Kp * error
        
        # Integral term with anti-windup
        self.integral_error += error * self.dt
        
        # Clamp integral to prevent excessive windup
        integral_limit = 0.5
        self.integral_error = np.clip(self.integral_error, -integral_limit, integral_limit)
        
        I = self.Ki * self.integral_error
        
        # Derivative term with filtering and quality check
        D = np.zeros(self.ny)
        for i in range(self.ny):
            if quality[i]:
                deriv_raw = (error[i] - self.last_error[i]) / self.dt
                # Low-pass filter on derivative
                self.filtered_deriv[i] = (self.deriv_filter_alpha * deriv_raw + 
                                         (1.0 - self.deriv_filter_alpha) * self.filtered_deriv[i])
                D[i] = self.Kd[i] * self.filtered_deriv[i]
        
        self.last_error = error.copy()
        
        # Compute control action
        u_raw = P + I + D
        
        # Apply actuator saturation with anti-windup
        u = np.clip(u_raw, self.u_min, self.u_max)
        
        # Anti-windup: reduce integral if saturated
        for i in range(self.nu):
            if u[i] != u_raw[i]:
                # Saturated - reduce integral contribution
                self.integral_error[i] *= 0.80
        
        # Rate limiting (slew rate protection)
        max_slew = 0.25  # per second
        u_slew_limited = np.clip(u, self.u_prev - max_slew * self.dt, 
                                 self.u_prev + max_slew * self.dt)
        
        # Track duty cycle
        duty_step = np.sum(np.abs(u_slew_limited - self.u_prev))
        self.duty_accumulator += duty_step
        
        # Reset duty accumulator periodically
        if t - self.last_duty_reset > 150.0:
            self.duty_accumulator = 0.0
            self.last_duty_reset = t
        
        # Adaptive integral reduction based on duty cycle
        if self.duty_accumulator > 0.012:
            self.integral_error *= 0.90
            self.duty_accumulator *= 0.80
        
        self.u_prev = u_slew_limited.copy()
        
        return u_slew_limited