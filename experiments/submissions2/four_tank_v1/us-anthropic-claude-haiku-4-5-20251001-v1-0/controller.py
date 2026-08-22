import numpy as np
from dataclasses import dataclass

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = brief.sample_time
        
        # Much more conservative PID gains to avoid overshooting unmeasured tanks
        # The safety violations suggest we're driving tanks 3 and 4 too high
        self.Kp = np.array([0.15, 0.15])
        self.Ki = np.array([0.03, 0.03])
        self.Kd = np.array([0.08, 0.08])
        
        # Integral state and anti-windup
        self.integral = np.array([0.0, 0.0])
        self.last_error = np.array([0.0, 0.0])
        self.last_output = np.array([3.0, 3.0])
        
        # Actuator limits
        self.u_min = np.array([0.0, 0.0])
        self.u_max = np.array([10.0, 10.0])
        
        # Duty cycle tracking (max 0.017 per step)
        self.max_duty_per_step = 0.17  # 0.17 V per 2s step
        
        # Low-pass filter for derivative (reduce noise sensitivity)
        self.alpha_d = 0.2
        self.d_filtered = np.array([0.0, 0.0])
        
        # Track measurement history for better state estimation
        self.y_prev = np.array([0.0, 0.0])
        self.meas_count = 0
        
    def reset(self):
        self.integral = np.array([0.0, 0.0])
        self.last_error = np.array([0.0, 0.0])
        self.last_output = np.array([3.0, 3.0])
        self.d_filtered = np.array([0.0, 0.0])
        self.y_prev = np.array([0.0, 0.0])
        self.meas_count = 0
    
    def step(self, t, y, r, quality):
        # Build error signal, being very conservative with bad measurements
        error = np.zeros(2)
        for i in range(2):
            if quality[i] and not np.isnan(r[i]):
                error[i] = r[i] - y[i]
                self.y_prev[i] = y[i]
            else:
                # If measurement is bad, use last known error but decay it
                error[i] = self.last_error[i] * 0.5
        
        # Proportional term - very conservative
        P = self.Kp * error
        
        # Integral term with aggressive anti-windup
        # Only integrate small errors to avoid windup
        integral_update = self.Ki * error * self.dt
        
        # Limit integral update magnitude
        integral_update = np.clip(integral_update, -0.05, 0.05)
        self.integral += integral_update
        
        # Clamp integral state itself to prevent excessive buildup
        self.integral = np.clip(self.integral, -0.5, 0.5)
        
        I = self.integral
        
        # Derivative term with aggressive low-pass filtering
        d_raw = np.zeros(2)
        for i in range(2):
            if quality[i]:
                d_raw[i] = (error[i] - self.last_error[i]) / self.dt
        
        self.d_filtered = self.alpha_d * d_raw + (1.0 - self.alpha_d) * self.d_filtered
        # Limit derivative action
        self.d_filtered = np.clip(self.d_filtered, -2.0, 2.0)
        D = self.Kd * self.d_filtered
        
        # Compute raw control output
        u_raw = P + I + D
        
        # Apply rate limiting to respect duty cycle constraint
        # Be very conservative here - limit changes significantly
        delta_u = u_raw - self.last_output
        max_change = min(self.max_duty_per_step, 0.1)  # Further limit rate of change
        delta_u = np.clip(delta_u, -max_change, max_change)
        u_limited = self.last_output + delta_u
        
        # Apply actuator saturation
        u = np.clip(u_limited, self.u_min, self.u_max)
        
        # Additional safety: if we're at a setpoint and error is small, hold steady
        for i in range(2):
            if not np.isnan(r[i]) and abs(error[i]) < 0.3:
                # Near setpoint - reduce aggressiveness
                u[i] = 0.9 * u[i] + 0.1 * self.last_output[i]
        
        # Update state for next iteration
        self.last_error = error.copy()
        self.last_output = u.copy()
        
        return u