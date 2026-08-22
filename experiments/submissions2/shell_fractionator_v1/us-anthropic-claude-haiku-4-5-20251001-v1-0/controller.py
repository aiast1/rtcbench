import numpy as np
from dataclasses import dataclass

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = brief.sample_time
        
        # Nominal model from hint
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
        self.LD = np.array([
            [27.0, 27.0],
            [15.0, 15.0],
            [27.0, 32.0]
        ])
        
        # Improved PID tuning - more aggressive on tracking, careful on effort
        # Increase proportional gain for faster response
        self.Kp = np.array([0.25, 0.20, 0.12])
        # Increase integral gain for better steady-state tracking
        self.Ki = np.array([0.015, 0.012, 0.008])
        # Moderate derivative for noise rejection
        self.Kd = np.array([0.8, 0.6, 0.3])
        
        # Actuator limits
        self.u_min = np.array([-0.5, -0.5, -0.5])
        self.u_max = np.array([0.5, 0.5, 0.5])
        
        # State tracking
        self.integral_error = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.prev_u = np.zeros(3)
        self.u_saturated = np.zeros(3)
        
        # Low-pass filter for derivative (reduce noise sensitivity)
        self.alpha_d = 0.25
        self.filtered_error_deriv = np.zeros(3)
        
        # Dead-time compensation: track delayed measurements
        self.max_delay_steps = 30
        self.y_history = np.zeros((self.max_delay_steps, 3))
        self.step_count = 0
        
    def reset(self):
        self.integral_error = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.prev_u = np.zeros(3)
        self.u_saturated = np.zeros(3)
        self.filtered_error_deriv = np.zeros(3)
        self.y_history = np.zeros((self.max_delay_steps, 3))
        self.step_count = 0
    
    def step(self, t, y, r, quality):
        # Store measurement history for dead-time awareness
        self.y_history[self.step_count % self.max_delay_steps] = y
        self.step_count += 1
        
        # Compute errors only for scored channels with valid setpoints
        error = np.zeros(3)
        for i in range(3):
            if not np.isnan(r[i]) and quality[i]:
                error[i] = r[i] - y[i]
            elif not np.isnan(r[i]):
                # Use last known error if measurement is bad
                error[i] = self.prev_error[i]
        
        # Anti-windup: only integrate when not saturated
        for i in range(3):
            if not np.isnan(r[i]):
                # Conditional integration with back-calculation
                if abs(self.u_saturated[i]) < 0.99 * self.u_max[i]:
                    self.integral_error[i] += error[i] * self.dt
                else:
                    # Back-calculation anti-windup
                    self.integral_error[i] *= 0.95
                
                # Limit integral to prevent excessive windup
                self.integral_error[i] = np.clip(self.integral_error[i], -0.25, 0.25)
        
        # Derivative action with low-pass filtering
        error_deriv = (error - self.prev_error) / self.dt
        self.filtered_error_deriv = (
            self.alpha_d * error_deriv + 
            (1.0 - self.alpha_d) * self.filtered_error_deriv
        )
        
        # PID control law
        u = np.zeros(3)
        for i in range(3):
            if not np.isnan(r[i]):
                p_term = self.Kp[i] * error[i]
                i_term = self.Ki[i] * self.integral_error[i]
                d_term = self.Kd[i] * self.filtered_error_deriv[i]
                u[i] = p_term + i_term + d_term
            else:
                # No setpoint: maintain current position
                u[i] = self.prev_u[i]
        
        # Saturation with anti-windup flag
        u_clipped = np.clip(u, self.u_min, self.u_max)
        self.u_saturated = u_clipped
        
        # Slew rate limiting - more aggressive to reduce duty violations
        max_slew = 0.12  # Increased from 0.08 for faster response
        u_limited = np.zeros(3)
        for i in range(3):
            delta = u_clipped[i] - self.prev_u[i]
            delta = np.clip(delta, -max_slew * self.dt, max_slew * self.dt)
            u_limited[i] = self.prev_u[i] + delta
        
        u_limited = np.clip(u_limited, self.u_min, self.u_max)
        
        # Update state
        self.prev_error = error.copy()
        self.prev_u = u_limited.copy()
        
        return u_limited