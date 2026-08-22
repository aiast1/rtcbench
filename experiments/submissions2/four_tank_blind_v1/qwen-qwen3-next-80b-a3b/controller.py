import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.reset()

    def reset(self):
        # Initialize controller states
        self.integral_error = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.prev_control = np.array([3.0, 3.0])  # Start at initial operating point
        self.prev_y = np.zeros(2)
        self.kp = np.array([0.8, 0.8])            # Proportional gain
        self.ki = np.array([0.02, 0.02])          # Integral gain (conservative)
        self.kd = np.array([0.1, 0.1])            # Derivative gain (low for noise rejection)
        self.sat_limit = 10.0
        self.min_output = 0.0
        self.sample_time = self.brief.sample_time
        self.last_time = 0.0
        self.antwindup_factor = 0.8  # Anti-windup factor to prevent integrator windup

    def step(self, t, y, r, quality):
        # Ensure we have valid measurements (handle bad quality)
        y_clean = np.where(quality, y, self.prev_y)
        self.prev_y = y_clean.copy()
        
        # Calculate error
        error = r - y_clean
        
        # Update integral term with anti-windup
        dt = t - self.last_time if self.last_time > 0 else self.sample_time
        dt = max(dt, 1e-6)  # Avoid division by zero
        
        # Anti-windup: limit integral action when actuators are saturated
        u_prev = self.prev_control
        u_sat = np.clip(u_prev, self.min_output, self.sat_limit)
        
        # Calculate integral contribution with anti-windup
        for i in range(2):
            if np.abs(u_sat[i] - self.min_output) < 1e-3 or np.abs(u_sat[i] - self.sat_limit) < 1e-3:
                # Actuator saturated, reduce integral contribution
                self.integral_error[i] += self.antwindup_factor * error[i] * dt
            else:
                self.integral_error[i] += error[i] * dt
        
        # Derivative term (filtered to reduce noise impact)
        if self.last_time > 0:
            d_error = (error - self.prev_error) / dt
        else:
            d_error = np.zeros(2)
        
        # PID control law
        u = (self.kp * error + 
             self.ki * self.integral_error + 
             self.kd * d_error)
        
        # Apply anti-windup: clamp control output
        u_clamped = np.clip(u, self.min_output, self.sat_limit)
        
        # Slew rate limiting (prevent abrupt changes)
        max_delta = 0.5  # Max change per step (V) - conservative for stiction and dynamics
        u_final = np.clip(u_clamped, 
                          self.prev_control - max_delta, 
                          self.prev_control + max_delta)
        
        # Ensure output stays within hard limits
        u_final = np.clip(u_final, self.min_output, self.sat_limit)
        
        # Update previous states
        self.prev_control = u_final.copy()
        self.prev_error = error.copy()
        self.last_time = t
        
        return u_final