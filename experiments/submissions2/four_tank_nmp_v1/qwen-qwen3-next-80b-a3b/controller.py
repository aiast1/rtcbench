import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.reset()

    def reset(self):
        self.integral_error = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.prev_control = np.array([3.0, 3.0])
        self.prev_y = np.zeros(2)
        self.last_t = 0.0
        self.kp = np.array([0.4, 0.4])
        self.ki = np.array([0.005, 0.005])
        self.kd = np.array([0.02, 0.02])
        self.sat_limit = 10.0
        self.min_output = 0.0
        self.sample_time = self.brief.sample_time
        self.antiwindup_factor = 1.0
        self.slew_limit = (self.sat_limit - self.min_output) / self.sample_time * 0.3
        self.safety_margin = 3.0  # cm buffer below overflow
        self.noise_filter_alpha = 0.1  # stronger filtering for noisy measurements
        self.output_limit = self.sat_limit - 1.0  # hard buffer to prevent saturation
        self.integral_clamp = 5.0  # max integral contribution

    def step(self, t, y, r, quality):
        # Strong exponential smoothing on measurements
        if self.last_t == 0.0:
            y_filtered = y.copy()
        else:
            y_filtered = self.noise_filter_alpha * y + (1 - self.noise_filter_alpha) * self.prev_y
        
        # Handle bad measurements
        y_filtered = np.where(quality, y_filtered, self.prev_y)
        
        # Apply safety margin: clamp to stay well below overflow
        y_filtered = np.clip(y_filtered, 0.0, 20.0 - self.safety_margin)
        
        # Calculate error
        error = r - y_filtered
        
        # Integral action with aggressive anti-windup
        dt = t - self.last_t
        if dt <= 0 or dt > 2.5:
            dt = self.sample_time
        
        self.integral_error += error * dt
        
        # Clamp integral to prevent windup
        self.integral_error = np.clip(self.integral_error, -self.integral_clamp, self.integral_clamp)
        
        # Anti-windup: disable integral if actuators are near limit
        for i in range(2):
            if (self.prev_control[i] >= self.output_limit and error[i] > 0) or \
               (self.prev_control[i] <= self.min_output + 0.1 and error[i] < 0):
                self.integral_error[i] -= error[i] * dt  # subtract error to unwind
        
        # Calculate control output
        control = self.kp * error + self.ki * self.integral_error + self.kd * (error - self.prev_error) / dt
        
        # Slew rate limiting (very conservative)
        control = np.clip(control, 
                         self.prev_control - self.slew_limit * dt,
                         self.prev_control + self.slew_limit * dt)
        
        # Hard actuator limits with safety buffer
        control = np.clip(control, self.min_output, self.output_limit)
        
        # Bumpless start: first output must be 3.0, 3.0
        if self.last_t == 0.0:
            control = np.array([3.0, 3.0])
        
        # Update history
        self.prev_error = error.copy()
        self.prev_y = y_filtered.copy()
        self.prev_control = control.copy()
        self.last_t = t
        
        return control