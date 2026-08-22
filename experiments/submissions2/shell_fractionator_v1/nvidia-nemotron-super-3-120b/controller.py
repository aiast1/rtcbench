import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = brief.sample_time
        self.n_y = 3  # AI-101, AI-102, TI-103
        self.n_u = 3  # FCV-201, FCV-202, FCV-203
        
        # Actuator limits
        self.u_min = np.array([-0.5, -0.5, -0.5])
        self.u_max = np.array([0.5, 0.5, 0.5])
        
        # Safety constraint: y3 (bottoms reflux temp) must be >= -0.5
        self.y3_min = -0.5
        
        # Controller tuning (optimized for performance while maintaining robustness)
        self.Kp = np.array([0.07, 0.06, 0.035])   # Slightly increased for better response
        self.Ki = np.array([0.0012, 0.0010, 0.0007])  # Adjusted for faster settling
        self.Kd = np.array([0.0, 0.0, 0.0])      # No derivative due to noise and delay
        
        # State
        self.x_int = np.zeros(self.n_y)      # Integral states
        self.u_prev = np.zeros(self.n_u)     # Previous actuator command
        self.last_good_y = np.zeros(self.n_y) # Last good measurement
        self.integral_clamp = np.array([1.8, 1.8, 1.8])  # Anti-windup clamp
        
    def reset(self):
        self.x_int = np.zeros(self.n_y)
        self.u_prev = np.zeros(self.n_u)
        self.last_good_y = np.zeros(self.n_y)
        
    def step(self, t, y, r, quality):
        # Handle bad measurements: use last good value
        y_used = np.where(quality, y, self.last_good_y)
        self.last_good_y = np.where(quality, y, self.last_good_y)
        
        # Error signal (only for scored channels)
        error = r - y_used
        # For non-scored channels (NaN in r), set error to 0
        error = np.where(np.isnan(error), 0.0, error)
        
        # Integral update with anti-windup (pre-update clamping)
        self.x_int += error * self.dt
        # Anti-windup: clamp integral based on actuator saturation
        for i in range(self.n_y):
            # If actuator is saturated, limit integral growth
            if self.u_prev[i] <= self.u_min[i] + 1e-3:
                self.x_int[i] = min(self.x_int[i], self.integral_clamp[i])
            elif self.u_prev[i] >= self.u_max[i] - 1e-3:
                self.x_int[i] = max(self.x_int[i], -self.integral_clamp[i])
            else:
                # Not saturated, allow integral to move freely within bounds
                self.x_int[i] = np.clip(self.x_int[i], -self.integral_clamp[i], self.integral_clamp[i])
        
        # PID control law (no derivative due to noise and delay)
        u_pid = self.Kp * error + self.Ki * self.x_int
        
        # Safety override: if y3 (bottoms reflux temp) is too low, increase FCV-203 to add reflux
        # y3 is index 2, actuator 2 is FCV-203 (bottoms reflux duty)
        if y_used[2] < self.y3_min:
            # Proportional safety action: increase reflux to raise temperature
            safety_gain = 0.25  # Balanced gain for safety response
            safety_action = safety_gain * (self.y3_min - y_used[2])
            u_pid[2] += safety_action
        
        # Anti-windup: back-calculation if actuator saturated
        u_raw = u_pid
        u_clipped = np.clip(u_raw, self.u_min, self.u_max)
        
        # Back-calculation anti-windup: adjust integral if saturated
        for i in range(self.n_u):
            if u_raw[i] != u_clipped[i]:  # Saturated
                # Back-calculate integral to prevent windup
                self.x_int[i] -= (u_raw[i] - u_clipped[i]) / (self.Ki[i] + 1e-8) * self.dt * 0.07
        
        # Final actuator command
        u = u_clipped
        
        # Rate limiting (slew rate) - tuned to minimize effort while avoiding chattering
        max_delta = 0.012  # Slightly increased to allow faster response while staying near duty limit
        delta = u - self.u_prev
        delta = np.clip(delta, -max_delta, max_delta)
        u = self.u_prev + delta
        u = np.clip(u, self.u_min, self.u_max)  # Re-clip after rate limit
        
        # Update state
        self.u_prev = u
        
        return u