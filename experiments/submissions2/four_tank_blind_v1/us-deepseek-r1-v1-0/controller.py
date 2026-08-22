import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.actuator_limits = (0.0, 10.0)
        
        # Aggressively tuned parameters with safety margins
        self.Kp1 = 0.55
        self.Ti1 = 30.0
        self.Ki1 = self.Kp1 / self.Ti1
        
        self.Kp2 = 0.55
        self.Ti2 = 30.0
        self.Ki2 = self.Kp2 / self.Ti2
        
        # Strong anti-windup compensation
        self.Kaw = 3.0
        
        # Faster setpoint tracking filter
        self.tau_filter = 10.0
        
        # Output rate limiting
        self.max_delta_u = 0.5  # V per step

    def reset(self):
        self.integral1 = 0.0
        self.integral2 = 0.0
        self.last_good_y1 = 12.26
        self.last_good_y2 = 12.78
        self.filtered_r1 = 12.26
        self.filtered_r2 = 12.78
        self.last_u = np.array([3.0, 3.0])
        self.first_step = True

    def step(self, t, y, r, quality):
        # Adaptive setpoint filtering
        alpha = self.sample_time / (self.tau_filter + self.sample_time)
        if not np.isnan(r[0]):
            self.filtered_r1 += alpha * (r[0] - self.filtered_r1)
        if not np.isnan(r[1]):
            self.filtered_r2 += alpha * (r[1] - self.filtered_r2)

        # Measurement validation and hold
        current_y1 = y[0] if quality[0] else self.last_good_y1
        self.last_good_y1 = current_y1
        
        current_y2 = y[1] if quality[1] else self.last_good_y2
        self.last_good_y2 = current_y2

        # Error calculation with split paths
        error_p1 = self.filtered_r1 - current_y1
        error_i1 = r[0] - current_y1 if not np.isnan(r[0]) else 0.0
        
        error_p2 = self.filtered_r2 - current_y2
        error_i2 = r[1] - current_y2 if not np.isnan(r[1]) else 0.0

        # Bumpless initialization
        if self.first_step:
            self.integral1 = (3.0 - self.Kp1 * error_p1) / self.Ki1
            self.integral2 = (3.0 - self.Kp2 * error_p2) / self.Ki2
            self.first_step = False

        # Control computation with clamping
        u1_unclamped = self.Kp1 * error_p1 + self.Ki1 * self.integral1
        u2_unclamped = self.Kp2 * error_p2 + self.Ki2 * self.integral2
        
        # Rate limiting
        delta_u1 = np.clip(u1_unclamped - self.last_u[0], -self.max_delta_u, self.max_delta_u)
        delta_u2 = np.clip(u2_unclamped - self.last_u[1], -self.max_delta_u, self.max_delta_u)
        u1 = np.clip(self.last_u[0] + delta_u1, *self.actuator_limits)
        u2 = np.clip(self.last_u[1] + delta_u2, *self.actuator_limits)
        self.last_u[:] = [u1, u2]

        # Anti-windup with dynamic clamping
        windup_correction1 = (u1 - u1_unclamped) * self.Kaw * self.sample_time
        windup_correction2 = (u2 - u2_unclamped) * self.Kaw * self.sample_time
        
        self.integral1 += self.Ki1 * error_i1 * self.sample_time + windup_correction1
        self.integral2 += self.Ki2 * error_i2 * self.sample_time + windup_correction2

        return np.array([u1, u2])