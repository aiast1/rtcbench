import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.Kp = np.array([[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]])
        self.Ki = np.array([[0.001, 0.0, 0.0], [0.0, 0.001, 0.0], [0.0, 0.0, 0.001]])
        self.integral = np.zeros(3)
        self.last_error = np.zeros(3)
        self.last_u = np.zeros(3)

    def reset(self):
        self.integral = np.zeros(3)
        self.last_error = np.zeros(3)
        self.last_u = np.zeros(3)

    def step(self, t, y, r, quality):
        valid_channels = np.isfinite(r) & quality
        error = np.zeros_like(r)
        error[valid_channels] = r[valid_channels] - y[valid_channels]
        delta_u = self.Kp @ error + np.diag(self.Ki) * self.integral
        u = self.last_u + delta_u
        u = np.clip(u, -0.5, 0.5)
        delta_u = u - self.last_u
        if np.any(np.abs(delta_u) > 0.016):
            u = self.last_u
            delta_u = np.zeros(3)
        self.integral += error * self.sample_time
        anti_windup = (np.clip(u, -0.5, 0.5) - u) / (np.max(self.Ki) * self.sample_time + 1e-6)
        self.integral += anti_windup
        self.integral[~valid_channels] = 0  # prevent integral windup on invalid channels
        self.last_u = u
        return u