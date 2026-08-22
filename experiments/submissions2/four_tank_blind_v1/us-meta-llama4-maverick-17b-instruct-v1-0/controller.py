import numpy as np
from dataclasses import dataclass

@dataclass
class TaskBrief:
    sample_time: float

class Controller:
    def __init__(self, brief: TaskBrief):
        self.sample_time = brief.sample_time
        self.Kp = np.array([[0.25, 0.0], [0.0, 0.25]])  # Increased Kp to improve tracking
        self.Ki = np.array([[0.05, 0.0], [0.0, 0.05]])
        self.integral = np.zeros(2)
        self.last_error = np.zeros(2)
        self.last_u = np.array([3.0, 3.0])

    def reset(self):
        self.integral = np.zeros(2)
        self.last_error = np.zeros(2)
        self.last_u = np.array([3.0, 3.0])

    def step(self, t, y, r, quality):
        error = r - y
        delta_u = np.dot(self.Kp, error) + np.dot(self.Ki, self.integral)
        u = self.last_u + delta_u
        u = np.clip(u, 0.0, 10.0)
        delta_u = u - self.last_u
        if np.any(np.abs(delta_u) > 0.034):  
            u = self.last_u + np.sign(delta_u) * 0.034
        self.integral += error * self.sample_time
        anti_windup = np.clip(self.integral, -10.0, 10.0) - self.integral
        self.integral += anti_windup
        self.last_u = u
        self.last_error = error
        return u