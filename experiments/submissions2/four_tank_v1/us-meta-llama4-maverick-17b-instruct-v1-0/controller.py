import numpy as np
from dataclasses import dataclass

@dataclass
class TaskBrief:
    sample_time: float

class Controller:
    def __init__(self, brief: TaskBrief):
        self.sample_time = brief.sample_time
        self.Kp = np.array([[0.8, 0.1], [0.1, 0.8]])
        self.Ki = np.array([[0.08, 0.0], [0.0, 0.08]])
        self.Kd = np.array([[0.05, 0.0], [0.0, 0.05]])
        self.integral = np.zeros(2)
        self.last_error = np.zeros(2)
        self.last_last_error = np.zeros(2)
        self.last_u = np.array([3.0, 3.0])

    def reset(self):
        self.integral = np.zeros(2)
        self.last_error = np.zeros(2)
        self.last_last_error = np.zeros(2)
        self.last_u = np.array([3.0, 3.0])

    def step(self, t, y, r, quality):
        error = r - y
        valid = np.isfinite(r) & quality
        self.integral[valid] += error[valid] * self.sample_time
        self.integral = np.clip(self.integral, -10, 10)  # anti-windup
        derivative = (error - self.last_error) / self.sample_time
        u = self.last_u + np.dot(self.Kp, error) + np.dot(self.Ki, self.integral) + np.dot(self.Kd, derivative)
        u = np.clip(u, 0, 10)  # actuator limits
        du = u - self.last_u
        if np.any(np.abs(du) > 0.017):  # check duty limit
            u = self.last_u + np.clip(du, -0.017, 0.017)  # limit du
        self.last_last_error = self.last_error
        self.last_error = error
        self.last_u = u
        return u