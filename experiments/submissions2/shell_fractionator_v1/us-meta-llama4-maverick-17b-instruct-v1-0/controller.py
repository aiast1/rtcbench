import numpy as np
from dataclasses import dataclass

@dataclass
class TaskBrief:
    sample_time: float

class Controller:
    def __init__(self, brief: TaskBrief):
        self.sample_time = brief.sample_time
        self.Kp = np.array([[0.02, 0.005, 0.002], [0.01, 0.02, 0.005], [0.002, 0.002, 0.02]])  # Further reduced PID gains
        self.Ki = np.array([[0.0004, 0.0001, 0.00005], [0.0002, 0.0004, 0.0001], [0.00005, 0.00005, 0.0004]])
        self.Kd = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # No derivative action due to noise
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.previous_u = np.zeros(3)

    def reset(self):
        self.integral = np.zeros(3)
        self.previous_error = np.zeros(3)
        self.previous_u = np.zeros(3)

    def step(self, t, y, r, quality):
        error = r - y
        error[~np.isfinite(r)] = 0  # Ignore non-scoring channels
        self.integral += error * self.sample_time
        derivative = (error - self.previous_error) / self.sample_time
        u = self.previous_u + np.dot(self.Kp, error) + np.dot(self.Ki, self.integral) + np.dot(self.Kd, derivative)
        u = np.clip(u, -0.5, 0.5)  # Enforce actuator limits
        du = u - self.previous_u
        if np.any(np.abs(du) > 0.0112 / self.sample_time):  # Check for actuator duty limit
            u = self.previous_u + np.clip(du, -0.0112 / self.sample_time, 0.0112 / self.sample_time)
        # Additional safety check for bottoms reflux temperature
        if y[2] < -0.5:
            u[2] += 0.01  # Increase bottoms reflux duty to prevent safety violation
            u = np.clip(u, -0.5, 0.5)
        self.integral -= np.dot(self.Ki, (u - np.clip(self.previous_u + np.dot(self.Kp, error), -0.5, 0.5))) * self.sample_time  # Anti-windup
        self.previous_error = error
        self.previous_u = u
        return u