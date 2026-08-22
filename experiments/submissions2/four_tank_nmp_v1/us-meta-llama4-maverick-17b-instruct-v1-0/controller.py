import numpy as np
from dataclasses import dataclass

@dataclass
class TaskBrief:
    sample_time: float

class Controller:
    def __init__(self, brief: TaskBrief):
        self.sample_time = brief.sample_time
        self.Kp = np.array([[0.3, 0.0], [0.0, 0.3]])
        self.Ki = np.array([[0.02, 0.0], [0.0, 0.02]])
        self.Kd = np.array([[0.0, 0.0], [0.0, 0.0]])
        self.integral = np.zeros(2)
        self.previous_error = np.zeros(2)
        self.previous_measurement = np.zeros(2)
        self.u = np.array([3.0, 3.0])  # initialize u to the starting condition

    def reset(self):
        self.integral = np.zeros(2)
        self.previous_error = np.zeros(2)
        self.previous_measurement = np.zeros(2)
        self.u = np.array([3.0, 3.0])  # reset u to the starting condition

    def step(self, t, y, r, quality):
        if not np.all(quality):
            return self.u  # hold the last control output if any measurement is bad
        error = r - y
        delta_measurement = (y - self.previous_measurement) / self.sample_time
        self.integral += error * self.sample_time
        self.integral = np.clip(self.integral, -5, 5)  # anti-windup
        delta_u = np.dot(self.Kp, error) + np.dot(self.Ki, self.integral) - np.dot(self.Kd, delta_measurement)
        delta_u = np.clip(delta_u, -0.0074, 0.0074)  # limit actuator slew rate
        self.u += delta_u
        self.u = np.clip(self.u, 0, 10)  # actuator limits
        self.previous_error = error
        self.previous_measurement = y
        return self.u