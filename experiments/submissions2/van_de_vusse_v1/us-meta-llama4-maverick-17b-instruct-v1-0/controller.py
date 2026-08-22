import numpy as np
from dataclasses import dataclass

@dataclass
class TaskBrief:
    sample_time: float

class Controller:
    def __init__(self, brief: TaskBrief):
        self.sample_time = brief.sample_time
        self.Kp_CB = np.array([[-0.1], [0]])  
        self.Ki_CB = np.array([[-0.01], [0]])  
        self.Kp_T = np.array([[0], [-5]])  
        self.Ki_T = np.array([[0], [-0.5]])  
        self.integral_CB = np.zeros((2, 1))
        self.integral_T = np.zeros((2, 1))
        self.last_y = None
        self.last_r = None
        self.last_u = np.array([14.19, -1113.5])

    def reset(self):
        self.integral_CB = np.zeros((2, 1))
        self.integral_T = np.zeros((2, 1))
        self.last_y = None
        self.last_r = None
        self.last_u = np.array([14.19, -1113.5])

    def step(self, t, y, r, quality):
        if self.last_y is None:
            self.last_y = y[:, None]
            self.last_r = r[:, None]
            return self.last_u  

        error_CB = r[0] - y[0]
        error_T = r[1] - y[1]

        self.integral_CB += error_CB * self.sample_time
        self.integral_T += error_T * self.sample_time

        delta_u_CB = self.Kp_CB * error_CB + self.Ki_CB * self.integral_CB
        delta_u_T = self.Kp_T * error_T + self.Ki_T * self.integral_T

        delta_u = np.array([delta_u_CB[0, 0], delta_u_T[1, 0]])

        u = self.last_u + delta_u * self.sample_time

        u[0] = np.clip(u[0], 3.0, 35.0)
        u[1] = np.clip(u[1], -9000.0, 0.0)

        # Anti-windup for integral terms
        if u[0] >= 35.0 or u[0] <= 3.0:
            self.integral_CB -= error_CB * self.sample_time
        if u[1] >= 0.0 or u[1] <= -9000.0:
            self.integral_T -= error_T * self.sample_time

        # Limit actuator slew rate
        delta_u_actual = u - self.last_u
        delta_u_max = np.array([0.1, 100]) * self.sample_time
        delta_u_actual = np.clip(delta_u_actual, -delta_u_max, delta_u_max)
        u = self.last_u + delta_u_actual

        self.last_y = y[:, None]
        self.last_r = r[:, None]
        self.last_u = u

        return u