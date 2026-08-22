import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.T_setpoint = 350.0
        self.Kp = 1.2
        self.Ki = 0.04
        self.Kd = 0.01
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_T = None
        self.actuator_limits = np.array([270.0, 340.0])
        self.actuator_previous = 300.0
        self.duty_limit = 0.01715 / self.sample_time

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_T = None
        self.actuator_previous = 300.0

    def step(self, t, y, r, quality):
        T = y[0]
        if not quality[0]:
            T = self.previous_T
        self.previous_T = T

        if np.isnan(r[0]):
            r[0] = self.T_setpoint

        error = r[0] - T

        if self.previous_T is not None:
            derivative = (error - self.previous_error) / self.sample_time
        else:
            derivative = 0.0

        self.integral += error * self.sample_time
        self.integral = np.clip(self.integral, -2.0, 2.0)  # anti-windup

        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        du = (u - self.actuator_previous) / self.sample_time
        if abs(du) > self.duty_limit:
            u = self.actuator_previous + np.sign(du) * self.duty_limit * self.sample_time
        u = np.clip(u, self.actuator_limits[0], self.actuator_limits[1])

        self.actuator_previous = u
        self.previous_error = error

        return np.array([u])