import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.Kp = 0.5
        self.Ki = 0.05 / self.sample_time
        self.Kd = 0.0
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_TT401 = None
        self.setpoint_schedule = [(0, 55.0), (500, 68.0), (1600, 48.0), (2600, 60.0), (3700, 65.0)]
        self.current_setpoint_index = 0
        self.output = 50.0  # Initialize output to starting value

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_TT401 = None
        self.current_setpoint_index = 0
        self.output = 50.0  # Reset output to starting value

    def step(self, t, y, r, quality):
        TT401 = y[0]
        if not quality[0]:
            TT401 = self.previous_TT401
        self.previous_TT401 = TT401

        # Update setpoint
        for i in range(len(self.setpoint_schedule) - 1, -1, -1):
            if self.setpoint_schedule[i][0] <= t:
                setpoint = self.setpoint_schedule[i][1]
                break

        error = setpoint - TT401

        # PI control with anti-windup and slew rate limit
        self.integral += self.Ki * error * self.sample_time
        self.integral = np.clip(self.integral, 0.0, 100.0)

        # Limit derivative action
        derivative = 0.0
        if self.previous_error is not None:
            derivative = self.Kd * (error - self.previous_error) / self.sample_time
        self.previous_error = error

        # Calculate control output with slew rate limit
        new_output = self.output + np.clip(self.Kp * error + derivative, -5.0, 5.0)
        self.output = np.clip(new_output, 0.0, 100.0)

        # Prevent actuator duty limit exceedance
        if abs(self.output - 50.0) > 10.0:
            self.output = 50.0 + np.sign(self.output - 50.0) * 10.0

        return np.array([self.output])