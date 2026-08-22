import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.setpoint_schedule = [(0, 6.5), (350, 8.2), (800, 9.9), (1300, 6.5)]
        self.current_setpoint_index = 0
        self.current_setpoint_time = self.setpoint_schedule[self.current_setpoint_index][0]
        self.current_setpoint = self.setpoint_schedule[self.current_setpoint_index][1]
        self.integral_term = 0
        self.previous_error = 0
        self.previous_output = 14.22
        self.anti_windup = False
        self.output_rate = 0

    def reset(self):
        self.integral_term = 0
        self.previous_error = 0
        self.previous_output = 14.22
        self.anti_windup = False
        self.current_setpoint_index = 0
        self.current_setpoint_time = self.setpoint_schedule[self.current_setpoint_index][0]
        self.current_setpoint = self.setpoint_schedule[self.current_setpoint_index][1]
        self.output_rate = 0

    def step(self, t, y, r, quality):
        if not quality[0]:
            y[0] = self.previous_y0 if hasattr(self, 'previous_y0') else y[0]
        self.previous_y0 = y[0]

        # Update setpoint
        for i in range(len(self.setpoint_schedule) - 1, -1, -1):
            if t >= self.setpoint_schedule[i][0]:
                if i > self.current_setpoint_index:
                    self.current_setpoint_index = i
                    self.current_setpoint_time = self.setpoint_schedule[i][0]
                    self.current_setpoint = self.setpoint_schedule[i][1]
                break

        if self.setpoint_schedule[self.current_setpoint_index][0] + 200 < t and self.current_setpoint_index < len(self.setpoint_schedule) - 1 and self.current_setpoint_index == 2:
            if self.current_setpoint < self.setpoint_schedule[self.current_setpoint_index + 1][1]:
                self.current_setpoint = min(self.current_setpoint + 0.1 * self.sample_time / 200, self.setpoint_schedule[self.current_setpoint_index + 1][1])

        error = self.current_setpoint - y[0]

        # PI controller with anti-windup
        kp = 1.5
        ki = 0.05
        kd = 0.0
        self.integral_term += ki * error * self.sample_time
        derivative_term = kd * (error - self.previous_error) / self.sample_time if self.sample_time > 0 else 0

        output = kp * error + self.integral_term + derivative_term

        # Anti-windup
        if output > 30.0:
            output = 30.0
            self.integral_term -= ki * error * self.sample_time
        elif output < 0.0:
            output = 0.0
            self.integral_term -= ki * error * self.sample_time

        # Limit actuator slew rate
        slew_limit = 0.0041 / self.sample_time
        output_rate = (output - self.previous_output) / self.sample_time
        if abs(output_rate) > slew_limit:
            output = self.previous_output + np.sign(output_rate) * slew_limit * self.sample_time
        self.previous_output = output

        # Ensure tank level stays within safe limits
        if y[1] < 5.0:
            output = max(output, 14.22)
        elif y[1] > 30.0:
            output = min(output, 14.22)

        self.previous_error = error

        return np.array([output])