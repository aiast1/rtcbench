import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.initial_actuator = 50.0
        self.Kp = 1.0  # Reduced proportional gain to reduce actuator duty
        self.Ki = 0.05  # Reduced integral gain to reduce actuator duty
        self.Kd = 0.0  
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_measurement = None
        self.previous_output = self.initial_actuator
        self.delta_output = 0.0
        self.previous_delta_output = 0.0

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_measurement = None
        self.previous_output = self.initial_actuator
        self.delta_output = 0.0
        self.previous_delta_output = 0.0

    def step(self, t, y, r, quality):
        measurement = y[0]
        setpoint = r[0]

        if np.isnan(setpoint):
            return np.array([self.previous_output])  

        if not quality[0]:
            measurement = self.previous_measurement

        error = setpoint - measurement

        self.integral += error * self.sample_time
        derivative = (error - self.previous_error) / self.sample_time if self.sample_time != 0 else 0

        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative + self.previous_output

        output = np.clip(output, 0.0, 100.0)

        # Anti-windup: stop integrating when output is saturated
        if output == 0.0 or output == 100.0:
            self.integral -= error * self.sample_time

        self.delta_output = output - self.previous_output
        self.delta_output = np.clip(self.delta_output, -1.0, 1.0)  # Limit the rate of change of output

        output = self.previous_output + self.delta_output

        self.previous_error = error
        self.previous_measurement = measurement
        self.previous_output = output
        self.previous_delta_output = self.delta_output

        return np.array([output])