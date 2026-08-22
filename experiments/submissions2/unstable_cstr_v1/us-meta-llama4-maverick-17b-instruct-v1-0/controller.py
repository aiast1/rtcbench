import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.Kp = -1.5  # Reduced proportional gain to be less aggressive
        self.Ki = -0.05  # Reduced integral gain to be less aggressive
        self.Kd = -0.2  # Reduced derivative gain to be less sensitive to noise
        self.integral = 0.0
        self.last_error = 0.0
        self.last_measurement = None
        self.last_output = np.array([300.0])  # Initial output

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_measurement = None
        self.last_output = np.array([300.0])  # Reset output

    def step(self, t, y, r, quality):
        if not quality[0]:
            return self.last_output  # If measurement is bad, hold last output

        measurement = y[0]
        setpoint = r[0]

        error = setpoint - measurement

        # Calculate derivative term
        if self.last_measurement is not None:
            derivative = (measurement - self.last_measurement) / self.brief.sample_time
        else:
            derivative = 0.0

        # Update integral term with anti-windup
        self.integral += error * self.brief.sample_time
        if self.last_output[0] >= 339.0 and error > 0:  # Softening the saturation limit check
            self.integral -= error * self.brief.sample_time  # Anti-windup when saturated high
        elif self.last_output[0] <= 271.0 and error < 0:  # Softening the saturation limit check
            self.integral -= error * self.brief.sample_time  # Anti-windup when saturated low

        # PID calculation
        output = self.Kp * error + self.Ki * self.integral - self.Kd * derivative

        # Clamp output to valid range and slew limit
        output = np.clip(output, 270.0, 340.0)
        output = np.clip(output, self.last_output[0] - 1.5 * self.brief.sample_time, self.last_output[0] + 1.5 * self.brief.sample_time)

        # Additional check to prevent too rapid changes
        if abs(output - self.last_output[0]) > 0.01715 / self.brief.sample_time:
            output = self.last_output[0] + np.sign(output - self.last_output[0]) * 0.01715 / self.brief.sample_time

        self.last_error = error
        self.last_measurement = measurement
        self.last_output = np.array([output])

        return self.last_output