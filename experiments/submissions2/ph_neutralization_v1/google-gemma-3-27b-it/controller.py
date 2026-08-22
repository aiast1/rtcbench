import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.pH_setpoint = 0.0
        self.last_output = 14.22  # Initial actuator value
        self.integral = 0.0
        self.anti_windup_gain = 1.0
        self.Kp = 1.0
        self.Ki = 0.01
        self.Kd = 0.1
        self.actuator_limit_low = 0.0
        self.actuator_limit_high = 30.0
        self.pH_limit_low = 4.0
        self.pH_limit_high = 10.5
        self.level_limit_low = 5.0
        self.level_limit_high = 30.0
        self.last_error = 0.0  # Initialize last_error

    def reset(self):
        self.pH_setpoint = 0.0
        self.integral = 0.0
        self.last_output = 14.22
        self.last_error = 0.0

    def step(self, t, y, r, quality):
        """t: seconds (float). y: numpy array of measured values, one per measurement.
        r: numpy array of setpoints, nan for channels that are not scored.
        quality: numpy bool array, False means that reading is stale/bad.
        Returns: numpy array of control values, one per actuator."""

        pH = y[0]
        level = y[1]

        # Setpoint handling
        if not np.isnan(r[0]):
            self.pH_setpoint = r[0]

        # Calculate error
        error = self.pH_setpoint - pH

        # Proportional term
        proportional = self.Kp * error

        # Integral term
        self.integral += self.Ki * error * self.sample_time

        # Anti-windup
        if self.integral > self.actuator_limit_high:
            self.integral = self.actuator_limit_high
        elif self.integral < self.actuator_limit_low:
            self.integral = self.actuator_limit_low

        # Derivative term
        if t > 0:
            derivative = self.Kd * (error - self.last_error) / self.sample_time
        else:
            derivative = 0.0

        # Calculate output
        output = proportional + self.integral + derivative

        # Saturate output
        output = np.clip(output, self.actuator_limit_low, self.actuator_limit_high)

        # Safety checks
        if pH < self.pH_limit_low or pH > self.pH_limit_high:
            return np.array([0.0])  # Zero out the control action
        if level < self.level_limit_low or level > self.level_limit_high:
            return np.array([0.0])

        # Actuator duty limit check
        duty_cycle = np.abs(output - self.last_output) / self.actuator_limit_high
        if duty_cycle > 0.0041:
            return np.array([self.last_output])

        self.last_output = output
        self.last_error = error

        return np.array([output])