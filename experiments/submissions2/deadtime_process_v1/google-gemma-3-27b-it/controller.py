import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.actuator_count = 1
        self.measurement_count = 1

        # PID gains (tuned for robustness) - reduced Ki and Kd
        self.Kp = 0.4
        self.Ki = 0.005  # Reduced integral gain
        self.Kd = 0.005  # Reduced derivative gain

        # Anti-windup gain
        self.anti_windup_gain = 0.1

        # Integrator state
        self.integrator = 0.0

        # Previous error
        self.previous_error = 0.0

        # Actuator limits
        self.actuator_min = 0.0
        self.actuator_max = 100.0

        # Safety margin
        self.safety_margin = 10.0  # Degrees C

    def reset(self):
        self.integrator = 0.0
        self.previous_error = 0.0

    def step(self, t, y, r, quality):
        """t: seconds (float). y: numpy array of measured values, one per measurement.
        r: numpy array of setpoints, nan for channels that are not scored.
        quality: numpy bool array, False means that reading is stale/bad.
        Returns: numpy array of control values, one per actuator."""

        # Get the temperature measurement and setpoint
        temperature = y[0]
        setpoint = r[0]

        # Check data quality
        if not quality[0]:
            return np.array([50.0])  # Hold last output if data is bad

        # Calculate the error
        error = setpoint - temperature

        # Calculate the proportional term
        proportional = self.Kp * error

        # Calculate the integral term
        self.integrator += self.Ki * error * self.sample_time

        # Anti-windup - more aggressive
        if self.integrator > self.actuator_max:
            self.integrator = self.actuator_max
        elif self.integrator < self.actuator_min:
            self.integrator = self.actuator_min

        # Calculate the derivative term
        derivative = self.Kd * (error - self.previous_error) / self.sample_time

        # Calculate the control output
        output = proportional + self.integrator + derivative

        # Safety check - limit output based on estimated temperature
        estimated_temperature = temperature + (output / 100.0) * (90.0 - temperature)  # Rough estimate
        if estimated_temperature > 90.0 + self.safety_margin:
            output = np.clip(output, self.actuator_min, (90.0 - temperature) / (90.0 - temperature) * 100.0)
        
        # Saturate the output
        output = np.clip(output, self.actuator_min, self.actuator_max)

        # Actuator duty limit check
        if output > 100.0:
            output = 100.0

        # Update previous error
        self.previous_error = error

        return np.array([output])