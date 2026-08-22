import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.actuator_limits = np.array([270.0, 340.0])
        self.reactor_temp_index = 0
        self.jacket_temp_index = 1
        self.actuator_index = 0

        # PID gains (tuned for robustness) - Reduced Ki to reduce integral windup
        self.Kp = 0.8
        self.Ki = 0.005  # Reduced Ki
        self.Kd = 0.1

        # Anti-windup gain
        self.anti_windup_gain = 0.5

        # Integrator clamp - Increased clamp
        self.integrator_clamp = 200.0

        # Previous error and integrator
        self.previous_error = 0.0
        self.integrator = 0.0

        # Actuator duty limit
        self.actuator_duty_limit = 0.01715

        # Previous actuator output
        self.previous_actuator_output = 300.0

    def reset(self):
        # Reset integrator and previous error
        self.integrator = 0.0
        self.previous_error = 0.0
        self.previous_actuator_output = 300.0

    def step(self, t, y, r, quality):
        """t: seconds (float). y: numpy array of measured values, one per measurement.
        r: numpy array of setpoints, nan for channels that are not scored.
        quality: numpy bool array, False means that reading is stale/bad.
        Returns: numpy array of control values, one per actuator."""

        if not quality[self.reactor_temp_index]:
            # If reactor temperature is not valid, maintain last output
            return np.array([self.previous_actuator_output])

        # Calculate error
        error = r[self.reactor_temp_index] - y[self.reactor_temp_index]

        # Proportional term
        P = self.Kp * error

        # Integral term
        self.integrator += self.Ki * error * self.sample_time
        self.integrator = np.clip(self.integrator, -self.integrator_clamp, self.integrator_clamp)
        I = self.integrator

        # Derivative term
        D = self.Kd * (error - self.previous_error) / self.sample_time

        # Calculate control output
        u = P + I + D

        # Anti-windup - More aggressive anti-windup
        if u > self.actuator_limits[1]:
            self.integrator -= (u - self.actuator_limits[1]) * self.anti_windup_gain * 2 # Increased anti-windup gain
            u = self.actuator_limits[1]
        elif u < self.actuator_limits[0]:
            self.integrator -= (u - self.actuator_limits[0]) * self.anti_windup_gain * 2 # Increased anti-windup gain
            u = self.actuator_limits[0]

        # Actuator saturation and duty cycle limit
        u = np.clip(u, self.actuator_limits[0], self.actuator_limits[1])

        # Duty cycle check - More conservative duty cycle check
        duty_cycle = np.abs(u - self.previous_actuator_output) / self.sample_time
        if duty_cycle > self.actuator_duty_limit * 0.8: # Reduced duty cycle threshold
            u = self.previous_actuator_output

        # Update previous error and actuator output
        self.previous_error = error
        self.previous_actuator_output = u

        # Safety checks
        if y[self.reactor_temp_index] > 470.0 or y[self.reactor_temp_index] < 300.0:
            return np.array([0.0])  # Zero out control if safety limits are violated

        return np.array([u])