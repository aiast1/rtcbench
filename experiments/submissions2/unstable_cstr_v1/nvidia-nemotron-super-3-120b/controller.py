import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.T_sp = 350.0
        self.T_min = 300.0
        self.T_max = 470.0
        self.coolant_min = 270.0
        self.coolant_max = 340.0
        self.max_actuator_travel_per_step = 0.01715
        self.max_slew_rate = 1.5
        self.stiction = 0.15

        # Reduced gains to minimize actuator travel and improve robustness
        self.Kp = -1.2
        self.Ki = -0.015
        self.Kd = -0.05

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = None
        self.prev_time = None
        self.prev_output = 300.0
        self._filtered_derivative = 0.0
        self.back_calc_gain = 0.05  # reduced back-calculation gain to limit integral windup correction

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = None
        self.prev_time = None
        self.prev_output = 300.0
        self._filtered_derivative = 0.0

    def step(self, t, y, r, quality):
        # Extract reactor temperature (TI-101)
        T_meas = y[0] if quality[0] else self.prev_measurement
        if T_meas is None:
            T_meas = 350.0
        if self.prev_measurement is None:
            self.prev_measurement = T_meas

        # Update setpoint from schedule
        if not np.isnan(r[0]):
            self.T_sp = r[0]

        error = self.T_sp - T_meas

        # Integral term
        self.integral += error * self.sample_time
        # Compute unsaturated output
        output_unsaturated = self.Kp * error + self.Ki * self.integral + self.Kd * self._filtered_derivative
        # Anti-windup via back-calculation
        output_clamped = np.clip(output_unsaturated, self.coolant_min, self.coolant_max)
        self.integral += self.back_calc_gain * (output_unsaturated - output_clamped) * self.sample_time
        output = self.Kp * error + self.Ki * self.integral + self.Kd * self._filtered_derivative

        # Actuator limits
        output = np.clip(output, self.coolant_min, self.coolant_max)

        # Slew rate limit
        max_delta = self.max_slew_rate * self.sample_time
        delta = output - self.prev_output
        if abs(delta) > max_delta:
            output = self.prev_output + np.sign(delta) * max_delta

        # Stiction: deadband to avoid chattering
        if abs(delta) < self.stiction:
            output = self.prev_output

        # Hard actuator travel limit per step
        delta = output - self.prev_output
        if abs(delta) > self.max_actuator_travel_per_step:
            output = self.prev_output + np.sign(delta) * self.max_actuator_travel_per_step

        # Final clamp
        output = np.clip(output, self.coolant_min, self.coolant_max)

        # Safety overrides
        if T_meas > 450.0:
            output = self.coolant_min
        elif T_meas < 310.0:
            output = self.coolant_max

        # Update state
        self.prev_error = error
        self.prev_measurement = T_meas
        self.prev_time = t
        self.prev_output = output

        return np.array([output])

    def _derivative(self, measurement, t):
        if self.prev_measurement is None or self.prev_time is None:
            return 0.0
        dt = t - self.prev_time
        if dt < 1e-6:
            return 0.0
        deriv = (measurement - self.prev_measurement) / dt
        alpha = 0.1  # increased filtering for derivative to reduce noise impact
        self._filtered_derivative = alpha * deriv + (1 - alpha) * self._filtered_derivative
        return self._filtered_derivative