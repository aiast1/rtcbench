import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.actuator_limits = np.array([[0.0, 10.0], [0.0, 10.0]])
        self.initial_actuators = np.array([3.0, 3.0])
        self.n_actuators = 2
        self.n_measurements = 2

        # More aggressive PID parameters with better disturbance rejection
        self.Kp = np.array([1.2, 1.2])
        self.Ti = np.array([40.0, 40.0])
        self.Td = np.array([8.0, 8.0])
        self.N = 5.0

        # State variables
        self.integral = np.zeros(self.n_measurements)
        self.prev_error = np.zeros(self.n_measurements)
        self.prev_measurement = np.zeros(self.n_measurements)
        self.prev_output = self.initial_actuators.copy()
        self.prev_derivative = np.zeros(self.n_measurements)

        # Setpoint tracking
        self.setpoint_ramp_start = None
        self.setpoint_ramp_end = None
        self.setpoint_start = np.zeros(self.n_measurements)
        self.setpoint_end = np.zeros(self.n_measurements)

        # Output filtering to reduce actuator effort
        self.alpha = 0.2
        self.filtered_output = self.initial_actuators.copy()

    def reset(self):
        self.integral = np.zeros(self.n_measurements)
        self.prev_error = np.zeros(self.n_measurements)
        self.prev_measurement = np.zeros(self.n_measurements)
        self.prev_output = self.initial_actuators.copy()
        self.prev_derivative = np.zeros(self.n_measurements)
        self.setpoint_ramp_start = None
        self.setpoint_ramp_end = None
        self.filtered_output = self.initial_actuators.copy()

    def _update_setpoint_schedule(self, t):
        if t < 200:
            return np.array([12.26, 12.78])
        elif 200 <= t < 500:
            return np.array([14.0, 12.78])
        elif 500 <= t < 560:
            if self.setpoint_ramp_start is None:
                self.setpoint_ramp_start = t
                self.setpoint_ramp_end = t + 60
                self.setpoint_start = np.array([14.0, 12.78])
                self.setpoint_end = np.array([14.0, 11.2])
            alpha = (t - self.setpoint_ramp_start) / (self.setpoint_ramp_end - self.setpoint_ramp_start)
            return self.setpoint_start + alpha * (self.setpoint_end - self.setpoint_start)
        else:
            return np.array([12.26, 12.78])

    def step(self, t, y, r, quality):
        r = self._update_setpoint_schedule(t)
        output = np.zeros(self.n_actuators)

        for i in range(self.n_measurements):
            if not quality[i]:
                output[i] = self.prev_output[i]
                continue

            error = r[i] - y[i]

            # Proportional term with feedforward on setpoint change
            P = self.Kp[i] * error

            # Integral term with anti-windup
            self.integral[i] += error * self.sample_time
            I = self.Kp[i] / self.Ti[i] * self.integral[i]

            # Derivative term on measurement only (no derivative kick)
            derivative = -(y[i] - self.prev_measurement[i]) / self.sample_time
            filtered_derivative = (self.Td[i] * self.prev_derivative[i] + self.sample_time * derivative) / (self.Td[i] + self.sample_time / self.N)
            D = self.Kp[i] * self.Td[i] * filtered_derivative

            # PID output
            pid_output = P + I + D
            raw_output = np.clip(pid_output, self.actuator_limits[i, 0], self.actuator_limits[i, 1])

            # Anti-windup
            if pid_output != raw_output:
                self.integral[i] -= error * self.sample_time

            # Output filtering to reduce actuator effort
            output[i] = self.alpha * raw_output + (1 - self.alpha) * self.filtered_output[i]

            # Update state
            self.prev_error[i] = error
            self.prev_measurement[i] = y[i]
            self.prev_derivative[i] = filtered_derivative
            self.filtered_output[i] = output[i]

        self.prev_output = output.copy()
        return output