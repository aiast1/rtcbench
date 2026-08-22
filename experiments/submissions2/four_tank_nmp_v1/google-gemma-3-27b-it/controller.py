import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.actuator_limits = np.array([0.0, 10.0])
        self.Kp = 0.8
        self.Ki = 0.005
        self.Kd = 0.05
        self.integrator_1 = 0.0
        self.integrator_2 = 0.0
        self.prev_error_1 = 0.0
        self.prev_error_2 = 0.0
        self.output_1 = 3.0
        self.output_2 = 3.0
        self.max_integrator_windup = 2.0  # Limit integrator windup

    def reset(self):
        self.integrator_1 = 0.0
        self.integrator_2 = 0.0
        self.prev_error_1 = 0.0
        self.prev_error_2 = 0.0
        self.output_1 = 3.0
        self.output_2 = 3.0

    def step(self, t, y, r, quality):
        """t: seconds (float). y: numpy array of measured values, one per measurement.
        r: numpy array of setpoints, nan for channels that are not scored.
        quality: numpy bool array, False means that reading is stale/bad.
        Returns: numpy array of control values, one per actuator."""

        if not quality[0] or not quality[1]:
            return np.array([self.output_1, self.output_2])

        error_1 = r[0] - y[0]
        error_2 = r[1] - y[1]

        # Anti-windup and clamping
        integrator_1_change = error_1 * self.Ki * self.sample_time
        self.integrator_1 += integrator_1_change
        self.integrator_1 = np.clip(self.integrator_1, -self.max_integrator_windup, self.max_integrator_windup)

        integrator_2_change = error_2 * self.Ki * self.sample_time
        self.integrator_2 += integrator_2_change
        self.integrator_2 = np.clip(self.integrator_2, -self.max_integrator_windup, self.max_integrator_windup)

        # PID control
        output_1 = self.Kp * error_1 + self.integrator_1 + self.Kd * (error_1 - self.prev_error_1) / self.sample_time
        output_2 = self.Kp * error_2 + self.integrator_2 + self.Kd * (error_2 - self.prev_error_2) / self.sample_time

        # Saturate outputs
        output_1 = np.clip(output_1, self.actuator_limits[0], self.actuator_limits[1])
        output_2 = np.clip(output_2, self.actuator_limits[0], self.actuator_limits[1])

        self.prev_error_1 = error_1
        self.prev_error_2 = error_2

        return np.array([output_1, output_2])