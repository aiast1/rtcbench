import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.actuator_limits = np.array([brief.actuators[0][0], brief.actuators[0][1],
                                         brief.actuators[1][0], brief.actuators[1][1]])
        self.setpoint_schedule = {
            0: np.array([1.09, 114.19]),
            900: np.array([0.928, 114.19]),
            2900: np.array([1.081, 114.19]),
            4600: np.array([0.892, 114.19]),
            6100: np.array([1.09, 114.19])
        }
        self.current_setpoint = np.array([1.09, 114.19])
        self.last_time = 0.0

        # PID parameters
        self.Kp = np.array([0.5, 0.1])  # Proportional gain for CB and T
        self.Ki = np.array([0.01, 0.005])  # Integral gain for CB and T
        self.Kd = np.array([0.0, 0.0])  # Derivative gain for CB and T

        # Anti-windup parameters
        self.anti_windup_gain = 0.1

        # Integrator states
        self.integrator = np.array([0.0, 0.0])

        # Previous measurement values
        self.previous_measurement = np.array([0.0, 0.0])

    def reset(self):
        self.current_setpoint = np.array([1.09, 114.19])
        self.last_time = 0.0
        self.integrator = np.array([0.0, 0.0])
        self.previous_measurement = np.array([0.0, 0.0])

    def step(self, t, y, r, quality):
        """t: seconds (float). y: numpy array of measured values, one per measurement.
        r: numpy array of setpoints, nan for channels that are not scored.
        quality: numpy bool array, False means that reading is stale/bad.
        Returns: numpy array of control values, one per actuator."""

        # Update setpoint
        for time, setpoint in self.setpoint_schedule.items():
            if t >= time and (time == 0 or t < list(self.setpoint_schedule.keys())[list(self.setpoint_schedule.keys()).index(time)+1]):
                self.current_setpoint = setpoint

        # Calculate error
        error = self.current_setpoint - y

        # Update integrator
        self.integrator += self.Ki * error * self.sample_time

        # Anti-windup
        self.integrator[0] = np.clip(self.integrator[0], self.actuator_limits[0], self.actuator_limits[1])
        self.integrator[1] = np.clip(self.integrator[1], self.actuator_limits[2], self.actuator_limits[3])

        # Calculate control output
        u = self.Kp * error + self.integrator + self.Kd * (y - self.previous_measurement) / self.sample_time

        # Saturate actuators
        u[0] = np.clip(u[0], self.actuator_limits[0], self.actuator_limits[1])
        u[1] = np.clip(u[1], self.actuator_limits[2], self.actuator_limits[3])

        # Update previous measurement
        self.previous_measurement = y

        # Safety check
        if y[1] > 150.0:
            u = np.array([self.actuator_limits[0], self.actuator_limits[2]])  # Zero output on safety violation

        return u