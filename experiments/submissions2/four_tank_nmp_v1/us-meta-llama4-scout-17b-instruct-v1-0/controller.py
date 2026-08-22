import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.dt = self.sample_time
        self.n_tanks = 4
        self.n_actuators = 2
        self.n_measurements = 2

        # Nominal model parameters
        self.A1, self.A2, self.A3, self.A4 = 28.0, 32.0, 28.0, 32.0
        self.a1, self.a2, self.a3, self.a4 = 0.071, 0.057, 0.071, 0.057
        self.k1, self.k2 = 3.14, 3.29
        self.gamma1, self.gamma2 = 0.43, 0.34

        # PID gains
        self.Kp = np.array([0.5, 0.5])
        self.Ki = np.array([0.05, 0.05])
        self.Kd = np.array([0.0, 0.0])

        # Integral terms
        self.integral = np.array([0.0, 0.0])

        # Previous error and measurement
        self.prev_error = np.array([0.0, 0.0])
        self.prev_measurement = np.array([0.0, 0.0])

        # Actuator limits
        self.actuator_limits = np.array([10.0, 10.0])

        # Setpoint schedule
        self.setpoint_schedule = [
            [0, np.array([11.28, 11.94])],
            [200, np.array([13.0, 11.94])],
            [500, np.array([13.0, 10.4])],
            [900, np.array([11.28, 11.94])]
        ]

        # Initialize previous control output
        self.prev_u = np.array([3.0, 3.0])

    def reset(self):
        self.integral = np.array([0.0, 0.0])
        self.prev_error = np.array([0.0, 0.0])
        self.prev_measurement = np.array([0.0, 0.0])
        self.prev_u = np.array([3.0, 3.0])

    def step(self, t, y, r, quality):
        # Get current setpoint
        setpoint = self.get_setpoint(t)

        # Calculate error
        error = setpoint - y

        # Update integral term with anti-windup
        self.integral += self.Ki * error * self.dt
        for i in range(self.n_measurements):
            if abs(self.integral[i]) > 100:
                self.integral[i] = np.sign(self.integral[i]) * 100

        # Calculate control output
        u = self.prev_u + self.Kp * (error - self.prev_error) + self.Ki * error * self.dt

        # Limit control output
        u = np.clip(u, 0, self.actuator_limits)

        # Limit actuator slew rate
        slew_limit = 0.0037 / self.dt
        u = np.clip(u, self.prev_u - slew_limit, self.prev_u + slew_limit)

        # Update previous error, measurement and control output
        self.prev_error = error
        self.prev_measurement = y
        self.prev_u = u

        return u

    def get_setpoint(self, t):
        for i in range(len(self.setpoint_schedule) - 1):
            if t >= self.setpoint_schedule[i][0] and t < self.setpoint_schedule[i + 1][0]:
                if i == 0:
                    return self.setpoint_schedule[i][1]
                else:
                    slope = (self.setpoint_schedule[i + 1][1] - self.setpoint_schedule[i][1]) / \
                            (self.setpoint_schedule[i + 1][0] - self.setpoint_schedule[i][0])
                    return self.setpoint_schedule[i][1] + slope * (t - self.setpoint_schedule[i][0])
        return self.setpoint_schedule[-1][1]