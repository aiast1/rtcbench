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
        self.k1, self.k2 = 3.33, 3.35
        self.gamma1, self.gamma2 = 0.7, 0.6

        # Controller gains
        self.Kp = np.array([[0.1, 0.0], [0.0, 0.1]])
        self.Ki = np.array([[0.01, 0.0], [0.0, 0.01]])
        self.Kd = np.array([[0.0, 0.0], [0.0, 0.0]])

        # Integral terms
        self.integral = np.zeros(self.n_actuators)

        # Previous error and measurement
        self.prev_error = np.zeros(self.n_measurements)
        self.prev_measurement = np.zeros(self.n_measurements)

        # Actuator limits
        self.actuator_limits = np.array([[0.0, 10.0], [0.0, 10.0]])

        # Setpoint schedule
        self.setpoint_schedule = np.array([
            [0, np.array([12.26, 12.78])],
            [200, np.array([14.0, 12.78])],
            [500, np.array([14.0, 11.2])],
            [900, np.array([12.26, 12.78])]
        ])

        # Initialize previous control output
        self.prev_u = np.array([3.0, 3.0])

    def reset(self):
        self.integral = np.zeros(self.n_actuators)
        self.prev_error = np.zeros(self.n_measurements)
        self.prev_measurement = np.zeros(self.n_measurements)
        self.prev_u = np.array([3.0, 3.0])

    def step(self, t, y, r, quality):
        # Get current setpoint
        setpoint = self.get_setpoint(t)

        # Calculate error
        error = setpoint - y

        # Update integral term with anti-windup
        for i in range(self.n_actuators):
            u_without_integral = self.Kp[i, 0] * error[0] + self.Kp[i, 1] * error[1] 
            integral_contribution = self.integral[i]
            proposed_u = u_without_integral + integral_contribution
            self.integral[i] = np.clip(proposed_u - u_without_integral, self.actuator_limits[i, 0] - u_without_integral, self.actuator_limits[i, 1] - u_without_integral)

        # Calculate control output
        u = np.array([self.Kp[0, 0] * error[0] + self.integral[0], self.Kp[1, 1] * error[1] + self.integral[1]])
        u = np.clip(u, self.actuator_limits[:, 0], self.actuator_limits[:, 1])

        # Slew rate limit
        u = np.clip(u, self.prev_u - 0.017, self.prev_u + 0.017)
        self.prev_u = u

        # Update previous error and measurement
        self.prev_error = error
        self.prev_measurement = y

        return u

    def get_setpoint(self, t):
        for i in range(len(self.setpoint_schedule) - 1):
            if t >= self.setpoint_schedule[i, 0] and t < self.setpoint_schedule[i + 1, 0]:
                if i == 0:
                    return self.setpoint_schedule[i, 1]
                else:
                    slope = (self.setpoint_schedule[i + 1, 1] - self.setpoint_schedule[i, 1]) / (self.setpoint_schedule[i + 1, 0] - self.setpoint_schedule[i, 0])
                    return self.setpoint_schedule[i, 1] + slope * (t - self.setpoint_schedule[i, 0])
        return self.setpoint_schedule[-1, 1]