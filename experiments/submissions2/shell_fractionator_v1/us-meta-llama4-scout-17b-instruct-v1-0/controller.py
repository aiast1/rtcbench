import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.model = {
            "K": np.array([
                [4.05, 1.77, 5.88],
                [5.39, 5.72, 6.9],
                [4.38, 4.42, 7.2]
            ]),
            "TAU": np.array([
                [50.0, 60.0, 50.0],
                [50.0, 60.0, 40.0],
                [33.0, 44.0, 19.0]
            ]),
            "L": np.array([
                [27.0, 28.0, 27.0],
                [18.0, 14.0, 15.0],
                [20.0, 22.0, 0.0]
            ]),
            "KD": np.array([
                [1.2, 1.44],
                [1.52, 1.83],
                [1.14, 1.26]
            ]),
            "TAUD": np.array([
                [45.0, 40.0],
                [25.0, 20.0],
                [27.0, 32.0]
            ]),
            "LD": np.array([
                [27.0, 27.0],
                [15.0, 15.0],
                [27.0, 32.0]
            ])
        }
        self.sample_time = brief.sample_time
        self.n_actuators = 3
        self.n_measurements = 3
        self.n_disturbances = 2
        self.actuators = np.zeros(self.n_actuators)
        self.integrals = np.zeros(self.n_measurements)
        self.prev_error = np.zeros(self.n_measurements)
        self.setpoint_schedule = np.array([
            [0.0, 0.0, 0.0],
            [0.2, -0.15, 0.0],
            [-0.1, 0.1, 0.05],
            [0.0, 0.0, 0.0]
        ])
        self.time = 0.0
        self.schedule_index = 0

    def reset(self):
        self.actuators = np.zeros(self.n_actuators)
        self.integrals = np.zeros(self.n_measurements)
        self.prev_error = np.zeros(self.n_measurements)
        self.time = 0.0
        self.schedule_index = 0

    def step(self, t, y, r, quality):
        self.time = t
        if self.time < 120:
            r = self.setpoint_schedule[0]
        elif self.time < 210:
            r = self.setpoint_schedule[1]
        elif self.time < 600:
            r = self.setpoint_schedule[2] + (self.setpoint_schedule[1] - self.setpoint_schedule[2]) * (600 - self.time) / 90
        else:
            r = self.setpoint_schedule[3]

        error = r - y
        self.integrals += self.sample_time * error
        delta_error = (error - self.prev_error) / self.sample_time
        self.prev_error = error

        # PID gains
        Kp = 0.5 * np.ones(self.n_measurements)
        Ki = 0.1 * np.ones(self.n_measurements)
        Kd = 0.05 * np.ones(self.n_measurements)

        u = -Kp * error - Ki * self.integrals - Kd * delta_error

        # Anti-windup
        for i in range(self.n_actuators):
            if u[i] > 0.5:
                u[i] = 0.5
            elif u[i] < -0.5:
                u[i] = -0.5

        # Limit actuator slew rate
        slew_limit = 0.0112 / self.sample_time
        for i in range(self.n_actuators):
            if u[i] > self.actuators[i] + slew_limit:
                u[i] = self.actuators[i] + slew_limit
            elif u[i] < self.actuators[i] - slew_limit:
                u[i] = self.actuators[i] - slew_limit

        self.actuators = u
        return u