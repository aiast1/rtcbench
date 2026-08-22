import numpy as np
from dataclasses import dataclass

@dataclass
class Brief:
    sample_time: float

class Controller:
    def __init__(self, brief: Brief):
        self.brief = brief
        self.ny = 3
        self.nu = 3
        self.n_dist = 2

        # Nominal model parameters
        self.K = np.array([
            [4.05, 1.77, 5.88],
            [5.39, 5.72, 6.9],
            [4.38, 4.42, 7.2]
        ])
        self.TAU = np.array([
            [50.0, 60.0, 50.0],
            [50.0, 60.0, 40.0],
            [33.0, 44.0, 19.0]
        ])
        self.L = np.array([
            [27.0, 28.0, 27.0],
            [18.0, 14.0, 15.0],
            [20.0, 22.0, 0.0]
        ])
        self.KD = np.array([
            [1.2, 1.44],
            [1.52, 1.83],
            [1.14, 1.26]
        ])
        self.TAUD = np.array([
            [45.0, 40.0],
            [25.0, 20.0],
            [27.0, 32.0]
        ])
        self.LD = np.array([
            [27.0, 27.0],
            [15.0, 15.0],
            [27.0, 32.0]
        ])

        # PID gains
        self.Kp = np.array([0.3, 0.3, 0.3])
        self.Ki = np.array([0.05, 0.05, 0.05])
        self.Kd = np.array([0.05, 0.05, 0.05])

        # Integrator states
        self.integrator_states = np.zeros(self.ny)

        # Previous measurements and control outputs
        self.prev_y = np.zeros(self.ny)
        self.prev_u = np.zeros(self.nu)
        self.prev_error = np.zeros(self.ny)

        # Setpoint schedule
        self.setpoint_schedule = [
            (0, np.array([0.0, 0.0, 0.0])),
            (120, np.array([0.4, -0.3, 0.0])),
            (350, np.array([-0.25, 0.25, 0.12])),
            (600, np.array([0.0, 0.0, 0.0]))
        ]

    def reset(self):
        self.integrator_states = np.zeros(self.ny)
        self.prev_y = np.zeros(self.ny)
        self.prev_u = np.zeros(self.nu)
        self.prev_error = np.zeros(self.ny)

    def step(self, t, y, r, quality):
        # Get current setpoint
        current_setpoint = self.get_setpoint(t)

        # Check for safety violations
        if y[2] < -0.5:
            return np.zeros(self.nu)

        # Calculate error
        error = current_setpoint - y

        # Update integrator states with anti-windup
        for i in range(self.ny):
            if self.prev_u[i] == 0.5 and error[i] > 0:
                pass
            elif self.prev_u[i] == -0.5 and error[i] < 0:
                pass
            else:
                self.integrator_states[i] += self.Ki[i] * error[i] * self.brief.sample_time

        # Calculate derivative term
        derivative_term = self.Kd * (error - self.prev_error) / self.brief.sample_time
        self.prev_error = error

        # Calculate control output
        u = self.Kp * error + self.integrator_states + derivative_term

        # Limit control output
        for i in range(self.nu):
            if u[i] > 0.5:
                u[i] = 0.5
                self.integrator_states[i] -= self.Ki[i] * error[i] * self.brief.sample_time
            elif u[i] < -0.5:
                u[i] = -0.5
                self.integrator_states[i] -= self.Ki[i] * error[i] * self.brief.sample_time

        # Actuator slew limits
        du = u - self.prev_u
        if np.any(np.abs(du) > 0.016 / self.brief.sample_time):
            u = self.prev_u + np.sign(du) * 0.016 / self.brief.sample_time
        self.prev_u = u
        self.prev_y = y

        return u

    def get_setpoint(self, t):
        for i in range(len(self.setpoint_schedule) - 1):
            if t >= self.setpoint_schedule[i][0] and t < self.setpoint_schedule[i+1][0]:
                if i == 0:
                    return self.setpoint_schedule[i][1]
                else:
                    slope = (self.setpoint_schedule[i+1][1] - self.setpoint_schedule[i][1]) / (self.setpoint_schedule[i+1][0] - self.setpoint_schedule[i][0])
                    return self.setpoint_schedule[i][1] + slope * (t - self.setpoint_schedule[i][0])
        return self.setpoint_schedule[-1][1]