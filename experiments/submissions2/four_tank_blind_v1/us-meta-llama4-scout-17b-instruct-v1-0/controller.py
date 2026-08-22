import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.n_tanks = 2
        self.n_actuators = 2
        self.Kp = np.array([[0.5, 0.0], [0.0, 0.5]])
        self.Ki = np.array([[0.05, 0.0], [0.0, 0.05]])
        self.Kd = np.array([[0.1, 0.0], [0.0, 0.1]])
        self.integrator = np.zeros(self.n_tanks)
        self.prev_error = np.zeros(self.n_tanks)
        self.prev_output = np.array([3.0, 3.0])
        self.setpoint_schedule = [
            (0, np.array([12.26, 12.78])),
            (200, np.array([14.0, 12.78])),
            (500, np.array([14.0, 11.2])),
            (900, np.array([12.26, 12.78]))
        ]

    def reset(self):
        self.integrator = np.zeros(self.n_tanks)
        self.prev_error = np.zeros(self.n_tanks)
        self.prev_output = np.array([3.0, 3.0])

    def step(self, t, y, r, quality):
        # Get the current setpoint
        setpoint = self.get_setpoint(t)

        # Check for safety violations
        if np.any(y < 0) or np.any(y > 20):
            return np.array([0.0, 0.0])

        # Calculate the error
        error = setpoint - y

        # Calculate the derivative of the error
        d_error = (error - self.prev_error) / self.sample_time
        self.prev_error = error

        # Update the integrator with anti-windup
        self.integrator += error * self.sample_time
        output = self.Kp @ error + self.Ki @ self.integrator + self.Kd @ d_error
        output = np.clip(output, 0, 10)

        # Limit the actuator travel
        delta_output = output - self.prev_output
        delta_output = np.clip(delta_output, -0.017, 0.017)
        output = self.prev_output + delta_output
        output = np.clip(output, 0, 10)

        self.prev_output = output

        return output

    def get_setpoint(self, t):
        current_setpoint = self.setpoint_schedule[0][1]
        for i in range(len(self.setpoint_schedule) - 1):
            if t >= self.setpoint_schedule[i][0] and t < self.setpoint_schedule[i+1][0]:
                if i == 0:
                    current_setpoint = self.setpoint_schedule[i][1]
                else:
                    slope = (self.setpoint_schedule[i+1][1] - self.setpoint_schedule[i][1]) / (self.setpoint_schedule[i+1][0] - self.setpoint_schedule[i][0])
                    current_setpoint = self.setpoint_schedule[i][1] + slope * (t - self.setpoint_schedule[i][0])
        return current_setpoint