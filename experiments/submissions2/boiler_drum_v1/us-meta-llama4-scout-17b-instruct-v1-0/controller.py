import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = brief.sample_time
        self.level_sp = 0.0
        self.level_prev = None
        self.error_prev = 0.0
        self.integral = 0.0
        self.feedwater_valve_prev = 45.0
        self.drum_pressure_prev = None

    def reset(self):
        self.level_prev = None
        self.error_prev = 0.0
        self.integral = 0.0
        self.drum_pressure_prev = None

    def step(self, t, y, r, quality):
        # Unpack measurements
        l_ind = y[0]
        p = y[1]
        ft_103 = y[2]
        ft_104 = y[3]

        # Unpack setpoints
        level_sp = r[0]

        # Check if measurements are valid
        if not quality[0] or not quality[1] or not quality[2] or not quality[3]:
            return np.array([self.feedwater_valve_prev])

        # Feedwater valve control
        kp = 0.3
        ki = 0.002 / self.dt
        kd = 0.1 * self.dt

        error = level_sp - l_ind
        self.integral += error * self.dt
        if self.feedwater_valve_prev >= 100.0:
            self.integral -= 0.05 * error * self.dt
        if self.feedwater_valve_prev <= 0.0:
            self.integral += 0.05 * error * self.dt

        if self.level_prev is not None:
            derivative = (error - self.error_prev) / self.dt
        else:
            derivative = 0.0
        self.error_prev = error
        self.level_prev = l_ind

        u = kp * error + ki * self.integral + kd * derivative

        # Actuator limits and slew rate limit
        u = max(0.0, min(100.0, u))
        slew_limit = 0.00169 / self.dt
        u = max(self.feedwater_valve_prev - slew_limit, min(self.feedwater_valve_prev + slew_limit, u))

        # Safety checks (very simplified)
        if l_ind > 250.0 or l_ind < -250.0 or p < 70.0 or p > 100.0:
            u = self.feedwater_valve_prev  # Hold valve position on safety violation

        self.feedwater_valve_prev = u
        return np.array([u])