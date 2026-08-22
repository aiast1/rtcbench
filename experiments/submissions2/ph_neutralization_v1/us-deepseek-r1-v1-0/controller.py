import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        self.Kp = 1.2
        self.Ki = 0.04
        self.integral = 0.0
        self.last_u = 14.22
        self.last_error = 0.0
        self.h_upper = 28.0
        self.h_lower = 7.0
        self.K_level = 0.4
        self.u_min = 0.0
        self.u_max = 30.0
        self.max_delta_u = 1.2
        self.first_step = True

    def reset(self):
        self.integral = 0.0
        self.last_u = 14.22
        self.last_error = 0.0
        self.first_step = True

    def step(self, t, y, r, quality):
        pH, h = y[0], y[1]
        sp = r[0]

        if self.first_step and quality[0]:
            error = sp - pH
            self.integral = (self.last_u - self.Kp * error)/self.Ki if self.Ki != 0 else 0.0
            self.first_step = False

        if quality[0]:
            error = sp - pH
        else:
            error = self.last_error

        self.integral += error * self.sample_time
        u_pi = self.Kp * error + self.Ki * self.integral
        u_pi_clamped = np.clip(u_pi, self.u_min, self.u_max)

        if (u_pi_clamped <= self.u_min and error < 0) or (u_pi_clamped >= self.u_max and error > 0):
            self.integral -= error * self.sample_time

        if h > self.h_upper:
            u_adj = u_pi_clamped - (h - self.h_upper)*self.K_level
        elif h < self.h_lower:
            u_adj = u_pi_clamped + (self.h_lower - h)*self.K_level
        else:
            u_adj = u_pi_clamped

        delta_u = u_adj - self.last_u
        delta_u = np.clip(delta_u, -self.max_delta_u, self.max_delta_u)
        u_final = np.clip(self.last_u + delta_u, self.u_min, self.u_max)

        self.last_u = u_final
        if quality[0]:
            self.last_error = error

        return np.array([u_final])