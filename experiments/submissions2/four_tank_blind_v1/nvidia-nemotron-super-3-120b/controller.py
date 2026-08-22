import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.n_y = len(brief.measurements)
        self.n_u = len(brief.actuators)
        self.y_min = np.array([m.min for m in brief.measurements])
        self.y_max = np.array([m.max for m in brief.measurements])
        self.u_min = np.array([a.min for a in brief.actuators])
        self.u_max = np.array([a.max for a in brief.actuators])
        self.r = np.zeros(self.n_y)
        self.r_old = np.zeros(self.n_y)
        self.e_int = np.zeros(self.n_y)
        self.e_prev = np.zeros(self.n_y)
        self.u_prev = np.array([3.0, 3.0])  # bumpless start
        self.alpha = 0.1  # derivative filter coefficient
        self.d_filt = np.zeros(self.n_y)
        self.Kp = np.array([0.06, 0.06])
        self.Ki = np.array([0.0015, 0.0015])
        self.Kd = np.array([0.0, 0.0])
        self.effort_weight = 0.5
        self.max_duty = 0.017
        self.dt = self.sample_time

    def reset(self):
        self.e_int = np.zeros(self.n_y)
        self.e_prev = np.zeros(self.n_y)
        self.u_prev = np.array([3.0, 3.0])
        self.d_filt = np.zeros(self.n_y)

    def step(self, t, y, r, quality):
        # Update setpoint, hold previous if NaN
        for i in range(self.n_y):
            if not np.isnan(r[i]):
                self.r[i] = r[i]
        # Error
        e = self.r - y
        # Integral with anti-windup via back-calculation
        self.e_int += e * self.dt
        # PID output
        u_p = self.Kp * e
        u_i = self.Ki * self.e_int
        # Derivative on measurement to avoid kick, filtered
        de = (y - self.e_prev) / self.dt if self.dt > 0 else 0.0
        self.d_filt = self.alpha * de + (1 - self.alpha) * self.d_filt
        u_d = -self.Kd * self.d_filt
        u_ff = self.u_prev  # bumpless feedforward
        u = u_ff + u_p + u_i + u_d
        # Anti-windup: back-calculation if saturated
        for i in range(self.n_u):
            if u[i] < self.u_min[i] or u[i] > self.u_max[i]:
                self.e_int[i] -= e[i] * self.dt  # undo integral step
        # Clip to actuator limits
        u = np.clip(u, self.u_min, self.u_max)
        # Duty limit: restrict actuator movement per step
        delta = u - self.u_prev
        max_delta = self.max_duty * (self.u_max - self.u_min)
        delta = np.clip(delta, -max_delta, max_delta)
        u = self.u_prev + delta
        # Final clip (ensure within limits after duty limit)
        u = np.clip(u, self.u_min, self.u_max)
        # Store for next step
        self.u_prev = u.copy()
        self.e_prev = y.copy()
        return u