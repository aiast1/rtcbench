import numpy as np

class Controller:
    def __init__(self, brief):
        self.dt = float(brief.sample_time)
        self.u_min = np.array([0.8, 0.8])
        self.u_max = np.array([5.0, 5.0])
        self.u0 = np.array([3.0, 3.0])
        self.kp = np.array([0.14, 0.12])
        self.ki = np.array([0.0075, 0.0065])
        self.kd = np.array([0.06, 0.05])
        self.beta = 0.35
        self.tau_f = 10.0
        self.reset()

    def reset(self):
        self.intg = np.zeros(2)
        self.y_f = None
        self.y_prev = None
        self.u_prev = self.u0.copy()

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float).copy()
        r = np.asarray(r, dtype=float).copy()
        q = np.asarray(quality, dtype=bool)
        for i in range(2):
            if (not q[i]) or (not np.isfinite(y[i])):
                y[i] = self.y_prev[i] if self.y_prev is not None else 11.5
        y2 = y[:2]
        if self.y_f is None:
            self.y_f = y2.copy()
            self.y_prev = y2.copy()
        alpha = self.dt / (self.tau_f + self.dt)
        self.y_f = (1.0 - alpha) * self.y_f + alpha * y2
        dy = (self.y_f - self.y_prev) / max(self.dt, 1e-6)
        self.y_prev = self.y_f.copy()

        r_use = np.where(np.isfinite(r[:2]), r[:2], y2)
        r_use = np.clip(r_use, 9.0, 13.5)
        e = r_use - y2
        p = self.kp * (self.beta * r_use - y2)
        d = -self.kd * np.clip(dy, -0.4, 0.4)
        u_try = self.u0 + p + self.intg + d
        for i in range(2):
            if u_try[i] >= self.u_max[i] and e[i] > 0:
                continue
            if u_try[i] <= self.u_min[i] and e[i] < 0:
                continue
            self.intg[i] += self.ki[i] * e[i] * self.dt
        self.intg = np.clip(self.intg, -2.5, 2.5)
        u = self.u0 + p + self.intg + d
        u = np.clip(u, self.u_min, self.u_max)
        du_max = 0.10
        u = self.u_prev + np.clip(u - self.u_prev, -du_max, du_max)
        u = np.clip(u, self.u_min, self.u_max)
        self.u_prev = u.copy()
        return u