import numpy as np

class Controller:
    def __init__(self, brief):
        self.dt = float(brief.sample_time)
        self.u_min = 0.0
        self.u_max = 100.0
        self.kp = 0.85
        self.ti = 240.0
        self.u = 50.0
        self.i = 0.0
        self.y_f = 55.0
        self.last_good = 55.0
        self.alpha = 0.12
        self.du_max = 0.55

    def reset(self):
        self.u = 50.0
        self.i = 0.0
        self.y_f = 55.0
        self.last_good = 55.0

    def step(self, t, y, r, quality):
        yv = float(y[0]) if np.isfinite(y[0]) else self.last_good
        qok = bool(quality[0]) if quality is not None else True
        if qok and np.isfinite(yv):
            self.last_good = yv
        else:
            yv = self.last_good
        self.y_f = self.alpha * yv + (1.0 - self.alpha) * self.y_f
        sp = float(r[0]) if r is not None and np.isfinite(r[0]) else 55.0
        # nominal static map: T ≈ 15 + 0.8 u  => u_ff = (sp-15)/0.8
        u_ff = (sp - 15.0) / 0.85
        err = sp - self.y_f
        p = self.kp * err
        u_try = u_ff + p + self.i
        sat_hi = u_try >= self.u_max - 1e-6
        sat_lo = u_try <= self.u_min + 1e-6
        if not ((sat_hi and err > 0.0) or (sat_lo and err < 0.0)):
            self.i += (self.kp * self.dt / self.ti) * err
        self.i = float(np.clip(self.i, -25.0, 25.0))
        u = u_ff + p + self.i
        if self.y_f > 72.0:
            u = min(u, 58.0)
        if self.y_f > 80.0:
            u = min(u, 38.0)
        if self.y_f > 85.0:
            u = min(u, 16.0)
        if self.y_f > 87.5:
            u = min(u, 6.0)
        u = float(np.clip(u, self.u_min, self.u_max))
        u = float(np.clip(u, self.u - self.du_max, self.u + self.du_max))
        self.u = u
        return np.array([self.u], dtype=float)