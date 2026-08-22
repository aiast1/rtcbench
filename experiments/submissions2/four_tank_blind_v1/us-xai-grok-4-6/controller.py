import numpy as np

class Controller:
    def __init__(self, brief):
        self.dt = float(brief.sample_time)
        self.u_min = 0.0
        self.u_max = 10.0
        self.u0 = np.array([3.0, 3.0], dtype=float)
        self.kp = np.array([0.85, 0.85])
        self.ki = np.array([0.038, 0.038])
        self.kd = np.array([0.18, 0.18])
        self.k12 = 0.12
        self.beta = 0.7
        self.u_slew = 0.28
        self.reset()

    def reset(self):
        self.u = self.u0.copy()
        self.integ = np.zeros(2)
        self.y_f = np.array([12.26, 12.78], dtype=float)
        self.y_prev = self.y_f.copy()
        self.dy_f = np.zeros(2)
        self.initialized = False

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float).copy()
        r = np.asarray(r, dtype=float).copy()
        q = np.asarray(quality, dtype=bool)

        for i in range(2):
            if not np.isfinite(y[i]):
                q[i] = False
            if q[i]:
                a = 0.55
                if not self.initialized:
                    self.y_f[i] = y[i]
                    self.y_prev[i] = y[i]
                else:
                    self.y_f[i] = (1.0 - a) * self.y_f[i] + a * y[i]

        if not self.initialized:
            self.initialized = True
            return self.u.copy()

        sp = np.array([
            r[0] if np.isfinite(r[0]) else self.y_f[0],
            r[1] if np.isfinite(r[1]) else self.y_f[1],
        ])

        e = sp - self.y_f
        e_p = self.beta * sp - self.y_f
        dy_raw = (self.y_f - self.y_prev) / max(self.dt, 1e-6)
        self.dy_f = 0.6 * self.dy_f + 0.4 * dy_raw
        self.y_prev = self.y_f.copy()

        scale = np.ones(2)
        for i in range(2):
            if self.y_f[i] > 17.5:
                scale[i] = max(0.25, (20.0 - self.y_f[i]) / 2.5)

        u_p = self.kp * e_p - self.kd * self.dy_f
        u_dec = np.array([
            u_p[0] - self.k12 * e_p[1],
            u_p[1] - self.k12 * e_p[0],
        ])
        u_des = self.u0 + u_dec + self.integ

        u_lim = np.clip(u_des, self.u_min, self.u_max)
        u_lim = self.u0 + scale * (u_lim - self.u0)
        u_lim = np.clip(u_lim, self.u_min, self.u_max)
        u_lim = np.clip(u_lim, self.u - self.u_slew, self.u + self.u_slew)

        for i in range(2):
            sat_hi = (u_des[i] > u_lim[i] + 1e-9) and (e[i] > 0)
            sat_lo = (u_des[i] < u_lim[i] - 1e-9) and (e[i] < 0)
            if not sat_hi and not sat_lo:
                self.integ[i] += self.ki[i] * e[i] * self.dt
            self.integ[i] = float(np.clip(self.integ[i], -4.5, 4.5))

        self.u = u_lim
        return self.u.copy()