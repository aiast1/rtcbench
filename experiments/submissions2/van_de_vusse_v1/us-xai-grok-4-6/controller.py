import numpy as np

class Controller:
    def __init__(self, brief):
        self.dt = float(getattr(brief, "sample_time", 10.0))
        self.u0 = np.array([14.19, -1113.5], dtype=float)
        self.umin = np.array([3.0, -9000.0], dtype=float)
        self.umax = np.array([35.0, 0.0], dtype=float)
        self.kp_c = -4.5
        self.ki_c = -0.12
        self.kp_t = 180.0
        self.ki_t = 4.0
        self.reset()

    def reset(self):
        self.u = self.u0.copy()
        self.ec_i = 0.0
        self.et_i = 0.0
        self.yf = None
        self.rT_f = 114.19
        self.rcb_f = 1.09

    def _sat(self, u):
        return np.clip(u, self.umin, self.umax)

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float).reshape(-1)
        r = np.asarray(r, dtype=float).reshape(-1)
        q = np.asarray(quality, dtype=bool).reshape(-1)
        if self.yf is None:
            self.yf = np.array([y[0] if np.isfinite(y[0]) else 1.09,
                                y[1] if np.isfinite(y[1]) else 114.19], dtype=float)
        else:
            a = 0.35
            if q.size > 0 and q[0] and np.isfinite(y[0]):
                self.yf[0] = (1.0 - a) * self.yf[0] + a * y[0]
            if q.size > 1 and q[1] and np.isfinite(y[1]):
                self.yf[1] = (1.0 - a) * self.yf[1] + a * y[1]
        cb, temp = float(self.yf[0]), float(self.yf[1])
        rcb = float(r[0]) if r.size > 0 and np.isfinite(r[0]) else self.rcb_f
        rT = float(r[1]) if r.size > 1 and np.isfinite(r[1]) else self.rT_f
        self.rcb_f = 0.4 * self.rcb_f + 0.6 * rcb
        self.rT_f = 0.4 * self.rT_f + 0.6 * rT

        # stay left of yield peak
        fv_lo, fv_hi = 7.0, 14.35
        # crude inverse: higher CB needs slightly lower F/V on this side
        uff = 12.8 - 8.0 * (self.rcb_f - 1.0)
        uff = float(np.clip(uff, fv_lo, fv_hi))

        ec = self.rcb_f - cb
        et = self.rT_f - temp
        dt = self.dt

        self.et_i += et * dt
        self.et_i = float(np.clip(self.et_i, -800.0, 800.0))
        qff = -950.0 - 18.0 * (self.rT_f - 114.19)
        uq = qff + self.kp_t * et + self.ki_t * self.et_i
        if temp > 128.0:
            uq -= 500.0 * (temp - 128.0)
        if temp > 138.0:
            uq = -9000.0
        uq = float(np.clip(uq, self.umin[1], self.umax[1]))
        if (uq <= self.umin[1] + 1e-6 and et < 0) or (uq >= self.umax[1] - 1e-6 and et > 0):
            self.et_i -= et * dt

        self.ec_i += ec * dt
        self.ec_i = float(np.clip(self.ec_i, -40.0, 40.0))
        uf = uff + self.kp_c * ec + self.ki_c * self.ec_i
        uf = float(np.clip(uf, fv_lo, fv_hi))
        if (uf <= fv_lo + 1e-9 and self.kp_c * ec < 0) or (uf >= fv_hi - 1e-9 and self.kp_c * ec > 0):
            self.ec_i -= ec * dt

        du_max = np.array([0.15, 150.0])
        u_des = np.array([uf, uq], dtype=float)
        du = np.clip(u_des - self.u, -du_max, du_max)
        self.u = self._sat(self.u + du)
        return self.u.copy()