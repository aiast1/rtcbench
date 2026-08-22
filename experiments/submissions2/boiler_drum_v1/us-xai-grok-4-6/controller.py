import numpy as np

class Controller:
    def __init__(self, brief):
        self.dt = float(brief.sample_time)
        self.reset()

    def reset(self):
        self.u = 45.0
        self.last_u = 45.0
        self.ie = 0.0
        self.yf = None
        self.spf = 0.0
        self.stm = 50.0
        self.fw = 50.0
        self.p = 85.0
        self.inited = False

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        q = np.asarray(quality, dtype=bool)
        raw = np.array([0.0, 85.0, 50.0, 50.0])
        if self.yf is None:
            self.yf = raw.copy()
        for i in range(4):
            if i < len(y) and bool(q[i]) and np.isfinite(y[i]):
                raw[i] = y[i]
            else:
                raw[i] = self.yf[i]
        self.yf = 0.75 * self.yf + 0.25 * raw
        lvl = float(self.yf[0])
        p = float(self.yf[1])
        stm = float(self.yf[2])
        fw = float(self.yf[3])
        if not self.inited:
            self.stm = stm
            self.fw = fw
            self.p = p
            self.spf = float(r[0]) if np.isfinite(r[0]) else 0.0
            self.inited = True
        self.stm = 0.9 * self.stm + 0.1 * stm
        self.fw = 0.9 * self.fw + 0.1 * fw
        self.p = 0.92 * self.p + 0.08 * p
        sp = float(r[0]) if np.isfinite(r[0]) else self.spf
        self.spf = 0.985 * self.spf + 0.015 * sp
        e = self.spf - lvl
        # shrink/swell: don't chase inverse response
        if t < 80.0:
            e = 0.0
        # inventory proxy — keep feed up if emptying
        imb = self.stm - self.fw
        if imb > 8.0:
            e += 0.3 * imb
        if imb < -8.0:
            e += 0.2 * imb
        if p < 78.0:
            e -= 2.0 * (78.0 - p)  # less cold feed if P low
        if p > 95.0:
            e += 1.5 * (p - 95.0)
        if lvl > 180.0:
            e -= (lvl - 180.0)
        if lvl < -180.0:
            e += (-180.0 - lvl)
        self.ie = float(np.clip(self.ie + 0.0004 * e * self.dt, -8.0, 8.0))
        # three element around 45%
        u = 45.0 + 0.35 * (self.stm - 50.0) + 0.08 * e + self.ie
        u = float(np.clip(u, 20.0, 75.0))
        du = 0.35
        u = float(np.clip(u, self.last_u - du, self.last_u + du))
        if (u >= 74.5 and e > 0) or (u <= 20.5 and e < 0):
            self.ie -= 0.0004 * e * self.dt
        self.last_u = u
        self.u = u
        return np.array([u], dtype=float)