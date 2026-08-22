import numpy as np

class Controller:
    def __init__(self, brief):
        self.dt = float(brief.sample_time)
        self.umin = np.array([0.0, 0.0], dtype=float)
        self.umax = np.array([10.0, 10.0], dtype=float)
        self.kp = np.array([0.72, 0.68], dtype=float)
        self.ki = np.array([0.028, 0.026], dtype=float)
        self.u0 = np.array([3.0, 3.0], dtype=float)
        self.reset()

    def reset(self):
        self.u = self.u0.copy()
        self.I = np.zeros(2, dtype=float)
        self.yf = np.array([12.26, 12.78], dtype=float)
        self.y_last = self.yf.copy()
        self.have_y = False
        self.sp_f = self.yf.copy()

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float).reshape(-1)
        r = np.asarray(r, dtype=float).reshape(-1)
        q = np.asarray(quality, dtype=bool).reshape(-1)

        ymeas = y[:2].copy()
        for i in range(2):
            if (not q[i]) or (not np.isfinite(ymeas[i])):
                ymeas[i] = self.y_last[i]
        self.y_last = ymeas.copy()

        a = 0.50
        if not self.have_y:
            self.yf = ymeas.copy()
            self.have_y = True
        else:
            self.yf = a * ymeas + (1.0 - a) * self.yf

        sp = r[:2].copy()
        for i in range(2):
            if not np.isfinite(sp[i]):
                sp[i] = self.sp_f[i]
        self.sp_f = 0.40 * sp + 0.60 * self.sp_f

        e = self.sp_f - self.yf
        high = np.maximum(self.yf - 18.2, 0.0)
        e = e - 1.2 * high

        up = self.kp * e
        self.I = np.clip(self.I + self.ki * e * self.dt, -3.5, 3.5)

        u_unsat = self.u0 + up + self.I
        u = np.clip(u_unsat, self.umin, self.umax)
        self.I = self.I - 0.65 * (u_unsat - u)

        du = np.clip(u - self.u, -0.18, 0.18)
        u = np.clip(self.u + du, self.umin, self.umax)
        self.u = u
        return u