import numpy as np

class Controller:
    def __init__(self, brief):
        self.dt = float(brief.sample_time)
        self.umin = 270.0
        self.umax = 340.0
        self.Kp = 6.0
        self.Ti = 120.0
        self.Td = 12.0
        self.beta = 0.3
        self.Tnom = 350.0
        self.u0 = 300.0
        self.I = 0.0
        self.yp = 350.0
        self.up = 300.0
        self.slew = 0.8 * self.dt
        self.Tf = 12.0
        self.dfilt = 0.0

    def reset(self):
        self.I = 0.0
        self.yp = 350.0
        self.up = 300.0
        self.dfilt = 0.0

    def step(self, t, y, r, quality):
        T = float(y[0]) if (quality is not None and len(quality) > 0 and quality[0]) else self.yp
        if not np.isfinite(T):
            T = self.yp
        T = float(np.clip(T, 280.0, 480.0))
        sp = float(r[0]) if (r is not None and np.isfinite(r[0])) else self.Tnom
        e = sp - T
        P = self.Kp * (self.beta * (sp - self.Tnom) - (T - self.Tnom))
        raw_d = (T - self.yp) / max(self.dt, 1e-6)
        a = self.dt / (self.Tf + self.dt)
        self.dfilt = (1.0 - a) * self.dfilt + a * raw_d
        D = -self.Kp * self.Td * self.dfilt
        u_unsat = self.u0 + P + self.I + D
        u = float(np.clip(u_unsat, self.umin, self.umax))
        du = float(np.clip(u - self.up, -self.slew, self.slew))
        u = self.up + du
        u = float(np.clip(u, self.umin, self.umax))
        if abs(u_unsat - u) < 0.2:
            self.I += (self.Kp * self.dt / self.Ti) * e
        self.I = float(np.clip(self.I, -30.0, 30.0))
        self.yp = T
        self.up = u
        return np.array([u], dtype=float)