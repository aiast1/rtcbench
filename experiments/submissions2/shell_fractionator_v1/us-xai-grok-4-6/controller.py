import numpy as np

class Controller:
    def __init__(self, brief):
        self.dt = float(getattr(brief, "sample_time", 1.0))
        self.umax = 0.48
        self.umin = -0.48
        self.rate = 0.010
        K = np.array([
            [4.05, 1.77, 5.88],
            [5.39, 5.72, 6.90],
            [4.38, 4.42, 7.20],
        ], dtype=float)
        self.Kinv = np.linalg.pinv(K)
        self.kp = np.array([0.045, 0.040, 0.055])
        self.ki = np.array([0.0007, 0.0006, 0.0010])
        self.reset()

    def reset(self):
        self.u = np.zeros(3)
        self.xi = np.zeros(3)
        self.yhat = np.zeros(3)
        self.have_y = False

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float).reshape(-1)
        r = np.asarray(r, dtype=float).reshape(-1)
        q = np.asarray(quality, dtype=bool).reshape(-1)
        if not self.have_y:
            self.yhat = np.where(np.isfinite(y[:3]), y[:3], 0.0)
            self.have_y = True
        for i in range(min(3, y.size)):
            if i < q.size and q[i] and np.isfinite(y[i]):
                self.yhat[i] = 0.7 * self.yhat[i] + 0.3 * y[i]
        rr = np.zeros(3)
        n = min(3, r.size)
        rr[:n] = np.where(np.isfinite(r[:n]), r[:n], 0.0)
        # never request bottoms temp below a safe margin
        rr[2] = max(rr[2], -0.05)
        e = rr - self.yhat
        # hard constraint overlay on TI-103
        tbot = self.yhat[2]
        if tbot < -0.15:
            e[2] += 1.5 * (-0.08 - tbot)
        if tbot < -0.28:
            e[2] += 4.0 * (-0.18 - tbot)
        v = self.kp * e + self.xi
        du_dec = self.Kinv @ v
        # if temperature is dangerously low, raise all MVs (all K3j > 0)
        if tbot < -0.32:
            lift = min(0.02, 0.08 * (-0.28 - tbot))
            du_dec = du_dec + lift
        if tbot < -0.40:
            du_dec = np.maximum(du_dec, 0.008)
        u_cmd = np.clip(self.u + du_dec, self.umin, self.umax)
        du = np.clip(u_cmd - self.u, -self.rate, self.rate)
        u_new = np.clip(self.u + du, self.umin, self.umax)
        sat_hi = u_new >= self.umax - 1e-9
        sat_lo = u_new <= self.umin + 1e-9
        freeze = (sat_hi & (e > 0)) | (sat_lo & (e < 0))
        # freeze all I if bottoms is being rescued
        if tbot < -0.35:
            freeze = np.ones(3, dtype=bool)
            self.xi[2] = max(self.xi[2], 0.05)
        self.xi = self.xi + self.ki * e * self.dt * (~freeze)
        self.xi = np.clip(self.xi, -0.5, 0.5)
        self.u = u_new
        return self.u.copy()