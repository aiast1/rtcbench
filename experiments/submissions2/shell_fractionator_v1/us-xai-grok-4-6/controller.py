import numpy as np

class Controller:
    def __init__(self, brief):
        self.dt = float(brief.sample_time)
        self.umin = np.array([-0.35, -0.35, -0.05], dtype=float)
        self.umax = np.array([0.45, 0.45, 0.5], dtype=float)
        self.du_max = 0.005
        K = np.array([
            [4.05, 1.77, 5.88],
            [5.39, 5.72, 6.90],
            [4.38, 4.42, 7.20],
        ], dtype=float)
        self.Kinv = np.linalg.pinv(K)
        self.kp = 0.016
        self.ki = 0.00035
        self.reset()

    def reset(self):
        self.u = np.array([0.0, 0.0, 0.08], dtype=float)
        self.ei = np.zeros(3)
        self.y_f = np.zeros(3)
        self.have_y = False

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float).copy()
        r = np.asarray(r, dtype=float).copy()
        q = np.asarray(quality, dtype=bool)
        for i in range(3):
            if not q[i] or not np.isfinite(y[i]):
                y[i] = self.y_f[i] if self.have_y else 0.0
            if not np.isfinite(r[i]):
                r[i] = 0.0
        r[2] = max(float(r[2]), 0.08)
        if not self.have_y:
            self.y_f = y.copy()
            self.have_y = True
        else:
            self.y_f = 0.92 * self.y_f + 0.08 * y
        e = r - self.y_f
        if self.y_f[2] < 0.0:
            e[2] += (0.05 - self.y_f[2]) * 3.0
        if self.y_f[2] < -0.15:
            e[2] += (-0.02 - self.y_f[2]) * 10.0
        self.ei = np.clip(self.ei + e * self.dt, -4.0, 4.0)
        v = self.kp * e + self.ki * self.ei
        du = self.Kinv @ v
        du[2] += 0.025 * e[2]
        # keep a heating bias on reflux duty
        if self.u[2] < 0.12:
            du[2] += 0.002
        du = np.clip(du, -self.du_max, self.du_max)
        u_des = np.clip(self.u + du, self.umin, self.umax)
        if self.y_f[2] < -0.1:
            u_des[2] = max(u_des[2], 0.15)
        if self.y_f[2] < -0.22:
            u_des[2] = max(u_des[2], 0.28)
            u_des[0] = max(u_des[0], 0.0)
            u_des[1] = max(u_des[1], 0.0)
        sat = (u_des <= self.umin + 1e-9) | (u_des >= self.umax - 1e-9)
        for i in range(3):
            if sat[i]:
                self.ei[i] *= 0.95
        self.u = u_des
        return self.u.copy()