import numpy as np


class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = float(brief.sample_time) if hasattr(brief, 'sample_time') else 2.0
        self.umin = 0.0
        self.umax = 10.0
        self.u0 = np.array([3.0, 3.0])
        self.hmax = 20.0
        self.hmin = 0.0
        # Push gains further; tracking dominates cost, effort still small.
        self.Kp = 1.15
        self.Ki = 0.024
        self.reset()

    def reset(self):
        self.u = self.u0.copy()
        self.i_term = np.zeros(2)
        self.last_valid = np.array([12.26, 12.78])
        self.prev_u = self.u0.copy()
        self.ema = np.array([12.26, 12.78])
        self.ema_init = False

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        q = np.asarray(quality, dtype=bool)

        meas = self.last_valid.copy()
        for i in range(2):
            if q[i] and np.isfinite(y[i]):
                meas[i] = y[i]
                self.last_valid[i] = y[i]

        if not self.ema_init:
            self.ema = meas.copy()
            self.ema_init = True
        else:
            alpha = 0.50
            self.ema = alpha * meas + (1.0 - alpha) * self.ema
        h = self.ema

        sp = np.array([r[0] if np.isfinite(r[0]) else h[0],
                       r[1] if np.isfinite(r[1]) else h[1]])
        sp = np.clip(sp, 1.0, 18.0)

        err = sp - h

        u_unsat = np.zeros(2)
        for i in range(2):
            p = self.Kp * err[i]
            i_new = self.i_term[i] + self.Ki * err[i] * self.dt
            out = self.u0[i] + p + i_new
            if out > self.umax:
                if err[i] < 0:
                    self.i_term[i] = i_new
            elif out < self.umin:
                if err[i] > 0:
                    self.i_term[i] = i_new
            else:
                self.i_term[i] = i_new
            u_unsat[i] = self.u0[i] + p + self.i_term[i]

        # Safety override for measured tanks (approach to overflow / empty)
        for i in range(2):
            if h[i] > 18.5:
                u_unsat[i] = min(u_unsat[i], self.u[i] - 0.5)
            if h[i] < 1.5:
                u_unsat[i] = max(u_unsat[i], self.u[i] + 0.5)

        u_cmd = np.clip(u_unsat, self.umin, self.umax)

        # Slew limiting - enough for fast steps, still smooth to keep duty low
        max_step = 0.55
        du = u_cmd - self.u
        du = np.clip(du, -max_step, max_step)
        self.u = self.u + du
        self.u = np.clip(self.u, self.umin, self.umax)

        self.prev_u = self.u.copy()
        return self.u.copy()