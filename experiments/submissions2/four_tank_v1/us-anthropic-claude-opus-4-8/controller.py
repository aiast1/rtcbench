import numpy as np


class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = float(brief.sample_time)
        self.u_lo = 0.0
        self.u_hi = 10.0
        self.u0 = np.array([3.0, 3.0], dtype=float)

        # PI gains - a bit more aggressive since prior run had margin.
        self.Kp = np.array([1.3, 1.3])
        self.Ki = np.array([0.10, 0.10])

        # slew limit on our command (V per step)
        self.slew = 0.8

        # setpoint filter time constant
        self.sp_tau = 6.0

    def reset(self):
        self.u = self.u0.copy()
        self.integ = np.zeros(2)
        self.y_filt = None
        self.r_filt = None
        self.last_good_y = None
        self.init_done = False

    def _filter(self, prev, new, tau):
        a = self.dt / (tau + self.dt)
        return prev + a * (new - prev)

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        q = np.asarray(quality, dtype=bool)

        meas = y[:2].copy()

        if self.last_good_y is None:
            self.last_good_y = meas.copy()
        for i in range(2):
            if not q[i] or not np.isfinite(meas[i]):
                meas[i] = self.last_good_y[i]
            else:
                self.last_good_y[i] = meas[i]

        # measurement filter (noise/quantization)
        if self.y_filt is None:
            self.y_filt = meas.copy()
        else:
            self.y_filt = self._filter(self.y_filt, meas, 3.0)

        sp = r[:2].copy()
        if self.r_filt is None:
            init_sp = np.where(np.isfinite(sp), sp, self.y_filt)
            self.r_filt = init_sp.copy()
        for i in range(2):
            if not np.isfinite(sp[i]):
                sp[i] = self.r_filt[i]
        self.r_filt = self._filter(self.r_filt, sp, self.sp_tau)

        if not self.init_done:
            err0 = self.r_filt - self.y_filt
            # bumpless start: PI output equals u0 initially
            self.integ = (self.u0 - self.Kp * err0)
            self.init_done = True

        err = self.r_filt - self.y_filt

        self.integ += self.Ki * err * self.dt
        u_unsat = self.Kp * err + self.integ

        u_sat = np.clip(u_unsat, self.u_lo, self.u_hi)

        # anti-windup: back-calculate integral when saturated
        for i in range(2):
            if u_sat[i] != u_unsat[i]:
                self.integ[i] = u_sat[i] - self.Kp[i] * err[i]

        # keep a safety margin from overflow: if measured level near top,
        # bias command down to protect unmeasured tanks from filling.
        for i in range(2):
            if self.y_filt[i] > 18.0:
                u_sat[i] = min(u_sat[i], self.u0[i])

        du = np.clip(u_sat - self.u, -self.slew, self.slew)
        self.u = self.u + du
        self.u = np.clip(self.u, self.u_lo, self.u_hi)

        return self.u.copy()