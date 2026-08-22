import numpy as np


class Controller:
    def __init__(self, brief):
        self.dt_nom = float(brief.sample_time)

        self.u0 = np.array([3.0, 3.0], dtype=float)
        self.u_min = np.array([1.65, 1.65], dtype=float)
        self.u_max = np.array([4.05, 4.05], dtype=float)

        # Static inverse around the nominal operating point.  The plant has
        # strong cross coupling because most pump flow is routed through the
        # diagonal upper tanks.
        self.inv_gain = np.array([
            [-0.215, 0.335],
            [0.343, -0.208],
        ], dtype=float)

        # Slightly faster than the initial commissioning settings while still
        # deliberately conservative for the inverse-response upper-tank paths.
        self.kp = np.array([0.070, 0.070], dtype=float)
        self.ki = 0.0050

        self.filter_tau = 6.0
        self.max_rate = 0.0065

        self.reset()

    def reset(self):
        self.initialized = False
        self.last_t = None
        self.yf = np.full(2, np.nan, dtype=float)
        self.ref = np.full(2, np.nan, dtype=float)
        self.z = np.zeros(2, dtype=float)
        self.u_last = self.u0.copy()

    def _levels(self):
        h = self.yf.copy()
        h[~np.isfinite(h)] = 12.0
        return np.clip(h, 0.0, 20.0)

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        quality = np.asarray(quality, dtype=bool)

        if not self.initialized:
            for i in range(2):
                good = (
                    i < y.size and i < quality.size and
                    bool(quality[i]) and np.isfinite(y[i])
                )
                self.yf[i] = np.clip(y[i], 0.0, 20.0) if good else 12.0

                if i < r.size and np.isfinite(r[i]):
                    self.ref[i] = float(r[i])
                else:
                    self.ref[i] = self.yf[i]

            self.last_t = float(t)
            self.initialized = True
            return self.u_last.copy()

        dt = float(t) - float(self.last_t)
        if not np.isfinite(dt) or dt <= 0.0:
            dt = self.dt_nom
        dt = float(np.clip(dt, 0.25 * self.dt_nom, 3.0 * self.dt_nom))
        self.last_t = float(t)

        for i in range(2):
            if i < r.size and np.isfinite(r[i]):
                self.ref[i] = float(r[i])

        valid = np.zeros(2, dtype=bool)
        alpha = dt / (self.filter_tau + dt)

        for i in range(2):
            good = (
                i < y.size and i < quality.size and
                bool(quality[i]) and np.isfinite(y[i])
            )
            if good:
                valid[i] = True
                yy = float(np.clip(y[i], 0.0, 20.0))
                self.yf[i] += alpha * (yy - self.yf[i])

        h = self._levels()
        err = self.ref - h

        # Quantization/noise deadband prevents needless valve movement.
        e_int = np.where(np.abs(err) >= 0.025, err, 0.0)
        e_int = np.where(valid, e_int, 0.0)
        e_p = np.where(valid, err, 0.0)

        z_trial = self.z + dt * self.ki * (self.inv_gain @ e_int)
        raw = self.u0 + z_trial + self.kp * e_p

        # Measured lower-tank high-level protection.  The unmeasured upper
        # tanks are protected primarily by conservative actuator limits.
        if h[0] > 15.0:
            b = h[0] - 15.0
            raw -= np.array([0.045 * b, 0.140 * b])

        if h[1] > 15.0:
            b = h[1] - 15.0
            raw -= np.array([0.140 * b, 0.045 * b])

        if h[0] > 18.4 or h[1] > 18.4:
            raw = np.minimum(raw, np.array([2.70, 2.70]))

        bounded = np.clip(raw, self.u_min, self.u_max)

        # Back-calculation prevents the static-decoupling integrators from
        # winding up when an actuator or safety limit is active.
        self.z = z_trial + 0.28 * (bounded - raw)

        du_limit = self.max_rate * dt
        u = self.u_last + np.clip(bounded - self.u_last, -du_limit, du_limit)
        u = np.clip(u, self.u_min, self.u_max)

        self.u_last = u
        return u.copy()