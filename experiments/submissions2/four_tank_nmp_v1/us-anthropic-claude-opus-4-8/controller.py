import numpy as np


class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = float(brief.sample_time)
        self.u_min = 0.0
        self.u_max = 10.0
        self.u0 = np.array([3.0, 3.0], dtype=float)

        # The unmeasured upper tanks (h3, h4) overflow at 20 cm too and get
        # MOST of each pump's flow. If we drive a pump too hard for too long
        # the upper tank overflows even though the measured lower tank looks
        # fine -> "invisible" safety violation. So the real constraint is on
        # cumulative/high pump voltage, not on the visible level.
        #
        # Fix: keep pump voltages conservative, add a strong cap on the
        # command, and slow integral action so we never push the upper
        # tanks toward overflow. Detune for worst-case robustness.

        self.Kp = 0.25
        self.Ki = 0.012
        self.u_slew = 0.4

        # Hard operating cap on pump voltage to protect unmeasured upper
        # tanks from overflow. Nominal steady is ~3V; setpoints near mid-range
        # need only modest increases. Cap well below max.
        self.u_cap = 6.0

    def reset(self):
        self.integ = np.zeros(2)
        self.u_prev = self.u0.copy()
        self.y_filt = None
        self.last_r = None

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        q = np.asarray(quality, dtype=bool)

        # --- measurement handling ---
        if self.y_filt is None:
            self.y_filt = np.array([y[0], y[1]], dtype=float)

        meas = self.y_filt.copy()
        for i in range(2):
            if q[i] and np.isfinite(y[i]):
                a = 0.4
                self.y_filt[i] = (1 - a) * self.y_filt[i] + a * y[i]
            meas[i] = self.y_filt[i]

        # --- setpoint handling ---
        sp = np.array([meas[0], meas[1]], dtype=float)
        for i in range(2):
            if np.isfinite(r[i]):
                sp[i] = r[i]
        if self.last_r is None:
            self.last_r = sp.copy()
        for i in range(2):
            d = sp[i] - self.last_r[i]
            mx = 0.3
            if d > mx:
                d = mx
            elif d < -mx:
                d = -mx
            self.last_r[i] += d
        spf = self.last_r

        err = spf - meas

        # --- PI control with anti-windup, capped output ---
        u = self.u0.copy()
        for i in range(2):
            integ_new = self.integ[i] + self.Ki * err[i] * self.dt
            raw = self.u0[i] + self.Kp * err[i] + integ_new

            hi = self.u_cap
            lo = self.u_min
            if raw > hi:
                if err[i] < 0:
                    self.integ[i] = integ_new
                clamped = hi
            elif raw < lo:
                if err[i] > 0:
                    self.integ[i] = integ_new
                clamped = lo
            else:
                self.integ[i] = integ_new
                clamped = raw

            u[i] = clamped

        # limit integral magnitude (conservative to protect upper tanks)
        self.integ = np.clip(self.integ, -2.5, 2.5)

        # --- slew limit ---
        for i in range(2):
            d = u[i] - self.u_prev[i]
            if d > self.u_slew:
                d = self.u_slew
            elif d < -self.u_slew:
                d = -self.u_slew
            u[i] = self.u_prev[i] + d

        u = np.clip(u, self.u_min, self.u_cap)
        self.u_prev = u.copy()
        return u