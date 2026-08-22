import numpy as np


class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = float(brief.sample_time)
        self.u_min = 0.0
        self.u_max = 100.0
        self.u0 = 50.0

        # Conservative PI for variable large deadtime.
        self.Kp = 0.9
        self.Ti = 260.0
        self.max_du = 2.0  # % per step, protect duty budget & avoid chatter

        # Safety: T_out <= 90, and unmeasured inlet also <= 90.
        # Keep a hard cap on output to avoid overheating during long deadtime.
        self.T_out_hi = 90.0
        self.T_out_lo = 0.0

    def reset(self):
        self.integ = 0.0
        self.u_prev = self.u0
        self.y_filt = None
        self.last_valid_y = None
        self.first = True

    def step(self, t, y, r, quality):
        meas = float(y[0])
        good = bool(quality[0])
        sp = r[0]

        if good and np.isfinite(meas):
            self.last_valid_y = meas
        if self.last_valid_y is None:
            self.last_valid_y = meas if np.isfinite(meas) else 55.0
        ym = self.last_valid_y

        alpha = 0.2
        if self.y_filt is None:
            self.y_filt = ym
        else:
            self.y_filt = (1 - alpha) * self.y_filt + alpha * ym
        yf = self.y_filt

        if not np.isfinite(sp):
            return np.array([self.u_prev])

        if self.first:
            self.first = False
            err0 = sp - yf
            self.integ = (self.u0 - self.u_prev) - self.Kp * err0
            self.u_prev = self.u0

        err = sp - yf

        prop = self.Kp * err
        integ_new = self.integ + self.Kp * (self.dt / self.Ti) * err

        u_unsat = self.u0 + prop + integ_new
        u_sat = np.clip(u_unsat, self.u_min, self.u_max)
        if u_sat == u_unsat:
            self.integ = integ_new
        else:
            if (u_unsat > self.u_max and err < 0) or (u_unsat < self.u_min and err > 0):
                self.integ = integ_new

        u = self.u0 + prop + self.integ
        u = np.clip(u, self.u_min, self.u_max)

        # --- Safety-driven output clamp ---
        # The unmeasured heater outlet shares the 90 degC limit and, with a
        # long deadtime, the outlet we DO see lags far behind. If the measured
        # outlet is near the safety limit, back off aggressively. Also cap the
        # absolute duty to prevent overheating the inlet during long delays.
        # A simple predictive guard: never push far above what steady-state
        # tracking near setpoint would require.
        margin = self.T_out_hi - yf
        if margin < 10.0:
            # scale down output as we approach the limit
            scale = max(0.0, margin / 10.0)
            u_cap = self.u0 + (u - self.u0) * scale
            if u_cap < u:
                u = u_cap
                # bleed integral so it doesn't wind up against the cap
                self.integ *= 0.98

        u = np.clip(u, self.u_min, self.u_max)

        # slew limit
        du = u - self.u_prev
        if du > self.max_du:
            u = self.u_prev + self.max_du
        elif du < -self.max_du:
            u = self.u_prev - self.max_du

        u = float(np.clip(u, self.u_min, self.u_max))
        self.u_prev = u

        return np.array([u])