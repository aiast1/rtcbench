import numpy as np

class Controller:
    def __init__(self, brief):
        self.dt = float(brief.sample_time)
        self.umin = 0.0
        self.umax = 30.0
        self.u0 = 14.22
        self.kp = 2.4
        self.ti = 120.0
        self.sp_tau = 25.0
        self.reset()

    def reset(self):
        self.u = float(self.u0)
        self.integ = 0.0
        self.sp_f = 6.5
        self.y_f = 6.5
        self.last_good_y = 6.5
        self.last_good_h = 17.5
        self.initialized = False

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        quality = np.asarray(quality, dtype=bool)

        ph_sp = float(r[0]) if np.isfinite(r[0]) else self.sp_f
        # never chase consent edges: delay + mismatch will overshoot
        ph_sp = float(np.clip(ph_sp, 5.6, 8.7))

        ph_ok = bool(quality[0]) and np.isfinite(y[0])
        ph = float(y[0]) if ph_ok else self.last_good_y
        if ph_ok:
            self.last_good_y = ph

        h_ok = (y.shape[0] > 1) and bool(quality[1]) and np.isfinite(y[1])
        h = float(y[1]) if h_ok else self.last_good_h
        if h_ok:
            self.last_good_h = h

        if not self.initialized:
            self.sp_f = ph_sp
            self.y_f = ph
            self.integ = 0.0
            self.initialized = True

        a_sp = self.dt / (self.sp_tau + self.dt)
        self.sp_f = (1.0 - a_sp) * self.sp_f + a_sp * ph_sp
        a_y = self.dt / (15.0 + self.dt)
        self.y_f = (1.0 - a_y) * self.y_f + a_y * ph

        err = self.sp_f - self.y_f
        gain_scale = 0.7 + 0.6 / (1.0 + 0.4 * (self.y_f - 7.0) ** 2)
        if self.y_f > 8.2:
            gain_scale *= 0.45
        kp = self.kp * gain_scale

        p = kp * err
        u_unsat = self.u0 + p + kp * self.integ / self.ti

        # high-pH cutback (true pH leads the analyzer)
        if ph > 8.4:
            u_unsat = min(u_unsat, 12.0)
        if ph > 8.8:
            u_unsat = min(u_unsat, 8.0)
        if ph > 9.2:
            u_unsat = min(u_unsat, 3.0)
        if ph > 9.6:
            u_unsat = 0.0
        if ph < 5.6:
            u_unsat = max(u_unsat, 18.0)
        if ph < 5.1:
            u_unsat = max(u_unsat, 24.0)

        # level [5, 30]; more base raises level
        if h > 23.0:
            u_unsat = min(u_unsat, 11.0)
        if h > 25.5:
            u_unsat = min(u_unsat, 6.0)
        if h > 27.5:
            u_unsat = min(u_unsat, 2.0)
        if h < 10.5:
            u_unsat = max(u_unsat, 16.0)
        if h < 8.0:
            u_unsat = max(u_unsat, 22.0)

        u_sat = float(np.clip(u_unsat, self.umin, self.umax))

        would_sat_hi = (u_unsat >= self.umax - 1e-9) and (err > 0)
        would_sat_lo = (u_unsat <= self.umin + 1e-9) and (err < 0)
        if not (would_sat_hi or would_sat_lo):
            self.integ += err * self.dt
        if abs(kp) > 1e-9:
            i_des = (u_sat - self.u0 - p) * self.ti / kp
            self.integ = 0.7 * self.integ + 0.3 * i_des
        self.integ = float(np.clip(self.integ, -250.0, 250.0))

        du_max = 0.28
        du = float(np.clip(u_sat - self.u, -du_max, du_max))
        self.u = float(np.clip(self.u + du, self.umin, self.umax))
        return np.array([self.u], dtype=float)