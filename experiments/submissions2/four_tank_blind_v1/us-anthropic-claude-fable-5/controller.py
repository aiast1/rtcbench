import numpy as np


class Controller:
    """
    Blind commissioning controller for the quadruple-tank process, round 3.

    Round-2 data: still zero violations, duty comfortably clear, tracking is
    ~93% of the cost. Push gains further while keeping the safety machinery:
      * Diagonal PI pairing (pump 1 -> tank 1, pump 2 -> tank 2).
      * Faster PI, lighter measurement/setpoint filtering, no derivative
        (measurement is noisy, delayed, quantized).
      * Conditional integration + back-calculation anti-windup.
      * Slew limit + tiny move deadband keep actuator duty under the trip.
      * Unmeasured upper tanks protected by capping sustained pump command;
        measured levels near 20 cm progressively choke both pumps.
    """

    def __init__(self, brief):
        self.dt = float(getattr(brief, "sample_time", 2.0)) or 2.0
        self.n_u = 2
        self.n_y = 2

        self.u_lo = 0.0
        self.u_hi = 7.0          # ceiling protects unmeasured upper tanks
        self.u_start = np.array([3.0, 3.0], dtype=float)

        # PI tuning (V per cm) - faster again than round 2
        self.Kp = 1.05
        self.Ti = 30.0                      # s
        self.Ki = self.Kp / self.Ti
        self.Tt = 12.0                      # anti-windup back-calc time

        # Filters
        self.tau_y = 3.0                    # measurement filter (s)
        self.tau_r = 3.0                    # setpoint filter (s)
        self.a_y = self.dt / (self.dt + self.tau_y)
        self.a_r = self.dt / (self.dt + self.tau_r)

        # Actuator shaping
        self.du_max = 1.0                   # V per 2 s step
        self.move_db = 0.006                # V

        # Safety thresholds on measured levels
        self.h_soft = 17.5
        self.h_hard = 19.2
        self.h_max = 20.0

        self.reset()

    def reset(self):
        self.I = self.u_start.copy()
        self.u_out = self.u_start.copy()
        self.yf = None
        self.rf = None
        self.y_good = None
        self.first = True

    # ------------------------------------------------------------------ #

    def _sanitize(self, y, quality):
        y = np.asarray(y, dtype=float).copy()
        q = np.asarray(quality, dtype=bool) if quality is not None else \
            np.ones_like(y, dtype=bool)
        if self.y_good is None:
            self.y_good = np.clip(np.nan_to_num(y, nan=10.0), 0.0, self.h_max)
        for i in range(len(y)):
            bad = (i < len(q) and not q[i]) or not np.isfinite(y[i])
            if bad:
                y[i] = self.y_good[i]
            else:
                y[i] = float(np.clip(y[i], 0.0, self.h_max))
                self.y_good[i] = y[i]
        return y

    def _setpoint(self, r):
        r = np.asarray(r, dtype=float)
        if self.rf is None:
            rr = np.where(np.isfinite(r), r, self.yf)
            return np.clip(rr, 1.0, 16.5)
        tgt = np.where(np.isfinite(r), r, self.rf)
        tgt = np.clip(tgt, 1.0, 16.5)       # never demand a level near trip
        return self.rf + self.a_r * (tgt - self.rf)

    # ------------------------------------------------------------------ #

    def step(self, t, y, r, quality):
        y = self._sanitize(y, quality)

        if self.yf is None:
            self.yf = y.copy()
        else:
            self.yf = self.yf + self.a_y * (y - self.yf)

        self.rf = self._setpoint(r)

        e = self.rf - self.yf

        # ---- PI with anti-windup -------------------------------------- #
        u_raw = self.Kp * e + self.I

        # Safety governor: choke pumps as any measured level nears overflow.
        worst = float(np.max(self.yf))
        g_all = np.clip((self.h_hard - worst) / (self.h_hard - self.h_soft),
                        0.0, 1.0)
        g_pair = np.clip((self.h_hard - self.yf) /
                         (self.h_hard - self.h_soft), 0.0, 1.0)
        u_cap = self.u_lo + np.minimum(g_all, g_pair) * (self.u_hi - self.u_lo)

        u_sat = np.clip(u_raw, self.u_lo, u_cap)

        for i in range(self.n_u):
            saturating_up = u_raw[i] > u_sat[i] and e[i] > 0.0
            saturating_dn = u_raw[i] < u_sat[i] and e[i] < 0.0
            if not (saturating_up or saturating_dn):
                self.I[i] += self.Ki * self.dt * e[i]
            self.I[i] += (u_sat[i] - u_raw[i]) * self.dt / self.Tt
            self.I[i] = float(np.clip(self.I[i], self.u_lo, self.u_hi))

        u_cmd = np.clip(self.Kp * e + self.I, self.u_lo, u_cap)

        # ---- slew limiting -------------------------------------------- #
        du = np.clip(u_cmd - self.u_out, -self.du_max, self.du_max)
        u_new = self.u_out + du

        # ---- move deadband -------------------------------------------- #
        emergency = worst > self.h_soft
        for i in range(self.n_u):
            if not emergency and abs(u_new[i] - self.u_out[i]) < self.move_db:
                u_new[i] = self.u_out[i]

        # Hard emergency: measured level essentially at trip -> drive down.
        if worst >= 19.6:
            u_new = np.minimum(u_new, np.maximum(self.u_out - self.du_max, 0.0))

        u_new = np.clip(u_new, 0.0, 10.0)

        if self.first:
            self.first = False
            u_new = np.clip(u_new, self.u_start - 0.25, self.u_start + 0.25)

        self.u_out = u_new
        return u_new.copy()