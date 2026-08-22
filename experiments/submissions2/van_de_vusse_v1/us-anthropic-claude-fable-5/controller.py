import numpy as np


class Controller:
    """
    Two-loop PI controller for the van de Vusse CSTR.

      Loop 1: C_B (analyzer, slow/delayed)  -> F/V   (FCV-101)
      Loop 2: T   (thermocouple, fast/noisy)-> Q_K   (TCV-201, cooling only)

    Design notes:
      * Operate strictly on the LEFT branch of the non-monotonic C_B vs F/V map:
        F/V is capped at 17 h^-1 (nominal peak ~14.7) so integral windup can never
        push the loop onto the wrong-gain branch.
      * Conditional integration (anti-windup) on both loops.
      * EWMA filtering of measurements; filters only updated on good-quality samples.
      * Output deadband + rate limiting to keep actuator travel far below the duty
        limit (no chattering on sensor noise).
      * Safety override: if filtered T exceeds 138 degC, drive Q_K toward full
        cooling with a fast rate limit. Since Q_K <= 0 (cooling only), the jacket
        temperature T_K cannot exceed the reactor temperature at steady state, so
        keeping T well below 150 keeps the unmeasured T_K safe as well.
      * No derivative action: the scored measurements are noisy and delayed.
      * Bumpless start: integrators are seeded with the as-found actuator values.
    """

    def __init__(self, brief):
        self.dt = 10.0
        try:
            st = getattr(brief, "sample_time", None)
            if st is not None and float(st) > 0.0:
                self.dt = float(st)
        except Exception:
            pass

        self.u_start = np.array([14.19, -1113.5], dtype=float)

        # ---- measurement filters (EWMA coefficients per sample) ----
        self.a_cb = 0.30      # analyzer: modest filtering (already slow)
        self.a_T = 0.15       # thermocouple: heavy filtering vs noise

        # ---- PI tuning ----
        self.Kp1 = 7.0        # (1/h) per (mol/L)
        self.Ti1 = 200.0      # s
        self.Kp2 = 120.0      # (kJ/h) per K
        self.Ti2 = 250.0      # s

        # ---- actuator limits ----
        self.F_lo, self.F_hi = 3.0, 17.0        # soft high cap: stay on left branch
        self.F_hard_lo, self.F_hard_hi = 3.0, 35.0
        self.Q_lo, self.Q_hi = -9000.0, 0.0

        # ---- travel management: rate limits per step & deadbands ----
        self.rl_F = 0.4          # 1/h per step
        self.rl_Q_up = 200.0     # kJ/h per step toward 0 (less cooling)
        self.rl_Q_dn = 800.0     # kJ/h per step toward -9000 (more cooling)
        self.rl_Q_safe = 1500.0  # fast cooling ramp under safety override
        self.db_F = 0.05         # 1/h
        self.db_Q = 25.0         # kJ/h

        # ---- safety ----
        self.T_safe = 138.0      # degC, override threshold (limit is 150)

        self.reset()

    def reset(self):
        self.cb_f = None
        self.T_f = None
        self.I1 = float(self.u_start[0])
        self.I2 = float(self.u_start[1])
        self.last = self.u_start.copy()
        self.r_hold = np.array([1.09, 114.19], dtype=float)

    # -------- helpers --------
    @staticmethod
    def _move(last, target, db, rl_up, rl_dn):
        """Deadband + asymmetric rate limit relative to current command."""
        d = target - last
        if abs(d) <= db:
            return last
        if d > 0.0:
            d = min(d, rl_up)
        else:
            d = max(d, -rl_dn)
        return last + d

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        if quality is None:
            q = np.array([True, True])
        else:
            q = np.asarray(quality).astype(bool)
            if q.size < 2:
                q = np.array([True, True])

        # ---- setpoint handling (hold last valid) ----
        if r is not None:
            r = np.asarray(r, dtype=float)
            for i in range(min(2, r.size)):
                if np.isfinite(r[i]):
                    self.r_hold[i] = r[i]
        r_cb, r_T = self.r_hold[0], self.r_hold[1]

        # ---- filter updates (only on good samples) ----
        good_cb = bool(q[0]) and y.size > 0 and np.isfinite(y[0])
        good_T = bool(q[1]) and y.size > 1 and np.isfinite(y[1])

        if self.cb_f is None:
            self.cb_f = y[0] if good_cb else r_cb
        elif good_cb:
            self.cb_f += self.a_cb * (y[0] - self.cb_f)

        if self.T_f is None:
            self.T_f = y[1] if good_T else r_T
        elif good_T:
            self.T_f += self.a_T * (y[1] - self.T_f)

        e1 = r_cb - self.cb_f
        e2 = r_T - self.T_f

        # ================= Loop 1: C_B -> F/V =================
        u1_un = self.I1 + self.Kp1 * e1
        integrate1 = True
        if (u1_un >= self.F_hi and e1 > 0.0) or (u1_un <= self.F_lo and e1 < 0.0):
            integrate1 = False  # conditional integration (anti-windup)
        if integrate1:
            self.I1 += (self.Kp1 * self.dt / self.Ti1) * e1
        self.I1 = float(np.clip(self.I1, self.F_lo, self.F_hi))
        u1_target = float(np.clip(self.I1 + self.Kp1 * e1, self.F_lo, self.F_hi))

        # ================= Loop 2: T -> Q_K =================
        u2_un = self.I2 + self.Kp2 * e2
        integrate2 = True
        if (u2_un >= self.Q_hi and e2 > 0.0) or (u2_un <= self.Q_lo and e2 < 0.0):
            integrate2 = False
        safety = self.T_f > self.T_safe
        if safety:
            integrate2 = False
        if integrate2:
            self.I2 += (self.Kp2 * self.dt / self.Ti2) * e2
        self.I2 = float(np.clip(self.I2, self.Q_lo, self.Q_hi))
        u2_target = float(np.clip(self.I2 + self.Kp2 * e2, self.Q_lo, self.Q_hi))

        if safety:
            # Force full cooling; bleed integrator so recovery is bumpless.
            u2_target = self.Q_lo
            self.I2 = min(self.I2, float(self.last[1]))

        # ---- deadband + rate limiting (duty / travel management) ----
        u1_cmd = self._move(float(self.last[0]), u1_target,
                            self.db_F, self.rl_F, self.rl_F)
        rl_dn = self.rl_Q_safe if safety else self.rl_Q_dn
        u2_cmd = self._move(float(self.last[1]), u2_target,
                            self.db_Q, self.rl_Q_up, rl_dn)

        # ---- hard clamps ----
        u1_cmd = float(np.clip(u1_cmd, self.F_hard_lo, self.F_hard_hi))
        u2_cmd = float(np.clip(u2_cmd, self.Q_lo, self.Q_hi))

        self.last = np.array([u1_cmd, u2_cmd], dtype=float)
        return self.last.copy()