import numpy as np


class Controller:
    """
    Van de Vusse CSTR - two-loop design.

    Loop A (C_B -> F/V, FCV-101)
        The static map C_B(F/V) is non-monotonic (peak near F/V ~ 14-15).
        A nominal first-principles STATIC model (steady-state solution of the
        A->B->C / 2A->D kinetics at the measured reactor temperature) is used
        as a feedforward inverse: given the C_B setpoint we solve for the F/V
        on the RISING (left) branch.  One scalar gain 's' of that model is
        identified from the initial steady state (the plant is handed over at
        steady state, which is a perfect anchor point).  A slow PI trim on top
        removes all remaining model mismatch and the mid-run disturbance.
        The command is hard-clamped below the model peak so the loop can never
        walk onto the wrong-sign branch.

    Loop B (T -> Q_K, TCV-201)
        The reactor thermal gain is only ~1e-3 degC per kJ/h, so the jacket PI
        needs gains of order 1e3 kJ/h/degC to hold the temperature while the
        feed rate is being swung over 2:1.  PI only (no derivative - the
        thermocouple is noisy/quantized), moderate measurement filtering, a
        move deadband and a slew limit keep the valve duty far below the limit.

    Safety: T (and hence the uninstrumented T_K, which sits BELOW T whenever
    Q_K <= 0) is kept near 114 degC.  Hard overrides slam cooling to the stop
    and raise the dilution rate if T ever climbs toward the 150 degC limit.
    """

    # ---------------- nominal model constants ----------------
    K10 = 1.287e12
    K20 = 1.287e12
    K30 = 9.043e9
    E1 = -9758.3
    E2 = -9758.3
    E3 = -8560.0
    CA0 = 5.1

    def __init__(self, brief):
        self.brief = brief
        dt = float(getattr(brief, "sample_time", 10.0))
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 10.0
        self.dt = dt

        # actuator hard limits
        self.f_lo, self.f_hi = 3.0, 35.0
        self.q_lo, self.q_hi = -9000.0, 0.0

        # operating window for the feed (stay on rising branch of C_B(F))
        self.f_min_op = 3.4
        self.f_max_op = 15.5

        # handover point
        self.f0 = 14.19
        self.q0 = -1113.5

        # ---- C_B loop (outer) ----
        self.Kp_cb = 10.0          # (1/h) per (mol/L)
        self.Ki_cb = 0.10          # (1/h) per (mol/L) per s
        self.cb_alpha = 0.25       # measurement filter (on fresh samples)
        self.trim_lim = 4.0
        self.p_lim_cb = 3.0
        self.f_slew = 1.0          # 1/h per sample
        self.f_db = 0.03           # 1/h move deadband

        # ---- T loop (inner) ----
        self.Kp_T = 900.0          # kJ/h per degC
        self.Ki_T = 10.0           # kJ/h per degC per s
        self.t_alpha = 0.25        # T filter (~30 s)
        self.q_slew = 1500.0       # kJ/h per sample
        self.q_db = 60.0           # kJ/h move deadband

        # safety guards on reactor temperature
        self.T_warn = 132.0
        self.T_hot = 138.0
        self.T_trip = 143.0

        self._kin_cache = {}
        self._peak_cache = {}

        self.reset()

    # ================================================================
    def reset(self):
        self.f_out = self.f0
        self.q_out = self.q0

        self.trim = 0.0
        self.i_T = 0.0

        self.cb_f = None
        self.T_f = None
        self.T_slow = None

        self.cb_good = None
        self.T_good = None

        self.s = None              # model output scaling
        self.s_ready = False

        self.r_cb = None
        self.r_T = 114.19

        self.f_ff = self.f0
        self.t_last = None
        self.n = 0

    # ---------------- helpers ----------------
    @staticmethod
    def _clip(x, lo, hi):
        if not np.isfinite(x):
            return lo if lo > 0 else 0.5 * (lo + hi)
        return lo if x < lo else (hi if x > hi else x)

    def _kin(self, Tc):
        key = round(float(Tc) * 4.0)
        v = self._kin_cache.get(key)
        if v is None:
            Tk = key / 4.0 + 273.15
            k1 = self.K10 * np.exp(self.E1 / Tk)
            k2 = self.K20 * np.exp(self.E2 / Tk)
            k3 = self.K30 * np.exp(self.E3 / Tk)
            v = (k1, k2, k3)
            if len(self._kin_cache) < 4000:
                self._kin_cache[key] = v
        return v

    def _cb_nom(self, F, kin):
        k1, k2, k3 = kin
        F = np.asarray(F, dtype=float)
        b = k1 + F
        disc = b * b + 8.0 * k3 * F * self.CA0
        disc = np.maximum(disc, 0.0)
        CA = (-b + np.sqrt(disc)) / (4.0 * k3)
        CA = np.maximum(CA, 0.0)
        return k1 * CA / (F + k2)

    def _peak(self, Tc):
        key = round(float(Tc) * 4.0)
        v = self._peak_cache.get(key)
        if v is None:
            kin = self._kin(key / 4.0)
            grid = np.arange(3.0, 20.001, 0.2)
            vals = self._cb_nom(grid, kin)
            i = int(np.argmax(vals))
            Fp = float(grid[i])
            if 0 < i < len(grid) - 1:
                y0, y1, y2 = vals[i - 1], vals[i], vals[i + 1]
                den = (y0 - 2.0 * y1 + y2)
                if abs(den) > 1e-12:
                    Fp = Fp - 0.5 * 0.2 * (y2 - y0) / den
            Fp = self._clip(Fp, 6.0, 19.0)
            cbp = float(self._cb_nom(np.array([Fp]), kin)[0])
            v = (Fp, cbp)
            if len(self._peak_cache) < 4000:
                self._peak_cache[key] = v
        return v

    def _invert(self, cb_target, Tc):
        """F/V on the rising branch that yields cb_target (model, scaled)."""
        kin = self._kin(Tc)
        Fp, cbp = self._peak(Tc)
        Fp = min(Fp, self.f_max_op)
        cbp = float(self._cb_nom(np.array([Fp]), kin)[0])
        tgt = cb_target / max(self.s, 1e-3)
        if tgt >= cbp:
            return Fp
        lo = self.f_min_op
        cb_lo = float(self._cb_nom(np.array([lo]), kin)[0])
        if tgt <= cb_lo:
            return lo
        hi = Fp
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if float(self._cb_nom(np.array([mid]), kin)[0]) < tgt:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ================================================================
    def step(self, t, y, r, quality):
        dt = self.dt
        if self.t_last is not None:
            d = float(t) - self.t_last
            if 0.1 < d < 10.0 * self.dt:
                dt = d
        self.t_last = float(t)
        self.n += 1

        y = np.asarray(y, dtype=float).ravel()
        r = np.asarray(r, dtype=float).ravel()
        if quality is None:
            q = np.ones(max(y.size, 2), dtype=bool)
        else:
            q = np.asarray(quality).ravel().astype(bool)
            if q.size < y.size:
                q = np.concatenate([q, np.ones(y.size - q.size, dtype=bool)])

        cb_raw = y[0] if y.size > 0 else np.nan
        T_raw = y[1] if y.size > 1 else np.nan
        cb_ok = bool(q[0]) and np.isfinite(cb_raw) and (0.0 <= cb_raw <= 1.6)
        T_ok = (y.size > 1 and bool(q[1]) and np.isfinite(T_raw)
                and (60.0 <= T_raw <= 200.0))

        if cb_ok:
            self.cb_good = float(cb_raw)
        if T_ok:
            self.T_good = float(T_raw)

        cb_m = self.cb_good if self.cb_good is not None else 1.09
        T_m = self.T_good if self.T_good is not None else 114.19

        # ---------------- filtering ----------------
        if self.cb_f is None:
            self.cb_f = cb_m
        elif cb_ok:
            self.cb_f += self.cb_alpha * (cb_m - self.cb_f)

        if self.T_f is None:
            self.T_f = T_m
            self.T_slow = T_m
        else:
            if T_ok:
                self.T_f += self.t_alpha * (T_m - self.T_f)
            self.T_slow += 0.03 * (self.T_f - self.T_slow)

        Tmod = self._clip(self.T_slow, 95.0, 140.0)

        # ---------------- setpoints ----------------
        if r.size > 0 and np.isfinite(r[0]):
            self.r_cb = self._clip(float(r[0]), 0.05, 1.5)
        elif self.r_cb is None:
            self.r_cb = self.cb_f
        if r.size > 1 and np.isfinite(r[1]):
            self.r_T = self._clip(float(r[1]), 90.0, 145.0)

        # ---------------- model scaling identification ----------------
        if self.s is None:
            kin = self._kin(Tmod)
            base = float(self._cb_nom(np.array([self.f0]), kin)[0])
            self.s = self._clip(self.cb_f / max(base, 1e-3), 0.5, 2.2)
        elif (not self.s_ready):
            if float(t) < 400.0 and abs(self.f_out - self.f0) < 0.6 and cb_ok:
                kin = self._kin(Tmod)
                base = float(self._cb_nom(np.array([self.f0]), kin)[0])
                s_meas = self._clip(self.cb_f / max(base, 1e-3), 0.5, 2.2)
                self.s += 0.2 * (s_meas - self.s)
            elif float(t) >= 400.0:
                self.s_ready = True

        # ---------------- C_B loop ----------------
        e_cb = self.r_cb - self.cb_f
        self.f_ff = self._invert(self.r_cb, Tmod)

        p_cb = self._clip(self.Kp_cb * e_cb, -self.p_lim_cb, self.p_lim_cb)
        trim_c = self.trim + self.Ki_cb * e_cb * dt
        trim_c = self._clip(trim_c, -self.trim_lim, self.trim_lim)

        f_uns = self.f_ff + p_cb + trim_c
        f_des = self._clip(f_uns, self.f_min_op, self.f_max_op)
        if (f_uns > self.f_max_op and e_cb > 0.0) or \
           (f_uns < self.f_min_op and e_cb < 0.0):
            # windup protection: hold the integrator
            trim_c = self.trim
            f_uns = self.f_ff + p_cb + trim_c
            f_des = self._clip(f_uns, self.f_min_op, self.f_max_op)
        self.trim = trim_c

        # ---------------- T loop ----------------
        e_T = self.r_T - self.T_f
        iT_c = self.i_T + self.Ki_T * e_T * dt
        iT_c = self._clip(iT_c, -8000.0, 8000.0)

        q_uns = self.q0 + self.Kp_T * e_T + iT_c
        q_des = self._clip(q_uns, self.q_lo, self.q_hi)
        if (q_uns > self.q_hi and e_T > 0.0) or (q_uns < self.q_lo and e_T < 0.0):
            iT_c = self.i_T
            q_uns = self.q0 + self.Kp_T * e_T + iT_c
            q_des = self._clip(q_uns, self.q_lo, self.q_hi)
        self.i_T = iT_c

        # ---------------- safety overrides ----------------
        hot = False
        if self.T_f > self.T_warn:
            hot = True
            q_des = min(q_des, -3000.0)
        if self.T_f > self.T_hot:
            q_des = self.q_lo
        if self.T_f > self.T_trip:
            q_des = self.q_lo
            f_des = max(f_des, 20.0)   # dilute / cool with fresh feed

        # ---------------- move shaping (travel & duty) ----------------
        df = f_des - self.f_out
        if abs(df) < self.f_db and not hot:
            f_new = self.f_out
        else:
            if df > self.f_slew:
                df = self.f_slew
            elif df < -self.f_slew:
                df = -self.f_slew
            f_new = self.f_out + df
        f_new = self._clip(f_new, self.f_lo, self.f_hi)

        dq = q_des - self.q_out
        if abs(dq) < self.q_db and not hot:
            q_new = self.q_out
        else:
            sl = self.q_slew if not hot else 4.0 * self.q_slew
            if dq > sl:
                dq = sl
            elif dq < -sl:
                dq = -sl
            q_new = self.q_out + dq
        q_new = self._clip(q_new, self.q_lo, self.q_hi)

        self.f_out = f_new
        self.q_out = q_new
        return np.array([self.f_out, self.q_out], dtype=float)