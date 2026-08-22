import numpy as np


class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = float(brief.sample_time)

        # Actuator limits
        self.fv_min, self.fv_max = 3.0, 35.0
        self.qk_min, self.qk_max = -9000.0, 0.0

        # Bumpless start
        self.fv0 = 14.19
        self.qk0 = -1113.5

        # Non-monotonic C_B vs F/V: peak around 14.7 h^-1.
        # Stay on the safe (lower-F/V) side to keep a monotonic
        # increasing relationship between F/V and C_B.
        self.fv_peak = 14.7

        # T safety limit; back off well below 150 for robustness
        self.T_limit = 150.0
        self.T_safe = 145.0

        self.reset()

    def reset(self):
        # --- C_B loop (moves F/V) ---
        self.cb_int = 0.0
        self.cb_last = None
        self.fv_cmd = self.fv0

        # --- T loop (moves Q_K) ---
        self.T_int = 0.0
        self.T_last = None
        self.qk_cmd = self.qk0

        # measurement holds
        self.cb_hold = None
        self.T_hold = None

        self.last_out = np.array([self.fv0, self.qk0], dtype=float)

        # filtered derivatives
        self.cb_d = 0.0
        self.T_d = 0.0

        self.first = True

    # ---- gains ----
    # C_B loop: slow analyzer, gentle. F/V increases C_B (on safe branch).
    CB_KP = 8.0
    CB_KI = 0.5
    CB_KD = 0.0

    # T loop: fast thermocouple. Q_K is cooling (negative). More cooling
    # (more negative Q_K) lowers T. So error = (T - setpoint) -> increase
    # cooling magnitude. We treat control on Q_K directly.
    T_KP = 250.0
    T_KI = 15.0
    T_KD = 30.0

    def _slew(self, cmd, prev, rate):
        d = cmd - prev
        if d > rate:
            d = rate
        elif d < -rate:
            d = -rate
        return prev + d

    def step(self, t, y, r, quality):
        dt = self.dt

        # --- read measurements with hold on bad quality ---
        cb_meas = y[0]
        T_meas = y[1]
        cb_ok = bool(quality[0]) if quality is not None else True
        T_ok = bool(quality[1]) if quality is not None else True

        if cb_ok and np.isfinite(cb_meas):
            self.cb_hold = cb_meas
        cb = self.cb_hold if self.cb_hold is not None else cb_meas

        if T_ok and np.isfinite(T_meas):
            self.T_hold = T_meas
        T = self.T_hold if self.T_hold is not None else T_meas

        # --- setpoints ---
        cb_sp = r[0]
        T_sp = r[1]
        if not np.isfinite(cb_sp):
            cb_sp = cb
        if not np.isfinite(T_sp):
            T_sp = T

        # ================= C_B loop -> F/V =================
        if cb is not None and np.isfinite(cb):
            e = cb_sp - cb  # want more C_B -> more F/V (safe branch)

            # derivative (filtered) on measurement
            if self.cb_last is None:
                dcb = 0.0
            else:
                dcb = (cb - self.cb_last) / dt
            self.cb_last = cb
            self.cb_d = 0.8 * self.cb_d + 0.2 * dcb

            # tentative integral
            fv_unsat = (self.fv0 + self.CB_KP * e + self.CB_KI * self.cb_int
                        - self.CB_KD * self.cb_d)

            fv_lo = self.fv_min
            fv_hi = self.fv_peak  # never cross the peak
            fv_clamped = min(max(fv_unsat, fv_lo), fv_hi)

            # anti-windup: only integrate if not saturating in that direction
            if (fv_unsat >= fv_lo or e > 0) and (fv_unsat <= fv_hi or e < 0):
                self.cb_int += e * dt
                # clamp integral
                self.cb_int = max(min(self.cb_int, 60.0), -60.0)

            self.fv_cmd = fv_clamped

        # ================= T loop -> Q_K =================
        if T is not None and np.isfinite(T):
            eT = T_sp - T  # positive -> too cold -> reduce cooling (Q_K -> 0)

            if self.T_last is None:
                dT = 0.0
            else:
                dT = (T - self.T_last) / dt
            self.T_last = T
            self.T_d = 0.7 * self.T_d + 0.3 * dT

            # Q_K positive-going when too cold; error eT>0 -> raise Q_K toward 0
            qk_unsat = (self.qk0 + self.T_KP * eT + self.T_KI * self.T_int
                        - self.T_KD * self.T_d)

            # Safety override: if T near limit, force maximum cooling
            if T > self.T_safe:
                over = (T - self.T_safe)
                # aggressively drive cooling
                qk_unsat = self.qk0 - 3000.0 - 2000.0 * over

            qk_clamped = min(max(qk_unsat, self.qk_min), self.qk_max)

            if (qk_unsat >= self.qk_min or eT < 0) and (qk_unsat <= self.qk_max or eT > 0):
                self.T_int += eT * dt
                self.T_int = max(min(self.T_int, 400.0), -400.0)

            self.qk_cmd = qk_clamped

        # ================= slew limiting for smooth motion =================
        prev = self.last_out
        # modest slew to avoid chatter / duty violation
        fv_out = self._slew(self.fv_cmd, prev[0], 0.5)
        qk_out = self._slew(self.qk_cmd, prev[1], 400.0)

        fv_out = min(max(fv_out, self.fv_min), self.fv_max)
        qk_out = min(max(qk_out, self.qk_min), self.qk_max)

        out = np.array([fv_out, qk_out], dtype=float)
        self.last_out = out
        self.first = False
        return out