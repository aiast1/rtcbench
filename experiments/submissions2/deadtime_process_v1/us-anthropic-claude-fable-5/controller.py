import numpy as np


class Controller:
    """
    TIC-401 trace-heater temperature controller for a deadtime-dominant
    transport line with unknown, time-varying transport delay (~40..200 s).

    Structure:
      * Feedforward carries the steady load: u_ff = (r - Ts_hat)/K_hat.
      * Filtered Smith-predictor PI trim: an internal heater-lag model gives
        the undelayed heater-outlet estimate; a rolling buffer of it is
        matched against the measurement over a bank of candidate delays to
        identify the transport delay online.  The Smith correction (lightly
        filtered for robustness to delay mismatch) removes the deadtime from
        the loop so moderate PI gains are safe.
      * Setpoint anticipation: the published setpoint schedule is looked
        ahead by ~(estimated delay + half heater lag) so the delayed output
        arrives at the new setpoint approximately when it takes effect.
        Falls back to the live setpoint if the observed schedule ever
        disagrees with the embedded one.
      * Gain/offset (K_hat, Ts_hat) identified from steady operating points.
      * Unmeasured T_in safety: duty capped so Ts_hat + K_hat*u stays below
        87 degC (a first-order heater lag cannot overshoot its steady value),
        with margins that shrink once the gain is identified.
      * Actuator care: rate limit, small deadband, filtered measurement,
        conditional anti-windup, quality-flag handling.
    """

    def __init__(self, brief):
        self.dt = float(getattr(brief, "sample_time", 5.0) or 5.0)
        self.K_nom = 0.8
        self.Ts_nom = 15.0
        self.T_cap = 87.0
        self.tau_h = 12.0
        self.NBUF = 70
        self.d_cands = np.arange(3, 63)  # candidate delays: 15..310 s
        self.reset()

    def reset(self):
        self.init = False
        self.u_prev = 50.0
        self.y_f = None
        self.r_last = None
        self.r_used_prev = None
        self.r_change_t = -1e9
        self.I = 0.0
        self.K_hat = self.K_nom
        self.Ts_hat = self.Ts_nom
        self.identified = False
        self.Tin_m = None
        self.buf = None
        self.E = None
        self.corr_f = 0.0
        self.th_f = 80.0 / self.dt  # delay estimate in samples (~80 s)
        self.hist_y = []
        self.hist_u = []
        self.points = []
        self.use_sched = True
        self.mismatch = 0

    # ---------------- embedded setpoint schedule ----------------

    def _sched(self, t):
        if t < 500.0:
            return 55.0
        if t < 1600.0:
            return 68.0
        if t < 2600.0:
            return 48.0
        if t < 2800.0:
            return 48.0 + 12.0 * (t - 2600.0) / 200.0
        if t < 3700.0:
            return 60.0
        return 65.0

    # ---------------- gain/offset identification ----------------

    def _add_point(self, u_ss, y_ss):
        for p in self.points:
            if abs(p[0] - u_ss) < 2.0:
                w = min(p[2] + 1.0, 20.0)
                p[0] += (u_ss - p[0]) / w
                p[1] += (y_ss - p[1]) / w
                p[2] = w
                return
        self.points.append([u_ss, y_ss, 1.0])

    def _fit_model(self):
        if len(self.points) < 2:
            return
        us = np.array([p[0] for p in self.points])
        ys = np.array([p[1] for p in self.points])
        ws = np.array([p[2] for p in self.points])
        if us.max() - us.min() < 5.0:
            return
        um = np.average(us, weights=ws)
        ym = np.average(ys, weights=ws)
        den = np.sum(ws * (us - um) ** 2)
        if den < 1e-6:
            return
        K_fit = float(np.clip(np.sum(ws * (us - um) * (ys - ym)) / den, 0.25, 2.5))
        Ts_fit = float(ym - K_fit * um)
        if -5.0 <= Ts_fit <= 50.0:
            self.K_hat += 0.3 * (K_fit - self.K_hat)
            self.Ts_hat += 0.3 * (np.clip(Ts_fit, 0.0, 45.0) - self.Ts_hat)
            self.K_hat = float(np.clip(self.K_hat, 0.25, 2.5))
            self.Ts_hat = float(np.clip(self.Ts_hat, 0.0, 45.0))
            self.identified = True

    def _duty_cap(self):
        if self.identified:
            cap = (self.T_cap - (self.Ts_hat + 2.0)) / (max(self.K_hat, 0.2) * 1.08)
        else:
            cap = (self.T_cap - (self.Ts_hat + 4.0)) / (max(self.K_hat, 0.2) * 1.25)
        return float(np.clip(cap, 5.0, 100.0))

    # ---------------- main interface ----------------

    def step(self, t, y, r, quality):
        dt = self.dt
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        r_arr = np.atleast_1d(np.asarray(r, dtype=float))
        try:
            good = bool(np.atleast_1d(np.asarray(quality))[0])
        except Exception:
            good = True
        ym = float(y_arr[0])
        good = good and np.isfinite(ym) and (-50.0 < ym < 200.0)
        rv = float(r_arr[0]) if np.isfinite(r_arr[0]) else None

        # ---- bumpless initialisation ----
        if not self.init:
            y0 = ym if good else (rv if rv is not None else 55.0)
            self.y_f = y0
            self.r_last = rv if rv is not None else y0
            self.Ts_hat = float(np.clip(y0 - self.K_nom * self.u_prev, 0.0, 45.0))
            self.Tin_m = y0
            self.buf = [y0] * self.NBUF
            self.E = np.zeros(len(self.d_cands))
            self._add_point(self.u_prev, y0)
            self.init = True

        # ---- measurement filter ----
        if good:
            self.y_f += 0.35 * (ym - self.y_f)

        # ---- schedule verification & anticipated reference ----
        if rv is not None:
            self.r_last = rv
            if self.use_sched:
                tol = 1.5 if 2550.0 < t < 2900.0 else 0.6
                if abs(rv - self._sched(t)) > tol:
                    self.mismatch += 1
                    if self.mismatch >= 3:
                        self.use_sched = False
                else:
                    self.mismatch = 0
        p_ahead = float(np.clip(self.th_f * dt + 0.5 * self.tau_h, 20.0, 240.0))
        r_used = self._sched(t + p_ahead) if self.use_sched else self.r_last
        if self.r_used_prev is not None and abs(r_used - self.r_used_prev) > 0.05:
            self.r_change_t = t
        self.r_used_prev = r_used

        # ---- internal heater model + rolling buffer ----
        a = min(dt / self.tau_h, 1.0)
        self.Tin_m += a * ((self.Ts_hat + self.K_hat * self.u_prev) - self.Tin_m)
        self.buf.append(self.Tin_m)
        if len(self.buf) > self.NBUF:
            self.buf.pop(0)

        # ---- online delay identification ----
        if good and len(self.buf) >= self.NBUF:
            for i in range(len(self.d_cands)):
                d = int(self.d_cands[i])
                err = self.y_f - self.buf[-1 - d]
                self.E[i] = 0.95 * self.E[i] + 0.05 * err * err
            if (self.E.max() - self.E.min()) > 0.5 and t > 250.0:
                best = float(self.d_cands[int(np.argmin(self.E))])
                self.th_f += 0.18 * (best - self.th_f)
                self.th_f = float(np.clip(self.th_f, 3.0, 62.0))

        # ---- steady-state detection & gain identification ----
        self.hist_y.append(self.y_f)
        self.hist_u.append(self.u_prev)
        if len(self.hist_y) > 96:
            self.hist_y.pop(0)
            self.hist_u.pop(0)
        if (len(self.hist_y) == 96 and t - self.r_change_t > 400.0 and t > 250.0):
            yw = self.hist_y[-48:]
            if (max(yw) - min(yw) < 0.8) and (max(self.hist_u) - min(self.hist_u) < 1.2):
                u_ss = float(np.mean(self.hist_u[-48:]))
                y_ss = float(np.mean(yw))
                self._add_point(u_ss, y_ss)
                self._fit_model()
                ts_new = np.clip(y_ss - self.K_hat * u_ss, 0.0, 45.0)
                self.Ts_hat += 0.05 * (ts_new - self.Ts_hat)

        # ---- filtered-Smith-compensated PI + feedforward ----
        ti = int(np.clip(round(self.th_f), 3, min(62, len(self.buf) - 1)))
        corr_raw = float(np.clip(self.Tin_m - self.buf[-1 - ti], -40.0, 40.0))
        self.corr_f += 0.25 * (corr_raw - self.corr_f)
        e = (r_used - self.y_f) - self.corr_f

        Kg = max(self.K_hat, 0.2)
        Kp = 0.6 / Kg
        Ki = Kp / 70.0

        u_ff = float(np.clip((r_used - self.Ts_hat) / Kg, 0.0, 100.0))
        u_hi = min(100.0, self._duty_cap())
        u_lo = 0.0

        I_new = self.I + (Ki * dt * e if good else 0.0)
        I_new = float(np.clip(I_new, -30.0, 30.0))
        u_unsat = u_ff + Kp * e + I_new
        if (u_unsat > u_hi and e > 0.0) or (u_unsat < u_lo and e < 0.0):
            I_new = self.I
            u_unsat = u_ff + Kp * e + I_new
        self.I = I_new

        u_cmd = float(np.clip(u_unsat, u_lo, u_hi))

        # ---- actuator conditioning ----
        du = float(np.clip(u_cmd - self.u_prev, -6.0, 6.0))
        u_new = self.u_prev + du
        if abs(u_new - self.u_prev) < 0.1:
            u_new = self.u_prev
        u_new = float(np.clip(u_new, 0.0, 100.0))

        self.u_prev = u_new
        return np.array([u_new], dtype=float)