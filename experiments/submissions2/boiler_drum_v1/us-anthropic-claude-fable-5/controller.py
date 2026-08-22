import numpy as np


class Controller:
    """
    Three-element feedwater control, commissioned for robustness.

    Post-mortem of earlier rounds:
      * Aggressive "helper" overrides (pressure guard forcing underfeed, a
        long-memory mass-balance guard vulnerable to flow-sensor bias) pinned
        the level against a limit for hundreds of periods. All overrides are
        now either removed or reduced to sign-clamps / near-trip emergencies.
      * Actuator duty was blown by noise passing through the feedforward.
        Fix: very heavy filtering of steam flow and pressure, a second filter
        on the valve target, and a hysteresis (dead-zone) output stage that
        only moves the valve when the target has genuinely moved.

    Structure:
      demand  w_dem = filtered steam flow + small level-PI trim (+/-8 kg/s)
      valve   u = 100*w_dem/(k_est*sqrt(dp)) + slow flow integral
      output  hysteresis + slew limit, emergency bypass near the real trips.
    """

    def __init__(self, brief):
        self.dt = float(getattr(brief, "sample_time", 5.0))
        self.u0 = 45.0
        self.p_pump = 120.0
        self.kv = 1000.0 / (720.0 * 20.0)   # mm/s per kg/s excess feed

        # level trim PI (kg/s)
        self.Kp = 0.06
        self.Ti = 300.0
        self.trim_lim = 8.0
        self.sp_rate = 0.6                  # mm/s internal SP rate limit

        # inner flow integral
        self.Ki_u = 0.010                   # %/(kg/s)/s
        self.ui_lim = 20.0

        # filters (EWMA alphas at 5 s)
        self.aL = 0.15
        self.aW = 0.05
        self.aP = 0.05
        self.aT = 0.25                      # valve-target smoothing

        # output stage
        self.hyst = 0.6                     # % dead-zone before moving
        self.max_move = 1.0                 # %/step normal
        self.max_move_em = 2.5              # %/step emergency
        self.duty_frac = 0.00169            # scored limit (fraction of range/step)

        self.reset()

    def reset(self):
        self.first = True
        self.L_f = None
        self.p_f = None
        self.ws_f = None
        self.wf_f = None
        self.k_est = 18.78
        self.trim_i = 0.0
        self.u_i = 0.0
        self.u_cmd = self.u0
        self.u_tgt_f = None
        self.r_int = None
        self.sp_last = 0.0
        self.D = 0.0                        # high-passed mass-balance (mm)
        self.travel = 0.0
        self.nsteps = 0

    def _filt(self, cur, new, alpha, ok, default):
        if cur is None:
            return float(new) if (ok and np.isfinite(new)) else float(default)
        if not ok or not np.isfinite(new):
            return cur
        return cur + alpha * (float(new) - cur)

    def step(self, t, y, r, quality):
        dt = self.dt
        try:
            q = np.asarray(quality, dtype=bool)
            ok = [bool(q[i]) if i < q.size else True for i in range(4)]
        except Exception:
            ok = [True, True, True, True]

        # ---------------- filtered measurements ----------------
        self.L_f = self._filt(self.L_f, y[0], self.aL, ok[0], 0.0)
        self.p_f = self._filt(self.p_f, y[1], self.aP, ok[1], 85.0)
        self.ws_f = self._filt(self.ws_f, y[2], self.aW, ok[2], 50.0)
        self.wf_f = self._filt(self.wf_f, y[3], self.aW, ok[3], 50.0)
        dp = max(self.p_pump - self.p_f, 4.0)

        # ---------------- setpoint (internally rate limited) ----------------
        if r is not None and len(r) > 0 and np.isfinite(r[0]):
            self.sp_last = float(r[0])
        if self.r_int is None:
            self.r_int = self.L_f
        self.r_int += float(np.clip(self.sp_last - self.r_int,
                                    -self.sp_rate * dt, self.sp_rate * dt))

        # -------- high-passed mass-balance (swell-blind inventory drift) -----
        self.D += ((self.wf_f - self.ws_f) * self.kv - self.D / 200.0) * dt
        self.D = float(np.clip(self.D, -400.0, 400.0))

        # ---------------- level PI trim (small authority) ----------------
        e = self.r_int - self.L_f
        trim_p = self.Kp * e
        unsat = trim_p + self.trim_i
        blocked = ((unsat >= self.trim_lim and e > 0.0) or
                   (unsat <= -self.trim_lim and e < 0.0) or
                   (self.u_cmd >= 99.0 and e > 0.0) or
                   (self.u_cmd <= 1.0 and e < 0.0))
        if not blocked:
            self.trim_i += (self.Kp / self.Ti) * e * dt
        self.trim_i = float(np.clip(self.trim_i, -self.trim_lim, self.trim_lim))
        trim = float(np.clip(trim_p + self.trim_i, -self.trim_lim, self.trim_lim))

        # ---------------- gentle guards (act only near real trips) -----------
        if self.L_f > 190.0:                       # carryover side (measured)
            trim -= min(0.2 * (self.L_f - 190.0), 12.0)
        if self.L_f < -190.0:                      # low indicated
            trim += min(0.2 * (-190.0 - self.L_f), 12.0)
        if self.D < -130.0:                        # hidden inventory drain
            trim += min(0.06 * (-130.0 - self.D), 8.0)
        # pressure: only clamp the SIGN of the trim, never force under/overfeed
        if self.p_f < 72.0:
            trim = min(trim, 2.0)                  # don't push extra cold water
        trim = float(np.clip(trim, -14.0, 14.0))

        # ---------------- feed demand ----------------
        w_dem = float(np.clip(self.ws_f + trim, 2.0, 95.0))

        # ---------------- valve gain estimate (slow) ----------------
        if ok[3] and np.isfinite(y[3]) and 8.0 < self.u_cmd < 96.0 and self.wf_f > 2.0:
            k_obs = self.wf_f / ((self.u_cmd / 100.0) * np.sqrt(dp))
            lr = 0.5 if self.first else 0.005
            self.k_est = float(np.clip(self.k_est + lr * (k_obs - self.k_est),
                                       8.0, 40.0))

        u_ff = 100.0 * w_dem / (self.k_est * np.sqrt(dp))

        # ---------------- inner flow integral ----------------
        if self.first:
            self.u_i = self.u0 - u_ff
        else:
            ef = w_dem - self.wf_f
            if (1.0 < self.u_cmd < 99.0) or (self.u_cmd <= 1.0 and ef > 0.0) or \
               (self.u_cmd >= 99.0 and ef < 0.0):
                self.u_i += self.Ki_u * ef * dt
            self.u_i = float(np.clip(self.u_i, -self.ui_lim, self.ui_lim))

        u_target = float(np.clip(u_ff + self.u_i, 0.0, 100.0))
        if self.u_tgt_f is None:
            self.u_tgt_f = u_target
        else:
            self.u_tgt_f += self.aT * (u_target - self.u_tgt_f)

        # ---------------- emergencies (near-trip only) ----------------
        em_close = self.L_f > 215.0
        em_open = (self.L_f < -215.0) or (self.D < -180.0)
        emergency = em_close or em_open

        # ---------------- output stage: hysteresis + slew ----------------
        if self.first:
            self.first = False
            self.nsteps += 1
            return np.array([self.u_cmd], dtype=float)

        self.nsteps += 1
        # duty budget watchdog -> conservative mode if burning travel too fast
        budget = 0.75 * self.duty_frac * 100.0 * self.nsteps + 10.0
        hyst = self.hyst if self.travel < budget else 1.2
        mmax = self.max_move if self.travel < budget else 0.6

        if em_close:
            tgt = min(self.u_tgt_f, self.u_cmd - self.max_move_em)
            du = float(np.clip(tgt - self.u_cmd, -self.max_move_em, self.max_move_em))
        elif em_open:
            tgt = max(self.u_tgt_f, self.u_cmd + self.max_move_em)
            du = float(np.clip(tgt - self.u_cmd, -self.max_move_em, self.max_move_em))
        else:
            gap = self.u_tgt_f - self.u_cmd
            if abs(gap) > hyst:
                du = float(np.clip(gap, -mmax, mmax))
            else:
                du = 0.0

        u_out = float(np.clip(self.u_cmd + du, 0.0, 100.0))
        self.travel += abs(u_out - self.u_cmd)
        self.u_cmd = u_out

        return np.array([u_out], dtype=float)