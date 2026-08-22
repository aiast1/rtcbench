import numpy as np


class Controller:
    """
    Quadruple-tank (four_tank_v1) commissioning controller  -- rev 3.

    What drives the cost here is tracking, not travel (measured effort is
    ~0.0007 against a weight of 0.5, i.e. ~3% of the score), so this revision
    spends a little more actuator motion to buy speed, while keeping the
    stability margins that protect the worst-case draw.

    Structure
    ---------
      * Static model-inversion FEEDFORWARD: the full 2x2 steady-state map
        (levels -> pump volts) is inverted, so a setpoint move on either
        channel immediately produces the right move on BOTH pumps.  This is
        also the decoupler; no dynamic inverse is used (its gain blows up as
        gamma1+gamma2 -> 1, which is exactly the draw we must survive).
      * RATE feedforward T/K * dr/dt (filtered, hard-clipped) so the 60 s ramp
        segment is followed with almost no lag.  r is noise-free.
      * Decentralised PI, pump1->h1, pump2->h2 (minimum-phase pairing).
        IMC tuning gives Kc = A/(gamma*k*lambda), which is INDEPENDENT of
        level, so the sqrt() gravity-drain gain fall-off needs no scheduling.
        lambda ~ 12 s -> predicted PM ~ 48 deg nominal, ~36 deg with +40%
        gain error, delay and filter included.
        Ti is set below the open-loop tau (Ti ~ 4*lambda) because the dominant
        error source is a load/mismatch offset, whose rejection tail decays
        with Ti, not with lambda.
      * NO derivative action (noisy, quantised, delayed, dropped samples).
      * Exact back-calculation anti-windup across saturation, slew and guards.
      * Unmeasured tanks 3/4 are protected by a deliberately PESSIMISTIC
        open-loop observer (cross-path gain x1.35, area x0.85) feeding a
        voltage-ceiling governor whose floor (4.5 V) is still far above the
        volts needed to hold any setpoint, so the guard cannot destabilise the
        loop but will stop a runaway fill of a tank we cannot see.
      * Output deadband + measurement filter keep noise dither (and hence
        actuator duty) small.
    """

    G = 981.0  # cm/s^2

    # ------------------------------------------------------------------ init
    def __init__(self, brief):
        try:
            dt = float(brief.sample_time)
        except Exception:
            dt = 2.0
        if (not np.isfinite(dt)) or dt <= 0.0:
            dt = 2.0
        self.dt_nom = dt

        # nominal plant (feedforward + hidden-state guarding only)
        self.A1, self.A2, self.A3, self.A4 = 28.0, 32.0, 28.0, 32.0
        self.a1, self.a2, self.a3, self.a4 = 0.071, 0.057, 0.071, 0.057
        self.k1, self.k2 = 3.33, 3.35
        self.gam1, self.gam2 = 0.7, 0.6

        # ---- PI tuning (lambda ~ 12 s) --------------------------------
        # nominal: K1=5.19 cm/V, T1=62 s ; K2=5.69 cm/V, T2=91 s
        self.Kc = np.array([0.95, 1.20])       # V/cm
        self.Ti = np.array([45.0, 58.0])       # s
        self.b = 0.75                           # setpoint weight on P

        self.tau_f = 3.5                        # measurement filter [s]
        self.tau_r = 6.0                        # setpoint-rate filter [s]

        # ---- feedforward gains ---------------------------------------
        self.ff_gain = 0.95                     # static FF de-rating
        self.cd = np.array([11.0, 15.0])        # ~T/K  (V per cm/s)
        self.cd_lim = 0.80                      # V, rate-FF clamp

        # ---- actuator handling ---------------------------------------
        self.u_hard_lo, self.u_hard_hi = 0.0, 10.0
        self.u_cap = 9.0
        self.du_max = 1.5                       # V per sample (soft slew)
        self.deadband = 0.015                   # V
        self.u_start = np.array([3.0, 3.0])

        # ---- hidden-tank guarding ------------------------------------
        self.sf = 1.35                          # pessimism on cross-path gain
        self.c3 = self.sf * (1.0 - self.gam2) * self.k2   # -> tank 3, per volt
        self.c4 = self.sf * (1.0 - self.gam1) * self.k1   # -> tank 4, per volt
        self.A3e = 0.85 * self.A3
        self.A4e = 0.85 * self.A4
        self.gov_h = np.array([0.0, 11.0, 15.0, 17.5])
        self.gov_u = np.array([10.0, 8.0, 6.0, 4.5])

        # ---- measured-tank overflow / dry-out guards -----------------
        self.hi_h = np.array([16.0, 19.0])
        self.hi_u = np.array([9.0, 0.2])
        self.lo_h = np.array([1.0, 3.0])
        self.lo_u = np.array([5.0, 0.0])

        self.r_default = np.array([12.26, 12.78])
        self.reset()

    # ----------------------------------------------------------------- reset
    def reset(self):
        self.t_prev = None
        self.first = True
        self.yf = np.array([np.nan, np.nan])
        self.rej = np.zeros(2, dtype=int)
        self.I = np.zeros(2)
        self.r_last = self.r_default.copy()
        self.r_prev = self.r_default.copy()
        self.rdot = np.zeros(2)
        self.u_t = self.u_start.copy()      # internal continuous command
        self.u_out = self.u_start.copy()    # emitted command
        self.h3e = (self.c3 * self.u_start[1] / self.a3) ** 2 / (2.0 * self.G)
        self.h4e = (self.c4 * self.u_start[0] / self.a4) ** 2 / (2.0 * self.G)

    # ------------------------------------------------------- feedforward map
    def _ff(self, r):
        """Steady-state inverse of the nominal plant: levels -> pump volts.
        a1*sqrt(2g h1) = g1*k1*v1 + (1-g2)*k2*v2
        a2*sqrt(2g h2) = g2*k2*v2 + (1-g1)*k1*v1
        """
        r = np.clip(np.asarray(r, dtype=float), 0.5, 20.0)
        B1 = self.a1 * np.sqrt(2.0 * self.G * r[0])
        B2 = self.a2 * np.sqrt(2.0 * self.G * r[1])
        D = self.gam1 + self.gam2 - 1.0                      # 0.3 nominal
        if abs(D) < 0.15:
            D = 0.15
        X = (self.gam2 * B1 - (1.0 - self.gam2) * B2) / D     # k1*v1
        Y = (self.gam1 * B2 - (1.0 - self.gam1) * B1) / D     # k2*v2
        v = np.array([X / self.k1, Y / self.k2])
        return np.clip(v, 0.2, 6.5)

    # ------------------------------------------------------------------ step
    def step(self, t, y, r, quality):
        y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
        r = np.atleast_1d(np.asarray(r, dtype=float)).ravel()
        if quality is None:
            q = np.ones(2, dtype=bool)
        else:
            q = np.atleast_1d(np.asarray(quality)).ravel().astype(bool)

        # ---- sample interval -----------------------------------------
        if self.t_prev is None:
            dt = self.dt_nom
        else:
            dt = float(t) - self.t_prev
            if (not np.isfinite(dt)) or dt <= 0.0:
                dt = self.dt_nom
            dt = min(max(dt, 0.25 * self.dt_nom), 4.0 * self.dt_nom)
        self.t_prev = float(t)

        # ---- setpoints (hold last valid) -----------------------------
        for i in range(2):
            if i < r.size and np.isfinite(r[i]):
                self.r_last[i] = float(r[i])
        rr = np.clip(self.r_last.copy(), 1.0, 18.0)

        # ---- filtered setpoint rate (rate feedforward) ---------------
        ar = dt / (self.tau_r + dt)
        raw_rdot = (rr - self.r_prev) / dt
        self.r_prev = rr.copy()
        if self.first:
            self.rdot[:] = 0.0
        else:
            self.rdot += ar * (raw_rdot - self.rdot)

        # ---- measurement conditioning --------------------------------
        alpha = dt / (self.tau_f + dt)
        for i in range(2):
            raw = y[i] if i < y.size else np.nan
            good = bool(q[i]) if i < q.size else True
            if (not good) or (not np.isfinite(raw)) or raw < -1.0 or raw > 25.0:
                continue                                    # stale: hold
            val = min(max(float(raw), 0.0), 20.0)
            if not np.isfinite(self.yf[i]):
                self.yf[i] = val
                self.rej[i] = 0
                continue
            if abs(val - self.yf[i]) > 3.0 and self.rej[i] < 3:
                self.rej[i] += 1                            # spike: skip
                continue
            self.rej[i] = 0
            self.yf[i] += alpha * (val - self.yf[i])
        for i in range(2):
            if not np.isfinite(self.yf[i]):
                self.yf[i] = rr[i]
        yf = self.yf.copy()

        # ---- feedforward ---------------------------------------------
        uff = self.ff_gain * self._ff(rr)
        uff = uff + np.clip(self.cd * self.rdot, -self.cd_lim, self.cd_lim)

        # ---- PI ------------------------------------------------------
        e = rr - yf
        p = self.Kc * (self.b * rr - yf)

        if self.first:
            self.I = self.u_start - uff - p     # exact bumpless start
            self.first = False
        else:
            self.I += (self.Kc / self.Ti) * e * dt

        u_un = uff + p + self.I

        # ---- limits: hidden-tank governor + measured-level guards ----
        cap_h4 = float(np.interp(self.h4e, self.gov_h, self.gov_u))
        cap_h3 = float(np.interp(self.h3e, self.gov_h, self.gov_u))
        cap = np.array([cap_h4, cap_h3])        # pump1 fills t4, pump2 fills t3
        lvl_cap = np.array([float(np.interp(yf[0], self.hi_h, self.hi_u)),
                            float(np.interp(yf[1], self.hi_h, self.hi_u))])
        hi = np.minimum(np.minimum(cap, lvl_cap), self.u_cap)
        hi = np.minimum(hi, self.u_hard_hi)

        lo = np.array([float(np.interp(yf[0], self.lo_h, self.lo_u)),
                       float(np.interp(yf[1], self.lo_h, self.lo_u))])
        lo = np.clip(lo, self.u_hard_lo, hi)

        u_t = np.clip(u_un, lo, hi)

        # ---- soft slew limit -----------------------------------------
        du = np.clip(u_t - self.u_t, -self.du_max, self.du_max)
        u_t = np.clip(self.u_t + du, self.u_hard_lo, self.u_hard_hi)

        # ---- exact back-calculation anti-windup ----------------------
        self.I += (u_t - u_un)
        self.u_t = u_t

        # ---- output deadband (travel / duty economy) -----------------
        for i in range(2):
            if abs(u_t[i] - self.u_out[i]) > self.deadband:
                self.u_out[i] = u_t[i]
        u_cmd = np.clip(self.u_out.copy(), self.u_hard_lo, self.u_hard_hi)

        # ---- pessimistic observers for the unmeasured upper tanks ----
        h3 = max(self.h3e, 0.0)
        h4 = max(self.h4e, 0.0)
        d3 = (self.c3 * u_cmd[1] - self.a3 * np.sqrt(2.0 * self.G * h3)) / self.A3e
        d4 = (self.c4 * u_cmd[0] - self.a4 * np.sqrt(2.0 * self.G * h4)) / self.A4e
        self.h3e = min(max(h3 + dt * d3, 0.0), 40.0)
        self.h4e = min(max(h4 + dt * d4, 0.0), 40.0)

        return u_cmd