"""
four_tank_blind_v1 commissioning controller  (revision 3)

Round-2 result: mean cost 0.0197, no violations, no duty breach, effort
0.0007 against a duty budget of 0.017 -> there is still ~10x head-room on
actuator movement, and essentially all of the cost is tracking lag.  So this
revision spends a little more movement to shorten every transient, while
keeping the same safety architecture that has now survived two rounds.

What changed
------------
1.  Feed-forward decoupling (the big one).  Gravity draining makes the
    STEADY-STATE map linear in sqrt(level):
        sqrt(h1) ~ g1*v1 + (1-g2)*v2 ,   sqrt(h2) ~ (1-g1)*v1 + g2*v2
    for any unknown split g1,g2 and any orifice/area draw.  Therefore, to
    raise h1 while holding h2, v1 must go UP and v2 must go DOWN - that sign
    is model-free, it holds for both the minimum-phase and the
    non-minimum-phase split.  The feed-forward now boosts the differential
    mode of its own move by 1.6 (deliberate UNDER-decoupling: the required
    ratio is 0.4...1.0, we apply 0.23, so we can never over-shoot the wrong
    way).  Feedback stays strictly decentralised, so this cannot affect
    closed-loop stability.

2.  Faster loops: Kc 0.60 -> 0.85, Ti 55 -> 40 s, PV filter 6 -> 5 s,
    lead gain 2.0 -> 3.0 with a shorter washout (45 -> 35 s).  The static
    gain of these tanks is dh/dv ~ 2h/v ~ 8 cm/V, so the setpoint steps only
    need ~0.2 V of standing change; the lead supplies a ~0.6 V decaying kick
    that triples the initial slope and then hands over to the integrator.

3.  Anti-windup changed from back-calculation to conditional integration.
    With a feed-forward pulse present, back-calculation was able to drive the
    integrator far negative while the ceiling was active and then leave the
    level low once the pulse washed out.  Conditional integration + the hard
    "sustainable command" clamp does the same job without that artefact.

4.  Baseline learning: the plant starts at steady state on u_start, so the
    first scans measure exactly the level u_start holds.  It is now averaged
    over 2 scans and blended with r(0) when the two agree, which de-noises
    the reference used by the whole feed-forward / barrier structure.

Safety (unchanged in spirit, all four tanks incl. the two unmeasured ones)
    * hard ceiling ~2x u_start: since every tank's steady level scales as
      v^2, no sustained command can put any tank near 20 cm;
    * measured-level barrier squeezing the ceiling to 0.97*u_hold(h) as h
      approaches 18 cm;
    * integral clamped so the sustainable command can never hold >17 cm;
    * low-level floor below 3 cm;
    * integral frozen on stale data, output deadband + slew limit for duty.
"""

import numpy as np


class Controller:
    # ------------------------- fixed tuning ------------------------------
    KC = 0.85          # V/cm    proportional gain (both loops)
    TI = 40.0          # s       integral time
    TD = 3.5           # s       derivative time (weak: noisy, delayed PV)
    TAU_F = 5.0        # s       PV filter
    TAU_D = 12.0       # s       derivative filter
    KL = 3.0           #         feed-forward lead gain
    TL = 35.0          # s       feed-forward lead washout
    LEAD_CLIP = 1.5    # V       max lead contribution per channel
    B_FF = 1.6         #         differential-mode boost on feed-forward only
    DU_MAX = 1.0       # V/step  slew limit
    DEADBAND = 0.008   # V       output deadband (duty / chatter guard)
    H_BAR_LO = 15.0    # cm      barrier starts
    H_BAR_HI = 18.0    # cm      barrier fully closed
    H_SS_MAX = 17.0    # cm      max level a *sustained* command may hold
    H_LOW = 3.0        # cm      low-level floor engages
    U_CAP_LEVEL = 19.0 # cm      transient ceiling reference
    U_CAP_MARGIN = 2.2 # V       transient ceiling head-room
    N_INIT = 2         #         scans used to learn the baseline
    STALE_MAX = 10     #         scans of bad data before freezing I

    def __init__(self, brief):
        self.brief = brief

        # ---- sample time --------------------------------------------------
        dt = 2.0
        for name in ("sample_time", "control_period", "dt", "period", "ts"):
            v = getattr(brief, name, None)
            if v is not None:
                try:
                    fv = float(v)
                    if fv > 0.0:
                        dt = fv
                except Exception:
                    pass
                break
        self.dt = float(dt)

        # ---- actuators ----------------------------------------------------
        self.nu = 2
        self.u_lo = np.zeros(self.nu)
        self.u_hi = np.full(self.nu, 10.0)

        u_start = np.array([3.0, 3.0])
        for name in ("actuator_start", "actuators_start", "u_start", "u0",
                     "actuator_initial", "u_init"):
            v = getattr(brief, name, None)
            if v is None:
                continue
            try:
                a = np.asarray(v, dtype=float).ravel()
                if a.size >= self.nu and np.all(np.isfinite(a[:self.nu])):
                    u_start = a[:self.nu].copy()
            except Exception:
                pass
            break
        self.u_start = np.clip(u_start, 0.2, 9.0)

        self.reset()

    # --------------------------------------------------------------- reset
    def reset(self):
        self.k = 0
        self.hf = None                        # filtered PV
        self.dst = None                       # derivative filter state
        self.I = np.zeros(self.nu)            # integral, volts
        self.u_prev = self.u_start.copy()
        self.h_ref = None                     # level held by u_start
        self.h_acc = np.zeros(self.nu)
        self.n_acc = 0
        self.r0 = None
        self.lead = None                      # lag state of u_hold(r)
        self.last_good = None
        self.stale = np.zeros(self.nu, dtype=int)

    # ---------------------------------------------------------------- util
    def _u_hold(self, h):
        """Voltage that holds level h in steady state (per channel)."""
        h = np.clip(np.asarray(h, dtype=float), 0.05, 40.0)
        return np.clip(self.u_start * np.sqrt(h / self.h_ref),
                       0.0, self.u_hi)

    # ---------------------------------------------------------------- step
    def step(self, t, y, r, quality):
        dt = self.dt
        n = self.nu

        y = np.asarray(y, dtype=float).ravel()
        r = np.asarray(r, dtype=float).ravel()
        if quality is None:
            q = np.ones(n, dtype=bool)
        else:
            q = np.asarray(quality).ravel()
            try:
                q = q.astype(bool)
            except Exception:
                q = np.ones(n, dtype=bool)
            if q.size < n:
                q = np.ones(n, dtype=bool)

        # ---------------- validate / hold measurements ---------------------
        meas = np.zeros(n)
        good = np.zeros(n, dtype=bool)
        for i in range(n):
            v = y[i] if y.size > i else np.nan
            ok = bool(q[i]) and np.isfinite(v) and (-1.0 <= v <= 26.0)
            if ok:
                meas[i] = v
                good[i] = True
            elif self.last_good is not None:
                meas[i] = self.last_good[i]
            else:
                meas[i] = r[i] if (r.size > i and np.isfinite(r[i])) else 12.0

        if self.last_good is None:
            self.last_good = meas.copy()
        else:
            for i in range(n):
                if good[i]:
                    self.last_good[i] = meas[i]
        for i in range(n):
            self.stale[i] = 0 if good[i] else self.stale[i] + 1

        # ---------------- PV filter ----------------------------------------
        af = dt / (self.TAU_F + dt)
        if self.hf is None:
            self.hf = meas.copy()
            self.dst = meas.copy()
        else:
            for i in range(n):
                if good[i]:
                    self.hf[i] += af * (meas[i] - self.hf[i])
        hf = self.hf

        # ---------------- commissioning baseline (first scans) -------------
        if self.h_ref is None:
            if self.r0 is None:
                self.r0 = np.array([r[i] if (r.size > i and np.isfinite(r[i]))
                                    else np.nan for i in range(n)])
            self.h_acc += meas
            self.n_acc += 1
            self.k += 1
            if self.n_acc >= self.N_INIT:
                base = self.h_acc / float(self.n_acc)
                for i in range(n):
                    r0i = self.r0[i]
                    if np.isfinite(r0i) and abs(r0i - base[i]) < 1.0:
                        base[i] = 0.5 * base[i] + 0.5 * r0i
                self.h_ref = np.clip(base, 1.0, 20.0)
            self.u_prev = self.u_start.copy()
            return self.u_start.copy()

        self.k += 1

        # ---------------- targets ------------------------------------------
        rt = np.zeros(n)
        track = np.zeros(n, dtype=bool)
        for i in range(n):
            ri = r[i] if (r.size > i and np.isfinite(r[i])) else np.nan
            if np.isfinite(ri):
                rt[i] = float(np.clip(ri, 0.2, 20.0))
                track[i] = True
            else:
                rt[i] = hf[i]

        e = rt - hf

        # ---------------- filtered PV derivative ---------------------------
        ad = dt / (self.TAU_D + dt)
        self.dst = self.dst + ad * (hf - self.dst)
        deriv = (hf - self.dst) / self.TAU_D          # ~ dPV/dt, low-passed

        # ---------------- static feed-forward + lead -----------------------
        uff = self._u_hold(rt)                        # sustainable part
        if self.lead is None:
            self.lead = uff.copy()
        else:
            al = dt / (self.TL + dt)
            self.lead += al * (uff - self.lead)
        lead_term = np.clip(self.KL * (uff - self.lead),
                            -self.LEAD_CLIP, self.LEAD_CLIP)

        # differential-mode boost on the feed-forward move only
        delta = (uff + lead_term) - self.u_start
        cm = 0.5 * (delta[0] + delta[1])
        dm = 0.5 * (delta[0] - delta[1])
        bias = np.array([self.u_start[0] + cm + self.B_FF * dm,
                         self.u_start[1] + cm - self.B_FF * dm])

        # ---------------- P and D ------------------------------------------
        P = self.KC * e
        D = -self.KC * self.TD * deriv

        # ---------------- limits (safety) ----------------------------------
        u_cap = np.minimum(
            self.u_hi,
            np.maximum(self._u_hold(np.full(n, self.U_CAP_LEVEL))
                       + self.U_CAP_MARGIN,
                       1.8 * self.u_start))
        uh_now = 0.97 * self._u_hold(hf)
        s = np.clip((self.H_BAR_HI - hf) / (self.H_BAR_HI - self.H_BAR_LO),
                    0.0, 1.0)
        u_max_dyn = np.clip(uh_now + s * (u_cap - uh_now), 0.0, self.u_hi)

        u_min_dyn = np.zeros(n)
        uh_hf = self._u_hold(hf)
        for i in range(n):
            if hf[i] < self.H_LOW:
                u_min_dyn[i] = min(u_max_dyn[i], 1.05 * float(uh_hf[i]) + 0.05)
        u_min_dyn = np.clip(u_min_dyn, 0.0, u_max_dyn)

        # ---------------- integral with conditional integration ------------
        Ki = self.KC / self.TI
        I_try = self.I.copy()
        for i in range(n):
            if track[i] and self.stale[i] <= self.STALE_MAX:
                I_try[i] = self.I[i] + Ki * e[i] * dt

        u_try = bias + P + D + I_try
        for i in range(n):
            if u_try[i] > u_max_dyn[i] and e[i] > 0.0:
                I_try[i] = self.I[i]                  # would wind up
            elif u_try[i] < u_min_dyn[i] and e[i] < 0.0:
                I_try[i] = self.I[i]

        # hard clamp: sustainable command must stay inside the safe band
        uh_ss_hi = self._u_hold(np.full(n, self.H_SS_MAX))
        uh_ss_lo = self._u_hold(0.25 * self.h_ref)
        I_hi = np.maximum(0.0, uh_ss_hi - uff)
        I_lo = np.minimum(0.0, uh_ss_lo - uff)
        self.I = np.clip(I_try, I_lo, I_hi)

        u_des = bias + P + D + self.I
        u_cmd = np.clip(u_des, u_min_dyn, u_max_dyn)

        # ---------------- slew limit, deadband -----------------------------
        du = np.clip(u_cmd - self.u_prev, -self.DU_MAX, self.DU_MAX)
        u_out = self.u_prev + du
        for i in range(n):
            if abs(u_out[i] - self.u_prev[i]) < self.DEADBAND:
                u_out[i] = self.u_prev[i]

        u_out = np.clip(u_out, self.u_lo, self.u_hi)
        u_out = np.where(np.isfinite(u_out), u_out, self.u_prev)

        self.u_prev = u_out.copy()
        return u_out