import numpy as np


class Controller:
    """
    Gain-scheduled PI controller (velocity form) for pH neutralization.

    Design notes:
      * The pH process gain varies by orders of magnitude across the titration
        curve.  A local linearization of the nominal reaction-invariant model
        gives an estimate of dpH/d(base flow) at the current operating point;
        the PI gain is scheduled with its inverse, clamped so that model
        mismatch cannot produce runaway gains.
      * Velocity (incremental) form gives inherent anti-windup: increments that
        cannot be applied (saturation / overrides) are discarded.
      * Proportional action acts on the filtered measurement only (no setpoint
        kick).  No derivative: the analyzer is ~20 s late and noisy.
      * First-order filtering of pH, a small move dead-band with an
        accumulator, and per-step slew limiting keep actuator travel/duty low.
      * Overrides on measured tank level (unscored but safety-constrained) and
        on extreme filtered pH clamp the allowed move direction near limits.
      * The internal reference is rate-limited so setpoint steps are followed
        smoothly, limiting overshoot near the 10.5 pH hard limit.
    """

    # nominal chemistry / plant (used ONLY for relative gain scheduling)
    Q1, Q2 = 16.6, 0.55
    WA1, WB1 = 0.003, 0.0
    WA2, WB2 = -0.03, 0.03
    WA3, WB3 = -0.00305, 5e-5
    PK1, PK2 = 6.35, 10.25
    LN10 = np.log(10.0)

    def __init__(self, brief):
        self.Ts = float(getattr(brief, "sample_time", 5.0))
        u0 = 14.22
        for name in ("u0", "actuator_initial", "actuators_initial", "u_init"):
            v = getattr(brief, name, None)
            if v is not None:
                try:
                    u0 = float(np.asarray(v).ravel()[0])
                except Exception:
                    pass
                break
        self.u_init = float(np.clip(u0, 0.0, 30.0))

        # actuator limits (leave a hair off the hard stops)
        self.u_lo, self.u_hi = 0.2, 29.5

        # tuning (SIMC-style; tightened again - duty margin remains large)
        self.tau_p = 150.0          # nominal residence time, s
        self.theta = 27.0           # delay + half sample, s
        self.tauc = 38.0            # desired closed-loop time constant, s
        self.Ti = 88.0              # integral time, s
        self.Kmin, self.Kmax = 0.05, 3.0   # clamp on scheduled process gain

        # filtering / move shaping
        self.tau_f = 9.0                        # pH filter time constant, s
        self.alpha = self.Ts / (self.Ts + self.tau_f)
        self.slew = 2.0                         # max |du| per step, mL/s
        self.db = 0.02                          # move dead-band, mL/s
        self.r_rate = 1.0                       # internal ref rate, pH/step

        # safety envelopes (conservative vs true limits: meas. is delayed)
        self.h_hi, self.h_lo = 27.5, 7.0        # level triggers (5..30 hard)
        self.ph_hi, self.ph_lo = 10.05, 4.7     # pH triggers (4..10.5 hard)

        self.reset()

    # ---- nominal titration-curve local gain -------------------------------
    def _proc_gain(self, ph, u):
        ph = float(np.clip(ph, 2.0, 12.0))
        q = self.Q1 + self.Q2 + max(u, 0.0)
        wa4 = (self.Q1 * self.WA1 + self.Q2 * self.WA2 + u * self.WA3) / q
        wb4 = (self.Q1 * self.WB1 + self.Q2 * self.WB2 + u * self.WB3) / q
        x1 = 10.0 ** (ph - self.PK2)
        x2 = 10.0 ** (self.PK1 - ph)
        num = 1.0 + 2.0 * x1
        den = 1.0 + x2 + x1
        g = num / den
        dnum = 2.0 * self.LN10 * x1
        dden = self.LN10 * (x1 - x2)
        gp = (dnum * den - num * dden) / (den * den)
        dFdu = (self.WA3 - wa4) / q + (self.WB3 - wb4) * g / q
        dFdpH = wb4 * gp + self.LN10 * (10.0 ** (ph - 14.0) + 10.0 ** (-ph))
        K = -dFdu / max(dFdpH, 1e-12)
        return float(np.clip(K, self.Kmin, self.Kmax))

    # -----------------------------------------------------------------------
    def reset(self):
        self.u = self.u_init
        self.yf = None          # filtered pH
        self.yf_prev = None
        self.r_int = None       # rate-limited internal reference
        self.pending = 0.0      # accumulated un-applied move
        self.h_last = None      # last good level

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        q = np.asarray(quality).astype(bool) if quality is not None \
            else np.ones(len(y), dtype=bool)

        ph_ok = bool(q[0]) and np.isfinite(y[0])
        lvl_ok = len(y) > 1 and bool(q[1]) and np.isfinite(y[1])
        if lvl_ok:
            self.h_last = float(y[1])
        h = self.h_last

        # --- first call: initialize bumplessly, hold the incoming output ---
        if self.yf is None:
            self.yf = float(y[0]) if ph_ok else 7.0
            self.yf_prev = self.yf
            self.r_int = self.yf
            return np.array([self.u])

        # --- measurement filter (hold on bad quality) ---
        self.yf_prev = self.yf
        if ph_ok:
            self.yf += self.alpha * (float(y[0]) - self.yf)

        # --- rate-limited internal reference ---
        r_tgt = float(r[0]) if (r is not None and np.isfinite(r[0])) else self.r_int
        r_tgt = float(np.clip(r_tgt, 4.8, 9.95))
        dr = np.clip(r_tgt - self.r_int, -self.r_rate, self.r_rate)
        self.r_int += dr

        e = self.r_int - self.yf

        # --- gain-scheduled PI increment (P on measurement) ---
        K = self._proc_gain(self.yf, self.u)
        Kc = self.tau_p / (K * (self.tauc + self.theta)) * 0.9
        dP = -Kc * (self.yf - self.yf_prev) if ph_ok else 0.0
        dI = Kc * (self.Ts / self.Ti) * e
        du_des = self.pending + dP + dI

        # --- move bounds: slew, level override, pH override ---
        lo, hi = -self.slew, self.slew
        if h is not None:
            hi = min(hi, 0.4 * (self.h_hi - h))   # forces down-moves when high
            lo = max(lo, 0.4 * (self.h_lo - h))   # forces up-moves when low
        hi = min(hi, 3.0 * (self.ph_hi - self.yf) if self.yf > self.ph_hi
                 else self.slew)
        lo = max(lo, 3.0 * (self.ph_lo - self.yf) if self.yf < self.ph_lo
                 else -self.slew)
        if lo > hi:
            lo = hi = 0.5 * (lo + hi)

        du = float(np.clip(du_des, lo, hi))

        # --- dead-band with accumulator (limits travel/duty, beats stiction) ---
        if abs(du) < self.db and lo <= 0.0 <= hi:
            du = 0.0
        self.pending = float(np.clip(du_des - du, -1.0, 1.0))

        # --- apply, respect absolute limits, discard clipped excess ---
        u_new = float(np.clip(self.u + du, self.u_lo, self.u_hi))
        self.u = u_new
        return np.array([self.u])