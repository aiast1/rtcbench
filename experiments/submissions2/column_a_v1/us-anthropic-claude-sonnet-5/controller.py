import numpy as np


class Controller:
    """
    Decentralized, bumpless-start PI controller for the 2x2 distillation
    column (L -> AT-101 distillate purity, V -> AT-102 bottoms purity).

    This process is extremely slow (dominant time constant ~194 min) compared
    with the 400 s scenario, so within one run the achievable tracking
    improvement from feedback is inherently small. Given that, the design
    priority is:

      1. NEVER violate safety (yD >= 0.85, xB <= 0.15, D >= 0.05, B >= 0.05).
      2. NEVER exceed the actuator duty (slew) limit.
      3. Make gentle, well-filtered, anti-windup-protected corrective moves
         towards the scheduled setpoints.

    Key robustness choices after the previous round's safety violations
    (caused by an overly aggressive PI + an ad-hoc "nudge" safety layer that
    could itself destabilise the loop):

      * Much lower proportional/integral gains and a tight integral clamp,
        so the commanded flows only drift slowly from the commissioned
        point -- appropriate given the process cannot respond meaningfully
        within 400 s anyway, and this keeps us far from the trip limits
        under any parameter draw.

      * A hard, *physics-based* safety clamp on the produced-flow spread
        (V - L), which directly bounds D = V - L and B = L + F - V away
        from the 0.05 kmol/min trip, independent of any measured signal
        (D and B are not measured). This replaces the previous two-sided
        ad-hoc nudge, which could fight the PI action and cause drift.

      * No derivative action (noisy, delayed measurement).

      * Slew-rate limiting well below the stated actuator duty limit
        (0.003 / step), applied *after* the safety clamp so the clamp
        cannot itself cause a duty violation.
    """

    def __init__(self, brief):
        self.brief = brief
        self.dt = float(getattr(brief, "sample_time", 1.0))

        # Actuator hard limits
        self.L_min, self.L_max = 1.5, 4.5
        self.V_min, self.V_max = 2.0, 5.0

        # Commissioned bumpless-start point
        self.L0 = 2.70629
        self.V0 = 3.20629

        # Feed rate (fixed design spec, not part of the parameter draw)
        self.F = 1.0

        # Conservative PI gains -- small moves only
        self.Kp1, self.Ki1 = 4.0, 0.06     # L loop  (AT-101)
        self.Kp2, self.Ki2 = 4.0, 0.06     # V loop  (AT-102)

        # Integral clamp (actuator units) -- keeps windup small
        self.I_max = 0.15

        # Measurement filter coefficient
        self.filt_alpha = 0.10

        # Slew limit per step -- comfortably below the 0.003 duty limit
        self.slew = 0.0015

        # Safety band on produced-flow spread s = V - L.
        # Nominal spread = V0 - L0 = 0.5, giving D = B = 0.5 at nominal F=1.
        # Require D = s >= margin and B = F - s >= margin.
        self.spread_margin = 0.15
        self.spread_min = self.spread_margin
        self.spread_max = self.F - self.spread_margin

        self.reset()

    def reset(self):
        self.L_cmd = self.L0
        self.V_cmd = self.V0

        self.y1f = None
        self.y2f = None

        self.i1 = 0.0
        self.i2 = 0.0

        self.r1_last = 0.99
        self.r2_last = 0.99

    def step(self, t, y, r, quality):
        y1, y2 = float(y[0]), float(y[1])
        q1 = bool(quality[0]) if len(quality) > 0 else True
        q2 = bool(quality[1]) if len(quality) > 1 else True

        # --- setpoints (hold last valid value if nan/stale) ---
        r1 = float(r[0]) if r is not None and len(r) > 0 else np.nan
        r2 = float(r[1]) if r is not None and len(r) > 1 else np.nan
        if np.isnan(r1):
            r1 = self.r1_last
        else:
            self.r1_last = r1
        if np.isnan(r2):
            r2 = self.r2_last
        else:
            self.r2_last = r2

        # --- measurement filtering (freeze on bad quality) ---
        if self.y1f is None:
            self.y1f = y1 if q1 else 0.99
        elif q1:
            self.y1f += self.filt_alpha * (y1 - self.y1f)

        if self.y2f is None:
            self.y2f = y2 if q2 else 0.99
        elif q2:
            self.y2f += self.filt_alpha * (y2 - self.y2f)

        # --- errors (direct acting) ---
        e1 = r1 - self.y1f
        e2 = r2 - self.y2f

        # --- integral update, frozen on bad quality ---
        if q1:
            self.i1 += self.Ki1 * e1 * self.dt
            self.i1 = float(np.clip(self.i1, -self.I_max, self.I_max))
        if q2:
            self.i2 += self.Ki2 * e2 * self.dt
            self.i2 = float(np.clip(self.i2, -self.I_max, self.I_max))

        # --- raw (unclamped) desired commands ---
        raw_L = self.L0 + self.Kp1 * e1 + self.i1
        raw_V = self.V0 + self.Kp2 * e2 + self.i2

        # anti-windup: trim integral if raw target exceeds hard limits
        if raw_L > self.L_max or raw_L < self.L_min:
            self.i1 *= 0.9
        if raw_V > self.V_max or raw_V < self.V_min:
            self.i2 *= 0.9

        raw_L = float(np.clip(raw_L, self.L_min, self.L_max))
        raw_V = float(np.clip(raw_V, self.V_min, self.V_max))

        # --- physics-based safety clamp on produced-flow spread ---
        spread = raw_V - raw_L
        if spread < self.spread_min:
            deficit = self.spread_min - spread
            raw_V += 0.5 * deficit
            raw_L -= 0.5 * deficit
        elif spread > self.spread_max:
            excess = spread - self.spread_max
            raw_V -= 0.5 * excess
            raw_L += 0.5 * excess

        raw_L = float(np.clip(raw_L, self.L_min, self.L_max))
        raw_V = float(np.clip(raw_V, self.V_min, self.V_max))

        # --- slew-rate limit the actual commanded move ---
        dL = float(np.clip(raw_L - self.L_cmd, -self.slew, self.slew))
        dV = float(np.clip(raw_V - self.V_cmd, -self.slew, self.slew))

        self.L_cmd = float(np.clip(self.L_cmd + dL, self.L_min, self.L_max))
        self.V_cmd = float(np.clip(self.V_cmd + dV, self.V_min, self.V_max))

        return np.array([self.L_cmd, self.V_cmd], dtype=float)