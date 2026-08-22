"""
Commissioning controller for column_a_v1 (LV configuration, high purity binary column).

Design notes
------------
*  The plant is a 41-stage high-purity column.  Its steady state gain matrix
   (Skogestad "column A") is strongly directional:

        dyD    =  0.878 dL - 0.864 dV
        d(xB)  =  1.082 dL - 1.096 dV          (xB = light key in bottoms)

   AT-102 reports the HEAVY key in the bottoms, i.e. y2 = 1 - xB, so

        dy2    = -1.082 dL + 1.096 dV

   Writing dL = s + d/2, dV = s - d/2  (s = internal / boil-up-and-reflux
   together, d = external split, since D = V - L = D0 - d):

        dyD = 0.014 s + 0.871 d
        dy2 = 0.014 s - 1.089 d

   -> the "difference" channel d is ~70x more effective per unit of valve
      travel than the "sum" channel s, and it is the ONLY one that can move
      the two purities in opposite directions (which is exactly what the
      220 s setpoint ramp asks for).  The s channel is nearly gain-less;
      it is used only gently (and as the safe recovery direction if either
      analyser starts to sag, because s does not change D or B at all).

*  D = V - L and B = L + F - V are *algebraic* in the two flows and are NOT
   measured.  They are the real safety trap.  Therefore the commanded
   difference V - L is hard-clamped inside [0.20, 0.70] kmol/min, which keeps
   both draws well above 0.05 even if the feed rate is drawn/disturbed
   +-20 % away from 1.0 kmol/min.  The clamp is re-applied after rate
   limiting so it can never be violated transiently.

*  The analysers run on a one minute cycle with delay, quantisation and
   dropouts, and the dominant time constant is ~194 min (>> the 400 s run).
   Hence: no derivative action, heavy measurement filtering, modest
   proportional gain, slow integral action with back-calculation anti-windup,
   and a firm command rate limit (0.004 kmol/min per second => duty
   0.0013 normalised, well under the 0.003 limit) so travel stays small and
   stiction is not excited.
"""

import numpy as np


class Controller:

    # ---------------------------------------------------------------- setup
    def __init__(self, brief):
        self.brief = brief

        # ---- sample time -------------------------------------------------
        dt = 1.0
        try:
            dt = float(brief.sample_time)
        except Exception:
            dt = 1.0
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 1.0
        self.dt_nom = dt

        # ---- commissioned duty point (bumpless start) --------------------
        u0 = np.array([2.70629, 3.20629], dtype=float)
        for nm in ("actuator_start", "actuators_start", "u_start",
                   "u0", "u_init", "actuator_init"):
            v = getattr(brief, nm, None)
            if v is None:
                continue
            try:
                a = np.asarray(v, dtype=float).ravel()
                if a.size >= 2 and np.all(np.isfinite(a[:2])):
                    u0 = a[:2].astype(float)
                    break
            except Exception:
                pass
        self.u0 = u0

        # ---- hard actuator limits ---------------------------------------
        self.umin = np.array([1.5, 2.0], dtype=float)
        self.umax = np.array([4.5, 5.0], dtype=float)
        # keep the start point strictly inside
        self.umin = np.minimum(self.umin, self.u0 - 0.01)
        self.umax = np.maximum(self.umax, self.u0 + 0.01)

        # ---- draw (D / B) protection on the commanded flows -------------
        self.diff0 = float(self.u0[1] - self.u0[0])          # nominally 0.5
        self.diff_min = max(0.15, self.diff0 - 0.30)         # protects D
        self.diff_max = min(0.75, self.diff0 + 0.20)         # protects B

        # ---- tuning ------------------------------------------------------
        # external / split loop  (strong, cheap, fast-ish)
        self.Kp_d = 40.0
        self.Ki_d = 40.0 / 900.0
        self.d_lo = -(self.diff_max - self.diff0)            # d = diff0 - diff
        self.d_hi = (self.diff0 - self.diff_min)
        # internal flow loop (weak gain -> small, gentle authority only)
        self.Kp_s = 20.0
        self.Ki_s = 20.0 / 1500.0
        self.s_lo = -0.15
        self.s_hi = 0.20
        # low-purity guard (raise internal flows: safe, does not touch D/B)
        self.guard_thresh = 0.945
        self.guard_gain = 5.0
        self.guard_max = 0.45

        # measurement filter / deadband / rate limit
        self.tau_f = 25.0                 # s, first order analyser filter
        self.deadband = 3.0e-4            # mole fraction, ~quantisation
        self.rate = 0.004                 # kmol/min per second, per valve

        self.reset()

    # ---------------------------------------------------------------- reset
    def reset(self):
        self.t_last = None
        self.raw = [np.nan, np.nan]        # last good analyser readings
        self.filt = [np.nan, np.nan]       # filtered analyser readings
        self.r_last = [np.nan, np.nan]
        self.i_d = 0.0
        self.i_s = 0.0
        self.u_cmd = self.u0.copy()

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _clip(x, lo, hi):
        return lo if x < lo else (hi if x > hi else x)

    def _db(self, e):
        if e > self.deadband:
            return e - self.deadband
        if e < -self.deadband:
            return e + self.deadband
        return 0.0

    def _fix_diff(self, u):
        """Keep V - L (and hence both product draws) inside safe bounds."""
        diff = u[1] - u[0]
        d_cl = self._clip(diff, self.diff_min, self.diff_max)
        if d_cl != diff:
            mid = 0.5 * (u[0] + u[1])
            u = np.array([mid - 0.5 * d_cl, mid + 0.5 * d_cl], dtype=float)
        return u

    # ---------------------------------------------------------------- step
    def step(self, t, y, r, quality):
        # ---------------- timing -----------------------------------------
        try:
            tf = float(t)
        except Exception:
            tf = 0.0
        if self.t_last is None:
            dt = self.dt_nom
        else:
            dt = tf - self.t_last
            if (not np.isfinite(dt)) or dt <= 0.0 or dt > 20.0 * self.dt_nom:
                dt = self.dt_nom
        self.t_last = tf

        # ---------------- measurements -----------------------------------
        try:
            ya = np.asarray(y, dtype=float).ravel()
        except Exception:
            ya = np.array([], dtype=float)
        try:
            qa = np.asarray(quality).ravel().astype(bool)
        except Exception:
            qa = np.array([], dtype=bool)

        for i in (0, 1):
            val = ya[i] if i < ya.size else np.nan
            ok = bool(np.isfinite(val)) and (0.30 < val < 1.05)
            if i < qa.size:
                ok = ok and bool(qa[i])
            if ok:
                self.raw[i] = float(val)
            if np.isfinite(self.raw[i]):
                if not np.isfinite(self.filt[i]):
                    self.filt[i] = self.raw[i]
                else:
                    a = dt / max(self.tau_f, dt)
                    if a > 1.0:
                        a = 1.0
                    self.filt[i] += a * (self.raw[i] - self.filt[i])

        # ---------------- setpoints --------------------------------------
        try:
            ra = np.asarray(r, dtype=float).ravel()
        except Exception:
            ra = np.array([], dtype=float)
        sp = [np.nan, np.nan]
        for i in (0, 1):
            v = ra[i] if i < ra.size else np.nan
            if np.isfinite(v):
                self.r_last[i] = float(v)
            sp[i] = self.r_last[i]

        # ---------------- errors -----------------------------------------
        e = [0.0, 0.0]
        for i in (0, 1):
            if np.isfinite(sp[i]) and np.isfinite(self.filt[i]):
                e[i] = self._db(sp[i] - self.filt[i])
            else:
                e[i] = 0.0

        ed = 0.5 * (e[0] - e[1])       # drives split   d = dL - dV
        es = 0.5 * (e[0] + e[1])       # drives internal flows s

        # ---------------- split (external) PI ----------------------------
        i_d_try = self.i_d + self.Ki_d * ed * dt
        d_raw = self.Kp_d * ed + i_d_try
        d = self._clip(d_raw, self.d_lo, self.d_hi)
        # back-calculation anti-windup
        self.i_d = self._clip(i_d_try,
                              self.d_lo - self.Kp_d * ed,
                              self.d_hi - self.Kp_d * ed)
        self.i_d = self._clip(self.i_d, -2.0 * abs(self.d_lo) - 0.5,
                              2.0 * abs(self.d_hi) + 0.5)

        # ---------------- internal flow PI -------------------------------
        i_s_try = self.i_s + self.Ki_s * es * dt
        s_raw = self.Kp_s * es + i_s_try
        s = self._clip(s_raw, self.s_lo, self.s_hi)
        self.i_s = self._clip(i_s_try,
                              self.s_lo - self.Kp_s * es,
                              self.s_hi - self.Kp_s * es)
        self.i_s = self._clip(self.i_s, -0.5, 0.5)

        # ---------------- low purity guard (safe direction) --------------
        meas = [v for v in self.filt if np.isfinite(v)]
        if meas:
            worst = min(meas)
            if worst < self.guard_thresh:
                g = self.guard_gain * (self.guard_thresh - worst)
                s += self._clip(g, 0.0, self.guard_max)

        s = self._clip(s, self.s_lo, self.s_hi + self.guard_max)

        # ---------------- compose flow targets ---------------------------
        tgt = np.array([self.u0[0] + s + 0.5 * d,
                        self.u0[1] + s - 0.5 * d], dtype=float)
        tgt = np.clip(tgt, self.umin, self.umax)
        tgt = self._fix_diff(tgt)
        tgt = np.clip(tgt, self.umin, self.umax)

        # ---------------- rate limit + final safety clamps ---------------
        step_max = self.rate * dt
        du = np.clip(tgt - self.u_cmd, -step_max, step_max)
        u = self.u_cmd + du
        u = self._fix_diff(u)
        u = np.clip(u, self.umin, self.umax)

        if not np.all(np.isfinite(u)):
            u = self.u0.copy()

        self.u_cmd = u
        return self.u_cmd.copy()