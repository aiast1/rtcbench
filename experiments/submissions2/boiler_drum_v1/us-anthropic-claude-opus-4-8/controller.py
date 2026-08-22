import numpy as np


class Controller:
    """
    Feedwater / drum-level controller for a natural-circulation boiler.

    Priorities after round 1:
      1. Kill the actuator duty breach: duty ~ how often/how much the valve
         reverses. Our rate limit + tiny deadband still let the valve wander
         every period. Fix: strong output low-pass + larger deadband so the
         command moves smoothly and rarely reverses.
      2. Kill the safety violations. The trip that bites is the UNMEASURED
         water inventory (-250 mm) during shrink/swell, plus pressure band.
         Make the loop gentler and lean the steady bias slightly toward
         holding inventory rather than chasing indicated level aggressively.
    """

    def __init__(self, brief):
        self.brief = brief
        self.dt = float(getattr(brief, "sample_time", 5.0))
        self.u0 = 45.0
        self.umin = 0.0
        self.umax = 100.0

        # outer level loop: gentle
        self.Kp_lvl = 0.06
        self.Ki_lvl = 0.0008

        # inner flow loop: moderate
        self.Kp_flow = 0.5
        self.Ki_flow = 0.03

        # inventory estimator
        self.inv_gain = 1.0
        self.inv_leak = 0.002

        self.inv_low = -250.0
        self.ind_high = 250.0

        self.reset()

    def reset(self):
        self.u = self.u0
        self.i_lvl = 0.0
        self.i_flow = 0.0
        self.inv_est = 0.0
        self.last_lvl = 0.0
        self.last_r = 0.0
        self.have_last = False
        self.prev_u = self.u0
        self.u_filt = self.u0

    def _clip(self, x, lo, hi):
        return max(lo, min(hi, x))

    def step(self, t, y, r, quality):
        dt = self.dt

        lvl = float(y[0]) if quality[0] else self.last_lvl
        press = float(y[1]) if len(y) > 1 else 85.0
        w_steam = float(y[2]) if (len(y) > 2 and quality[2]) else None
        w_feed = float(y[3]) if (len(y) > 3 and quality[3]) else None

        if not self.have_last:
            self.last_lvl = lvl
            self.inv_est = lvl
            self.have_last = True

        sp = r[0]
        if sp is None or (isinstance(sp, float) and np.isnan(sp)):
            sp = self.last_r
        sp = float(sp)
        self.last_r = sp

        # ---- inventory estimator ----
        if (w_feed is not None) and (w_steam is not None):
            imb = w_feed - w_steam
        else:
            imb = 0.0
        self.inv_est += self.inv_gain * imb * dt * 0.02
        self.inv_est += self.inv_leak * (lvl - self.inv_est) * dt

        # ---- outer level loop ----
        err_lvl = sp - lvl

        inv_margin = self.inv_est - self.inv_low
        guard = 0.0
        if inv_margin < 150.0:
            guard = (150.0 - inv_margin) * 0.05   # push feed up (protect inventory)

        ind_margin = self.ind_high - lvl
        if ind_margin < 150.0:
            guard -= (150.0 - ind_margin) * 0.05  # back off (protect carryover)

        self.i_lvl += self.Ki_lvl * err_lvl * dt
        self.i_lvl = self._clip(self.i_lvl, -6.0, 6.0)

        feed_bias = self.Kp_lvl * err_lvl + self.i_lvl + guard
        feed_bias = self._clip(feed_bias, -10.0, 12.0)

        # ---- inner flow loop ----
        if (w_feed is not None) and (w_steam is not None):
            feed_target = max(0.0, w_steam + feed_bias)
            err_flow = feed_target - w_feed
            self.i_flow += self.Ki_flow * err_flow * dt
            self.i_flow = self._clip(self.i_flow, -25.0, 25.0)
            u_cmd = self.u0 + self.Kp_flow * err_flow + self.i_flow
        else:
            self.i_flow += self.Ki_flow * err_lvl * 0.05 * dt
            self.i_flow = self._clip(self.i_flow, -25.0, 25.0)
            u_cmd = self.u0 + 0.08 * err_lvl + self.i_flow

        # pressure protection (band 70..100, keep interior)
        if press < 75.0:
            u_cmd -= (75.0 - press) * 1.2
        if press > 96.0:
            u_cmd += (press - 96.0) * 0.4

        u_cmd = self._clip(u_cmd, self.umin, self.umax)

        # anti-windup
        if u_cmd >= self.umax - 1e-6 and self.i_flow > 0:
            self.i_flow -= self.Ki_flow * 1.0 * dt
        if u_cmd <= self.umin + 1e-6 and self.i_flow < 0:
            self.i_flow += self.Ki_flow * 1.0 * dt

        # ---- heavy output smoothing to respect duty limit ----
        # first-order low-pass on the command
        alpha = 0.25
        self.u_filt = self.u_filt + alpha * (u_cmd - self.u_filt)

        # rate limit
        max_step = 1.5  # % per 5 s
        du = self.u_filt - self.prev_u
        if du > max_step:
            du = max_step
        elif du < -max_step:
            du = -max_step
        u_out = self.prev_u + du

        # large deadband to stop chatter / reversals (duty protection)
        if abs(u_out - self.prev_u) < 0.25:
            u_out = self.prev_u
            self.u_filt = self.prev_u

        u_out = self._clip(u_out, self.umin, self.umax)
        self.prev_u = u_out
        self.u = u_out
        self.last_lvl = lvl

        return np.array([u_out], dtype=float)