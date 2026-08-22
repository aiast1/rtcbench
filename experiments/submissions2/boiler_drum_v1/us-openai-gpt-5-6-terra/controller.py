import numpy as np


class Controller:
    def __init__(self, brief):
        self.dt_nom = float(brief.sample_time)
        self.reset()

    def reset(self):
        self.last_t = None
        self.started = False

        self.u = 45.0
        self.level = 0.0
        self.p = 85.0
        self.steam = 50.0
        self.feed = 50.0

        self.inventory = 0.0
        self.bias = 0.0
        self.inv_ref = 0.0
        self.sp = 0.0
        self.i = 0.0

        self.capacity = 0.1878
        self.last_p = 85.0

    @staticmethod
    def _valid(q, i):
        try:
            return bool(q[i])
        except Exception:
            return True

    def step(self, t, y, r, quality):
        t = float(t)
        if self.last_t is None:
            dt = self.dt_nom
        else:
            dt = float(np.clip(t - self.last_t, 1.0, 10.0))
        self.last_t = t

        ql = self._valid(quality, 0)
        qp = self._valid(quality, 1)
        qs = self._valid(quality, 2)
        qf = self._valid(quality, 3)

        if ql and np.isfinite(y[0]):
            self.level = float(np.clip(y[0], -500.0, 500.0))

        if qp and np.isfinite(y[1]):
            x = float(np.clip(y[1], 60.0, 110.0))
            a = dt / (5.0 + dt)
            self.p += a * (x - self.p)

        if qs and np.isfinite(y[2]):
            x = float(np.clip(y[2], 0.0, 100.0))
            a = dt / (3.0 + dt)
            self.steam += a * (x - self.steam)

        if qf and np.isfinite(y[3]):
            x = float(np.clip(y[3], 0.0, 100.0))
            a = dt / (4.0 + dt)
            self.feed += a * (x - self.feed)

        if np.size(r) > 0 and np.isfinite(r[0]):
            self.sp = float(np.clip(r[0], -150.0, 150.0))

        if not self.started:
            self.started = True
            self.last_p = self.p
            return np.array([45.0], dtype=float)

        p = self.p
        steam = self.steam
        feed = self.feed

        # Relative water inventory from the independently measured mass
        # balance.  A 20 m2 drum area corresponds to approximately 20 kg/mm.
        self.inventory += (feed - steam) * dt / 20.0
        self.inventory = float(np.clip(self.inventory, -350.0, 400.0))

        dpdt = (p - self.last_p) / max(dt, 1.0)
        self.last_p = p

        # Learn the static LT/inventory difference only in quiet conditions.
        # This excludes the shrink/swell interval after a mass-flow change.
        if ql and abs(feed - steam) < 0.35 and abs(dpdt) < 0.012:
            z = float(np.clip(self.level - self.inventory, -130.0, 130.0))
            a = dt / (600.0 + dt)
            self.bias += a * (z - self.bias)

        # Leave a substantial unmeasured-inventory reserve.  In particular,
        # do not follow the negative indicated-level request into low water:
        # swell can make LT appear healthy while inventory is falling.
        desired = self.sp - self.bias
        desired = float(np.clip(desired, 35.0, 125.0))

        # A slow reference is intentional.  Steam-flow feedforward handles
        # load changes; rapid net cold-feed additions are unsafe for pressure
        # and cause large transient shrink.
        ref_rate = 0.075
        self.inv_ref += float(np.clip(desired - self.inv_ref,
                                      -ref_rate * dt, ref_rate * dt))

        # Hidden low-water protection has priority over setpoint tracking.
        if self.inventory < -20.0:
            self.inv_ref = max(self.inv_ref, 45.0)
        if self.inventory < -80.0:
            self.inv_ref = max(self.inv_ref, 80.0)
        if self.inventory < -140.0:
            self.inv_ref = max(self.inv_ref, 125.0)

        err = self.inv_ref - self.inventory
        kp = 0.070
        ki = 0.000010
        i_try = float(np.clip(self.i + ki * err * dt, -7.0, 7.0))
        excess = kp * err + i_try

        # Pressure margin prevents the level loop from worsening a firing-lag
        # pressure depression.  It is not allowed to defeat low-water action.
        if p < 77.0 and self.inventory > -70.0:
            excess = min(excess, 1.2)
        if p < 74.0 and self.inventory > -70.0:
            excess = min(excess, 0.2)

        # Direct, increasingly forceful carryover protection.  This is based
        # on LT because the upper trip is an indicated two-phase level limit.
        if ql and self.inventory > -115.0:
            if self.level > 135.0:
                excess = min(excess, -0.16 * (self.level - 135.0))
            if self.level > 180.0:
                excess = min(excess, -7.2 - 0.34 * (self.level - 180.0))
            if self.level > 220.0:
                excess = min(excess, -21.0)

        # Conversely, a very low LT is useful as an additional conservative
        # warning, although it is not used as the inventory measurement.
        if ql and self.level < -150.0:
            excess = max(excess, 4.0 + 0.08 * (-150.0 - self.level))

        root_dp = np.sqrt(max(4.0, 120.0 - p))
        max_flow = min(105.0, 100.0 * self.capacity * root_dp)

        requested = steam + excess
        flow_target = float(np.clip(requested, 0.0, max_flow))

        high_sat = requested > max_flow and err > 0.0
        low_sat = requested < 0.0 and err < 0.0
        if not (high_sat or low_sat):
            self.i = i_try

        u_target = 100.0 * flow_target / max(0.05,
                                              self.capacity * root_dp)
        u_target = float(np.clip(u_target, 0.0, 100.0))

        # Filter normal corrections for low travel.  High-level protection is
        # permitted a faster closing move, since carryover has no recovery
        # margin once the indicated surface approaches the separator.
        alpha = dt / (11.0 + dt)
        u_smooth = self.u + alpha * (u_target - self.u)

        max_move = 0.48
        if ql and self.level > 180.0 and u_smooth < self.u:
            max_move = 0.75
        if self.inventory < -120.0 and u_smooth > self.u:
            max_move = 0.65

        u_new = self.u + float(np.clip(u_smooth - self.u,
                                       -max_move, max_move))
        self.u = float(np.clip(u_new, 0.0, 100.0))

        return np.array([self.u], dtype=float)