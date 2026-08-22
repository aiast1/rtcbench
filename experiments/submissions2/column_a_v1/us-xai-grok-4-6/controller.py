import numpy as np

class Controller:
    def __init__(self, brief):
        self.dt = float(brief.sample_time)
        self.L0 = 2.70629
        self.V0 = 3.20629
        self.F = 1.0
        self.D0 = self.V0 - self.L0  # 0.5
        self.reset()

    def reset(self):
        self.L = self.L0
        self.V = self.V0
        self.ei_q = 0.0
        self.ei_m = 0.0
        self.yd_f = 0.99
        self.yb_f = 0.99
        self.init = False

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        q = np.asarray(quality, dtype=bool)
        yd = float(y[0]) if y.size else np.nan
        yb = float(y[1]) if y.size > 1 else np.nan
        rd = float(r[0]) if r.size and np.isfinite(r[0]) else 0.99
        rb = float(r[1]) if r.size > 1 and np.isfinite(r[1]) else 0.99
        qd = bool(q[0]) if q.size else False
        qb = bool(q[1]) if q.size > 1 else False

        if qd and np.isfinite(yd):
            self.yd_f = yd if not self.init else 0.06 * yd + 0.94 * self.yd_f
        if qb and np.isfinite(yb):
            self.yb_f = yb if not self.init else 0.06 * yb + 0.94 * self.yb_f
        self.init = True

        ed = rd - self.yd_f
        eb = rb - self.yb_f
        # quality (both products): raise L and V together
        eq = 0.5 * (ed + eb)
        # imbalance: more L vs V if distillate lagging
        em = ed - eb

        freeze = (self.yd_f < 0.90) or (self.yb_f < 0.90)
        if not freeze:
            if qd or qb:
                self.ei_q += eq * self.dt
            if qd and qb:
                self.ei_m += em * self.dt
        self.ei_q = float(np.clip(self.ei_q, -0.3, 0.3))
        self.ei_m = float(np.clip(self.ei_m, -0.2, 0.2))

        dq = 0.45 * eq + 0.0012 * self.ei_q
        dm = 0.25 * em + 0.0006 * self.ei_m
        dq = float(np.clip(dq, -0.0008, 0.0008))
        dm = float(np.clip(dm, -0.0005, 0.0005))

        # L += dq + dm, V += dq - dm  keeps D almost constant if dm small
        L = self.L + dq + dm
        V = self.V + dq - dm

        # hard keep D,B in [0.28, 0.72]
        dmin, dmax = 0.28, 0.72
        D = V - L
        if D < dmin:
            mid = 0.5 * (dmin - D)
            V += mid
            L -= mid
        elif D > dmax:
            mid = 0.5 * (D - dmax)
            V -= mid
            L += mid
        B = L + self.F - V
        if B < dmin:
            mid = 0.5 * (dmin - B)
            L += mid
            V -= mid
        elif B > dmax:
            mid = 0.5 * (B - dmax)
            L -= mid
            V += mid

        L = float(np.clip(L, 1.6, 4.4))
        V = float(np.clip(V, 2.1, 4.9))
        # final D,B
        D = V - L
        if D < dmin:
            V = L + dmin
        if D > dmax:
            V = L + dmax
        B = L + self.F - V
        if B < dmin:
            V = L + self.F - dmin
        if B > dmax:
            V = L + self.F - dmax
        L = float(np.clip(L, 1.6, 4.4))
        V = float(np.clip(V, 2.1, 4.9))

        self.L, self.V = L, V
        return np.array([self.L, self.V], dtype=float)