import numpy as np


class Controller:
    """
    Four-tank (non-minimum-phase splitter setting) level controller.

    Structure
    ---------
    * Pairing: with gamma < 0.5 most of each pump's flow goes to the DIAGONAL
      upper tank, which drains into the other lower tank.  So pump 1 mainly
      drives h2 and pump 2 mainly drives h1  ->  cross pairing.
    * Static model-inverse feedforward from the setpoint vector: solves the
      2x2 steady-state flow balance for (u1,u2).  At the nominal draw this
      gives exactly [3,3] for the initial setpoint, so the start is bumpless
      and setpoint steps/ramps get instant, well-shaped action without needing
      high loop gain.
    * Two PI loops (cross paired) on top of the feedforward, mild gain
      scheduling on sqrt(level) (gravity-drain gain falls with level),
      clamping anti-windup, no derivative (noisy + delayed measurement).
    * Unmeasured-tank protection: the upper tank levels h3, h4 are RECONSTRUCTED
      from a mass balance on the measured lower tanks
          a3*sqrt(2 g h3) = A1*dh1/dt + a1*sqrt(2 g h1) - gamma1*k1*u1
      plus a lead term from the known inflow, then used to cap the pump
      commands well before the 20 cm overflow.  A second, model-free layer
      caps a slow lag of the commands themselves in case the estimator is
      starved by bad-quality samples.
    """

    G2 = 2.0 * 981.0
    SQ2G = np.sqrt(2.0 * 981.0)

    # nominal model (hint) -- only used for feedforward / estimation
    A1, A2, A3, A4 = 28.0, 32.0, 28.0, 32.0
    a1, a2, a3, a4 = 0.071, 0.057, 0.071, 0.057
    k1, k2 = 3.14, 3.29
    g1, g2 = 0.43, 0.34

    U_MIN, U_ABS_MAX = 0.0, 7.5
    U0 = 3.0
    Y_MAX = 20.0

    KP = 0.62               # V/cm
    TI = 85.0               # s
    F_ALPHA = 0.35          # measurement filter
    DU_MAX = 0.75           # V per sample
    DEADBAND = 0.008        # V command deadband (duty protection)
    H_SCALE = 12.0

    # upper-tank protection
    H_UP_SOFT = 11.0        # start backing off
    H_UP_HARD = 16.0        # fully backed off
    CAP_HI = 7.5
    CAP_LO = 1.5

    # measured-level protection
    SAFE_HI = 16.5
    HARD_HI = 18.5

    def __init__(self, brief):
        try:
            self.dt = float(getattr(brief, "sample_time", 2.0))
        except Exception:
            self.dt = 2.0
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            self.dt = 2.0
        # feedforward matrix:  M @ u = rhs(r)
        self.M = np.array([[self.g1 * self.k1, (1.0 - self.g2) * self.k2],
                           [(1.0 - self.g1) * self.k1, self.g2 * self.k2]])
        try:
            self.Minv = np.linalg.inv(self.M)
        except Exception:
            self.Minv = None
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.u = np.array([self.U0, self.U0], dtype=float)
        self.u_cmd = self.u.copy()
        self.integ = np.zeros(2)
        self.yf = None
        self.yf_prev = None
        self.y_last = None
        self.r_last = np.array([11.28, 11.94], dtype=float)
        self.t_prev = None
        self.dh = np.zeros(2)          # filtered level derivatives
        self.q_up = None               # filtered upper-tank outflows [q3, q4]
        self.h_up = None               # filtered upper-tank level estimates
        self.u_lag = np.array([self.U0, self.U0])   # short lag (delay align)
        self.u_bar = np.array([self.U0, self.U0])   # slow lag (duty/level proxy)
        self.first = True

    # ------------------------------------------------------------------
    def _uff(self, r):
        """static model inverse: steady-state pump voltages for setpoint r"""
        rhs = np.array([self.a1 * self.SQ2G * np.sqrt(max(r[0], 0.05)),
                        self.a2 * self.SQ2G * np.sqrt(max(r[1], 0.05))])
        if self.Minv is None:
            return np.array([self.U0, self.U0])
        u = self.Minv @ rhs
        if not np.all(np.isfinite(u)):
            return np.array([self.U0, self.U0])
        return np.clip(u, 0.2, 6.5)

    @staticmethod
    def _lpf(state, new, dt, tau):
        a = dt / max(tau, dt)
        a = min(a, 1.0)
        return (1.0 - a) * state + a * new

    # ------------------------------------------------------------------
    def step(self, t, y, r, quality):
        dt = self.dt
        if self.t_prev is not None:
            d = float(t) - self.t_prev
            if 0.05 < d < 60.0:
                dt = d
        self.t_prev = float(t)

        # ---------------- measurements ----------------
        y = np.asarray(y, dtype=float).ravel()
        if y.size < 2:
            y = np.concatenate([y, np.full(2 - y.size, np.nan)])
        try:
            q = np.asarray(quality, dtype=bool).ravel()
            if q.size < 2:
                q = np.ones(2, dtype=bool)
        except Exception:
            q = np.ones(2, dtype=bool)

        if self.y_last is None:
            base = np.array([11.28, 11.94], dtype=float)
            for i in range(2):
                if q[i] and np.isfinite(y[i]):
                    base[i] = y[i]
            self.y_last = base.copy()

        good = np.zeros(2, dtype=bool)
        ymeas = self.y_last.copy()
        for i in range(2):
            if q[i] and np.isfinite(y[i]):
                v = float(np.clip(y[i], 0.0, self.Y_MAX))
                ymeas[i] = v
                self.y_last[i] = v
                good[i] = True

        if self.yf is None:
            self.yf = ymeas.copy()
            self.yf_prev = ymeas.copy()
        else:
            a = self.F_ALPHA
            self.yf_prev = self.yf.copy()
            self.yf = (1.0 - a) * self.yf + a * ymeas
        h = self.yf

        # filtered derivatives (slow: only used for the upper-tank estimator)
        draw = (self.yf - self.yf_prev) / dt
        self.dh = self._lpf(self.dh, draw, dt, 20.0)

        # ---------------- setpoints ----------------
        sp = self.r_last.copy()
        rr = np.asarray(r, dtype=float).ravel()
        for i in range(min(2, rr.size)):
            if np.isfinite(rr[i]):
                sp[i] = rr[i]
        sp = np.clip(sp, 0.5, self.Y_MAX - 2.0)
        self.r_last = sp.copy()

        e = sp - h                      # [e1, e2] on measured tanks

        # ---------------- upper tank estimation ----------------
        ul = self.u_lag
        # q3 feeds tank1 (from pump2), q4 feeds tank2 (from pump1)
        q3 = self.A1 * self.dh[0] + self.a1 * self.SQ2G * np.sqrt(max(h[0], 0.0)) \
            - self.g1 * self.k1 * ul[0]
        q4 = self.A2 * self.dh[1] + self.a2 * self.SQ2G * np.sqrt(max(h[1], 0.0)) \
            - self.g2 * self.k2 * ul[1]
        q3 = float(np.clip(q3, 0.0, 60.0))
        q4 = float(np.clip(q4, 0.0, 60.0))
        qm = np.array([q3, q4])
        if self.q_up is None:
            self.q_up = qm.copy()
        else:
            self.q_up = self._lpf(self.q_up, qm, dt, 15.0)

        h3_raw = float(np.clip((self.q_up[0] / self.a3) ** 2 / self.G2, 0.0, 30.0))
        h4_raw = float(np.clip((self.q_up[1] / self.a4) ** 2 / self.G2, 0.0, 30.0))
        hraw = np.array([h3_raw, h4_raw])
        if self.h_up is None:
            self.h_up = hraw.copy()
        else:
            self.h_up = self._lpf(self.h_up, hraw, dt, 20.0)

        # anticipation: net fill rate of the upper tanks with present commands
        qin3 = (1.0 - self.g2) * self.k2 * self.u[1]
        qin4 = (1.0 - self.g1) * self.k1 * self.u[0]
        dh3 = (qin3 - self.q_up[0]) / self.A3
        dh4 = (qin4 - self.q_up[1]) / self.A4
        h3p = float(np.clip(self.h_up[0] + 30.0 * dh3, 0.0, 30.0))
        h4p = float(np.clip(self.h_up[1] + 30.0 * dh4, 0.0, 30.0))

        # caps: pump 2 fills tank 3, pump 1 fills tank 4
        span = max(self.H_UP_HARD - self.H_UP_SOFT, 1e-6)
        x3 = float(np.clip((h3p - self.H_UP_SOFT) / span, 0.0, 1.0))
        x4 = float(np.clip((h4p - self.H_UP_SOFT) / span, 0.0, 1.0))
        cap = np.array([self.CAP_HI - (self.CAP_HI - self.CAP_LO) * x4,
                        self.CAP_HI - (self.CAP_HI - self.CAP_LO) * x3])

        # model-free backup layer: slow lag of the commands
        xb = np.clip((self.u_bar - 4.5) / 1.5, 0.0, 1.0)
        cap = np.minimum(cap, 6.6 - 4.1 * xb)

        # measured-level overflow guard
        hmax = float(np.max(ymeas))
        if hmax > self.SAFE_HI:
            f = float(np.clip((hmax - self.SAFE_HI) /
                              max(self.HARD_HI - self.SAFE_HI, 1e-6), 0.0, 1.0))
            cap = np.minimum(cap, self.U0 * (1.0 - f))
            self.integ *= (1.0 - 0.3 * f)

        cap = np.clip(cap, 0.0, self.U_ABS_MAX)

        # ---------------- feedforward + PI (cross paired) ----------------
        uff = self._uff(sp)
        if self.first:
            # bumpless: make the initial total command equal to the plant's u0
            self.integ = np.array([self.U0, self.U0]) - uff - self.KP * np.array([e[1], e[0]])
            self.integ = np.clip(self.integ, -3.0, 3.0)
            self.first = False

        e_loop = np.array([e[1], e[0]])          # u1 <- h2 error, u2 <- h1 error
        h_loop = np.array([max(h[1], 1.0), max(h[0], 1.0)])
        gs = np.clip(np.sqrt(h_loop / self.H_SCALE), 0.75, 1.3)
        kp = self.KP * gs
        ki = kp / self.TI

        u_unsat = np.zeros(2)
        for i in range(2):
            trial = self.integ[i] + ki[i] * e_loop[i] * dt
            p = kp[i] * e_loop[i]
            raw = uff[i] + p + trial
            hi = min(cap[i], self.U_ABS_MAX)
            if (raw > hi and e_loop[i] > 0.0) or (raw < self.U_MIN and e_loop[i] < 0.0):
                pass                              # freeze integrator
            else:
                self.integ[i] = float(np.clip(trial, -4.0, 4.0))
            u_unsat[i] = uff[i] + p + self.integ[i]

        u_unsat = np.clip(u_unsat, self.U_MIN, np.minimum(cap, self.U_ABS_MAX))

        # ---------------- rate limit, deadband ----------------
        du = np.clip(u_unsat - self.u_cmd, -self.DU_MAX, self.DU_MAX)
        u_new = np.clip(self.u_cmd + du, self.U_MIN, self.U_ABS_MAX)
        self.u_cmd = u_new

        out = self.u.copy()
        for i in range(2):
            if abs(u_new[i] - out[i]) > self.DEADBAND:
                out[i] = u_new[i]
        self.u = np.clip(out, self.U_MIN, self.U_ABS_MAX)

        # lags of the applied command
        self.u_lag = self._lpf(self.u_lag, self.u, dt, 6.0)
        self.u_bar = self._lpf(self.u_bar, self.u, dt, 60.0)

        return self.u.copy()