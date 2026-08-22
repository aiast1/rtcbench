import numpy as np


class Controller:
    """
    Static-decoupling velocity-form PI for the quadruple-tank process in the
    non-minimum-phase configuration (gamma1 + gamma2 < 1, cross-coupling
    dominant: pump 1 mostly drives tank 2 via tank 4, pump 2 mostly drives
    tank 1 via tank 3).

    Design choices:
      * Steady-state gain matrix G from the nominal model, linearised at the
        initial operating levels; control moves are mapped through inv(G) so
        each PI loop sees an (approximately) decoupled unity-gain plant.
      * Velocity (incremental) PI -> inherent anti-windup and bumpless start
        at the as-found actuator position [3.0, 3.0].
      * Conservative tuning (closed loop ~ 2 minutes): the plant has an RHP
        zero, transport delay, noise, and per-scenario parameter draws, so
        robustness beats aggression.
      * Heavy first-order measurement filtering + a small move deadband with
        a pending-move accumulator: keeps actuator travel/duty far below the
        limit without losing integral action.
      * Hard caps on pump voltages chosen so the UNMEASURED upper tanks
        (3 and 4) cannot be driven past their 20 cm overflow even with
        adverse parameter draws; plus a proportional pull-back if a measured
        lower tank approaches overflow.
    """

    def __init__(self, brief):
        self.dt = float(getattr(brief, "sample_time", 2.0) or 2.0)

        # ---- nominal model -> steady-state gain matrix at h ~ [11.3, 11.9]
        g = 981.0
        a1, a2 = 0.071, 0.057
        k1, k2 = 3.14, 3.29
        g1, g2 = 0.43, 0.34
        h0 = np.array([11.28, 11.94])
        # dh_ss/dq_in for each lower tank (cm per cm^3/s)
        c1 = np.sqrt(2.0 * h0[0] / g) / a1
        c2 = np.sqrt(2.0 * h0[1] / g) / a2
        G = np.array([
            [g1 * k1 * c1, (1.0 - g2) * k2 * c1],
            [(1.0 - g1) * k1 * c2, g2 * k2 * c2],
        ])
        self.Ginv = np.linalg.inv(G)

        # ---- PI tuning in decoupled (level, cm) coordinates
        self.Kp = 0.35                # V-equivalent per cm after decoupling
        self.Ki = self.Kp / 110.0     # Ti ~ 110 s

        # ---- actuator handling
        self.u_init = np.array([3.0, 3.0])
        self.u_min = np.array([0.2, 0.2])
        # cap keeps unmeasured upper tanks below overflow even for adverse
        # parameter draws (nominal safe limits are ~6.3 / 6.5 V)
        self.u_max = np.array([4.7, 4.7])
        self.rate = 0.25              # V per control period slew clamp
        self.deadband = 0.01          # V, minimum applied move

        # ---- measurement filter (tau ~ 8 s)
        self.alpha = self.dt / (8.0 + self.dt)

        self.reset()

    def reset(self):
        self.u = self.u_init.copy()
        self.yf = None
        self.e_prev = None
        self.pend = np.zeros(2)   # accumulated not-yet-applied move

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)[:2]
        if quality is None:
            q = np.ones(2, dtype=bool)
        else:
            q = np.asarray(quality).astype(bool)[:2]
        good = q & np.isfinite(y)

        # ---- filtered measurement, hold last value on bad samples
        if self.yf is None:
            init = np.where(np.isfinite(y), y, np.array([11.28, 11.94]))
            self.yf = init.astype(float)
        self.yf = np.where(good, self.yf + self.alpha * (y - self.yf), self.yf)

        # ---- error (nan setpoint -> hold, zero error)
        rr = np.asarray(r, dtype=float)[:2]
        rr = np.where(np.isfinite(rr), rr, self.yf)
        e = rr - self.yf
        if self.e_prev is None:
            self.e_prev = e.copy()

        # ---- velocity-form PI in decoupled coordinates
        dv = self.Kp * (e - self.e_prev) + self.Ki * self.dt * e
        self.e_prev = e.copy()
        du = self.Ginv @ dv

        # ---- measured-level overflow guard (both pumps feed both tanks)
        over = np.maximum(self.yf - 17.5, 0.0)
        if over.sum() > 0.0:
            du = du - 0.3 * over.sum()

        # ---- accumulate, apply deadband + slew limit
        self.pend += du
        move = np.where(np.abs(self.pend) >= self.deadband, self.pend, 0.0)
        move = np.clip(move, -self.rate, self.rate)
        self.pend -= move

        u_new = np.clip(self.u + move, self.u_min, self.u_max)

        # ---- anti-windup: dump pending demand that pushes into a limit
        at_hi = (u_new >= self.u_max - 1e-9) & (self.pend > 0.0)
        at_lo = (u_new <= self.u_min + 1e-9) & (self.pend < 0.0)
        self.pend = np.where(at_hi | at_lo, 0.0, self.pend)

        self.u = u_new
        return self.u.copy()