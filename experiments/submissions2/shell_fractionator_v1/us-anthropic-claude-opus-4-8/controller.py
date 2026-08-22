import numpy as np


class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = float(brief.sample_time)
        self.n = 3
        self.m = 3

        # Nominal gain matrix (steady-state), from model hint
        self.K = np.array([
            [4.05, 1.77, 5.88],
            [5.39, 5.72, 6.90],
            [4.38, 4.42, 7.20],
        ], dtype=float)

        # Robust decoupler: use a regularized pseudo-inverse of K so that
        # the controller works in a "decoupled" command space.
        # Regularization keeps it robust to gain mismatch.
        KtK = self.K.T @ self.K
        reg = 0.15 * np.trace(KtK) / self.n * np.eye(self.n)
        self.Kinv = np.linalg.solve(KtK + reg, self.K.T)

        # Per-channel (decoupled) PI gains. Conservative because of long
        # dead times (15-28 s) and mismatch. Tuned in the decoupled space
        # where effective gain ~ 1.
        # slow integral to remove offset, small proportional.
        self.Kp = np.array([0.35, 0.35, 0.35])
        self.Ki = np.array([0.010, 0.010, 0.012])

        # actuator limits
        self.u_lo = -0.5
        self.u_hi = 0.5

        # slew / duty limit: max change per step
        self.du_max = 0.014  # under the 0.016 duty limit with margin

        self.reset()

    def reset(self):
        self.u = np.zeros(self.m)          # actuator command
        self.integ = np.zeros(self.n)      # integral in decoupled/error space
        self.y_prev = np.zeros(self.n)
        self.y_filt = np.zeros(self.n)
        self.r_prev = None
        self.first = True

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        q = np.asarray(quality, dtype=bool)

        # Handle bad/stale readings: hold previous filtered value
        y_use = np.array(self.y_filt, dtype=float)
        for i in range(self.n):
            if i < len(q) and q[i] and np.isfinite(y[i]):
                y_use[i] = y[i]

        if self.first:
            # initialize filter from first good reading
            for i in range(self.n):
                if i < len(q) and q[i] and np.isfinite(y[i]):
                    y_use[i] = y[i]
            self.y_filt = y_use.copy()
            self.first = False

        # Low-pass filter to tame noise before any derivative-like action
        alpha = 0.25
        self.y_filt = (1 - alpha) * self.y_filt + alpha * y_use

        # Setpoints: replace nan with 0
        rr = np.where(np.isfinite(r), r, 0.0)

        # Tracking error in output space
        err = rr - self.y_filt

        # Map error to decoupled command space
        e_dec = self.Kinv @ err

        # PI in decoupled space
        # Proportional term
        p_term = self.Kp * e_dec

        # Integral update (trapezoidal-ish, simple)
        self.integ += self.Ki * e_dec * self.dt

        u_unsat = p_term + self.integ

        # Clamp to actuator limits
        u_sat = np.clip(u_unsat, self.u_lo, self.u_hi)

        # Anti-windup: back-calculate integral when saturated
        sat_excess = u_unsat - u_sat
        if np.any(np.abs(sat_excess) > 1e-9):
            self.integ -= sat_excess

        # Slew / duty rate limiting
        du = u_sat - self.u
        du = np.clip(du, -self.du_max, self.du_max)
        u_new = self.u + du
        u_new = np.clip(u_new, self.u_lo, self.u_hi)

        self.u = u_new
        return self.u.copy()