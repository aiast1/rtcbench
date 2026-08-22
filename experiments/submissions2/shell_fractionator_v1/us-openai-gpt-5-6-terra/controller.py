import numpy as np


class Controller:
    def __init__(self, brief):
        self.dt_nom = float(brief.sample_time)

        self.K = np.array([
            [4.05, 1.77, 5.88],
            [5.39, 5.72, 6.90],
            [4.38, 4.42, 7.20],
        ], dtype=float)

        # A lightly regularized nominal inverse is used as both the static
        # setpoint allocator and the direction for integral correction.
        reg = 0.006
        self.D = np.linalg.solve(
            self.K.T @ self.K + reg * np.eye(3),
            self.K.T
        )

        self.u_min = -0.5
        self.u_max = 0.5

        # Most of the process-side smoothing is already supplied by the long
        # process lags and dead times.  Avoid adding appreciable controller
        # delay to a requested composition move.
        self.ref_tau = 3.0

        # Gains are expressed in decoupled output coordinates.  Integral
        # action is intentionally appreciably stronger than the previous
        # design: steady gain mismatch and the unreported heat-duty upset are
        # otherwise corrected too slowly over this scenario length.
        self.kp = 0.155
        self.ki = 0.0060

        # This remains below the duty threshold, while making the large top
        # composition move reach its required valve position sooner.
        self.max_du_per_sec = 0.014

        self.reset()

    def reset(self):
        self.last_t = None
        self.initialized = False
        self.y_hold = np.zeros(3, dtype=float)
        self.r_filt = np.zeros(3, dtype=float)
        self.i_term = np.zeros(3, dtype=float)
        self.u_last = np.zeros(3, dtype=float)

    def step(self, t, y, r, quality):
        t = float(t)
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        quality = np.asarray(quality, dtype=bool)

        if self.last_t is None:
            dt = self.dt_nom
        else:
            dt = t - self.last_t
            if not np.isfinite(dt) or dt <= 0.0:
                dt = self.dt_nom
            dt = min(dt, 3.0 * self.dt_nom)
        self.last_t = t

        valid_y = quality[:3] & np.isfinite(y[:3])
        if not self.initialized:
            self.y_hold[:] = 0.0
            self.y_hold[valid_y] = y[:3][valid_y]
            self.initialized = True
        else:
            self.y_hold[valid_y] = y[:3][valid_y]

        r_target = self.r_filt.copy()
        valid_r = np.isfinite(r[:3])
        r_target[valid_r] = r[:3][valid_r]
        r_target = np.clip(r_target, -0.55, 0.55)

        alpha = dt / (self.ref_tau + dt)
        self.r_filt += alpha * (r_target - self.r_filt)

        err = self.r_filt - self.y_hold
        err[~valid_y] = 0.0

        # Work in valve coordinates after applying the nominal decoupler.
        u_ff = self.D @ self.r_filt
        u_p = self.kp * (self.D @ err)

        i_candidate = self.i_term + self.ki * dt * (self.D @ err)
        i_candidate = np.clip(i_candidate, -0.34, 0.34)

        # Preserve margin to the hard temperature constraint.  A common
        # positive move is robust because every nominal temperature gain is
        # positive; it is only active near the lower safety region.
        safety_bias = 0.0
        if valid_y[2]:
            temp = self.y_hold[2]
            if temp < -0.13:
                safety_bias += min(0.16, 0.60 * (-0.13 - temp))
            if temp < -0.27:
                safety_bias += min(0.24, 1.30 * (-0.27 - temp))

        u_pre = u_ff + u_p + i_candidate + safety_bias
        u_sat = np.clip(u_pre, self.u_min, self.u_max)

        # Back calculation keeps the integrator aligned with attainable valve
        # positions during the constrained high top-product setpoint.
        sat_gap = u_sat - u_pre
        if np.any(np.abs(sat_gap) > 1e-12):
            i_candidate = np.clip(
                i_candidate + 0.25 * sat_gap,
                -0.34,
                0.34
            )

        self.i_term = i_candidate

        u_cmd = u_ff + u_p + self.i_term + safety_bias
        u_cmd = np.clip(u_cmd, self.u_min, self.u_max)

        max_move = self.max_du_per_sec * dt
        u_out = self.u_last + np.clip(
            u_cmd - self.u_last,
            -max_move,
            max_move
        )
        u_out = np.clip(u_out, self.u_min, self.u_max)

        self.u_last = u_out.copy()
        return u_out