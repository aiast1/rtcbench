import numpy as np


class Controller:
    """
    Robust, moderately aggressive PI controller for the measured lower tanks.

    Independent error components are additionally treated in common and
    differential coordinates.  This is useful for the four-tank arrangement:
    common pump motion changes total inventory, while opposing pump motion is
    effective for a one-tank setpoint change after the diagonal upper-tank path
    has responded.
    """

    def __init__(self, brief):
        self.dt_nominal = float(brief.sample_time)

        self.u_bias = np.array([3.0, 3.0], dtype=float)
        self.u_min = np.array([0.0, 0.0], dtype=float)
        self.u_max = np.array([5.35, 5.35], dtype=float)

        # Common mode is deliberately somewhat slower than differential mode.
        # This reduces unnecessary total inventory excursions for one-level
        # setpoint moves while improving lower-tank tracking.
        self.kp_common = 0.205
        self.ki_common = 0.0044
        self.kp_diff = 0.285
        self.ki_diff = 0.0056

        self.filter_tau = 5.5
        self.rate_up = np.array([0.070, 0.070], dtype=float)
        self.rate_down = np.array([0.125, 0.125], dtype=float)

        self.integral_limit = 3.35
        self.aw_gain = 0.32
        self.move_deadband = 0.0012

        self.reset()

    def reset(self):
        self.u = self.u_bias.copy()
        self.i_common = 0.0
        self.i_diff = 0.0
        self.yf = np.zeros(2, dtype=float)
        self.r_hold = np.zeros(2, dtype=float)
        self.initialized = False
        self.last_t = None

    def step(self, t, y, r, quality):
        y_in = np.asarray(y, dtype=float).reshape(-1)
        r_in = np.asarray(r, dtype=float).reshape(-1)
        q_in = np.asarray(quality, dtype=bool).reshape(-1)

        yv = np.zeros(2, dtype=float)
        rv = np.full(2, np.nan, dtype=float)
        qv = np.zeros(2, dtype=bool)

        n = min(2, y_in.size)
        if n:
            yv[:n] = y_in[:n]
        n = min(2, r_in.size)
        if n:
            rv[:n] = r_in[:n]
        n = min(2, q_in.size)
        if n:
            qv[:n] = q_in[:n]

        valid_y = qv & np.isfinite(yv)
        yv = np.clip(yv, 0.0, 20.0)
        valid_r = np.isfinite(rv)

        if not self.initialized:
            self.yf[:] = np.where(valid_y, yv, 0.0)
            self.r_hold[:] = self.yf
            self.r_hold[valid_r] = rv[valid_r]
            self.r_hold[:] = np.clip(self.r_hold, 0.0, 20.0)
            self.last_t = float(t)
            self.initialized = True
            return self.u.copy()

        dt = float(t) - float(self.last_t)
        self.last_t = float(t)
        if not np.isfinite(dt) or dt <= 0.0:
            dt = self.dt_nominal
        dt = float(np.clip(dt, 0.5 * self.dt_nominal, 3.0 * self.dt_nominal))

        self.r_hold[valid_r] = rv[valid_r]
        self.r_hold[:] = np.clip(self.r_hold, 0.0, 20.0)

        alpha = dt / (self.filter_tau + dt)
        for i in range(2):
            if valid_y[i]:
                self.yf[i] += alpha * (yv[i] - self.yf[i])

        upper = self.u_max.copy()
        high = -np.inf
        if np.any(valid_y):
            high = float(np.max(self.yf[valid_y]))
            if high > 17.1:
                cap = 5.35 - 1.12 * (high - 17.1)
                upper[:] = np.minimum(upper, np.clip(cap, 1.55, 5.35))

        # If either transmitter is stale, retain the existing command.  This
        # avoids using a stale level to create a differential inventory move.
        if not np.all(valid_y):
            target = self.u.copy()
        else:
            e = self.r_hold - self.yf
            e_common = 0.5 * (e[0] + e[1])
            e_diff = 0.5 * (e[0] - e[1])

            p_common = self.kp_common * e_common
            p_diff = self.kp_diff * e_diff

            raw = np.array([
                self.u_bias[0] + p_common + p_diff + self.i_common + self.i_diff,
                self.u_bias[1] + p_common - p_diff + self.i_common - self.i_diff,
            ])

            limited = np.clip(raw, self.u_min, upper)

            # Transform actuator saturation discrepancy back to controller
            # common/differential coordinates for anti-windup.
            sat_common = 0.5 * ((limited[0] - raw[0]) + (limited[1] - raw[1]))
            sat_diff = 0.5 * ((limited[0] - raw[0]) - (limited[1] - raw[1]))

            self.i_common += self.ki_common * e_common * dt + self.aw_gain * sat_common
            self.i_diff += self.ki_diff * e_diff * dt + self.aw_gain * sat_diff
            self.i_common = float(np.clip(
                self.i_common, -self.integral_limit, self.integral_limit
            ))
            self.i_diff = float(np.clip(
                self.i_diff, -self.integral_limit, self.integral_limit
            ))

            target = np.array([
                self.u_bias[0] + p_common + p_diff + self.i_common + self.i_diff,
                self.u_bias[1] + p_common - p_diff + self.i_common - self.i_diff,
            ])
            target = np.clip(target, self.u_min, upper)

        # Safety-biased common cap limits all presently unobserved upper-tank
        # feed paths whenever a measured lower level approaches its boundary.
        if high > 18.35:
            cap = float(np.clip(3.35 - 0.90 * (high - 18.35), 1.15, 3.35))
            target[:] = np.minimum(target, cap)

        du = target - self.u
        du = np.clip(du, -self.rate_down * dt, self.rate_up * dt)
        du[np.abs(du) < self.move_deadband] = 0.0

        self.u += du
        self.u = np.clip(self.u, self.u_min, upper)
        return self.u.copy()