import numpy as np


class Controller:
    """
    Low-travel robust PI controller with steady-state decoupled feedforward.

    The pump commands are biased at the known 3 V operating point.  Setpoint
    movements are handled primarily by a nominal inverse DC gain, while a
    filtered feedback PI term rejects parameter mismatch and disturbances.
    """

    def __init__(self, brief):
        self.dt_nom = float(brief.sample_time)
        self.reset()

    def reset(self):
        self.initialized = False
        self.last_t = None

        self.yf = np.zeros(2, dtype=float)
        self.r0 = np.zeros(2, dtype=float)
        self.ierr = np.zeros(2, dtype=float)
        self.u = np.array([3.0, 3.0], dtype=float)

        # Moderately fast feedback; derivative is intentionally omitted because
        # measurements include delay, quantization, noise, and dropped samples.
        self.kp = np.array([0.150, 0.145], dtype=float)
        self.ki = np.array([0.00215, 0.00205], dtype=float)

        # Feedback shaping approximately cancels the lower-tank DC cross paths.
        self.feedback_decouple = np.array([
            [1.0, -0.46],
            [-0.40, 1.0],
        ], dtype=float)

        # Nominal inverse DC gain, slightly softened for plant-to-plant mismatch.
        # Rows are pump voltages; columns are lower tank level setpoints.
        self.ff = np.array([
            [0.235, -0.115],
            [-0.105, 0.218],
        ], dtype=float)

        self.measurement_tau = 3.5
        self.error_deadband = 0.014
        self.output_deadband = 0.0015
        self.max_step = 0.045

        # Retain margin against hidden upper-tank overflow.
        self.u_min = np.array([0.0, 0.0], dtype=float)
        self.u_max = np.array([4.25, 4.25], dtype=float)

    @staticmethod
    def _valid(y, quality, i):
        return (
            i < len(y)
            and i < len(quality)
            and bool(quality[i])
            and np.isfinite(y[i])
        )

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        quality = np.asarray(quality, dtype=bool)

        if not self.initialized:
            for i in range(2):
                if self._valid(y, quality, i):
                    self.yf[i] = np.clip(y[i], 0.0, 20.0)
                else:
                    self.yf[i] = 0.0

                if i < len(r) and np.isfinite(r[i]):
                    self.r0[i] = np.clip(r[i], 0.0, 20.0)
                else:
                    self.r0[i] = self.yf[i]

            self.last_t = float(t)
            self.initialized = True
            return self.u.copy()

        dt = float(t) - float(self.last_t)
        self.last_t = float(t)
        if not np.isfinite(dt) or dt <= 0.0:
            dt = self.dt_nom
        dt = float(np.clip(dt, 0.5 * self.dt_nom, 3.0 * self.dt_nom))

        alpha = dt / (self.measurement_tau + dt)
        valid = np.zeros(2, dtype=bool)
        for i in range(2):
            if self._valid(y, quality, i):
                ym = np.clip(y[i], 0.0, 20.0)
                self.yf[i] += alpha * (ym - self.yf[i])
                valid[i] = True

        rr = self.r0.copy()
        for i in range(2):
            if i < len(r) and np.isfinite(r[i]):
                rr[i] = np.clip(r[i], 0.0, 20.0)

        err = rr - self.yf
        eint = err.copy()
        eint[np.abs(eint) < self.error_deadband] = 0.0
        eint[~valid] = 0.0

        ff_move = self.ff.dot(rr - self.r0)
        p_move = self.feedback_decouple.dot(self.kp * err)
        i_move = self.feedback_decouple.dot(self.ki * self.ierr)
        u_pre = 3.0 + ff_move + p_move + i_move

        # Per-integrator conditional integration.  Do not integrate an error
        # that would drive any saturated pump farther into its active limit.
        for i in range(2):
            if not valid[i] or eint[i] == 0.0:
                continue

            direction = self.feedback_decouple[:, i] * eint[i]
            drives_high = np.any(
                (u_pre >= self.u_max - 0.01) & (direction > 0.0)
            )
            drives_low = np.any(
                (u_pre <= self.u_min + 0.01) & (direction < 0.0)
            )
            if not drives_high and not drives_low:
                self.ierr[i] += eint[i] * dt

        self.ierr = np.clip(self.ierr, -420.0, 420.0)

        i_move = self.feedback_decouple.dot(self.ki * self.ierr)
        u_desired = 3.0 + ff_move + p_move + i_move
        u_desired = np.clip(u_desired, self.u_min, self.u_max)

        # Conservative measured-level override provides protection for both
        # lower tanks and the hidden upper inventories.
        ymax = float(np.max(self.yf))
        if ymax >= 19.0:
            u_desired = np.minimum(u_desired, np.array([0.10, 0.10]))
        elif ymax >= 18.0:
            u_desired = np.minimum(u_desired, np.array([0.65, 0.65]))
        elif ymax >= 17.0:
            u_desired = np.minimum(u_desired, np.array([2.35, 2.35]))

        delta = np.clip(
            u_desired - self.u,
            -self.max_step,
            self.max_step,
        )
        delta[np.abs(delta) < self.output_deadband] = 0.0

        self.u = np.clip(self.u + delta, self.u_min, self.u_max)
        return self.u.copy()