import numpy as np


class Controller:
    """Nonlinear feedforward with robust PI feedback for the CSTR."""

    def __init__(self, brief):
        self.brief = brief
        self.dt = float(brief.sample_time)
        self.reset()

    def reset(self):
        self.u0 = np.array([14.19, -1113.5], dtype=float)
        self.u = self.u0.copy()

        self.first_step = True
        self.cb_anchor = np.nan
        self.cb_filtered = np.nan
        self.t_filtered = np.nan
        self.r_cb_filtered = np.nan
        self.r_t_filtered = np.nan

        self.cb_integral = 0.0
        self.t_integral = 0.0
        self.last_r_cb = 1.09
        self.last_r_t = 114.19

        # Nominal low-dilution branch, rescaled from the measured initial point.
        temperature_k = 114.19 + 273.15
        k1 = 1.287e12 * np.exp(-9758.3 / temperature_k)
        k2 = 1.287e12 * np.exp(-9758.3 / temperature_k)
        k3 = 9.043e9 * np.exp(-8560.0 / temperature_k)
        ca0 = 5.1

        self.d_grid = np.linspace(3.0, 14.35, 1000)
        b = self.d_grid + k1
        ca = (
            -b + np.sqrt(b * b + 4.0 * k3 * self.d_grid * ca0)
        ) / (2.0 * k3)
        self.cb_grid = k1 * ca / (self.d_grid + k2)
        self.cb_nominal_anchor = float(
            np.interp(self.u0[0], self.d_grid, self.cb_grid)
        )

    @staticmethod
    def _valid(y, quality, index):
        return (
            index < y.size
            and index < quality.size
            and bool(quality[index])
            and np.isfinite(y[index])
        )

    def _dilution_feedforward(self, cb_setpoint):
        if np.isfinite(self.cb_anchor) and self.cb_anchor > 0.1:
            target = (
                self.cb_nominal_anchor * cb_setpoint / self.cb_anchor
            )
        else:
            target = cb_setpoint

        target = float(
            np.clip(target, self.cb_grid[0], self.cb_grid[-1])
        )
        return float(np.interp(target, self.cb_grid, self.d_grid))

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        quality = np.asarray(quality, dtype=bool)

        cb_good = self._valid(y, quality, 0)
        temp_good = self._valid(y, quality, 1)

        if r.size > 0 and np.isfinite(r[0]):
            self.last_r_cb = float(r[0])
        if r.size > 1 and np.isfinite(r[1]):
            self.last_r_t = float(r[1])

        if cb_good:
            cb_measured = float(np.clip(y[0], 0.0, 1.5))
            if not np.isfinite(self.cb_filtered):
                self.cb_filtered = cb_measured
            if not np.isfinite(self.cb_anchor):
                self.cb_anchor = max(cb_measured, 0.1)

        if temp_good:
            temp_measured = float(np.clip(y[1], 70.0, 170.0))
            if not np.isfinite(self.t_filtered):
                self.t_filtered = temp_measured

        if not np.isfinite(self.r_cb_filtered):
            self.r_cb_filtered = self.last_r_cb
        if not np.isfinite(self.r_t_filtered):
            self.r_t_filtered = self.last_r_t

        # Explicitly preserve the in-service steady actuator values initially.
        if self.first_step:
            self.first_step = False
            return self.u.copy()

        # Setpoint shaping is short relative to the process dynamics but prevents
        # abrupt commands from exciting stiction and analyzer delay.
        alpha_r_cb = self.dt / (10.0 + self.dt)
        alpha_r_t = self.dt / (8.0 + self.dt)
        self.r_cb_filtered += alpha_r_cb * (
            self.last_r_cb - self.r_cb_filtered
        )
        self.r_t_filtered += alpha_r_t * (
            self.last_r_t - self.r_t_filtered
        )

        if cb_good:
            alpha_cb = self.dt / (15.0 + self.dt)
            self.cb_filtered += alpha_cb * (
                cb_measured - self.cb_filtered
            )

            # Average only the initial steady data; stopping early avoids
            # contaminating the calibration with subsequent control movement.
            if float(t) <= 120.0 and abs(self.last_r_cb - 1.09) < 0.01:
                self.cb_anchor += 0.04 * (
                    self.cb_filtered - self.cb_anchor
                )

        if temp_good:
            alpha_t = self.dt / (12.0 + self.dt)
            self.t_filtered += alpha_t * (
                temp_measured - self.t_filtered
            )

        # Concentration loop. Feedforward supplies the large scheduled moves;
        # feedback corrects plant-draw mismatch and the mid-run disturbance.
        d_ff = self._dilution_feedforward(self.r_cb_filtered)
        cb_error = 0.0

        if cb_good and np.isfinite(self.cb_filtered):
            cb_error = self.r_cb_filtered - self.cb_filtered
            integral_error = cb_error if abs(cb_error) > 0.002 else 0.0

            old_integral = self.cb_integral
            self.cb_integral += 0.030 * integral_error * self.dt
            self.cb_integral = float(
                np.clip(self.cb_integral, -6.5, 3.0)
            )

            trial = d_ff + 9.0 * cb_error + self.cb_integral
            if (
                (trial >= 14.35 and integral_error > 0.0)
                or (trial <= 3.0 and integral_error < 0.0)
            ):
                self.cb_integral = old_integral

        d_target = d_ff + 9.0 * cb_error + self.cb_integral
        d_target = float(np.clip(d_target, 3.0, 14.35))

        d_alpha = self.dt / (10.0 + self.dt)
        d_move = d_alpha * (d_target - self.u[0])
        d_move = float(np.clip(d_move, -0.85, 0.85))

        if abs(d_move) >= 0.005:
            self.u[0] = float(
                np.clip(self.u[0] + d_move, 3.0, 14.35)
            )

        # Approximate feed sensible-heat compensation followed by fast PI.
        q_ff = self.u0[1] + 250.0 * (self.u[0] - self.u0[0])
        temp_error = 0.0

        if temp_good and np.isfinite(self.t_filtered):
            temp_error = self.r_t_filtered - self.t_filtered
            integral_error = (
                temp_error if abs(temp_error) > 0.012 else 0.0
            )

            old_integral = self.t_integral
            self.t_integral += 1.55 * integral_error * self.dt
            self.t_integral = float(
                np.clip(self.t_integral, -5200.0, 5200.0)
            )

            trial = q_ff + 390.0 * temp_error + self.t_integral
            if (
                (trial >= 0.0 and integral_error > 0.0)
                or (trial <= -9000.0 and integral_error < 0.0)
            ):
                self.t_integral = old_integral

        q_target = q_ff + 390.0 * temp_error + self.t_integral
        q_target = float(np.clip(q_target, -9000.0, 0.0))

        # High-temperature protection remains independent of normal feedback.
        emergency = False
        if temp_good and np.isfinite(self.t_filtered):
            if self.t_filtered > 124.0:
                q_override = self.u0[1] - 600.0 * (
                    self.t_filtered - 124.0
                )
                q_target = min(
                    q_target,
                    float(np.clip(q_override, -9000.0, 0.0)),
                )
            emergency = self.t_filtered > 134.0

        q_rate = 750.0 if emergency else 260.0
        q_move = float(
            np.clip(q_target - self.u[1], -q_rate, q_rate)
        )

        if abs(q_move) >= 8.0:
            self.u[1] = float(
                np.clip(self.u[1] + q_move, -9000.0, 0.0)
            )

        return self.u.copy()