import numpy as np


class Controller:
    """Conservative multivariable PI controller with hidden-draw protection."""

    def __init__(self, brief):
        self.brief = brief
        self.dt_nominal = float(brief.sample_time)

        self.u0 = np.array([2.70629, 3.20629], dtype=float)
        self.u_min = np.array([1.5, 2.0], dtype=float)
        self.u_max = np.array([4.5, 5.0], dtype=float)

        # Common-mode flow primarily changes separation.
        self.kp_common = 20.0
        self.ki_common = 0.070
        self.ff_common = 38.0

        # Differential flow redistributes production between the products.
        # This loop is intentionally restrained because V-L and F-(V-L)
        # are unmeasured safety-critical product draws.
        self.kp_split = 3.0
        self.ki_split = 0.010

        self.filter_tau = 12.0
        self.backcalc_gain = 0.015

        # q = (V-L)-0.5. These limits retain substantial draw margin.
        self.q_min = -0.08
        self.q_max = 0.06

        self.c_min = -0.25
        self.c_max = 1.20

        self.max_output_step = 0.004
        self.output_deadband = 3.0e-4

        self.reset()

    def reset(self):
        self.initialized = False
        self.last_t = None
        self.y_filt = np.array([0.99, 0.99], dtype=float)
        self.last_good = self.y_filt.copy()
        self.i_common = 0.0
        self.i_split = 0.0
        self.u = self.u0.copy()

    @staticmethod
    def _finite(value):
        return bool(np.isfinite(value))

    def _setpoints(self, r):
        rr = np.asarray(r, dtype=float).reshape(-1)
        result = np.array([0.99, 0.99], dtype=float)
        for i in range(min(2, rr.size)):
            if self._finite(rr[i]):
                result[i] = float(np.clip(rr[i], 0.85, 0.9995))
        return result

    def _update_measurements(self, y, quality, dt):
        yy = np.asarray(y, dtype=float).reshape(-1)
        qq = np.asarray(quality, dtype=bool).reshape(-1)
        valid = np.zeros(2, dtype=bool)

        alpha = 1.0 - np.exp(-dt / max(self.filter_tau, dt))

        for i in range(2):
            if (
                i < yy.size
                and i < qq.size
                and qq[i]
                and self._finite(yy[i])
            ):
                measurement = float(np.clip(yy[i], 0.80, 1.01))
                self.last_good[i] = measurement
                self.y_filt[i] += alpha * (measurement - self.y_filt[i])
                valid[i] = True

        return valid

    def _draw_feedforward(self, r):
        # At nominal feed composition, component balance gives
        # D = (zF-xB)/(xD-xB), with xB = 1-bottoms-heavy-purity.
        x_d = float(r[0])
        x_b = float(1.0 - r[1])
        denominator = x_d - x_b

        if denominator <= 0.2:
            return 0.0

        distillate = (0.5 - x_b) / denominator
        return float(np.clip(distillate - 0.5, self.q_min, self.q_max))

    def _project_coordinates(self, c_raw, q_raw):
        q = float(np.clip(q_raw, self.q_min, self.q_max))

        c_lower = max(
            self.c_min,
            self.u_min[0] - self.u0[0] + 0.5 * q,
            self.u_min[1] - self.u0[1] - 0.5 * q,
        )
        c_upper = min(
            self.c_max,
            self.u_max[0] - self.u0[0] + 0.5 * q,
            self.u_max[1] - self.u0[1] - 0.5 * q,
        )

        return float(np.clip(c_raw, c_lower, c_upper)), q

    def _to_actuators(self, c, q):
        target = np.array(
            [
                self.u0[0] + c - 0.5 * q,
                self.u0[1] + c + 0.5 * q,
            ],
            dtype=float,
        )
        return np.clip(target, self.u_min, self.u_max)

    def _protect_commanded_draws(self, u):
        result = np.asarray(u, dtype=float).copy()
        difference = float(result[1] - result[0])
        safe_difference = float(
            np.clip(difference, 0.5 + self.q_min, 0.5 + self.q_max)
        )

        if abs(safe_difference - difference) > 1.0e-12:
            mean_flow = 0.5 * (result[0] + result[1])
            result[0] = mean_flow - 0.5 * safe_difference
            result[1] = mean_flow + 0.5 * safe_difference

        return np.clip(result, self.u_min, self.u_max)

    def step(self, t, y, r, quality):
        t = float(t)

        if self.last_t is None:
            dt = self.dt_nominal
        else:
            dt = float(np.clip(
                t - self.last_t,
                0.0,
                5.0 * self.dt_nominal,
            ))
            if dt <= 0.0:
                dt = self.dt_nominal
        self.last_t = t

        rr = self._setpoints(r)
        yy = np.asarray(y, dtype=float).reshape(-1)
        qq = np.asarray(quality, dtype=bool).reshape(-1)

        if not self.initialized:
            for i in range(2):
                if (
                    i < yy.size
                    and i < qq.size
                    and qq[i]
                    and self._finite(yy[i])
                ):
                    value = float(np.clip(yy[i], 0.80, 1.01))
                    self.y_filt[i] = value
                    self.last_good[i] = value

            self.initialized = True
            self.u = self.u0.copy()
            return self.u.copy()

        valid = self._update_measurements(yy, qq, dt)

        e_top = float(rr[0] - self.y_filt[0])
        e_bottom = float(rr[1] - self.y_filt[1])
        e_common = 0.5 * (e_top + e_bottom)
        e_split = e_bottom - e_top

        r_common = 0.5 * (rr[0] + rr[1])
        c_ff = self.ff_common * (r_common - 0.99)
        q_ff = self._draw_feedforward(rr)

        # Early separation override prevents delayed analyzers from allowing
        # a moderate purity deterioration to approach the trip boundary.
        minimum_purity = float(np.min(self.y_filt))
        safety_boost = 18.0 * max(0.0, 0.965 - minimum_purity)

        c_raw = (
            c_ff
            + self.kp_common * e_common
            + self.i_common
            + safety_boost
        )
        q_raw = q_ff + self.kp_split * e_split + self.i_split
        c_sat, q_sat = self._project_coordinates(c_raw, q_raw)

        if np.all(valid):
            self.i_common += dt * (
                self.ki_common * e_common
                + self.backcalc_gain * (c_sat - c_raw)
            )
            self.i_split += dt * (
                self.ki_split * e_split
                + self.backcalc_gain * (q_sat - q_raw)
            )

            self.i_common = float(np.clip(self.i_common, -0.35, 1.20))
            self.i_split = float(np.clip(self.i_split, -0.07, 0.05))

            c_raw = (
                c_ff
                + self.kp_common * e_common
                + self.i_common
                + safety_boost
            )
            q_raw = q_ff + self.kp_split * e_split + self.i_split
            c_sat, q_sat = self._project_coordinates(c_raw, q_raw)

        target = self._to_actuators(c_sat, q_sat)

        delta = target - self.u
        delta[np.abs(delta) < self.output_deadband] = 0.0
        delta = np.clip(
            delta,
            -self.max_output_step,
            self.max_output_step,
        )

        self.u = np.clip(self.u + delta, self.u_min, self.u_max)
        self.u = self._protect_commanded_draws(self.u)

        return self.u.copy()