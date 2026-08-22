import numpy as np


class Controller:
    def __init__(self, brief):
        self.dt_nominal = float(brief.sample_time)

        self.k1 = 3.14
        self.k2 = 3.29
        self.gamma1 = 0.43
        self.gamma2 = 0.34
        self.sqrt_2g = np.sqrt(2.0 * 981.0)

        # Steady-state pump-to-lower-tank flow map. The off-diagonal
        # terms include flow routed through the upper tanks.
        self.M = np.array([
            [self.gamma1 * self.k1, (1.0 - self.gamma2) * self.k2],
            [(1.0 - self.gamma1) * self.k1, self.gamma2 * self.k2]
        ], dtype=float)
        self.Minv = np.linalg.inv(self.M)

        self.a_lower = np.array([0.071, 0.057], dtype=float)
        self.r_base = np.array([11.28, 11.94], dtype=float)
        self.u_base = np.array([3.0, 3.0], dtype=float)

        # Preserve substantial margin to hidden upper-tank overflow.
        self.u_min = np.array([0.45, 0.45], dtype=float)
        self.u_max = np.array([4.20, 4.20], dtype=float)

        # Fast proportional action follows the direct lower-tank paths.
        # Slower integral action uses the steady-state inverse allocation.
        self.kp_direct = np.array([0.14, 0.14], dtype=float)
        self.ki_flow = np.array([0.0032, 0.0032], dtype=float)
        self.integral_limit = np.array([2.2, 2.2], dtype=float)

        self.measurement_tau = 7.0
        self.reference_tau = 15.0
        self.dynamic_lead = np.array([9.0, 9.0], dtype=float)

        self.output_tau = 3.5
        self.max_output_rate = np.array([0.030, 0.030], dtype=float)
        self.output_deadband = 0.0035
        self.antiwindup_rate = 0.045

        self.reset()

    def reset(self):
        self.initialized = False
        self.last_t = None
        self.elapsed = 0.0

        self.yf = self.r_base.copy()
        self.rf = self.r_base.copy()
        self.last_r = self.r_base.copy()

        self.integral = np.zeros(2, dtype=float)
        self.u = self.u_base.copy()

        # Conservative initial hidden-level envelope.
        upper_inflow_at_3 = np.array([
            (1.0 - self.gamma2) * self.k2 * 3.0,
            (1.0 - self.gamma1) * self.k1 * 3.0
        ], dtype=float)
        upper_a = np.array([0.071, 0.057], dtype=float)
        ratio = (
            1.20 * upper_inflow_at_3
            / (0.80 * upper_a * self.sqrt_2g)
        )
        self.upper_envelope = ratio * ratio

    def _steady_feedforward(self, reference):
        reference = np.clip(reference, 0.0, 20.0)
        q = self.a_lower * self.sqrt_2g * np.sqrt(reference)
        q_base = self.a_lower * self.sqrt_2g * np.sqrt(self.r_base)
        return self.u_base + self.Minv @ (q - q_base)

    def _update_upper_envelope(self, dt):
        # h3 is fed by pump 2 and h4 by pump 1. Parameter choices here
        # intentionally overestimate hidden levels.
        area_worst = np.array([0.80 * 28.0, 0.80 * 32.0], dtype=float)
        outlet_worst = np.array([
            0.80 * 0.071,
            0.80 * 0.057
        ], dtype=float)

        inflow_worst = np.array([
            1.20 * (1.0 - self.gamma2) * self.k2 * self.u[1],
            1.20 * (1.0 - self.gamma1) * self.k1 * self.u[0]
        ], dtype=float)

        outflow_worst = (
            outlet_worst
            * self.sqrt_2g
            * np.sqrt(np.maximum(self.upper_envelope, 0.0))
        )

        self.upper_envelope += (
            dt * (inflow_worst - outflow_worst) / area_worst
        )
        self.upper_envelope = np.clip(self.upper_envelope, 0.0, 30.0)

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float).reshape(-1)
        r = np.asarray(r, dtype=float).reshape(-1)
        quality = np.asarray(quality, dtype=bool).reshape(-1)

        if self.last_t is None:
            dt = self.dt_nominal
        else:
            dt = float(np.clip(
                float(t) - self.last_t,
                0.25 * self.dt_nominal,
                3.0 * self.dt_nominal
            ))
        self.last_t = float(t)
        self.elapsed += dt

        valid_y = np.zeros(2, dtype=bool)
        valid_r = np.zeros(2, dtype=bool)

        for i in range(min(2, y.size, quality.size)):
            valid_y[i] = bool(quality[i]) and np.isfinite(y[i])
        for i in range(min(2, r.size)):
            valid_r[i] = np.isfinite(r[i])

        requested_r = self.last_r.copy()
        for i in range(2):
            if valid_r[i]:
                requested_r[i] = np.clip(r[i], 0.0, 20.0)
        self.last_r = requested_r

        if not self.initialized:
            for i in range(2):
                if valid_y[i]:
                    self.yf[i] = np.clip(y[i], 0.0, 20.0)
            self.rf = requested_r.copy()
            self.initialized = True
            return self.u.copy()

        self._update_upper_envelope(dt)

        alpha_y = 1.0 - np.exp(-dt / self.measurement_tau)
        for i in range(2):
            if valid_y[i]:
                measured = np.clip(y[i], 0.0, 20.0)
                self.yf[i] += alpha_y * (measured - self.yf[i])

        previous_rf = self.rf.copy()
        alpha_r = 1.0 - np.exp(-dt / self.reference_tau)
        self.rf += alpha_r * (requested_r - self.rf)
        reference_rate = (self.rf - previous_rf) / max(dt, 1.0e-9)

        error = self.rf - self.yf

        # Integrate only channels with fresh measurements.
        integral_delta = self.ki_flow * error * dt
        integral_delta[~valid_y] = 0.0
        self.integral += integral_delta
        self.integral = np.clip(
            self.integral,
            -self.integral_limit,
            self.integral_limit
        )

        startup_enable = min(1.0, self.elapsed / 10.0)

        steady_ff = self._steady_feedforward(self.rf)

        # The lead term initially moves the directly connected pump in the
        # required direction, avoiding the inverse response that would result
        # from applying the steady-state inverse immediately.
        dynamic_ff = self.dynamic_lead * reference_rate
        proportional = startup_enable * self.kp_direct * error
        integral_actuation = self.Minv @ self.integral

        raw_target = (
            steady_ff
            + dynamic_ff
            + proportional
            + integral_actuation
        )

        # Measured lower-level safety override.
        maximum_measured = float(np.max(self.yf))
        if maximum_measured > 16.5:
            raw_target -= 0.50 * (maximum_measured - 16.5)

        # Hidden upper-tank safety overrides.
        if self.upper_envelope[0] > 17.5:
            raw_target[1] -= 0.42 * (self.upper_envelope[0] - 17.5)
        if self.upper_envelope[1] > 17.5:
            raw_target[0] -= 0.42 * (self.upper_envelope[1] - 17.5)

        clipped_target = np.clip(raw_target, self.u_min, self.u_max)

        # Back-calculate actuator saturation into the flow-domain integrator.
        saturation_error = clipped_target - raw_target
        if np.any(np.abs(saturation_error) > 1.0e-12):
            self.integral += (
                self.antiwindup_rate
                * dt
                * (self.M @ saturation_error)
            )
            self.integral = np.clip(
                self.integral,
                -self.integral_limit,
                self.integral_limit
            )

        beta_u = 1.0 - np.exp(-dt / self.output_tau)
        proposed = self.u + beta_u * (clipped_target - self.u)

        max_delta = self.max_output_rate * dt
        delta = np.clip(proposed - self.u, -max_delta, max_delta)
        candidate = np.clip(self.u + delta, self.u_min, self.u_max)

        # Suppress quantization-noise motion and stiction dithering.
        hold = np.abs(candidate - self.u) < self.output_deadband
        candidate[hold] = self.u[hold]

        self.u = candidate
        return self.u.copy()