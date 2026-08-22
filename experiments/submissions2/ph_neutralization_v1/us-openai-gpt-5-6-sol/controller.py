import numpy as np


class Controller:
    """Gain-scheduled 2-DOF PID with nonlinear titration compensation."""

    def __init__(self, brief):
        self.sample_time = float(brief.sample_time)
        self.u_start = 14.22
        self.reset()

    @staticmethod
    def _equivalent_base_flow(ph):
        """Map pH to nominal steady-state NaOH flow."""
        ph = float(np.clip(ph, 2.0, 12.5))

        hydrogen = 10.0 ** (-ph)
        hydroxide = 1.0e-14 / max(hydrogen, 1.0e-16)
        water_charge = hydrogen - hydroxide

        k1 = 10.0 ** (-6.35)
        k2 = 10.0 ** (-10.25)
        species_denominator = (
            hydrogen * hydrogen + k1 * hydrogen + k1 * k2
        )

        alpha1 = k1 * hydrogen / species_denominator
        alpha2 = k1 * k2 / species_denominator
        carbonate_charge = alpha1 + 2.0 * alpha2

        acid_term = 16.6 * 0.003 + 0.55 * (-0.03)
        buffer_term = 0.55 * 0.03
        fixed_flow = 16.6 + 0.55

        denominator = (
            0.00305
            - 0.00005 * carbonate_charge
            + water_charge
        )
        numerator = (
            acid_term
            + buffer_term * carbonate_charge
            - fixed_flow * water_charge
        )

        if abs(denominator) < 1.0e-9:
            denominator = np.copysign(1.0e-9, denominator)

        return float(np.clip(numerator / denominator, -10.0, 40.0))

    def reset(self):
        self.first_step = True
        self.last_t = None

        self.u = self.u_start
        self.integral = 0.0

        self.ph_filtered = 6.5
        self.level_filtered = 12.0
        self.have_ph = False
        self.have_level = False

        self.anchor_initialized = False
        self.anchor_coordinate = self._equivalent_base_flow(6.5)
        self.last_setpoint = 6.5

        self.last_coordinate = None
        self.last_coordinate_t = None
        self.coordinate_rate = 0.0

    def _update_measurements(self, t, y, quality, dt):
        ph_valid = (
            y.size > 0
            and quality.size > 0
            and bool(quality[0])
            and np.isfinite(y[0])
            and 0.0 <= float(y[0]) <= 14.0
        )
        level_valid = (
            y.size > 1
            and quality.size > 1
            and bool(quality[1])
            and np.isfinite(y[1])
            and 0.0 <= float(y[1]) <= 35.0
        )

        if ph_valid:
            ph = float(y[0])
            if not self.have_ph:
                self.ph_filtered = ph
                self.have_ph = True
            else:
                alpha = dt / (5.0 + dt)
                self.ph_filtered += alpha * (ph - self.ph_filtered)

            coordinate = self._equivalent_base_flow(self.ph_filtered)
            if self.last_coordinate is not None:
                rate_dt = max(float(t) - self.last_coordinate_t, 0.25)
                raw_rate = (coordinate - self.last_coordinate) / rate_dt
                raw_rate = float(np.clip(raw_rate, -0.35, 0.35))
                rate_alpha = rate_dt / (10.0 + rate_dt)
                self.coordinate_rate += rate_alpha * (
                    raw_rate - self.coordinate_rate
                )

            self.last_coordinate = coordinate
            self.last_coordinate_t = float(t)

        if level_valid:
            level = float(y[1])
            if not self.have_level:
                self.level_filtered = level
                self.have_level = True
            else:
                alpha = dt / (8.0 + dt)
                self.level_filtered += alpha * (
                    level - self.level_filtered
                )

        return ph_valid, level_valid

    def _level_limits(self):
        if not self.have_level:
            return 8.0, 24.5

        level = self.level_filtered
        lower = 8.0
        upper = 24.5

        if level < 9.0:
            lower = max(lower, 12.0 + 2.0 * (9.0 - level))

        if level > 24.0:
            upper = min(upper, 24.5 - 2.0 * (level - 24.0))

        lower = float(np.clip(lower, 8.0, 22.0))
        upper = float(np.clip(upper, 10.0, 24.5))

        if lower > upper:
            midpoint = 0.5 * (lower + upper)
            lower = midpoint
            upper = midpoint

        return lower, upper

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        quality = np.asarray(quality, dtype=bool)

        if self.last_t is None:
            dt = self.sample_time
        else:
            dt = float(np.clip(
                float(t) - self.last_t,
                0.25 * self.sample_time,
                2.0 * self.sample_time
            ))
        self.last_t = float(t)

        ph_valid, level_valid = self._update_measurements(
            t, y, quality, dt
        )

        if ph_valid and not self.anchor_initialized:
            self.anchor_coordinate = self._equivalent_base_flow(
                self.ph_filtered
            )
            self.anchor_initialized = True

        if r.size > 0 and np.isfinite(r[0]):
            requested_setpoint = float(r[0])
            self.last_setpoint = requested_setpoint
        else:
            requested_setpoint = self.last_setpoint

        # Small upper margin accounts for the downstream analyzer delay.
        controlled_setpoint = float(np.clip(
            requested_setpoint, 4.25, 9.88
        ))

        if self.first_step:
            self.first_step = False
            self.u = self.u_start
            return np.array([self.u], dtype=float)

        target_coordinate = self._equivalent_base_flow(
            controlled_setpoint
        )
        measured_coordinate = self._equivalent_base_flow(
            self.ph_filtered
        )
        error = target_coordinate - measured_coordinate

        feedforward = (
            self.u_start
            + target_coordinate
            - self.anchor_coordinate
        )

        kp = 0.82
        ki = kp / 105.0
        kd = 7.0

        derivative = -kd * self.coordinate_rate
        integral_trial = self.integral

        if ph_valid:
            integral_trial += ki * dt * error

        integral_trial = float(np.clip(integral_trial, -8.0, 8.0))

        lower, upper = self._level_limits()
        raw_trial = (
            feedforward
            + kp * error
            + integral_trial
            + derivative
        )

        # Conditional anti-windup at actuator and level-derived limits.
        if (
            (raw_trial > upper and error > 0.0)
            or (raw_trial < lower and error < 0.0)
        ):
            integral_trial = self.integral
            raw_trial = (
                feedforward
                + kp * error
                + integral_trial
                + derivative
            )

        self.integral = integral_trial
        desired = float(np.clip(raw_trial, lower, upper))

        emergency = False
        safety_ph = self.ph_filtered
        if ph_valid:
            safety_ph = float(y[0])

        if safety_ph > 10.05:
            emergency = True
            reduction = 0.4 + 2.0 * max(0.0, safety_ph - 10.05)
            desired = min(desired, self.u - reduction)

        if safety_ph < 4.45:
            emergency = True
            increase = 0.4 + 2.0 * max(0.0, 4.45 - safety_ph)
            desired = max(desired, self.u + increase)

        if level_valid and self.level_filtered > 28.0:
            emergency = True
            desired = min(desired, upper)

        if level_valid and self.level_filtered < 6.5:
            emergency = True
            desired = max(desired, lower)

        desired = float(np.clip(desired, lower, upper))

        max_step = 0.95 if emergency else 0.65
        delta = float(np.clip(desired - self.u, -max_step, max_step))

        # Avoid spending actuator duty on quantization-scale corrections.
        if not emergency and abs(delta) < 0.04:
            delta = 0.0

        self.u = float(np.clip(self.u + delta, lower, upper))
        return np.array([self.u], dtype=float)