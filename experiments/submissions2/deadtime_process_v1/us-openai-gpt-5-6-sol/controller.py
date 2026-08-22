import numpy as np


class Controller:
    """Delay-aware adaptive-feedforward PI controller."""

    def __init__(self, brief):
        self.brief = brief
        self.dt_nominal = float(brief.sample_time)
        self.reset()

    def reset(self):
        self.u = 50.0
        self.integral = 0.0
        self.p_term = 0.0

        self.y_filt = np.nan
        self.initial_y = np.nan
        self.last_valid_y = np.nan
        self.last_r = 55.0
        self.last_t = None

        self.initialized = False
        self.first_call = True

        self.k_est = 0.8
        self.k_ff = 0.8
        self.u_safe_max = 77.0

        self.kp = 0.68
        self.ki = 0.0052
        self.filter_tau = 16.0
        self.integral_limit = 22.0

        self.feedback_hold_until = 0.0
        self.reference_hold_time = 225.0

        self.step_monitor_active = False
        self.step_start_time = 0.0
        self.step_y0 = np.nan
        self.step_r0 = 55.0
        self.step_r1 = 55.0
        self.release_progress = 0.72

        self.max_step = 3.0
        self.move_deadband = 0.12
        self.output_resolution = 0.1

    def _initialize(self, t, y_value, reference):
        y_value = float(y_value)
        self.y_filt = y_value
        self.initial_y = y_value
        self.last_valid_y = y_value

        # The process is initially at steady state with 50% heater duty.
        # This provides a useful one-point estimate of heater gain. Blending
        # with nominal gain limits sensitivity to supply-temperature mismatch.
        measured_gain = (y_value - 15.0) / 50.0
        self.k_est = float(np.clip(measured_gain, 0.35, 1.20))
        self.k_ff = float(np.clip(
            0.75 * self.k_est + 0.25 * 0.8,
            0.45,
            1.10
        ))

        # Conservative hidden heater-outlet constraint.
        allowable_rise = max(0.0, 88.0 - y_value)
        cap = 50.0 + allowable_rise / (1.18 * self.k_est)
        self.u_safe_max = float(np.clip(cap, 55.0, 94.0))

        self.initialized = True

        # If the initial equilibrium is off setpoint, feedforward corrects it.
        # Hold feedback until that correction has traversed the line.
        if abs(reference - y_value) > 0.35:
            self.feedback_hold_until = t + self.reference_hold_time
            self.step_monitor_active = True
            self.step_start_time = t
            self.step_y0 = y_value
            self.step_r0 = y_value
            self.step_r1 = reference

    def _feedforward(self, reference):
        if not self.initialized:
            return 50.0 + (float(reference) - 55.0) / 0.8

        # Anchoring at the measured initial equilibrium makes this command
        # bumpless when reference equals the initial process temperature.
        return 50.0 + (float(reference) - self.initial_y) / self.k_ff

    def _start_step_monitor(self, t, old_reference, new_reference):
        if not self.initialized or not np.isfinite(self.y_filt):
            return

        self.feedback_hold_until = max(
            self.feedback_hold_until,
            t + self.reference_hold_time
        )
        self.step_monitor_active = True
        self.step_start_time = t
        self.step_y0 = float(self.y_filt)
        self.step_r0 = float(old_reference)
        self.step_r1 = float(new_reference)

    def step(self, t, y, r, quality):
        t = float(t)
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        quality = np.asarray(quality, dtype=bool)

        if self.last_t is None:
            dt = self.dt_nominal
        else:
            dt = float(np.clip(
                t - self.last_t,
                0.2 * self.dt_nominal,
                2.0 * self.dt_nominal
            ))
        self.last_t = t

        reference = self.last_r
        if r.size > 0 and np.isfinite(r[0]):
            reference = float(r[0])

        measurement_good = (
            y.size > 0
            and quality.size > 0
            and bool(quality[0])
            and np.isfinite(y[0])
        )

        if measurement_good and not self.initialized:
            self._initialize(t, float(y[0]), reference)

        if self.first_call:
            self.first_call = False
            self.last_r = reference
            self.u = 50.0
            return np.array([self.u], dtype=float)

        if measurement_good:
            measured = float(y[0])
            alpha = dt / (self.filter_tau + dt)

            if np.isfinite(self.y_filt):
                self.y_filt += alpha * (measured - self.y_filt)
            else:
                self.y_filt = measured

            self.last_valid_y = measured

        reference_change = reference - self.last_r

        # Full setpoint steps get response-based deadtime handling. Ramp
        # increments extend the hold but do not repeatedly reset the monitor.
        if abs(reference_change) >= 1.0:
            self._start_step_monitor(t, self.last_r, reference)
        elif abs(reference_change) > 0.05:
            self.feedback_hold_until = max(
                self.feedback_hold_until,
                t + self.reference_hold_time
            )

        self.last_r = reference

        if self.step_monitor_active and measurement_good:
            requested_change = self.step_r1 - self.step_r0
            measured_change = self.y_filt - self.step_y0

            if abs(requested_change) > 0.35:
                progress = measured_change / requested_change

                # Resume feedback once most of the new material has reached
                # the outlet, avoiding correction based on the old condition.
                if progress >= self.release_progress:
                    self.feedback_hold_until = t
                    self.step_monitor_active = False

            if t >= self.feedback_hold_until:
                self.step_monitor_active = False

        u_ff = self._feedforward(reference)
        hold_active = t < self.feedback_hold_until

        if self.initialized and not hold_active and np.isfinite(self.y_filt):
            error = reference - self.y_filt
            self.p_term = self.kp * error

            if measurement_good:
                trial_integral = float(np.clip(
                    self.integral + self.ki * error * dt,
                    -self.integral_limit,
                    self.integral_limit
                ))

                trial_raw = u_ff + self.p_term + trial_integral
                pushing_high = trial_raw > self.u_safe_max and error > 0.0
                pushing_low = trial_raw < 0.0 and error < 0.0

                if not (pushing_high or pushing_low):
                    self.integral = trial_integral
        else:
            self.p_term = 0.0

        raw_target = u_ff + self.integral + self.p_term
        target = float(np.clip(raw_target, 0.0, self.u_safe_max))

        if self.initialized and not hold_active:
            if raw_target > self.u_safe_max:
                self.integral += 0.10 * (self.u_safe_max - raw_target)
            elif raw_target < 0.0:
                self.integral += 0.10 * (0.0 - raw_target)

            self.integral = float(np.clip(
                self.integral,
                -self.integral_limit,
                self.integral_limit
            ))

        delta = target - self.u
        if abs(delta) >= self.move_deadband:
            delta = float(np.clip(delta, -self.max_step, self.max_step))
            new_u = self.u + delta
            new_u = self.output_resolution * np.round(
                new_u / self.output_resolution
            )
            self.u = float(np.clip(new_u, 0.0, self.u_safe_max))

        return np.array([self.u], dtype=float)