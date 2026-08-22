import numpy as np


class Controller:
    """
    Robust PI/lead controller for the unstable middle CSTR branch.

    The normal controller uses limited command activity and a deadband. Nonlinear
    cooling authority is introduced as the reactor departs above its setpoint,
    with separate high- and low-temperature recovery logic.
    """

    def __init__(self, brief):
        self.brief = brief
        self.sample_time = float(brief.sample_time)

        self.u_min = 270.0
        self.u_max = 340.0
        self.u_nom = 300.0
        self.t_nom = 350.0

        # Normal-loop tuning. Feedback on absolute temperature stabilizes the
        # open-loop unstable middle branch; setpoint weighting avoids large
        # command steps when the schedule changes.
        self.kp = 3.6
        self.ki = 0.017
        self.beta = 0.48
        self.setpoint_feedforward = -0.30

        self.measurement_tau = 4.5
        self.slope_tau = 10.0
        self.lead_time = 4.5

        self.integral_limit = 24.0
        self.command_deadband = 0.34
        self.reversal_deadband = 0.55
        self.normal_rate = 0.9

        self.reset()

    def reset(self):
        self.initialized = False
        self.last_t = None
        self.filtered_temperature = self.t_nom
        self.temperature_slope = 0.0
        self.integral = 0.0
        self.last_command = self.u_nom
        self.last_setpoint = self.t_nom
        self.last_move = 0.0
        self.bad_count = 0
        self.high_recovery = False
        self.low_recovery = False

    @staticmethod
    def _finite(value):
        return bool(np.isfinite(value))

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        quality = np.asarray(quality, dtype=bool)

        if self.last_t is None:
            dt = self.sample_time
        else:
            dt = float(t) - self.last_t
            if not np.isfinite(dt) or dt <= 0.0:
                dt = self.sample_time
            dt = float(np.clip(
                dt, 0.5 * self.sample_time, 2.0 * self.sample_time
            ))
        self.last_t = float(t)

        if r.size and self._finite(r[0]):
            setpoint = float(np.clip(r[0], 346.0, 354.0))
            self.last_setpoint = setpoint
        else:
            setpoint = self.last_setpoint

        temperature_valid = (
            y.size > 0
            and quality.size > 0
            and bool(quality[0])
            and self._finite(y[0])
            and 275.0 <= float(y[0]) <= 485.0
        )
        jacket_valid = (
            y.size > 1
            and quality.size > 1
            and bool(quality[1])
            and self._finite(y[1])
            and 265.0 <= float(y[1]) <= 345.0
        )

        measured_temperature = (
            float(y[0]) if temperature_valid
            else self.filtered_temperature
        )
        measured_jacket = (
            float(y[1]) if jacket_valid
            else self.last_command
        )

        if not self.initialized:
            self.filtered_temperature = measured_temperature
            self.temperature_slope = 0.0
            self.integral = float(np.clip(
                self.u_nom
                - (
                    self.u_nom
                    + self.setpoint_feedforward * (setpoint - self.t_nom)
                    + self.kp * (
                        self.beta * (setpoint - self.t_nom)
                        - (self.filtered_temperature - self.t_nom)
                    )
                ),
                -self.integral_limit,
                self.integral_limit
            ))
            self.last_command = self.u_nom
            self.initialized = True
            return np.array([self.u_nom], dtype=float)

        if temperature_valid:
            self.bad_count = 0
            previous_filtered = self.filtered_temperature

            alpha_y = dt / (self.measurement_tau + dt)
            self.filtered_temperature += alpha_y * (
                measured_temperature - self.filtered_temperature
            )

            observed_slope = (
                self.filtered_temperature - previous_filtered
            ) / max(dt, 1.0e-9)

            alpha_slope = dt / (self.slope_tau + dt)
            self.temperature_slope += alpha_slope * (
                observed_slope - self.temperature_slope
            )
            self.temperature_slope = float(np.clip(
                self.temperature_slope, -1.5, 1.5
            ))
        else:
            self.bad_count += 1
            # Predict only through short dropouts, then freeze and decay.
            prediction_weight = max(
                0.0, 0.40 - 0.10 * (self.bad_count - 1)
            )
            self.filtered_temperature += (
                prediction_weight * dt * self.temperature_slope
            )
            self.temperature_slope *= np.exp(-dt / 9.0)

        predicted_temperature = (
            self.filtered_temperature
            + self.lead_time * self.temperature_slope
        )
        predicted_temperature = float(np.clip(
            predicted_temperature,
            self.filtered_temperature - 5.0,
            self.filtered_temperature + 5.0
        ))

        feedback_error = (
            self.beta * (setpoint - self.t_nom)
            - (predicted_temperature - self.t_nom)
        )
        tracking_error = setpoint - self.filtered_temperature

        feedforward = (
            self.u_nom
            + self.setpoint_feedforward * (setpoint - self.t_nom)
        )
        proportional = self.kp * feedback_error

        old_integral = self.integral
        integral_candidate = float(np.clip(
            old_integral + self.ki * tracking_error * dt,
            -self.integral_limit,
            self.integral_limit
        ))

        candidate = feedforward + proportional + integral_candidate

        # Smooth nonlinear authority outside the normal tracking band. It is
        # inactive close to setpoint and acts in the stabilizing direction.
        hot_departure = max(
            0.0, predicted_temperature - (setpoint + 1.0)
        )
        cold_departure = max(
            0.0, (setpoint - 2.0) - predicted_temperature
        )
        candidate -= (
            1.15 * hot_departure * hot_departure
            + 0.7 * hot_departure
        )
        candidate += (
            0.45 * cold_departure * cold_departure
            + 0.5 * cold_departure
        )

        # Conditional integration at limits and while strongly departing.
        blocks_high = candidate >= self.u_max and tracking_error > 0.0
        blocks_low = candidate <= self.u_min and tracking_error < 0.0
        hot_windup = hot_departure > 2.0 and tracking_error < 0.0
        cold_windup = cold_departure > 4.0 and tracking_error > 0.0

        if blocks_high or blocks_low or hot_windup or cold_windup:
            self.integral = old_integral
        else:
            self.integral = integral_candidate

        target = feedforward + proportional + self.integral
        target -= (
            1.15 * hot_departure * hot_departure
            + 0.7 * hot_departure
        )
        target += (
            0.45 * cold_departure * cold_departure
            + 0.5 * cold_departure
        )

        hottest = max(
            self.filtered_temperature,
            predicted_temperature,
            measured_temperature if temperature_valid
            else self.filtered_temperature
        )
        coldest = min(
            self.filtered_temperature,
            predicted_temperature,
            measured_temperature if temperature_valid
            else self.filtered_temperature
        )

        # Early high-side detection prevents ignition despite thermowell lag.
        if (
            predicted_temperature >= 357.0
            or self.filtered_temperature >= 358.0
            or (
                predicted_temperature >= max(354.0, setpoint + 1.5)
                and self.temperature_slope >= 0.065
            )
        ):
            self.high_recovery = True
            self.low_recovery = False

        if self.high_recovery:
            if (
                predicted_temperature <= max(352.0, setpoint + 0.2)
                and self.filtered_temperature <= max(353.0, setpoint + 0.5)
                and self.temperature_slope <= 0.0
            ):
                self.high_recovery = False
                self.integral = float(np.clip(
                    self.last_command - feedforward - proportional,
                    -self.integral_limit,
                    self.integral_limit
                ))
            else:
                if hottest >= 365.0 or self.temperature_slope >= 0.20:
                    target = self.u_min
                else:
                    target = min(target, 282.0)

        # Low-side recovery is deliberately late because normal coolant/feed
        # conditions naturally keep the cold branch well above the 300 K trip.
        if not self.high_recovery and (
            predicted_temperature <= 320.0
            or (
                predicted_temperature <= 330.0
                and self.temperature_slope <= -0.16
            )
        ):
            self.low_recovery = True

        if self.low_recovery:
            if (
                coldest >= 340.0
                and self.temperature_slope >= -0.01
            ):
                self.low_recovery = False
                self.integral = float(np.clip(
                    self.last_command - feedforward - proportional,
                    -self.integral_limit,
                    self.integral_limit
                ))
            else:
                target = self.u_max

        target = float(np.clip(target, self.u_min, self.u_max))
        emergency = self.high_recovery or self.low_recovery

        if emergency:
            # Use the skid's full physical slew capability only for safety.
            max_change = 1.5 * dt
            delta = float(np.clip(
                target - self.last_command, -max_change, max_change
            ))
            command = self.last_command + delta
        else:
            desired_change = target - self.last_command

            close_to_command = (
                abs(desired_change) < self.command_deadband
            )
            close_to_delivered = (
                jacket_valid
                and abs(target - measured_jacket) < 0.22
                and abs(desired_change) < 0.65
            )

            # Suppress small direction reversals caused by quantization/noise.
            reversing_small_move = (
                self.last_move != 0.0
                and desired_change * self.last_move < 0.0
                and abs(desired_change) < self.reversal_deadband
            )

            if close_to_command or close_to_delivered or reversing_small_move:
                command = self.last_command
                delta = 0.0
            else:
                max_change = self.normal_rate * dt
                delta = float(np.clip(
                    desired_change, -max_change, max_change
                ))
                command = self.last_command + delta

        command = float(np.clip(command, self.u_min, self.u_max))
        actual_move = command - self.last_command
        if abs(actual_move) > 1.0e-9:
            self.last_move = actual_move
        self.last_command = command

        return np.array([command], dtype=float)