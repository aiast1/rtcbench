import numpy as np


class Controller:
    """Robust dynamic-decoupling PI controller for the four-tank process."""

    def __init__(self, brief):
        self.sample_time = float(brief.sample_time)

        self.gravity = 981.0
        self.area_lower = np.array([28.0, 32.0], dtype=float)
        self.area_upper = np.array([28.0, 32.0], dtype=float)
        self.orifice_lower = np.array([0.071, 0.057], dtype=float)
        self.orifice_upper = np.array([0.071, 0.057], dtype=float)

        self.k1 = 3.33
        self.k2 = 3.35
        self.gamma1 = 0.70
        self.gamma2 = 0.60

        # Direct pump paths into the measured lower tanks.
        self.direct_gain = np.array(
            [self.gamma1 * self.k1, self.gamma2 * self.k2],
            dtype=float,
        )

        # Pump 2 -> tank 3 -> tank 1 and pump 1 -> tank 4 -> tank 2.
        self.upper_input_gain = np.array(
            [
                (1.0 - self.gamma2) * self.k2,
                (1.0 - self.gamma1) * self.k1,
            ],
            dtype=float,
        )

        # Channel-specific tuning compensates for the larger, slower tank 2.
        self.kp = np.array([2.20, 2.40], dtype=float)
        self.integral_time = np.array([52.0, 55.0], dtype=float)
        self.antiwindup_time = 18.0

        self.measurement_tau = 4.5
        self.command_tau = 2.5

        # These pump limits also keep the hidden tanks well below overflow.
        self.u_min = np.array([0.20, 0.20], dtype=float)
        self.u_max = np.array([5.50, 5.50], dtype=float)
        self.integral_limit = np.array([4.5, 4.5], dtype=float)

        self.max_command_step = 0.16
        self.command_deadband = 0.010

        self.reset()

    def reset(self):
        self.initialized = False
        self.last_t = None

        self.y_filtered = np.zeros(2, dtype=float)
        self.integral = np.zeros(2, dtype=float)

        self.u_state = np.array([3.0, 3.0], dtype=float)
        self.u_command = np.array([3.0, 3.0], dtype=float)

        # Nominal hidden-tank equilibrium at the specified initial output.
        hidden_inflow = np.array(
            [
                self.upper_input_gain[0] * 3.0,
                self.upper_input_gain[1] * 3.0,
            ],
            dtype=float,
        )
        self.h_upper = (
            (hidden_inflow / self.orifice_upper) ** 2
            / (2.0 * self.gravity)
        )

    def _outflow(self, level, orifice):
        level = np.maximum(np.asarray(level, dtype=float), 0.0)
        return orifice * np.sqrt(2.0 * self.gravity * level)

    def _propagate_hidden_tanks(self, dt):
        count = max(1, int(np.ceil(dt / self.sample_time)))
        sub_dt = dt / float(count)

        for _ in range(count):
            inflow = np.array(
                [
                    self.upper_input_gain[0] * self.u_command[1],
                    self.upper_input_gain[1] * self.u_command[0],
                ],
                dtype=float,
            )
            outflow = self._outflow(
                self.h_upper, self.orifice_upper
            )
            self.h_upper += (
                sub_dt
                * (inflow - outflow)
                / self.area_upper
            )
            self.h_upper = np.clip(self.h_upper, 0.0, 19.0)

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float).reshape(-1)
        r = np.asarray(r, dtype=float).reshape(-1)
        quality = np.asarray(quality, dtype=bool).reshape(-1)

        if self.last_t is None:
            dt = self.sample_time
        else:
            dt = float(t) - self.last_t
            if not np.isfinite(dt) or dt <= 0.0:
                dt = self.sample_time
            dt = float(np.clip(
                dt,
                0.5 * self.sample_time,
                3.0 * self.sample_time,
            ))
        self.last_t = float(t)

        valid_y = quality[:2] & np.isfinite(y[:2])

        if not self.initialized:
            fallback = np.where(
                np.isfinite(r[:2]), r[:2], 12.0
            )
            initial_y = np.where(valid_y, y[:2], fallback)
            self.y_filtered[:] = np.clip(initial_y, 0.0, 20.0)
            self.initialized = True
            return self.u_command.copy()

        self._propagate_hidden_tanks(dt)

        measurement_alpha = 1.0 - np.exp(
            -dt / self.measurement_tau
        )
        self.y_filtered[valid_y] += measurement_alpha * (
            y[:2][valid_y] - self.y_filtered[valid_y]
        )
        self.y_filtered = np.clip(
            self.y_filtered, 0.0, 20.0
        )

        valid_r = np.isfinite(r[:2])
        reference = np.where(
            valid_r, r[:2], self.y_filtered
        )
        reference = np.clip(reference, 0.25, 18.5)

        error = reference - self.y_filtered
        active_error = np.where(
            valid_y & valid_r, error, 0.0
        )

        reference_outflow = self._outflow(
            reference, self.orifice_lower
        )

        # Local gravity-outflow sensitivity dq/dh.
        flow_slope = reference_outflow / np.maximum(
            2.0 * reference, 0.5
        )

        proportional_flow = self.kp * flow_slope * error

        self.integral += (
            self.kp
            * flow_slope
            / self.integral_time
            * active_error
            * dt
        )
        self.integral = np.clip(
            self.integral,
            -self.integral_limit,
            self.integral_limit,
        )

        requested_total_flow = (
            reference_outflow
            + proportional_flow
            + self.integral
        )

        hidden_outflow = self._outflow(
            self.h_upper, self.orifice_upper
        )

        # Compensate the currently arriving delayed upper-tank flows.
        unconstrained_u = np.array(
            [
                (
                    requested_total_flow[0]
                    - hidden_outflow[0]
                ) / self.direct_gain[0],
                (
                    requested_total_flow[1]
                    - hidden_outflow[1]
                ) / self.direct_gain[1],
            ],
            dtype=float,
        )

        target_u = np.clip(
            unconstrained_u, self.u_min, self.u_max
        )

        achieved_total_flow = np.array(
            [
                self.direct_gain[0] * target_u[0]
                + hidden_outflow[0],
                self.direct_gain[1] * target_u[1]
                + hidden_outflow[1],
            ],
            dtype=float,
        )

        # Back-calculation anti-windup in physical flow coordinates.
        self.integral += (
            dt / self.antiwindup_time
        ) * (achieved_total_flow - requested_total_flow)
        self.integral = np.clip(
            self.integral,
            -self.integral_limit,
            self.integral_limit,
        )

        maximum_measured = -np.inf
        if np.any(valid_y):
            maximum_measured = float(
                np.max(y[:2][valid_y])
            )
            if maximum_measured > 18.0:
                target_u = np.minimum(target_u, 0.40)
            elif maximum_measured > 16.8:
                target_u = np.minimum(target_u, 1.50)

        maximum_hidden = float(np.max(self.h_upper))
        if maximum_hidden > 17.5:
            target_u = np.minimum(target_u, 0.50)
        elif maximum_hidden > 15.5:
            target_u = np.minimum(target_u, 1.75)

        command_alpha = 1.0 - np.exp(
            -dt / self.command_tau
        )
        desired_state = self.u_state + command_alpha * (
            target_u - self.u_state
        )

        delta_u = np.clip(
            desired_state - self.u_state,
            -self.max_command_step,
            self.max_command_step,
        )
        self.u_state = np.clip(
            self.u_state + delta_u,
            self.u_min,
            self.u_max,
        )

        emergency = (
            maximum_measured > 16.8
            or maximum_hidden > 15.5
        )
        if emergency:
            extra_drop = np.clip(
                target_u - self.u_state, -0.30, 0.0
            )
            self.u_state = np.clip(
                self.u_state + extra_drop,
                self.u_min,
                self.u_max,
            )

        # Suppress small noise-driven moves while retaining fast transients.
        move = (
            np.abs(self.u_state - self.u_command)
            >= self.command_deadband
        )
        self.u_command[move] = self.u_state[move]
        self.u_command = np.clip(
            self.u_command, self.u_min, self.u_max
        )

        return self.u_command.copy()