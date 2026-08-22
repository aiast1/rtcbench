import numpy as np


class Controller:
    """Robust multivariable predictive controller for the fractionator."""

    def __init__(self, brief):
        self.brief = brief
        self.dt = float(brief.sample_time)

        self.ny = 3
        self.nu = 3

        self.K = np.array(
            [
                [4.05, 1.77, 5.88],
                [5.39, 5.72, 6.90],
                [4.38, 4.42, 7.20],
            ],
            dtype=float,
        )
        self.tau = np.array(
            [
                [50.0, 60.0, 50.0],
                [50.0, 60.0, 40.0],
                [33.0, 44.0, 19.0],
            ],
            dtype=float,
        )
        self.delay = np.array(
            [
                [27.0, 28.0, 27.0],
                [18.0, 14.0, 15.0],
                [20.0, 22.0, 0.0],
            ],
            dtype=float,
        )

        self.prediction_horizon = max(90, int(round(130.0 / self.dt)))
        self.block_length = max(4, int(round(10.0 / self.dt)))
        self.nblocks = 13
        self.history_length = max(300, int(round(400.0 / self.dt)))

        self.output_weights = np.array([1.0, 1.0, 1.15], dtype=float)
        self.rate_penalty = 4000.0
        self.rate_difference_penalty = 7000.0

        self.rate_limits = np.array([0.0060, 0.0060, 0.0065])
        self.total_rate_limit = 0.0135

        self._build_prediction_data()
        self.reset()

    def _step_response(self, seconds):
        seconds = np.asarray(seconds, dtype=float)
        scalar = seconds.ndim == 0
        x = seconds.reshape(-1, 1, 1)

        active_time = np.maximum(x - self.delay[None, :, :], 0.0)
        response = self.K[None, :, :] * (
            1.0 - np.exp(-active_time / self.tau[None, :, :])
        )
        response = np.where(x > self.delay[None, :, :], response, 0.0)

        return response[0] if scalar else response

    def _build_prediction_data(self):
        p = self.prediction_horizon
        nvar = self.nblocks * self.nu

        self.H = np.zeros((p * self.ny, nvar), dtype=float)

        for horizon in range(1, p + 1):
            row = slice((horizon - 1) * self.ny, horizon * self.ny)

            for future_move in range(horizon):
                block = min(
                    future_move // self.block_length,
                    self.nblocks - 1,
                )
                age = (horizon - future_move) * self.dt
                response = self._step_response(age)

                col = slice(
                    block * self.nu,
                    (block + 1) * self.nu,
                )
                self.H[row, col] += response

        self.row_weights = np.tile(
            np.sqrt(self.output_weights),
            p,
        )
        weighted_H = self.H * self.row_weights[:, None]

        regularization = self.rate_penalty * np.eye(nvar)

        difference = np.zeros(
            ((self.nblocks - 1) * self.nu, nvar),
            dtype=float,
        )
        row_index = 0
        for block in range(1, self.nblocks):
            for actuator in range(self.nu):
                difference[
                    row_index,
                    (block - 1) * self.nu + actuator,
                ] = -1.0
                difference[
                    row_index,
                    block * self.nu + actuator,
                ] = 1.0
                row_index += 1

        regularization += (
            self.rate_difference_penalty
            * difference.T.dot(difference)
        )

        normal = weighted_H.T.dot(weighted_H) + regularization
        normal += 1e-9 * np.eye(nvar)

        try:
            self.control_gain = np.linalg.solve(normal, weighted_H.T)
        except np.linalg.LinAlgError:
            self.control_gain = np.linalg.pinv(normal).dot(weighted_H.T)

        maximum_index = (
            self.history_length + self.prediction_horizon + 8
        )
        ages = np.arange(maximum_index, dtype=float) * self.dt
        step_responses = self._step_response(ages)
        self.response_complement = (
            self.K[None, :, :] - step_responses
        )

    def reset(self):
        self.u = np.zeros(self.nu, dtype=float)
        self.previous_rate = np.zeros(self.nu, dtype=float)
        self.move_history = []

        self.last_y = np.zeros(self.ny, dtype=float)
        self.filtered_y = np.zeros(self.ny, dtype=float)
        self.filtered_reference = np.zeros(self.ny, dtype=float)
        self.output_bias = np.zeros(self.ny, dtype=float)

        self.previous_temperature = 0.0
        self.safety_latch = False
        self.initialized = False

    def _sanitize_measurements(self, y, quality):
        values = np.asarray(y, dtype=float).reshape(-1)
        flags = np.asarray(quality, dtype=bool).reshape(-1)

        measured = self.last_y.copy()
        good = np.zeros(self.ny, dtype=bool)

        count = min(self.ny, values.size, flags.size)
        for index in range(count):
            if flags[index] and np.isfinite(values[index]):
                measured[index] = float(values[index])
                good[index] = True

        self.last_y[good] = measured[good]
        return measured, good

    def _current_model_output(self):
        model_output = self.K.dot(self.u)

        if self.move_history:
            history = np.asarray(self.move_history, dtype=float)
            ages = np.arange(1, history.shape[0] + 1, dtype=int)
            residual = np.einsum(
                "nij,nj->i",
                self.response_complement[ages],
                history,
            )
            model_output -= residual

        return model_output

    def _free_prediction(self):
        p = self.prediction_horizon
        prediction = np.tile(self.K.dot(self.u), (p, 1))

        if self.move_history:
            history = np.asarray(self.move_history, dtype=float)
            history_ages = np.arange(
                1,
                history.shape[0] + 1,
                dtype=int,
            )
            future_ages = np.arange(1, p + 1, dtype=int)
            indices = future_ages[:, None] + history_ages[None, :]

            residual = np.einsum(
                "hnij,nj->hi",
                self.response_complement[indices],
                history,
            )
            prediction -= residual

        prediction += self.output_bias[None, :]
        return prediction

    def _update_safety_latch(self, temperature, rate, valid):
        if not valid:
            return

        if temperature < -0.30:
            self.safety_latch = True
        elif temperature < -0.20 and rate < -0.0025:
            self.safety_latch = True
        elif (
            self.safety_latch
            and temperature > -0.11
            and rate > -0.0005
        ):
            self.safety_latch = False

    def _limit_rate(self, requested_rate):
        rate = np.clip(
            requested_rate,
            -self.rate_limits,
            self.rate_limits,
        )

        total = float(np.sum(np.abs(rate)))
        if total > self.total_rate_limit:
            rate *= self.total_rate_limit / total

        lower = -0.5 - self.u
        upper = 0.5 - self.u
        return np.clip(rate, lower, upper)

    def step(self, t, y, r, quality):
        measured, good = self._sanitize_measurements(y, quality)

        supplied_reference = np.asarray(r, dtype=float).reshape(-1)
        reference = self.filtered_reference.copy()

        for index in range(min(self.ny, supplied_reference.size)):
            if np.isfinite(supplied_reference[index]):
                reference[index] = float(supplied_reference[index])

        if not self.initialized:
            self.filtered_y = measured.copy()
            self.last_y = measured.copy()
            self.filtered_reference = reference.copy()
            self.previous_temperature = measured[2]
            self.initialized = True
            return self.u.copy()

        measurement_alpha = self.dt / (1.8 + self.dt)
        self.filtered_y[good] += measurement_alpha * (
            measured[good] - self.filtered_y[good]
        )

        reference_alpha = self.dt / (4.0 + self.dt)
        self.filtered_reference += reference_alpha * (
            reference - self.filtered_reference
        )

        model_now = self._current_model_output()
        measured_bias = self.filtered_y - model_now

        bias_alpha = self.dt / (12.0 + self.dt)
        self.output_bias[good] += bias_alpha * (
            measured_bias[good] - self.output_bias[good]
        )
        self.output_bias = np.clip(self.output_bias, -0.8, 0.8)

        temperature_rate = (
            self.filtered_y[2] - self.previous_temperature
        ) / max(self.dt, 1e-12)
        self.previous_temperature = self.filtered_y[2]

        self._update_safety_latch(
            self.filtered_y[2],
            temperature_rate,
            bool(good[2]),
        )

        target = self.filtered_reference.copy()

        if good[2]:
            if self.filtered_y[2] < -0.20:
                target[2] = max(target[2], 0.04)
            if self.safety_latch:
                target[2] = max(target[2], 0.16)

        free_prediction = self._free_prediction().reshape(-1)
        target_vector = np.tile(target, self.prediction_horizon)

        weighted_residual = (
            target_vector - free_prediction
        ) * self.row_weights

        planned_rates = self.control_gain.dot(weighted_residual)
        requested_rate = planned_rates[:self.nu]

        rate_smoothing = 0.40
        requested_rate = (
            (1.0 - rate_smoothing) * self.previous_rate
            + rate_smoothing * requested_rate
        )

        if self.safety_latch and good[2]:
            temperature = self.filtered_y[2]
            protective_rate = np.clip(
                0.0035 + 0.035 * (-0.22 - temperature),
                0.0035,
                0.0080,
            )

            requested_rate[2] = max(
                requested_rate[2],
                protective_rate,
            )
            requested_rate[0] = max(requested_rate[0], 0.0)
            requested_rate[1] = max(requested_rate[1], 0.0)

            if self.u[2] > 0.47 and temperature < -0.26:
                requested_rate[0] = max(requested_rate[0], 0.0012)
                requested_rate[1] = max(requested_rate[1], 0.0012)

        requested_rate[np.abs(requested_rate) < 5e-5] = 0.0
        applied_rate = self._limit_rate(requested_rate)

        new_u = np.clip(self.u + applied_rate, -0.5, 0.5)
        actual_move = new_u - self.u

        self.u = new_u
        self.previous_rate = actual_move.copy()

        self.move_history.insert(0, actual_move.copy())
        if len(self.move_history) > self.history_length:
            self.move_history.pop()

        return self.u.copy()