import numpy as np


class Controller:
    def __init__(self, brief):
        self.sample_time = float(brief.sample_time)

        # Reverse acting: hotter reactor requires colder coolant.
        self.kp = 8.0
        self.ki = 0.020
        self.kd = 10.0

        self.u_min = 270.0
        self.u_max = 340.0
        self.max_command_rate = 1.5

        self.temp_tau = 4.5
        self.rate_tau = 7.5

        self.reset()

    def reset(self):
        self.u = 300.0
        self.i_term = 0.0
        self.last_t = None
        self.last_sp = 350.0
        self.tf = None
        self.rate = 0.0

    def step(self, t, y, r, quality):
        dt = self.sample_time
        if self.last_t is not None:
            observed_dt = float(t) - self.last_t
            if np.isfinite(observed_dt) and observed_dt > 0.1:
                dt = float(np.clip(observed_dt, 0.5, 8.0))
        self.last_t = float(t)

        sp = self.last_sp
        if len(r) > 0 and np.isfinite(r[0]):
            sp = float(r[0])
            self.last_sp = sp

        valid_temp = (
            len(y) > 0
            and len(quality) > 0
            and bool(quality[0])
            and np.isfinite(y[0])
        )

        if not valid_temp:
            return np.array([self.u], dtype=float)

        temp = float(y[0])

        if self.tf is None:
            self.tf = temp
            self.rate = 0.0

        previous_tf = self.tf
        alpha_temp = dt / (self.temp_tau + dt)
        self.tf += alpha_temp * (temp - self.tf)

        raw_rate = (self.tf - previous_tf) / max(dt, 1.0e-6)
        alpha_rate = dt / (self.rate_tau + dt)
        self.rate += alpha_rate * (raw_rate - self.rate)

        error = self.tf - sp
        if abs(error) < 0.02:
            error = 0.0

        # Keep a substantial recovery margin around the actual safety limits.
        # The lower guard is intentionally early because, once this CSTR has
        # fallen off the middle branch, delayed maximum heating may not recover
        # it before the product-dropout limit.
        high_prediction = self.tf + 28.0 * max(self.rate, 0.0)
        low_prediction = self.tf + 22.0 * min(self.rate, 0.0)

        high_risk = temp >= 357.0 or high_prediction >= 356.0
        low_risk = temp <= 315.0 or low_prediction <= 314.0

        if high_risk:
            desired = self.u_min
            # Positive integral drives coolant colder, assisting high-side
            # recovery.  Remove any stored heating demand.
            self.i_term = max(self.i_term, 0.0)
        elif low_risk:
            desired = self.u_max
            # Negative integral drives coolant warmer, assisting low-side
            # recovery.  Remove any stored cooling demand.
            self.i_term = min(self.i_term, 0.0)
        else:
            candidate_i = np.clip(self.i_term + error * dt, -500.0, 500.0)

            unsaturated = (
                300.0
                - self.kp * error
                - self.ki * candidate_i
                - self.kd * self.rate
            )
            desired = float(np.clip(unsaturated, self.u_min, self.u_max))

            # Conditional integration gives saturation anti-windup while still
            # allowing the integral to unwind when the error reverses.
            pushing_lower_limit = unsaturated < self.u_min and error > 0.0
            pushing_upper_limit = unsaturated > self.u_max and error < 0.0
            if not (pushing_lower_limit or pushing_upper_limit):
                self.i_term = candidate_i

        desired = float(np.clip(desired, self.u_min, self.u_max))

        max_step = self.max_command_rate * dt
        self.u += float(np.clip(desired - self.u, -max_step, max_step))
        self.u = float(np.clip(self.u, self.u_min, self.u_max))

        return np.array([self.u], dtype=float)