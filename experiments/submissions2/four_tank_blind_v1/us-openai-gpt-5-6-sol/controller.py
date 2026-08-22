import numpy as np


class Controller:
    """Conservative fast PI control for the two measured lower-tank levels."""

    def __init__(self, brief):
        self.dt = float(brief.sample_time)

        # Increased bandwidth while retaining ample robustness to uncertain
        # gains, transport delay, cross-coupling, and stiction.
        self.kp = np.array([0.46, 0.46], dtype=float)
        self.ki = np.array([0.0062, 0.0062], dtype=float)

        self.filter_tau = 4.5
        self.filter_alpha = self.dt / (self.filter_tau + self.dt)

        self.hard_lo = np.array([0.0, 0.0], dtype=float)
        self.hard_hi = np.array([10.0, 10.0], dtype=float)

        # Unmeasured upper tanks motivate conservative normal pump limits.
        self.soft_lo = np.array([0.15, 0.15], dtype=float)
        self.soft_hi = np.array([5.80, 5.80], dtype=float)

        # Normal slew stays below 0.017 V/s at the stated scan period.
        self.normal_step_limit = min(0.032, 0.016 * self.dt)
        self.emergency_step_limit = 0.080

        self.output_deadband = 0.0025
        self.error_deadband = 0.010
        self.integral_limit = 3.2
        self.antiwindup_gain = 0.10

        self.reset()

    def reset(self):
        self.u = np.array([3.0, 3.0], dtype=float)
        self.integral = np.zeros(2, dtype=float)
        self.yf = np.zeros(2, dtype=float)
        self.y_initialized = np.zeros(2, dtype=bool)
        self.first_step = True

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float).reshape(-1)
        r = np.asarray(r, dtype=float).reshape(-1)
        quality = np.asarray(quality, dtype=bool).reshape(-1)

        good = np.zeros(2, dtype=bool)

        for i in range(2):
            finite = i < y.size and np.isfinite(y[i])
            valid = finite and i < quality.size and bool(quality[i])
            good[i] = valid

            if not self.y_initialized[i]:
                if finite:
                    self.yf[i] = y[i]
                    self.y_initialized[i] = True
            elif valid:
                innovation = np.clip(y[i] - self.yf[i], -1.25, 1.25)
                self.yf[i] += self.filter_alpha * innovation

        # Bumpless transfer from the stated in-service operating point.
        if self.first_step:
            self.first_step = False
            return self.u.copy()

        active = np.zeros(2, dtype=bool)
        error = np.zeros(2, dtype=float)

        for i in range(2):
            if i < r.size and np.isfinite(r[i]) and self.y_initialized[i]:
                active[i] = True
                error[i] = r[i] - self.yf[i]

        integral_error = error.copy()
        integral_error[np.abs(integral_error) < self.error_deadband] = 0.0
        integral_error[~active] = 0.0
        integral_error[~good] = 0.0

        candidate_integral = (
            self.integral + self.ki * self.dt * integral_error
        )
        candidate_integral = np.clip(
            candidate_integral, -self.integral_limit, self.integral_limit
        )

        raw = 3.0 + candidate_integral + self.kp * error
        raw[~active] = self.u[~active]

        valid_levels = self.yf[good & self.y_initialized]
        high_level = (
            float(np.max(valid_levels)) if valid_levels.size else -np.inf
        )

        emergency = high_level > 17.0
        if emergency:
            # Both pumps have direct or indirect paths to both lower tanks.
            raw -= 1.0 * (high_level - 17.0)
            candidate_integral = np.minimum(candidate_integral, 0.0)

        target = np.clip(raw, self.soft_lo, self.soft_hi)

        # Back-calculation for operating saturation. Slew limiting is excluded
        # so the integrator can continue building the command during transients.
        candidate_integral += self.antiwindup_gain * (target - raw)
        self.integral = np.clip(
            candidate_integral, -self.integral_limit, self.integral_limit
        )

        step_limit = (
            self.emergency_step_limit if emergency
            else self.normal_step_limit
        )
        delta = np.clip(target - self.u, -step_limit, step_limit)
        delta[np.abs(delta) < self.output_deadband] = 0.0

        self.u = np.clip(self.u + delta, self.hard_lo, self.hard_hi)
        return self.u.copy()