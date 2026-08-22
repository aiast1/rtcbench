import numpy as np


class Controller:
    def __init__(self, brief):
        self.dt_nominal = float(brief.sample_time)
        self.L0 = 2.70629
        self.V0 = 3.20629

        # Conservative tuning for a slow, delayed composition process.
        self.kp = np.array([8.0, 8.0], dtype=float)
        self.ki = np.array([0.035, 0.035], dtype=float)

        # Measurement filtering and small deadband prevent noise-driven travel.
        self.filter_tau = 12.0
        self.deadband = 2.5e-4

        # Keep commanded motion below the duty/chatter limit.
        self.max_move_per_second = 0.0018

        # Deliberately tighter than the physical limits, leaving room for
        # valve stiction and unmeasured product-rate constraints.
        self.L_min = 1.55
        self.L_max = 4.45
        self.V_min = 2.05
        self.V_max = 4.95

        # V - L is the distillate rate for the nominal material balance.
        # The limits retain substantial positive distillate and bottoms flow.
        self.d_min = 0.30
        self.d_max = 0.70

        self.reset()

    def reset(self):
        self.integral = np.zeros(2, dtype=float)
        self.y_filt = np.array([0.99, 0.99], dtype=float)
        self.r_last = np.array([0.99, 0.99], dtype=float)
        self.u_last = np.array([self.L0, self.V0], dtype=float)
        self.t_last = None
        self.initialized = False

    def _project_safe(self, u):
        L = float(np.clip(u[0], self.L_min, self.L_max))
        V = float(np.clip(u[1], self.V_min, self.V_max))

        d = V - L
        if d < self.d_min:
            V = L + self.d_min
        elif d > self.d_max:
            V = L + self.d_max

        if V > self.V_max:
            V = self.V_max
            L = min(L, V - self.d_min)
        if V < self.V_min:
            V = self.V_min
            L = max(L, V - self.d_max)

        L = float(np.clip(L, self.L_min, self.L_max))
        V = float(np.clip(V, self.V_min, self.V_max))

        d = V - L
        if d < self.d_min:
            L = min(L, self.V_max - self.d_min)
            V = L + self.d_min
        elif d > self.d_max:
            V = max(V, self.L_min + self.d_max)
            L = V - self.d_max

        return np.array([
            np.clip(L, self.L_min, self.L_max),
            np.clip(V, self.V_min, self.V_max),
        ], dtype=float)

    def step(self, t, y, r, quality):
        t = float(t)

        if self.t_last is None:
            dt = self.dt_nominal
        else:
            dt = float(np.clip(t - self.t_last, 0.0, 5.0))
            if dt <= 0.0:
                dt = self.dt_nominal
        self.t_last = t

        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        quality = np.asarray(quality, dtype=bool)

        for i in range(2):
            if i < r.size and np.isfinite(r[i]):
                self.r_last[i] = float(np.clip(r[i], 0.90, 1.0))

        valid = np.zeros(2, dtype=bool)
        for i in range(2):
            valid[i] = (
                i < y.size
                and i < quality.size
                and bool(quality[i])
                and np.isfinite(y[i])
                and 0.80 <= y[i] <= 1.05
            )

        if not self.initialized:
            for i in range(2):
                if valid[i]:
                    self.y_filt[i] = y[i]
            self.initialized = True
            return self.u_last.copy()

        alpha = dt / (self.filter_tau + dt)
        for i in range(2):
            if valid[i]:
                self.y_filt[i] += alpha * (y[i] - self.y_filt[i])

        error = self.r_last - self.y_filt
        error[np.abs(error) < self.deadband] = 0.0

        # Conditional integration: do not wind further into actuator limits.
        old_integral = self.integral.copy()
        for i in range(2):
            if valid[i]:
                self.integral[i] += self.ki[i] * error[i] * dt

        self.integral = np.clip(self.integral, -0.75, 0.75)

        desired = np.array([
            self.L0 + self.kp[0] * error[0] + self.integral[0],
            self.V0 + self.kp[1] * error[1] + self.integral[1],
        ], dtype=float)
        projected = self._project_safe(desired)

        # Back off an integrator if its requested direction is blocked.
        for i in range(2):
            if valid[i]:
                if desired[i] > projected[i] and error[i] > 0.0:
                    self.integral[i] = old_integral[i]
                elif desired[i] < projected[i] and error[i] < 0.0:
                    self.integral[i] = old_integral[i]

        desired = np.array([
            self.L0 + self.kp[0] * error[0] + self.integral[0],
            self.V0 + self.kp[1] * error[1] + self.integral[1],
        ], dtype=float)
        desired = self._project_safe(desired)

        max_step = self.max_move_per_second * max(dt, self.dt_nominal)
        du = np.clip(desired - self.u_last, -max_step, max_step)
        u = self._project_safe(self.u_last + du)

        self.u_last = u
        return u.copy()