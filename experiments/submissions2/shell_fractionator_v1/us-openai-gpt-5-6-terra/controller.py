import numpy as np


class Controller:
    def __init__(self, brief):
        self.dt_nominal = float(brief.sample_time)

        self.K = np.array([
            [4.05, 1.77, 5.88],
            [5.39, 5.72, 6.90],
            [4.38, 4.42, 7.20],
        ], dtype=float)
        self.Kinv = np.linalg.inv(self.K)

        # Conservative virtual-output PI tuning.  The controller works in
        # output units, then uses the nominal steady-state inverse as a
        # decoupler.
        self.kp = np.array([0.30, 0.30, 0.25], dtype=float)
        self.ki = np.array([0.0030, 0.0030, 0.0025], dtype=float)

        self.u_lo = np.array([-0.5, -0.5, -0.5], dtype=float)
        self.u_hi = np.array([0.5, 0.5, 0.5], dtype=float)

        self.ref_tau = 9.0
        self.meas_alpha = 0.22
        self.normal_slew = 0.0025
        self.safety_slew = 0.0045

        self.reset()

    def reset(self):
        self.last_t = None
        self.yhat = np.zeros(3, dtype=float)
        self.ref = np.zeros(3, dtype=float)
        self.integral = np.zeros(3, dtype=float)
        self.u = np.zeros(3, dtype=float)
        self.initialized = False
        self.last_good = np.zeros(3, dtype=bool)

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float).reshape(-1)
        r = np.asarray(r, dtype=float).reshape(-1)
        quality = np.asarray(quality, dtype=bool).reshape(-1)

        if self.last_t is None:
            dt = self.dt_nominal
        else:
            dt = float(t) - float(self.last_t)
            if not np.isfinite(dt) or dt <= 0.0:
                dt = self.dt_nominal
            dt = min(max(dt, 0.25), 2.0)
        self.last_t = float(t)

        valid = np.zeros(3, dtype=bool)
        valid[:min(3, y.size, quality.size)] = quality[:min(3, y.size, quality.size)]
        finite_y = np.zeros(3, dtype=bool)
        finite_y[:min(3, y.size)] = np.isfinite(y[:min(3, y.size)])
        valid &= finite_y

        if not self.initialized:
            for i in range(3):
                if valid[i]:
                    self.yhat[i] = y[i]
                    self.last_good[i] = True
            self.initialized = True

        for i in range(3):
            if valid[i]:
                if not self.last_good[i]:
                    self.yhat[i] = y[i]
                    self.last_good[i] = True
                else:
                    self.yhat[i] += self.meas_alpha * (y[i] - self.yhat[i])

        target = self.ref.copy()
        for i in range(min(3, r.size)):
            if np.isfinite(r[i]):
                target[i] = r[i]

        ref_alpha = min(1.0, dt / (self.ref_tau + dt))
        self.ref += ref_alpha * (target - self.ref)

        error = self.ref - self.yhat
        # Do not integrate or apply proportional kicks from stale analyzers.
        error_for_control = error.copy()
        error_for_control[~valid] = 0.0

        z_unsat = self.kp * error_for_control + self.integral
        u_base_unsat = self.Kinv.dot(z_unsat)
        u_base_sat = np.clip(u_base_unsat, self.u_lo, self.u_hi)

        # Back-calculation against only hard valve saturation.  Rate limiting
        # is deliberately excluded so the PI controller can work through
        # stiction and actuator lag.
        z_sat = self.K.dot(u_base_sat)
        antiwindup = 0.12 * (z_sat - z_unsat)
        self.integral += dt * self.ki * error_for_control + antiwindup
        self.integral = np.clip(self.integral, -0.75, 0.75)

        z_cmd = self.kp * error_for_control + self.integral
        u_target = self.Kinv.dot(z_cmd)

        # Temperature-floor protection.  FCV-203 is the fastest and strongest
        # temperature actuator.  This layer only becomes active well above the
        # hard floor, and is intentionally biased toward safe positive duty.
        temp_for_safety = self.yhat[2]
        if valid[2]:
            temp_for_safety = y[2]

        safety_bias = 0.0
        if np.isfinite(temp_for_safety):
            safety_bias = max(0.0, (-0.16 - temp_for_safety) * 0.90)
            if temp_for_safety < -0.30:
                safety_bias += (-0.30 - temp_for_safety) * 0.60

        u_target[2] += min(safety_bias, 0.32)
        u_target = np.clip(u_target, self.u_lo, self.u_hi)

        delta = u_target - self.u
        slew = np.full(3, self.normal_slew * dt, dtype=float)

        if u_target[2] > self.u[2] and safety_bias > 0.0:
            slew[2] = self.safety_slew * dt

        delta = np.clip(delta, -slew, slew)
        self.u = np.clip(self.u + delta, self.u_lo, self.u_hi)

        return self.u.copy()