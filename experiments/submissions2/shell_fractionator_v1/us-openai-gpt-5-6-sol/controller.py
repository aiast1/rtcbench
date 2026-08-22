import numpy as np


class Controller:
    def __init__(self, brief):
        self.dt_nominal = float(brief.sample_time)

        self.K = np.array(
            [
                [4.05, 1.77, 5.88],
                [5.39, 5.72, 6.90],
                [4.38, 4.42, 7.20],
            ],
            dtype=float,
        )
        self.Kinv = np.linalg.inv(self.K)

        # Conservative multivariable PI tuning. The temperature loop is faster
        # because FCV-203 has no nominal dead time to TI-103.
        self.kp = np.array([0.62, 0.62, 0.75], dtype=float)
        self.ti = np.array([58.0, 58.0, 40.0], dtype=float)

        self.u_min = np.full(3, -0.5, dtype=float)
        self.u_max = np.full(3, 0.5, dtype=float)

        self.reset()

    def reset(self):
        self.u = np.zeros(3, dtype=float)
        self.yf = np.zeros(3, dtype=float)
        self.integral = np.zeros(3, dtype=float)
        self.temp_slope = 0.0
        self.temp_raw = 0.0
        self.temp_bad_age = 0.0
        self.last_t = None
        self.initialized = False

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float).reshape(-1)[:3]
        r = np.asarray(r, dtype=float).reshape(-1)[:3]
        quality = np.asarray(quality, dtype=bool).reshape(-1)[:3]

        if self.last_t is None:
            dt = self.dt_nominal
        else:
            dt = float(t) - self.last_t
            dt = float(np.clip(dt, 0.2 * self.dt_nominal, 2.0 * self.dt_nominal))
        self.last_t = float(t)

        valid = quality & np.isfinite(y)

        if not self.initialized:
            self.yf[:] = np.where(valid, y, 0.0)
            if valid[2]:
                self.temp_raw = float(y[2])
            else:
                self.temp_raw = float(self.yf[2])
            self.initialized = True
            self.u[:] = 0.0
            return self.u.copy()

        old_yf = self.yf.copy()

        # Filter quantization and analyzer noise, while holding rejected samples.
        alpha_y = dt / (3.0 + dt)
        self.yf[valid] += alpha_y * (y[valid] - self.yf[valid])

        measured_temp_rate = (self.yf[2] - old_yf[2]) / max(dt, 1.0e-9)
        alpha_slope = dt / (8.0 + dt)
        if valid[2]:
            self.temp_slope += alpha_slope * (
                measured_temp_rate - self.temp_slope
            )
            self.temp_raw = float(y[2])
            self.temp_bad_age = 0.0
        else:
            self.temp_bad_age += dt
            self.temp_slope *= np.exp(-dt / 20.0)

        scored = np.isfinite(r)
        r_used = np.where(scored, r, self.yf)
        error = np.clip(r_used - self.yf, -1.2, 1.2)

        # Avoid integrating quantization-scale errors and never integrate a bad
        # measurement.
        integral_error = error.copy()
        integral_error[np.abs(integral_error) < 0.003] = 0.0
        integral_enable = valid & scored
        self.integral[integral_enable] += (
            dt
            * self.kp[integral_enable]
            / self.ti[integral_enable]
            * integral_error[integral_enable]
        )
        self.integral = np.clip(self.integral, -1.5, 1.5)

        virtual_output = self.kp * error + self.integral
        raw_base = self.Kinv @ virtual_output
        raw = raw_base.copy()

        # Predict a short distance ahead and give TI-103 priority before the
        # hard floor is approached. FCV-203 is used directly because its
        # nominal path to temperature has zero dead time.
        temp_now = min(float(self.yf[2]), float(self.temp_raw))
        if not valid[2]:
            temp_now += min(self.temp_slope, 0.0) * self.temp_bad_age

        temp_prediction = temp_now + 12.0 * min(self.temp_slope, 0.0)
        safety_boost = 1.4 * max(0.0, -0.22 - temp_prediction)
        safety_boost = min(safety_boost, 0.45)
        raw[2] += safety_boost

        clipped = np.clip(raw, self.u_min, self.u_max)

        # Back-calculate only hard-saturation error. Slew limiting is not fed
        # back, avoiding cancellation of the temperature protection action.
        saturation_error = clipped - raw
        if np.any(np.abs(saturation_error) > 1.0e-12):
            self.integral += (
                dt / 22.0
            ) * (self.K @ saturation_error)
            self.integral = np.clip(self.integral, -1.5, 1.5)

        delta = clipped - self.u
        delta[np.abs(delta) < 0.0004] = 0.0

        emergency = safety_boost > 1.0e-5
        if emergency:
            # Prioritize positive FCV-203 travel while keeping total commanded
            # travel below the duty threshold.
            d3 = float(np.clip(delta[2], -0.003, 0.009))
            if d3 < 0.0:
                d3 = 0.0

            remaining = max(0.0, 0.0105 - abs(d3))
            d01 = np.clip(delta[:2], -0.004, 0.004)
            travel01 = float(np.sum(np.abs(d01)))
            if travel01 > remaining and travel01 > 0.0:
                d01 *= remaining / travel01

            applied_delta = np.array([d01[0], d01[1], d3], dtype=float)
        else:
            applied_delta = np.clip(delta, -0.006, 0.006)
            total_travel = float(np.sum(np.abs(applied_delta)))
            if total_travel > 0.0075:
                applied_delta *= 0.0075 / total_travel

        self.u = np.clip(self.u + applied_delta, self.u_min, self.u_max)
        return self.u.copy()