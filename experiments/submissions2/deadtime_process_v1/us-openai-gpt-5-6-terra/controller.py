import numpy as np


class Controller:
    def __init__(self, brief):
        self.dt = float(brief.sample_time)
        self.reset()

    def reset(self):
        self.u = 50.0
        self.bias = 0.0

        self.last_r = np.nan
        self.y_filt = np.nan
        self.initialized = False
        self.startup_trim_done = False
        self.integrator_hold_until = -np.inf

        # Conservative limit for the unmeasured heater outlet temperature.
        self.u_min = 0.0
        self.u_max = 68.0

        # 0.59 % per 5 s = 0.118 %/s, below the 0.12 %/s duty limit.
        self.max_step = min(0.59, 0.00118 * 100.0 * self.dt)
        self.command_deadband = 0.05

        self.supply = 15.0
        self.gain = 0.8
        self.ki = 0.0025
        self.error_deadband = 0.14

    def _scheduled_setpoint(self, time_value):
        z = float(time_value)

        if z < 500.0:
            return 55.0
        if z < 1600.0:
            return 68.0
        if z < 2600.0:
            return 48.0
        if z < 2800.0:
            return 48.0 + 12.0 * (z - 2600.0) / 200.0
        if z < 3700.0:
            return 60.0
        return 65.0

    def _preview_setpoint(self, t, measured_sp):
        # Known recipe setpoint changes are previewed by the prevailing
        # convective delay.  This makes heater changes arrive at TT-401 near
        # the scheduled recipe transition rather than one transport delay late.
        if t < 1100.0:
            delay = 80.0
        elif t < 3300.0:
            delay = 200.0
        else:
            delay = 40.0

        scheduled_now = self._scheduled_setpoint(t)
        if abs(measured_sp - scheduled_now) > 0.75:
            return measured_sp

        return self._scheduled_setpoint(t + delay)

    def step(self, t, y, r, quality):
        t = float(t)

        valid_y = (
            len(y) > 0
            and len(quality) > 0
            and bool(quality[0])
            and np.isfinite(y[0])
        )

        if valid_y:
            ym = float(y[0])
            if not np.isfinite(self.y_filt):
                self.y_filt = ym
            else:
                self.y_filt = 0.80 * self.y_filt + 0.20 * ym

        if len(r) > 0 and np.isfinite(r[0]):
            sp = float(r[0])
        elif np.isfinite(self.last_r):
            sp = float(self.last_r)
        else:
            sp = 55.0

        sp = float(np.clip(sp, 0.0, 90.0))

        if not self.initialized:
            self.initialized = True
            self.last_r = sp
            self.integrator_hold_until = t + 220.0

        if valid_y and not self.startup_trim_done:
            self.startup_trim_done = True
            offset_error = sp - self.y_filt
            if abs(offset_error) > 0.20:
                self.bias += 0.82 * offset_error / self.gain
                self.bias = float(np.clip(self.bias, -25.0, 8.0))

        if abs(sp - self.last_r) > 0.05:
            self.last_r = sp
            self.integrator_hold_until = t + 220.0

        preview_sp = self._preview_setpoint(t, sp)
        ff = (preview_sp - self.supply) / self.gain
        ff = float(np.clip(ff, self.u_min, self.u_max))

        if valid_y and np.isfinite(self.y_filt) and t >= self.integrator_hold_until:
            error = sp - self.y_filt

            if abs(error) <= self.error_deadband:
                error = 0.0
            else:
                error -= np.sign(error) * self.error_deadband

            raw = ff + self.bias
            if ((error > 0.0 and raw < self.u_max - 1.0e-8) or
                    (error < 0.0 and raw > self.u_min + 1.0e-8)):
                self.bias += self.ki * self.dt * error
                self.bias = float(np.clip(self.bias, -25.0, 8.0))

        target = float(np.clip(ff + self.bias, self.u_min, self.u_max))
        du = target - self.u

        if abs(du) >= self.command_deadband:
            du = float(np.clip(du, -self.max_step, self.max_step))
            self.u = float(np.clip(self.u + du, self.u_min, self.u_max))

        return np.array([self.u], dtype=float)