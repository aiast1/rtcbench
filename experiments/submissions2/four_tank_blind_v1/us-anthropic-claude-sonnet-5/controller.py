import numpy as np

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = getattr(brief, "sample_time", 2.0)
        self.n_meas = 2
        self.n_act = 2

        # Actuator limits
        self.u_min = 0.0
        self.u_max = 10.0
        self.u_init = np.array([3.0, 3.0])

        # PI gains (decentralized) - pushed further since round 2 showed
        # headroom (no violations, duty well under limit).
        self.Kp = np.array([0.75, 0.75])
        self.Ti = np.array([65.0, 65.0])   # integral time constant (s)
        self.Ki = self.Kp / self.Ti

        # Back-calculation anti-windup time constant (s)
        self.Tt = np.array([20.0, 20.0])

        # Setpoint internal slew limiter (cm/s) - loosened further for
        # faster tracking while still protecting unmeasured tanks.
        self.sp_slew = 0.18  # cm per second

        # Output rate limiter (V per control step)
        self.max_du = 0.15

        self.reset()

    def reset(self):
        self.integ = self.u_init.copy()
        self.u_prev = self.u_init.copy()
        self.y_prev = np.array([np.nan, np.nan])
        self.sp_filt = None
        self.first_call = True

    def _slew_sp(self, r):
        if self.sp_filt is None:
            self.sp_filt = r.copy()
            return self.sp_filt
        max_step = self.sp_slew * self.dt
        delta = r - self.sp_filt
        delta = np.clip(delta, -max_step, max_step)
        self.sp_filt = self.sp_filt + delta
        return self.sp_filt

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float).copy()
        r = np.asarray(r, dtype=float).copy()
        q = np.asarray(quality, dtype=bool)

        # Handle stale/bad samples: hold last good value
        for i in range(self.n_meas):
            if not q[i] or not np.isfinite(y[i]):
                if np.isfinite(self.y_prev[i]):
                    y[i] = self.y_prev[i]
                else:
                    y[i] = r[i] if np.isfinite(r[i]) else 10.0
            else:
                self.y_prev[i] = y[i]

        # Safety clamp on measured values before use (defensive)
        y = np.clip(y, 0.0, 20.0)

        # Setpoint slew-limiting to keep transients gentle (protects unmeasured tanks)
        r_use = r.copy()
        if self.sp_filt is not None:
            for i in range(self.n_meas):
                if not np.isfinite(r_use[i]):
                    r_use[i] = self.sp_filt[i]
        else:
            for i in range(self.n_meas):
                if not np.isfinite(r_use[i]):
                    r_use[i] = y[i]

        sp = self._slew_sp(r_use)

        err = sp - y

        # Bumpless first call: align integrator with current output/init
        if self.first_call:
            self.integ = self.u_init.copy()
            self.u_prev = self.u_init.copy()
            self.first_call = False

        u_cmd = np.zeros(self.n_act)

        for i in range(self.n_act):
            # Proportional term
            p_term = self.Kp[i] * err[i]

            u_unclamped = p_term + self.integ[i]
            u_sat = np.clip(u_unclamped, self.u_min, self.u_max)

            # Back-calculation anti-windup: integrator is nudged toward a
            # value that would have produced the saturated output, at rate
            # governed by Tt.
            sat_error = u_sat - u_unclamped
            integ_update = self.Ki[i] * err[i] * self.dt + (self.dt / self.Tt[i]) * sat_error
            self.integ[i] = self.integ[i] + integ_update

            u_cmd[i] = u_sat

        # Output rate limiting (avoid chattering, respect duty limit)
        du = u_cmd - self.u_prev
        du = np.clip(du, -self.max_du, self.max_du)
        u_out = self.u_prev + du

        # Final hard clamp
        u_out = np.clip(u_out, self.u_min, self.u_max)

        self.u_prev = u_out.copy()

        return u_out