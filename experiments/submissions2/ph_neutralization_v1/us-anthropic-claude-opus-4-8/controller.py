import numpy as np


class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = float(brief.sample_time)
        self.u_min = 0.0
        self.u_max = 30.0
        self.u0 = 14.22

        self.pH_lo = 4.0
        self.pH_hi = 10.5
        self.pH_lo_soft = 4.6
        self.pH_hi_soft = 10.1

        self.h_lo = 5.0
        self.h_hi = 30.0
        self.h_lo_soft = 7.0
        self.h_hi_soft = 28.0

        self.reset()

    def reset(self):
        self.u = self.u0
        self.u_prev = self.u0
        self.integ = 0.0
        self.pH_filt = None
        self.pH_prev = None
        self.last_good_pH = None
        self.last_good_h = None
        self.r_filt = None

    def _gain_schedule(self, pH):
        d = abs(pH - 7.0)
        sched = 0.25 + 0.15 * min(d, 5.0)
        return sched

    def step(self, t, y, r, quality):
        pH_meas = float(y[0])
        h_meas = float(y[1])
        pH_ok = bool(quality[0])
        h_ok = bool(quality[1])

        if pH_ok and np.isfinite(pH_meas):
            self.last_good_pH = pH_meas
        pH_use = self.last_good_pH if self.last_good_pH is not None else pH_meas

        if h_ok and np.isfinite(h_meas):
            self.last_good_h = h_meas
        h_use = self.last_good_h if self.last_good_h is not None else h_meas

        alpha = 0.4
        if self.pH_filt is None:
            self.pH_filt = pH_use
        else:
            self.pH_filt = (1 - alpha) * self.pH_filt + alpha * pH_use
        pH = self.pH_filt

        sp = r[0]
        if not np.isfinite(sp):
            sp = pH
        # track setpoint more tightly (less smoothing) since effort has margin
        if self.r_filt is None:
            self.r_filt = sp
        else:
            beta = 0.45
            self.r_filt = (1 - beta) * self.r_filt + beta * sp
        sp_use = self.r_filt

        err = sp_use - pH

        gsched = self._gain_schedule(pH)

        # push gains a bit further; effort remains small and duty within limit
        Kp = gsched * 1.9
        Ki = gsched * 0.045

        if self.pH_prev is None:
            dpH = 0.0
        else:
            dpH = (pH - self.pH_prev) / self.dt
        self.pH_prev = pH
        Kd = gsched * 0.5

        integ_new = self.integ + Ki * err * self.dt

        u_unclamped = self.u0 + Kp * err + integ_new - Kd * dpH

        if pH > self.pH_hi_soft:
            excess = pH - self.pH_hi_soft
            u_unclamped -= 6.0 * excess
        if pH < self.pH_lo_soft:
            deficit = self.pH_lo_soft - pH
            u_unclamped += 6.0 * deficit

        if h_use > self.h_hi_soft:
            over = h_use - self.h_hi_soft
            u_unclamped -= 2.0 * over
        if h_use < self.h_lo_soft:
            under = self.h_lo_soft - h_use
            u_unclamped += 1.0 * under

        u_cmd = float(np.clip(u_unclamped, self.u_min, self.u_max))

        saturated_hi = (u_unclamped > self.u_max) and (err > 0)
        saturated_lo = (u_unclamped < self.u_min) and (err < 0)
        if not (saturated_hi or saturated_lo):
            self.integ = integ_new
        self.integ = float(np.clip(self.integ, -12.0, 12.0))

        max_step = 2.2
        du = u_cmd - self.u_prev
        if du > max_step:
            u_cmd = self.u_prev + max_step
        elif du < -max_step:
            u_cmd = self.u_prev - max_step

        u_cmd = float(np.clip(u_cmd, self.u_min, self.u_max))
        self.u_prev = u_cmd
        self.u = u_cmd

        return np.array([u_cmd], dtype=float)