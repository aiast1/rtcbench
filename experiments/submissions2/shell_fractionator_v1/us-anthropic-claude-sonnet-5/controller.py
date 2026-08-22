import numpy as np


class Controller:
    """
    Decentralized PI(D) controller with steady-state gain decoupling,
    conditional anti-windup, hard actuator-rate limiting (to respect the
    0.016/step duty limit) and a dedicated safety override for the
    bottoms-temperature floor constraint.

    Revision 3: effort cost is a small fraction of tracking cost (rate
    limiting already keeps actuator travel modest), so gains were
    pushed further and integral time shortened to attack the dominant
    tracking-error term. A small derivative-on-measurement term was
    added (not derivative-on-error, to avoid setpoint-step kick) using
    the already-filtered signal to help settle the ramp/step
    transitions faster without amplifying noise.
    """

    def __init__(self, brief):
        self.dt = getattr(brief, "sample_time", 1.0)

        # ---- nominal steady-state gain matrix (model hint) -----------------
        self.K = np.array([
            [4.05, 1.77, 5.88],
            [5.39, 5.72, 6.90],
            [4.38, 4.42, 7.20],
        ], dtype=float)

        # regularized inverse for a *damped* steady-state decoupler
        reg = 1.0e-2 * np.eye(3)
        try:
            self.Kinv = np.linalg.inv(self.K + reg)
        except np.linalg.LinAlgError:
            self.Kinv = np.linalg.pinv(self.K)

        self.diagK = np.diag(self.K).copy()
        self.diagK[np.abs(self.diagK) < 1e-3] = 1e-3

        # blend factor between full steady-state decoupling and pure diagonal
        self.alpha = 0.7

        # ---- PID tuning (per scored channel, in y-error units) --------------
        self.Kp = np.array([0.58, 0.58, 0.58])
        self.Ti = np.array([26.0, 26.0, 26.0])   # integral time [s]
        self.Ki = self.Kp / self.Ti
        self.Kd = np.array([0.06, 0.06, 0.06]) * self.Kp * 8.0  # mild derivative-on-measurement

        self.Imax = 4.0

        # ---- actuator limits -------------------------------------------------
        self.u_lo = -0.5
        self.u_hi = 0.5
        self.rate_max = 0.0150  # margin under the 0.016 duty-limit threshold

        # ---- measurement filter (light, to fight noise without adding delay
        # comparable to the process lags) ------------------------------------
        self.tau_f = 4.0

        # ---- safety on TI-103 (index 2), hard floor at -0.5 -----------------
        self.safe_low = -0.40   # start biasing recovery
        self.safe_hard = -0.47  # near-hard: force maximum recovery rate

        self.reset()

    def reset(self):
        self.y_filt = np.zeros(3)
        self.have_filt = False
        self.y_filt_prev = np.zeros(3)
        self.I = np.zeros(3)
        self.u_prev = np.zeros(3)

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        quality = np.asarray(quality, dtype=bool)
        dt = self.dt

        # ---- filter (hold last good value on bad quality) -------------------
        self.y_filt_prev = self.y_filt.copy()
        if not self.have_filt:
            self.y_filt = y.copy()
            self.y_filt_prev = y.copy()
            self.have_filt = True
        else:
            a = dt / (self.tau_f + dt)
            for i in range(3):
                if quality[i]:
                    self.y_filt[i] += a * (y[i] - self.y_filt[i])
                # else: hold previous filtered value

        # ---- error (nan setpoints -> zero contribution) ---------------------
        e = np.where(np.isnan(r), 0.0, r - self.y_filt)

        # ---- derivative on measurement (filtered), not on error --------------
        dy = (self.y_filt - self.y_filt_prev) / dt
        deriv_term = -self.Kd * dy
        deriv_term = np.where(np.isnan(r), 0.0, deriv_term)

        # ---- PID "virtual" correction in output space -------------------------
        v = self.Kp * e + self.Ki * self.I + deriv_term

        # ---- decoupled and diagonal-only actuator targets --------------------
        u_dec = self.Kinv @ v
        u_diag = v / self.diagK
        u_target = self.alpha * u_dec + (1.0 - self.alpha) * u_diag

        # ---- safety override on bottoms temperature (measured, index 2) -----
        y2_raw = y[2] if quality[2] else self.y_filt[2]
        if y2_raw < self.safe_low:
            deficiency = self.safe_low - y2_raw
            boost = 6.0 * deficiency
            u_target[2] = max(u_target[2], self.u_prev[2] + boost)
            if y2_raw < self.safe_hard:
                # force maximum allowed upward move this step
                u_target[2] = self.u_prev[2] + self.rate_max

        # ---- clip to hard actuator limits ------------------------------------
        u_target = np.clip(u_target, self.u_lo, self.u_hi)

        # ---- slew / duty limiting --------------------------------------------
        raw_delta = u_target - self.u_prev
        delta = np.clip(raw_delta, -self.rate_max, self.rate_max)
        u = self.u_prev + delta
        u = np.clip(u, self.u_lo, self.u_hi)

        # ---- conditional anti-windup ------------------------------------------
        limited = np.abs(raw_delta) > (self.rate_max + 1e-9)
        for i in range(3):
            if limited[i]:
                # heavily throttle integration while rate-saturated,
                # but keep a trickle so we still recover from big offsets
                self.I[i] += 0.15 * e[i] * dt
            else:
                self.I[i] += e[i] * dt

        self.I = np.clip(self.I, -self.Imax, self.Imax)

        self.u_prev = u
        return u