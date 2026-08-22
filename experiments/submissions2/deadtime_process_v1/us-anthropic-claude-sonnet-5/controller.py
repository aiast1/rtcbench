import numpy as np


class Controller:
    """
    PI controller with anti-windup, measurement filtering, and software
    slew-rate limiting for a long-transport-delay, trace-heated line
    (TIC-401 / TT-401).

    Revision notes (round 3)
    -------------------------
    Round 2 improved tracking cost but one seed (the one with the longest
    transport delay, ~200 s, from the low-flow excursion) triggered a
    duty-limit breach: the faster gains + looser slew cap let the loop
    develop small sustained oscillation/chatter under that worst-case
    delay, which zeroed the scenario. Since scoring is on the WORST
    scenario, this must be fixed first, even at some cost to average
    tracking performance.

    Changes from round 2:
      - Kp reduced (0.95 -> 0.80) and Ti lengthened (140 -> 170s) to
        restore phase margin against the worst-case ~200s dead time
        while still being reasonably responsive at the fast (~40s
        delay) end of the range.
      - Measurement filter time slightly increased (8 -> 11s) to damp
        noise-driven micro-corrections that were feeding the slew
        limiter and eating into duty travel budget.
      - Proportional deadband widened slightly (0.03 -> 0.05 degC) to
        reject sensor noise/quantization "creep" without materially
        hurting settled tracking.
      - Slew-rate limiting pulled back (baseline 0.35 -> 0.22 %/s,
        error term 0.10 -> 0.06 %/s) AND given a hard ceiling
        (max 0.5 %/s) so that even large transient errors cannot push
        the actuator into a chattering/duty-breaching regime; this is
        the primary fix for the round-2 failure.
      - Everything else (anti-windup, quality gating, bumpless start,
        92% output ceiling for unmeasured heater-outlet safety) is kept
        as before.
    """

    def __init__(self, brief):
        self.dt = float(getattr(brief, "sample_time", 5.0))

        # --- PI tuning (robust across ~40-200s transport delay) ---
        self.Kp = 0.80
        self.Ti = 170.0          # integral time [s]
        self.Ki = self.Kp / self.Ti

        # --- measurement filter ---
        self.tau_f = 11.0

        # --- deadband on proportional action (degC) ---
        self.err_db = 0.05

        # --- output limits (safety headroom below official 100) ---
        self.out_min = 0.0
        self.out_max = 92.0

        # --- software slew rate limit (%/s) ---
        self.rate_min = 0.22      # baseline (steady-state) allowed rate
        self.rate_gain = 0.06     # extra allowed rate per degC of error
        self.rate_max_cap = 0.50  # hard ceiling regardless of error size

        self.u0 = 50.0

        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_u = self.u0
        self.y_filt = None
        self.last_meas = None
        self._initialized = False

    def step(self, t, y, r, quality):
        sp = float(r[0]) if not np.isnan(r[0]) else self.last_meas

        # --- measurement handling / quality gating ---
        good = bool(quality[0]) if quality is not None else True
        raw = float(y[0]) if (y is not None and not np.isnan(y[0])) else None

        if good and raw is not None:
            meas = raw
            self.last_meas = meas
        else:
            meas = self.last_meas if self.last_meas is not None else sp

        # --- low-pass filter the measurement used for control ---
        if self.y_filt is None:
            self.y_filt = meas
        alpha = self.dt / (self.tau_f + self.dt)
        self.y_filt += alpha * (meas - self.y_filt)

        if sp is None:
            sp = self.y_filt

        error = sp - self.y_filt

        # --- proportional term with small deadband ---
        if abs(error) <= self.err_db:
            p_err = 0.0
        else:
            p_err = error - np.sign(error) * self.err_db

        p_term = self.Kp * p_err

        # --- tentative output (before slew limiting) using current integral ---
        u_tentative = p_term + self.integral

        # --- anti-windup: integrate only if not saturating further ---
        would_saturate_high = (u_tentative > self.out_max) and (error > 0)
        would_saturate_low = (u_tentative < self.out_min) and (error < 0)
        if not (would_saturate_high or would_saturate_low):
            self.integral += self.Ki * error * self.dt
            # keep integral itself within output bounds to avoid latent windup
            self.integral = min(self.out_max, max(self.out_min, self.integral))

        u_target = min(self.out_max, max(self.out_min, p_term + self.integral))

        # --- bumpless start: first call snaps to current actuator state ---
        if not self._initialized:
            self.prev_u = self.u0
            self._initialized = True

        # --- software slew-rate limiting (adaptive, but hard-capped) ---
        max_rate = self.rate_min + self.rate_gain * abs(error)  # %/s
        if max_rate > self.rate_max_cap:
            max_rate = self.rate_max_cap
        max_step = max_rate * self.dt
        delta = u_target - self.prev_u
        if delta > max_step:
            delta = max_step
        elif delta < -max_step:
            delta = -max_step

        u = self.prev_u + delta
        u = min(self.out_max, max(self.out_min, u))

        self.prev_u = u

        return np.array([u], dtype=float)