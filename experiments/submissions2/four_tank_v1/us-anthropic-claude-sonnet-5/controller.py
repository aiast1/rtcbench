import numpy as np


class Controller:
    """
    Decentralized PI controller for the four-tank process (measured: h1, h2).
    Designed for robustness across a family of plants (mismatch tier):
      - filtered measurements (reduce noise/quality dropout impact)
      - rate-limited setpoint tracking (soft ramps instead of hard steps)
      - PI with conditional (clamping) anti-windup
      - actuator slew-rate limiting to respect duty-cycle / chatter limits
      - no derivative term (noisy, delayed measurement -> unsafe D action)

    Revision notes (round 3): round-2 results showed tracking error still
    dominating cost while actuator effort remained far under the duty
    budget (effort ~0.0004 vs track ~0.017). Pushed gains further and
    reduced filtering/settling lag to close the residual tracking gap,
    while keeping slew and anti-windup protections so the unmeasured
    tanks (h3, h4) and duty-cycle limit stay safe.
    """

    def __init__(self, brief):
        self.dt = float(getattr(brief, "sample_time", 2.0))

        # actuator limits
        self.u_min = 0.0
        self.u_max = 10.0

        # nominal bumpless start
        self.u0 = np.array([3.0, 3.0])

        # PI gains (further increased vs round-2; effort budget has ample margin)
        self.Kp = np.array([0.62, 0.62])
        self.Ki = np.array([0.0065, 0.0065])   # Ti ~ 95 s

        # measurement low-pass filter time constant (s) -- lighter filtering
        self.tau_meas = 3.0
        # setpoint smoothing time constant (s) -- faster tracking of steps
        self.tau_sp = 5.0
        # actuator slew limit per control step (V) -- loosened further, still safe
        self.max_slew = 0.55

        self.reset()

    def reset(self):
        self.u_prev = self.u0.copy()
        # integrator carries the bumpless bias directly
        self.integ = self.u0.copy()
        self.yf = None
        self.rf = None
        self._initialized = False

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        q = np.asarray(quality, dtype=bool)

        n = len(y)

        if not self._initialized:
            self.yf = y.copy()
            # initialise setpoint filter to first real setpoint (bumpless)
            r0 = np.where(np.isnan(r), y, r)
            self.rf = r0.copy()
            self._initialized = True

        alpha_y = self.dt / (self.tau_meas + self.dt)
        alpha_r = self.dt / (self.tau_sp + self.dt)

        # --- update filtered measurement, honoring quality flags ---
        for i in range(n):
            if q[i]:
                self.yf[i] = self.yf[i] + alpha_y * (y[i] - self.yf[i])
            # else: hold previous filtered value (stale sample ignored)

        # --- update smoothed setpoint (only for scored / non-nan channels) ---
        for i in range(n):
            if not np.isnan(r[i]):
                self.rf[i] = self.rf[i] + alpha_r * (r[i] - self.rf[i])

        e = self.rf - self.yf

        # candidate unclamped PI output
        u_unclamped = self.integ + self.Kp * e
        u_clamped = np.clip(u_unclamped, self.u_min, self.u_max)

        # --- conditional (clamping) anti-windup: only integrate if it would
        # not push us further into saturation ---
        for i in range(len(u_unclamped)):
            saturated_high = (u_unclamped[i] > self.u_max) and (e[i] > 0.0)
            saturated_low = (u_unclamped[i] < self.u_min) and (e[i] < 0.0)
            if not (saturated_high or saturated_low):
                self.integ[i] += self.Ki[i] * e[i] * self.dt
                # keep integrator itself within actuator bounds for safety
                self.integ[i] = float(np.clip(self.integ[i], self.u_min, self.u_max))

        # --- actuator slew-rate limiting to protect duty cycle / avoid chatter ---
        delta = np.clip(u_clamped - self.u_prev, -self.max_slew, self.max_slew)
        u_out = self.u_prev + delta
        u_out = np.clip(u_out, self.u_min, self.u_max)

        self.u_prev = u_out
        return u_out