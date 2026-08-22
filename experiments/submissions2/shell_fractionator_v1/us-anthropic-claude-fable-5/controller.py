import numpy as np


class Controller:
    """
    Static-decoupling PI controller for the 3x3 fractionator, with a
    layered predictive safety override on the bottoms reflux temperature.

    Design notes:
      - Nominal DC-gain matrix inverse gives steady-state decoupling; three
        independent PI loops run in the virtual (unit-DC-gain) space.
      - Dead times 15-28 s with tau 20-60 s -> IMC-like PI (lambda ~ L),
        no derivative (noisy, quantized, delayed measurements).
      - Measurement low-pass (frozen on bad quality) + setpoint low-pass
        keep actuator travel and duty low.
      - Anti-windup via back-calculation against the delivered actuation;
        this also lets the other two loops absorb safety-override moves.
      - Move-suppression deadband so stiction/noise cannot cause chatter
        (duty breach is fatal).
      - SAFETY (y3 >= -0.5, hard): act on a trend-based prediction of y3.
          1. Soft: below -0.18 (predicted), inflate channel-3 error so the
             PI pushes back early - this is also the tracking direction
             (setpoint is >= 0), so it costs little.
          2. Hard: below -0.30 (predicted), directly drive FCV-203 up
             (largest gain, ~zero dead time to TI-103) with the deadband
             bypassed. Hysteresis prevents mode chatter.
      - Channel 3 carries the strongest integral action so disturbance
        rejection there is fast, minimizing time spent in override.
      - Bumpless start: integrators zero, u starts at [0,0,0].
    """

    def __init__(self, brief):
        self.dt = float(getattr(brief, "sample_time", 1.0) or 1.0)
        K = np.array([[4.05, 1.77, 5.88],
                      [5.39, 5.72, 6.90],
                      [4.38, 4.42, 7.20]], dtype=float)
        self.K = K
        self.Kinv = np.linalg.inv(K)

        # PI gains in decoupled (unit-gain) coordinates, per second.
        self.Kp = np.array([0.65, 0.70, 0.70])
        self.Ki = np.array([0.016, 0.018, 0.022])
        self.kaw = 0.10          # anti-windup back-calculation gain (1/s)

        self.u_lo, self.u_hi = -0.5, 0.5
        self.du_max = 0.03       # per-step rate limit (units/step)
        self.du_min = 4e-4       # move suppression deadband (anti-chatter)

        self.tau_f = 3.0         # measurement filter time constant (s)
        self.tau_r = 5.0         # setpoint filter time constant (s)

        # Safety parameters for y3 (hard floor at -0.5).
        self.safe_soft = -0.18   # start pushing back here (predicted)
        self.soft_gain = 3.0
        self.safe_hard = -0.30   # direct actuator override here (predicted)
        self.safe_exit = -0.12   # hysteresis exit for hard mode
        self.pred_horizon = 15.0 # seconds of trend extrapolation

        self.reset()

    def reset(self):
        self.I = np.zeros(3)
        self.u = np.zeros(3)
        self.yf = None
        self.rf = None
        self.r_last = np.zeros(3)
        self.d2f = 0.0           # filtered derivative of yf[2]
        self.y2_prev = 0.0
        self.safe_mode = False

    def step(self, t, y, r, quality):
        dt = self.dt
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        if quality is None:
            q = np.ones(3, dtype=bool)
        else:
            q = np.asarray(quality, dtype=bool)

        rr = np.where(np.isfinite(r), r, self.r_last)
        self.r_last = rr.copy()

        if self.yf is None:
            self.yf = np.where(np.isfinite(y), y, 0.0)
            self.rf = rr.copy()
            self.y2_prev = self.yf[2]

        # Measurement filter; freeze on bad/stale/non-finite samples.
        a = dt / (self.tau_f + dt)
        for i in range(3):
            ok = (i < q.shape[0] and q[i]) and np.isfinite(y[i])
            if ok:
                self.yf[i] += a * (y[i] - self.yf[i])

        # Trend estimate and short-horizon prediction of y3.
        raw_d2 = (self.yf[2] - self.y2_prev) / dt
        self.y2_prev = self.yf[2]
        self.d2f += 0.25 * (raw_d2 - self.d2f)
        y2_pred = self.yf[2] + self.pred_horizon * min(self.d2f, 0.0)
        y2_low = min(self.yf[2], y2_pred)

        # Setpoint filter.
        b = dt / (self.tau_r + dt)
        self.rf += b * (rr - self.rf)

        e = self.rf - self.yf

        # --- Layer 1: soft safety push-back on channel 3 ---
        if y2_low < self.safe_soft:
            e[2] = max(e[2], self.soft_gain * (self.safe_soft - y2_low))

        # Hard-mode hysteresis.
        if y2_low < self.safe_hard:
            self.safe_mode = True
        elif self.yf[2] > self.safe_exit and y2_pred > self.safe_exit:
            self.safe_mode = False

        # PI in virtual-output space, then static decoupling.
        v = self.Kp * e + self.I
        u_raw = self.Kinv @ v

        u_clipped = np.clip(u_raw, self.u_lo, self.u_hi)
        du = np.clip(u_clipped - self.u, -self.du_max, self.du_max)
        du[np.abs(du) < self.du_min] = 0.0
        u_new = np.clip(self.u + du, self.u_lo, self.u_hi)

        # --- Layer 2: hard override -> drive FCV-203 up fast ---
        if self.safe_mode:
            kick = np.clip(1.5 * (self.safe_hard - y2_low), 0.0, 0.08)
            u2_min = self.u[2] + kick
            if y2_low < self.safe_hard:
                # never let FCV-203 drop while deep in the danger zone
                u2_min = max(u2_min, self.u[2])
            u_new[2] = np.clip(max(u_new[2], u2_min), self.u_lo, self.u_hi)

        # Anti-windup: integrate error, bleed toward delivered actuation.
        v_act = self.K @ u_new
        self.I += self.Ki * e * dt + self.kaw * (v_act - v) * dt
        self.I = np.clip(self.I, -3.0, 3.0)

        self.u = u_new
        return self.u.copy()