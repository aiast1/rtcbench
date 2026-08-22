import numpy as np


class Controller:
    """
    Robust 2x2 PI(+light filtered D) with nominal steady-state feedforward for
    the quadruple-tank process (minimum-phase pairing: pump1->tank1, pump2->tank2).

    Round-3 changes (effort was ~5% of cost, tracking dominated; duty far under
    the limit, so trade a little effort for speed):
      - Raised PI gains ~55% (lambda ~ 14 s instead of ~22 s).
      - Added a mild, heavily filtered derivative on the measurement to keep
        the higher-gain loops well damped under parameter draws and delay.
      - Higher setpoint weight (beta 0.85), faster setpoint filter (2.5 s),
        lighter measurement filter (alpha 0.7), larger rate limit (0.9 V/step).
      - Kept: inverse-static-gain feedforward, back-calculation anti-windup,
        micro-move deadband (duty protection), pessimistic observer for the
        unmeasured upper tanks with dynamic caps, bumpless start at [3,3].
    """

    G = 981.0  # cm/s^2

    def __init__(self, brief):
        self.dt = float(getattr(brief, "sample_time", 2.0))
        self.u0 = np.array([3.0, 3.0], dtype=float)

        # Nominal model (hint): used for feedforward and the safety observer.
        self.a1, self.a2 = 0.071, 0.057
        self.A3, self.A4 = 28.0, 32.0
        self.a3, self.a4 = 0.071, 0.057
        self.k1, self.k2 = 3.33, 3.35
        self.g1, self.g2 = 0.7, 0.6

        # Static gain matrix (flow balance):  M @ u = a_i * sqrt(2 g h_i)
        self.M = np.array(
            [[self.g1 * self.k1, (1.0 - self.g2) * self.k2],
             [(1.0 - self.g1) * self.k1, self.g2 * self.k2]], dtype=float)
        self.Minv = np.linalg.inv(self.M)

        # PID gains (lambda ~ 14 s on nominal linearization at ~12.5 cm).
        self.Kp = np.array([0.85, 1.05], dtype=float)
        self.Ki = np.array([0.0145, 0.0125], dtype=float)
        self.Td = 4.0                                # s, mild damping term
        self.Kd = self.Kp * self.Td
        self.tau_d = 6.0                             # s, derivative filter
        self.beta = 0.85                             # proportional SP weight
        self.Tt = 10.0                               # anti-windup time const

        # Signal conditioning
        self.meas_alpha = 0.7    # measurement low-pass per step
        self.sp_tau = 2.5        # setpoint filter time constant, s

        # Actuator shaping
        self.du_max = 0.9        # V per 2 s step
        self.move_db = 0.008     # V, ignore micro-moves (no dither)
        self.u_lo = 0.0
        self.u_hi_base = 6.5     # static cap protects unmeasured tanks

        self.reset()

    # ------------------------------------------------------------------ #

    def reset(self):
        self.initialized = False
        self.yf = np.array([12.26, 12.78], dtype=float)
        self.yf_prev = self.yf.copy()
        self.y_last_good = self.yf.copy()
        self.rf = self.yf.copy()
        self.I = np.zeros(2, dtype=float)
        self.d_f = np.zeros(2, dtype=float)          # filtered dy/dt
        self.u_prev = self.u0.copy()
        # Pessimistic observer states for unmeasured tanks 3 and 4.
        self.h3 = self._ss_upper((1.0 - self.g2) * self.k2, self.a3, self.u0[1])
        self.h4 = self._ss_upper((1.0 - self.g1) * self.k1, self.a4, self.u0[0])

    def _ss_upper(self, kin, aout, u):
        qin = 1.15 * kin * u
        qout_coef = 0.90 * aout
        h = (qin / qout_coef) ** 2 / (2.0 * self.G)
        return float(np.clip(h, 0.0, 25.0))

    def _observer_update(self, u):
        dt = self.dt
        qin3 = 1.15 * (1.0 - self.g2) * self.k2 * u[1]
        qout3 = 0.90 * self.a3 * np.sqrt(2.0 * self.G * max(self.h3, 0.0))
        self.h3 = float(np.clip(self.h3 + dt * (qin3 - qout3) / self.A3, 0.0, 25.0))

        qin4 = 1.15 * (1.0 - self.g1) * self.k1 * u[0]
        qout4 = 0.90 * self.a4 * np.sqrt(2.0 * self.G * max(self.h4, 0.0))
        self.h4 = float(np.clip(self.h4 + dt * (qin4 - qout4) / self.A4, 0.0, 25.0))

    def _dynamic_caps(self):
        cap2 = self.u_hi_base - np.clip((self.h3 - 13.0) * 0.8, 0.0, 4.0)
        cap1 = self.u_hi_base - np.clip((self.h4 - 13.0) * 0.8, 0.0, 4.0)
        return np.array([max(cap1, 2.0), max(cap2, 2.0)], dtype=float)

    def _u_ff(self, r):
        b = np.array([self.a1 * np.sqrt(2.0 * self.G * max(r[0], 0.1)),
                      self.a2 * np.sqrt(2.0 * self.G * max(r[1], 0.1))])
        u = self.Minv @ b
        return np.clip(u, 0.5, self.u_hi_base)

    # ------------------------------------------------------------------ #

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        q = np.asarray(quality, dtype=bool) if quality is not None else np.ones(2, bool)

        # --- validate / hold measurements ---
        ym = self.y_last_good.copy()
        for i in range(2):
            if i < y.size and q[min(i, q.size - 1)] and np.isfinite(y[i]):
                ym[i] = np.clip(y[i], 0.0, 20.0)
        self.y_last_good = ym.copy()

        # --- setpoints (hold last if nan) ---
        r_use = self.rf.copy()
        for i in range(2):
            if i < r.size and np.isfinite(r[i]):
                r_use[i] = np.clip(r[i], 1.0, 15.5)

        # --- first-call bumpless initialization ---
        if not self.initialized:
            self.yf = ym.copy()
            self.yf_prev = ym.copy()
            self.rf = r_use.copy()
            uff0 = self._u_ff(self.rf)
            e0 = self.beta * self.rf - self.yf
            self.I = self.u0 - uff0 - self.Kp * e0
            self.u_prev = self.u0.copy()
            self.initialized = True
            return self.u0.copy()

        # --- filters ---
        self.yf = self.yf + self.meas_alpha * (ym - self.yf)
        a_sp = self.dt / (self.dt + self.sp_tau)
        self.rf = self.rf + a_sp * (r_use - self.rf)

        # --- filtered derivative of measurement (damping only, no SP kick) ---
        dy = (self.yf - self.yf_prev) / self.dt
        self.yf_prev = self.yf.copy()
        a_d = self.dt / (self.dt + self.tau_d)
        self.d_f = self.d_f + a_d * (dy - self.d_f)

        # --- feedforward + PID with setpoint weighting ---
        uff = self._u_ff(self.rf)
        e = self.rf - self.yf
        u_raw = (uff + self.Kp * (self.beta * self.rf - self.yf)
                 + self.I - self.Kd * self.d_f)

        # --- saturation limits (static + upper-tank protection) ---
        u_hi = self._dynamic_caps()
        u_sat = np.clip(u_raw, self.u_lo, u_hi)

        # --- rate limit relative to last commanded output ---
        u_cmd = np.clip(u_sat, self.u_prev - self.du_max, self.u_prev + self.du_max)

        # --- tiny-move deadband (protect actuator duty budget) ---
        for i in range(2):
            if abs(u_cmd[i] - self.u_prev[i]) < self.move_db:
                u_cmd[i] = self.u_prev[i]

        u_cmd = np.clip(u_cmd, 0.0, 10.0)

        # --- integrator update with back-calculation anti-windup ---
        self.I += self.dt * (self.Ki * e + (u_cmd - u_raw) / self.Tt)
        self.I = np.clip(self.I, -4.0, 8.0)

        # --- update pessimistic upper-tank observer ---
        self._observer_update(u_cmd)

        self.u_prev = u_cmd.copy()
        return u_cmd.copy()