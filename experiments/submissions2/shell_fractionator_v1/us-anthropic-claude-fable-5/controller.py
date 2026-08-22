import numpy as np


class Controller:
    """
    Static-decoupling multiloop 2-DOF PI controller for the 3x3 fractionator.

    Round 3 changes (effort remains ~100x under budget, tracking dominates):
      * Steady-state reference feedforward in the decoupled space
        (v ~= r at steady state since the decoupled plant has unit gain),
        so the loop does not wait on the integrator to reach a new target.
      * Setpoint weighting (beta) on the proportional term to avoid overshoot
        from the combined ff + P kick.
      * Slightly faster PI (Kp up, Ti down) and lighter filtering.
    Robustness features kept: regularized gain-matrix decoupling,
    back-calculation anti-windup vs. the actually applied u, rate limiting,
    and an early safety override for the TI-103 floor at -0.5.
    """

    def __init__(self, brief):
        self.dt = float(getattr(brief, "sample_time", 1.0) or 1.0)

        # Nominal steady-state gains (model hint)
        self.K = np.array([
            [4.05, 1.77, 5.88],
            [5.39, 5.72, 6.90],
            [4.38, 4.42, 7.20],
        ])
        # Regularized inverse for robustness to plant/model mismatch
        U, s, Vt = np.linalg.svd(self.K)
        s_inv = s / (s * s + (0.05 * s[0]) ** 2)
        self.Kinv = (Vt.T * s_inv) @ U.T

        # 2-DOF PI tuning in decoupled (unit-gain) coordinates.
        self.Kp = np.array([1.10, 1.20, 1.30])
        self.Ti = np.array([40.0, 35.0, 28.0])
        self.Tt = np.array([13.0, 12.0, 9.0])   # anti-windup tracking time
        self.beta = 0.4                          # setpoint weight on P term
        self.kff = 0.80                          # steady-state feedforward

        # Actuator handling (duty headroom is large; allow real moves)
        self.umin, self.umax = -0.5, 0.5
        self.du_max = 0.030
        self.deadband = 0.0003

        # Filters
        self.tau_y = 2.0     # measurement filter (light)
        self.tau_r = 4.0     # reference shaping (softens step kick)

        # Safety: y[2] hard floor at -0.5 -> intervene early
        self.y2_soft = -0.25
        self.y2_hard = -0.35

        self.reset()

    def reset(self):
        self.yf = None                 # filtered measurements
        self.rf = None                 # shaped references
        self.I = np.zeros(3)           # integrator states (decoupled space)
        self.u = np.zeros(3)           # last applied actuator vector
        self.r_last = np.zeros(3)      # last good setpoints

    def step(self, t, y, r, quality):
        dt = self.dt

        # --- sanitize inputs -------------------------------------------------
        y = np.asarray(y, dtype=float)
        r_in = np.asarray(r, dtype=float).copy()
        q = np.asarray(quality).astype(bool) if quality is not None \
            else np.ones(3, dtype=bool)

        for i in range(3):
            if not np.isfinite(r_in[i]):
                r_in[i] = self.r_last[i]
        self.r_last = r_in.copy()

        # --- measurement filtering (hold on bad quality / nan) --------------
        if self.yf is None:
            self.yf = np.where(np.isfinite(y), y, 0.0).copy()
        a_y = dt / (self.tau_y + dt)
        for i in range(3):
            if q[i] and np.isfinite(y[i]):
                self.yf[i] += a_y * (y[i] - self.yf[i])

        # --- reference shaping (bumpless, softens step kicks) ----------------
        if self.rf is None:
            self.rf = r_in.copy()
        a_r = dt / (self.tau_r + dt)
        self.rf += a_r * (r_in - self.rf)

        # --- errors + safety override on TI-103 -------------------------------
        e = self.rf - self.yf                    # for the integrator
        ep = self.beta * self.rf - self.yf       # setpoint-weighted P error

        y2 = self.yf[2]
        if q[2] and np.isfinite(y[2]):
            y2 = min(y2, y[2])  # respect a raw low reading immediately
        if y2 < self.y2_soft:
            e_safe = (self.y2_soft + 0.05) - y2
            e[2] = max(e[2], e_safe)
            ep[2] = max(ep[2], e_safe)
            if y2 < self.y2_hard:
                e[2] = max(e[2], 2.0 * ((self.y2_soft + 0.05) - y2))
                ep[2] = e[2]

        # --- 2-DOF PI + feedforward in decoupled coordinates ------------------
        v = self.kff * self.rf + self.Kp * ep + self.I

        u_des = self.Kinv @ v
        u_des = np.clip(u_des, self.umin, self.umax)

        # rate limit + deadband relative to last applied u
        du = np.clip(u_des - self.u, -self.du_max, self.du_max)
        du[np.abs(du) < self.deadband] = 0.0
        u_new = np.clip(self.u + du, self.umin, self.umax)

        # --- integrator update with back-calculation anti-windup -------------
        v_applied = self.K @ u_new
        Ki = self.Kp / self.Ti
        self.I += Ki * e * dt + (v_applied - v) * (dt / self.Tt)
        self.I = np.clip(self.I, -4.0, 4.0)

        self.u = u_new
        return self.u.copy()