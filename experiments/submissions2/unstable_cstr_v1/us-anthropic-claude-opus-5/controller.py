import numpy as np


class Controller:
    """
    Observer-based (Kalman) feedback-linearizing controller for an
    open-loop-unstable exothermic CSTR whose only handle is the jacket
    coolant supply temperature.

    Structure
    ---------
      * 3-state estimator  x = [T, Ca, d]   (d = lumped heat-balance bias,
        K/s) driven by the NONLINEAR nominal model and corrected with a
        constant steady-state Kalman gain computed from the linearisation.
        The estimator input is the *measured* delivered jacket temperature
        (TI-102) when available, so slew limits, stiction and saturation are
        seen by the observer -> no integrator windup, ever.
      * control law solves the energy balance for the jacket temperature that
        gives  dT/dt = sp_dot - lambda*(T - sp).  The bias state d supplies
        the integral action (offset free).
      * temperature-dependent command envelope: both actuator extremes are
        unsafe on this unit (270 K parks the reactor near 296 K, 340 K ignites
        it), so extreme commands are only permitted when the reactor
        temperature actually calls for them.
      * small dead-band + slew clamp keeps actuator duty well below the limit.
    """

    # ------------------------------------------------------------------
    def __init__(self, brief):
        self.brief = brief
        self.dt_nom = float(getattr(brief, "sample_time", 3.0))

        self.umin = 270.0
        self.umax = 340.0
        self.u_start = 300.0

        # --- nominal model, SI-ish, per SECOND ---
        self.tau = 100.0 / 100.0 * 60.0                 # V/q  [s]
        self.a = 50000.0 / (100.0 * 1000.0 * 0.239) / 60.0   # UA/(V rho Cp) [1/s]
        self.beta = 50000.0 / 239.0                     # -dH/(rho Cp) [K/(mol/L)]
        self.k0 = 7.2e10 / 60.0                         # [1/s]
        self.EoR = 8750.0
        self.Tf = 350.0
        self.Caf = 1.0

        # --- tuning ---
        self.lam = 0.050          # closed-loop temperature bandwidth [1/s]
        self.nd = 1               # assumed measurement delay [samples]
        self.R_meas = 0.25        # measurement variance [K^2]
        self.Qd = np.diag([4.0e-3, 1.0e-7, 1.5e-5])
        self.du_max = 4.0         # per-step slew clamp [K]  (skid: 1.5 K/s)
        self.deadband = 0.15      # K, matches valve stiction

        self.Kf = self._design_gain(self.dt_nom)
        self.reset()

    # ------------------------------------------------------------------
    # design helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _expm(M):
        M = np.asarray(M, dtype=float)
        nrm = float(np.max(np.sum(np.abs(M), axis=1))) if M.size else 0.0
        s = 0
        while nrm > 0.5 and s < 40:
            nrm *= 0.5
            s += 1
        A = M / (2.0 ** s)
        n = M.shape[0]
        E = np.eye(n)
        term = np.eye(n)
        for i in range(1, 16):
            term = term.dot(A) / float(i)
            E = E + term
        for _ in range(s):
            E = E.dot(E)
        return E

    def _kfun(self, T):
        T = max(min(float(T), 520.0), 250.0)
        return self.k0 * np.exp(-self.EoR / T)

    def _design_gain(self, dt):
        T0, Ca0 = 350.0, 0.5
        k = self._kfun(T0)
        dk = k * self.EoR / (T0 * T0)
        A = np.zeros((3, 3))
        A[0, 0] = -1.0 / self.tau - self.a + self.beta * Ca0 * dk
        A[0, 1] = self.beta * k
        A[0, 2] = 1.0
        A[1, 0] = -Ca0 * dk
        A[1, 1] = -1.0 / self.tau - k
        B = np.array([[self.a], [0.0], [0.0]])

        M = np.zeros((4, 4))
        M[:3, :3] = A * dt
        M[:3, 3:4] = B * dt
        E = self._expm(M)
        Ad = E[:3, :3]

        # steady-state Kalman gain (predict / update form)
        P = np.diag([1.0, 1.0e-2, 1.0e-4])
        Kf = np.array([0.3, 0.0, 0.0])
        for _ in range(4000):
            P = Ad.dot(P).dot(Ad.T) + self.Qd
            den = P[0, 0] + self.R_meas
            if den <= 1e-12 or not np.isfinite(den):
                break
            Knew = P[:, 0] / den
            P = P - np.outer(Knew, P[0, :])
            P = 0.5 * (P + P.T)
            if np.max(np.abs(Knew - Kf)) < 1e-12:
                Kf = Knew
                break
            Kf = Knew
        if not np.all(np.isfinite(Kf)):
            Kf = np.array([0.35, 0.0, 0.002])
        # keep the gain sane
        Kf[0] = float(np.clip(Kf[0], 0.05, 0.75))
        Kf[1] = float(np.clip(Kf[1], -0.05, 0.05))
        Kf[2] = float(np.clip(Kf[2], -0.02, 0.02))
        return Kf

    # ------------------------------------------------------------------
    def reset(self):
        self.T_hat = 350.0
        self.Ca_hat = 0.5
        self.d_hat = 0.0
        self.Tpred_hist = []
        self.u_prev = self.u_start
        self.u_eff = self.u_start
        self.T_meas_last = 350.0
        self.have_meas = False
        self.sp_f = None
        self.sp_dot = 0.0
        self.t_last = None
        self.bad_count = 0
        self.initialised = False

    # ------------------------------------------------------------------
    def _predict(self, dt):
        n = 6
        h = dt / float(n)
        T = self.T_hat
        Ca = self.Ca_hat
        d = self.d_hat
        u = self.u_eff
        for _ in range(n):
            k = self._kfun(T)
            dT = (self.Tf - T) / self.tau + self.beta * k * Ca - self.a * (T - u) + d
            dCa = (self.Caf - Ca) / self.tau - k * Ca
            T = T + h * dT
            Ca = Ca + h * dCa
            T = min(max(T, 250.0), 520.0)
            Ca = min(max(Ca, 0.0), 1.2)
        if not (np.isfinite(T) and np.isfinite(Ca)):
            T, Ca = self.T_meas_last, 0.5
        self.T_hat = T
        self.Ca_hat = Ca
        self.d_hat = d

    # ------------------------------------------------------------------
    def step(self, t, y, r, quality):
        # ---------------- housekeeping ----------------
        dt = self.dt_nom
        if self.t_last is not None:
            d_ = float(t) - self.t_last
            if 0.5 < d_ < 20.0:
                dt = d_
        self.t_last = float(t)

        y = np.asarray(y, dtype=float).ravel()
        if quality is None:
            q = np.ones(max(y.size, 2), dtype=bool)
        else:
            q = np.asarray(quality).ravel().astype(bool)
            if q.size < y.size:
                q = np.concatenate([q, np.ones(y.size - q.size, dtype=bool)])

        # reactor temperature
        T_ok = False
        if y.size > 0 and q[0] and np.isfinite(y[0]) and 250.0 < y[0] < 520.0:
            T_meas = float(y[0])
            T_ok = True
            self.T_meas_last = T_meas
            self.have_meas = True
            self.bad_count = 0
        else:
            T_meas = self.T_meas_last
            self.bad_count += 1

        # delivered jacket temperature (observer input)
        if y.size > 1 and q[1] and np.isfinite(y[1]) and 250.0 < y[1] < 360.0:
            self.u_eff = float(np.clip(y[1], self.umin, self.umax))
        else:
            self.u_eff = float(np.clip(self.u_prev, self.umin, self.umax))

        # ---------------- initialisation ----------------
        if not self.initialised:
            self.initialised = True
            T0 = T_meas if T_ok else 350.0
            self.T_hat = T0
            k0 = self._kfun(T0)
            self.Ca_hat = (self.Caf / self.tau) / (1.0 / self.tau + k0)
            self.d_hat = 0.0
            self.u_prev = self.u_start
            self.u_eff = self.u_start
            self.Tpred_hist = [T0] * (self.nd + 2)
            self.sp_f = T0
            return np.array([self.u_start], dtype=float)

        # ---------------- setpoint shaping ----------------
        sp_raw = 350.0
        if r is not None:
            rr = np.asarray(r, dtype=float).ravel()
            if rr.size > 0 and np.isfinite(rr[0]):
                sp_raw = float(np.clip(rr[0], 310.0, 400.0))
        if self.sp_f is None:
            self.sp_f = sp_raw
        rate_max = 0.30 * dt
        target = self.sp_f + float(np.clip(sp_raw - self.sp_f, -rate_max, rate_max))
        alpha = dt / (4.0 + dt)
        sp_new = self.sp_f + alpha * (target - self.sp_f)
        self.sp_dot = (sp_new - self.sp_f) / dt
        self.sp_f = sp_new
        sp = self.sp_f

        # ---------------- estimator: predict ----------------
        self._predict(dt)
        self.Tpred_hist.append(self.T_hat)
        if len(self.Tpred_hist) > 40:
            self.Tpred_hist.pop(0)

        # ---------------- estimator: correct ----------------
        if T_ok:
            idx = len(self.Tpred_hist) - 1 - self.nd
            if idx < 0:
                idx = 0
            inn = T_meas - self.Tpred_hist[idx]
            inn = float(np.clip(inn, -8.0, 8.0))
            self.T_hat += self.Kf[0] * inn
            self.Ca_hat += self.Kf[1] * inn
            self.d_hat += self.Kf[2] * inn
            self.T_hat = float(np.clip(self.T_hat, 250.0, 520.0))
            self.Ca_hat = float(np.clip(self.Ca_hat, 0.0, 1.2))
            self.d_hat = float(np.clip(self.d_hat, -0.5, 0.5))
            self.Tpred_hist[-1] = self.T_hat

        # ---------------- control law ----------------
        Th = self.T_hat
        k = self._kfun(Th)
        g = (self.Tf - Th) / self.tau + self.beta * k * self.Ca_hat + self.d_hat
        des = self.sp_dot - self.lam * (Th - sp)
        u_ideal = Th + (des - g) / self.a
        if not np.isfinite(u_ideal):
            u_ideal = self.u_prev

        # ---------------- safety envelope on the command ----------------
        T_use = max(Th, T_meas)          # be pessimistic on the hot side
        T_cold = min(Th, T_meas)         # and on the cold side
        u_hi = float(np.clip(340.0 - 2.5 * (T_use - 352.0), 285.0, 340.0))
        u_lo = float(np.clip(270.0 + 2.5 * (338.0 - T_cold), 270.0, 335.0))
        if u_lo > u_hi:
            mid = 0.5 * (u_lo + u_hi)
            u_lo = u_hi = mid
        u = float(np.clip(u_ideal, max(self.umin, u_lo), min(self.umax, u_hi)))

        # hard guards
        if T_use > 385.0:
            u = self.umin
        if T_cold < 318.0:
            u = self.umax
        if T_use > 470.0 - 60.0:     # far excursion: dump cooling regardless
            u = self.umin

        # ---------------- slew clamp / dead-band / duty ----------------
        u = float(np.clip(u, self.u_prev - self.du_max, self.u_prev + self.du_max))
        if abs(u - self.u_prev) < self.deadband and 315.0 < T_use < 372.0:
            u = self.u_prev
        u = float(np.clip(u, self.umin, self.umax))
        if not np.isfinite(u):
            u = self.u_prev
        self.u_prev = u
        return np.array([u], dtype=float)