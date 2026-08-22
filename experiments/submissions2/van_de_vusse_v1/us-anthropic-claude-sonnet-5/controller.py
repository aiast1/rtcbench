import numpy as np


class Controller:
    """
    Van de Vusse CSTR controller.

    Two loops:
      - Reactor temperature T  -> jacket duty Q_K   (fast, monotonic, PI with
        incremental/anti-windup structure).
      - Product conc.       C_B -> feed dilution F/V.  This loop combines a
        *model-based feed-forward* (nominal steady-state van de Vusse
        inversion, evaluated on the low-F branch below the C_B peak) with a
        small PI trim on the residual error measured by the slow analyzer.
        The feed-forward gives fast, anticipatory moves on setpoint changes
        (steps/ramps) while the PI trim removes the steady-state offset
        caused by plant-model mismatch.  All actuator moves are slew-limited
        per step to respect duty limits and stiction, and the trim
        integrator only ever proposes bounded increments (structural
        anti-windup).

    A hard safety override on measured T protects both T and the
    unmeasured jacket temperature TK (Q_K is cooling-only, so limiting T
    also limits TK).
    """

    # ---- nominal model constants (from commissioning brief) ----
    K10 = 1.287e12
    K20 = 1.287e12
    K30 = 9.043e9
    E1 = -9758.3
    E2 = -9758.3
    E3 = -8560.0
    CA0 = 5.1

    def __init__(self, brief):
        self.dt = float(getattr(brief, "sample_time", 10.0))

        # hard actuator limits
        self.u_lo = np.array([3.0, -9000.0])
        self.u_hi = np.array([35.0, 0.0])

        # conservative operating band for F/V (stay on the ascending,
        # well-behaved branch, clear of the non-monotonic region)
        self.fv_lo = 5.0
        self.fv_hi = 16.5

        # deadbands (roughly at measurement noise/quantization level)
        self.db_cb = 0.004
        self.db_t = 0.05

        # trim PI gains for the C_B loop (small - feed-forward carries the
        # bulk of the response)
        self.Kp_cb = 0.6
        self.Ki_cb = 0.004
        self.trim_lim = 3.0  # limit trim contribution (1/h)

        # PI gains for the temperature loop
        self.Kp_t = 60.0
        self.Ki_t = 3.0

        # per-step slew limits (own duty-cycle protection)
        self.max_d_fv = 0.30
        self.max_d_qk = 200.0

        # grid for feed-forward inversion
        self._fv_grid = np.linspace(self.fv_lo, self.fv_hi, 241)

        self.reset()

    def reset(self):
        # bumpless start at the plant's known operating point
        self.u = np.array([14.19, -1113.5])

        self.e_t_prev = 0.0
        self.trim = 0.0
        self.trim_prev_err = 0.0

        self.cb_filt = None
        self.t_filt = None

    # ------------------------------------------------------------------
    def _cb_ss(self, F, T):
        """Nominal steady-state C_B as a function of F/V at temperature T."""
        Tk = T + 273.15
        k1 = self.K10 * np.exp(self.E1 / Tk)
        k2 = self.K20 * np.exp(self.E2 / Tk)
        k3 = self.K30 * np.exp(self.E3 / Tk)
        disc = (F + k1) ** 2 + 4.0 * k3 * F * self.CA0
        disc = np.maximum(disc, 0.0)
        CA = (-(F + k1) + np.sqrt(disc)) / (2.0 * k3)
        CA = np.maximum(CA, 0.0)
        CB = k1 * CA / (F + k2)
        return CB

    def _feedforward_fv(self, cb_target, T):
        """Invert the nominal steady-state map on the ascending branch."""
        cb_grid = self._cb_ss(self._fv_grid, T)
        idx = int(np.argmin(np.abs(cb_grid - cb_target)))
        return float(self._fv_grid[idx])

    # ------------------------------------------------------------------
    def step(self, t, y, r, quality):
        y_cb = float(y[0])
        y_t = float(y[1])
        r_cb = float(r[0]) if np.isfinite(r[0]) else np.nan
        r_t = float(r[1]) if np.isfinite(r[1]) else np.nan

        q_cb = bool(quality[0]) if len(quality) > 0 else True
        q_t = bool(quality[1]) if len(quality) > 1 else True

        # --- filtering with quality gating (hold last good value) ---
        if self.t_filt is None:
            self.t_filt = y_t
        elif q_t:
            self.t_filt = 0.5 * y_t + 0.5 * self.t_filt
        t_meas = self.t_filt

        if self.cb_filt is None:
            self.cb_filt = y_cb
        elif q_cb:
            self.cb_filt = 0.5 * y_cb + 0.5 * self.cb_filt
        cb_meas = self.cb_filt

        u_fv, u_qk = self.u[0], self.u[1]

        # =========================================================
        # Temperature loop: Q_K (fast, monotonic, direct PI control)
        # =========================================================
        if np.isfinite(r_t):
            e_t = r_t - t_meas
            if abs(e_t) < self.db_t:
                e_t_eff = 0.0
            else:
                e_t_eff = e_t

            d_qk = (self.Kp_t * (e_t_eff - self.e_t_prev)
                    + self.Ki_t * self.dt * e_t_eff)
            d_qk = float(np.clip(d_qk, -self.max_d_qk, self.max_d_qk))
            u_qk = self.u[1] + d_qk
            self.e_t_prev = e_t_eff
        u_qk = float(np.clip(u_qk, self.u_lo[1], self.u_hi[1]))

        # =========================================================
        # Concentration loop: feed-forward + small PI trim
        # =========================================================
        if np.isfinite(r_cb):
            # temperature used for the model inversion: prefer the T
            # setpoint (steady target) if available & finite, else the
            # filtered measurement.
            T_for_model = r_t if np.isfinite(r_t) else t_meas
            fv_ff = self._feedforward_fv(r_cb, T_for_model)

            e_cb = r_cb - cb_meas
            if abs(e_cb) < self.db_cb:
                e_cb_eff = 0.0
            else:
                e_cb_eff = e_cb

            d_trim = (self.Kp_cb * (e_cb_eff - self.trim_prev_err)
                      + self.Ki_cb * self.dt * e_cb_eff)
            self.trim = float(np.clip(self.trim + d_trim,
                                       -self.trim_lim, self.trim_lim))
            self.trim_prev_err = e_cb_eff

            fv_target = float(np.clip(fv_ff + self.trim,
                                       self.fv_lo, self.fv_hi))

            # move toward target with slew limiting (duty protection)
            d_fv = float(np.clip(fv_target - self.u[0],
                                  -self.max_d_fv, self.max_d_fv))
            u_fv = self.u[0] + d_fv

        u_fv = float(np.clip(u_fv, self.fv_lo, self.fv_hi))
        u_fv = float(np.clip(u_fv, self.u_lo[0], self.u_hi[0]))

        # =========================================================
        # Hard safety override on reactor temperature.
        # Q_K is cooling-only (<=0), so keeping T bounded also keeps
        # the unmeasured jacket temperature TK bounded.
        # =========================================================
        if q_t:
            if y_t > 145.0:
                u_qk = self.u_lo[1]  # full cooling
            elif y_t > 140.0:
                u_qk = min(u_qk, -6000.0)

        self.u = np.array([u_fv, u_qk])
        return self.u.copy()