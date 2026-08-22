import numpy as np


class Controller:
    """
    Model-based feedforward + bounded PI trim controller for the (mismatched)
    quadruple-tank unit.

    Key design choices, driven by the fact that the two UPPER tanks (h3, h4)
    are never measured but must never overflow:

      * A nominal steady-state feedforward is computed every step by
        inverting the algebraic quad-tank steady-state relations

            a1*sqrt(2g h1) = gamma1*k1*u1 + (1-gamma2)*k2*u2
            a2*sqrt(2g h2) = (1-gamma1)*k1*u1 + gamma2*k2*u2

        for the requested (h1,h2) setpoint.  This gets the actuators close
        to the right operating point immediately, without relying on slow,
        windup-prone integral action to climb there.

      * A small PI trim removes residual steady-state error caused by the
        actual (mismatched) plant differing from the nominal model.  The
        trim's integral is clamped and frozen under saturation (classic
        anti-windup).

      * The *total* command (feedforward + trim) is hard-capped well below
        the 0-10V actuator limits.  The caps are derived from the nominal
        model so that even sustained maximum command cannot push the
        unmeasured upper tank (h3 or h4) past 20 cm, with a safety margin
        to cover per-scenario parameter mismatch.  This is the primary
        defence against the hidden-state overflow that caused the previous
        safety violations.

      * Slew-rate limiting on the actuator command bounds actuator travel
        per step (protects the duty-cycle budget and avoids chattering),
        and also gives an automatically bumpless start from the actuators'
        initial operating point.

      * Stale/bad samples (per `quality`) are held rather than fed into the
        filter/controller, and a light EMA filter attenuates sensor noise
        ahead of the derivative-free PI action (no derivative term is used,
        given noise + delay make D action unsafe here).
    """

    # ---- nominal model (from commissioning brief) ----
    A1, A2, A3, A4 = 28.0, 32.0, 28.0, 32.0
    a1, a2, a3, a4 = 0.071, 0.057, 0.071, 0.057
    k1, k2 = 3.14, 3.29
    gamma1, gamma2 = 0.43, 0.34
    G = 981.0  # cm/s^2

    def __init__(self, brief):
        self.brief = brief
        self.dt = float(getattr(brief, "sample_time", 2.0))

        self.u_lo = 0.0
        self.u_hi_hard = 10.0
        self.u_start = np.array([3.0, 3.0], dtype=float)

        # ---- conservative safety caps on total command ----
        # Solve, from the nominal steady-state relations, the voltage that
        # would put the corresponding UNMEASURED upper tank exactly at the
        # 20 cm limit, then apply a safety margin.
        two_g_hmax = 2.0 * self.G * 20.0
        sqrt_term = np.sqrt(two_g_hmax)

        u1_overflow = (self.a4 * sqrt_term) / ((1.0 - self.gamma1) * self.k1)
        u2_overflow = (self.a3 * sqrt_term) / ((1.0 - self.gamma2) * self.k2)

        margin = 0.82
        self.u_cap = np.array([u1_overflow * margin, u2_overflow * margin])
        # never exceed hard actuator limits either, and keep a sane floor
        self.u_cap = np.clip(self.u_cap, 1.0, self.u_hi_hard)

        # ---- feedforward matrix (steady-state inversion) ----
        # M * [u1;u2] = [a1*sqrt(2g h1); a2*sqrt(2g h2)]
        self.M = np.array([
            [self.gamma1 * self.k1,        (1.0 - self.gamma2) * self.k2],
            [(1.0 - self.gamma1) * self.k1, self.gamma2 * self.k2],
        ])
        self.M_inv = np.linalg.inv(self.M)

        # ---- PI trim gains (small, correct residual model mismatch) ----
        self.Kp = np.array([0.18, 0.18])
        self.Ki = np.array([0.006, 0.006])
        self.trim_limit = np.array([2.0, 2.0])   # V, trim clamp around feedforward

        # gain scheduling reference (loop gain falls as level falls)
        self.h_ref = 12.0
        self.sched_lo = 0.7
        self.sched_hi = 1.6
        self.h_floor = 2.0

        # measurement filtering
        self.meas_alpha = 0.35

        # actuator slew limiting (protects duty budget, bumpless start)
        self.max_step = 0.10  # V per control period

        self._init_state()

    def _init_state(self):
        self.y_filt = None
        self.u_prev = self.u_start.copy()
        self.trim_int = np.zeros(2)
        self.last_r = None
        self.first_call = True

    def reset(self):
        self._init_state()

    def _feedforward(self, h_targets):
        h_safe = np.clip(h_targets, 0.5, 19.5)
        lhs = np.array([
            self.a1 * np.sqrt(2.0 * self.G * h_safe[0]),
            self.a2 * np.sqrt(2.0 * self.G * h_safe[1]),
        ])
        u_ff = self.M_inv @ lhs
        u_ff = np.clip(u_ff, 0.0, self.u_cap)
        return u_ff

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        quality = (np.asarray(quality, dtype=bool)
                   if quality is not None else np.ones_like(y, dtype=bool))

        if self.first_call or self.y_filt is None:
            y_use = y.copy()
            self.y_filt = y.copy()
            self.first_call = False
        else:
            y_use = np.where(quality, y, self.y_filt)
            self.y_filt = self.meas_alpha * y_use + (1.0 - self.meas_alpha) * self.y_filt
            y_use = self.y_filt

        # setpoint handling: hold last valid target for NaN (unscored) channels
        if self.last_r is None:
            self.last_r = np.where(np.isnan(r), y_use, r)
        else:
            self.last_r = np.where(np.isnan(r), self.last_r, r)
        r_use = self.last_r

        # --- feedforward from nominal steady-state model ---
        u_ff = self._feedforward(r_use)

        # --- gain-scheduled PI trim on top of feedforward ---
        e = r_use - y_use
        h_safe = np.maximum(y_use, self.h_floor)
        sched = np.sqrt(self.h_ref / h_safe)
        sched = np.clip(sched, self.sched_lo, self.sched_hi)

        Kp_eff = self.Kp * sched
        Ki_eff = self.Ki * sched

        trim_int_tentative = self.trim_int + e * self.dt
        trim_tentative = Kp_eff * e + Ki_eff * trim_int_tentative
        trim_tentative = np.clip(trim_tentative, -self.trim_limit, self.trim_limit)

        u_tentative = u_ff + trim_tentative
        u_tentative_capped = np.clip(u_tentative, self.u_lo, self.u_cap)

        # anti-windup: freeze integral if update pushes further into saturation
        saturated_hi = u_tentative > self.u_cap
        saturated_lo = u_tentative < self.u_lo
        pushing_hi = saturated_hi & (e > 0)
        pushing_lo = saturated_lo & (e < 0)
        freeze = pushing_hi | pushing_lo

        self.trim_int = np.where(freeze, self.trim_int, trim_int_tentative)

        trim = Kp_eff * e + Ki_eff * self.trim_int
        trim = np.clip(trim, -self.trim_limit, self.trim_limit)

        u_cmd = u_ff + trim
        u_cmd = np.clip(u_cmd, self.u_lo, self.u_cap)

        # --- slew-rate limiting (duty protection, bumpless behaviour) ---
        delta = np.clip(u_cmd - self.u_prev, -self.max_step, self.max_step)
        u_out = self.u_prev + delta

        # final hard safety clamp (never exceed true actuator limits either)
        u_out = np.clip(u_out, self.u_lo, np.minimum(self.u_cap, self.u_hi_hard))

        self.u_prev = u_out.copy()
        return u_out