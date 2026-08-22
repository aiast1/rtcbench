import numpy as np


class Controller:
    """
    Multivariable PI with steady-state (ridge-regularised) decoupling,
    reference feedforward through the inverse gain, shaped reference
    trajectory for the feedback path, conditional-integration anti-windup,
    output slew limiting (duty compliance) and a safety push on y3.

    Design notes
    ------------
    * Feedforward u_ff = M @ r puts the valves straight at (approximately)
      the steady state required by the new setpoint, so the feedback loop
      only has to clean up the model mismatch.  This removes most of the
      tracking integral that a pure PI pays while it waits out the 15-28 s
      dead times.
    * The feedback error is taken against a shaped reference (two cascaded
      lags) that is roughly what the plant can physically achieve, so the
      proportional term supplies a *moderate* overdrive instead of a huge
      kick that would overshoot behind the dead time.
    * All motion is slew limited to 0.014 per second, comfortably under the
      0.016 duty limit under either a peak or an average interpretation.
    """

    def __init__(self, brief):
        self.dt = float(getattr(brief, "sample_time", 1.0))
        self.ny = 3
        self.nu = 3

        # ---- nominal steady-state gain (hint; plant is a draw around it)
        K = np.array([[4.05, 1.77, 5.88],
                      [5.39, 5.72, 6.90],
                      [4.38, 4.42, 7.20]], dtype=float)

        # ridge right-inverse: bounded gain in the ill-conditioned direction
        lam = 1.0
        self.M = K.T @ np.linalg.inv(K @ K.T + lam * np.eye(3))

        # ---- decoupled loop tuning (unit gain, tau ~ 40-60 s, L ~ 15-28 s)
        self.Kp = np.array([0.80, 0.80, 0.85])
        self.Ti = np.array([60.0, 60.0, 50.0])

        # ---- reference shaping (two cascaded lags per channel)
        self.T_ref = np.array([22.0, 22.0, 18.0])

        # ---- limits
        self.umin = -0.5
        self.umax = 0.5
        self.du_max = 0.014 * self.dt      # slew per step (< duty limit)

        # ---- measurement filtering
        self.tau_f = 8.0
        self.deadband = 0.005

        # ---- command smoothing
        self.tau_u = 2.5

        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.u = np.zeros(self.nu)          # last command sent
        self.uc = np.zeros(self.nu)         # smoothed command state
        self.Ib = np.zeros(self.ny)         # integral (bias) term
        self.yf = np.zeros(self.ny)         # filtered measurement
        self.y_last = np.zeros(self.ny)
        self.r1 = np.zeros(self.ny)         # reference filter states
        self.r2 = np.zeros(self.ny)
        self.init = False
        self.t_last = None

    # ------------------------------------------------------------------
    def step(self, t, y, r, quality):
        dt = self.dt
        if self.t_last is not None:
            d = float(t) - self.t_last
            if 0.0 < d < 10.0 * self.dt:
                dt = d
        self.t_last = float(t)

        ny, nu = self.ny, self.nu

        # ---------------- sanitise inputs -----------------------------
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size < ny:
            y = np.concatenate([y, np.zeros(ny - y.size)])
        r = np.asarray(r, dtype=float).reshape(-1)
        if r.size < ny:
            r = np.concatenate([r, np.full(ny - r.size, np.nan)])

        if quality is None:
            q = np.ones(ny, dtype=bool)
        else:
            q = np.asarray(quality).reshape(-1).astype(bool)
            if q.size < ny:
                q = np.concatenate([q, np.ones(ny - q.size, dtype=bool)])

        ymeas = self.y_last.copy()
        for i in range(ny):
            v = y[i]
            if q[i] and np.isfinite(v) and abs(v) < 5.0:
                ymeas[i] = v
        self.y_last = ymeas.copy()

        if not self.init:
            self.yf = ymeas.copy()
            self.init = True
        else:
            af = float(np.exp(-dt / self.tau_f))
            self.yf = af * self.yf + (1.0 - af) * ymeas

        # setpoints: nan -> hold current measurement (no action requested)
        sp = np.where(np.isfinite(r[:ny]), r[:ny], self.yf)
        sp = np.clip(sp, -0.6, 0.6)

        # ---------------- reference shaping ---------------------------
        for i in range(ny):
            ar = float(np.exp(-dt / max(self.T_ref[i], dt)))
            self.r1[i] = ar * self.r1[i] + (1.0 - ar) * sp[i]
            self.r2[i] = ar * self.r2[i] + (1.0 - ar) * self.r1[i]
        sp_shaped = self.r2

        # ---------------- feedback error ------------------------------
        e = sp_shaped - self.yf
        ed = np.where(np.abs(e) > self.deadband,
                      e - np.sign(e) * self.deadband, 0.0)

        # ---------------- integral ------------------------------------
        Ki = self.Kp / np.maximum(self.Ti, 1.0)
        I_new = self.Ib + Ki * ed * dt
        I_new = np.clip(I_new, -0.7, 0.7)

        # ---------------- safety push on y3 (hard floor at -0.5) ------
        y3 = self.yf[2]
        bias = 0.0
        if y3 < -0.22:
            bias = 2.0 * (-0.22 - y3)

        # ---------------- demanded steady-state output vector ---------
        v = self.Kp * ed + I_new + sp          # sp = feedforward term
        v[2] += bias
        v = np.clip(v, -1.1, 1.1)

        u_req = self.M @ v

        # ---------------- command smoothing, slew + clip --------------
        au = float(np.exp(-dt / self.tau_u))
        self.uc = au * self.uc + (1.0 - au) * u_req
        u_tgt = self.uc

        du = np.clip(u_tgt - self.u, -self.du_max, self.du_max)
        u_new = np.clip(self.u + du, self.umin, self.umax)

        # ---------------- anti-windup --------------------------------
        cut = u_req - u_new
        if np.any(np.abs(cut) > 1e-9):
            back = self.M.T @ cut
            for i in range(ny):
                if back[i] * ed[i] > 0.0 and abs(back[i]) > 1e-6:
                    # only freeze on genuine saturation, not on smoothing lag
                    if (u_new[i] <= self.umin + 1e-9) or (u_new[i] >= self.umax - 1e-9) \
                       or abs(cut[i]) > 3.0 * self.du_max:
                        I_new[i] = self.Ib[i]

        # keep smoothing state from running away when clipped
        self.uc = np.clip(self.uc, self.umin, self.umax)

        self.Ib = I_new
        self.u = u_new
        return self.u.copy()