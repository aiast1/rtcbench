import numpy as np


class Controller:
    """
    Two-point composition control of a binary distillation column ("column A").

    Structure:
      * L (reflux)  <- AT-101 (distillate purity),  direct acting
      * V (boilup)  <- AT-102 (bottoms purity),     direct acting
      Diagonal PI loops (inverse-based decoupling is deliberately avoided:
      with the column's high condition number, input-gain mismatch destabilises
      an inverse decoupler, while detuned diagonal PI is robust).

    A static-model feedforward (steady-state gain inverse, deliberately
    detuned) moves the bulk of the flows on setpoint changes so the feedback
    loops only have to trim; the PI integrators remove residual offset from
    plant/model mismatch and disturbances.

    Safety:
      * V - L (the distillate draw D) is kept inside [0.12, 0.72] so that both
        unmeasured draws D and B = L + F - V stay well above the 0.05 trip,
        even if the (unmeasured) feed is drawn well away from nominal.
      * Hard actuator limits and output rate limiting.
      * Anti-windup by back-calculation, measurement filtering and output
        smoothing to keep actuator travel/duty low.
    """

    # nominal duty point
    U0 = np.array([2.70629, 3.20629])
    R0 = np.array([0.99, 0.99])

    # actuator hard limits
    U_LO = np.array([1.5, 2.0])
    U_HI = np.array([4.5, 5.0])

    # safe band for D = V - L (trip at 0.05; feed may be off-nominal)
    D_MIN = 0.12
    D_MAX = 0.72

    def __init__(self, brief):
        dt = getattr(brief, "sample_time", 1.0)
        self.dt = float(dt) if dt else 1.0

        # steady-state gains of (yD, bottoms-purity) w.r.t. (L, V)
        # (classic column-A linearisation: 0.878/-0.864 ; -1.082/+1.096)
        G = np.array([[0.878, -0.864],
                      [-1.082, 1.096]])
        # detuned inverse for feedforward only (mismatch tolerance)
        self.Ginv = 0.9 * np.linalg.inv(G)

        # PI tuning (per loop, engineering units: kmol/min per mole fraction)
        self.Kc = np.array([12.0, 12.0])
        self.Ti = np.array([12.0, 12.0])     # integral time (steps ~ minutes)
        self.Tt = 8.0                        # anti-windup back-calc time
        self.I_MAX = 1.2                     # integrator clamp (flow units)

        # filters (in units of control steps)
        self.tau_y = 2.0     # measurement filter
        self.tau_ff = 4.0    # feedforward shaping
        self.tau_u = 2.0     # output smoothing
        self.rate = 0.08     # max |du| per step per actuator

        self.reset()

    def reset(self):
        self.yf = None                    # filtered measurement
        self.y_hold = self.R0.copy()      # last good raw reading
        self.r_hold = self.R0.copy()      # last good setpoints
        self.I = np.zeros(2)              # integrator (flow units)
        self.uff = self.U0.copy()         # shaped feedforward
        self.usm = self.U0.copy()         # smoothed command
        self.u_prev = self.U0.copy()      # last output (rate limiting)

    # ---------- helpers -------------------------------------------------

    def _split_clamp(self, u):
        """Keep D = V - L inside the safe band and inside hard limits."""
        L, V = float(u[0]), float(u[1])
        d = V - L
        dc = min(max(d, self.D_MIN), self.D_MAX)
        if dc != d:
            m = 0.5 * (L + V)
            L = m - 0.5 * dc
            V = m + 0.5 * dc
        L = min(max(L, self.U_LO[0]), self.U_HI[0])
        V = min(max(V, self.U_LO[1]), self.U_HI[1])
        # re-check split after hard limiting
        d = V - L
        if d < self.D_MIN:
            V = min(max(L + self.D_MIN, self.U_LO[1]), self.U_HI[1])
            if V - L < self.D_MIN:
                L = min(max(V - self.D_MIN, self.U_LO[0]), self.U_HI[0])
        elif d > self.D_MAX:
            V = min(max(L + self.D_MAX, self.U_LO[1]), self.U_HI[1])
            if V - L > self.D_MAX:
                L = min(max(V - self.D_MAX, self.U_LO[0]), self.U_HI[0])
        return np.array([L, V])

    # ---------- main ----------------------------------------------------

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        q = np.asarray(quality, dtype=bool) if quality is not None \
            else np.ones(2, dtype=bool)

        # --- sanitise measurements (hold last good on bad/stale/insane) ---
        for i in range(2):
            yi = y[i] if i < y.size else np.nan
            ok = (i < q.size and q[i]) and np.isfinite(yi) and 0.5 <= yi <= 1.001
            if ok:
                self.y_hold[i] = yi
        ym = self.y_hold.copy()

        # --- sanitise setpoints ---
        for i in range(2):
            ri = r[i] if i < r.size else np.nan
            if np.isfinite(ri) and 0.8 <= ri <= 1.0:
                self.r_hold[i] = ri
        rs = self.r_hold.copy()

        # --- measurement filter ---
        if self.yf is None:
            self.yf = ym.copy()
        a = 1.0 / max(self.tau_y, 1.0)
        self.yf += a * (ym - self.yf)

        # --- feedforward from setpoints (shaped) ---
        uff_target = self.U0 + self.Ginv @ (rs - self.R0)
        b = 1.0 / max(self.tau_ff, 1.0)
        self.uff += b * (uff_target - self.uff)

        # --- diagonal PI feedback ---
        e = rs - self.yf     # e1>0 -> raise L ; e2>0 -> raise V (direct acting)
        self.I += self.Kc * e * (self.dt / self.Ti)
        self.I = np.clip(self.I, -self.I_MAX, self.I_MAX)
        u_fb = self.Kc * e + self.I
        u_cmd = self.uff + u_fb

        # --- anti-windup: back-calculate against safe/limited command ---
        u_sat = self._split_clamp(u_cmd)
        self.I += (self.dt / self.Tt) * (u_sat - u_cmd)
        self.I = np.clip(self.I, -self.I_MAX, self.I_MAX)

        # --- output smoothing (keeps travel / duty low) ---
        c = 1.0 / max(self.tau_u, 1.0)
        self.usm += c * (u_sat - self.usm)

        # --- rate limit ---
        du = np.clip(self.usm - self.u_prev, -self.rate, self.rate)
        u_out = self.u_prev + du

        # --- final safety clamp ---
        u_out = self._split_clamp(u_out)

        self.u_prev = u_out.copy()
        return u_out