import numpy as np


class Controller:
    """
    Multivariable DMC-style (step-response) predictive controller for the
    3x3 shell-fractionator loop.

      * nominal FOPDT model matrix (with dead-time margin) for prediction,
        blocked move horizon, Tikhonov move regularisation so the
        ill-conditioned plant direction is not blindly inverted,
      * BOX-CONSTRAINED QP (projected accelerated gradient) on the absolute
        valve levels over the blocked horizon -> correct error trade-off when a
        valve saturates, and structural anti-windup,
      * output-bias (DMC) correction -> offset free, rejects the unmeasured
        upstream circulating-duty disturbances; the internal model is driven
        with the ACTUALLY APPLIED (clipped / rate-limited) signal,
      * per-channel reference funnel + setpoint-rate preview for the ramp,
      * slow, bounded oscillation detector that can only DETUNE (never
        aggravate) if the drawn plant fights the nominal model,
      * graded, soft-then-hard protection of the TI-103 floor (y3 >= -0.5).
    """

    # ------------------------------------------------------------------ init
    def __init__(self, brief):
        try:
            ts = float(brief.sample_time)
        except Exception:
            ts = 1.0
        if not np.isfinite(ts) or ts <= 0.0:
            ts = 1.0
        self.ts = ts

        self.ny = 3
        self.nu = 3
        self.umin = -0.5
        self.umax = 0.5

        # ---- nominal model (NOT the as-running plant: only a guide)
        self.K = np.array([[4.05, 1.77, 5.88],
                           [5.39, 5.72, 6.90],
                           [4.38, 4.42, 7.20]], dtype=float)
        TAU = np.array([[50., 60., 50.],
                        [50., 60., 40.],
                        [33., 44., 19.]], dtype=float)
        LN = np.array([[27., 28., 27.],
                       [18., 14., 15.],
                       [20., 22., 0.]], dtype=float)

        self.tau = np.maximum(TAU, 2.0 * ts)
        self.a = np.exp(-ts / self.tau)
        self.oma = 1.0 - self.a
        # dead-time margin: over-estimating delay is conservative
        self.Ld = np.maximum(np.rint(LN / ts).astype(int) + 2, 1)
        self.Lmax = int(self.Ld.max())

        # ---- horizon / blocking
        self.P = int(max(60, round(260.0 / ts)))
        self.blocks = sorted(set(int(round(x / ts)) for x in (0.0, 15.0, 40.0, 90.0)))
        self.Nb = len(self.blocks)
        self.nv = self.nu * self.Nb

        # ---- tuning (balanced for worst-case robustness)
        self.T_ref = np.array([32.0, 28.0, 24.0])   # reference funnel  [s]
        self.lam = 1.5                              # move penalty
        self.T_bias = 16.0                          # bias filter       [s]
        self.T_yf = 5.0                             # meas filter       [s]
        self.du_max = 0.030 * ts
        self.du_max_safe = 0.060 * ts
        self.deadband = 0.0012

        # ---- setpoint-rate preview
        self.T_rdot = 6.0
        self.step_thr = 0.02
        self.lead_cap = 60.0

        self.Jidx = np.tile(np.arange(self.nu), (self.ny, 1))

        self._build()

        kk = np.arange(1, self.P + 1) * ts
        self.dec = np.exp(-kk[None, :] / self.T_ref[:, None])
        self.lead = np.minimum(kk, self.lead_cap)
        self.beta_b = 1.0 - np.exp(-ts / self.T_bias)
        self.beta_y = 1.0 - np.exp(-ts / self.T_yf)
        self.beta_r = 1.0 - np.exp(-ts / self.T_rdot)

        # ---- oscillation detector
        self.Nd = max(4, int(round(12.0 / ts)))
        self.hlen = 2 * self.Nd + 2
        self.beta_osc = 1.0 - np.exp(-ts / 60.0)
        self.g_dn = 1.0 - np.exp(-ts / 10.0)
        self.g_up = 1.0 - np.exp(-ts / 150.0)

        self.reset()

    # ------------------------------------------------------- model build-up
    def _build(self):
        P, ny, nu, Nb = self.P, self.ny, self.nu, self.Nb

        n = np.arange(P + 1)
        s = np.zeros((ny, nu, P + 1))
        for i in range(ny):
            for j in range(nu):
                idx = n - self.Ld[i, j]
                m = idx >= 1
                s[i, j, m] = self.K[i, j] * (1.0 - self.a[i, j] ** idx[m])

        kk = np.arange(1, P + 1)
        Su = np.zeros((ny * P, nu * Nb))
        for i in range(ny):
            for j in range(nu):
                for bi, mb in enumerate(self.blocks):
                    nn = kk - mb
                    col = np.zeros(P)
                    ok = nn >= 0
                    col[ok] = s[i, j, nn[ok]]
                    Su[i * P:(i + 1) * P, j * Nb + bi] = col
        self.Su = Su

        # increments from absolute block levels:  d = D v + c
        D = np.zeros((self.nv, self.nv))
        for j in range(nu):
            for b in range(Nb):
                k = j * Nb + b
                D[k, k] = 1.0
                if b > 0:
                    D[k, k - 1] = -1.0
        self.D = D
        self.Dt = D.T
        self.b0idx = np.array([j * Nb for j in range(nu)])

        self.ws_n = self._make_set(np.array([1.0, 1.0, 1.0]), self.lam)
        self.ws_s = self._make_set(np.array([1.0, 1.0, 3.0]), self.lam)
        self.ws_e = self._make_set(np.array([0.5, 0.5, 8.0]), self.lam)

    def _make_set(self, q, lam):
        P = self.P
        Qd = np.repeat(np.asarray(q, dtype=float), P) / float(P)
        SuTQ = np.ascontiguousarray(self.Su.T * Qd[None, :])
        M = SuTQ @ self.Su + lam * np.eye(self.nv)
        M = 0.5 * (M + M.T)
        A = self.Dt @ M @ self.D
        A = 0.5 * (A + A.T)
        ev = np.linalg.eigvalsh(A)
        L = float(max(ev.max(), 1e-6))
        return {'SuTQ': SuTQ, 'M': M, 'A': A, 'step': 1.0 / L}

    # ---------------------------------------------------------- QP solution
    def _solve(self, ws, E, u0, v0):
        c = np.zeros(self.nv)
        c[self.b0idx] = -u0
        gl = ws['SuTQ'] @ E
        qv = self.Dt @ (ws['M'] @ c - gl)
        A = ws['A']
        st = ws['step']
        v = np.clip(v0, self.umin, self.umax)
        vp = v.copy()
        for k in range(1, 51):
            w = v + ((k - 1.0) / (k + 2.0)) * (v - vp)
            vp = v
            g = A @ w + qv
            v = np.clip(w - st * g, self.umin, self.umax)
        if not np.all(np.isfinite(v)):
            v = np.clip(v0, self.umin, self.umax)
        return v

    # ----------------------------------------------------------------- reset
    def reset(self):
        self.x = np.zeros((self.ny, self.nu))
        self.U = np.zeros((self.Lmax + 2, self.nu))    # U[m] = u(t-1-m)
        self.b = np.zeros(self.ny)
        self.yf = np.zeros(self.ny)
        self.y_last = np.zeros(self.ny)
        self.r_last = np.zeros(self.ny)
        self.rdot = np.zeros(self.ny)
        self.v = np.zeros(self.nv)
        self.first = True
        self.lev = 0
        self.u = np.zeros(self.nu)
        self.ebuf = np.zeros((self.hlen, self.ny))
        self.ecnt = 0
        self.osc_n = np.zeros(self.ny)
        self.osc_d = np.zeros(self.ny)
        self.g = 1.0

    # ------------------------------------------------------------------ step
    def step(self, t, y, r, quality):
        ny, nu, P = self.ny, self.nu, self.P

        # ---------- read / sanitise measurements -------------------------
        ym = np.zeros(ny)
        yin = np.asarray(y, dtype=float).ravel()
        good = np.ones(ny, dtype=bool)
        if quality is not None:
            qa = np.asarray(quality).ravel()
            for i in range(min(ny, qa.size)):
                good[i] = bool(qa[i])
        for i in range(ny):
            v = yin[i] if i < yin.size else np.nan
            if (not good[i]) or (not np.isfinite(v)):
                ym[i] = self.y_last[i]
                good[i] = False
            else:
                ym[i] = v
                self.y_last[i] = v

        # ---------- setpoints + ramp-rate estimate ------------------------
        rin = np.asarray(r, dtype=float).ravel()
        rr = self.r_last.copy()
        for i in range(ny):
            if i < rin.size and np.isfinite(rin[i]):
                rr[i] = float(rin[i])
        rr = np.clip(rr, -0.6, 0.6)
        dr = rr - self.r_last
        if self.first:
            self.rdot[:] = 0.0
        else:
            for i in range(ny):
                if abs(dr[i]) > self.step_thr:
                    self.rdot[i] = 0.0
                else:
                    self.rdot[i] += self.beta_r * (dr[i] / self.ts - self.rdot[i])
        self.r_last = rr.copy()

        if self.first:
            self.yf = ym.copy()
        else:
            self.yf = self.yf + self.beta_y * (ym - self.yf)

        # ---------- internal model advance (applied inputs) ---------------
        U = self.U
        uin = U[self.Ld, self.Jidx]
        self.x = self.a * self.x + self.oma * self.K * uin
        ymod = self.x.sum(axis=1)

        # ---------- bias / integral action --------------------------------
        d = ym - ymod
        if self.first:
            self.b = d.copy()
            self.first = False
        else:
            for i in range(ny):
                if good[i]:
                    self.b[i] += self.beta_b * (d[i] - self.b[i])
        self.b = np.clip(self.b, -1.5, 1.5)

        # ---------- free response (hold last applied input) ---------------
        u0 = U[0].copy()
        xf = self.x.copy()
        f = np.empty((ny, P))
        k0 = min(self.Lmax, P)
        for k in range(1, k0 + 1):
            m = self.Ld - k
            np.clip(m, 0, None, out=m)
            uk = U[m, self.Jidx]
            xf = self.a * xf + self.oma * self.K * uk
            f[:, k - 1] = xf.sum(axis=1)
        if P > k0:
            nn = np.arange(1, P - k0 + 1)
            A3 = self.a[:, :, None] ** nn[None, None, :]
            ss = (self.K * u0[None, :])[:, :, None]
            xr = A3 * xf[:, :, None] + (1.0 - A3) * ss
            f[:, k0:] = xr.sum(axis=1)

        p3 = f[2] + self.b[2]
        h2 = max(4, P // 2)
        p3min = float(np.min(p3[:h2]))

        # ---------- graded TI-103 floor protection ------------------------
        y3 = ym[2]
        lev = self.lev
        if lev >= 2:
            if y3 > -0.36:
                lev = 1
        elif lev == 1:
            if y3 > -0.26 and p3min > -0.32:
                lev = 0
        if y3 < -0.43:
            lev = 2
        elif lev == 0 and (y3 < -0.34 or p3min < -0.40):
            lev = 1
        self.lev = lev

        rr_eff = rr.copy()
        if lev >= 1:
            rr_eff[2] = max(rr_eff[2], 0.0)
        if lev >= 2:
            rr_eff[2] = max(rr_eff[2], 0.10)

        if lev >= 2:
            ws = self.ws_e
            dumax = self.du_max_safe
        elif lev == 1:
            ws = self.ws_s
            dumax = self.du_max_safe
        else:
            ws = self.ws_n
            dumax = self.du_max

        # ---------- oscillation detector (detune only) --------------------
        e_now = rr - self.yf
        self.ebuf[1:] = self.ebuf[:-1]
        self.ebuf[0] = e_now
        self.ecnt += 1
        if self.ecnt > 2 * self.Nd + 1:
            d1 = self.ebuf[0] - self.ebuf[self.Nd]
            d2 = self.ebuf[self.Nd] - self.ebuf[2 * self.Nd]
            pr = d1 * d2
            for i in range(ny):
                if abs(d1[i]) > 0.004 and abs(d2[i]) > 0.004:
                    self.osc_n[i] += self.beta_osc * (max(0.0, -pr[i]) - self.osc_n[i])
                    self.osc_d[i] += self.beta_osc * (abs(pr[i]) - self.osc_d[i])
            osc = float(np.max(self.osc_n / (self.osc_d + 1e-12)))
        else:
            osc = 0.0
        g_t = 1.0 - 1.1 * max(0.0, osc - 0.35)
        g_t = min(1.0, max(0.5, g_t))
        if g_t < self.g:
            self.g += self.g_dn * (g_t - self.g)
        else:
            self.g += self.g_up * (g_t - self.g)
        gain = 1.0 if lev >= 2 else self.g

        # ---------- reference funnel (with ramp preview) ------------------
        rdot = self.rdot.copy()
        if lev >= 1:
            rdot[:] = 0.0
        rfut = rr_eff[:, None] + rdot[:, None] * self.lead[None, :]
        np.clip(rfut, -0.62, 0.62, out=rfut)
        err0 = rr_eff - self.yf
        tgt = rfut - err0[:, None] * self.dec
        E = (tgt - f - self.b[:, None]).reshape(-1)
        if not np.all(np.isfinite(E)):
            E = np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)

        # ---------- constrained QP ----------------------------------------
        vsol = self._solve(ws, E, u0, self.v)
        self.v = vsol
        du = gain * (vsol[self.b0idx] - u0)
        if not np.all(np.isfinite(du)):
            du = np.zeros(nu)

        # ---------- dead-band, rate limit, hard clamp ---------------------
        du = np.where(np.abs(du) > self.deadband, du, 0.0)
        du = np.clip(du, -dumax, dumax)
        if lev >= 2:
            du[2] = max(du[2], 0.03 * self.ts)
        unew = np.clip(u0 + du, self.umin, self.umax)
        if not np.all(np.isfinite(unew)):
            unew = np.clip(u0, self.umin, self.umax)

        # ---------- shift applied-input history ---------------------------
        self.U[1:] = self.U[:-1]
        self.U[0] = unew
        self.u = unew
        return unew.copy()