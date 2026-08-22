import numpy as np


class Controller:
    """
    Three-element feedwater control for a natural-circulation drum boiler.

    Structure
    ---------
    * Inner loop : feedwater FLOW controller (FT-104 -> FCV-201), PI with a
      valve-authority (pressure) feedforward.  This removes the "same position
      passes more water when the drum sags" nonlinearity and lets the outer
      loop think in kg/s.
    * Feedforward: steam flow (FT-103) is passed straight through to the flow
      demand.  Inventory therefore stays balanced through a load change even
      while the indicated level is swelling/shrinking.
    * Outer loop : level controller acting on an INVENTORY ESTIMATE built as a
      complementary filter,   l_est = HP(integral of (w_fw - w_steam)) + LP(LT-101).
      The flow integral carries no shrink/swell inverse response, so the loop
      can be tightened; the slow LP(LT-101) path removes the drift of the flow
      integral and makes the steady state track the *indicated* setpoint (which
      is what is scored).
    * Soft barriers keep indicated level away from the +250 mm carryover trip
      and the estimated inventory away from the -250 mm low-water trip, with the
      inventory barrier taking priority.
    * Output is rate limited, dead-banded and quantised to keep actuator duty
      and travel small.
    """

    # ------------------------------------------------------------------ init
    def __init__(self, brief):
        # ---------- sample time ----------
        dt = 5.0
        try:
            dt = float(getattr(brief, 'sample_time', 5.0))
        except Exception:
            dt = 5.0
        if (not np.isfinite(dt)) or dt <= 0.0:
            dt = 5.0
        self.dt = dt

        # ---------- bumpless start ----------
        u0 = 45.0
        for nm in ('actuators_start', 'actuator_start', 'u_start', 'u0',
                   'actuator_initial', 'u_init', 'actuators_initial'):
            v = getattr(brief, nm, None)
            if v is None:
                continue
            try:
                u0 = float(np.asarray(v, dtype=float).ravel()[0])
                break
            except Exception:
                pass
        if not np.isfinite(u0):
            u0 = 45.0
        self.u0 = float(min(max(u0, 0.0), 100.0))

        # ---------- process scaling ----------
        self.k_inv = 0.07        # mm of level per kg of water added
        self.p_pump = 120.0      # assumed pump discharge, bar (nominal)

        # ---------- measurement filters (s) ----------
        self.tau_l = 10.0
        self.tau_w = 20.0
        self.tau_p = 25.0

        # ---------- observer ----------
        self.tau_b = 250.0       # bias / void-offset adaptation

        # ---------- outer (level) loop ----------
        self.Kp = 0.13           # (kg/s) per mm
        self.Ki = 0.00035        # (kg/s) per mm per s
        self.I_lim = 10.0
        self.trim_lim = 16.0
        self.sp_rate = 0.60      # mm/s internal setpoint slew
        self.ff_lim = 12.0

        # ---------- barriers ----------
        self.ind_hi = 165.0      # start pushing down on indicated level
        self.inv_lo = -170.0     # start pushing up on estimated inventory
        self.Kb = 0.12
        self.bar_hi_lim = 10.0
        self.bar_lo_lim = 12.0

        # ---------- pressure guard ----------
        self.p_lo = 74.0
        self.p_hi = 96.0
        self.Kp_guard = 1.0
        self.p_guard_lim = 6.0

        # ---------- inner (flow) loop ----------
        self.Kpw = 0.35          # % per (kg/s)
        self.Kiw = 0.020         # % per (kg/s) per s

        # ---------- output conditioning ----------
        self.du_lim = 2.5        # % per sample
        self.db = 0.25           # % dead band
        self.quant = 0.1         # % command resolution

        self.reset()

    # ----------------------------------------------------------------- reset
    def reset(self):
        self.init_done = False
        self.hold = np.array([0.0, 85.0, 50.0, 50.0], dtype=float)

        self.l_f = 0.0
        self.p_f = 85.0
        self.ws_f = 50.0
        self.wf_f = 50.0

        self.l_inv = 0.0
        self.b = 0.0
        self.l_est = 0.0

        self.I = 0.0
        self.Iu = self.u0
        self.dp_ref = max(self.p_pump - 85.0, 5.0)

        self.r_s = 0.0
        self.r_target = 0.0

        self.u_out = self.u0
        self.u_last = self.u0
        self.sat = 0

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _c(x, lo, hi):
        return float(min(max(x, lo), hi))

    def _lp(self, state, x, tau):
        a = self.dt / (tau + self.dt)
        return state + (x - state) * a

    # ------------------------------------------------------------------ step
    def step(self, t, y, r, quality):
        try:
            u = self._core(t, y, r, quality)
            if not np.isfinite(u):
                u = self.u_out
        except Exception:
            u = self.u_out
        u = self._c(u, 0.0, 100.0)
        self.u_out = u
        self.u_last = u
        return np.array([u], dtype=float)

    # ------------------------------------------------------------------ core
    def _core(self, t, y, r, quality):
        dt = self.dt

        # -------------------- read measurements, hold bad ones -------------
        ya = np.asarray(y, dtype=float).ravel()
        n = ya.size
        if quality is None:
            qa = np.ones(4, dtype=bool)
        else:
            qa = np.asarray(quality).ravel()
            try:
                qa = qa.astype(bool)
            except Exception:
                qa = np.ones(4, dtype=bool)
        for i in range(4):
            v = ya[i] if i < n else np.nan
            g = bool(qa[i]) if i < qa.size else True
            if g and np.isfinite(v):
                self.hold[i] = float(v)

        l_raw = self._c(self.hold[0], -600.0, 600.0)
        p_raw = self._c(self.hold[1], 55.0, 120.0)
        ws_raw = self._c(self.hold[2], 0.0, 100.0)
        wf_raw = self._c(self.hold[3], 0.0, 100.0)

        # -------------------- one-time initialisation ---------------------
        if not self.init_done:
            self.l_f = l_raw
            self.p_f = p_raw
            self.ws_f = ws_raw
            self.wf_f = wf_raw
            self.l_inv = 0.0
            self.b = l_raw
            self.l_est = l_raw
            self.r_s = l_raw
            self.r_target = l_raw
            self.I = 0.0
            self.Iu = self.u0
            self.dp_ref = max(self.p_pump - p_raw, 5.0)
            self.u_out = self.u0
            self.init_done = True
            return self.u0

        # -------------------- filtering ------------------------------------
        self.l_f = self._lp(self.l_f, l_raw, self.tau_l)
        self.p_f = self._lp(self.p_f, p_raw, self.tau_p)
        self.ws_f = self._lp(self.ws_f, ws_raw, self.tau_w)
        self.wf_f = self._lp(self.wf_f, wf_raw, self.tau_w)

        l_ind = self.l_f
        p = self.p_f
        ws = self.ws_f
        wf = self.wf_f

        # -------------------- inventory observer --------------------------
        self.l_inv += self.k_inv * (wf - ws) * dt
        self.l_inv = self._c(self.l_inv, -5000.0, 5000.0)
        self.b += (l_ind - self.l_inv - self.b) * (dt / self.tau_b)
        self.b = self._c(self.b, -1500.0, 1500.0)
        l_est = self.l_inv + self.b
        l_est = self._c(l_est, -600.0, 600.0)
        self.l_est = l_est

        # -------------------- setpoint shaping ----------------------------
        rt = self.r_target
        try:
            ra = np.asarray(r, dtype=float).ravel()
            if ra.size > 0 and np.isfinite(ra[0]):
                rt = float(ra[0])
        except Exception:
            pass
        rt = self._c(rt, -300.0, 220.0)
        self.r_target = rt

        step_lim = self.sp_rate * dt
        d = self._c(rt - self.r_s, -step_lim, step_lim)
        self.r_s += d
        rdot = d / dt

        # -------------------- outer level loop ----------------------------
        e = self.r_s - l_est
        ff = self._c(rdot / max(self.k_inv, 1e-3), -self.ff_lim, self.ff_lim)

        # barriers -------------------------------------------------------
        # inventory has priority: fade the carryover barrier out if inventory
        # is already low
        fac = self._c((l_est - (self.inv_lo + 10.0)) / 100.0, 0.0, 1.0)
        bar_hi = -fac * self.Kb * max(0.0, l_ind - self.ind_hi)
        bar_hi = self._c(bar_hi, -self.bar_hi_lim, 0.0)
        bar_lo = self.Kb * max(0.0, self.inv_lo - l_est)
        bar_lo = self._c(bar_lo, 0.0, self.bar_lo_lim)

        # pressure guard (mild): cold feedwater depresses drum pressure
        pg = 0.0
        if p < self.p_lo:
            pg = -self.Kp_guard * (self.p_lo - p)
        elif p > self.p_hi:
            pg = self.Kp_guard * (p - self.p_hi)
        pg = self._c(pg, -self.p_guard_lim, self.p_guard_lim)

        # integral with conditional update (anti-windup)
        I_try = self._c(self.I + self.Ki * e * dt, -self.I_lim, self.I_lim)
        base = self.Kp * e + ff + bar_hi + bar_lo
        trim_try = base + I_try
        if trim_try > self.trim_lim or trim_try < -self.trim_lim:
            # only allow updates that pull the demand back inside
            if abs(I_try) < abs(self.I):
                self.I = I_try
        elif self.sat != 0 and (self.sat * (self.Ki * e) > 0.0):
            pass  # valve on its stop, pushing further would wind up
        else:
            self.I = I_try

        trim = self._c(base + self.I, -self.trim_lim, self.trim_lim)

        w_dem = ws + trim + pg
        w_dem = self._c(w_dem, 0.0, 95.0)

        # -------------------- inner flow loop -----------------------------
        e_w = w_dem - wf

        dp = max(self.p_pump - p, 4.0)
        corr = np.sqrt(self.dp_ref / dp)
        corr = self._c(corr, 0.80, 1.25)

        Iu_try = self.Iu + self.Kiw * e_w * dt
        Iu_try = self._c(Iu_try, -20.0, 130.0)
        u_pi = Iu_try + self.Kpw * e_w
        u_raw = u_pi * corr
        u_sat = self._c(u_raw, 0.0, 100.0)

        if u_raw > 100.0 or u_raw < 0.0:
            # back-calculation anti-windup on the flow integrator
            Iu_try = self._c(u_sat / corr - self.Kpw * e_w, -20.0, 130.0)
            self.sat = 1 if u_raw > 100.0 else -1
        else:
            self.sat = 0
        self.Iu = Iu_try

        # -------------------- output conditioning -------------------------
        du = self._c(u_sat - self.u_out, -self.du_lim, self.du_lim)
        u_cmd = self.u_out + du

        if abs(u_cmd - self.u_out) < self.db:
            u_cmd = self.u_out
        else:
            u_cmd = round(u_cmd / self.quant) * self.quant

        return self._c(u_cmd, 0.0, 100.0)