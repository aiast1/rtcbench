import numpy as np


class Controller:
    """
    Trace-heated transport line (TIC-401 -> TT-401), delay-dominant with
    strongly varying transport delay (40 s ... 200 s) and per-scenario
    parameter draws.

    Structure (one fixed design, no scenario detection):

      * heavy first-order filter on the noisy / quantised / dropped-sample
        measurement, value held on bad quality;
      * static feed-forward  u_ff = u_anchor + S*(r - y_anchor)  built from the
        *measured* initial steady state (u=50, y0).  This removes the initial
        offset and most of every setpoint step immediately, which is the only
        thing that can act inside the dead time;
      * delay-shifted reference for the feedback error.  The error actually
        used is the sign-consistent minimum of (r - y) and (r_delayed - y);
        it is zero while the output is still "on its way", which kills the
        integral wind-up that a pure PI unavoidably gets during 200 s of
        transport delay, and is conservative when the delay estimate is wrong
        in either direction;
      * simple, bounded on-line estimates of the transport delay (from the
        observed dead time after each setpoint step) and of the static gain
        (from steady (u,y) pairs), both hard-clipped;
      * PI tuned for the *worst* delay (gain-margin checked for 2x gain error),
        back-calculation anti-windup, integral slowly off-loaded into the
        feed-forward anchor at steady state;
      * command ceiling derived from an upper bound of the process gain so the
        unmeasured heater outlet T_in stays well below its 90 degC limit;
      * slew limit + small dead band on the command so measurement noise never
        reaches the valve (actuator-duty protection).
    """

    # ------------------------------------------------------------------ init
    def __init__(self, brief):
        self.brief = brief

        # ---- sample time ------------------------------------------------
        dt = 5.0
        try:
            v = float(getattr(brief, "sample_time", 5.0))
            if np.isfinite(v) and v > 0.0:
                dt = v
        except Exception:
            dt = 5.0
        self.dt = float(dt)

        # ---- number of actuators ---------------------------------------
        self.n_u = 1
        for nm in ("actuators", "actuator_names", "n_actuators", "n_u"):
            v = getattr(brief, nm, None)
            if v is None:
                continue
            try:
                if hasattr(v, "__len__"):
                    self.n_u = max(1, int(len(v)))
                else:
                    self.n_u = max(1, int(v))
            except Exception:
                pass
            break

        # ---- actuator limits / start point ------------------------------
        self.u_lo = 0.0
        self.u_hi = 100.0
        self.u_start = 50.0
        for nm in ("actuator_start", "actuators_start", "u_start", "u0",
                   "initial_actuators", "actuator_initial", "u_init"):
            v = getattr(brief, nm, None)
            if v is None:
                continue
            try:
                self.u_start = float(np.asarray(v, dtype=float).ravel()[0])
                break
            except Exception:
                pass
        if not np.isfinite(self.u_start):
            self.u_start = 50.0
        self.u_start = float(np.clip(self.u_start, self.u_lo, self.u_hi))

        # ---- fixed design constants ------------------------------------
        self.tau_y = 20.0        # measurement filter [s]
        self.tau_d = 60.0        # slope filter [s]

        self.kp_n = 0.25         # Kp * K_process  (dimensionless)
        self.ki_n = 0.0032       # Ki * K_process  [1/s]

        self.K_nom = 0.8         # nominal degC per % duty
        self.K_min = 0.35
        self.K_max = 1.60
        self.gmarg = 1.35        # gain upper-bound factor for the safety cap

        self.T_in_max = 84.0     # keep estimated heater outlet below this (limit 90)
        self.u_hard = 95.0       # never command more than this

        self.L_init = 140.0      # initial transport-delay estimate [s]
        self.L_min = 40.0
        self.L_max = 250.0
        self.L_shift_max = 260.0

        self.I_lim = 30.0        # integral authority [%]
        self.rate = 5.0          # max command move per sample [%]
        self.db = 0.10           # command dead band [%]

        self.nbuf = int(self.L_shift_max / self.dt) + 6

        self.reset()

    # ----------------------------------------------------------------- reset
    def reset(self):
        self.started = False
        self.t_last = 0.0

        self.yf = np.nan
        self.dyf = 0.0
        self.bad = 0

        self.y0_ref = 55.0       # measured steady output at u0_ref
        self.u0_ref = self.u_start
        self.y_anchor = 55.0     # feed-forward anchor (output)
        self.u_anchor = self.u_start   # feed-forward anchor (duty)

        self.I = 0.0
        self.u_prev = self.u_start

        self.r_prev = None
        self.t_r_change = -1.0e9
        self.rbuf = []

        self.K_est = self.K_nom
        self.L = self.L_init

        self.det_on = False
        self.det_t0 = 0.0
        self.det_y0 = 0.0
        self.det_dr = 0.0

        self.init_n = 0
        self.init_sum = 0.0

        self.uhist = []

    # ------------------------------------------------------------------ step
    def step(self, t, y, r, quality):
        dt = self.dt

        # ---------------- housekeeping of inputs -------------------------
        try:
            t = float(t)
            if not np.isfinite(t):
                t = self.t_last + dt
        except Exception:
            t = self.t_last + dt

        ym = np.nan
        try:
            ya = np.asarray(y, dtype=float).ravel()
            if ya.size > 0:
                ym = float(ya[0])
        except Exception:
            ym = np.nan

        ok = True
        if quality is not None:
            try:
                qa = np.asarray(quality).ravel()
                if qa.size > 0:
                    ok = bool(qa[0])
            except Exception:
                ok = True
        if not np.isfinite(ym):
            ok = False

        rr = np.nan
        try:
            if r is not None:
                ra = np.asarray(r, dtype=float).ravel()
                if ra.size > 0:
                    rr = float(ra[0])
        except Exception:
            rr = np.nan
        if not np.isfinite(rr):
            if self.r_prev is not None and np.isfinite(self.r_prev):
                rr = float(self.r_prev)
            elif ok:
                rr = ym
            else:
                rr = 55.0

        # ---------------- first call: bumpless -----------------------------
        if not self.started:
            self.started = True
            base = ym if ok else rr
            if not np.isfinite(base):
                base = 55.0
            self.yf = float(base)
            self.dyf = 0.0
            self.y0_ref = float(base)
            self.y_anchor = float(base)
            self.u0_ref = float(self.u_start)
            self.u_anchor = float(self.u_start)
            self.u_prev = float(self.u_start)
            self.I = 0.0
            self.r_prev = float(rr)
            self.t_r_change = t
            self.t_last = t
            # pretend the setpoint used to sit at the measured output, so the
            # delayed reference produces no wind-up while the FF acts
            self.rbuf = [float(base)] * self.nbuf
            self.rbuf.append(float(rr))
            self.init_n = 1 if ok else 0
            self.init_sum = float(base) if ok else 0.0
            self.uhist = [float(self.u_start)]
            return np.full(self.n_u, float(self.u_start), dtype=float)

        # ---------------- integrity guards --------------------------------
        if not np.isfinite(self.yf):
            self.yf = ym if ok else self.y0_ref
        if not np.isfinite(self.I):
            self.I = 0.0
        if not np.isfinite(self.u_prev):
            self.u_prev = self.u_start
        if not np.isfinite(self.dyf):
            self.dyf = 0.0

        # ---------------- measurement filtering ---------------------------
        if ok:
            self.bad = 0
            inn = float(np.clip(ym - self.yf, -15.0, 15.0))
            a = dt / (self.tau_y + dt)
            ynew = self.yf + a * inn
        else:
            self.bad += 1
            ynew = self.yf
        slope = (ynew - self.yf) / dt
        self.dyf += (dt / (self.tau_d + dt)) * (slope - self.dyf)
        self.yf = float(ynew)

        # refine the initial steady-state anchor over the first few samples
        if ok and t <= 25.0:
            self.init_n += 1
            self.init_sum += ym
            m = self.init_sum / max(1, self.init_n)
            if np.isfinite(m):
                self.y0_ref = float(m)
                self.y_anchor = float(m)

        # ---------------- setpoint bookkeeping ----------------------------
        dr = rr - (self.r_prev if self.r_prev is not None else rr)
        if abs(dr) > 0.02:
            self.t_r_change = t
        if abs(dr) >= 3.0:
            self.det_on = True
            self.det_t0 = t
            self.det_y0 = self.yf
            self.det_dr = dr
        self.r_prev = float(rr)

        self.rbuf.append(float(rr))
        if len(self.rbuf) > self.nbuf + 4:
            del self.rbuf[0:len(self.rbuf) - (self.nbuf + 4)]

        # ---------------- transport-delay estimation ----------------------
        if self.det_on:
            if (t - self.det_t0) > 420.0:
                self.det_on = False
            else:
                thr = max(0.8, 0.12 * abs(self.det_dr))
                sgn = 1.0 if self.det_dr > 0.0 else -1.0
                if (self.yf - self.det_y0) * sgn > thr:
                    Lm = (t - self.det_t0) - 12.0
                    Lm = float(np.clip(Lm, self.L_min, self.L_max))
                    self.L = 0.4 * self.L + 0.6 * Lm
                    self.det_on = False
        self.L = float(np.clip(self.L, self.L_min, self.L_max))

        # slight over-estimate: sluggish is safe, wind-up is not
        L_use = float(np.clip(1.25 * self.L + 15.0, self.L_min, self.L_shift_max))
        k = int(round(L_use / dt))
        idx = len(self.rbuf) - 1 - k
        if idx < 0:
            idx = 0
        r_sh = float(self.rbuf[idx])
        if not np.isfinite(r_sh):
            r_sh = rr

        # ---------------- error gating -----------------------------------
        e = rr - self.yf
        e_sh = r_sh - self.yf
        if not np.isfinite(e):
            e = 0.0
        if not np.isfinite(e_sh):
            e_sh = e
        if e * e_sh <= 0.0:
            e_use = 0.0
        else:
            e_use = (1.0 if e > 0.0 else -1.0) * min(abs(e), abs(e_sh))

        # ---------------- gains / feed-forward ---------------------------
        K = float(np.clip(self.K_est, self.K_min, self.K_max))
        S = 1.0 / K
        Kp = self.kp_n * S
        Ki = self.ki_n * S

        u_ff = self.u_anchor + S * (rr - self.y_anchor)
        if not np.isfinite(u_ff):
            u_ff = self.u_prev

        # ---------------- safety ceiling on the duty ---------------------
        K_ub = min(K * self.gmarg, 1.9)
        u_cap = self.u0_ref + (self.T_in_max - self.y0_ref) / max(K_ub, 0.2)
        if not np.isfinite(u_cap):
            u_cap = 70.0
        u_cap = float(min(u_cap, self.u_hard, self.u_hi))
        u_cap = float(max(u_cap, 0.0))

        # ---------------- PI with anti-windup ----------------------------
        if self.bad * dt < 60.0:
            self.I += Ki * e_use * dt
        self.I = float(np.clip(self.I, -self.I_lim, self.I_lim))

        u_raw = u_ff + Kp * e_use + self.I
        if not np.isfinite(u_raw):
            u_raw = self.u_prev

        if u_raw > u_cap:
            self.I -= (u_raw - u_cap)
            u_raw = u_cap
        elif u_raw < self.u_lo:
            self.I -= (u_raw - self.u_lo)
            u_raw = self.u_lo
        self.I = float(np.clip(self.I, -self.I_lim, self.I_lim))

        # ---------------- slew limit + dead band -------------------------
        du = u_raw - self.u_prev
        if du > self.rate:
            u_cmd = self.u_prev + self.rate
        elif du < -self.rate:
            u_cmd = self.u_prev - self.rate
        elif abs(du) < self.db:
            u_cmd = self.u_prev
        else:
            u_cmd = u_raw

        u_cmd = float(np.clip(u_cmd, self.u_lo, min(u_cap, self.u_hi)))
        if not np.isfinite(u_cmd):
            u_cmd = float(np.clip(self.u_prev, self.u_lo, self.u_hi))

        # ---------------- steady state: off-load I, learn gain -----------
        self.uhist.append(u_cmd)
        if len(self.uhist) > 80:
            del self.uhist[0:len(self.uhist) - 80]
        if len(self.uhist) >= 40:
            w = self.uhist[-40:]
            uspan = max(w) - min(w)
        else:
            uspan = 99.0

        steady = (ok and (t - self.t_r_change) > 320.0
                  and abs(self.dyf) < 0.004 and abs(e) < 2.5 and uspan < 1.5)

        if steady:
            tr = 0.03 * self.I
            self.I -= tr
            self.u_anchor = float(np.clip(self.u_anchor + tr, -25.0, 125.0))

            du_ref = u_cmd - self.u0_ref
            dy_ref = self.yf - self.y0_ref
            if abs(du_ref) > 6.0 and dy_ref * du_ref > 0.0:
                K_obs = float(np.clip(dy_ref / du_ref, 0.30, 1.80))
                if np.isfinite(K_obs):
                    self.K_est += 0.03 * (K_obs - self.K_est)
                    self.K_est = float(np.clip(self.K_est, self.K_min, self.K_max))
        if not np.isfinite(self.K_est):
            self.K_est = self.K_nom

        # ---------------- commit -----------------------------------------
        self.u_prev = u_cmd
        self.t_last = t

        out = np.full(self.n_u, u_cmd, dtype=float)
        if not np.all(np.isfinite(out)):
            out = np.full(self.n_u, float(self.u_start), dtype=float)
        return out