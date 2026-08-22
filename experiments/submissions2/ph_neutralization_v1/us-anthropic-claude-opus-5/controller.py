"""
pH neutralization controller (ph_neutralization_v1, tier: mismatch)

Strategy
--------
1.  Linearising transform.  The titration curve makes the pH->base-flow gain
    vary by an order of magnitude over the operating range.  The measured pH is
    mapped through a charge-balance / reaction-invariant function g(pH) (net
    strong-base equivalent).  In g-space the plant is nearly first-order plus
    dead time with an almost draw-independent gain, so ONE fixed PI tuning
    works everywhere.
2.  Robust PI on g (no derivative: analyser is noisy, quantized, ~20 s late),
    setpoint weighting, back-calculated anti-windup.
3.  Level override: high-level cap and low-level floor on the caustic demand,
    because level is unmanaged and rides on base flow.  Safety wins over pH.
4.  Duty management: output rate limit + move deadband, so the valve does not
    chatter.
"""

import numpy as np


def _g(pH, Wb=0.0005, pK1=6.35, pK2=10.25):
    """Net strong-base equivalent (reaction invariant) at a given pH."""
    pH = float(np.clip(pH, 0.5, 13.5))
    h = 10.0 ** (-pH)
    oh = 10.0 ** (pH - 14.0)
    # buffer (carbonate) contribution, mean charge per buffer molecule
    a1 = 10.0 ** (pH - pK1)
    a2 = 10.0 ** (2.0 * pH - pK1 - pK2)
    denom = 1.0 + a1 + a2
    zbar = (a1 + 2.0 * a2) / denom
    return oh - h + Wb * zbar


class Controller:
    # ---------------- construction ----------------
    def __init__(self, brief):
        self.dt = float(getattr(brief, "sample_time", 5.0))

        # actuator limits
        self.u_min = 0.0
        self.u_max = 30.0
        try:
            lim = brief.actuators[0]
            self.u_min = float(getattr(lim, "min", self.u_min))
            self.u_max = float(getattr(lim, "max", self.u_max))
        except Exception:
            pass

        # bumpless start
        self.u0 = 14.22
        try:
            a0 = np.asarray(getattr(brief, "actuator_start",
                                    getattr(brief, "u0", [14.22])),
                            dtype=float).ravel()
            if a0.size >= 1 and np.isfinite(a0[0]):
                self.u0 = float(a0[0])
        except Exception:
            pass

        # ---- PI tuning in g-space (SIMC-ish, detuned for robustness) ----
        # dead time ~20 s analyser + ~5 s hold; tank tau ~ 100 s (A1/q)
        self.theta = 25.0
        self.tau = 110.0
        # nominal gain dg/du  [ (mol/L) per (mL/s) ] : caustic strength / flow
        self.Kp_g = 4000.0          # mL/s per unit g  (proportional)
        self.Ti = 170.0             # integral time, s
        self.b = 0.45               # setpoint weight on proportional term

        # measurement filter
        self.tau_f = 12.0

        # output shaping
        self.rate_lim = 1.6 * self.dt / 5.0   # mL/s per sample
        self.deadband = 0.045                 # mL/s, suppress dithering

        # level guard
        self.h_hi = 30.0
        self.h_lo = 5.0
        self.h_hi_act = 26.5     # start backing off here
        self.h_lo_act = 8.5
        self.k_h = 2.6           # mL/s per cm

        self.reset()

    # ---------------- reset ----------------
    def reset(self):
        self.u = self.u0
        self.u_cmd = self.u0
        self.I = self.u0                 # integrator holds bias (in mL/s)
        self.pH_f = None
        self.h_f = None
        self.r_f = None
        self.g_meas = None
        self.last_good_pH = None
        self.last_good_h = None
        self.k = 0
        self.first = True
        self.r_prev = None
        self.sp_rate = 0.0

    # ---------------- helpers ----------------
    def _clip(self, v):
        return float(min(max(v, self.u_min), self.u_max))

    # ---------------- step ----------------
    def step(self, t, y, r, quality):
        dt = self.dt
        y = np.asarray(y, dtype=float).ravel()
        r = np.asarray(r, dtype=float).ravel()
        try:
            q = np.asarray(quality, dtype=bool).ravel()
        except Exception:
            q = np.ones(y.size, dtype=bool)
        if q.size < y.size:
            q = np.concatenate([q, np.ones(y.size - q.size, dtype=bool)])

        # ---------- read pH ----------
        pH_ok = q[0] and np.isfinite(y[0]) and 0.0 < y[0] < 14.0
        if pH_ok:
            pH = float(y[0])
            self.last_good_pH = pH
        elif self.last_good_pH is not None:
            pH = self.last_good_pH
        else:
            pH = 7.0

        # ---------- read level ----------
        h_ok = (y.size > 1) and q[1] and np.isfinite(y[1])
        if h_ok:
            h = float(y[1])
            self.last_good_h = h
        elif self.last_good_h is not None:
            h = self.last_good_h
        else:
            h = 17.0

        # ---------- filters ----------
        a = dt / (self.tau_f + dt)
        if self.pH_f is None:
            self.pH_f = pH
        else:
            self.pH_f += a * (pH - self.pH_f)
        ah = dt / (8.0 + dt)
        if self.h_f is None:
            self.h_f = h
        else:
            self.h_f += ah * (h - self.h_f)

        # ---------- setpoint ----------
        rs = r[0] if (r.size > 0 and np.isfinite(r[0])) else self.pH_f
        rs = float(np.clip(rs, 4.6, 10.1))
        if self.r_f is None:
            self.r_f = rs
            self.r_prev = rs
        # gentle setpoint smoothing to avoid kick, plus ramp-rate estimate
        self.sp_rate = 0.85 * self.sp_rate + 0.15 * ((rs - self.r_prev) / dt)
        self.r_prev = rs
        self.r_f += (dt / (20.0 + dt)) * (rs - self.r_f)
        r_use = self.r_f

        # ---------- transform to g-space ----------
        g_y = _g(self.pH_f)
        g_r = _g(r_use)
        # small lead on the ramp: anticipate where sp will be one dead time out
        g_r_lead = _g(float(np.clip(r_use + self.sp_rate * 0.6 * self.theta,
                                    4.6, 10.1)))
        e = g_r_lead - g_y                     # error for integral action
        e_p = self.b * g_r_lead - g_y          # error for proportional action

        # ---------- PI ----------
        P = self.Kp_g * e_p
        if self.first:
            # initialise integrator so first output == u0 (bumpless)
            self.I = self.u0 - P
            self.first = False
        else:
            self.I += (self.Kp_g * dt / self.Ti) * e

        u_pi = P + self.I

        # ---------- level override (cap / floor) ----------
        u_cap = self.u_max
        u_floor = self.u_min
        if self.h_f > self.h_hi_act:
            u_cap = self.u - self.k_h * (self.h_f - self.h_hi_act)
            if self.h_f > 28.5:
                u_cap = min(u_cap, self.u - 3.0 * (self.h_f - 28.5) - 0.5)
            u_cap = max(u_cap, self.u_min)
        if self.h_f < self.h_lo_act:
            u_floor = self.u + self.k_h * (self.h_lo_act - self.h_f)
            if self.h_f < 7.0:
                u_floor = max(u_floor, self.u + 3.0 * (7.0 - self.h_f) + 0.5)
            u_floor = min(u_floor, self.u_max)

        u_des = u_pi
        if u_des > u_cap:
            u_des = u_cap
        if u_des < u_floor:
            u_des = u_floor

        # ---------- pH safety back-off (consent 4.0 .. 10.5 on true effluent) ----------
        # measurement is late, so keep a healthy margin on what we can see.
        if self.pH_f > 10.05:
            u_des = min(u_des, self.u - 2.0 * (self.pH_f - 10.05))
        elif self.pH_f > 9.85:
            u_des = min(u_des, self.u + 0.35)
        if self.pH_f < 4.9:
            u_des = max(u_des, self.u + 2.0 * (4.9 - self.pH_f))
        elif self.pH_f < 5.4:
            u_des = max(u_des, self.u - 0.35)

        u_des = self._clip(u_des)

        # ---------- rate limit ----------
        du = u_des - self.u
        if du > self.rate_lim:
            du = self.rate_lim
        elif du < -self.rate_lim:
            du = -self.rate_lim
        u_new = self._clip(self.u + du)

        # ---------- deadband to protect duty ----------
        if abs(u_new - self.u) < self.deadband:
            u_new = self.u

        # ---------- anti-windup: back-calculate integrator ----------
        self.I = u_new - P
        # keep integrator sane
        self.I = float(np.clip(self.I, self.u_min - 8.0, self.u_max + 8.0))

        self.u = u_new
        self.k += 1
        return np.array([self.u], dtype=float)