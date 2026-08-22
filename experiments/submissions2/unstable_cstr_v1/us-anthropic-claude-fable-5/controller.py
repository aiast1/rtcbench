"""
Controller for unstable_cstr_v1 (open-loop unstable middle steady state).

Round-3 revision: REVERT to the round-1 design and tuning.

Round 1 scored mean 0.00509 (safe, no duty breach). Round 2's changes
(setpoint-weighted P with a lagged reference, softer filter gains, less
derivative) more than doubled cost on every seed - the softened
proportional/derivative action let the unstable pole express itself, so
both tracking AND travel got worse (the loop had to work harder chasing
larger excursions). The lesson: on an open-loop unstable plant, crisp
stabilisation is what minimises travel, not gentler gains.

Design (as in round 1):
  * PID with Kp = 2.8, Ti = 55 s, Td = 8 s - well above the stabilising
    minimum gain, tolerant of the sensor delay and 3 s sample time,
  * derivative on the measurement only, taken from an alpha-beta
    (position/velocity) tracker that coasts through dropped samples and
    clamps spike innovations,
  * back-calculation anti-windup consistent with saturation AND the
    self-imposed rate limit matching the skid slew (1.5 K/s),
  * small deadband (below the 0.15 K valve stiction) to suppress
    noise-driven travel and respect the actuator duty limit,
  * hard safety latches: full cooling above 420 K (vessel relieves at
    470), full heating below 312 K (product drops out below 300), with
    hysteresis on release.
"""

import numpy as np


class Controller:
    def __init__(self, brief):
        # Sample time
        try:
            self.dt = float(brief.sample_time)
        except Exception:
            self.dt = 3.0
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            self.dt = 3.0

        # Actuator limits (coolant supply temperature)
        self.u_min = 270.0
        self.u_max = 340.0
        self.u_nom = 300.0          # bumpless start value
        self.slew = 1.5             # K/s skid slew -> our own rate limit
        self.deadband = 0.06        # K, below valve stiction (~0.15 K)

        # PID tuning (error in K reactor -> K coolant)  [round-1 values]
        self.Kp = 2.8
        self.Ti = 55.0              # s
        self.Td = 8.0               # s
        self.Kd = self.Kp * self.Td # derivative gain on dT/dt
        self.Taw = 12.0             # s, anti-windup back-calculation

        # Alpha-beta measurement filter (position/velocity tracker)
        self.alpha = 0.55
        self.beta = 0.12
        self.innov_clip = 8.0       # K, spike rejection
        self.vel_clip = 3.0         # K/s, physically plausible bound

        # Safety override thresholds on the temperature estimate
        self.hot_trip = 420.0       # force full cooling above this
        self.hot_release = 385.0
        self.cold_trip = 312.0      # force full heating below this
        self.cold_release = 322.0

        self.reset()

    def reset(self):
        self.I = self.u_nom         # integrator holds the output bias
        self.That = None            # filtered temperature estimate
        self.vel = 0.0              # filtered dT/dt estimate (K/s)
        self.u_prev = self.u_nom
        self.sp = 350.0
        self.hot_latch = False
        self.cold_latch = False
        self.bad_count = 0

    def step(self, t, y, r, quality):
        dt = self.dt

        # ---------- setpoint (hold last good value if nan) ----------
        try:
            if r is not None and np.isfinite(r[0]):
                self.sp = float(r[0])
        except Exception:
            pass

        # ---------- measurement validation ----------
        try:
            Tm = float(y[0])
        except Exception:
            Tm = np.nan
        good = np.isfinite(Tm) and (280.0 <= Tm <= 480.0)
        try:
            good = good and bool(np.asarray(quality).ravel()[0])
        except Exception:
            pass

        # ---------- alpha-beta filter: smooth T and dT/dt ----------
        if self.That is None:
            self.That = Tm if good else self.sp
            self.vel = 0.0

        Tpred = self.That + self.vel * dt
        if good:
            resid = Tm - Tpred
            # innovation clamp: reject single-sample spikes
            if resid > self.innov_clip:
                resid = self.innov_clip
            elif resid < -self.innov_clip:
                resid = -self.innov_clip
            self.That = Tpred + self.alpha * resid
            self.vel = self.vel + self.beta * resid / dt
            self.bad_count = 0
        else:
            # coast on the model prediction, gently bleed velocity
            self.That = Tpred
            self.vel *= 0.9
            self.bad_count += 1

        # keep estimates physical
        self.That = min(max(self.That, 280.0), 480.0)
        self.vel = min(max(self.vel, -self.vel_clip), self.vel_clip)

        # ---------- PID (derivative on measurement) ----------
        e = self.sp - self.That
        u_raw = self.I + self.Kp * e - self.Kd * self.vel

        # saturation
        u = min(max(u_raw, self.u_min), self.u_max)

        # rate limit to what the skid can actually deliver
        du_max = self.slew * dt
        lo = self.u_prev - du_max
        hi = self.u_prev + du_max
        u = min(max(u, lo), hi)

        # stiction-aware deadband: don't nibble at the valve
        if abs(u - self.u_prev) < self.deadband:
            u = self.u_prev

        # ---------- safety overrides (with hysteresis) ----------
        if self.hot_latch:
            if self.That < self.hot_release:
                self.hot_latch = False
        elif self.That > self.hot_trip:
            self.hot_latch = True

        if self.cold_latch:
            if self.That > self.cold_release:
                self.cold_latch = False
        elif self.That < self.cold_trip:
            self.cold_latch = True

        if self.hot_latch:
            u = self.u_min          # full cooling, skid slews on its own
        elif self.cold_latch:
            u = self.u_max          # full heating

        # ---------- integrator with back-calculation anti-windup ----------
        self.I += (self.Kp * dt / self.Ti) * e + (dt / self.Taw) * (u - u_raw)
        self.I = min(max(self.I, self.u_min - 5.0), self.u_max + 5.0)

        self.u_prev = u
        return np.array([u], dtype=float)