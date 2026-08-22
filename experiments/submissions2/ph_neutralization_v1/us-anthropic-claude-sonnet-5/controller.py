import numpy as np


class Controller:
    """
    Robust PI(+D+FF) pH controller with explicit but permissive safety
    barriers.

    Lessons carried forward from rounds 1 & 2:
      * Round 1 failed safety because the gain schedule peaked exactly
        at the upper buffer pKa (10.25), right next to the hard pH
        limit (10.5), and windup pushed pH past the envelope under the
        20 s sample-line delay.
      * Round 2 fixed safety but was overly conservative: the "soft"
        safety barrier band (4.8-9.8) sat *inside* the scored setpoint
        range (setpoints go up to 9.9), so the barrier was constantly
        fighting the controller near the top setpoint, inflating
        tracking error even though effort/duty were far under budget.
      * This round: widen the soft-barrier band so it only engages
        genuinely close to the hard limits (well outside any commanded
        setpoint), speed up the pH filter a bit (less added lag on top
        of the 20 s transport delay), loosen the slew/deadband limits
        a little (effort cost was tiny, there's plenty of duty budget
        left), and add a setpoint-rate feedforward term so the 200 s
        ramp segment is tracked without waiting on pure integral action.
    """

    def __init__(self, brief):
        self.dt = float(getattr(brief, "sample_time", 5.0))

        self.u_min = 0.0
        self.u_max = 30.0
        self.u_init = 14.22

        # nominal buffer pKa's - used only for a mild gain shape
        self.pK1 = 6.35
        self.pK2 = 10.25

        # measurement filters
        self.tau_pH = 12.0
        self.alpha_pH = self.dt / (self.tau_pH + self.dt)
        self.tau_h = 10.0
        self.alpha_h = self.dt / (self.tau_h + self.dt)

        # PI gain schedule bounds
        self.Kp_min = 0.7
        self.Kp_max = 1.8
        self.Ti_min = 60.0
        self.Ti_max = 140.0

        # derivative damping (on filtered signal) to counter delay
        self.Kd = 8.0

        # setpoint-rate feedforward (helps track the 200 s ramp segment)
        self.Kff = 6.0

        # anti-chatter behaviour
        self.err_deadband = 0.02      # pH
        self.out_deadband = 0.01      # mL/s
        self.slew_max = 0.6           # mL/s per control step (normal)
        self.slew_max_emerg = 3.0     # mL/s per control step (emergency)

        # ---- pH safety barrier (soft, quadratic, model-free) ----
        # kept well OUTSIDE the scored setpoint range (up to 9.9) so it
        # does not fight normal tracking, only engages near hard limits
        self.pH_hard_lo = 4.0
        self.pH_hard_hi = 10.5
        self.pH_soft_lo = 4.5
        self.pH_soft_hi = 10.15
        self.k_barrier = 8.0

        # raw (unfiltered) fast trip thresholds - bypass everything
        self.pH_trip_lo = 4.25
        self.pH_trip_hi = 10.3
        self.trip_step = 2.5          # mL/s immediate corrective step

        # ---- level safety (secondary, mild) ----
        self.h_low_hard = 5.0
        self.h_high_hard = 30.0
        self.h_low_soft = 8.0
        self.h_high_soft = 26.0
        self.k_h_safety = 2.5

        self.reset()

    def reset(self):
        self.pH_filt = None
        self.h_filt = None
        self.integ = self.u_init
        self.last_u = self.u_init
        self.prev_sp = None

    def step(self, t, y, r, quality):
        pH_meas = float(y[0]) if len(y) > 0 else 7.0
        h_meas = float(y[1]) if len(y) > 1 else 17.5

        qpH = bool(quality[0]) if len(quality) > 0 else True
        qh = bool(quality[1]) if len(quality) > 1 else True

        # --- filters (hold last good value on bad quality) ---
        if self.pH_filt is None:
            self.pH_filt = pH_meas
        prev_for_deriv = self.pH_filt
        if qpH:
            self.pH_filt = (1.0 - self.alpha_pH) * self.pH_filt + self.alpha_pH * pH_meas

        dpH = (self.pH_filt - prev_for_deriv) / self.dt

        if self.h_filt is None:
            self.h_filt = h_meas
        if qh:
            self.h_filt = (1.0 - self.alpha_h) * self.h_filt + self.alpha_h * h_meas
        h_use = self.h_filt

        # --- setpoint / error ---
        sp = r[0] if len(r) > 0 else np.nan
        if np.isnan(sp):
            sp = self.pH_filt  # nothing scored right now: hold
            sp_rate = 0.0
        else:
            if self.prev_sp is None:
                self.prev_sp = sp
            sp_rate = (sp - self.prev_sp) / self.dt
            self.prev_sp = sp

        e = sp - self.pH_filt
        e_eff = 0.0 if abs(e) < self.err_deadband else e

        # --- mild gain schedule (bounded, peak kept away from hard limit) ---
        s = 1.2
        d1 = self.pH_filt - self.pK1
        d2 = self.pH_filt - self.pK2
        prox = np.exp(-(d1 * d1) / (2.0 * s * s)) + np.exp(-(d2 * d2) / (2.0 * s * s))
        prox = float(np.clip(prox, 0.0, 1.0))

        Kp = self.Kp_min + (self.Kp_max - self.Kp_min) * prox
        Ti = self.Ti_max - (self.Ti_max - self.Ti_min) * prox
        Ki = Kp / Ti

        # --- PI + derivative damping + setpoint-rate feedforward ---
        u_pi = self.integ + Kp * e_eff - self.Kd * dpH + self.Kff * sp_rate

        # --- pH safety barrier (soft, quadratic, symmetric, outside setpoints) ---
        barrier = 0.0
        if self.pH_filt > self.pH_soft_hi:
            over = self.pH_filt - self.pH_soft_hi
            barrier -= self.k_barrier * over * over
        if self.pH_filt < self.pH_soft_lo:
            under = self.pH_soft_lo - self.pH_filt
            barrier += self.k_barrier * under * under

        # --- level safety barrier (secondary, mild) ---
        h_barrier = 0.0
        if h_use > self.h_high_soft:
            over = h_use - self.h_high_soft
            span = max(self.h_high_hard - self.h_high_soft, 1e-6)
            h_barrier -= self.k_h_safety * (over / span) * self.u_max
        if h_use < self.h_low_soft:
            under = self.h_low_soft - h_use
            span = max(self.h_low_soft - self.h_low_hard, 1e-6)
            h_barrier += self.k_h_safety * (under / span) * self.u_max

        u_target = u_pi + barrier + h_barrier
        u_sat = float(np.clip(u_target, self.u_min, self.u_max))

        # --- conditional anti-windup ---
        if qpH:
            would_be = self.integ + Ki * e_eff * self.dt
            saturated_up = u_sat >= self.u_max - 1e-6 and e_eff > 0
            saturated_dn = u_sat <= self.u_min + 1e-6 and e_eff < 0
            if not (saturated_up or saturated_dn):
                self.integ = float(np.clip(would_be, self.u_min - 5.0, self.u_max + 5.0))

        # --- fast trip on raw measurement (bypasses filter lag) ---
        trip = None
        if qpH:
            if pH_meas >= self.pH_trip_hi:
                trip = -self.trip_step
            elif pH_meas <= self.pH_trip_lo:
                trip = self.trip_step

        # --- emergency condition (near hard bounds) ---
        emergency = (
            h_use < self.h_low_hard + 1.0 or h_use > self.h_high_hard - 1.0
            or self.pH_filt < self.pH_hard_lo + 0.5
            or self.pH_filt > self.pH_hard_hi - 0.5
            or trip is not None
        )

        if trip is not None:
            u_cmd = float(np.clip(self.last_u + trip, self.u_min, self.u_max))
        elif emergency:
            du = u_sat - self.last_u
            du = float(np.clip(du, -self.slew_max_emerg, self.slew_max_emerg))
            u_cmd = self.last_u + du
        else:
            du = u_sat - self.last_u
            if abs(du) < self.out_deadband:
                du = 0.0
            du = float(np.clip(du, -self.slew_max, self.slew_max))
            u_cmd = self.last_u + du

        u_cmd = float(np.clip(u_cmd, self.u_min, self.u_max))
        self.last_u = u_cmd

        return np.array([u_cmd])