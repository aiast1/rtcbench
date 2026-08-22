import numpy as np


class Controller:
    """
    Controller for the unstable middle-steady-state CSTR.

    Design summary
    ---------------
    The reactor sits on the *unstable* (saddle-type) middle steady state.
    A linearisation of the nominal model around T=350 K shows a single
    unstable real pole with time constant of order 20 s, and shows that
    a simple proportional output-feedback on reactor temperature is
    *analytically* enough to stabilise it once the proportional gain
    exceeds roughly 1.1 K(coolant)/K(reactor).  In practice measurement
    noise/quantisation/dropouts, transport delay, actuator slew and
    valve stiction eat a large part of that margin, so the feedback
    gain is pushed well above the analytic minimum and complemented
    with:

      * a fixed (setpoint-only) nonlinear steady-state feed-forward for
        the coolant temperature, obtained by inverting the nominal
        energy/species balance -- this removes almost all of the
        static offset so the feedback loop only has to fight
        disturbances/mismatch, not do the whole job;
      * a nonlinear gain boost that increases proportional action once
        the error grows beyond a modest band, to arrest large
        excursions quickly without over-driving the loop near the
        setpoint (which would risk actuator chatter / duty violation);
      * a hard, last-resort safety clamp that commands full cooling /
        full heating if the reactor gets close to a trip, regardless
        of what the PID term says;
      * back-calculation anti-windup so the integrator can never wind
        up beyond what the (rate-limited, saturated) actuator can
        actually deliver;
      * light filtering + light derivative action (derivative acts on
        the filtered measurement, not on the error, to avoid kick) --
        enough to add damping without amplifying noise into a
        duty-limit violation;
      * output rate limiting matched to (just under) the coolant
        skid's slew capability, plus a tiny deadband to avoid
        chattering.
    """

    def __init__(self, brief):
        self.dt = float(getattr(brief, "sample_time", 3.0))

        # ---- Nominal process model (from commissioning brief) ----
        self.q = 100.0
        self.V = 100.0
        self.rho = 1000.0
        self.Cp = 0.239
        self.dH = -50000.0
        self.EoverR = 8750.0
        self.k0 = 72000000000.0
        self.UA = 50000.0
        self.Caf = 1.0
        self.Tf = 350.0

        # ---- Actuator limits ----
        self.u_min = 270.0
        self.u_max = 340.0

        # ---- Core feedback gains ----
        self.Kp = 14.0          # K(coolant) per K(error)  -- large margin
        self.Ki = 0.02          # 1/s
        self.Kd = 2.0           # K per (K/s), lightly used

        # ---- Nonlinear gain boost for large excursions ----
        self.boost_band = 6.0   # K, error beyond this gets extra gain
        self.Kp_boost = 3.0     # extra K/K beyond the band

        # ---- Filters ----
        self.tau_meas = 3.0     # measurement low-pass filter (s)
        self.tau_deriv = 5.0    # derivative low-pass filter (s)

        # ---- Output shaping ----
        self.rate_limit = 1.45 * self.dt   # K per control step (< skid slew*dt)
        self.deadband = 0.05               # K, ignore tiny moves (chatter guard)

        # integral clamp (bounds Ki*integral contribution to +/-40 K)
        self.integral_limit = 40.0 / max(self.Ki, 1e-9)

        # ---- Emergency (last-resort) safety thresholds ----
        self.hard_hi = 445.0    # force full cooling above this (trip is 470)
        self.hard_lo = 308.0    # force full heating below this (trip is 300)
        self.soft_hi = 400.0    # start extra-aggressive cooling boost
        self.soft_lo = 320.0    # start extra-aggressive heating boost

        self.reset()

    def reset(self):
        self.integral = 0.0
        self.filt_meas = None
        self.prev_filt = None
        self.filt_deriv = 0.0
        self.u_prev = 300.0
        self.last_good_meas = 350.0

    # ---------------------------------------------------------------
    def _feedforward(self, Tsp):
        """Nominal-model steady-state coolant temperature for a target
        reactor temperature Tsp (energy + component balance), evaluated
        ONLY at the setpoint (never at the live measurement -- doing the
        latter would just retrace the unstable open-loop manifold)."""
        Tsp = float(np.clip(Tsp, 300.0, 470.0))
        k = self.k0 * np.exp(-self.EoverR / Tsp)
        Ca = self.q * self.Caf / (self.q + self.V * k)
        gen = (-self.dH) * self.V * k * Ca
        conv = self.q * self.rho * self.Cp * (self.Tf - Tsp)
        Tc = Tsp - (conv + gen) / self.UA
        return float(np.clip(Tc, self.u_min, self.u_max))

    # ---------------------------------------------------------------
    def step(self, t, y, r, quality):
        dt = self.dt

        # --- measurement handling (quality / dropouts) ---
        meas = float(y[0])
        good = True
        if quality is not None and len(quality) > 0:
            good = bool(quality[0])
        if (not good) or (not np.isfinite(meas)):
            meas = self.last_good_meas
        else:
            self.last_good_meas = meas

        raw_meas = meas  # unfiltered (but dropout-protected) value, for fast safety checks

        # --- measurement low-pass filter ---
        if self.filt_meas is None:
            self.filt_meas = meas
        alpha = dt / (self.tau_meas + dt)
        self.filt_meas += alpha * (meas - self.filt_meas)

        # --- setpoint ---
        sp = float(r[0]) if np.isfinite(r[0]) else 350.0

        # --- feedforward from nominal steady-state model (setpoint only) ---
        Tc_ff = self._feedforward(sp)

        # --- error ---
        error = sp - self.filt_meas

        # --- proportional with nonlinear gain boost for large deviations ---
        P = self.Kp * error
        excess = abs(error) - self.boost_band
        if excess > 0.0:
            P += self.Kp_boost * excess * np.sign(error)

        # --- extra continuous boost near soft safety thresholds ---
        if self.filt_meas > self.soft_hi:
            P -= self.Kp_boost * (self.filt_meas - self.soft_hi)
        elif self.filt_meas < self.soft_lo:
            P += self.Kp_boost * (self.soft_lo - self.filt_meas)

        # --- derivative on filtered measurement (avoid kick), filtered again ---
        if self.prev_filt is None:
            self.prev_filt = self.filt_meas
        raw_deriv = (self.filt_meas - self.prev_filt) / dt
        self.prev_filt = self.filt_meas
        beta = dt / (self.tau_deriv + dt)
        self.filt_deriv += beta * (raw_deriv - self.filt_deriv)
        D = -self.Kd * self.filt_deriv

        # --- integral term (anti-windup applied after saturation, below) ---
        I_term = self.Ki * self.integral

        u_unclamped = Tc_ff + P + I_term + D

        # --- hard, last-resort safety override ---
        if raw_meas >= self.hard_hi or self.filt_meas >= self.hard_hi:
            u_unclamped = self.u_min
        elif raw_meas <= self.hard_lo or self.filt_meas <= self.hard_lo:
            u_unclamped = self.u_max

        # --- saturate to actuator hard limits ---
        u_cmd = float(np.clip(u_unclamped, self.u_min, self.u_max))

        # --- rate limiting (respect skid slew) ---
        delta = u_cmd - self.u_prev
        if delta > self.rate_limit:
            delta = self.rate_limit
        elif delta < -self.rate_limit:
            delta = -self.rate_limit
        u_new = self.u_prev + delta

        # --- deadband to avoid chattering (duty-limit safety) ---
        if abs(u_new - self.u_prev) < self.deadband:
            u_new = self.u_prev

        u_new = float(np.clip(u_new, self.u_min, self.u_max))

        # --- anti-windup via back-calculation: force the integral to be
        #     exactly consistent with what the actuator actually did, so
        #     it can never wind up beyond what is deliverable. ---
        if self.Ki > 1e-12:
            self.integral = (u_new - Tc_ff - P - D) / self.Ki
            self.integral = float(np.clip(self.integral,
                                           -self.integral_limit,
                                           self.integral_limit))
        else:
            self.integral += error * dt

        self.u_prev = u_new

        return np.array([u_new])