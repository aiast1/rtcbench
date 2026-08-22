import numpy as np


class Controller:
    """
    Decentralized PI controller (one loop per actuator/measurement pair,
    matching the physical pairing FCV-201->AI-101, FCV-202->AI-102,
    FCV-203->TI-103) with:
      - SIMC-style PI tuning derived from the nominal FOPDT model, detuned
        (tau_c = 2*L) for robustness to per-scenario parameter mismatch,
      - hard actuator saturation,
      - an explicit per-step rate limiter kept safely below the actuator
        duty-limit threshold to avoid any chattering/duty violation,
      - back-calculation anti-windup so the integrator always reflects the
        actually-applied (possibly clamped/rate-limited) command.

    No cross-loop decoupling is used: with independently-drawn gains/lags/
    delays per scenario, a nominal-model decoupler could easily destabilize
    an unlucky draw. Conservative decentralized loops with strong anti-windup
    and rate-limiting are the robust choice for the worst-case scoring rule.
    """

    def __init__(self, brief):
        self.dt = float(getattr(brief, "sample_time", 1.0))

        # Nominal diagonal FOPDT parameters (K_ii, L_ii, tau_ii) from the
        # model hint. Bottoms loop dead time floored to a small positive
        # value for robust tuning (nominal is ~0, but real draw will be >0).
        self.K = np.array([4.05, 5.72, 7.2])
        self.L = np.array([27.0, 14.0, 5.0])
        self.TAU = np.array([50.0, 60.0, 19.0])

        # SIMC PI tuning with tau_c = 2*L (robust / detuned choice)
        tau_c = 2.0 * self.L
        self.Kc = (1.0 / self.K) * (self.TAU / (self.L + tau_c))
        self.tau_I = np.minimum(self.TAU, 4.0 * (tau_c + self.L))

        # Actuator limits
        self.u_min = -0.5
        self.u_max = 0.5

        # Per-step rate limit, kept comfortably under the 0.0112 duty limit
        self.rate_max = 0.0095

        self.n = 3
        self.reset()

    def reset(self):
        self.u_prev = np.zeros(self.n)
        self.integral = np.zeros(self.n)
        self.y_last = np.zeros(self.n)
        self._started = False

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        quality = np.asarray(quality, dtype=bool)

        # Hold last-good value for stale/bad readings
        if not self._started:
            self.y_last = y.copy()
            self._started = True

        y_use = np.where(quality, y, self.y_last)
        self.y_last = np.where(quality, y, self.y_last)

        # All three channels are scored; guard against stray NaNs anyway
        r_use = np.where(np.isnan(r), 0.0, r)

        e = r_use - y_use

        # Update integral state (will be corrected below for anti-windup)
        integral_trial = self.integral + e * self.dt

        u_unclamped = self.Kc * e + (self.Kc / self.tau_I) * integral_trial

        # Hard actuator saturation
        u_sat = np.clip(u_unclamped, self.u_min, self.u_max)

        # Per-step rate limiting relative to previous applied command
        delta = u_sat - self.u_prev
        delta = np.clip(delta, -self.rate_max, self.rate_max)
        u_out = self.u_prev + delta
        u_out = np.clip(u_out, self.u_min, self.u_max)

        # Anti-windup: back-calculate the integral so that, given the
        # actually applied output, the PI equation is self-consistent.
        # This prevents windup whenever saturation or rate limiting bites.
        safe_Kc = np.where(np.abs(self.Kc) < 1e-9, 1e-9, self.Kc)
        integral_corrected = (u_out - self.Kc * e) * (self.tau_I / safe_Kc)
        self.integral = integral_corrected

        self.u_prev = u_out.copy()

        return u_out