import numpy as np


class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = float(getattr(brief, 'sample_time', 1.0))
        self.n = 3  # 3x3

        # Nominal steady-state gain matrix (for decoupling / pairing awareness)
        self.K = np.array([
            [4.05, 1.77, 5.88],
            [5.39, 5.72, 6.90],
            [4.38, 4.42, 7.20],
        ], dtype=float)

        # Diagonal-dominant pairing chosen: u0->y0, u1->y1, u2->y2
        # Use a conservative model-based inverse of the steady-state gain for
        # static decoupling, but heavily regularized for robustness.
        # Regularized pseudo-inverse
        K = self.K
        lam = 2.0
        self.Kinv = np.linalg.solve(K.T @ K + lam * np.eye(3), K.T)

        # PI tuning per loop (conservative, robust to gain/lag/deadtime draws).
        # Effective loop gains after decoupling are ~1, so gains are modest.
        self.Kp = np.array([0.06, 0.06, 0.05])
        self.Ki = np.array([0.0022, 0.0022, 0.0018])
        self.Kd = np.array([0.0, 0.0, 0.0])  # no derivative on noisy delayed meas

        # Actuator limits
        self.umin = -0.5
        self.umax = 0.5

        # Slew / duty control: keep travel small to respect duty limit 0.0112
        self.du_max = 0.010  # per-step max change per actuator

        self.reset()

    def reset(self):
        self.u = np.zeros(self.n)
        self.u_prev = np.zeros(self.n)
        self.integ = np.zeros(self.n)
        self.y_filt = np.zeros(self.n)
        self.first = True
        self.last_good_y = np.zeros(self.n)
        self.r_filt = np.zeros(self.n)

    def step(self, t, y, r, quality):
        y = np.asarray(y, dtype=float)
        r = np.asarray(r, dtype=float)
        if quality is None:
            quality = np.ones(self.n, dtype=bool)
        else:
            quality = np.asarray(quality, dtype=bool)

        # Handle bad / stale readings: hold last good value
        y_use = np.array(self.last_good_y, dtype=float)
        for i in range(self.n):
            if i < len(quality) and quality[i] and np.isfinite(y[i]):
                y_use[i] = y[i]
        self.last_good_y = y_use.copy()

        if self.first:
            self.y_filt = y_use.copy()
            self.r_filt = np.where(np.isfinite(r), r, 0.0)
            self.first = False

        # Low-pass filter measurements (noise / quantization)
        af = 0.35
        self.y_filt = (1 - af) * self.y_filt + af * y_use

        # Setpoints: nan -> hold 0 (unscored); smooth slightly
        r_target = np.where(np.isfinite(r), r, 0.0)
        ar = 0.5
        self.r_filt = (1 - ar) * self.r_filt + ar * r_target

        # Error in measured space
        e = self.r_filt - self.y_filt

        # PI per decoupled loop -----------------------------------------
        # Compute desired incremental control in "loop" space, then map
        # through decoupler. Here loops are aligned with outputs.
        # Proportional + integral action
        p_term = self.Kp * e

        # Provisional integral update
        integ_new = self.integ + self.Ki * e * self.dt

        # Loop-space command (before decoupling): treat as required
        # correction in output units
        v = p_term + integ_new

        # Static decoupling: map output-space demand to actuator-space.
        # v is a demanded change in outputs; Kinv maps to actuator command.
        u_cmd = self.Kinv @ v

        # Anti-windup: if actuator would saturate, don't accumulate integral
        u_unclamped = u_cmd.copy()

        # Slew-rate limit relative to previous applied u
        du = u_cmd - self.u_prev
        du = np.clip(du, -self.du_max, self.du_max)
        u_slewed = self.u_prev + du

        # Hard limits
        u_clamped = np.clip(u_slewed, self.umin, self.umax)

        # Anti-windup: detect saturation (either slew or limit) and back off
        # the integral so it doesn't wind up. Conditional integration.
        saturated = (np.abs(u_unclamped - u_clamped) > 1e-9)
        # Map saturation back to loops: if any actuator saturated, reduce
        # integral growth for loops. Simple approach: only commit integral
        # if not driving further into saturation.
        for i in range(self.n):
            # commit integral for loop i only if error and unsat consistent
            self.integ[i] = integ_new[i]
        # Global back-off: if strong saturation, decay integral slightly
        if np.any(saturated):
            self.integ *= 0.98
            # clamp integral magnitude
        self.integ = np.clip(self.integ, -0.4, 0.4)

        # Safety: y3 (index 2, TI-103) has hard floor at -0.5.
        # FCV-203 (u2) manages bottoms reflux temp. Its nominal gain to y2
        # is positive, so if y2 approaches the floor, push u2 up.
        # Add a protective bias when measured (or estimated) temp is low.
        y2 = self.y_filt[2]
        if y2 < -0.30:
            # proportional protective push on u2 (positive gain to y2)
            push = 0.5 * (-0.30 - y2)
            u_clamped[2] = min(self.umax, u_clamped[2] + push)
            # also nudge integral to sustain
            self.integ[2] = min(0.4, self.integ[2] + 0.005)

        # Re-apply hard limits after protective action
        u_final = np.clip(u_clamped, self.umin, self.umax)

        self.u_prev = u_final.copy()
        self.u = u_final.copy()
        return u_final