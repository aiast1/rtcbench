import numpy as np


class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.dt = float(brief.sample_time)
        self.u_min = 270.0
        self.u_max = 340.0
        self.u_start = 300.0
        self.T_hi = 470.0
        self.T_lo = 300.0
        self.slew = 1.5 * self.dt  # K per step

        # Gentle PID. The plant is open-loop unstable so we need enough
        # proportional gain to stabilize, but effort/duty and noise limit us.
        self.Kp = 4.0
        self.Ki = 0.12
        self.Kd = 1.2

        self.reset()

    def reset(self):
        self.integ = 0.0
        self.u = self.u_start
        self.prev_T = None
        self.T_filt = None
        self.d_filt = 0.0
        self.last_good_T = None

    def step(self, t, y, r, quality):
        T = float(y[0])
        q_T = bool(quality[0]) if quality is not None and len(quality) > 0 else True

        sp = r[0]
        if sp is None or (isinstance(sp, float) and np.isnan(sp)):
            sp = 350.0
        sp = float(sp)

        # Handle bad/stale reactor reading: hold last good value.
        if not q_T or not np.isfinite(T):
            if self.last_good_T is not None:
                T = self.last_good_T
            else:
                T = sp
        else:
            self.last_good_T = T

        # Filter the measurement (noisy, delayed) - moderate low pass.
        if self.T_filt is None:
            self.T_filt = T
        alpha = 0.35
        self.T_filt = (1 - alpha) * self.T_filt + alpha * T
        Tf = self.T_filt

        e = Tf - sp

        # Derivative on filtered measurement.
        if self.prev_T is None:
            deriv = 0.0
        else:
            deriv = (Tf - self.prev_T) / self.dt
        self.prev_T = Tf
        beta = 0.4
        self.d_filt = (1 - beta) * self.d_filt + beta * deriv

        bias = self.u_start

        denom = self.Ki

        # Compute unsaturated output using current integral.
        # Physics: higher coolant temp -> hotter reactor. If reactor is too
        # hot (e>0) we cool -> lower coolant. So u = bias - Kp*e - Ki*I - Kd*d.
        def compute(integ):
            return bias - self.Kp * e - self.Ki * integ - self.Kd * self.d_filt

        u_trial = compute(self.integ + e * self.dt)

        # Conditional integration (anti-windup): only integrate if not pushing
        # further into saturation.
        integrate = True
        if u_trial >= self.u_max and e < 0:
            integrate = False
        if u_trial <= self.u_min and e > 0:
            integrate = False
        if integrate:
            self.integ += e * self.dt

        u_unsat = compute(self.integ)
        u_clamped = min(self.u_max, max(self.u_min, u_unsat))

        # Back-calculate integral to keep it consistent when saturated.
        if u_clamped != u_unsat and denom != 0.0:
            self.integ = (bias - self.Kp * e - self.Kd * self.d_filt - u_clamped) / denom

        u_target = u_clamped

        # Slew limit.
        du = u_target - self.u
        if du > self.slew:
            du = self.slew
        elif du < -self.slew:
            du = -self.slew

        # Deadband to avoid chatter / duty breach. Larger band to control duty.
        # Only suppress very small moves.
        if abs(du) < 0.15:
            du = 0.0

        self.u = self.u + du
        self.u = min(self.u_max, max(self.u_min, self.u))

        return np.array([self.u], dtype=float)