import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.y_min = np.array([0.0])  # TT-401 min
        self.y_max = np.array([100.0])  # TT-401 max
        self.u_min = np.array([0.0])  # TIC-401 min
        self.u_max = np.array([100.0])  # TIC-401 max
        self.integral = np.zeros(1)  # integral state
        self.last_error = np.zeros(1)  # for derivative
        self.last_u = np.array([50.0])  # bumpless start
        self.last_t = None
        self.Kp = 0.4
        self.Ki = 0.015
        self.Kd = 0.0
        self.alpha = 0.3  # derivative filter coefficient
        self.derivative_filter = np.zeros(1)  # filtered derivative state
        self.y_last = None  # last good measurement
        self.max_integral = 50.0 / self.Ki  # clamp integral to prevent windup

    def reset(self):
        self.integral = np.zeros(1)
        self.last_error = np.zeros(1)
        self.last_u = np.array([50.0])
        self.last_t = None
        self.derivative_filter = np.zeros(1)
        self.y_last = None

    def step(self, t, y, r, quality):
        # Handle first call
        if self.last_t is None:
            self.last_t = t
            if quality[0] and not np.isnan(r[0]):
                self.y_last = y[0]
                error = r[0] - y[0]
            else:
                self.y_last = 50.0  # assume mid-scale if bad
                error = 0.0
            self.last_error = np.array([error])
            self.derivative_filter = np.array([0.0])
            return self.last_u.copy()

        # Compute error (only if measurement is good and setpoint is valid)
        if quality[0] and not np.isnan(r[0]):
            error = r[0] - y[0]
            self.y_last = y[0]  # update last good measurement
        else:
            error = 0.0  # no correction if bad measurement or no setpoint

        # Proportional term
        P = self.Kp * error

        # Integral term with anti-windup via clamping and back-calculation
        self.integral += error * self.sample_time
        # Clamp integral to prevent excessive windup
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        I = self.Ki * self.integral

        # Derivative term on measurement (to avoid derivative kick) with filtering
        if self.y_last is not None:
            # Use current y if good, else last good y for derivative calculation
            y_used = y[0] if quality[0] else self.y_last
            dt = t - self.last_t
            if dt > 1e-8:
                dy = (y_used - self.y_last) / dt
            else:
                dy = 0.0
            # First-order filter on derivative: tau = alpha * sample_time
            tau = self.alpha * self.sample_time
            if tau > 0:
                self.derivative_filter += (self.sample_time / tau) * (dy - self.derivative_filter)
            else:
                self.derivative_filter = np.array([dy])
            D = -self.Kd * self.derivative_filter  # negative because we differentiate measurement
            # Update y_last only if we used a new measurement
            if quality[0]:
                self.y_last = y[0]
        else:
            D = 0.0
            if self.y_last is None:
                self.y_last = y[0] if quality[0] else 50.0

        # PID output
        u_pid = P + I + D

        # Anti-windup: back-calculation if output saturates
        u_saturated = np.clip(u_pid, self.u_min[0], self.u_max[0])
        if u_saturated != u_pid:
            # Back-calculate integral to reduce windup
            self.integral += (u_saturated - u_pid) / (self.Ki * self.sample_time + 1e-8)
            # Re-clamp after back-calculation
            self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)

        u_final = u_saturated

        # Slew rate limiting: max 2% per step (reduced further to avoid stiction and duty limit issues)
        max_delta = 2.0  # % per 5s step
        delta = u_final - self.last_u[0]
        if abs(delta) > max_delta:
            u_final = self.last_u[0] + np.sign(delta) * max_delta

        # Ensure final output within hard limits
        u_final = np.clip(u_final, self.u_min[0], self.u_max[0])

        # Update state
        self.last_u = np.array([u_final])
        self.last_t = t
        self.last_error = np.array([error])

        return self.last_u.copy()