import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.n_act = 3
        self.n_meas = 3

        # More aggressive tuning for better tracking while maintaining robustness
        self.Kp = np.diag([0.15, 0.15, 0.08])
        self.Ki = np.diag([0.003, 0.003, 0.0015])
        self.Kd = np.diag([1.5, 1.5, 0.8])

        # Anti-windup tracking
        self.u_prev = np.zeros(self.n_act)
        self.ui_prev = np.zeros(self.n_act)
        self.ud_prev = np.zeros(self.n_act)
        self.e_prev = np.zeros(self.n_meas)
        self.y_prev = np.zeros(self.n_meas)

        # Filter for derivative action
        self.tau_f = 5.0
        self.alpha = self.sample_time / (self.tau_f + self.sample_time)

        # Safety limits
        self.u_min = -0.5 * np.ones(self.n_act)
        self.u_max = 0.5 * np.ones(self.n_act)
        self.y_min = np.array([-np.inf, -np.inf, -0.5])

        # Bumpless transfer
        self.initial_actuators = np.array([0.0, 0.0, 0.0])

        # Setpoint handling with smoother transitions
        self.setpoint_schedule = [
            (0, [0.0, 0.0, 0.0]),
            (120, [0.4, -0.3, 0.0]),
            (350, [-0.25, 0.25, 0.12]),
            (440, [-0.25, 0.25, 0.12]),
            (600, [0.0, 0.0, 0.0])
        ]
        self.current_setpoint = np.zeros(self.n_meas)

        # Feedforward compensation for disturbances
        self.disturbance_model = np.array([
            [1.2, 1.44],
            [1.52, 1.83],
            [1.14, 1.26]
        ])

    def reset(self):
        self.u_prev = self.initial_actuators.copy()
        self.ui_prev = np.zeros(self.n_act)
        self.ud_prev = np.zeros(self.n_act)
        self.e_prev = np.zeros(self.n_meas)
        self.y_prev = np.zeros(self.n_meas)
        self.current_setpoint = np.zeros(self.n_meas)

    def step(self, t, y, r, quality):
        # Update current setpoint based on schedule
        for time, sp in reversed(self.setpoint_schedule):
            if t >= time:
                if time == 350 and t < 440:
                    ramp_progress = (t - 350) / 90
                    self.current_setpoint = np.array([
                        0.4 + (-0.25 - 0.4) * ramp_progress,
                        -0.3 + (0.25 + 0.3) * ramp_progress,
                        0.0 + (0.12 - 0.0) * ramp_progress
                    ])
                else:
                    self.current_setpoint = np.array(sp)
                break

        # Use provided setpoint where not nan, otherwise use current
        sp = np.where(np.isnan(r), self.current_setpoint, r)

        # Calculate error
        e = sp - y

        # PID calculation with anti-windup
        up = self.Kp @ e
        ui = self.ui_prev + self.Ki @ e * self.sample_time
        ud_filtered = self.alpha * (self.Kd @ (e - self.e_prev) / self.sample_time) + (1 - self.alpha) * self.ud_prev

        # Feedforward compensation (assuming disturbances d are available)
        # Since we don't have access to disturbances, we'll skip this part
        # u_ff = -self.disturbance_model @ d
        u = up + ui + ud_filtered  # + u_ff

        # Anti-windup: clamp integral term when output saturates
        u_saturated = np.clip(u, self.u_min, self.u_max)
        ui = ui + (u_saturated - u)  # Back-calculate integral term

        # Safety check on bottoms temperature
        if y[2] < -0.5:
            u_saturated = self.u_prev.copy()  # Freeze actuators if safety violated
            ui = self.ui_prev  # Prevent integral windup

        # Update states
        self.ui_prev = ui
        self.ud_prev = ud_filtered
        self.e_prev = e.copy()
        self.u_prev = u_saturated.copy()
        self.y_prev = y.copy()

        return u_saturated