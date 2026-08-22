import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.nu = 3  # 3 actuators
        self.ny = 3  # 3 measurements
        self.u_limits = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])

        # Conservative PID tuning
        self.Kp = np.diag([0.03, 0.03, 0.05])
        self.Ti = np.diag([120.0, 120.0, 60.0])
        self.Td = np.diag([3.0, 3.0, 1.5])

        # State variables
        self.u_prev = np.zeros(self.nu)
        self.ui_prev = np.zeros(self.nu)
        self.e_prev = np.zeros(self.ny)
        self.de_prev = np.zeros(self.ny)
        self.y_prev = np.zeros(self.ny)

        # Setpoint tracking
        self.r = np.zeros(self.ny)
        self.setpoint_changes = [
            (0, [0.0, 0.0, 0.0]),
            (120, [0.2, -0.15, 0.0]),
            (350, [-0.1, 0.1, 0.05]),
            (600, [0.0, 0.0, 0.0])
        ]
        self.ramp_durations = [0, 0, 90, 0]
        self.next_setpoint_idx = 0
        self.ramp_start_time = None
        self.ramp_duration = 0
        self.ramp_start_sp = None
        self.ramp_end_sp = None

        # Dead-time compensation
        self.dead_time = np.array([[27, 28, 27], [18, 14, 15], [20, 22, 0]])
        self.max_dead_time = int(np.max(self.dead_time) / self.sample_time) + 1
        self.u_history = np.zeros((self.max_dead_time, self.nu))

    def reset(self):
        self.u_prev = np.zeros(self.nu)
        self.ui_prev = np.zeros(self.nu)
        self.e_prev = np.zeros(self.ny)
        self.de_prev = np.zeros(self.ny)
        self.y_prev = np.zeros(self.ny)
        self.r = np.zeros(self.ny)
        self.next_setpoint_idx = 0
        self.ramp_start_time = None
        self.u_history = np.zeros((self.max_dead_time, self.nu))

    def step(self, t, y, r, quality):
        # Update setpoints
        self._update_setpoints(t)

        # Safety check (TI-103 must stay above -0.5)
        if y[2] < -0.5:
            return self.u_prev  # Emergency hold

        # Filter measurements
        y_filtered = self._filter_measurements(y, quality)

        # Calculate errors
        e = self.r - y_filtered
        de = (e - self.e_prev) / self.sample_time

        # PID calculation
        u_p = self.Kp @ e
        u_i = self.ui_prev + (self.Kp / self.Ti) @ (e * self.sample_time)
        u_d = (self.Kp * self.Td) @ de

        u = u_p + u_i + u_d

        # Anti-windup
        u_saturated = np.clip(u, self.u_limits[0], self.u_limits[1])
        ui = u_i + (u_saturated - u)  # Back-calculation
        ui = np.clip(ui, self.u_limits[0], self.u_limits[1])
        u = u_p + ui + u_d

        # Dead-time compensation
        u_compensated = self._compensate_dead_time(u)

        # Slew rate limiting
        u_out = np.clip(
            u_compensated,
            self.u_prev - 0.01,  # Conservative slew limit
            self.u_prev + 0.01
        )
        u_out = np.clip(u_out, self.u_limits[0], self.u_limits[1])

        # Update states
        self.u_prev = u_out
        self.ui_prev = ui
        self.e_prev = e
        self.de_prev = de
        self.y_prev = y_filtered

        # Update history
        self.u_history = np.roll(self.u_history, -1, axis=0)
        self.u_history[-1, :] = u_out

        return u_out

    def _update_setpoints(self, t):
        if self.next_setpoint_idx < len(self.setpoint_changes):
            change_time, sp = self.setpoint_changes[self.next_setpoint_idx]
            ramp_duration = self.ramp_durations[self.next_setpoint_idx]
            if t >= change_time:
                if ramp_duration > 0:
                    if self.ramp_start_time is None:
                        self.ramp_start_time = t
                        self.ramp_duration = ramp_duration
                        self.ramp_start_sp = self.r.copy()
                        self.ramp_end_sp = np.array(sp)
                    progress = min((t - self.ramp_start_time) / self.ramp_duration, 1.0)
                    self.r = self.ramp_start_sp + (self.ramp_end_sp - self.ramp_start_sp) * progress
                else:
                    self.r = np.array(sp)
                if t >= change_time + ramp_duration:
                    self.next_setpoint_idx += 1
                    self.ramp_start_time = None

    def _filter_measurements(self, y, quality):
        y_filtered = y.copy()
        for i in range(self.ny):
            if not quality[i]:
                y_filtered[i] = self.y_prev[i]
            else:
                alpha = 0.2
                y_filtered[i] = alpha * y[i] + (1 - alpha) * self.y_prev[i]
        return y_filtered

    def _compensate_dead_time(self, u):
        u_compensated = u.copy()
        for i in range(self.ny):
            for j in range(self.nu):
                dt = int(self.dead_time[i, j] / self.sample_time)
                if dt > 0 and dt < self.max_dead_time:
                    u_compensated[j] += (self.Kp[i, j] * (u[j] - self.u_history[-dt-1, j]))
        return u_compensated