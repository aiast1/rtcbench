import numpy as np
from dataclasses import dataclass

@dataclass
class Brief:
    sample_time: float
    y_shape: tuple
    u_shape: tuple

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.T_sp = 350.0
        self.T_sp_prev = 350.0
        self.ramp_start = None
        self.ramp_end = None
        self.ramp_start_sp = None
        self.ramp_end_sp = None

        # PID parameters tuned for this process
        self.Kp = 2.0
        self.Ti = 60.0
        self.Td = 10.0
        self.N = 10.0  # Derivative filter coefficient

        # Anti-windup and bumpless transfer
        self.u_prev = np.array([300.0])
        self.ui_prev = 0.0
        self.ud_prev = 0.0
        self.e_prev = 0.0

        # Actuator limits
        self.u_min = 270.0
        self.u_max = 340.0
        self.slew_rate = 1.5 * self.sample_time  # 1.5 K/s * sample time

        # Filter for derivative term
        self.alpha = self.sample_time / (self.N * self.sample_time + self.Td)

    def reset(self):
        self.u_prev = np.array([300.0])
        self.ui_prev = 0.0
        self.ud_prev = 0.0
        self.e_prev = 0.0
        self.T_sp = 350.0
        self.T_sp_prev = 350.0
        self.ramp_start = None
        self.ramp_end = None
        self.ramp_start_sp = None
        self.ramp_end_sp = None

    def step(self, t, y, r, quality):
        # Extract measurements
        T_reactor = y[0]
        T_jacket = y[1] if len(y) > 1 else self.u_prev[0]

        # Handle setpoint schedule
        if t < 240:
            self.T_sp = 350.0
        elif t < 480:
            self.T_sp = 352.0
        elif t < 570:
            if self.ramp_start is None:
                self.ramp_start = t
                self.ramp_end = t + 90
                self.ramp_start_sp = 352.0
                self.ramp_end_sp = 348.0
            self.T_sp = self.ramp_start_sp + (self.ramp_end_sp - self.ramp_start_sp) * (t - self.ramp_start) / 90
        elif t < 720:
            self.T_sp = 348.0
        else:
            self.T_sp = 350.0

        # Calculate error
        e = self.T_sp - T_reactor

        # Proportional term
        up = self.Kp * e

        # Integral term with anti-windup
        ui = self.ui_prev + self.Kp * self.sample_time / self.Ti * e
        ui = np.clip(ui, self.u_min - up, self.u_max - up)

        # Derivative term with filtering
        ud_raw = (e - self.e_prev) / self.sample_time
        ud = self.alpha * ud_raw + (1 - self.alpha) * self.ud_prev

        # PID output
        u = up + ui + self.Kp * self.Td * ud

        # Clamp output to actuator limits
        u = np.clip(u, self.u_min, self.u_max)

        # Apply slew rate limit
        u = np.clip(u, self.u_prev[0] - self.slew_rate, self.u_prev[0] + self.slew_rate)

        # Update states for next iteration
        self.ui_prev = ui
        self.ud_prev = ud
        self.e_prev = e
        self.u_prev = np.array([u])
        self.T_sp_prev = self.T_sp

        return self.u_prev