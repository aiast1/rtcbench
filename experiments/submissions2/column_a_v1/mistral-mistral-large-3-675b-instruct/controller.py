import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.nu = 2
        self.ny = 2

        # More conservative tuning to prevent safety violations
        self.Kp = np.array([0.05, 0.05])  # Reduced proportional gain
        self.Ti = np.array([600.0, 600.0])  # Slower integral action
        self.Td = np.array([0.0, 0.0])

        # Anti-windup and bumpless transfer
        self.integral = np.zeros(self.nu)
        self.prev_error = np.zeros(self.ny)
        self.prev_output = np.array([2.70629, 3.20629])
        self.prev_u = self.prev_output.copy()

        # Actuator limits
        self.u_min = np.array([1.5, 2.0])
        self.u_max = np.array([4.5, 5.0])

        # Rate limiting
        self.max_duty = 0.002  # Reduced from 0.003
        self.max_step = self.max_duty * self.sample_time

        # Safety margins
        self.min_purity = 0.86  # Above the 0.85 trip point
        self.min_draw = 0.06  # Above the 0.05 trip point

        # Setpoint tracking with ramp limits
        self.setpoint = np.array([0.99, 0.99])
        self.setpoint_ramp = 0.0005  # max change per second

        # Feedforward terms to maintain product draws
        self.LV_ratio = 2.70629 / 3.20629

    def reset(self):
        self.integral = np.zeros(self.nu)
        self.prev_error = np.zeros(self.ny)
        self.prev_output = np.array([2.70629, 3.20629])
        self.prev_u = self.prev_output.copy()
        self.setpoint = np.array([0.99, 0.99])

    def step(self, t, y, r, quality):
        # Update setpoints with ramp limiting
        for i in range(self.ny):
            if not np.isnan(r[i]):
                delta = r[i] - self.setpoint[i]
                delta = np.clip(delta, -self.setpoint_ramp*self.sample_time, self.setpoint_ramp*self.sample_time)
                self.setpoint[i] += delta

        # Compute error with safety checks
        error = np.zeros(self.ny)
        for i in range(self.ny):
            if not np.isnan(r[i]):
                # Apply safety margin to error calculation
                if i == 0:  # Distillate purity
                    error[i] = max(0, self.setpoint[i] - y[i]) if y[i] > self.min_purity else 0
                else:  # Bottoms purity (heavy key)
                    xB = 1 - y[i]
                    error[i] = max(0, self.setpoint[i] - y[i]) if xB < (1 - self.min_purity) else 0

        # PID calculation with anti-windup and feedforward
        output = np.zeros(self.nu)
        for i in range(self.nu):
            # Proportional term
            p_term = self.Kp[i] * error[i]

            # Integral term with anti-windup
            self.integral[i] += error[i] * self.sample_time
            i_term = self.Kp[i] / self.Ti[i] * self.integral[i]

            # Feedforward term to maintain LV ratio
            if i == 0:  # Reflux
                ff_term = self.LV_ratio * self.prev_u[1]
            else:  # Boilup
                ff_term = self.prev_u[0] / self.LV_ratio

            # Total output
            output[i] = p_term + i_term + ff_term

            # Anti-windup
            if output[i] < self.u_min[i] or output[i] > self.u_max[i]:
                self.integral[i] -= error[i] * self.sample_time

        # Clamp output to actuator limits
        output = np.clip(output, self.u_min, self.u_max)

        # Rate limiting
        delta_u = output - self.prev_u
        delta_u = np.clip(delta_u, -self.max_step, self.max_step)
        output = self.prev_u + delta_u

        # Update previous values
        self.prev_error = error.copy()
        self.prev_u = output.copy()

        return output