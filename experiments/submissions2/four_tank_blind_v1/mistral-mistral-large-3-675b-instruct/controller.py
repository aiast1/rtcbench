import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.n_meas = 2
        self.n_ctrl = 2
        self.ctrl_limits = np.array([0.0, 10.0])
        self.meas_limits = np.array([0.0, 20.0])

        # PID parameters
        self.Kp = np.array([0.3, 0.3])
        self.Ti = np.array([150.0, 150.0])
        self.Td = np.array([2.0, 2.0])
        self.N = 5.0  # Derivative filter coefficient

        # Anti-windup
        self.anti_windup = True
        self.aw_gain = 0.2

        # Initialize state
        self.reset()

    def reset(self):
        self.integral = np.zeros(self.n_meas)
        self.prev_error = np.zeros(self.n_meas)
        self.prev_derivative = np.zeros(self.n_meas)
        self.prev_measurement = np.zeros(self.n_meas)
        self.prev_output = np.array([3.0, 3.0])

    def step(self, t, y, r, quality):
        # Update setpoints
        if t < 200:
            setpoint = np.array([12.26, 12.78])
        elif t < 500:
            setpoint = np.array([14.0, 12.78])
        elif t < 560:
            progress = (t - 500) / 60.0
            setpoint = np.array([14.0, 12.78 + (11.2 - 12.78) * progress])
        else:
            setpoint = np.array([12.26, 12.78])

        # Calculate errors
        error = np.where(quality, setpoint - y, self.prev_error)

        # PID calculation
        output = np.zeros(self.n_ctrl)
        for i in range(self.n_ctrl):
            # Proportional term
            p_term = self.Kp[i] * error[i]

            # Integral term
            self.integral[i] += error[i] * self.sample_time
            i_term = self.integral[i] / self.Ti[i]

            # Derivative term
            if quality[i]:
                derivative = (error[i] - self.prev_error[i]) / self.sample_time
                filtered_derivative = (self.Td[i] * self.prev_derivative[i] +
                                     self.N * derivative * self.sample_time) / (self.Td[i] + self.N * self.sample_time)
            else:
                filtered_derivative = self.prev_derivative[i]
            d_term = self.Td[i] * filtered_derivative

            # Calculate output
            output[i] = p_term + i_term + d_term

            # Anti-windup
            if self.anti_windup:
                if output[i] > self.ctrl_limits[1] or output[i] < self.ctrl_limits[0]:
                    self.integral[i] -= error[i] * self.sample_time * self.aw_gain

            # Clamp output
            output[i] = np.clip(output[i], self.ctrl_limits[0], self.ctrl_limits[1])

        # Update state
        self.prev_error = error.copy()
        self.prev_derivative = np.array([filtered_derivative]*self.n_meas)
        self.prev_measurement = y.copy()
        self.prev_output = output.copy()

        return output