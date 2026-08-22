import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.actuator_limits = np.array([
            [3.0, 35.0],      # FCV-101: F/V (1/h)
            [-9000.0, 0.0]    # TCV-201: Q_K (kJ/h)
        ])

        # Use the starting actuator values from the brief
        self.initial_actuators = np.array([14.19, -1113.5])

        # PID parameters for C_B (index 0) and T (index 1)
        self.Kp = np.array([-1.0, 0.3])
        self.Ki = np.array([-0.02, 0.01])
        self.Kd = np.array([-0.3, 0.05])

        # Anti-windup and bumpless transfer
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.prev_output = self.initial_actuators.copy()
        self.prev_measurement = np.zeros(2)

        # Safety constraints
        self.T_max = 150.0

        # Filter for derivative term (low-pass filter)
        self.tau_d = 5.0  # seconds
        self.alpha_d = self.sample_time / (self.tau_d + self.sample_time)
        self.derivative_filtered = np.zeros(2)

        # Feedforward for known non-monotonicity in C_B vs F/V
        self.FV_optimal = 14.7  # h^-1, nominal optimal point

    def reset(self):
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.prev_output = self.initial_actuators.copy()
        self.prev_measurement = np.zeros(2)
        self.derivative_filtered = np.zeros(2)

    def step(self, t, y, r, quality):
        # Initialize output
        output = self.prev_output.copy()

        # Process measurements and setpoints
        for i in range(2):
            if not quality[i] or np.isnan(r[i]):
                # If measurement is bad or setpoint is NaN, skip this step
                continue

            error = r[i] - y[i]

            # Proportional term
            p_term = self.Kp[i] * error

            # Integral term with anti-windup
            self.integral[i] += error * self.sample_time
            i_term = self.Ki[i] * self.integral[i]

            # Derivative term with filtering
            if t > 0:  # Avoid derivative kick at first step
                derivative = (error - self.prev_error[i]) / self.sample_time
                self.derivative_filtered[i] = self.alpha_d * derivative + (1 - self.alpha_d) * self.derivative_filtered[i]
            d_term = self.Kd[i] * self.derivative_filtered[i]

            # PID sum
            pid_sum = p_term + i_term + d_term

            # Apply feedforward for C_B (index 0)
            if i == 0:
                # If F/V is above optimal, reduce it to avoid non-monotonic region
                if self.prev_output[0] > self.FV_optimal:
                    ff_term = -0.05 * (self.prev_output[0] - self.FV_optimal)
                else:
                    ff_term = 0.0
                pid_sum += ff_term

            # Clamp output to actuator limits
            output[i] = np.clip(pid_sum, self.actuator_limits[i, 0], self.actuator_limits[i, 1])

            # Anti-windup: clamp integral term if output is saturated
            if output[i] != pid_sum:
                self.integral[i] -= error * self.sample_time  # Back-calculate integral

            # Update previous error
            self.prev_error[i] = error

        # Safety checks (T must stay <= 150.0)
        if quality[1] and y[1] > self.T_max - 5.0:  # Conservative margin
            output[1] = min(output[1], -1000.0)  # Increase cooling aggressively

        # Update previous output and measurement
        self.prev_output = output.copy()
        self.prev_measurement = y.copy()

        return output