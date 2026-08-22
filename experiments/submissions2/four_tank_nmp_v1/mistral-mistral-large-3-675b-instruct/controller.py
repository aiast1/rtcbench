import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.actuator_limits = np.array([[0.0, 10.0], [0.0, 10.0]])
        self.initial_actuators = np.array([3.0, 3.0])

        # Conservative PID parameters with safety margins
        self.Kp = np.array([0.3, 0.3])
        self.Ki = np.array([0.01, 0.01])
        self.Kd = np.array([0.05, 0.05])

        # Anti-windup and bumpless transfer
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.prev_output = self.initial_actuators.copy()
        self.prev_measurement = np.zeros(2)

        # Filter for derivative term
        self.tau = 1.0  # Increased filter time constant
        self.alpha = self.sample_time / (self.tau + self.sample_time)
        self.filtered_derivative = np.zeros(2)

        # Safety buffers
        self.safety_buffer = 1.0  # cm buffer from safety limits

    def reset(self):
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.prev_output = self.initial_actuators.copy()
        self.prev_measurement = np.zeros(2)
        self.filtered_derivative = np.zeros(2)

    def step(self, t, y, r, quality):
        # Initialize output
        output = self.prev_output.copy()

        # Apply safety buffer to setpoints
        safe_r = r.copy()
        safe_r[0] = np.clip(r[0], self.safety_buffer, 20.0 - self.safety_buffer)
        safe_r[1] = np.clip(r[1], self.safety_buffer, 20.0 - self.safety_buffer)

        # Calculate error
        error = safe_r - y

        # Only update integral if measurement is good
        for i in range(2):
            if quality[i]:
                # Integral with anti-windup
                self.integral[i] += error[i] * self.sample_time
                max_integral = (self.actuator_limits[i][1] - self.actuator_limits[i][0]) / (2 * self.Ki[i])
                self.integral[i] = np.clip(self.integral[i], -max_integral, max_integral)

        # Calculate derivative term with filtering
        derivative = (error - self.prev_error) / self.sample_time
        self.filtered_derivative = self.alpha * derivative + (1 - self.alpha) * self.filtered_derivative

        # PID calculation
        for i in range(2):
            if quality[i]:
                output[i] = (
                    self.Kp[i] * error[i] +
                    self.Ki[i] * self.integral[i] +
                    self.Kd[i] * self.filtered_derivative[i]
                )
            else:
                # If measurement is bad, hold last output
                output[i] = self.prev_output[i]

        # Bumpless transfer with reduced step size
        output = np.clip(output, self.prev_output - 0.1, self.prev_output + 0.1)

        # Apply actuator limits
        output = np.clip(output, self.actuator_limits[0][0], self.actuator_limits[0][1])

        # Enhanced anti-windup using back-calculation
        for i in range(2):
            if (output[i] >= self.actuator_limits[i][1] and error[i] > 0) or \
               (output[i] <= self.actuator_limits[i][0] and error[i] < 0):
                self.integral[i] -= error[i] * self.sample_time * 0.5  # More aggressive back-calculation

        # Update previous values
        self.prev_error = error.copy()
        self.prev_output = output.copy()
        self.prev_measurement = y.copy()

        return output