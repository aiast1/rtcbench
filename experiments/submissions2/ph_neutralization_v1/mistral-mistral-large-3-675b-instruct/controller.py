import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.n_act = 1
        self.n_meas = 2

        # Conservative PID parameters
        self.Kp = 0.8
        self.Ti = 200.0
        self.Td = 5.0
        self.N = 5.0

        # State variables
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.prev_output = 14.22
        self.filtered_ph = 6.5
        self.ph_filter_alpha = 0.1

        # Safety limits with buffer
        self.level_low = 8.0
        self.level_high = 27.0
        self.ph_low = 4.2
        self.ph_high = 10.3

        # Rate limiting
        self.max_output_change = 0.3
        self.output_lim_low = 0.0
        self.output_lim_high = 30.0

        # Disturbance detection
        self.prev_ph = 6.5
        self.disturbance_threshold = 0.5
        self.disturbance_counter = 0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_derivative = 0.0
        self.prev_output = 14.22
        self.filtered_ph = 6.5
        self.prev_ph = 6.5
        self.disturbance_counter = 0

    def step(self, t, y, r, quality):
        # Extract measurements
        ph_meas = y[0]
        level_meas = y[1]

        # Update setpoint
        ph_sp = r[0]
        if t < 350:
            ph_sp = 6.5
        elif 350 <= t < 800:
            ph_sp = 8.2
        elif 800 <= t < 1000:
            ph_sp = 8.2 + (9.9 - 8.2) * (t - 800) / 200
        elif 1000 <= t < 1300:
            ph_sp = 9.9
        else:
            ph_sp = 6.5

        # Filter pH measurement
        if quality[0]:
            self.filtered_ph = self.ph_filter_alpha * ph_meas + (1 - self.ph_filter_alpha) * self.filtered_ph
        ph = self.filtered_ph

        # Safety checks - override control if needed
        if quality[1]:
            if level_meas < self.level_low:
                output = min(self.prev_output + self.max_output_change, self.output_lim_high)
                self.integral = 0.0
                self.prev_output = output
                return np.array([output], dtype=np.float64)
            elif level_meas > self.level_high:
                output = max(self.prev_output - self.max_output_change, self.output_lim_low)
                self.integral = 0.0
                self.prev_output = output
                return np.array([output], dtype=np.float64)

        if ph < self.ph_low or ph > self.ph_high:
            if ph < self.ph_low:
                output = min(self.prev_output + self.max_output_change, self.output_lim_high)
            else:
                output = max(self.prev_output - self.max_output_change, self.output_lim_low)
            self.integral = 0.0
            self.prev_output = output
            return np.array([output], dtype=np.float64)

        # Disturbance detection
        ph_change = abs(ph - self.prev_ph)
        if ph_change > self.disturbance_threshold:
            self.disturbance_counter = 10
        self.prev_ph = ph

        # Calculate error
        error = ph_sp - ph

        # Proportional term
        proportional = self.Kp * error

        # Integral term with anti-windup
        if self.disturbance_counter > 0:
            self.disturbance_counter -= 1
        else:
            self.integral += error * self.sample_time
        integral = self.Kp / self.Ti * self.integral

        # Derivative term with filtering
        derivative = (error - self.prev_error) / self.sample_time if self.sample_time > 0 else 0.0
        filtered_derivative = self.prev_derivative + (derivative - self.prev_derivative) * self.N / (self.N + self.sample_time / self.Td)
        derivative_term = self.Kp * self.Td * filtered_derivative

        # PID output
        output = proportional + integral + derivative_term

        # Anti-windup
        if output > self.output_lim_high:
            output = self.output_lim_high
            self.integral = (output - proportional - derivative_term) * self.Ti / self.Kp
        elif output < self.output_lim_low:
            output = self.output_lim_low
            self.integral = (output - proportional - derivative_term) * self.Ti / self.Kp

        # Rate limiting
        output = np.clip(output, self.prev_output - self.max_output_change, self.prev_output + self.max_output_change)
        output = np.clip(output, self.output_lim_low, self.output_lim_high)

        # Update state
        self.prev_error = error
        self.prev_derivative = filtered_derivative
        self.prev_output = output

        return np.array([output], dtype=np.float64)