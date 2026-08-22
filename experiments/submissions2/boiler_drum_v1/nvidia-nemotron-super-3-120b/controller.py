import numpy as np

class Controller:
    def __init__(self, brief):
        self.sample_time = brief.sample_time
        self.setpoint = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 45.0  # Bumpless start
        self.integral_clamp = 10.0  # Further reduced to minimize windup
        self.max_delta_output = 0.3  # Strict limit on actuator movement per step to avoid duty limit breach
        self.Kp = 0.02  # Low proportional gain to minimize oscillation and actuator movement
        self.Ki = 0.0001  # Very low integral gain to prevent windup and slow drift
        self.Kd = 0.0  # Derivative disabled due to noise and delay
        self.last_level = None
        self.last_t = None
        self.output_history = []  # Track recent outputs to detect and dampen chattering
        self.consecutive_same_output = 0  # Count consecutive identical outputs to detect stiction or stall

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 45.0
        self.last_level = None
        self.last_t = None
        self.output_history = []
        self.consecutive_same_output = 0

    def step(self, t, y, r, quality):
        # If level measurement is bad, hold last good output
        if not quality[0]:
            return np.array([np.clip(self.prev_output, 0.0, 100.0)])

        level = y[0]  # LT-101: indicated drum level in mm
        setpoint = r[0] if not np.isnan(r[0]) else self.setpoint
        self.setpoint = setpoint

        # Initialize on first good measurement
        if self.last_level is None:
            self.last_level = level
            self.last_t = t
            return np.array([np.clip(self.prev_output, 0.0, 100.0)])

        # Compute error
        error = setpoint - level

        # Integral term with anti-windup
        self.integral += error * self.sample_time
        self.integral = np.clip(self.integral, -self.integral_clamp, self.integral_clamp)

        # PID output (derivative disabled)
        output = self.Kp * error + self.Ki * self.integral

        # Limit actuator movement per step to prevent chattering and duty limit violation
        delta = output - self.prev_output
        if abs(delta) > self.max_delta_output:
            output = self.prev_output + np.sign(delta) * self.max_delta_output

        # Additional smoothing: limit change based on recent history to avoid high-frequency movement
        self.output_history.append(output)
        if len(self.output_history) > 4:
            self.output_history.pop(0)
        if len(self.output_history) == 4:
            # Use median of last 4 outputs to dampen oscillation
            sorted_hist = sorted(self.output_history)
            median_output = (sorted_hist[1] + sorted_hist[2]) / 2.0
            # Blend current output with median to reduce jitter
            output = 0.8 * output + 0.2 * median_output

        # Final clamp to physical limits
        output = np.clip(output, 0.0, 100.0)

        # Update state
        self.prev_error = error
        self.prev_output = output
        self.last_level = level
        self.last_t = t

        return np.array([output])