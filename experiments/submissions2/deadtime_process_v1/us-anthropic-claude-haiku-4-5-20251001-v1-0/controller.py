import numpy as np
from dataclasses import dataclass

class Controller:
    def __init__(self, brief):
        self.brief = brief
        self.sample_time = brief.sample_time
        
        # Increase gains gradually while respecting duty cycle
        # Previous gains were too conservative - we can be more responsive
        self.Kp = 0.8
        self.Ki = 0.06
        self.Kd = 5.0
        
        # Anti-windup: integral clamp
        self.integral_max = 25.0
        self.integral_min = -25.0
        
        # State tracking
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_measurement = None
        self.prev_output = 50.0
        
        # Derivative filter
        self.derivative_filter_alpha = 0.25
        self.filtered_derivative = 0.0
        
        # Slew rate limiter - increase slightly but stay safe
        # 0.0012 duty limit is very tight, but we can use ~0.08% per step
        self.max_slew_rate = 0.016  # % per second = 0.08% per 5s step
        
        # Measurement smoothing
        self.measurement_filter_alpha = 0.3
        self.filtered_measurement = None
        
    def reset(self):
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_measurement = None
        self.prev_output = 50.0
        self.filtered_derivative = 0.0
        self.filtered_measurement = None
        
    def step(self, t, y, r, quality):
        # Extract measurement and setpoint
        if not quality[0]:
            # Use previous filtered measurement if current is bad
            if self.filtered_measurement is not None:
                measurement = self.filtered_measurement
            else:
                measurement = 55.0
        else:
            measurement = y[0]
            # Apply low-pass filter to measurement to reduce noise
            if self.filtered_measurement is None:
                self.filtered_measurement = measurement
            else:
                self.filtered_measurement = (
                    self.measurement_filter_alpha * measurement +
                    (1.0 - self.measurement_filter_alpha) * self.filtered_measurement
                )
            measurement = self.filtered_measurement
            self.prev_measurement = measurement
        
        setpoint = r[0]
        
        # Compute error
        error = setpoint - measurement
        
        # Proportional term
        p_term = self.Kp * error
        
        # Integral term with anti-windup
        self.integral_error += error * self.sample_time
        self.integral_error = np.clip(
            self.integral_error,
            self.integral_min,
            self.integral_max
        )
        i_term = self.Ki * self.integral_error
        
        # Derivative term with low-pass filtering
        raw_derivative = (error - self.prev_error) / self.sample_time
        self.filtered_derivative = (
            self.derivative_filter_alpha * raw_derivative +
            (1.0 - self.derivative_filter_alpha) * self.filtered_derivative
        )
        d_term = self.Kd * self.filtered_derivative
        
        self.prev_error = error
        
        # Compute raw PID output
        raw_output = p_term + i_term + d_term
        
        # Add bias to maintain steady-state near 50%
        output = raw_output + 50.0
        
        # Apply hard limits
        output = np.clip(output, 0.0, 100.0)
        
        # Apply slew rate limiting
        max_change = self.max_slew_rate * self.sample_time
        output = np.clip(
            output,
            self.prev_output - max_change,
            self.prev_output + max_change
        )
        
        # Final hard limit after slew
        output = np.clip(output, 0.0, 100.0)
        
        self.prev_output = output
        
        return np.array([output])